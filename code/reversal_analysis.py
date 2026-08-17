"""
reversal_analysis.py — Contingency reversal analysis for tACS Bandit study

Analyzes how participants detect and adapt to contingency reversals
(when the high-reward option switches).

Key outputs:
- Reversal identification within each run
- Reversal-locked accuracy curves
- Trials-to-criterion computation
- Post-reversal accuracy summary
- Reversal-locked WSLS analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Optional, Dict, List, Tuple

from config import (
    SUBJECT_INFO,
    COLOR_SHAM,
    COLOR_ACTIVE,
    AGE_MIN,
    AGE_MAX,
    AGE_COLORSCALE,
    PLOTLY_TEMPLATE,
    FONT_FAMILY,
)


# =============================================================================
# Age Color Utility
# =============================================================================

def _get_age_colormap():
    """Get matplotlib colormap for age gradient."""
    import matplotlib.colors as mcolors
    return mcolors.LinearSegmentedColormap.from_list('blue_gold', ['#1565C0', '#FFB300'])


def _age_to_rgb(age: float, age_min: float = AGE_MIN, age_max: float = AGE_MAX) -> str:
    """Convert age to RGB color string for Plotly."""
    cmap = _get_age_colormap()
    norm = np.clip((age - age_min) / (age_max - age_min), 0, 1)
    rgba = cmap(norm)
    return f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'


# =============================================================================
# Reversal Identification
# =============================================================================

def identify_reversals(
    df: pd.DataFrame,
    window_pre: int = 5,
    window_post: int = 15,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Detect contingency reversals and compute each trial's position
    relative to the nearest reversal.
    
    A reversal occurs when current_good changes from one trial to the next
    within a run. The first trial of the new contingency is trial_from_rev = 0.
    
    Parameters
    ----------
    df : DataFrame
        Trial-level data with columns: subject_id, run, current_good, condition
    window_pre : int
        Number of trials before reversal to include in window
    window_post : int
        Number of trials after reversal to include in window
    verbose : bool
        If True, print summary
    
    Returns
    -------
    DataFrame
        Input data with added columns:
        - is_reversal: True on the first trial after a reversal
        - reversal_id: unique identifier for each reversal
        - trial_from_rev: trial position relative to nearest reversal
        - in_rev_window: True if within analysis window
    """
    df = df.copy()
    df['is_reversal'] = False
    # Object dtype, not NaN: reversal_id holds strings like '10369_run1_rev0',
    # and initializing with np.nan gives the column float64, which pandas 2.x
    # refuses to hold a string in (it raises rather than silently upcasting).
    df['reversal_id'] = pd.Series(index=df.index, dtype='object')
    df['trial_from_rev'] = np.nan
    df['in_rev_window'] = False
    
    for (sub_id, run), group in df.groupby(['subject_id', 'run']):
        idx = group.index
        good = group['current_good'].values
        
        # Detect reversal points (where current_good changes)
        rev_mask = np.zeros(len(good), dtype=bool)
        rev_mask[1:] = good[1:] != good[:-1]
        rev_trials = np.where(rev_mask)[0]
        
        if len(rev_trials) == 0:
            continue
        
        # For each reversal, assign trial positions
        for rev_i, rev_pos in enumerate(rev_trials):
            rev_id = f'{sub_id}_run{run}_rev{rev_i}'
            df.loc[idx[rev_pos], 'is_reversal'] = True
            
            # Window around this reversal
            for offset in range(-window_pre, window_post + 1):
                t = rev_pos + offset
                if 0 <= t < len(idx):
                    trial_idx = idx[t]
                    # Only assign if closer to this reversal than any prior
                    current_dist = df.loc[trial_idx, 'trial_from_rev']
                    if pd.isna(current_dist) or abs(offset) < abs(current_dist):
                        df.loc[trial_idx, 'trial_from_rev'] = offset
                        df.loc[trial_idx, 'reversal_id'] = rev_id
                        df.loc[trial_idx, 'in_rev_window'] = True
    
    if verbose:
        n_reversals = df['is_reversal'].sum()
        n_subjects = df.loc[df['is_reversal'], 'subject_id'].nunique()
        print(f'Identified {n_reversals} reversals across {n_subjects} subjects')
        print(f'Trials in reversal windows: {df["in_rev_window"].sum()} / {len(df)}')
        
        # Per-subject breakdown
        rev_counts = df[df['is_reversal']].groupby(['subject_id', 'condition']).size()
        print(f'\nReversals per subject × condition:')
        print(rev_counts.unstack(fill_value=0).to_string())
    
    return df


def propagate_reversal_columns(
    data: pd.DataFrame,
    data_clean: pd.DataFrame,
    data_h1: pd.DataFrame,
    data_h2: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Copy reversal columns from full data to filtered datasets.
    
    Parameters
    ----------
    data : DataFrame
        Full dataset with reversal columns
    data_clean, data_h1, data_h2 : DataFrame
        Filtered datasets
    
    Returns
    -------
    tuple of (data_clean, data_h1, data_h2) with reversal columns added
    """
    rev_cols = ['is_reversal', 'reversal_id', 'trial_from_rev', 'in_rev_window']
    
    for col in rev_cols:
        if col in data.columns:
            data_clean = data_clean.copy()
            data_h1 = data_h1.copy()
            data_h2 = data_h2.copy()
            
            data_clean[col] = data.loc[data_clean.index, col].values
            data_h1[col] = data.loc[data_h1.index, col].values
            data_h2[col] = data.loc[data_h2.index, col].values
    
    return data_clean, data_h1, data_h2


# =============================================================================
# Reversal-Locked Accuracy
# =============================================================================

def compute_reversal_accuracy(
    data: pd.DataFrame,
    conditions: List[str] = ['sham', 'active']
) -> pd.DataFrame:
    """
    Compute p(correct) at each trial position relative to reversal.
    
    Parameters
    ----------
    data : DataFrame
        Trial-level data with reversal columns and 'correct' column
    conditions : list
        Conditions to include
    
    Returns
    -------
    DataFrame
        Accuracy by subject × condition × trial_from_rev
    """
    rev_data = data[
        data['in_rev_window'] & 
        data['choice'].notna() &
        data['condition'].isin(conditions)
    ].copy()
    
    # Convert 'correct' to numeric (handles string 'True'/'False')
    if rev_data['correct'].dtype == object:
        rev_data['correct_num'] = rev_data['correct'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})
    else:
        rev_data['correct_num'] = rev_data['correct'].astype(int)
    
    rev_acc = (rev_data
        .groupby(['subject_id', 'condition', 'trial_from_rev'])['correct_num']
        .mean()
        .reset_index()
        .rename(columns={'correct_num': 'p_correct'})
    )
    
    return rev_acc


def compute_post_reversal_accuracy(
    data: pd.DataFrame,
    window: Tuple[int, int] = (1, 10)
) -> pd.DataFrame:
    """
    Compute mean accuracy in post-reversal window.
    
    Parameters
    ----------
    data : DataFrame
        Trial-level data with reversal columns
    window : tuple
        (start, end) trial positions to include
    
    Returns
    -------
    DataFrame
        Mean post-reversal accuracy by subject × condition
    """
    post_rev = data[
        data['in_rev_window'] &
        data['choice'].notna() &
        data['trial_from_rev'].between(*window)
    ].copy()
    
    # Convert 'correct' to numeric (handles string 'True'/'False')
    if post_rev['correct'].dtype == object:
        post_rev['correct_num'] = post_rev['correct'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})
    else:
        post_rev['correct_num'] = post_rev['correct'].astype(int)
    
    post_rev_acc = (post_rev
        .groupby(['subject_id', 'condition'])['correct_num']
        .mean()
        .reset_index()
        .rename(columns={'correct_num': 'post_rev_accuracy'})
    )
    
    return post_rev_acc


# =============================================================================
# Trials to Criterion
# =============================================================================

def compute_trials_to_criterion(
    data: pd.DataFrame,
    criterion: int = 3
) -> pd.DataFrame:
    """
    For each reversal, compute trials until criterion consecutive correct.
    
    Parameters
    ----------
    data : DataFrame
        Trial-level data with reversal columns
    criterion : int
        Number of consecutive correct choices required
    
    Returns
    -------
    DataFrame
        One row per reversal with trials_to_criterion and reached_criterion
    """
    results = []
    
    post_rev_data = data[
        data['in_rev_window'] & 
        (data['trial_from_rev'] >= 0)
    ]
    
    for rev_id, group in post_rev_data.groupby('reversal_id'):
        if pd.isna(rev_id):
            continue
        
        group = group.sort_values('trial_from_rev')
        sub_id = group['subject_id'].iloc[0]
        condition = group['condition'].iloc[0]
        
        # Find first run of `criterion` consecutive correct
        # Convert to Python list to avoid pandas/numpy type issues
        correct_vals = group['correct'].tolist()
        consecutive = 0
        ttc = np.nan
        
        for i, c in enumerate(correct_vals):
            # Convert to boolean - handle string 'True'/'False', bool, int, etc.
            if isinstance(c, str):
                is_correct = c.strip().upper() == 'TRUE'
            elif isinstance(c, (bool, np.bool_)):
                is_correct = bool(c)
            elif isinstance(c, (int, float, np.integer, np.floating)):
                is_correct = bool(c)
            else:
                is_correct = False
            
            if is_correct:
                consecutive += 1
                if consecutive >= int(criterion):
                    ttc = i - int(criterion) + 1
                    break
            else:
                consecutive = 0
        
        results.append({
            'reversal_id': rev_id,
            'subject_id': sub_id,
            'condition': condition,
            'trials_to_criterion': ttc,
            'reached_criterion': not np.isnan(ttc),
        })
    
    return pd.DataFrame(results)


# =============================================================================
# Visualization
# =============================================================================

def plot_reversal_accuracy(
    rev_acc: pd.DataFrame,
    conditions: List[str] = ['sham', 'active'],
    show_fig: bool = True
) -> go.Figure:
    """
    Plot reversal-locked accuracy curves with individual subjects and group mean.
    
    Parameters
    ----------
    rev_acc : DataFrame
        Output from compute_reversal_accuracy()
    conditions : list
        Conditions to plot
    show_fig : bool
        If True, display figure
    
    Returns
    -------
    go.Figure
    """
    colors = {'sham': COLOR_SHAM, 'active': '#E65100'}
    
    fig = go.Figure()
    
    for cond in conditions:
        cond_data = rev_acc[rev_acc['condition'] == cond]
        
        # Individual subject lines (faint)
        for sub_id in cond_data['subject_id'].unique():
            sub_data = cond_data[cond_data['subject_id'] == sub_id].sort_values('trial_from_rev')
            fig.add_trace(go.Scatter(
                x=sub_data['trial_from_rev'],
                y=sub_data['p_correct'],
                mode='lines',
                line=dict(color=colors[cond], width=0.8),
                opacity=0.2,
                text=f'sub-{sub_id} ({cond})',
                hoverinfo='text+y',
                showlegend=False,
            ))
        
        # Group mean
        group_mean = cond_data.groupby('trial_from_rev')['p_correct'].agg(['mean', 'sem']).reset_index()
        fig.add_trace(go.Scatter(
            x=group_mean['trial_from_rev'],
            y=group_mean['mean'],
            mode='lines+markers',
            line=dict(color=colors[cond], width=2.5),
            marker=dict(size=5, color=colors[cond]),
            name=cond.capitalize(),
            error_y=dict(type='data', array=group_mean['sem'], visible=True,
                        color=colors[cond], thickness=1, width=3),
            hovertemplate='Trial %{x}: %{y:.3f}<extra>' + cond.capitalize() + '</extra>',
        ))
    
    # Reversal marker
    fig.add_vline(x=0, line=dict(color='#CC0000', width=1.5, dash='dash'),
                  annotation_text='Reversal', annotation_position='top',
                  annotation_font=dict(size=12, color='#CC0000'))
    
    # Chance line
    fig.add_hline(y=0.5, line=dict(color='#999999', width=1, dash='dot'))
    
    fig.update_layout(
        height=420, width=750,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=60, r=40, t=40, b=50),
        font=dict(family=FONT_FAMILY, size=14),
        legend=dict(x=0.85, y=0.98, font=dict(size=13)),
        xaxis_title='Trials from Reversal',
        yaxis_title='p(correct)',
        yaxis_range=[0, 1.02],
    )
    
    fig.update_xaxes(showgrid=False, zeroline=False,
                     showline=True, linewidth=1, linecolor='black',
                     tickfont=dict(size=13))
    fig.update_yaxes(showgrid=False,
                     showline=True, linewidth=1, linecolor='black',
                     tickfont=dict(size=13))
    
    if show_fig:
        fig.show(config=dict(toImageButtonOptions=dict(filename='reversal_locked_accuracy')))
    
    return fig


def plot_trials_to_criterion(
    ttc: pd.DataFrame,
    age_lookup: Dict,
    conditions: List[str] = ['sham', 'active'],
    show_fig: bool = True
) -> go.Figure:
    """
    Plot trials-to-criterion comparison between conditions.
    
    Parameters
    ----------
    ttc : DataFrame
        Output from compute_trials_to_criterion()
    age_lookup : dict
        {subject_id: age} mapping
    conditions : list
        Conditions to plot
    show_fig : bool
        If True, display figure
    
    Returns
    -------
    go.Figure
    """
    # Average TTC per subject × condition (for subjects who reached criterion)
    ttc_subj = (ttc[ttc['condition'].isin(conditions) & ttc['reached_criterion']]
        .groupby(['subject_id', 'condition'])['trials_to_criterion']
        .mean()
        .reset_index()
    )
    
    fig = go.Figure()
    rng = np.random.default_rng(42)
    cap_w = 0.08
    
    for i, cond in enumerate(conditions):
        cond_data = ttc_subj[ttc_subj['condition'] == cond]
        ages = [age_lookup.get(s, np.nan) for s in cond_data['subject_id']]
        jitter = rng.uniform(-0.12, 0.12, size=len(cond_data))
        
        # Individual points
        fig.add_trace(go.Scatter(
            x=i + jitter,
            y=cond_data['trials_to_criterion'],
            mode='markers',
            marker=dict(size=10, opacity=0.75,
                        color=[_age_to_rgb(a) for a in ages],
                        line=dict(width=0.5, color='white')),
            text=[f'sub-{s}<br>Age: {age_lookup.get(s, "?"):.0f}<br>TTC: {t:.1f}'
                  for s, t in zip(cond_data['subject_id'], cond_data['trials_to_criterion'])],
            hoverinfo='text',
            showlegend=False,
        ))
        
        # Mean + SEM
        vals = cond_data['trials_to_criterion'].dropna()
        if len(vals) > 0:
            mean = vals.mean()
            sem = vals.sem() if len(vals) > 1 else 0
            
            # Error bar
            fig.add_trace(go.Scatter(
                x=[i, i], y=[mean - sem, mean + sem],
                mode='lines', line=dict(color='#404040', width=1.5),
                showlegend=False, hoverinfo='skip'
            ))
            
            # Caps
            for y_cap in [mean + sem, mean - sem]:
                fig.add_trace(go.Scatter(
                    x=[i - cap_w, i + cap_w], y=[y_cap, y_cap],
                    mode='lines', line=dict(color='#404040', width=1.5),
                    showlegend=False, hoverinfo='skip'
                ))
            
            # Mean line
            fig.add_trace(go.Scatter(
                x=[i - 0.2, i + 0.2], y=[mean, mean],
                mode='lines', line=dict(color='#404040', width=2),
                showlegend=False, hoverinfo='text',
                text=[f'{cond.capitalize()} mean: {mean:.1f}'] * 2,
            ))
    
    fig.update_layout(
        height=400, width=450,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=60, r=40, t=30, b=50),
        font=dict(family=FONT_FAMILY, size=14),
        xaxis=dict(tickvals=list(range(len(conditions))),
                  ticktext=[c.capitalize() for c in conditions]),
        yaxis_title='Trials to Criterion',
    )
    
    fig.update_xaxes(showgrid=False, zeroline=False,
                     showline=True, linewidth=1, linecolor='black',
                     tickfont=dict(size=15))
    fig.update_yaxes(showgrid=False, rangemode='tozero',
                     showline=True, linewidth=1, linecolor='black',
                     tickfont=dict(size=13))
    
    if show_fig:
        fig.show(config=dict(toImageButtonOptions=dict(filename='trials_to_criterion')))
    
    return fig


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_reversal_analysis(
    data: pd.DataFrame,
    data_clean: pd.DataFrame,
    conditions: List[str] = ['sham', 'active'],
    criterion: int = 3,
    show_plots: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run complete reversal analysis pipeline.
    
    Parameters
    ----------
    data : DataFrame
        Full dataset (will have reversal columns added)
    data_clean : DataFrame
        Behaviorally cleaned dataset
    conditions : list
        Conditions to analyze
    criterion : int
        Consecutive correct trials for TTC analysis
    show_plots : bool
        If True, display visualizations
    verbose : bool
        If True, print summaries
    
    Returns
    -------
    dict with keys:
        'data': DataFrame with reversal columns
        'rev_acc': reversal-locked accuracy
        'post_rev_acc': post-reversal accuracy summary
        'ttc': trials-to-criterion data
        'accuracy_fig': accuracy curve figure
        'ttc_fig': TTC comparison figure
    """
    # Identify reversals
    data = identify_reversals(data, verbose=verbose)
    
    # Propagate to data_clean
    rev_cols = ['is_reversal', 'reversal_id', 'trial_from_rev', 'in_rev_window']
    for col in rev_cols:
        data_clean = data_clean.copy()
        data_clean[col] = data.loc[data_clean.index, col].values
    
    # Compute accuracy metrics
    rev_acc = compute_reversal_accuracy(data_clean, conditions)
    post_rev_acc = compute_post_reversal_accuracy(data_clean)
    ttc = compute_trials_to_criterion(data_clean, criterion=criterion)
    
    if verbose:
        print(f'\nTrials-to-criterion ({criterion} consecutive correct):')
        print(ttc.groupby('condition')['trials_to_criterion'].describe().round(1))
        print(f'\nReversals reaching criterion: {ttc["reached_criterion"].sum()} / {len(ttc)}')
        
        print('\nPost-reversal accuracy (trials 1–10):')
        print(post_rev_acc.groupby('condition')['post_rev_accuracy']
              .agg(['mean', 'std', 'count'])
              .round(3)
              .to_string())
    
    # --- Statistical Tests ---
    stats_results = {}
    
    if 'sham' in conditions and 'active' in conditions:
        from scipy import stats as sp_stats
        
        # 1. Trials-to-criterion: subject-level means, then paired t-test
        ttc_subj = (ttc[ttc['condition'].isin(['sham', 'active'])]
                    .groupby(['subject_id', 'condition'])['trials_to_criterion']
                    .mean()
                    .unstack('condition'))
        
        ttc_paired = ttc_subj.dropna()
        
        if len(ttc_paired) >= 3:
            t_ttc, p_ttc = sp_stats.ttest_rel(ttc_paired['sham'], ttc_paired['active'])
            diff_ttc = ttc_paired['active'] - ttc_paired['sham']
            dz_ttc = diff_ttc.mean() / diff_ttc.std() if diff_ttc.std() > 0 else 0
            
            stats_results['ttc'] = {
                'n': len(ttc_paired),
                't': t_ttc,
                'p': p_ttc,
                'dz': dz_ttc,
                'sham_mean': ttc_paired['sham'].mean(),
                'active_mean': ttc_paired['active'].mean(),
                'diff_mean': diff_ttc.mean(),
            }
            
            if verbose:
                print(f'\n--- Trials-to-Criterion: Sham vs Active (paired) ---')
                print(f'  N = {len(ttc_paired)} subjects')
                print(f'  Sham: M = {ttc_paired["sham"].mean():.2f}, SD = {ttc_paired["sham"].std():.2f}')
                print(f'  Active: M = {ttc_paired["active"].mean():.2f}, SD = {ttc_paired["active"].std():.2f}')
                print(f'  Δ = {diff_ttc.mean():.2f} trials')
                print(f'  t({len(ttc_paired)-1}) = {t_ttc:.3f}, p = {p_ttc:.3f}, dz = {dz_ttc:.3f}')
        
        # 2. Post-reversal accuracy: paired t-test
        acc_paired = post_rev_acc[post_rev_acc['condition'].isin(['sham', 'active'])].copy()
        acc_wide = acc_paired.pivot(index='subject_id', columns='condition', values='post_rev_accuracy')
        acc_wide = acc_wide.dropna()
        
        if len(acc_wide) >= 3:
            t_acc, p_acc = sp_stats.ttest_rel(acc_wide['sham'], acc_wide['active'])
            diff_acc = acc_wide['active'] - acc_wide['sham']
            dz_acc = diff_acc.mean() / diff_acc.std() if diff_acc.std() > 0 else 0
            
            stats_results['post_rev_accuracy'] = {
                'n': len(acc_wide),
                't': t_acc,
                'p': p_acc,
                'dz': dz_acc,
                'sham_mean': acc_wide['sham'].mean(),
                'active_mean': acc_wide['active'].mean(),
                'diff_mean': diff_acc.mean(),
            }
            
            if verbose:
                print(f'\n--- Post-Reversal Accuracy: Sham vs Active (paired) ---')
                print(f'  N = {len(acc_wide)} subjects')
                print(f'  Sham: M = {acc_wide["sham"].mean():.3f}, SD = {acc_wide["sham"].std():.3f}')
                print(f'  Active: M = {acc_wide["active"].mean():.3f}, SD = {acc_wide["active"].std():.3f}')
                print(f'  Δ = {diff_acc.mean():.3f}')
                print(f'  t({len(acc_wide)-1}) = {t_acc:.3f}, p = {p_acc:.3f}, dz = {dz_acc:.3f}')
        
        # 3. Recovery curve: test at key timepoints (trial 1, trial 5)
        if verbose:
            print(f'\n--- Accuracy at Key Timepoints ---')
        
        for trial_pos in [1, 5]:
            trial_data = rev_acc[
                (rev_acc['trial_from_rev'] == trial_pos) &
                (rev_acc['condition'].isin(['sham', 'active']))
            ]
            trial_wide = trial_data.pivot(index='subject_id', columns='condition', values='p_correct')
            trial_wide = trial_wide.dropna()
            
            if len(trial_wide) >= 3:
                t_trial, p_trial = sp_stats.ttest_rel(trial_wide['sham'], trial_wide['active'])
                diff_trial = trial_wide['active'] - trial_wide['sham']
                dz_trial = diff_trial.mean() / diff_trial.std() if diff_trial.std() > 0 else 0
                
                stats_results[f'accuracy_trial_{trial_pos}'] = {
                    'n': len(trial_wide),
                    't': t_trial,
                    'p': p_trial,
                    'dz': dz_trial,
                }
                
                if verbose:
                    sig = '*' if p_trial < 0.05 else ''
                    print(f'  Trial +{trial_pos}: Sham={trial_wide["sham"].mean():.3f}, '
                          f'Active={trial_wide["active"].mean():.3f}, '
                          f't={t_trial:.2f}, p={p_trial:.3f}{sig}')
        
        # --- Age-Related Analyses ---
        age_lookup = data.groupby('subject_id')['age'].first()
        
        if verbose:
            print(f'\n--- Age-Related Analyses ---')
        
        # 4. Age × TTC (sham condition = baseline individual differences)
        ttc_sham_subj = (ttc[ttc['condition'] == 'sham']
                         .groupby('subject_id')['trials_to_criterion']
                         .mean())
        ttc_age_df = pd.DataFrame({
            'age': age_lookup,
            'ttc_sham': ttc_sham_subj
        }).dropna()
        
        if len(ttc_age_df) >= 5:
            r_ttc_age, p_ttc_age = sp_stats.pearsonr(ttc_age_df['age'], ttc_age_df['ttc_sham'])
            stats_results['age_ttc_sham'] = {
                'r': r_ttc_age,
                'p': p_ttc_age,
                'n': len(ttc_age_df),
            }
            if verbose:
                sig = '*' if p_ttc_age < 0.05 else ''
                print(f'  Age × TTC (sham): r = {r_ttc_age:.3f}, p = {p_ttc_age:.3f}{sig}, n = {len(ttc_age_df)}')
        
        # 5. Age × Post-reversal accuracy (sham)
        acc_sham = post_rev_acc[post_rev_acc['condition'] == 'sham'].set_index('subject_id')
        acc_age_df = pd.DataFrame({
            'age': age_lookup,
            'post_rev_acc': acc_sham['post_rev_accuracy']
        }).dropna()
        
        if len(acc_age_df) >= 5:
            r_acc_age, p_acc_age = sp_stats.pearsonr(acc_age_df['age'], acc_age_df['post_rev_acc'])
            stats_results['age_post_rev_acc_sham'] = {
                'r': r_acc_age,
                'p': p_acc_age,
                'n': len(acc_age_df),
            }
            if verbose:
                sig = '*' if p_acc_age < 0.05 else ''
                print(f'  Age × Post-Rev Accuracy (sham): r = {r_acc_age:.3f}, p = {p_acc_age:.3f}{sig}, n = {len(acc_age_df)}')
        
        # 6. Age × tACS effect on TTC (does age moderate stimulation effect?)
        if 'ttc' in stats_results and len(ttc_paired) >= 5:
            delta_ttc = ttc_paired['active'] - ttc_paired['sham']
            delta_age_df = pd.DataFrame({
                'age': age_lookup.loc[delta_ttc.index],
                'delta_ttc': delta_ttc
            }).dropna()
            
            if len(delta_age_df) >= 5:
                r_delta_age, p_delta_age = sp_stats.pearsonr(delta_age_df['age'], delta_age_df['delta_ttc'])
                stats_results['age_delta_ttc'] = {
                    'r': r_delta_age,
                    'p': p_delta_age,
                    'n': len(delta_age_df),
                }
                if verbose:
                    sig = '*' if p_delta_age < 0.05 else ''
                    print(f'  Age × Δ TTC (active-sham): r = {r_delta_age:.3f}, p = {p_delta_age:.3f}{sig}, n = {len(delta_age_df)}')
        
        # 7. Age × tACS effect on post-reversal accuracy
        if 'post_rev_accuracy' in stats_results and len(acc_wide) >= 5:
            delta_acc = acc_wide['active'] - acc_wide['sham']
            delta_acc_age_df = pd.DataFrame({
                'age': age_lookup.loc[delta_acc.index],
                'delta_acc': delta_acc
            }).dropna()
            
            if len(delta_acc_age_df) >= 5:
                r_delta_acc_age, p_delta_acc_age = sp_stats.pearsonr(
                    delta_acc_age_df['age'], delta_acc_age_df['delta_acc'])
                stats_results['age_delta_post_rev_acc'] = {
                    'r': r_delta_acc_age,
                    'p': p_delta_acc_age,
                    'n': len(delta_acc_age_df),
                }
                if verbose:
                    sig = '*' if p_delta_acc_age < 0.05 else ''
                    print(f'  Age × Δ Post-Rev Acc: r = {r_delta_acc_age:.3f}, p = {p_delta_acc_age:.3f}{sig}, n = {len(delta_acc_age_df)}')
    
    # Plots
    accuracy_fig = None
    ttc_fig = None
    
    if show_plots:
        accuracy_fig = plot_reversal_accuracy(rev_acc, conditions, show_fig=True)
        
        age_lookup = data.groupby('subject_id')['age'].first().to_dict()
        ttc_fig = plot_trials_to_criterion(ttc, age_lookup, conditions, show_fig=True)
    
    return {
        'data': data,
        'data_clean': data_clean,
        'rev_acc': rev_acc,
        'post_rev_acc': post_rev_acc,
        'ttc': ttc,
        'stats': stats_results,
        'accuracy_fig': accuracy_fig,
        'ttc_fig': ttc_fig,
    }


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing reversal_analysis module...")
    print("Functions: identify_reversals, compute_reversal_accuracy,")
    print("           compute_trials_to_criterion, run_reversal_analysis")
