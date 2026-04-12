"""
blinding_analysis.py — Blinding integrity analysis for tACS Bandit study

Assesses whether participants could discriminate active from sham stimulation.
Uses signal detection theory metrics (d', criterion) and accuracy measures.

Key outputs:
- Contingency table (actual condition × participant guess)
- Signal detection metrics: d', criterion, hit rate, false alarm rate
- Accuracy statistics with binomial and chi-square tests
- Breakdown by counterbalance order and run position
- Visualizations: confusion matrix, ROC plot, accuracy by condition
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Optional, Dict, List, Tuple
import warnings

from config import (
    SUBJECT_INFO,
    COLOR_SHAM,
    COLOR_ACTIVE,
    COLOR_GREEN,
    COLOR_RED,
    CB_COLORS,
    PLOTLY_TEMPLATE,
    FONT_FAMILY,
)


# =============================================================================
# Data Extraction
# =============================================================================

def extract_blinding_data(
    data: pd.DataFrame,
    subject_info: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Extract run-level blinding data from trial-level behavioral data.
    
    The stim_guess column is constant within each run (asked once at end).
    
    Parameters
    ----------
    data : DataFrame
        Trial-level behavioral data with columns: subject_id, session, run,
        stim_guess, stim_condition, run_type
    subject_info : dict, optional
        Subject registry for filtering
    
    Returns
    -------
    DataFrame
        Run-level blinding data with columns:
        subject_id, session, run, run_type, actual, actual_label,
        guess, guess_label, correct, counterbalance
    """
    if subject_info is None:
        subject_info = SUBJECT_INFO
    
    valid_subjects = set(str(s) for s in subject_info.keys())
    blinding_records = []
    
    # Group by subject, session, run
    group_cols = ['subject_id']
    if 'session' in data.columns:
        group_cols.append('session')
    group_cols.append('run')
    
    for name, run_df in data.groupby(group_cols):
        if len(group_cols) == 3:
            sub, ses, run = name
        else:
            sub, run = name
            ses = 1
        
        if str(sub) not in valid_subjects:
            continue
        
        # Get guess value (should be constant within run)
        if 'stim_guess' not in run_df.columns:
            continue
        
        guess_vals = run_df['stim_guess'].dropna().unique()
        if len(guess_vals) == 0:
            continue
        guess = guess_vals[0]
        
        # Get actual condition
        if 'stim_condition' in run_df.columns:
            condition = run_df['stim_condition'].iloc[0]
        elif 'condition' in run_df.columns:
            condition = run_df['condition'].iloc[0]
        else:
            continue
        
        run_type = run_df['run_type'].iloc[0] if 'run_type' in run_df.columns else ''
        
        # Skip baseline runs
        if condition == 'baseline' or run_type == 'baseline':
            continue
        
        # Encode actual condition
        actual_stim = 1 if condition in ['active', 'stim', 'stimulation'] else 0
        
        # Encode guess
        guess_lower = str(guess).lower().strip()
        if guess_lower in ['yes', 'y', 'stim', 'stimulation', 'active', '1', 'true']:
            guess_stim = 1
        elif guess_lower in ['no', 'n', 'sham', 'control', '0', 'false']:
            guess_stim = 0
        else:
            continue  # Skip ambiguous guesses
        
        blinding_records.append({
            'subject_id': sub,
            'session': ses,
            'run': run,
            'run_type': run_type,
            'actual': actual_stim,
            'actual_label': 'Active' if actual_stim else 'Sham',
            'guess': guess_stim,
            'guess_label': 'Active' if guess_stim else 'Sham',
            'correct': int(actual_stim == guess_stim),
        })
    
    blind_df = pd.DataFrame(blinding_records)
    
    # Add counterbalance info
    if len(blind_df) > 0:
        blind_df['counterbalance'] = blind_df['subject_id'].apply(
            lambda x: subject_info.get(str(x), {}).get('counterbalance', 'Unknown')
        )
    
    return blind_df


# =============================================================================
# Signal Detection Metrics
# =============================================================================

def calc_dprime(
    hit_rate: float,
    fa_rate: float,
    n_signal: int,
    n_noise: int
) -> float:
    """
    Calculate d' (d-prime) sensitivity index with log-linear correction.
    
    Uses Hautus (1995) log-linear correction to avoid infinite values
    when hit rate = 1 or false alarm rate = 0.
    
    Parameters
    ----------
    hit_rate : float
        Proportion of "signal" trials correctly identified
    fa_rate : float
        Proportion of "noise" trials incorrectly identified as signal
    n_signal : int
        Number of signal trials
    n_noise : int
        Number of noise trials
    
    Returns
    -------
    float
        d' value (0 = no discrimination, higher = better discrimination)
    """
    # Log-linear correction (Hautus, 1995)
    hit_adj = (hit_rate * n_signal + 0.5) / (n_signal + 1)
    fa_adj = (fa_rate * n_noise + 0.5) / (n_noise + 1)
    
    return stats.norm.ppf(hit_adj) - stats.norm.ppf(fa_adj)


def calc_criterion(
    hit_rate: float,
    fa_rate: float,
    n_signal: int,
    n_noise: int
) -> float:
    """
    Calculate c (criterion) response bias.
    
    Negative c = bias toward saying "signal" (liberal)
    Positive c = bias toward saying "noise" (conservative)
    
    Parameters
    ----------
    hit_rate, fa_rate : float
        Hit and false alarm rates
    n_signal, n_noise : int
        Number of signal and noise trials
    
    Returns
    -------
    float
        Criterion value
    """
    hit_adj = (hit_rate * n_signal + 0.5) / (n_signal + 1)
    fa_adj = (fa_rate * n_noise + 0.5) / (n_noise + 1)
    
    return -0.5 * (stats.norm.ppf(hit_adj) + stats.norm.ppf(fa_adj))


def compute_blinding_metrics(blind_df: pd.DataFrame) -> Dict:
    """
    Compute all signal detection and accuracy metrics.
    
    Parameters
    ----------
    blind_df : DataFrame
        Blinding data from extract_blinding_data()
    
    Returns
    -------
    dict with keys:
        n_obs, n_subjects, hits, misses, false_alarms, correct_rejections,
        hit_rate, fa_rate, dprime, criterion, accuracy, binom_p, chi2, chi_p
    """
    if len(blind_df) == 0:
        return {}
    
    # Basic counts
    stim_trials = blind_df[blind_df['actual'] == 1]
    sham_trials = blind_df[blind_df['actual'] == 0]
    
    n_stim = len(stim_trials)
    n_sham = len(sham_trials)
    
    # Hit/FA counts
    hits = int(stim_trials['guess'].sum()) if n_stim > 0 else 0
    misses = n_stim - hits
    false_alarms = int(sham_trials['guess'].sum()) if n_sham > 0 else 0
    correct_rejections = n_sham - false_alarms
    
    # Rates
    hit_rate = stim_trials['guess'].mean() if n_stim > 0 else np.nan
    fa_rate = sham_trials['guess'].mean() if n_sham > 0 else np.nan
    
    # Signal detection metrics
    if n_stim > 0 and n_sham > 0 and not np.isnan(hit_rate) and not np.isnan(fa_rate):
        dprime = calc_dprime(hit_rate, fa_rate, n_stim, n_sham)
        criterion = calc_criterion(hit_rate, fa_rate, n_stim, n_sham)
    else:
        dprime = np.nan
        criterion = np.nan
    
    # Accuracy
    accuracy = blind_df['correct'].mean()
    k = int(blind_df['correct'].sum())
    n = len(blind_df)
    
    # Binomial test vs chance (50%)
    try:
        binom_result = stats.binomtest(k, n, 0.5)
        binom_p = binom_result.pvalue
    except AttributeError:
        # Older scipy versions
        binom_p = stats.binom_test(k, n, 0.5)
    
    # Chi-square test
    contingency = pd.crosstab(blind_df['actual'], blind_df['guess'])
    if contingency.shape == (2, 2):
        chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
    else:
        chi2, chi_p, dof = np.nan, np.nan, np.nan
    
    return {
        'n_obs': n,
        'n_subjects': blind_df['subject_id'].nunique(),
        'n_stim': n_stim,
        'n_sham': n_sham,
        'hits': hits,
        'misses': misses,
        'false_alarms': false_alarms,
        'correct_rejections': correct_rejections,
        'hit_rate': hit_rate,
        'fa_rate': fa_rate,
        'dprime': dprime,
        'criterion': criterion,
        'accuracy': accuracy,
        'n_correct': k,
        'binom_p': binom_p,
        'chi2': chi2,
        'chi_p': chi_p,
    }


# =============================================================================
# Summary by Subgroup
# =============================================================================

def compute_metrics_by_group(
    blind_df: pd.DataFrame,
    group_col: str
) -> pd.DataFrame:
    """
    Compute blinding metrics for each level of a grouping variable.
    
    Parameters
    ----------
    blind_df : DataFrame
        Blinding data
    group_col : str
        Column to group by (e.g., 'counterbalance', 'run')
    
    Returns
    -------
    DataFrame
        Metrics for each group level
    """
    records = []
    
    for level in sorted(blind_df[group_col].unique()):
        group_df = blind_df[blind_df[group_col] == level]
        metrics = compute_blinding_metrics(group_df)
        metrics[group_col] = level
        records.append(metrics)
    
    return pd.DataFrame(records)


# =============================================================================
# Printing Functions
# =============================================================================

def print_blinding_summary(blind_df: pd.DataFrame, metrics: Dict):
    """Print formatted blinding analysis summary."""
    print('='*70)
    print('Blinding Integrity Analysis')
    print('='*70)
    
    print(f"\nData: {metrics['n_obs']} run-level observations from {metrics['n_subjects']} participants")
    
    # Contingency table
    contingency = pd.crosstab(
        blind_df['actual_label'],
        blind_df['guess_label'],
        margins=True
    )
    contingency = contingency.reindex(
        index=['Active', 'Sham', 'All'],
        columns=['Active', 'Sham', 'All']
    )
    
    print(f'\n--- Contingency Table ---')
    print('                 Participant Guess')
    print('Actual Condition   Active    Sham    Total')
    print('-'*45)
    for actual in ['Active', 'Sham']:
        row = contingency.loc[actual]
        print(f'{actual:^16} {row["Active"]:>8} {row["Sham"]:>8} {row["All"]:>8}')
    print('-'*45)
    totals = contingency.loc['All']
    print(f'{"Total":^16} {totals["Active"]:>8} {totals["Sham"]:>8} {totals["All"]:>8}')
    
    # Signal detection metrics
    print(f'\n--- Signal Detection Metrics ---')
    print(f"  Hits (say Active | Active):           {metrics['hits']:>3} / {metrics['n_stim']} = {metrics['hit_rate']:.1%}")
    print(f"  Misses (say Sham | Active):           {metrics['misses']:>3} / {metrics['n_stim']} = {1-metrics['hit_rate']:.1%}")
    print(f"  False Alarms (say Active | Sham):     {metrics['false_alarms']:>3} / {metrics['n_sham']} = {metrics['fa_rate']:.1%}")
    print(f"  Correct Rejections (say Sham | Sham): {metrics['correct_rejections']:>3} / {metrics['n_sham']} = {1-metrics['fa_rate']:.1%}")
    
    # d' interpretation
    print(f"\n  d' (sensitivity):  {metrics['dprime']:+.2f}", end='')
    if abs(metrics['dprime']) < 0.5:
        print('  (near zero — no discrimination)')
    elif abs(metrics['dprime']) < 1.0:
        print('  (moderate — some discrimination)')
    else:
        print('  (WARNING: substantial discrimination)')
    
    # Criterion interpretation
    print(f"  c (criterion):     {metrics['criterion']:+.2f}", end='')
    if metrics['criterion'] > 0.3:
        print('  (bias toward "Sham")')
    elif metrics['criterion'] < -0.3:
        print('  (bias toward "Active")')
    else:
        print('  (no strong bias)')
    
    # Accuracy tests
    print(f"\n  Overall accuracy: {metrics['accuracy']:.1%} ({metrics['n_correct']}/{metrics['n_obs']})")
    print(f"  Binomial vs 50%: p = {metrics['binom_p']:.3f}")
    print(f"  χ² = {metrics['chi2']:.2f}, p = {metrics['chi_p']:.3f}")
    
    # By counterbalance
    if 'counterbalance' in blind_df.columns:
        print(f'\n--- By Counterbalance Order ---')
        cb_stats = compute_metrics_by_group(blind_df, 'counterbalance')
        header = "Order    N_runs        Hits          FA       d'      Acc"
        print(header)
        print('-'*60)
        for _, row in cb_stats.iterrows():
            print(f"{row['counterbalance']:<8} {row['n_obs']:>7} "
                  f"{row['hit_rate']:>11.1%} {row['fa_rate']:>11.1%} "
                  f"{row['dprime']:>+7.2f} {row['accuracy']:>8.1%}")
    
    # Summary
    print('\n' + '='*70)
    print('Summary')
    print('='*70)
    
    blinding_intact = (
        metrics['binom_p'] > 0.05 and 
        metrics['chi_p'] > 0.05 and 
        abs(metrics['dprime']) < 1.0
    )
    
    if blinding_intact:
        print('\n✓ Blinding appears INTACT:')
        print(f"  • Accuracy ({metrics['accuracy']:.1%}) not significantly different from chance")
        print(f"  • d' ({metrics['dprime']:.2f}) indicates minimal discrimination ability")
        print(f"  • No significant association between actual condition and guess")
    else:
        print('\n⚠ Blinding may be COMPROMISED:')
        if metrics['binom_p'] < 0.05:
            print(f"  • Accuracy ({metrics['accuracy']:.1%}) significantly different from chance (p = {metrics['binom_p']:.3f})")
        if abs(metrics['dprime']) >= 1.0:
            print(f"  • d' ({metrics['dprime']:.2f}) suggests substantial discrimination ability")
        if metrics['chi_p'] < 0.05:
            print(f"  • Guesses significantly associated with actual condition (χ² p = {metrics['chi_p']:.3f})")


# =============================================================================
# Visualization
# =============================================================================

def plot_confusion_matrix(
    blind_df: pd.DataFrame,
    show_fig: bool = True
) -> go.Figure:
    """Create confusion matrix heatmap."""
    contingency = pd.crosstab(
        blind_df['actual_label'],
        blind_df['guess_label']
    )
    contingency = contingency.reindex(
        index=['Active', 'Sham'],
        columns=['Active', 'Sham']
    ).fillna(0)
    
    conf_matrix = contingency.values
    conf_labels = [['Hit', 'Miss'], ['FA', 'CR']]
    conf_pcts = conf_matrix / conf_matrix.sum() * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Guessed Active', 'Guessed Sham'],
        y=['Actual Active', 'Actual Sham'],
        colorscale=[[0, '#f7fbff'], [0.5, '#6baed6'], [1, '#08519c']],
        showscale=False,
        hovertemplate='%{y} → %{x}<br>Count: %{z}<extra></extra>',
    ))
    
    # Add annotations
    for i in range(2):
        for j in range(2):
            count = conf_matrix[i, j]
            pct = conf_pcts[i, j]
            label = conf_labels[i][j]
            color = 'white' if count > conf_matrix.max() * 0.5 else 'black'
            fig.add_annotation(
                x=j, y=i,
                text=f'<b>{label}</b><br>{int(count)}<br>({pct:.1f}%)',
                showarrow=False,
                font=dict(size=14, color=color),
            )
    
    fig.update_layout(
        height=400, width=450,
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY, size=11),
        title=dict(text='Blinding: Confusion Matrix', font=dict(size=14)),
        xaxis=dict(title='Participant Guess', side='bottom', showgrid=False),
        yaxis=dict(title='Actual Condition', showgrid=False, autorange='reversed'),
    )
    
    if show_fig:
        fig.show(config=dict(toImageButtonOptions=dict(filename='blinding_confusion_matrix')))
    
    return fig


def plot_roc(
    blind_df: pd.DataFrame,
    metrics: Dict,
    show_fig: bool = True
) -> go.Figure:
    """Create ROC-style plot (hit rate vs false alarm rate)."""
    fig = go.Figure()
    
    # Diagonal (chance)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='gray', width=1.5, dash='dash'),
        showlegend=False,
        hoverinfo='skip',
    ))
    
    # Overall point
    fig.add_trace(go.Scatter(
        x=[metrics['fa_rate']],
        y=[metrics['hit_rate']],
        mode='markers',
        marker=dict(size=18, color=COLOR_SHAM, symbol='diamond',
                   line=dict(width=2, color='white')),
        name='Overall',
        hovertemplate=f"Overall<br>Hit: {metrics['hit_rate']:.1%}<br>FA: {metrics['fa_rate']:.1%}<br>d' = {metrics['dprime']:.2f}<extra></extra>",
    ))
    
    # By counterbalance
    if 'counterbalance' in blind_df.columns:
        cb_stats = compute_metrics_by_group(blind_df, 'counterbalance')
        for _, row in cb_stats.iterrows():
            cb = row['counterbalance']
            if pd.notna(row['hit_rate']) and pd.notna(row['fa_rate']):
                fig.add_trace(go.Scatter(
                    x=[row['fa_rate']],
                    y=[row['hit_rate']],
                    mode='markers',
                    marker=dict(size=12, color=CB_COLORS.get(cb, 'gray'), symbol='circle',
                               line=dict(width=1, color='white')),
                    name=f'CB {cb}',
                    hovertemplate=f"CB {cb}<br>Hit: {row['hit_rate']:.1%}<br>FA: {row['fa_rate']:.1%}<br>d' = {row['dprime']:.2f}<extra></extra>",
                ))
    
    # Annotation for diagonal
    fig.add_annotation(
        x=0.75, y=0.65,
        text="d' = 0<br>(no discrimination)",
        showarrow=True,
        arrowhead=2,
        ax=30, ay=30,
        font=dict(size=10, color='gray'),
    )
    
    # Summary annotation
    fig.add_annotation(
        x=0.98, y=0.02, xref='paper', yref='paper',
        text=f"<b>Overall</b><br>d' = {metrics['dprime']:.2f}<br>Acc = {metrics['accuracy']:.1%}<br>χ² p = {metrics['chi_p']:.3f}",
        showarrow=False,
        font=dict(size=10),
        align='right',
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='gray', borderwidth=1,
        xanchor='right', yanchor='bottom',
    )
    
    fig.update_layout(
        height=450, width=500,
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY, size=11),
        title=dict(text='Blinding: Hit Rate vs False Alarm Rate', font=dict(size=14)),
        xaxis=dict(title='False Alarm Rate (say Active | Sham)',
                  range=[-0.05, 1.05], showgrid=False, linecolor='black', linewidth=1),
        yaxis=dict(title='Hit Rate (say Active | Active)',
                  range=[-0.05, 1.05], showgrid=False, linecolor='black', linewidth=1),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.9)'),
    )
    
    if show_fig:
        fig.show(config=dict(toImageButtonOptions=dict(filename='blinding_roc_plot')))
    
    return fig


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_blinding_analysis(
    data: pd.DataFrame,
    show_plots: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run complete blinding integrity analysis.
    
    Parameters
    ----------
    data : DataFrame
        Trial-level behavioral data (e.g., data_clean)
    show_plots : bool
        If True, display visualizations
    verbose : bool
        If True, print summary
    
    Returns
    -------
    dict with keys:
        'blinding_data': DataFrame of run-level blinding data
        'metrics': dict of overall metrics
        'cb_stats': DataFrame of metrics by counterbalance
        'run_stats': DataFrame of metrics by run
        'confusion_fig': confusion matrix figure
        'roc_fig': ROC plot figure
        'blinding_intact': bool indicating if blinding appears intact
    """
    # Suppress RuntimeWarnings from norm.ppf edge cases
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Extract blinding data
    blind_df = extract_blinding_data(data)
    
    if len(blind_df) == 0:
        print('No blinding data available.')
        return {'blinding_data': blind_df, 'metrics': {}, 'blinding_intact': None}
    
    # Compute metrics
    metrics = compute_blinding_metrics(blind_df)
    
    # Print summary
    if verbose:
        print_blinding_summary(blind_df, metrics)
    
    # Subgroup analyses
    cb_stats = compute_metrics_by_group(blind_df, 'counterbalance') if 'counterbalance' in blind_df.columns else None
    run_stats = compute_metrics_by_group(blind_df, 'run') if 'run' in blind_df.columns else None
    
    # Plots
    confusion_fig = plot_confusion_matrix(blind_df, show_fig=show_plots) if show_plots else None
    roc_fig = plot_roc(blind_df, metrics, show_fig=show_plots) if show_plots else None
    
    # Determine if blinding intact
    blinding_intact = (
        metrics.get('binom_p', 0) > 0.05 and
        metrics.get('chi_p', 0) > 0.05 and
        abs(metrics.get('dprime', 999)) < 1.0
    )
    
    return {
        'blinding_data': blind_df,
        'metrics': metrics,
        'cb_stats': cb_stats,
        'run_stats': run_stats,
        'confusion_fig': confusion_fig,
        'roc_fig': roc_fig,
        'blinding_intact': blinding_intact,
    }


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing blinding_analysis module...")
    print(f"Subjects in registry: {len(SUBJECT_INFO)}")
    print("Functions: extract_blinding_data, compute_blinding_metrics, run_blinding_analysis")
