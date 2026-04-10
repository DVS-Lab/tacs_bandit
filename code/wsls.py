"""
wsls.py — Win-Stay/Lose-Shift analysis for tACS Bandit study

Computes and visualizes WSLS (win-stay/lose-shift) behavioral parameters:
- p(stay | win): probability of repeating choice after reward
- p(shift | lose): probability of switching choice after no reward

Key outputs:
- WSLS computation for H1 (sham) and H2 (active vs sham)
- Paired visualization with individual subjects
- Within-subject difference analysis with statistical tests
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
# Age Color Utility (shared with sample_descriptives)
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
# Core WSLS Computation
# =============================================================================

def compute_wsls(
    df: pd.DataFrame,
    groupby_cols: List[str] = ['subject_id', 'condition']
) -> pd.DataFrame:
    """
    Compute WSLS parameters for each grouping.
    
    Parameters
    ----------
    df : DataFrame
        Trial-level data with valid_wsls, prev_reward, stay, shift columns
    groupby_cols : list
        Columns to group by (e.g., ['subject_id'] or ['subject_id', 'condition'])
    
    Returns
    -------
    DataFrame with columns:
        [groupby_cols], p_stay_win, p_shift_lose, n_win_trials, n_lose_trials, n_total_trials
    """
    # Filter to valid WSLS trials
    valid = df[df['valid_wsls']].copy()
    
    results = []
    for name, group in valid.groupby(groupby_cols):
        # Handle single vs multiple groupby columns
        if isinstance(name, tuple):
            row = dict(zip(groupby_cols, name))
        else:
            row = {groupby_cols[0]: name}
        
        # Split by previous outcome
        win_trials = group[group['prev_reward'] == True]
        lose_trials = group[group['prev_reward'] == False]
        
        # Compute WSLS rates
        row['p_stay_win'] = win_trials['stay'].mean() if len(win_trials) > 0 else np.nan
        row['p_shift_lose'] = lose_trials['shift'].mean() if len(lose_trials) > 0 else np.nan
        row['n_win_trials'] = len(win_trials)
        row['n_lose_trials'] = len(lose_trials)
        row['n_total_trials'] = len(group)
        
        results.append(row)
    
    return pd.DataFrame(results)


def compute_wsls_h1_h2(
    data_h1: pd.DataFrame,
    data_h2: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute WSLS for both H1 (sham-only) and H2 (active vs sham) analyses.
    
    Parameters
    ----------
    data_h1 : DataFrame
        Sham-condition data with behavioral exclusions applied
    data_h2 : DataFrame
        Active + sham data with all exclusions applied
    
    Returns
    -------
    tuple of (wsls_h1, wsls_h2)
        wsls_h1: grouped by subject only
        wsls_h2: grouped by subject × condition
    """
    wsls_h1 = compute_wsls(data_h1, groupby_cols=['subject_id'])
    wsls_h2 = compute_wsls(data_h2, groupby_cols=['subject_id', 'condition'])
    
    return wsls_h1, wsls_h2


# =============================================================================
# Statistical Tests
# =============================================================================

def compute_wsls_difference_stats(
    wsls_h2: pd.DataFrame,
    params: List[str] = ['p_stay_win', 'p_shift_lose']
) -> Dict:
    """
    Compute within-subject difference statistics (active - sham).
    
    Parameters
    ----------
    wsls_h2 : DataFrame
        WSLS data with subject_id, condition columns
    params : list
        WSLS parameters to analyze
    
    Returns
    -------
    dict
        Keys are parameter names, values are dicts with:
        n, mean, sd, sem, t, p, dz (Cohen's d), ci_low, ci_high, w_p (Wilcoxon)
    """
    stat_results = {}
    
    for param in params:
        pivot = wsls_h2.pivot(index='subject_id', columns='condition', values=param)
        pivot = pivot.dropna(subset=['sham', 'active'])
        
        diff = pivot['active'] - pivot['sham']
        n = len(diff)
        
        if n < 2:
            stat_results[param] = {'n': n, 'error': 'Insufficient data'}
            continue
        
        mean_diff = diff.mean()
        sd_diff = diff.std(ddof=1)
        sem_diff = sd_diff / np.sqrt(n)
        
        # One-sample t-test: is mean difference different from 0?
        t_stat, p_val = stats.ttest_1samp(diff, 0)
        
        # Effect size: Cohen's dz (mean diff / SD of diffs)
        dz = mean_diff / sd_diff if sd_diff > 0 else 0
        
        # 95% CI for mean difference
        ci_low = mean_diff - 1.96 * sem_diff
        ci_high = mean_diff + 1.96 * sem_diff
        
        # Wilcoxon signed-rank (non-parametric)
        try:
            w_stat, w_p = stats.wilcoxon(diff, alternative='two-sided')
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        
        stat_results[param] = {
            'n': n,
            'mean': mean_diff,
            'sd': sd_diff,
            'sem': sem_diff,
            't': t_stat,
            'p': p_val,
            'dz': dz,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'w_p': w_p,
            'diff_values': diff,  # Store for plotting
        }
    
    return stat_results


def print_wsls_stats(stat_results: Dict):
    """Print formatted WSLS difference statistics."""
    print('='*70)
    print('WSLS Within-Subject Differences: Active − Sham')
    print('='*70)
    
    param_labels = {
        'p_stay_win': 'Δ p(stay | win)',
        'p_shift_lose': 'Δ p(shift | lose)'
    }
    
    for param, label in param_labels.items():
        if param not in stat_results:
            continue
        
        res = stat_results[param]
        if 'error' in res:
            print(f'\n{label}: {res["error"]}')
            continue
        
        print(f'\n{label}:')
        print(f'  N = {res["n"]}')
        print(f'  Mean Δ = {res["mean"]:+.3f} (SD = {res["sd"]:.3f})')
        print(f'  95% CI: [{res["ci_low"]:+.3f}, {res["ci_high"]:+.3f}]')
        print(f'  One-sample t-test: t({res["n"]-1}) = {res["t"]:.2f}, p = {res["p"]:.3f}')
        print(f"  Cohen's dz = {res['dz']:.2f}")
        
        if not np.isnan(res['w_p']):
            print(f'  Wilcoxon signed-rank: p = {res["w_p"]:.3f}')
        
        # Interpretation
        if res['p'] < 0.05:
            direction = 'higher' if res['mean'] > 0 else 'lower'
            print(f'  → Significant: Active {direction} than Sham')
        elif res['p'] < 0.10:
            direction = 'higher' if res['mean'] > 0 else 'lower'
            print(f'  → Trend (p < .10): Active {direction} than Sham')
        else:
            print(f'  → Not significant')


# =============================================================================
# Visualization
# =============================================================================

def plot_wsls_paired(
    wsls_h2: pd.DataFrame,
    age_lookup: Dict,
    show_fig: bool = True
) -> go.Figure:
    """
    Create paired strip plots for WSLS parameters (sham vs active).
    
    Parameters
    ----------
    wsls_h2 : DataFrame
        WSLS data with subject_id, condition, p_stay_win, p_shift_lose
    age_lookup : dict
        {subject_id: age} mapping for coloring
    show_fig : bool
        If True, display the figure
    
    Returns
    -------
    go.Figure
    """
    condition_order = ['sham', 'active']
    cap_w = 0.08
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['p(stay | win)', 'p(shift | lose)']
    )
    
    rng = np.random.default_rng(42)
    
    # Consistent subject-level jitter
    subjects = wsls_h2['subject_id'].unique()
    subject_jitter = {s: rng.uniform(-0.08, 0.08) for s in subjects}
    
    for col, param in enumerate(['p_stay_win', 'p_shift_lose'], start=1):
        plot_data = wsls_h2[wsls_h2['condition'].isin(condition_order)].copy()
        
        # Pivot for pairing
        pivot = plot_data.pivot(index='subject_id', columns='condition', values=param)
        pivot = pivot.dropna(subset=condition_order)
        
        # Draw paired lines (behind points)
        for subj in pivot.index:
            y_vals = pivot.loc[subj, condition_order].values
            jitter = subject_jitter.get(subj, 0)
            x_vals = [0 + jitter, 1 + jitter]
            
            age = age_lookup.get(subj, np.nan)
            base_color = _age_to_rgb(age)
            rgba_color = base_color.replace('rgb', 'rgba').replace(')', ',0.35)')
            
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines',
                line=dict(color=rgba_color, width=1.5),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=col)
        
        # Plot individual points
        for i, cond in enumerate(condition_order):
            cond_data = plot_data[plot_data['condition'] == cond]
            jitter = np.array([subject_jitter.get(s, 0) for s in cond_data['subject_id']])
            
            fig.add_trace(go.Scatter(
                x=i + jitter,
                y=cond_data[param],
                mode='markers',
                marker=dict(
                    size=9, opacity=0.85,
                    color=[_age_to_rgb(age_lookup.get(s, np.nan)) for s in cond_data['subject_id']],
                    line=dict(width=0.5, color='white')
                ),
                text=[f'sub-{s}<br>Age: {age_lookup.get(s, "?")}' for s in cond_data['subject_id']],
                hoverinfo='text+y',
                showlegend=False,
            ), row=1, col=col)
            
            # Mean + SEM
            vals = cond_data[param].dropna()
            if len(vals) > 0:
                mean = vals.mean()
                sem = vals.sem() if len(vals) > 1 else 0
                
                # Error bar
                fig.add_trace(go.Scatter(
                    x=[i, i], y=[mean - sem, mean + sem],
                    mode='lines', line=dict(color='#404040', width=1.5),
                    showlegend=False, hoverinfo='skip'
                ), row=1, col=col)
                
                # Caps
                for y_cap in [mean + sem, mean - sem]:
                    fig.add_trace(go.Scatter(
                        x=[i - cap_w, i + cap_w], y=[y_cap, y_cap],
                        mode='lines', line=dict(color='#404040', width=1.5),
                        showlegend=False, hoverinfo='skip'
                    ), row=1, col=col)
                
                # Mean line
                fig.add_trace(go.Scatter(
                    x=[i - 0.2, i + 0.2], y=[mean, mean],
                    mode='lines', line=dict(color='#404040', width=2),
                    showlegend=False, hoverinfo='text',
                    text=[f'{cond.capitalize()} mean: {mean:.3f}'] * 2,
                ), row=1, col=col)
        
        # Axes
        fig.update_xaxes(
            tickvals=[0, 1],
            ticktext=['Sham', 'Active'],
            row=1, col=col
        )
        fig.update_yaxes(range=[0, 1.02], row=1, col=col)
    
    # Layout
    n_subjects = wsls_h2['subject_id'].nunique()
    
    fig.update_layout(
        height=420, width=800,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=60, r=100, t=50, b=40),
        font=dict(family=FONT_FAMILY, size=14),
    )
    
    fig.update_xaxes(showgrid=False, zeroline=False,
                     showline=True, linewidth=1, linecolor='black',
                     tickfont=dict(size=15))
    fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='black',
                     tickfont=dict(size=15))
    
    # Age colorbar
    dummy_ages = np.linspace(AGE_MIN, AGE_MAX, 50)
    fig.add_trace(go.Scatter(
        x=[None]*50, y=[None]*50, mode='markers',
        marker=dict(size=0.1, color=dummy_ages,
                    colorscale=AGE_COLORSCALE,
                    cmin=AGE_MIN, cmax=AGE_MAX,
                    colorbar=dict(x=1.06, y=0.55, len=0.6, thickness=10,
                                  tickfont=dict(size=15))),
        showlegend=False, hoverinfo='skip',
    ))
    
    fig.add_annotation(text='Age', x=1.11, y=0.93,
                       xref='paper', yref='paper',
                       showarrow=False, font=dict(size=15))
    
    # N annotation
    fig.add_annotation(
        text=f'N = {n_subjects}',
        x=0.02, y=0.02,
        xref='paper', yref='paper',
        showarrow=False,
        font=dict(size=12, color='#666666'),
        xanchor='left'
    )
    
    if show_fig:
        fig.show(config=dict(toImageButtonOptions=dict(filename='wsls_paired')))
    
    return fig


def plot_wsls_difference(
    wsls_h2: pd.DataFrame,
    stat_results: Dict,
    age_lookup: Dict,
    show_fig: bool = True
) -> go.Figure:
    """
    Create difference plot (Active - Sham) for WSLS parameters.
    
    Parameters
    ----------
    wsls_h2 : DataFrame
        WSLS data
    stat_results : dict
        Output from compute_wsls_difference_stats()
    age_lookup : dict
        {subject_id: age} mapping
    show_fig : bool
        If True, display figure
    
    Returns
    -------
    go.Figure
    """
    params = ['p_stay_win', 'p_shift_lose']
    x_labels = ['Δ p(stay | win)', 'Δ p(shift | lose)']
    cap_w = 0.15
    
    fig = go.Figure()
    rng = np.random.default_rng(42)
    
    for i, param in enumerate(params):
        if param not in stat_results or 'diff_values' not in stat_results[param]:
            continue
        
        diff = stat_results[param]['diff_values']
        subjects = diff.index
        
        jitter = rng.uniform(-0.12, 0.12, size=len(diff))
        
        # Individual points
        fig.add_trace(go.Scatter(
            x=np.full(len(diff), i) + jitter,
            y=diff.values,
            mode='markers',
            marker=dict(
                size=10, opacity=0.85,
                color=[_age_to_rgb(age_lookup.get(s, np.nan)) for s in subjects],
                line=dict(width=0.5, color='white')
            ),
            text=[f'sub-{s}<br>Age: {age_lookup.get(s, "?")}<br>Δ: {d:.3f}'
                  for s, d in zip(subjects, diff.values)],
            hoverinfo='text+y',
            showlegend=False
        ))
        
        # Mean + SEM
        vals = diff.values
        if len(vals) > 0:
            mean = np.mean(vals)
            sem = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            
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
                text=[f'Mean Δ: {mean:.3f}'] * 2
            ))
            
            # Stats annotation
            res = stat_results[param]
            p_str = f'p = {res["p"]:.3f}' if res["p"] >= 0.001 else 'p < .001'
            sig_marker = '*' if res["p"] < 0.05 else ('†' if res["p"] < 0.10 else '')
            
            fig.add_annotation(
                x=i, y=max(vals) + 0.08,
                text=f'dz = {res["dz"]:.2f}, {p_str}{sig_marker}',
                showarrow=False,
                font=dict(size=10, color='#404040'),
            )
    
    # Zero reference line
    fig.add_hline(y=0, line=dict(color='black', width=1, dash='dash'))
    
    # Layout
    n_subjects = wsls_h2['subject_id'].nunique()
    
    fig.update_xaxes(
        tickvals=[0, 1], ticktext=x_labels,
        showgrid=False, zeroline=False, showline=True,
        linewidth=1, linecolor='black', tickfont=dict(size=15)
    )
    fig.update_yaxes(
        showgrid=False, showline=True, linewidth=1,
        linecolor='black', tickfont=dict(size=15)
    )
    
    fig.update_layout(
        height=450, width=650,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=60, r=100, t=50, b=40),
        font=dict(family=FONT_FAMILY, size=14),
        title='Within-Subject Differences (Active − Sham)'
    )
    
    # Age colorbar
    dummy_ages = np.linspace(AGE_MIN, AGE_MAX, 50)
    fig.add_trace(go.Scatter(
        x=[None]*50, y=[None]*50, mode='markers',
        marker=dict(size=0.1, color=dummy_ages,
                    colorscale=AGE_COLORSCALE,
                    cmin=AGE_MIN, cmax=AGE_MAX,
                    colorbar=dict(x=1.15, y=0.5, len=0.7, thickness=10,
                                  tickfont=dict(size=15))),
        showlegend=False, hoverinfo='skip'
    ))
    
    fig.add_annotation(text='Age', x=1.2, y=0.9,
                       xref='paper', yref='paper',
                       showarrow=False, font=dict(size=15))
    
    # N annotation
    fig.add_annotation(
        text=f'N = {n_subjects}', x=0.02, y=0.02,
        xref='paper', yref='paper',
        showarrow=False, font=dict(size=12, color='#666666'),
        xanchor='left'
    )
    
    if show_fig:
        fig.show(config=dict(toImageButtonOptions=dict(filename='wsls_difference')))
    
    return fig


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_wsls_analysis(
    data_h1: pd.DataFrame,
    data_h2: pd.DataFrame,
    data: pd.DataFrame,
    show_plots: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run complete WSLS analysis pipeline.
    
    Parameters
    ----------
    data_h1 : DataFrame
        Sham-condition data (for H1)
    data_h2 : DataFrame
        Active + sham data (for H2)
    data : DataFrame
        Full dataset (for age lookup)
    show_plots : bool
        If True, display visualizations
    verbose : bool
        If True, print summaries
    
    Returns
    -------
    dict with keys:
        'wsls_h1': DataFrame of H1 WSLS
        'wsls_h2': DataFrame of H2 WSLS
        'stat_results': dict of statistical test results
        'paired_fig': paired plot figure
        'diff_fig': difference plot figure
    """
    # Compute WSLS
    wsls_h1, wsls_h2 = compute_wsls_h1_h2(data_h1, data_h2)
    
    if verbose:
        print('H1 WSLS (sham, clean runs):')
        print(wsls_h1.to_string(index=False))
        print()
        print('H2 WSLS (active vs. sham, clean runs):')
        print(wsls_h2.to_string(index=False))
    
    # Statistical tests
    stat_results = compute_wsls_difference_stats(wsls_h2)
    
    if verbose:
        print()
        print_wsls_stats(stat_results)
    
    # Age lookup for plotting
    age_lookup = data.groupby('subject_id')['age'].first().to_dict()
    
    # Plots
    paired_fig = None
    diff_fig = None
    
    if show_plots:
        paired_fig = plot_wsls_paired(wsls_h2, age_lookup, show_fig=True)
        diff_fig = plot_wsls_difference(wsls_h2, stat_results, age_lookup, show_fig=True)
    
    return {
        'wsls_h1': wsls_h1,
        'wsls_h2': wsls_h2,
        'stat_results': stat_results,
        'paired_fig': paired_fig,
        'diff_fig': diff_fig,
    }


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing wsls module...")
    print("Functions: compute_wsls, compute_wsls_h1_h2, compute_wsls_difference_stats")
    print("           plot_wsls_paired, plot_wsls_difference, run_wsls_analysis")
