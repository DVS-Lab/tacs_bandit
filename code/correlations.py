"""
correlations.py — Correlation matrix and heatmap utilities for tACS Bandit study

Computes pairwise Pearson correlations among pre-registered and exploratory
variables. Displays as lower-triangle heatmap with significance stars.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, List, Tuple
import plotly.graph_objects as go

from config import (
    PLOTLY_TEMPLATE,
    FONT_FAMILY,
)


# =============================================================================
# Variable Definitions
# =============================================================================

# Primary: pre-registered variables
PRIMARY_VARS = [
    ('age', 'Age'),
    ('global_composite', 'Global Cog.'),
    ('attention_composite', 'Attention'),
    ('memory_composite', 'Ep. Memory'),
    ('speed_composite', 'Proc. Speed'),
    ('spsrq_sr', 'SPSRQ-SR'),
    ('spsrq_sp', 'SPSRQ-SP'),
    ('sham_p_stay_win', 'p(stay|win)'),
    ('sham_p_shift_lose', 'p(shift|lose)'),
    ('sham_alpha', 'α (learn rate)'),
    ('sham_beta', 'β (inv. temp)'),
    ('theta_p95', 'θ Reactivity'),
]

# Secondary: exploratory cognitive/demographic
SECONDARY_VARS = [
    ('ef_composite', 'Exec. Func.'),
    ('scd_q', 'SCD-Q'),
    ('kbit_iq', 'KBIT IQ'),
    ('education_years', 'Education'),
]

# Tertiary: exploratory surveys
TERTIARY_VARS = [
    ('crt_total', 'CRT Total'),
    ('ffmq_total', 'FFMQ Total'),
    ('ffmq_total_no_obs', 'FFMQ (no Obs)'),
    ('ffmq_observe', 'FFMQ Observe'),
    ('ffmq_describe', 'FFMQ Describe'),
    ('ffmq_actaware', 'FFMQ ActAware'),
    ('ffmq_nonjudge', 'FFMQ NonJudge'),
    ('ffmq_nonreact', 'FFMQ NonReact'),
    ('bpsqi_global', 'B-PSQI Global'),
    ('bpsqi_latency', 'Sleep Latency'),
    ('bpsqi_duration', 'Sleep Duration'),
    ('bpsqi_disturbance', 'Sleep Disturb.'),
    ('bpsqi_quality', 'Sleep Quality'),
    ('bbs_avg', 'BBS Avg'),
]


# =============================================================================
# Variable Filtering
# =============================================================================

def get_available_vars(
    var_list: List[Tuple[str, str]],
    df: pd.DataFrame,
    min_n: int = 3
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Filter variable list to those with sufficient data.
    
    Parameters
    ----------
    var_list : list of (column, label) tuples
        Variables to check
    df : DataFrame
        Data
    min_n : int
        Minimum non-null count required
    
    Returns
    -------
    available : list of (column, label) tuples
    dropped : list of labels that were dropped
    """
    available = []
    dropped = []
    
    for var, label in var_list:
        if var in df.columns and df[var].notna().sum() >= min_n:
            available.append((var, label))
        else:
            dropped.append(label)
    
    return available, dropped


def get_all_available_vars(
    df: pd.DataFrame,
    include_tertiary: bool = True,
    min_n: int = 3,
    verbose: bool = True
) -> Tuple[List[Tuple[str, str]], Dict]:
    """
    Get all available variables across primary, secondary, tertiary groups.
    
    Parameters
    ----------
    df : DataFrame
        Data
    include_tertiary : bool
        Whether to include tertiary (exploratory survey) variables
    min_n : int
        Minimum non-null count required
    verbose : bool
        If True, print summary
    
    Returns
    -------
    all_vars : list of (column, label) tuples
    info : dict with counts and dropped lists
    """
    primary_avail, primary_dropped = get_available_vars(PRIMARY_VARS, df, min_n)
    secondary_avail, secondary_dropped = get_available_vars(SECONDARY_VARS, df, min_n)
    
    if include_tertiary:
        tertiary_avail, tertiary_dropped = get_available_vars(TERTIARY_VARS, df, min_n)
    else:
        tertiary_avail, tertiary_dropped = [], []
    
    all_vars = primary_avail + secondary_avail + tertiary_avail
    
    info = {
        'n_primary': len(primary_avail),
        'n_secondary': len(secondary_avail),
        'n_tertiary': len(tertiary_avail),
        'n_total': len(all_vars),
        'primary_dropped': primary_dropped,
        'secondary_dropped': secondary_dropped,
        'tertiary_dropped': tertiary_dropped,
    }
    
    if verbose:
        if primary_dropped:
            print(f'Primary variables excluded (n<{min_n}): {primary_dropped}')
        if secondary_dropped:
            print(f'Secondary variables excluded (n<{min_n}): {secondary_dropped}')
        if tertiary_dropped and include_tertiary:
            print(f'Tertiary variables excluded (n<{min_n}): {tertiary_dropped}')
        
        print(f'\nVariables included: {info["n_total"]} total')
        print(f'  Primary: {info["n_primary"]}')
        print(f'  Secondary: {info["n_secondary"]}')
        if include_tertiary:
            print(f'  Tertiary: {info["n_tertiary"]}')
    
    return all_vars, info


# =============================================================================
# Correlation Computation
# =============================================================================

def compute_correlation_matrix(
    df: pd.DataFrame,
    variables: List[Tuple[str, str]]
) -> Dict:
    """
    Compute pairwise Pearson correlation matrix with p-values.
    
    Parameters
    ----------
    df : DataFrame
        Data
    variables : list of (column, label) tuples
        Variables to correlate
    
    Returns
    -------
    dict with keys:
        'r_matrix': correlation coefficients (n_vars × n_vars)
        'p_matrix': p-values
        'n_matrix': pairwise sample sizes
        'var_names': column names
        'var_labels': display labels
    """
    var_names = [v[0] for v in variables]
    var_labels = [v[1] for v in variables]
    n_vars = len(var_names)
    
    r_matrix = np.full((n_vars, n_vars), np.nan)
    p_matrix = np.full((n_vars, n_vars), np.nan)
    n_matrix = np.full((n_vars, n_vars), 0, dtype=int)
    
    for i in range(n_vars):
        for j in range(n_vars):
            # Coerce to numeric
            xi = pd.to_numeric(df[var_names[i]], errors='coerce')
            xj = pd.to_numeric(df[var_names[j]], errors='coerce')
            
            mask = xi.notna() & xj.notna()
            n_ij = mask.sum()
            n_matrix[i, j] = n_ij
            
            if n_ij >= 3:
                r, p = stats.pearsonr(xi[mask], xj[mask])
                r_matrix[i, j] = r
                p_matrix[i, j] = p
    
    return {
        'r_matrix': r_matrix,
        'p_matrix': p_matrix,
        'n_matrix': n_matrix,
        'var_names': var_names,
        'var_labels': var_labels,
    }


def get_significant_correlations(
    corr_result: Dict,
    alpha: float = 0.05,
    min_r: float = 0.0
) -> pd.DataFrame:
    """
    Extract significant correlations from correlation matrix.
    
    Parameters
    ----------
    corr_result : dict
        Output from compute_correlation_matrix()
    alpha : float
        Significance threshold
    min_r : float
        Minimum absolute correlation to include
    
    Returns
    -------
    DataFrame with significant pairs
    """
    r_matrix = corr_result['r_matrix']
    p_matrix = corr_result['p_matrix']
    n_matrix = corr_result['n_matrix']
    labels = corr_result['var_labels']
    
    n_vars = len(labels)
    results = []
    
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if (not np.isnan(p_matrix[i, j]) and
                p_matrix[i, j] < alpha and
                abs(r_matrix[i, j]) >= min_r):
                
                results.append({
                    'var1': labels[i],
                    'var2': labels[j],
                    'r': r_matrix[i, j],
                    'p': p_matrix[i, j],
                    'n': n_matrix[i, j],
                })
    
    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values('p')
    
    return df


# =============================================================================
# Heatmap Visualization
# =============================================================================

def plot_correlation_heatmap(
    corr_result: Dict,
    group_sizes: Optional[Dict] = None,
    title: str = 'Correlation Matrix',
    show_fig: bool = True
) -> go.Figure:
    """
    Create lower-triangle correlation heatmap.
    
    Parameters
    ----------
    corr_result : dict
        Output from compute_correlation_matrix()
    group_sizes : dict, optional
        Keys 'n_primary', 'n_secondary', 'n_tertiary' for divider lines
    title : str
        Figure title
    show_fig : bool
        If True, display figure
    
    Returns
    -------
    go.Figure : Plotly figure object
    """
    r_matrix = corr_result['r_matrix']
    p_matrix = corr_result['p_matrix']
    n_matrix = corr_result['n_matrix']
    var_labels = corr_result['var_labels']
    n_vars = len(var_labels)
    
    # Lower triangle mask
    tri_mask = np.triu(np.ones_like(r_matrix, dtype=bool), k=0)
    r_display = r_matrix.copy()
    r_display[tri_mask] = np.nan
    
    # Annotation text with significance stars
    annot_text = []
    for i in range(n_vars):
        row = []
        for j in range(n_vars):
            if tri_mask[i, j]:
                row.append('')
            elif np.isnan(r_matrix[i, j]):
                row.append('—')
            else:
                stars = ''
                if p_matrix[i, j] < 0.001:
                    stars = '***'
                elif p_matrix[i, j] < 0.01:
                    stars = '**'
                elif p_matrix[i, j] < 0.05:
                    stars = '*'
                row.append(f'{r_matrix[i, j]:.2f}{stars}')
        annot_text.append(row)
    
    # Hover text
    hover_text = []
    for i in range(n_vars):
        row = []
        for j in range(n_vars):
            if tri_mask[i, j]:
                row.append('')
            else:
                r_val = f'{r_matrix[i, j]:.3f}' if not np.isnan(r_matrix[i, j]) else 'N/A'
                p_val = f'{p_matrix[i, j]:.4f}' if not np.isnan(p_matrix[i, j]) else 'N/A'
                row.append(f'{var_labels[i]} × {var_labels[j]}<br>'
                          f'r = {r_val}, p = {p_val}<br>n = {int(n_matrix[i, j])}')
        hover_text.append(row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=r_display,
        x=var_labels,
        y=var_labels,
        colorscale=[[0, '#1565C0'], [0.5, '#FFFFFF'], [1, '#E65100']],
        zmid=0,
        zmin=-1,
        zmax=1,
        text=annot_text,
        texttemplate='%{text}',
        textfont=dict(size=9),
        hovertext=hover_text,
        hoverinfo='text',
        colorbar=dict(title='r', thickness=10, len=0.5, tickfont=dict(size=11)),
        xgap=1,
        ygap=1,
    ))
    
    # Add divider lines between groups
    if group_sizes:
        n_primary = group_sizes.get('n_primary', 0)
        n_secondary = group_sizes.get('n_secondary', 0)
        n_tertiary = group_sizes.get('n_tertiary', 0)
        
        divider_positions = []
        if n_secondary > 0:
            divider_positions.append(n_primary - 0.5)
        if n_tertiary > 0:
            divider_positions.append(n_primary + n_secondary - 0.5)
        
        for pos in divider_positions:
            fig.add_hline(y=pos, line_dash='dot', line_color='gray', line_width=1.5)
            fig.add_vline(x=pos, line_dash='dot', line_color='gray', line_width=1.5)
        
        # Legend annotation
        legend_parts = []
        if n_primary > 0:
            legend_parts.append(f'Primary (1-{n_primary})')
        if n_secondary > 0:
            legend_parts.append(f'Secondary ({n_primary+1}-{n_primary+n_secondary})')
        if n_tertiary > 0:
            legend_parts.append(f'Tertiary ({n_primary+n_secondary+1}-{n_vars})')
        
        if len(legend_parts) > 1:
            fig.add_annotation(
                text='Dashed lines separate: ' + ', '.join(legend_parts),
                xref='paper', yref='paper',
                x=0.5, y=-0.12,
                showarrow=False,
                font=dict(size=9, color='gray')
            )
    
    # Layout
    fig.update_layout(
        height=max(700, 32 * n_vars + 200),
        width=max(850, 32 * n_vars + 200),
        template=PLOTLY_TEMPLATE,
        margin=dict(l=130, r=60, t=50, b=130),
        font=dict(family=FONT_FAMILY, size=11),
        xaxis=dict(side='bottom', tickangle=-45),
        yaxis=dict(autorange='reversed'),
        title=dict(text=title, font=dict(size=14)),
    )
    
    if show_fig:
        fig.show(config=dict(toImageButtonOptions=dict(filename='correlation_heatmap')))
    
    return fig


def print_pairwise_n_summary(corr_result: Dict) -> None:
    """Print summary of pairwise sample sizes."""
    n_matrix = corr_result['n_matrix']
    var_labels = corr_result['var_labels']
    
    # Lower triangle only
    lower_tri = np.tril(n_matrix, k=-1).astype(float)
    lower_tri[lower_tri == 0] = np.nan
    
    print(f'\nPairwise sample sizes (min/max):')
    print(f'  Min N: {int(np.nanmin(lower_tri))}')
    print(f'  Max N: {int(np.nanmax(lower_tri))}')
    print(f'  Median N: {int(np.nanmedian(lower_tri))}')


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_correlation_analysis(
    subj_df: pd.DataFrame,
    include_tertiary: bool = True,
    show_plot: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run complete correlation analysis.
    
    Parameters
    ----------
    subj_df : DataFrame
        Subject-level data
    include_tertiary : bool
        Whether to include tertiary (exploratory) variables
    show_plot : bool
        If True, display heatmap
    verbose : bool
        If True, print summaries
    
    Returns
    -------
    dict with keys:
        'variables': list of (column, label) tuples
        'info': variable availability info
        'corr_result': correlation matrix data
        'significant': DataFrame of significant pairs
        'fig': Plotly figure (if show_plot)
    """
    results = {}
    
    # Get available variables
    variables, info = get_all_available_vars(subj_df, include_tertiary, verbose=verbose)
    results['variables'] = variables
    results['info'] = info
    
    if len(variables) < 2:
        if verbose:
            print('\nInsufficient variables for correlation analysis.')
        return results
    
    # Compute correlations
    corr_result = compute_correlation_matrix(subj_df, variables)
    results['corr_result'] = corr_result
    
    # Get significant pairs
    sig_df = get_significant_correlations(corr_result)
    results['significant'] = sig_df
    
    if verbose:
        print_pairwise_n_summary(corr_result)
        
        if len(sig_df) > 0:
            print(f'\nSignificant correlations (p < .05): {len(sig_df)}')
            for _, row in sig_df.head(10).iterrows():
                print(f"  {row['var1']} × {row['var2']}: r = {row['r']:.3f}, p = {row['p']:.4f}")
            if len(sig_df) > 10:
                print(f'  ... and {len(sig_df) - 10} more')
    
    # Plot
    if show_plot:
        fig = plot_correlation_heatmap(
            corr_result,
            group_sizes=info,
            title='Pre-registered & Exploratory Variable Correlations',
            show_fig=True
        )
        results['fig'] = fig
    
    return results


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing correlations module...")
    print("Functions: compute_correlation_matrix, plot_correlation_heatmap")
    print("           get_significant_correlations, run_correlation_analysis")
    print(f"Variable sets: {len(PRIMARY_VARS)} primary, {len(SECONDARY_VARS)} secondary, {len(TERTIARY_VARS)} tertiary")
