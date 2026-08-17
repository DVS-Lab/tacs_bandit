"""
efield_analysis.py — Electric field analysis for tACS Bandit study

Analyzes whether simulated electric field strength in the target ROI
(left DLPFC / F3) predicts tACS efficacy and relates to age.

Key questions:
1. Does e-field strength moderate stimulation effects? (Δ DV ~ mean_magnE)
2. Does age predict e-field strength? (anatomical differences / atrophy)

Key outputs:
- E-field descriptives
- Age × e-field correlation
- E-field moderation of stimulation effects for multiple DVs
- Visualization of relationships
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
    COLOR_GOLD,
    AGE_MIN,
    AGE_MAX,
    PLOTLY_TEMPLATE,
    FONT_FAMILY,
)


# =============================================================================
# Constants
# =============================================================================

# Default e-field CSV path (can be overridden).
#
# Absolute, via config, deliberately. This was a bare relative filename, which
# resolved against the caller's working directory -- so running from code/
# silently picked up a stale dissertation-era extract (35 subjects, values 2x
# too large under the old peak-to-peak convention, no t1_only flag) instead of
# the current 66-subject file. That extract now lives in data/archive/.
from config import EFIELD_CSV_PATH

EFIELD_CSV_DEFAULT = EFIELD_CSV_PATH

# Primary e-field metric
EFIELD_METRIC = 'mean_magnE'

# E-field units
EFIELD_UNITS = 'V/m'


# =============================================================================
# Age Color Utility
# =============================================================================

def _get_age_colormap():
    """Get matplotlib colormap for age gradient (blue → gold)."""
    import matplotlib.colors as mcolors
    return mcolors.LinearSegmentedColormap.from_list('blue_gold', ['#1565C0', '#FFB300'])


def _age_to_rgb(age: float, age_min: float = AGE_MIN, age_max: float = AGE_MAX) -> str:
    """Convert age to RGB color string for Plotly."""
    cmap = _get_age_colormap()
    norm = np.clip((age - age_min) / (age_max - age_min), 0, 1)
    rgba = cmap(norm)
    return f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'


# =============================================================================
# Data Loading
# =============================================================================

def load_efield_data(
    filepath: str = EFIELD_CSV_DEFAULT,
    cond_df: pd.DataFrame = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load e-field simulation data and merge with subject info.
    
    Parameters
    ----------
    filepath : str
        Path to e-field CSV file
    cond_df : DataFrame, optional
        Condition-level data containing age (from accuracy_analysis)
    verbose : bool
        Print summary
    
    Returns
    -------
    DataFrame with e-field data and age
    """
    # Load e-field data
    efield = pd.read_csv(filepath)
    
    # Ensure subject_id is int for merging
    efield['subject_id'] = efield['subject_id'].astype(int)
    
    # Add age from cond_df if provided
    if cond_df is not None and 'age' in cond_df.columns:
        # Handle case where subject_id might be index or column
        if 'subject_id' in cond_df.columns:
            # Ensure consistent int type for subject_id
            cond_df_copy = cond_df.copy()
            cond_df_copy['subject_id'] = cond_df_copy['subject_id'].astype(int)
            age_lookup = cond_df_copy.groupby('subject_id')['age'].first().to_dict()
        else:
            # subject_id is the index
            age_lookup = cond_df.groupby(cond_df.index)['age'].first().to_dict()
            # Convert keys to int for matching
            age_lookup = {int(k): v for k, v in age_lookup.items()}
        
        efield['age'] = efield['subject_id'].map(age_lookup)
    else:
        efield['age'] = np.nan
    
    # Add counterbalance from SUBJECT_INFO
    cb_lookup = {int(s): info['counterbalance'] for s, info in SUBJECT_INFO.items()}
    efield['counterbalance'] = efield['subject_id'].map(cb_lookup)
    
    if verbose:
        print("="*60)
        print("E-FIELD DATA SUMMARY")
        print("="*60)
        print(f"\nSubjects with e-field data: {len(efield)}")
        print(f"Subjects in SUBJECT_INFO: {len(SUBJECT_INFO)}")
        
        # Check overlap
        efield_subjs = set(efield['subject_id'])
        study_subjs = set(int(s) for s in SUBJECT_INFO.keys())
        overlap = efield_subjs & study_subjs
        missing_efield = study_subjs - efield_subjs
        
        print(f"Overlap: {len(overlap)}")
        if missing_efield:
            print(f"Missing e-field data: {sorted(missing_efield)}")
        
        print(f"\n{EFIELD_METRIC} ({EFIELD_UNITS}):")
        print(f"  Mean: {efield[EFIELD_METRIC].mean():.4f}")
        print(f"  SD: {efield[EFIELD_METRIC].std():.4f}")
        print(f"  Range: {efield[EFIELD_METRIC].min():.4f} - {efield[EFIELD_METRIC].max():.4f}")
        
        n_with_age = efield['age'].notna().sum()
        if n_with_age > 0:
            print(f"\nSubjects with age data: {n_with_age}")
            print(f"Age range: {efield['age'].min():.0f} - {efield['age'].max():.0f}")
        else:
            print("\nAge data not available (provide cond_df with age column)")
    
    return efield


# =============================================================================
# Age × E-field Analysis
# =============================================================================

def test_age_efield_relationship(
    efield_df: pd.DataFrame,
    verbose: bool = True
) -> Dict:
    """
    Test whether age predicts e-field strength.
    
    Older adults may have weaker e-fields due to cortical atrophy,
    increased CSF, or skull thickness changes.
    
    Parameters
    ----------
    efield_df : DataFrame
        Output from load_efield_data()
    verbose : bool
        Print results
    
    Returns
    -------
    dict with correlation results
    """
    df = efield_df[['subject_id', 'age', EFIELD_METRIC]].dropna()
    
    if len(df) < 5:
        print("Insufficient data for age × e-field analysis")
        return {}
    
    r, p = stats.pearsonr(df['age'], df[EFIELD_METRIC])
    
    # Also compute Spearman for robustness
    rho, p_spearman = stats.spearmanr(df['age'], df[EFIELD_METRIC])
    
    results = {
        'r': r,
        'p': p,
        'rho': rho,
        'p_spearman': p_spearman,
        'n': len(df),
        'age_range': (df['age'].min(), df['age'].max()),
        'efield_range': (df[EFIELD_METRIC].min(), df[EFIELD_METRIC].max()),
        'data': df,
    }
    
    if verbose:
        print("\n" + "="*60)
        print("AGE × E-FIELD RELATIONSHIP")
        print("="*60)
        sig = '*' if p < 0.05 else ''
        print(f"\nPearson r = {r:.3f}, p = {p:.3f}{sig}")
        print(f"Spearman ρ = {rho:.3f}, p = {p_spearman:.3f}")
        print(f"N = {len(df)}")
        
        if r < 0:
            print("\nInterpretation: Older adults show weaker e-fields")
        else:
            print("\nInterpretation: No evidence of age-related e-field reduction")
    
    return results


def plot_age_vs_efield(
    efield_df: pd.DataFrame,
    show_fig: bool = True
) -> go.Figure:
    """
    Scatter plot: Age (x) vs E-field strength (y).
    
    Parameters
    ----------
    efield_df : DataFrame
        Output from load_efield_data()
    show_fig : bool
        Display figure
    
    Returns
    -------
    go.Figure
    """
    df = efield_df[['subject_id', 'age', EFIELD_METRIC]].dropna()
    
    if len(df) < 2:
        print("Insufficient data for age × e-field plot (need at least 2 subjects with age)")
        return None
    
    r, p = stats.pearsonr(df['age'], df[EFIELD_METRIC])
    
    fig = go.Figure()
    
    # Scatter points colored by age
    for _, row in df.iterrows():
        color = _age_to_rgb(row['age'])
        
        fig.add_trace(go.Scatter(
            x=[row['age']],
            y=[row[EFIELD_METRIC]],
            mode='markers',
            marker=dict(size=12, color=color, line=dict(width=1, color='white')),
            showlegend=False,
            hovertemplate=f"Sub-{int(row['subject_id'])}<br>"
                          f"Age: {row['age']:.0f}<br>"
                          f"E-field: {row[EFIELD_METRIC]:.4f} {EFIELD_UNITS}"
        ))
    
    # Regression line
    slope, intercept = np.polyfit(df['age'], df[EFIELD_METRIC], 1)
    x_line = np.linspace(df['age'].min() - 2, df['age'].max() + 2, 100)
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=intercept + slope * x_line,
        mode='lines',
        line=dict(color='#404040', width=2, dash='dash'),
        showlegend=False
    ))
    
    # Stats annotation
    sig = '*' if p < 0.05 else ''
    fig.add_annotation(
        x=0.98, y=0.98, xref='paper', yref='paper',
        text=f"r = {r:.3f}, p = {p:.3f}{sig}<br>N = {len(df)}",
        showarrow=False,
        font=dict(size=12),
        bgcolor='rgba(255,255,255,0.8)',
        xanchor='right', yanchor='top'
    )
    
    fig.update_layout(
        title=f"Age vs. E-field Strength in Left DLPFC<br>"
              f"<sup>Does cortical atrophy reduce field strength?</sup>",
        xaxis_title="Age (years)",
        yaxis_title=f"Mean E-field Magnitude ({EFIELD_UNITS})",
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=500,
        width=550
    )
    
    if show_fig:
        fig.show()
    
    return fig


# =============================================================================
# E-field Moderation of Stimulation Effects
# =============================================================================

def compute_efield_moderation(
    efield_df: pd.DataFrame,
    cond_df: pd.DataFrame = None,
    wsls_h2: pd.DataFrame = None,
    rw_df: pd.DataFrame = None,
    ddm_df: pd.DataFrame = None,
    verbose: bool = True
) -> Dict:
    """
    Test whether e-field strength predicts stimulation efficacy.
    
    For each DV, tests: Δ(active - sham) ~ mean_magnE
    
    Parameters
    ----------
    efield_df : DataFrame
        Output from load_efield_data()
    cond_df : DataFrame, optional
        Condition-level accuracy/win_rate (from accuracy_analysis)
    wsls_h2 : DataFrame, optional
        WSLS results with sham/active columns
    rw_df : DataFrame, optional
        R-W parameters by subject × condition
    ddm_df : DataFrame, optional
        DDM parameters by subject × condition
    verbose : bool
        Print results
    
    Returns
    -------
    dict with moderation results for each DV
    """
    results = {}
    
    # Prepare e-field lookup with int index
    efield_lookup = efield_df.set_index('subject_id')[[EFIELD_METRIC, 'age']].copy()
    efield_lookup.index = efield_lookup.index.astype(int)
    
    # -------------------------------------------------------------------------
    # Accuracy / Win Rate moderation
    # -------------------------------------------------------------------------
    if cond_df is not None:
        # Ensure subject_id is int before pivoting
        cond_df_copy = cond_df.copy()
        cond_df_copy['subject_id'] = cond_df_copy['subject_id'].astype(int)
        
        for metric in ['accuracy', 'win_rate']:
            pivot = cond_df_copy.pivot(index='subject_id', columns='condition', values=metric)
            
            if 'sham' not in pivot.columns or 'active' not in pivot.columns:
                continue
            
            paired = pivot[['sham', 'active']].dropna()
            paired['delta'] = paired['active'] - paired['sham']
            
            # Merge with e-field (both should have int index now)
            merged = paired.join(efield_lookup, how='inner').dropna()
            
            if len(merged) >= 5:
                r, p = stats.pearsonr(merged[EFIELD_METRIC], merged['delta'])
                
                results[metric] = {
                    'r': r,
                    'p': p,
                    'n': len(merged),
                    'efield_mean': merged[EFIELD_METRIC].mean(),
                    'delta_mean': merged['delta'].mean(),
                    'data': merged,
                }
    
    # -------------------------------------------------------------------------
    # WSLS moderation
    # -------------------------------------------------------------------------
    if wsls_h2 is not None:
        # wsls_h2 is in long format: subject_id, condition, p_stay_win, p_shift_lose
        for dv in ['p_stay_win', 'p_shift_lose']:
            if dv not in wsls_h2.columns:
                continue
            
            # Ensure subject_id is int
            wsls_copy = wsls_h2.copy()
            wsls_copy['subject_id'] = wsls_copy['subject_id'].astype(int)
            
            # Pivot to wide format
            pivot = wsls_copy.pivot(index='subject_id', columns='condition', values=dv)
            
            if 'sham' not in pivot.columns or 'active' not in pivot.columns:
                continue
            
            paired = pivot[['sham', 'active']].dropna()
            paired['delta'] = paired['active'] - paired['sham']
            
            # Merge with e-field
            merged = paired.join(efield_lookup, how='inner').dropna()
            
            if len(merged) >= 5:
                r, p = stats.pearsonr(merged[EFIELD_METRIC], merged['delta'])
                results[f'wsls_{dv}'] = {'r': r, 'p': p, 'n': len(merged), 'data': merged}
    
    # -------------------------------------------------------------------------
    # R-W moderation
    # -------------------------------------------------------------------------
    if rw_df is not None:
        for param in ['alpha', 'beta']:
            # Try to construct paired data
            if 'condition' in rw_df.columns and param in rw_df.columns:
                sham_data = rw_df[rw_df['condition'] == 'sham'][['subject_id', param]].set_index('subject_id')
                active_data = rw_df[rw_df['condition'] == 'active'][['subject_id', param]].set_index('subject_id')
                
                if len(sham_data) > 0 and len(active_data) > 0:
                    rw_paired = sham_data.join(active_data, lsuffix='_sham', rsuffix='_active', how='inner')
                    rw_paired['delta'] = rw_paired[f'{param}_active'] - rw_paired[f'{param}_sham']
                    
                    merged = rw_paired.join(efield_lookup, how='inner').dropna()
                    
                    if len(merged) >= 5:
                        r, p = stats.pearsonr(merged[EFIELD_METRIC], merged['delta'])
                        results[f'rw_{param}'] = {'r': r, 'p': p, 'n': len(merged), 'data': merged}
    
    # -------------------------------------------------------------------------
    # DDM moderation
    # -------------------------------------------------------------------------
    if ddm_df is not None:
        for param in ['v', 'a', 't']:
            sham_col = f'{param}_sham'
            active_col = f'{param}_active'
            
            if sham_col in ddm_df.columns and active_col in ddm_df.columns:
                ddm_data = ddm_df[[sham_col, active_col]].copy()
                ddm_data['delta'] = ddm_data[active_col] - ddm_data[sham_col]
                
                merged = ddm_data.join(efield_lookup, how='inner').dropna()
                
                if len(merged) >= 5:
                    r, p = stats.pearsonr(merged[EFIELD_METRIC], merged['delta'])
                    results[f'ddm_{param}'] = {'r': r, 'p': p, 'n': len(merged), 'data': merged}
    
    if verbose:
        print("\n" + "="*60)
        print("E-FIELD MODERATION OF STIMULATION EFFECTS")
        print("="*60)
        print(f"\nQuestion: Does e-field strength predict tACS efficacy?")
        print(f"Test: Δ(active - sham) ~ {EFIELD_METRIC}")
        
        if results:
            print(f"\n{'DV':<25} {'r':>8} {'p':>10} {'n':>5} {'sig':>5}")
            print("-"*55)
            
            for dv, r in sorted(results.items()):
                sig = '*' if r['p'] < 0.05 else ''
                print(f"{dv:<25} {r['r']:>8.3f} {r['p']:>10.3f} {r['n']:>5} {sig:>5}")
        else:
            print("\nNo DVs available for moderation analysis.")
    
    return results


def plot_efield_vs_delta(
    moderation_results: Dict,
    dv: str = 'accuracy',
    show_fig: bool = True
) -> go.Figure:
    """
    Scatter plot: E-field strength (x) vs stimulation effect (y).
    
    Parameters
    ----------
    moderation_results : dict
        Output from compute_efield_moderation()
    dv : str
        Which DV to plot
    show_fig : bool
        Display figure
    
    Returns
    -------
    go.Figure
    """
    if dv not in moderation_results:
        print(f"No moderation results for {dv}")
        return None
    
    res = moderation_results[dv]
    data = res['data']
    
    fig = go.Figure()
    
    # Scatter points colored by age
    for subj in data.index:
        age = data.loc[subj, 'age']
        color = _age_to_rgb(age)
        
        fig.add_trace(go.Scatter(
            x=[data.loc[subj, EFIELD_METRIC]],
            y=[data.loc[subj, 'delta']],
            mode='markers',
            marker=dict(size=12, color=color, line=dict(width=1, color='white')),
            showlegend=False,
            hovertemplate=f"Sub-{subj}<br>Age: {age:.0f}<br>"
                          f"E-field: {data.loc[subj, EFIELD_METRIC]:.4f}<br>"
                          f"Δ{dv}: {data.loc[subj, 'delta']:+.3f}"
        ))
    
    # Regression line
    x = data[EFIELD_METRIC]
    y = data['delta']
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min() - 0.01, x.max() + 0.01, 100)
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=intercept + slope * x_line,
        mode='lines',
        line=dict(color='#404040', width=2, dash='dash'),
        showlegend=False
    ))
    
    # Reference line at y=0
    fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.7)
    
    # Stats annotation
    sig = '*' if res['p'] < 0.05 else ''
    fig.add_annotation(
        x=0.98, y=0.98, xref='paper', yref='paper',
        text=f"r = {res['r']:.3f}, p = {res['p']:.3f}{sig}<br>N = {res['n']}",
        showarrow=False,
        font=dict(size=12),
        bgcolor='rgba(255,255,255,0.8)',
        xanchor='right', yanchor='top'
    )
    
    # Labels
    dv_label = dv.replace('_', ' ').title()
    
    fig.update_layout(
        title=f"E-field Strength Moderates tACS Effect<br>"
              f"<sup>Mean E-field vs. Stimulation Effect (Active − Sham) on {dv_label}</sup>",
        xaxis_title=f"Mean E-field Magnitude ({EFIELD_UNITS})",
        yaxis_title=f"Δ {dv_label} (Active − Sham)",
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=500,
        width=550
    )
    
    if show_fig:
        fig.show()
    
    return fig


# =============================================================================
# Summary Visualization
# =============================================================================

def plot_efield_moderation_summary(
    moderation_results: Dict,
    show_fig: bool = True
) -> go.Figure:
    """
    Forest plot summarizing e-field moderation across all DVs.
    
    Parameters
    ----------
    moderation_results : dict
        Output from compute_efield_moderation()
    show_fig : bool
        Display figure
    
    Returns
    -------
    go.Figure
    """
    if not moderation_results:
        print("No moderation results to plot")
        return None
    
    # Sort by effect size
    dvs = sorted(moderation_results.keys(), key=lambda x: moderation_results[x]['r'], reverse=True)
    
    rs = [moderation_results[dv]['r'] for dv in dvs]
    ps = [moderation_results[dv]['p'] for dv in dvs]
    ns = [moderation_results[dv]['n'] for dv in dvs]
    
    # Colors based on significance
    colors = ['#2E7D32' if p < 0.05 else '#1565C0' if p < 0.10 else '#757575' for p in ps]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=rs,
        y=dvs,
        orientation='h',
        marker=dict(color=colors),
        text=[f"r={r:.2f}, p={p:.3f}" for r, p in zip(rs, ps)],
        textposition='outside',
        hovertemplate="%{y}<br>r = %{x:.3f}<extra></extra>"
    ))
    
    # Reference line at 0
    fig.add_vline(x=0, line_dash='solid', line_color='black', line_width=1)
    
    fig.update_layout(
        title=f"E-field Moderation of tACS Effects Across DVs<br>"
              f"<sup>Correlation: Δ(active - sham) ~ {EFIELD_METRIC}</sup>",
        xaxis_title="Correlation (r)",
        yaxis_title="",
        xaxis=dict(range=[-0.6, 0.6]),
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=400,
        width=650
    )
    
    if show_fig:
        fig.show()
    
    return fig


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_efield_analysis(
    efield_path: str = EFIELD_CSV_DEFAULT,
    cond_df: pd.DataFrame = None,
    wsls_h2: pd.DataFrame = None,
    rw_df: pd.DataFrame = None,
    ddm_df: pd.DataFrame = None,
    show_plots: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run complete e-field analysis pipeline.
    
    Parameters
    ----------
    efield_path : str
        Path to e-field CSV
    cond_df : DataFrame, optional
        Condition-level accuracy/win_rate
    wsls_h2 : DataFrame, optional
        WSLS results
    rw_df : DataFrame, optional
        R-W parameters
    ddm_df : DataFrame, optional
        DDM parameters
    show_plots : bool
        Display visualizations
    verbose : bool
        Print results
    
    Returns
    -------
    dict with all analysis results and figures
    """
    results = {
        'efield_data': None,
        'age_relationship': None,
        'moderation': None,
        'figures': {}
    }
    
    if verbose:
        print("="*70)
        print("E-FIELD ANALYSIS")
        print("="*70)
    
    # -------------------------------------------------------------------------
    # 1. Load e-field data
    # -------------------------------------------------------------------------
    efield_df = load_efield_data(efield_path, cond_df=cond_df, verbose=verbose)
    results['efield_data'] = efield_df
    
    # -------------------------------------------------------------------------
    # 2. Age × e-field relationship
    # -------------------------------------------------------------------------
    age_results = test_age_efield_relationship(efield_df, verbose=verbose)
    results['age_relationship'] = age_results
    
    # -------------------------------------------------------------------------
    # 3. E-field moderation of stimulation effects
    # -------------------------------------------------------------------------
    mod_results = compute_efield_moderation(
        efield_df,
        cond_df=cond_df,
        wsls_h2=wsls_h2,
        rw_df=rw_df,
        ddm_df=ddm_df,
        verbose=verbose
    )
    results['moderation'] = mod_results
    
    # -------------------------------------------------------------------------
    # 4. Visualizations
    # -------------------------------------------------------------------------
    if show_plots:
        if verbose:
            print("\n" + "="*70)
            print("GENERATING VISUALIZATIONS")
            print("="*70)
        
        # Age × e-field
        results['figures']['age_vs_efield'] = plot_age_vs_efield(efield_df, show_fig=True)
        
        # E-field × Δ accuracy
        if 'accuracy' in mod_results:
            results['figures']['efield_vs_delta_accuracy'] = plot_efield_vs_delta(
                mod_results, dv='accuracy', show_fig=True
            )
        
        # E-field × Δ win_rate
        if 'win_rate' in mod_results:
            results['figures']['efield_vs_delta_winrate'] = plot_efield_vs_delta(
                mod_results, dv='win_rate', show_fig=True
            )
        
        # E-field × Δ WSLS p(stay|win)
        if 'wsls_p_stay_win' in mod_results:
            results['figures']['efield_vs_delta_p_stay_win'] = plot_efield_vs_delta(
                mod_results, dv='wsls_p_stay_win', show_fig=True
            )
        
        # E-field × Δ WSLS p(shift|lose)
        if 'wsls_p_shift_lose' in mod_results:
            results['figures']['efield_vs_delta_p_shift_lose'] = plot_efield_vs_delta(
                mod_results, dv='wsls_p_shift_lose', show_fig=True
            )
        
        # Summary forest plot
        if mod_results:
            results['figures']['moderation_summary'] = plot_efield_moderation_summary(
                mod_results, show_fig=True
            )
    
    if verbose:
        print("\n" + "="*70)
        print("E-FIELD ANALYSIS COMPLETE")
        print("="*70)
    
    return results


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing efield_analysis module...")
    print("\nMain functions:")
    print("  - load_efield_data()")
    print("  - test_age_efield_relationship()")
    print("  - plot_age_vs_efield()")
    print("  - compute_efield_moderation()")
    print("  - plot_efield_vs_delta()")
    print("  - plot_efield_moderation_summary()")
    print("  - run_efield_analysis()")
