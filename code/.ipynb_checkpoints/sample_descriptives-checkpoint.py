"""
sample_descriptives.py — Sample characterization for tACS Bandit study

Functions for:
- Loading and summarizing demographic data from REDCap
- Age distribution visualization
- Cap size (estimated vs actual) analysis
- Sample summary tables
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional, Dict, Tuple
from pathlib import Path

from config import (
    SUBJECT_INFO,
    REDCAP_TACS_PATH,
    AGE_MIN,
    AGE_MAX,
    AGE_CMAP_COLORS,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_PURPLE,
    PLOTLY_TEMPLATE,
    FONT_FAMILY,
)


# =============================================================================
# Age Color Utilities
# =============================================================================

def get_age_colormap():
    """Get matplotlib colormap for age gradient."""
    import matplotlib.colors as mcolors
    return mcolors.LinearSegmentedColormap.from_list('blue_gold', AGE_CMAP_COLORS)


def age_to_rgb(age: float, age_min: float = AGE_MIN, age_max: float = AGE_MAX) -> str:
    """Convert age to RGB color string for Plotly."""
    cmap = get_age_colormap()
    norm = np.clip((age - age_min) / (age_max - age_min), 0, 1)
    rgba = cmap(norm)
    return f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'


# =============================================================================
# Demographics Loading
# =============================================================================

def load_demographics(
    redcap_path: Optional[Path] = None,
    subject_info: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Load demographic data from tACS Bandit REDCap export.
    
    Parameters
    ----------
    redcap_path : Path, optional
        Path to REDCap export; defaults to REDCAP_TACS_PATH
    subject_info : dict, optional
        Subject registry for filtering; defaults to SUBJECT_INFO
    
    Returns
    -------
    DataFrame
        Demographics for study participants with columns:
        subject_id, age, Gender, Race, Ethnicity
    """
    if redcap_path is None:
        redcap_path = REDCAP_TACS_PATH
    if subject_info is None:
        subject_info = SUBJECT_INFO
    
    redcap = pd.read_csv(redcap_path)
    redcap = redcap.rename(columns={'Record ID': 'subject_id', 'Age in years:': 'age'})
    redcap['subject_id'] = redcap['subject_id'].astype(str)
    
    # Filter to study participants
    demo = redcap[redcap['subject_id'].isin(subject_info.keys())].copy()
    
    return demo


def print_demographics_summary(demo: pd.DataFrame):
    """Print formatted demographics summary table."""
    n = len(demo)
    
    if n == 0:
        print("No demographic data available.")
        return
    
    print(f'N = {n}')
    
    # Age
    if 'age' in demo.columns and demo['age'].notna().any():
        age_mean = demo['age'].mean()
        age_sd = demo['age'].std()
        age_min = demo['age'].min()
        age_max = demo['age'].max()
        print(f'Age: M = {age_mean:.1f}, SD = {age_sd:.1f}, range = {age_min:.0f}–{age_max:.0f}')
    
    # Gender
    if 'Gender' in demo.columns:
        print(f'\nGender:')
        for val, count in demo['Gender'].value_counts().items():
            print(f'  {val}: {count} ({100*count/n:.0f}%)')
    
    # Race
    if 'Race' in demo.columns:
        print(f'\nRace:')
        for val, count in demo['Race'].value_counts().items():
            print(f'  {val}: {count} ({100*count/n:.0f}%)')
    
    # Ethnicity
    if 'Ethnicity' in demo.columns:
        print(f'\nEthnicity:')
        for val, count in demo['Ethnicity'].value_counts().items():
            print(f'  {val}: {count} ({100*count/n:.0f}%)')


# =============================================================================
# Age Distribution Visualization
# =============================================================================

def plot_age_distribution(
    demo: pd.DataFrame,
    show_fig: bool = True
) -> go.Figure:
    """
    Create age distribution strip plot with blue→gold gradient.
    
    Parameters
    ----------
    demo : DataFrame
        Demographics data with 'age', 'subject_id', 'Gender' columns
    show_fig : bool
        If True, display the figure
    
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    n = len(demo)
    
    if n == 0 or 'age' not in demo.columns:
        print("No age data available for plotting.")
        return None
    
    age_mean = demo['age'].mean()
    age_sd = demo['age'].std()
    
    # Add jitter for strip plot
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.15, 0.15, size=n)
    
    fig = go.Figure()
    
    # Scatter points colored by age
    fig.add_trace(go.Scatter(
        x=demo['age'],
        y=jitter,
        mode='markers',
        marker=dict(
            size=12,
            opacity=0.8,
            color=[age_to_rgb(a) for a in demo['age']],
            line=dict(width=0.5, color='white')
        ),
        text=[
            f"sub-{s}<br>Age: {a:.1f}<br>{g}" 
            for s, a, g in zip(
                demo['subject_id'], 
                demo['age'], 
                demo.get('Gender', [''] * n)
            )
        ],
        hoverinfo='text',
        showlegend=False,
    ))
    
    # Mean + SD reference lines
    fig.add_vline(
        x=age_mean, 
        line=dict(color='#404040', width=1.5, dash='solid'),
        annotation_text=f'M = {age_mean:.1f}', 
        annotation_position='top'
    )
    fig.add_vline(x=age_mean - age_sd, line=dict(color='#999999', width=1, dash='dash'))
    fig.add_vline(x=age_mean + age_sd, line=dict(color='#999999', width=1, dash='dash'))
    
    fig.update_layout(
        height=200,
        width=700,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=80, t=30, b=40),
        font=dict(family=FONT_FAMILY, size=14),
        yaxis=dict(visible=False),
        xaxis=dict(title='Age (years)', range=[18, 82]),
    )
    
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor='black',
        tickfont=dict(size=13)
    )
    
    if show_fig:
        fig.show(config=dict(toImageButtonOptions=dict(filename='sample_age_distribution')))
    
    return fig


# =============================================================================
# Cap Size Analysis
# =============================================================================

def load_cap_size_data(
    redcap_path: Optional[Path] = None,
    subject_info: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Load cap size data (estimated from ICV vs actual from session).
    
    Parameters
    ----------
    redcap_path : Path, optional
        Path to REDCap export
    subject_info : dict, optional
        Subject registry for filtering
    
    Returns
    -------
    DataFrame
        Cap size data with columns: subject_id, icv, cap_estimated, cap_actual,
        est_numeric, act_numeric, diff, abs_diff, match
    """
    if redcap_path is None:
        redcap_path = REDCAP_TACS_PATH
    if subject_info is None:
        subject_info = SUBJECT_INFO
    
    valid_subjects = set(int(s) for s in subject_info.keys() if s.isdigit())
    
    redcap = pd.read_csv(redcap_path)
    
    # Extract relevant columns
    cap_df = redcap[['Record ID', 'Intracranial Volume', 'Cap size', 'Cap size.1']].copy()
    cap_df.columns = ['subject_id', 'icv', 'cap_estimated', 'cap_actual']
    
    # Filter to numeric subject IDs and valid ICV
    cap_df = cap_df[cap_df['subject_id'].astype(str).str.match(r'^\d+$')]
    cap_df['subject_id'] = cap_df['subject_id'].astype(int)
    cap_df = cap_df[cap_df['icv'].notna()].copy()
    
    # Filter to study participants
    cap_df = cap_df[cap_df['subject_id'].isin(valid_subjects)].copy()
    
    # Map to numeric for comparison
    cap_map = {'Small': 1, 'Medium': 2, 'Large': 3, 'Extra Large': 4}
    cap_df['est_numeric'] = cap_df['cap_estimated'].map(cap_map)
    cap_df['act_numeric'] = cap_df['cap_actual'].map(cap_map)
    
    # Calculate mismatch
    cap_df['diff'] = cap_df['est_numeric'] - cap_df['act_numeric']
    cap_df['abs_diff'] = cap_df['diff'].abs()
    cap_df['match'] = cap_df['abs_diff'] == 0
    
    return cap_df


def print_cap_size_summary(cap_df: pd.DataFrame):
    """Print cap size comparison summary."""
    # Filter to those with both estimates
    cap_both = cap_df[
        cap_df['est_numeric'].notna() & 
        cap_df['act_numeric'].notna()
    ].copy()
    
    print('='*70)
    print('Estimated vs Actual Cap Size Comparison')
    print('='*70)
    print(f"\nSubjects with ICV data: {len(cap_df)}")
    print(f"Subjects with both estimated and actual: {len(cap_both)}")
    
    if len(cap_both) > 0:
        print(f"  Matches (same size): {cap_both['match'].sum()}")
        print(f"  Off by 1 size: {(cap_both['abs_diff'] == 1).sum()}")
        print(f"  Off by 2 sizes: {(cap_both['abs_diff'] == 2).sum()}")
        
        # Direction of mismatches
        mismatches = cap_both[~cap_both['match']]
        if len(mismatches) > 0:
            print(f"\nMismatch direction:")
            print(f"  Actual smaller than estimated: {(mismatches['diff'] > 0).sum()}")
            print(f"  Actual larger than estimated: {(mismatches['diff'] < 0).sum()}")
    
    # Full table
    print(f"\n{'Subject':>8} {'ICV':>8} {'Estimated':>14} {'Actual':>12} {'Status':>12}")
    print('-'*60)
    
    for _, row in cap_df.sort_values('subject_id').iterrows():
        est = row['cap_estimated'] if pd.notna(row['cap_estimated']) else '—'
        act = row['cap_actual'] if pd.notna(row['cap_actual']) else '—'
        
        if pd.isna(row['est_numeric']) or pd.isna(row['act_numeric']):
            status = 'Incomplete'
        elif row['est_numeric'] == row['act_numeric']:
            status = '✓ Match'
        else:
            diff = int(row['est_numeric'] - row['act_numeric'])
            status = f'✗ {diff:+d} size'
        
        print(f"{int(row['subject_id']):>8} {row['icv']:>8.1f} {est:>14} {act:>12} {status:>12}")


def plot_cap_size_comparison(
    cap_df: pd.DataFrame,
    show_fig: bool = True
) -> go.Figure:
    """
    Create scatter plot of estimated vs actual cap size.
    
    Parameters
    ----------
    cap_df : DataFrame
        Cap size data from load_cap_size_data()
    show_fig : bool
        If True, display the figure
    
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Filter to those with both estimates
    cap_both = cap_df[
        cap_df['est_numeric'].notna() & 
        cap_df['act_numeric'].notna()
    ].copy()
    
    if len(cap_both) == 0:
        print("No complete cap size data for plotting.")
        return None
    
    fig = go.Figure()
    
    # Add jitter for visibility
    np.random.seed(42)
    jitter_est = cap_both['est_numeric'] + np.random.uniform(-0.12, 0.12, len(cap_both))
    jitter_act = cap_both['act_numeric'] + np.random.uniform(-0.12, 0.12, len(cap_both))
    
    # Color by match status
    colors = []
    symbols = []
    for _, row in cap_both.iterrows():
        if row['abs_diff'] == 0:
            colors.append(COLOR_GREEN)  # Match
            symbols.append('circle')
        elif row['abs_diff'] == 1:
            colors.append(COLOR_RED)  # Off by 1
            symbols.append('circle')
        else:
            colors.append(COLOR_PURPLE)  # Off by 2+
            symbols.append('diamond')
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=jitter_est,
        y=jitter_act,
        mode='markers',
        marker=dict(size=14, color=colors, symbol=symbols,
                    line=dict(width=1.5, color='white')),
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            'ICV: %{customdata[1]:.0f} mm³<br>'
            'Estimated: %{customdata[2]}<br>'
            'Actual: %{customdata[3]}<br>'
            'Diff: %{customdata[4]}<extra></extra>'
        ),
        customdata=np.column_stack([
            cap_both['subject_id'].values,
            cap_both['icv'].values,
            cap_both['cap_estimated'].values,
            cap_both['cap_actual'].values,
            ['Match' if d == 0 else f'{int(d):+d} size' for d in cap_both['diff'].values]
        ]),
        showlegend=False,
    ))
    
    # Diagonal reference line (perfect agreement)
    fig.add_trace(go.Scatter(
        x=[0.5, 4.5], y=[0.5, 4.5],
        mode='lines',
        line=dict(color='gray', width=1.5, dash='dash'),
        showlegend=False,
        hoverinfo='skip',
    ))
    
    # Legend markers
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers', name='Match',
        marker=dict(size=12, color=COLOR_GREEN, symbol='circle'),
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers', name='Off by 1',
        marker=dict(size=12, color=COLOR_RED, symbol='circle'),
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers', name='Off by 2+',
        marker=dict(size=12, color=COLOR_PURPLE, symbol='diamond'),
    ))
    
    # Summary annotation
    n_match = cap_both['match'].sum()
    n_total = len(cap_both)
    pct_match = 100 * n_match / n_total
    
    fig.add_annotation(
        x=0.02, y=0.98, xref='paper', yref='paper',
        text=f"<b>N = {n_total}</b><br>Match: {n_match} ({pct_match:.0f}%)<br>Mismatch: {n_total - n_match}",
        showarrow=False,
        font=dict(size=11),
        align='left',
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='gray', borderwidth=1,
        xanchor='left', yanchor='top',
    )
    
    fig.update_layout(
        height=500, width=550,
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY, size=11),
        title=dict(text='Estimated vs Actual Cap Size', font=dict(size=14)),
        xaxis=dict(
            title='Estimated Cap Size (from ICV)',
            tickvals=[1, 2, 3, 4],
            ticktext=['Small', 'Medium', 'Large', 'XL'],
            range=[0.5, 4.5],
            showgrid=False, linecolor='black', linewidth=1,
        ),
        yaxis=dict(
            title='Actual Cap Size (Session Notes)',
            tickvals=[1, 2, 3, 4],
            ticktext=['Small', 'Medium', 'Large', 'XL'],
            range=[0.5, 4.5],
            showgrid=False, linecolor='black', linewidth=1,
        ),
        legend=dict(x=0.98, y=0.02, xanchor='right', yanchor='bottom',
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='gray', borderwidth=1),
    )
    
    if show_fig:
        fig.show(config=dict(toImageButtonOptions=dict(filename='cap_size_estimated_vs_actual')))
    
    return fig


# =============================================================================
# Combined Summary
# =============================================================================

def run_sample_descriptives(
    redcap_path: Optional[Path] = None,
    show_plots: bool = True
) -> Dict:
    """
    Run full sample descriptives analysis.
    
    Parameters
    ----------
    redcap_path : Path, optional
        Path to REDCap export
    show_plots : bool
        If True, display plots
    
    Returns
    -------
    dict with keys:
        'demographics': DataFrame of demographic data
        'cap_size': DataFrame of cap size data
        'age_fig': age distribution figure
        'cap_fig': cap size comparison figure
    """
    print('='*70)
    print('Sample Descriptives')
    print('='*70)
    print()
    
    # Demographics
    demo = load_demographics(redcap_path)
    print_demographics_summary(demo)
    print()
    
    # Age plot
    age_fig = plot_age_distribution(demo, show_fig=show_plots)
    
    # Cap size
    cap_df = load_cap_size_data(redcap_path)
    print()
    print_cap_size_summary(cap_df)
    
    # Cap size plot
    cap_fig = plot_cap_size_comparison(cap_df, show_fig=show_plots)
    
    return {
        'demographics': demo,
        'cap_size': cap_df,
        'age_fig': age_fig,
        'cap_fig': cap_fig,
    }


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing sample_descriptives module...")
    print(f"REDCap path: {REDCAP_TACS_PATH}")
    print(f"Subjects in registry: {len(SUBJECT_INFO)}")
