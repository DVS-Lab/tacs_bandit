"""
survey_exploration.py — Exploratory survey measure analyses for tACS Bandit study

Systematically tests relationships between all available survey/clinical
measures and behavioral/computational DVs from the bandit task.

Steps:
  1. Correlation matrix: all survey measures × all DVs (baseline + change)
  2. FDR correction for multiple comparisons (Benjamini-Hochberg)
  3. Scatter plots for significant relationships (with age gradient)
  4. Regression models for strongest candidates, controlling for age + cognition
  5. Survey measures as moderators of tACS effects (Δ scores)

Usage in main_analyses.ipynb:
    from survey_exploration import run_survey_exploration
    survey_results = run_survey_exploration(subj_df, show_plots=True)
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from typing import Optional, Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

from config import (
    AGE_MIN,
    AGE_MAX,
    AGE_COLORSCALE,
    PLOTLY_TEMPLATE,
    FONT_FAMILY,
)


# =============================================================================
# Constants
# =============================================================================

# Survey measure groups with readable labels
SURVEY_GROUPS = {
    'Reward/Punishment Sensitivity': {
        'spsrq_sr': 'SPSRQ-SR (Reward)',
        'spsrq_sp': 'SPSRQ-SP (Punishment)',
    },
    'Mood & Anxiety': {
        'susd_depression': 'SUSD Depression',
        'susd_mania': 'SUSD Mania',
        'scaared_total': 'SCAARED Total',
        'scaared_somatic': 'SCAARED Somatic',
        'scaared_gad': 'SCAARED GAD',
        'scaared_separation': 'SCAARED Separation',
        'scaared_social': 'SCAARED Social',
        'promis_anxiety': 'PROMIS Anxiety',
        'promis_depression': 'PROMIS Depression',
    },
    'Substance Use & Gambling': {
        'audit_total': 'AUDIT (Alcohol)',
        'dudit_total': 'DUDIT (Drugs)',
        'sogs_total': 'SOGS (Gambling)',
        'norc_lifetime': 'NORC Lifetime',
        'norc_past_year': 'NORC Past Year',
    },
    'Sleep & Fatigue': {
        'bpsqi_global': 'BPSQI Global',
        'bpsqi_latency': 'BPSQI Latency',
        'bpsqi_duration': 'BPSQI Duration',
        'bpsqi_disturbance': 'BPSQI Disturbance',
        'bpsqi_quality': 'BPSQI Quality',
        'promis_fatigue': 'PROMIS Fatigue',
        'promis_sleep': 'PROMIS Sleep',
    },
    'Trauma & Adversity': {
        'ctq_total': 'CTQ Total',
        'ctq_emotional_abuse': 'CTQ Emotional Abuse',
        'ctq_physical_abuse': 'CTQ Physical Abuse',
        'ctq_sexual_abuse': 'CTQ Sexual Abuse',
        'ctq_emotional_neglect': 'CTQ Emotional Neglect',
        'ctq_physical_neglect': 'CTQ Physical Neglect',
    },
    'Social & Wellbeing': {
        'mspss_total': 'MSPSS Total',
        'mspss_friends': 'MSPSS Friends',
        'mspss_family': 'MSPSS Family',
        'mspss_significant_other': 'MSPSS Sig. Other',
        'promis_social': 'PROMIS Social',
        'promis_pain': 'PROMIS Pain',
        'promis_physical': 'PROMIS Physical',
    },
    'Cognitive Style & Reflection': {
        'crt_total': 'CRT (Cog. Reflection)',
        'ffmq_total': 'FFMQ Total (Mindfulness)',
        'ffmq_observe': 'FFMQ Observe',
        'ffmq_describe': 'FFMQ Describe',
        'ffmq_actaware': 'FFMQ Act w/ Awareness',
        'ffmq_nonjudge': 'FFMQ Non-Judge',
        'ffmq_nonreact': 'FFMQ Non-React',
        'bbs_avg': 'BBS (Boredom)',
    },
    'Technology & Media': {
        'bsmas_total': 'BSMAS (Social Media)',
        'nbs_total': 'NBS (Nomophobia)',
    },
    'Emotion Regulation': {
        'eros_ext_improving': 'EROS Ext Improving',
        'eros_ext_worsening': 'EROS Ext Worsening',
        'eros_int_improving': 'EROS Int Improving',
        'eros_int_worsening': 'EROS Int Worsening',
    },
    'Subjective Cognition': {
        'scd_q': 'SCD-Q',
        'har_score': 'HAR Score',
    },
}

# DVs to test against
BASELINE_DVS = {
    'sham_p_stay_win': 'p(stay|win)',
    'sham_p_shift_lose': 'p(shift|lose)',
    'sham_alpha': 'α (learning rate)',
    'sham_beta': 'β (inv. temperature)',
    'sham_accuracy': 'Accuracy',
    'sham_win_rate': 'Win Rate',
}

CHANGE_DVS = {
    'delta_p_stay_win': 'Δ p(stay|win)',
    'delta_p_shift_lose': 'Δ p(shift|lose)',
    'delta_alpha': 'Δ α',
    'delta_beta': 'Δ β',
    'delta_accuracy': 'Δ Accuracy',
    'delta_win_rate': 'Δ Win Rate',
}


# =============================================================================
# Age color utilities (consistent with hypothesis_tests.py)
# =============================================================================

def _get_age_colormap():
    import matplotlib.colors as mcolors
    return mcolors.LinearSegmentedColormap.from_list('blue_gold', ['#1565C0', '#FFB300'])


def _age_to_rgb(age: float) -> str:
    cmap = _get_age_colormap()
    norm = np.clip((age - AGE_MIN) / (AGE_MAX - AGE_MIN), 0, 1)
    rgba = cmap(norm)
    return f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'


# =============================================================================
# Step 1: Pairwise correlations with coverage audit
# =============================================================================

def compute_correlation_matrix(
    subj_df: pd.DataFrame,
    min_n: int = 10,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlations between all available survey
    measures and all behavioral/computational DVs.

    Returns a DataFrame with columns:
        survey_var, survey_label, survey_group, dv_var, dv_label, dv_type,
        r, p_uncorrected, n, significant_uncorrected
    """
    # Flatten survey measures, keeping only those present in subj_df
    survey_vars = []
    for group_name, group_vars in SURVEY_GROUPS.items():
        for var, label in group_vars.items():
            if var in subj_df.columns:
                n_valid = subj_df[var].notna().sum()
                if n_valid >= min_n:
                    survey_vars.append((var, label, group_name, n_valid))

    # Combine baseline and change DVs
    all_dvs = []
    for var, label in BASELINE_DVS.items():
        if var in subj_df.columns:
            all_dvs.append((var, label, 'baseline'))
    for var, label in CHANGE_DVS.items():
        if var in subj_df.columns:
            all_dvs.append((var, label, 'change'))

    if verbose:
        print('=' * 70)
        print('SURVEY × DV CORRELATION MATRIX')
        print('=' * 70)
        print(f'\nSurvey measures available (n ≥ {min_n}): {len(survey_vars)}')
        for var, label, group, n_valid in survey_vars:
            print(f'  {label:35s}  n = {n_valid:2d}  [{group}]')
        print(f'\nDVs: {len(all_dvs)}')
        for var, label, dv_type in all_dvs:
            print(f'  {label:25s}  ({dv_type})')

    rows = []
    for s_var, s_label, s_group, _ in survey_vars:
        for d_var, d_label, d_type in all_dvs:
            # Ensure numeric
            s_vals = pd.to_numeric(subj_df[s_var], errors='coerce')
            d_vals = pd.to_numeric(subj_df[d_var], errors='coerce')
            valid = s_vals.notna() & d_vals.notna()
            n = valid.sum()

            if n >= min_n:
                r, p = stats.pearsonr(s_vals[valid], d_vals[valid])
                rows.append({
                    'survey_var': s_var,
                    'survey_label': s_label,
                    'survey_group': s_group,
                    'dv_var': d_var,
                    'dv_label': d_label,
                    'dv_type': d_type,
                    'r': r,
                    'p_uncorrected': p,
                    'n': n,
                })

    corr_df = pd.DataFrame(rows)

    if len(corr_df) == 0:
        if verbose:
            print('\n⚠ No valid survey × DV pairs found.')
        return corr_df

    if verbose:
        print(f'\nTotal correlations computed: {len(corr_df)}')
        print(f'  Baseline DV pairs: {(corr_df["dv_type"] == "baseline").sum()}')
        print(f'  Change DV pairs: {(corr_df["dv_type"] == "change").sum()}')

    return corr_df


# =============================================================================
# Step 2: FDR correction
# =============================================================================

def apply_fdr_correction(
    corr_df: pd.DataFrame,
    alpha: float = 0.05,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Apply Benjamini-Hochberg FDR correction to the correlation matrix.

    Correction is applied separately for:
      - Baseline DVs (survey → sham behavior)
      - Change DVs (survey → tACS effect)

    Adds columns: p_fdr, significant_fdr
    """
    if len(corr_df) == 0:
        return corr_df

    corr_df = corr_df.copy()
    corr_df['p_fdr'] = np.nan
    corr_df['significant_fdr'] = False
    corr_df['significant_uncorrected'] = corr_df['p_uncorrected'] < alpha

    for dv_type in ['baseline', 'change']:
        mask = corr_df['dv_type'] == dv_type
        if mask.sum() == 0:
            continue

        p_vals = corr_df.loc[mask, 'p_uncorrected'].values
        reject, p_corrected, _, _ = multipletests(p_vals, alpha=alpha, method='fdr_bh')

        corr_df.loc[mask, 'p_fdr'] = p_corrected
        corr_df.loc[mask, 'significant_fdr'] = reject

    if verbose:
        n_uncorr = corr_df['significant_uncorrected'].sum()
        n_fdr = corr_df['significant_fdr'].sum()
        print(f'\n--- FDR Correction (α = {alpha}) ---')
        print(f'  Significant (uncorrected): {n_uncorr}')
        print(f'  Significant (FDR):         {n_fdr}')

        if n_uncorr > 0:
            print(f'\n  Uncorrected hits (p < {alpha}):')
            hits = corr_df[corr_df['significant_uncorrected']].sort_values('p_uncorrected')
            for _, row in hits.iterrows():
                fdr_flag = '✓ FDR' if row['significant_fdr'] else '  FDR ns'
                print(f'    {row["survey_label"]:30s} × {row["dv_label"]:20s}'
                      f'  r = {row["r"]:+.3f}, p = {row["p_uncorrected"]:.4f}'
                      f', n = {row["n"]:2d}  [{fdr_flag}]')

    return corr_df


# =============================================================================
# Step 3: Summary tables by group
# =============================================================================

def print_group_summary(
    corr_df: pd.DataFrame,
    dv_type: str = 'baseline',
    top_n: int = 5,
    verbose: bool = True
) -> None:
    """Print a summary table of strongest correlations by survey group."""
    subset = corr_df[corr_df['dv_type'] == dv_type].copy()
    if len(subset) == 0:
        return

    type_label = 'Baseline (Sham)' if dv_type == 'baseline' else 'tACS Change (Δ)'

    if verbose:
        print(f'\n{"=" * 70}')
        print(f'TOP CORRELATIONS: Survey × {type_label} DVs')
        print(f'{"=" * 70}')

        # Top N overall
        top = subset.reindex(subset['p_uncorrected'].abs().sort_values().index).head(top_n)
        print(f'\nTop {top_n} (by p-value):')
        for _, row in top.iterrows():
            sig = '*' if row['significant_uncorrected'] else ''
            fdr = ' [FDR*]' if row['significant_fdr'] else ''
            print(f'  {row["survey_label"]:30s} × {row["dv_label"]:20s}'
                  f'  r = {row["r"]:+.3f}, p = {row["p_uncorrected"]:.4f}{sig}'
                  f', n = {row["n"]:2d}{fdr}')

        # By group
        for group_name in subset['survey_group'].unique():
            g = subset[subset['survey_group'] == group_name]
            hits = g[g['significant_uncorrected']]
            if len(hits) > 0:
                print(f'\n  [{group_name}] — {len(hits)} uncorrected hit(s):')
                for _, row in hits.sort_values('p_uncorrected').iterrows():
                    fdr = ' [FDR*]' if row['significant_fdr'] else ''
                    print(f'    {row["survey_label"]:30s} × {row["dv_label"]:20s}'
                          f'  r = {row["r"]:+.3f}, p = {row["p_uncorrected"]:.4f}'
                          f', n = {row["n"]:2d}{fdr}')


# =============================================================================
# Step 4: Scatter plots for significant or top relationships
# =============================================================================

def plot_survey_scatters(
    subj_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    dv_type: str = 'baseline',
    max_plots: int = 8,
    p_threshold: float = 0.05,
    show_fig: bool = True,
    verbose: bool = True
) -> Optional[go.Figure]:
    """
    Generate scatter plots for the strongest survey × DV relationships.

    Selects up to max_plots relationships with p < p_threshold (uncorrected),
    sorted by p-value. Falls back to top max_plots if none are significant.
    """
    subset = corr_df[corr_df['dv_type'] == dv_type].copy()
    if len(subset) == 0:
        return None

    # Select which correlations to plot
    sig = subset[subset['p_uncorrected'] < p_threshold].sort_values('p_uncorrected')
    if len(sig) == 0:
        # Fall back to top N by p-value
        sig = subset.sort_values('p_uncorrected').head(max_plots)
        if verbose:
            print(f'\n  No significant correlations at p < {p_threshold}; '
                  f'plotting top {len(sig)} by p-value.')

    to_plot = sig.head(max_plots)
    n_plots = len(to_plot)
    if n_plots == 0:
        return None

    # Layout
    n_cols = min(n_plots, 3)
    n_rows = int(np.ceil(n_plots / n_cols))

    subplot_titles = []
    for _, row in to_plot.iterrows():
        sig_marker = '*' if row['significant_uncorrected'] else ''
        fdr_marker = ' [FDR]' if row['significant_fdr'] else ''
        subplot_titles.append(
            f'{row["survey_label"]} × {row["dv_label"]}'
            f'<br><sub>r={row["r"]:+.3f}, p={row["p_uncorrected"]:.3f}{sig_marker}{fdr_marker}, n={row["n"]}</sub>'
        )

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )

    for idx, (_, row) in enumerate(to_plot.iterrows()):
        r_idx = idx // n_cols + 1
        c_idx = idx % n_cols + 1

        s_vals = pd.to_numeric(subj_df[row['survey_var']], errors='coerce')
        d_vals = pd.to_numeric(subj_df[row['dv_var']], errors='coerce')
        ages = subj_df['age']
        valid = s_vals.notna() & d_vals.notna() & ages.notna()

        x = s_vals[valid].values
        y = d_vals[valid].values
        a = ages[valid].values

        # Individual points with age gradient
        for xi, yi, ai in zip(x, y, a):
            fig.add_trace(go.Scatter(
                x=[xi], y=[yi],
                mode='markers',
                marker=dict(size=8, color=_age_to_rgb(ai),
                            line=dict(width=0.5, color='white')),
                opacity=0.8,
                hovertemplate=f'Age: {ai:.0f}<br>{row["survey_label"]}: {xi:.2f}'
                              f'<br>{row["dv_label"]}: {yi:.3f}<extra></extra>',
                showlegend=False,
            ), row=r_idx, col=c_idx)

        # Regression line
        if len(x) >= 3:
            slope, intercept, _, _, _ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 50)
            y_line = intercept + slope * x_line
            line_color = '#C62828' if row['significant_uncorrected'] else '#757575'
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                line=dict(color=line_color, width=2, dash='solid' if row['significant_uncorrected'] else 'dash'),
                showlegend=False,
                hoverinfo='skip',
            ), row=r_idx, col=c_idx)

        fig.update_xaxes(title_text=row['survey_label'], row=r_idx, col=c_idx)
        fig.update_yaxes(title_text=row['dv_label'], row=r_idx, col=c_idx)

    type_label = 'Baseline (Sham)' if dv_type == 'baseline' else 'tACS Change (Δ)'
    fig.update_layout(
        height=350 * n_rows,
        width=350 * n_cols,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=60, r=80, t=80, b=60),
        font=dict(family=FONT_FAMILY, size=12),
        title=dict(text=f'Survey × {type_label} DVs: Top Correlations',
                   font=dict(size=15)),
    )

    fig.update_xaxes(showgrid=False, zeroline=False,
                     showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='black')

    # Age colorbar
    fig.add_trace(go.Scatter(
        x=[None] * 50, y=[None] * 50, mode='markers',
        marker=dict(size=0.1, color=np.linspace(AGE_MIN, AGE_MAX, 50),
                    colorscale=AGE_COLORSCALE,
                    cmin=AGE_MIN, cmax=AGE_MAX,
                    colorbar=dict(x=1.04, y=0.5, len=0.5, thickness=10,
                                  title='Age', titleside='right', tickfont=dict(size=10))),
        showlegend=False, hoverinfo='skip',
    ))

    filename = f'survey_scatters_{dv_type}'
    if show_fig:
        fig.show(config=dict(toImageButtonOptions=dict(filename=filename)))

    return fig


# =============================================================================
# Step 5: Regression follow-up for significant hits
# =============================================================================

def regression_followup(
    subj_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    covariates: List[str] = None,
    p_threshold: float = 0.05,
    verbose: bool = True
) -> Dict:
    """
    For each uncorrected-significant correlation, run a regression controlling
    for age and global cognitive function to test whether the survey measure
    contributes unique variance.

    Returns dict of {(survey_var, dv_var): regression_result}
    """
    if covariates is None:
        covariates = ['age', 'global_composite']

    sig = corr_df[corr_df['significant_uncorrected']].copy()
    if len(sig) == 0:
        if verbose:
            print('\nNo significant correlations to follow up.')
        return {}

    if verbose:
        print(f'\n{"=" * 70}')
        print(f'REGRESSION FOLLOW-UP: Controlling for {", ".join(covariates)}')
        print(f'{"=" * 70}')

    results = {}
    for _, row in sig.sort_values('p_uncorrected').iterrows():
        s_var = row['survey_var']
        d_var = row['dv_var']
        s_label = row['survey_label']
        d_label = row['dv_label']

        # Build predictor list: survey var + covariates
        predictors = [s_var] + [c for c in covariates if c != s_var]

        # Get clean data
        all_cols = [d_var] + predictors
        available = [c for c in all_cols if c in subj_df.columns]
        if len(available) < len(all_cols):
            missing = set(all_cols) - set(available)
            if verbose:
                print(f'\n  {s_label} × {d_label}: missing columns {missing}, skipping.')
            continue

        df = subj_df[all_cols].copy()
        for col in all_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()

        if len(df) < len(predictors) + 2:
            if verbose:
                print(f'\n  {s_label} × {d_label}: insufficient data (n={len(df)}), skipping.')
            continue

        y = df[d_var].values
        X = df[predictors].copy()

        # Standardize for comparable coefficients
        X_std = (X - X.mean()) / X.std()
        X_std = sm.add_constant(X_std)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = sm.OLS(y, X_std).fit()

        # Also run base model (covariates only) for ΔR²
        X_base = df[[c for c in covariates if c in df.columns]].copy()
        X_base_std = (X_base - X_base.mean()) / X_base.std()
        X_base_std = sm.add_constant(X_base_std)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            base_model = sm.OLS(y, X_base_std).fit()

        delta_r2 = model.rsquared - base_model.rsquared

        result = {
            'model': model,
            'base_model': base_model,
            'n': len(df),
            'r2': model.rsquared,
            'adj_r2': model.rsquared_adj,
            'delta_r2': delta_r2,
            'survey_beta': model.params[1],  # First predictor after constant
            'survey_p': model.pvalues[1],
            'bivariate_r': row['r'],
            'bivariate_p': row['p_uncorrected'],
        }
        results[(s_var, d_var)] = result

        if verbose:
            sig_marker = '*' if result['survey_p'] < 0.05 else ''
            print(f'\n  {d_label} ~ {s_label} + {" + ".join(covariates)}')
            print(f'    N = {result["n"]}')
            print(f'    Full model: R² = {result["r2"]:.3f}, Adj R² = {result["adj_r2"]:.3f}')
            print(f'    Base model: R² = {base_model.rsquared:.3f}')
            print(f'    ΔR² ({s_label}): {delta_r2:+.3f}')
            print(f'    {s_label}: β = {result["survey_beta"]:+.3f}, '
                  f'p = {result["survey_p"]:.4f}{sig_marker}')
            print(f'    (bivariate: r = {row["r"]:+.3f}, p = {row["p_uncorrected"]:.4f})')

            # Print all coefficients
            for pred, beta, p in zip(
                ['const'] + predictors,
                model.params,
                model.pvalues
            ):
                if pred == 'const':
                    continue
                s = '*' if p < 0.05 else ''
                print(f'    {pred:25s}: β = {beta:+.3f}, p = {p:.4f}{s}')

    return results


# =============================================================================
# Step 6: Heatmap visualization
# =============================================================================

def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    dv_type: str = 'baseline',
    min_n: int = 15,
    show_fig: bool = True,
    verbose: bool = True
) -> Optional[go.Figure]:
    """
    Generate a heatmap of survey × DV correlations.
    Cells with n < min_n are grayed out. Significant correlations are annotated.
    """
    subset = corr_df[(corr_df['dv_type'] == dv_type)].copy()
    if len(subset) == 0:
        return None

    # Pivot to matrix form
    r_matrix = subset.pivot(index='survey_label', columns='dv_label', values='r')
    p_matrix = subset.pivot(index='survey_label', columns='dv_label', values='p_uncorrected')
    n_matrix = subset.pivot(index='survey_label', columns='dv_label', values='n')
    fdr_matrix = subset.pivot(index='survey_label', columns='dv_label', values='significant_fdr')

    # Build annotation text
    annot = r_matrix.copy().astype(str)
    for i in range(len(r_matrix)):
        for j in range(len(r_matrix.columns)):
            r_val = r_matrix.iloc[i, j]
            p_val = p_matrix.iloc[i, j]
            n_val = n_matrix.iloc[i, j]
            fdr_val = fdr_matrix.iloc[i, j]

            if pd.isna(r_val) or n_val < min_n:
                annot.iloc[i, j] = ''
            elif fdr_val:
                annot.iloc[i, j] = f'{r_val:+.2f}**'
            elif p_val < 0.05:
                annot.iloc[i, j] = f'{r_val:+.2f}*'
            else:
                annot.iloc[i, j] = f'{r_val:+.2f}'

    # Mask low-n cells
    r_display = r_matrix.copy()
    r_display[n_matrix < min_n] = np.nan

    type_label = 'Baseline (Sham)' if dv_type == 'baseline' else 'tACS Change (Δ)'

    fig = go.Figure(data=go.Heatmap(
        z=r_display.values,
        x=r_display.columns.tolist(),
        y=r_display.index.tolist(),
        text=annot.values,
        texttemplate='%{text}',
        textfont=dict(size=10),
        colorscale='RdBu_r',
        zmid=0,
        zmin=-0.6,
        zmax=0.6,
        colorbar=dict(title='r', thickness=12, len=0.7),
        hoverongaps=False,
        hovertemplate='%{y} × %{x}<br>r = %{z:.3f}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(text=f'Survey × {type_label}: Correlation Heatmap<br>'
                        f'<sub>* p < .05 uncorrected, ** p < .05 FDR</sub>',
                   font=dict(size=14)),
        height=max(400, 22 * len(r_display)),
        width=max(600, 100 * len(r_display.columns) + 200),
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY, size=11),
        margin=dict(l=200, r=80, t=80, b=80),
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed'),
    )

    filename = f'survey_heatmap_{dv_type}'
    if show_fig:
        fig.show(config=dict(toImageButtonOptions=dict(filename=filename)))

    return fig


# =============================================================================
# Step 7: tACS moderation analysis
# =============================================================================

def test_survey_moderation(
    subj_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Test whether survey measures moderate the tACS effect.

    For each survey × Δ DV correlation that is significant (uncorrected),
    run a regression: Δ DV ~ survey_measure + age + global_composite
    to test whether the moderation holds after controlling for covariates.
    """
    change_sig = corr_df[
        (corr_df['dv_type'] == 'change') &
        (corr_df['significant_uncorrected'])
    ].copy()

    if verbose:
        print(f'\n{"=" * 70}')
        print(f'SURVEY MODERATION OF tACS EFFECTS')
        print(f'{"=" * 70}')

        if len(change_sig) == 0:
            print('\n  No survey measures significantly correlate with change scores.')
            print('  Reporting top 5 by p-value for reference:\n')

            top5 = corr_df[corr_df['dv_type'] == 'change'].sort_values('p_uncorrected').head(5)
            for _, row in top5.iterrows():
                print(f'    {row["survey_label"]:30s} × {row["dv_label"]:20s}'
                      f'  r = {row["r"]:+.3f}, p = {row["p_uncorrected"]:.4f}'
                      f', n = {row["n"]:2d}')
        else:
            print(f'\n  {len(change_sig)} significant moderators found.')

    return change_sig


# =============================================================================
# Main entry point
# =============================================================================

def run_survey_exploration(
    subj_df: pd.DataFrame,
    show_plots: bool = True,
    verbose: bool = True,
    min_n: int = 10,
    fdr_alpha: float = 0.05,
    max_scatter_plots: int = 8
) -> Dict:
    """
    Run the full survey exploration pipeline.

    Parameters
    ----------
    subj_df : pd.DataFrame
        Subject-level DataFrame with survey measures, behavioral DVs,
        and computational parameters.
    show_plots : bool
        Whether to display Plotly figures inline.
    verbose : bool
        Whether to print detailed output.
    min_n : int
        Minimum sample size for a correlation to be computed.
    fdr_alpha : float
        Alpha level for FDR correction.
    max_scatter_plots : int
        Maximum number of scatter plots per DV type.

    Returns
    -------
    dict with keys:
        'corr_df': Full correlation DataFrame with FDR
        'regression_results': Follow-up regressions for significant hits
        'figures': Dict of Plotly figures
    """
    results = {'figures': {}}

    # Step 1: Compute all correlations
    corr_df = compute_correlation_matrix(subj_df, min_n=min_n, verbose=verbose)
    if len(corr_df) == 0:
        return results

    # Step 2: FDR correction
    corr_df = apply_fdr_correction(corr_df, alpha=fdr_alpha, verbose=verbose)
    results['corr_df'] = corr_df

    # Step 3: Group summaries
    print_group_summary(corr_df, dv_type='baseline', verbose=verbose)
    print_group_summary(corr_df, dv_type='change', verbose=verbose)

    # Step 4: Heatmaps
    for dv_type in ['baseline', 'change']:
        fig = plot_correlation_heatmap(corr_df, dv_type=dv_type,
                                        show_fig=show_plots, verbose=verbose)
        if fig is not None:
            results['figures'][f'heatmap_{dv_type}'] = fig

    # Step 5: Scatter plots
    for dv_type in ['baseline', 'change']:
        fig = plot_survey_scatters(subj_df, corr_df, dv_type=dv_type,
                                   max_plots=max_scatter_plots,
                                   show_fig=show_plots, verbose=verbose)
        if fig is not None:
            results['figures'][f'scatters_{dv_type}'] = fig

    # Step 6: Regression follow-up
    reg_results = regression_followup(subj_df, corr_df, verbose=verbose)
    results['regression_results'] = reg_results

    # Step 7: tACS moderation
    mod_df = test_survey_moderation(subj_df, corr_df, verbose=verbose)
    results['moderation_hits'] = mod_df

    # Summary
    if verbose:
        n_total = len(corr_df)
        n_uncorr = corr_df['significant_uncorrected'].sum()
        n_fdr = corr_df['significant_fdr'].sum()
        n_reg = sum(1 for v in reg_results.values() if v['survey_p'] < 0.05)

        print(f'\n{"=" * 70}')
        print(f'SURVEY EXPLORATION SUMMARY')
        print(f'{"=" * 70}')
        print(f'  Total correlations tested: {n_total}')
        print(f'  Uncorrected significant (p < .05): {n_uncorr}')
        print(f'  FDR significant: {n_fdr}')
        print(f'  Survived regression follow-up: {n_reg}')
        print(f'  Heatmaps generated: {sum(1 for k in results["figures"] if "heatmap" in k)}')
        print(f'  Scatter plots generated: {sum(1 for k in results["figures"] if "scatter" in k)}')

    return results
