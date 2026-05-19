"""
presentation_plots.py — Publication-ready figures for tACS Bandit dissertation

Generates figures for slides/manuscript using matplotlib with consistent styling.
All figures are saved as SVG and PNG for flexibility.

Usage:
    from presentation_plots import generate_all_figures
    generate_all_figures(subj_df, trial_df, output_dir='figures/')

Or generate individually:
    from presentation_plots import plot_jn, plot_age_delta_scatter, plot_learning_curves
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm, t as t_dist
from pathlib import Path
from typing import Optional, Tuple

from config import AGE_MIN, AGE_MAX


# =============================================================================
# Global style settings
# =============================================================================

def _set_style():
    """Set publication-ready matplotlib defaults."""
    sns.set_style('white')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
    })


def _get_age_colormap():
    return mcolors.LinearSegmentedColormap.from_list('blue_gold', ['#1565C0', '#FFB300'])


def _age_to_color(age, ages=None):
    cmap = _get_age_colormap()
    vmin = ages.min() if ages is not None else AGE_MIN
    vmax = ages.max() if ages is not None else AGE_MAX
    norm_val = np.clip((age - vmin) / (vmax - vmin), 0, 1)
    return cmap(norm_val)


def _save_fig(fig, filepath, formats=('svg', 'png')):
    """Save figure in multiple formats."""
    path = Path(filepath)
    for fmt in formats:
        fig.savefig(path.with_suffix(f'.{fmt}'), format=fmt, bbox_inches='tight')
    print(f'  Saved: {path.with_suffix(".svg")} (.svg, .png)')


# =============================================================================
# Plot 1/1b/1c: Johnson-Neyman plots
# =============================================================================

def plot_jn(
    dv: str,
    predictor: str,
    moderator: str,
    data: pd.DataFrame,
    covariates: list = None,
    dv_label: str = None,
    predictor_label: str = None,
    moderator_label: str = None,
    interaction_p: float = None,
    delta_r2: float = None,
    figsize: Tuple = (7, 5),
    output_path: str = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Generate a publication-ready Johnson-Neyman plot.

    Style matches the istart-mid-clean reference:
      - Black slope line (linewidth=3)
      - Gray CI band (alpha=0.3)
      - Green significance shading
      - Red dashed zero line
      - Rug plot of moderator values
    """
    _set_style()

    if covariates is None:
        covariates = []
    covariates = [c for c in covariates if c != predictor and c != moderator and c != dv]

    all_cols = list(dict.fromkeys([dv, predictor, moderator] + covariates))
    df = data[all_cols].copy()
    for col in all_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()

    if len(df) < 6:
        print(f'  Insufficient data for J-N plot (n={len(df)})')
        return None

    y = df[dv].values
    dv_lab = dv_label or dv
    pred_lab = predictor_label or predictor
    mod_lab = moderator_label or moderator

    # Mean-center predictor and covariates (NOT moderator)
    pred_c = df[predictor].values - df[predictor].mean()
    mod_vals = df[moderator].values
    mod_mean = mod_vals.mean()

    # Build design matrix
    X_dict = {'const': np.ones(len(df)), predictor: pred_c, moderator: mod_vals - mod_mean}
    for cov in covariates:
        X_dict[cov] = df[cov].values - df[cov].mean()
    int_name = f'{predictor}x{moderator}'
    X_dict[int_name] = pred_c * (mod_vals - mod_mean)

    X = np.column_stack(list(X_dict.values()))
    col_names = list(X_dict.keys())

    model = sm.OLS(y, X).fit()

    b_pred = model.params[col_names.index(predictor)]
    b_int = model.params[col_names.index(int_name)]
    pred_idx = col_names.index(predictor)
    int_idx = col_names.index(int_name)

    # Simple slopes across moderator range
    mod_range = np.linspace(df[moderator].min() - 2, df[moderator].max() + 2, 300)
    mod_centered = mod_range - mod_mean
    simple_slopes = b_pred + b_int * mod_centered

    # Standard errors
    vcov = model.cov_params()
    var_b1 = vcov[pred_idx, pred_idx]
    var_b3 = vcov[int_idx, int_idx]
    cov_b1b3 = vcov[pred_idx, int_idx]

    se_slopes = np.sqrt(var_b1 + mod_centered**2 * var_b3 + 2 * mod_centered * cov_b1b3)

    # CI and p-values
    df_resid = model.df_resid
    t_crit = t_dist.ppf(0.975, df_resid)
    lower = simple_slopes - t_crit * se_slopes
    upper = simple_slopes + t_crit * se_slopes

    t_vals = simple_slopes / se_slopes
    pvals = 2 * (1 - t_dist.cdf(np.abs(t_vals), df_resid))

    # J-N points (analytic)
    a_coef = b_int**2 - t_crit**2 * var_b3
    b_coef = 2 * b_pred * b_int - 2 * t_crit**2 * cov_b1b3
    c_coef = b_pred**2 - t_crit**2 * var_b1
    discriminant = b_coef**2 - 4 * a_coef * c_coef

    jn_points = []
    if discriminant >= 0 and abs(a_coef) > 1e-10:
        w1 = (-b_coef + np.sqrt(discriminant)) / (2 * a_coef) + mod_mean
        w2 = (-b_coef - np.sqrt(discriminant)) / (2 * a_coef) + mod_mean
        mod_min, mod_max = df[moderator].min(), df[moderator].max()
        span = mod_max - mod_min
        for pt in [w1, w2]:
            if mod_min - 0.15 * span <= pt <= mod_max + 0.15 * span:
                jn_points.append(pt)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # Gray CI band
    ax.fill_between(mod_range, lower, upper, color='gray', alpha=0.3, label='95% CI')

    # Green significance shading
    sig_mask = pvals < 0.05
    ax.fill_between(mod_range, lower, upper, where=sig_mask,
                    color='green', alpha=0.2, label='Significant (p < .05)')

    # Black slope line
    ax.plot(mod_range, simple_slopes, color='black', linewidth=3,
            label=f'Simple slope of {pred_lab}')

    # Red dashed zero line
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)

    # J-N point annotations
    for jn in jn_points:
        ax.axvline(jn, color='#C62828', linestyle=':', linewidth=1.5, alpha=0.7)
        y_pos = ax.get_ylim()[1] * 0.85
        ax.annotate(f'J-N = {jn:.1f}', xy=(jn, y_pos),
                    fontsize=11, color='#C62828', ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#C62828', alpha=0.8))

    # Stats annotation
    stats_parts = []
    if interaction_p is not None:
        stats_parts.append(f'p = {interaction_p:.3f}')
    if delta_r2 is not None:
        stats_parts.append(f'ΔR² = {delta_r2:.3f}')
    stats_parts.append(f'N = {len(df)}')
    stats_text = ', '.join(stats_parts)
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=11, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.8))

    ax.set_xlabel(mod_lab, fontsize=14)
    ax.set_ylabel(f'Simple Slope of\n{pred_lab} → {dv_lab}', fontsize=14)
    ax.set_title('Johnson-Neyman Plot', fontsize=16)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)

    sns.despine(ax=ax)
    fig.tight_layout()

    if output_path:
        _save_fig(fig, output_path)
    if show:
        plt.show()

    return fig


# =============================================================================
# Plots 2–3: Age × Δ scatterplots
# =============================================================================

def plot_age_delta_scatter(
    data: pd.DataFrame,
    delta_var: str,
    delta_label: str,
    r_val: float = None,
    p_val: float = None,
    figsize: Tuple = (5.5, 4.5),
    output_path: str = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Publication-ready scatterplot: Age × Δ (active − sham).
    Points colored by age gradient, regression line with CI.
    """
    _set_style()

    df = data[['age', delta_var]].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()

    if len(df) < 5:
        return None

    x, y = df['age'].values, df[delta_var].values
    ages = df['age'].values

    # Compute r and p if not provided
    if r_val is None or p_val is None:
        r_val, p_val = stats.pearsonr(x, y)

    fig, ax = plt.subplots(figsize=figsize)

    # Regression line with CI via seaborn
    sns.regplot(x=x, y=y, ax=ax,
                scatter=False,
                line_kws=dict(color='#333333', linewidth=2),
                ci=95,
                color='gray')

    # Individual points with age gradient
    cmap = _get_age_colormap()
    colors = [_age_to_color(a, ages) for a in ages]
    ax.scatter(x, y, c=colors, s=55, edgecolors='white', linewidths=0.6, zorder=5)

    # Zero reference line
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.6)

    # Stats annotation
    sig = '*' if p_val < 0.05 else '†' if p_val < 0.10 else ''
    ax.text(0.98, 0.98, f'r = {r_val:+.3f}, p = {p_val:.3f}{sig}\nN = {len(df)}',
            transform=ax.transAxes, fontsize=11, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.8))

    # Colorbar
    sm_cbar = plt.cm.ScalarMappable(cmap=cmap,
                                      norm=plt.Normalize(vmin=ages.min(), vmax=ages.max()))
    sm_cbar.set_array([])
    cbar = fig.colorbar(sm_cbar, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Age', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel('Age', fontsize=14)
    ax.set_ylabel(delta_label, fontsize=14)

    sns.despine(ax=ax)
    fig.tight_layout()

    if output_path:
        _save_fig(fig, output_path)
    if show:
        plt.show()

    return fig


def plot_age_delta_panel(
    data: pd.DataFrame,
    figsize: Tuple = (11, 4.5),
    output_path: str = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Side-by-side panel: Age × Δ Accuracy and Age × Δ Win Rate.
    """
    _set_style()

    pairs = [
        ('delta_accuracy', 'Δ Accuracy (Active − Sham)', .339, .043),
        ('delta_win_rate', 'Δ Win Rate (Active − Sham)', .381, .022),
    ]

    # Check which variables exist
    available = [(v, l, r, p) for v, l, r, p in pairs if v in data.columns]
    if not available:
        print('  delta_accuracy / delta_win_rate not found in data.')
        return None

    n_panels = len(available)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    for ax, (delta_var, delta_label, r_expected, p_expected) in zip(axes, available):
        df = data[['age', delta_var]].copy()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()

        x, y = df['age'].values, df[delta_var].values
        r_val, p_val = stats.pearsonr(x, y)
        ages = df['age'].values

        # Regression line with CI
        sns.regplot(x=x, y=y, ax=ax,
                    scatter=False,
                    line_kws=dict(color='#333333', linewidth=2),
                    ci=95, color='gray')

        # Points with age gradient
        cmap = _get_age_colormap()
        colors = [_age_to_color(a, ages) for a in ages]
        ax.scatter(x, y, c=colors, s=55, edgecolors='white', linewidths=0.6, zorder=5)

        ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.6)

        sig = '*' if p_val < 0.05 else '†' if p_val < 0.10 else ''
        ax.text(0.98, 0.98, f'r = {r_val:+.3f}, p = {p_val:.3f}{sig}\nN = {len(df)}',
                transform=ax.transAxes, fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='gray', alpha=0.8))

        ax.set_xlabel('Age', fontsize=14)
        ax.set_ylabel(delta_label, fontsize=14)
        sns.despine(ax=ax)

    # Single colorbar for the panel
    sm_cbar = plt.cm.ScalarMappable(cmap=_get_age_colormap(),
                                      norm=plt.Normalize(vmin=AGE_MIN, vmax=AGE_MAX))
    sm_cbar.set_array([])
    cbar = fig.colorbar(sm_cbar, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label('Age', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()

    if output_path:
        _save_fig(fig, output_path)
    if show:
        plt.show()

    return fig


# =============================================================================
# Plots 4–5: Learning curves
# =============================================================================

def plot_within_run_learning_curves(
    trial_df: pd.DataFrame,
    n_bins: int = 10,
    figsize: Tuple = (7, 5),
    output_path: str = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot 4: Within-run learning curves, Sham vs Active.
    X-axis: trial bin within run, Y-axis: mean accuracy.
    Lines with SE ribbons.
    """
    _set_style()

    df = trial_df.copy()
    df['correct_num'] = pd.to_numeric(df['correct'].map({'True': 1, 'False': 0, True: 1, False: 0, 1: 1, 0: 0}),
                                       errors='coerce')
    if 'correct_num' not in df.columns or df['correct_num'].isna().all():
        # Try direct numeric
        df['correct_num'] = pd.to_numeric(df['correct'], errors='coerce')

    df['stim_condition'] = df['stim_condition'].str.strip().str.lower()
    df = df[df['stim_condition'].isin(['sham', 'active'])].copy()

    # Create trial bins within each run
    df['trial_bin'] = pd.cut(df['trial_in_contingency'] if 'trial_in_contingency' in df.columns
                             else df['trial_num'],
                             bins=n_bins, labels=False) + 1

    # If trial_in_contingency doesn't exist, use trial_num
    if 'trial_in_contingency' not in df.columns:
        df['trial_bin'] = df.groupby(['subject_id', 'run', 'stim_condition'])['trial_num'].transform(
            lambda x: pd.cut(x.rank(), bins=n_bins, labels=False) + 1
        )

    # Aggregate: subject × condition × bin
    subj_bin = df.groupby(['subject_id', 'stim_condition', 'trial_bin'])['correct_num'].mean().reset_index()
    subj_bin.columns = ['subject_id', 'condition', 'bin', 'accuracy']

    # Group means and SEs
    group = subj_bin.groupby(['condition', 'bin'])['accuracy'].agg(['mean', 'sem']).reset_index()
    group.columns = ['condition', 'bin', 'mean', 'sem']

    fig, ax = plt.subplots(figsize=figsize)

    colors = {'sham': '#1565C0', 'active': '#E64A19'}
    labels = {'sham': 'Sham', 'active': 'Active'}

    for cond in ['sham', 'active']:
        cond_data = group[group['condition'] == cond].sort_values('bin')
        x = cond_data['bin'].values
        y = cond_data['mean'].values
        se = cond_data['sem'].values

        ax.plot(x, y, color=colors[cond], linewidth=2.5, label=labels[cond],
                marker='o', markersize=5)
        ax.fill_between(x, y - se, y + se, color=colors[cond], alpha=0.15)

    # Annotate improvements
    for cond in ['sham', 'active']:
        cond_data = group[group['condition'] == cond].sort_values('bin')
        first = cond_data.iloc[0]['mean']
        last = cond_data.iloc[-1]['mean']
        improvement = last - first
        x_pos = cond_data.iloc[-1]['bin']
        y_pos = last + 0.01
        ax.annotate(f'{improvement:+.1%}',
                    xy=(x_pos, y_pos), fontsize=10, color=colors[cond],
                    fontweight='bold', ha='center', va='bottom')

    ax.set_xlabel('Trial Bin', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Within-Run Learning Curves', fontsize=16)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.set_ylim(0.45, 0.85)

    sns.despine(ax=ax)
    fig.tight_layout()

    if output_path:
        _save_fig(fig, output_path)
    if show:
        plt.show()

    return fig


def plot_contingency_learning_curves(
    trial_df: pd.DataFrame,
    max_trial: int = 12,
    figsize: Tuple = (7, 5),
    output_path: str = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot 5: Contingency-phase learning curves, Sham vs Active.
    X-axis: trial number within contingency phase, Y-axis: mean accuracy.
    """
    _set_style()

    df = trial_df.copy()
    df['correct_num'] = pd.to_numeric(df['correct'].map({'True': 1, 'False': 0, True: 1, False: 0, 1: 1, 0: 0}),
                                       errors='coerce')
    if df['correct_num'].isna().all():
        df['correct_num'] = pd.to_numeric(df['correct'], errors='coerce')

    df['stim_condition'] = df['stim_condition'].str.strip().str.lower()
    df = df[df['stim_condition'].isin(['sham', 'active'])].copy()

    if 'trial_in_contingency' not in df.columns:
        print('  trial_in_contingency column not found. Cannot generate contingency learning curves.')
        return None

    # Cap trial number for cleaner plotting
    df['trial_capped'] = df['trial_in_contingency'].clip(upper=max_trial)

    # Bin the last few trials together
    df.loc[df['trial_in_contingency'] >= max_trial, 'trial_capped'] = max_trial

    # Aggregate: subject × condition × trial
    subj_trial = df.groupby(['subject_id', 'stim_condition', 'trial_capped'])['correct_num'].mean().reset_index()
    subj_trial.columns = ['subject_id', 'condition', 'trial', 'accuracy']

    # Group means and SEs
    group = subj_trial.groupby(['condition', 'trial'])['accuracy'].agg(['mean', 'sem']).reset_index()
    group.columns = ['condition', 'trial', 'mean', 'sem']

    fig, ax = plt.subplots(figsize=figsize)

    colors = {'sham': '#1565C0', 'active': '#E64A19'}
    labels_map = {'sham': 'Sham', 'active': 'Active'}

    for cond in ['sham', 'active']:
        cond_data = group[group['condition'] == cond].sort_values('trial')
        x = cond_data['trial'].values
        y = cond_data['mean'].values
        se = cond_data['sem'].values

        ax.plot(x, y, color=colors[cond], linewidth=2.5, label=labels_map[cond],
                marker='o', markersize=5)
        ax.fill_between(x, y - se, y + se, color=colors[cond], alpha=0.15)

    # Custom x-axis labels
    tick_labels = [str(i) for i in range(1, max_trial)] + [f'{max_trial}+']
    tick_vals = list(range(1, max_trial + 1))
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels)

    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Chance')

    ax.set_xlabel('Trial Within Contingency Phase', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Contingency-Phase Learning Curves', fontsize=16)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.set_ylim(0.3, 0.85)

    sns.despine(ax=ax)
    fig.tight_layout()

    if output_path:
        _save_fig(fig, output_path)
    if show:
        plt.show()

    return fig


# =============================================================================
# Master function
# =============================================================================

def generate_all_figures(
    subj_df: pd.DataFrame,
    trial_df: pd.DataFrame = None,
    output_dir: str = 'figures/',
    show: bool = True,
):
    """
    Generate all presentation/publication figures.

    Parameters
    ----------
    subj_df : DataFrame
        Subject-level data with all behavioral and survey variables.
    trial_df : DataFrame
        Trial-level data for learning curve plots. If None, learning curves are skipped.
    output_dir : str
        Directory to save figures.
    show : bool
        Whether to display figures inline.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('GENERATING PUBLICATION FIGURES')
    print('=' * 60)

    # --- Slide 26: J-N Plots ---
    print('\n--- Slide 26: Johnson-Neyman Plots ---')

    # Plot 1: Cognition × Age → p(shift|lose)
    print('\nPlot 1: Cognition × Age → p(shift|lose)')
    plot_jn(
        dv='sham_p_shift_lose', predictor='global_composite', moderator='age',
        data=subj_df, covariates=['education_years'],
        dv_label='p(shift|lose)', predictor_label='Global Cognition',
        moderator_label='Age',
        interaction_p=0.013, delta_r2=0.221,
        output_path=str(out / 'jn_cognition_age_shift'),
        show=show,
    )

    # Plot 1b: Cognition × Age → α
    print('\nPlot 1b: Cognition × Age → α (learning rate)')
    plot_jn(
        dv='sham_alpha', predictor='global_composite', moderator='age',
        data=subj_df, covariates=['education_years'],
        dv_label='α (Learning Rate)', predictor_label='Global Cognition',
        moderator_label='Age',
        interaction_p=0.021, delta_r2=0.200,
        output_path=str(out / 'jn_cognition_age_alpha'),
        show=show,
    )

    # Plot 1c: Cognition × Age → β
    print('\nPlot 1c: Cognition × Age → β (inv. temperature)')
    plot_jn(
        dv='sham_beta', predictor='global_composite', moderator='age',
        data=subj_df, covariates=['education_years'],
        dv_label='β (Inv. Temperature)', predictor_label='Global Cognition',
        moderator_label='Age',
        interaction_p=0.017, delta_r2=0.181,
        output_path=str(out / 'jn_cognition_age_beta'),
        show=show,
    )

    # --- Slide 27: Age × Δ Scatterplots ---
    print('\n--- Slide 27: Age × Stimulation Effect ---')

    print('\nPlots 2–3: Age × Δ Accuracy and Δ Win Rate (panel)')
    plot_age_delta_panel(
        data=subj_df,
        output_path=str(out / 'age_delta_accuracy_winrate_panel'),
        show=show,
    )

    # Also generate individual plots
    print('\nPlot 2: Age × Δ Accuracy')
    plot_age_delta_scatter(
        data=subj_df, delta_var='delta_accuracy',
        delta_label='Δ Accuracy (Active − Sham)',
        output_path=str(out / 'age_delta_accuracy'),
        show=show,
    )

    print('\nPlot 3: Age × Δ Win Rate')
    plot_age_delta_scatter(
        data=subj_df, delta_var='delta_win_rate',
        delta_label='Δ Win Rate (Active − Sham)',
        output_path=str(out / 'age_delta_winrate'),
        show=show,
    )

    # --- Slide 28: Learning Curves ---
    if trial_df is not None:
        print('\n--- Slide 28: Learning Curves ---')

        print('\nPlot 4: Within-run learning curves')
        plot_within_run_learning_curves(
            trial_df=trial_df,
            output_path=str(out / 'learning_curves_within_run'),
            show=show,
        )

        print('\nPlot 5: Contingency-phase learning curves')
        plot_contingency_learning_curves(
            trial_df=trial_df,
            output_path=str(out / 'learning_curves_contingency'),
            show=show,
        )
    else:
        print('\n--- Slide 28: Skipped (no trial_df provided) ---')
        print('  To generate learning curves, pass trial_df to generate_all_figures().')
        print('  trial_df should be the trial-level DataFrame with columns:')
        print('    subject_id, run, stim_condition, trial_num, trial_in_contingency, correct')

    print(f'\n{"=" * 60}')
    print(f'All figures saved to: {out.resolve()}')
    print(f'{"=" * 60}')
