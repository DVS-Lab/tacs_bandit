"""
hypothesis_tests.py — Pre-registered hypothesis tests for tACS Bandit study

Implements tests from pre-registration:

H1 (Sham-only, individual differences):
- H1.1: Cognitive function → baseline WSLS
- H1.2: SPSRQ sensitivity → baseline WSLS (independent of cognition)

H2 (Active vs. Sham):
- H2.1: Paired t-tests with TOST equivalence testing
- H2.2: Age moderating tACS-induced change

All tests use the appropriate filtered datasets (data_h1, data_h2) and
subj_df for subject-level analyses.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from typing import Optional, Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
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


def _age_to_rgb(age: float) -> str:
    """Convert age to RGB color string."""
    cmap = _get_age_colormap()
    norm = np.clip((age - AGE_MIN) / (AGE_MAX - AGE_MIN), 0, 1)
    rgba = cmap(norm)
    return f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'


# =============================================================================
# OLS Regression Helper
# =============================================================================

def run_ols(
    dv: str,
    predictors: List[str],
    data: pd.DataFrame,
    verbose: bool = True,
    model_label: Optional[str] = None
) -> Optional[Dict]:
    """
    Run OLS regression with standardized predictors.
    
    Parameters
    ----------
    dv : str
        Dependent variable column name
    predictors : list
        Predictor column names
    data : DataFrame
        Subject-level data
    verbose : bool
        If True, print summary
    model_label : str, optional
        Label for printed output
    
    Returns
    -------
    dict with model, r2, adj_r2, f_stat, f_pval, coefficients
    """
    # Filter to complete cases
    cols = [dv] + predictors
    df = data[cols].dropna()
    
    if len(df) < len(predictors) + 2:
        if verbose:
            print(f'  {model_label or dv}: Insufficient data (n={len(df)})')
        return None
    
    # Prepare design matrix
    X = df[predictors].copy()
    y = df[dv].values
    
    # Standardize predictors for comparable coefficients
    X_std = (X - X.mean()) / X.std()
    X_std = sm.add_constant(X_std)
    
    # Fit model
    model = sm.OLS(y, X_std).fit()
    
    result = {
        'model': model,
        'n': len(df),
        'r2': model.rsquared,
        'adj_r2': model.rsquared_adj,
        'f_stat': model.fvalue,
        'f_pval': model.f_pvalue,
        'coefficients': model.params.to_dict(),
        'pvalues': model.pvalues.to_dict(),
    }
    
    if verbose:
        label = model_label or f'{dv} ~ {" + ".join(predictors)}'
        print(f'\n{label}')
        print(f'  N = {len(df)}')
        print(f'  R² = {model.rsquared:.3f}, Adj R² = {model.rsquared_adj:.3f}')
        print(f'  F({model.df_model:.0f}, {model.df_resid:.0f}) = {model.fvalue:.2f}, p = {model.f_pvalue:.4f}')
        print(f'  Coefficients (standardized):')
        for pred in predictors:
            coef = model.params[pred]
            p = model.pvalues[pred]
            sig = '*' if p < 0.05 else ''
            print(f'    {pred}: β = {coef:+.3f}, p = {p:.4f} {sig}')
    
    return result


def compute_bivariate_correlation(
    x_col: str,
    y_col: str,
    data: pd.DataFrame,
    verbose: bool = True
) -> Optional[Dict]:
    """Compute Pearson correlation between two variables."""
    df = data[[x_col, y_col]].dropna()
    
    if len(df) < 3:
        return None
    
    r, p = stats.pearsonr(df[x_col], df[y_col])
    
    if verbose:
        print(f'  r({x_col}, {y_col}) = {r:.3f}, p = {p:.4f}, n = {len(df)}')
    
    return {'r': r, 'p': p, 'n': len(df)}


# =============================================================================
# H1.1: Cognitive Function → Baseline WSLS
# =============================================================================

def test_h1_1(
    subj_df: pd.DataFrame,
    verbose: bool = True
) -> Dict:
    """
    H1.1: Cognitive function predicting baseline (sham) WSLS.
    
    Model: sham_p_stay_win / sham_p_shift_lose ~ global_composite + age + education
    
    Parameters
    ----------
    subj_df : DataFrame
        Subject-level data with cognitive composites and sham WSLS
    verbose : bool
        If True, print results
    
    Returns
    -------
    dict with h1_1a (stay|win) and h1_1b (shift|lose) results
    """
    if verbose:
        print('='*70)
        print('H1.1: Cognitive function predicting baseline (sham) behavior')
        print('='*70)
    
    results = {}
    
    # H1.1a: p(stay|win) ~ global_composite + age + education
    predictors = ['global_composite', 'age', 'education_years']
    h1_1a = run_ols('sham_p_stay_win', predictors, subj_df, verbose=verbose,
                    model_label='H1.1a: p(stay|win) ~ Global Cog + Age + Education')
    results['h1_1a'] = h1_1a
    
    # H1.1b: p(shift|lose) ~ global_composite + age + education
    h1_1b = run_ols('sham_p_shift_lose', predictors, subj_df, verbose=verbose,
                    model_label='H1.1b: p(shift|lose) ~ Global Cog + Age + Education')
    results['h1_1b'] = h1_1b
    
    # Bivariate correlations for visualization
    if verbose:
        print('\nBivariate correlations:')
        for dv in ['sham_p_stay_win', 'sham_p_shift_lose']:
            if dv in subj_df.columns:
                compute_bivariate_correlation('global_composite', dv, subj_df)
    
    return results


# =============================================================================
# H1.2: SPSRQ Sensitivity → Baseline WSLS
# =============================================================================

def test_h1_2(
    subj_df: pd.DataFrame,
    h1_1_results: Optional[Dict] = None,
    verbose: bool = True
) -> Dict:
    """
    H1.2: Reward/punishment sensitivity predicting baseline WSLS,
    independent of cognitive function.
    
    Models:
    - p(stay|win) ~ SPSRQ-SR + global_composite + age + education
    - p(shift|lose) ~ SPSRQ-SP + global_composite + age + education
    
    Key test: ΔR² from adding SPSRQ to H1.1 models.
    
    Parameters
    ----------
    subj_df : DataFrame
        Subject-level data
    h1_1_results : dict, optional
        Results from test_h1_1() for ΔR² computation
    verbose : bool
        If True, print results
    
    Returns
    -------
    dict with h1_2a, h1_2b, and delta_r2 values
    """
    if verbose:
        print('='*70)
        print('H1.2: Reward/punishment sensitivity predicting baseline behavior')
        print('      (independent of cognitive function)')
        print('='*70)
    
    results = {}
    
    # H1.2a: p(stay|win) ~ SR + global_composite + age + education
    predictors_a = ['spsrq_sr', 'global_composite', 'age', 'education_years']
    h1_2a = run_ols('sham_p_stay_win', predictors_a, subj_df, verbose=verbose,
                    model_label='H1.2a: p(stay|win) ~ SPSRQ-SR + Global Cog + Age + Education')
    results['h1_2a'] = h1_2a
    
    # ΔR² for SR
    if h1_2a is not None and h1_1_results and h1_1_results.get('h1_1a'):
        delta_r2_a = h1_2a['r2'] - h1_1_results['h1_1a']['r2']
        results['delta_r2_sr'] = delta_r2_a
        if verbose:
            print(f'  ΔR² (adding SR to H1.1a): {delta_r2_a:+.3f}')
            sr_p = h1_2a['pvalues'].get('spsrq_sr', np.nan)
            print(f'  SR unique contribution: p = {sr_p:.4f}')
    
    # H1.2b: p(shift|lose) ~ SP + global_composite + age + education
    predictors_b = ['spsrq_sp', 'global_composite', 'age', 'education_years']
    h1_2b = run_ols('sham_p_shift_lose', predictors_b, subj_df, verbose=verbose,
                    model_label='H1.2b: p(shift|lose) ~ SPSRQ-SP + Global Cog + Age + Education')
    results['h1_2b'] = h1_2b
    
    # ΔR² for SP
    if h1_2b is not None and h1_1_results and h1_1_results.get('h1_1b'):
        delta_r2_b = h1_2b['r2'] - h1_1_results['h1_1b']['r2']
        results['delta_r2_sp'] = delta_r2_b
        if verbose:
            print(f'  ΔR² (adding SP to H1.1b): {delta_r2_b:+.3f}')
            sp_p = h1_2b['pvalues'].get('spsrq_sp', np.nan)
            print(f'  SP unique contribution: p = {sp_p:.4f}')
    
    # Complementary: Task-derived β as sensitivity measure
    if verbose:
        print('\n' + '-'*50)
        print('Complementary: Task-derived β as sensitivity measure')
        print('-'*50)
    
    predictors_beta = ['sham_beta', 'global_composite', 'age', 'education_years']
    run_ols('sham_p_stay_win', predictors_beta, subj_df, verbose=verbose,
            model_label='p(stay|win) ~ β + Global Cog + Age + Education')
    run_ols('sham_p_shift_lose', predictors_beta, subj_df, verbose=verbose,
            model_label='p(shift|lose) ~ β + Global Cog + Age + Education')
    
    return results


# =============================================================================
# H2.1: Paired t-tests with TOST
# =============================================================================

def paired_ttest_with_tost(
    sham_col: str,
    active_col: str,
    data: pd.DataFrame,
    label: str,
    equiv_bound_dz: float = 0.3,
    verbose: bool = True
) -> Optional[Dict]:
    """
    Paired t-test with Cohen's dz and TOST equivalence test.
    
    Parameters
    ----------
    sham_col : str
        Sham condition column
    active_col : str
        Active condition column
    data : DataFrame
        Subject-level data
    label : str
        Label for output
    equiv_bound_dz : float
        Equivalence bound in Cohen's dz units
    verbose : bool
        If True, print results
    
    Returns
    -------
    dict with n, t, p, dz, p_tost, ci_low, ci_high
    """
    df = data[[sham_col, active_col]].dropna()
    n = len(df)
    
    if n < 3:
        if verbose:
            print(f'  {label}: Insufficient data (n={n})')
        return None
    
    diff = df[active_col] - df[sham_col]
    t_stat, p_val = stats.ttest_rel(df[active_col], df[sham_col])
    dz = diff.mean() / diff.std() if diff.std() > 0 else 0
    
    # 95% CI
    se = diff.std() / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_low = diff.mean() - t_crit * se
    ci_high = diff.mean() + t_crit * se
    
    # TOST equivalence test
    bound_raw = equiv_bound_dz * diff.std()
    
    # Upper bound test: H0: diff >= +bound
    t_upper = (diff.mean() - bound_raw) / se
    p_upper = stats.t.cdf(t_upper, df=n-1)
    
    # Lower bound test: H0: diff <= -bound
    t_lower = (diff.mean() + bound_raw) / se
    p_lower = 1 - stats.t.cdf(t_lower, df=n-1)
    
    p_tost = max(p_upper, p_lower)
    
    result = {
        'n': n,
        't': t_stat,
        'p': p_val,
        'dz': dz,
        'mean_diff': diff.mean(),
        'sd_diff': diff.std(),
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_tost': p_tost,
        'sham_mean': df[sham_col].mean(),
        'sham_sd': df[sham_col].std(),
        'active_mean': df[active_col].mean(),
        'active_sd': df[active_col].std(),
    }
    
    if verbose:
        print(f'  {label}:')
        print(f'    N = {n}')
        print(f'    Sham:   M = {result["sham_mean"]:.3f}, SD = {result["sham_sd"]:.3f}')
        print(f'    Active: M = {result["active_mean"]:.3f}, SD = {result["active_sd"]:.3f}')
        print(f'    Diff:   M = {diff.mean():.4f}, SD = {diff.std():.4f}')
        print(f'    95% CI: [{ci_low:.4f}, {ci_high:.4f}]')
        print(f"    t({n-1}) = {t_stat:.3f}, p = {p_val:.4f}, Cohen's dz = {dz:.3f}")
        
        if p_val >= 0.05:
            print(f'    TOST equivalence (±{equiv_bound_dz} dz): p = {p_tost:.4f}', end='')
            if p_tost < 0.05:
                print(' → equivalent')
            else:
                print(' → inconclusive')
        print()
    
    return result


# =============================================================================
# Exploratory: Age × Predictor Interactions for H1
# =============================================================================

def run_ols_with_interaction(
    dv: str,
    predictor: str,
    moderator: str,
    data: pd.DataFrame,
    covariates: List[str] = None,
    verbose: bool = True,
    model_label: Optional[str] = None,
) -> Optional[Dict]:
    """
    Run OLS with a predictor × moderator interaction term.

    All continuous variables are mean-centered before creating the interaction
    to reduce multicollinearity and aid interpretation.
    """
    if covariates is None:
        covariates = []

    # Deduplicate
    covariates = [c for c in covariates if c != predictor and c != moderator and c != dv]

    all_cols = list(dict.fromkeys([dv, predictor, moderator] + covariates))
    df = data[all_cols].copy()
    for col in all_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()

    if len(df) < len(covariates) + 4:
        if verbose:
            print(f'  {model_label or dv}: Insufficient data (n={len(df)})')
        return None

    y = df[dv].values

    # Mean-center for interaction
    pred_c = df[predictor] - df[predictor].mean()
    mod_c = df[moderator] - df[moderator].mean()
    interaction = pred_c * mod_c

    X_dict = {predictor: pred_c, moderator: mod_c}
    for cov in covariates:
        X_dict[cov] = df[cov] - df[cov].mean()
    interaction_name = f'{predictor} × {moderator}'
    X_dict[interaction_name] = interaction

    X = pd.DataFrame(X_dict)
    X_std = (X - X.mean()) / X.std()
    X_std = sm.add_constant(X_std)

    model = sm.OLS(y, X_std).fit()

    # Model WITHOUT interaction for ΔR²
    X_no_int = X.drop(columns=[interaction_name])
    X_no_int_std = (X_no_int - X_no_int.mean()) / X_no_int.std()
    X_no_int_std = sm.add_constant(X_no_int_std)
    model_no_int = sm.OLS(y, X_no_int_std).fit()

    delta_r2 = model.rsquared - model_no_int.rsquared
    int_idx = list(X_std.columns).index(interaction_name)

    result = {
        'model': model,
        'model_no_interaction': model_no_int,
        'n': len(df),
        'r2': model.rsquared,
        'adj_r2': model.rsquared_adj,
        'r2_no_interaction': model_no_int.rsquared,
        'delta_r2_interaction': delta_r2,
        'interaction_beta': model.params[int_idx],
        'interaction_p': model.pvalues[int_idx],
        'f_stat': model.fvalue,
        'f_pval': model.f_pvalue,
    }

    if verbose:
        label = model_label or f'{dv} ~ {predictor} × {moderator}'
        print(f'\n{label}')
        print(f'  N = {result["n"]}')
        print(f'  Full model (with interaction): R² = {result["r2"]:.3f}, '
              f'Adj R² = {result["adj_r2"]:.3f}')
        print(f'  Without interaction:           R² = {result["r2_no_interaction"]:.3f}')
        print(f'  ΔR² (interaction):             {delta_r2:+.3f}')
        print(f'  Coefficients (standardized):')
        for pred, beta, p in zip(X_std.columns, model.params, model.pvalues):
            if pred == 'const':
                continue
            s = '*' if p < 0.05 else ''
            print(f'    {pred:30s}: β = {beta:+.3f}, p = {p:.4f} {s}')

    return result


def test_h1_age_interactions(
    subj_df: pd.DataFrame,
    verbose: bool = True
) -> Dict:
    """
    Exploratory: Test whether age moderates H1 predictor-behavior relationships.

    For each H1 predictor (global cognition, SPSRQ-SR, SPSRQ-SP), tests whether
    the predictor × age interaction predicts baseline WSLS and RW parameters.
    """
    results = {}

    if verbose:
        print('=' * 70)
        print('Exploratory: Age as Moderator of H1 Predictor-Behavior Relationships')
        print('=' * 70)

    # H1.1: Global Cognition × Age → WSLS
    if verbose:
        print('\n--- Global Cognition × Age → Baseline WSLS ---')

    results['cog_x_age_stay'] = run_ols_with_interaction(
        dv='sham_p_stay_win', predictor='global_composite', moderator='age',
        data=subj_df, covariates=['education_years'],
        verbose=verbose,
        model_label='p(stay|win) ~ Global Cog × Age + Education')

    results['cog_x_age_shift'] = run_ols_with_interaction(
        dv='sham_p_shift_lose', predictor='global_composite', moderator='age',
        data=subj_df, covariates=['education_years'],
        verbose=verbose,
        model_label='p(shift|lose) ~ Global Cog × Age + Education')

    # H1.2: SPSRQ × Age → WSLS
    if verbose:
        print('\n--- SPSRQ × Age → Baseline WSLS ---')

    results['sr_x_age_stay'] = run_ols_with_interaction(
        dv='sham_p_stay_win', predictor='spsrq_sr', moderator='age',
        data=subj_df, covariates=['global_composite', 'education_years'],
        verbose=verbose,
        model_label='p(stay|win) ~ SPSRQ-SR × Age + Global Cog + Education')

    results['sp_x_age_shift'] = run_ols_with_interaction(
        dv='sham_p_shift_lose', predictor='spsrq_sp', moderator='age',
        data=subj_df, covariates=['global_composite', 'education_years'],
        verbose=verbose,
        model_label='p(shift|lose) ~ SPSRQ-SP × Age + Global Cog + Education')

    # RW parameters
    if verbose:
        print('\n--- Global Cognition × Age → Baseline RW Parameters ---')

    results['cog_x_age_alpha'] = run_ols_with_interaction(
        dv='sham_alpha', predictor='global_composite', moderator='age',
        data=subj_df, covariates=['education_years'],
        verbose=verbose,
        model_label='α ~ Global Cog × Age + Education')

    results['cog_x_age_beta'] = run_ols_with_interaction(
        dv='sham_beta', predictor='global_composite', moderator='age',
        data=subj_df, covariates=['education_years'],
        verbose=verbose,
        model_label='β ~ Global Cog × Age + Education')

    # Summary
    if verbose:
        sig = [(k, v) for k, v in results.items()
               if v is not None and v['interaction_p'] < 0.05]
        marginal = [(k, v) for k, v in results.items()
                    if v is not None and 0.05 <= v['interaction_p'] < 0.10]
        print(f'\n--- Summary ---')
        print(f'  Significant interactions (p < .05): {len(sig)}')
        for k, v in sig:
            print(f'    {k}: β = {v["interaction_beta"]:+.3f}, p = {v["interaction_p"]:.4f}')
        if marginal:
            print(f'  Marginal interactions (.05 < p < .10): {len(marginal)}')
            for k, v in marginal:
                print(f'    {k}: β = {v["interaction_beta"]:+.3f}, p = {v["interaction_p"]:.4f}')

    return results


def plot_johnson_neyman(
    dv: str,
    predictor: str,
    moderator: str,
    data: pd.DataFrame,
    covariates: List[str] = None,
    dv_label: Optional[str] = None,
    predictor_label: Optional[str] = None,
    moderator_label: Optional[str] = None,
    n_points: int = 200,
    show_fig: bool = True,
) -> Optional[go.Figure]:
    """
    Create a Johnson-Neyman plot showing the conditional effect of predictor
    on DV across values of the moderator.

    The plot shows:
      - The simple slope of predictor → DV at each moderator value (blue line)
      - 95% confidence band (shaded)
      - Region(s) of significance (where CI excludes zero, shaded green)
      - Johnson-Neyman point(s) (vertical dashed red lines)
      - Rug plot of actual moderator values

    Parameters
    ----------
    dv, predictor, moderator : str
        Column names in data
    data : DataFrame
    covariates : list, optional
        Additional covariates to include in the model
    dv_label, predictor_label, moderator_label : str, optional
        Display labels for axes/title
    """
    if covariates is None:
        covariates = []
    covariates = [c for c in covariates if c != predictor and c != moderator and c != dv]

    all_cols = list(dict.fromkeys([dv, predictor, moderator] + covariates))
    df = data[all_cols].copy()
    for col in all_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()

    if len(df) < 6:
        return None

    y = df[dv].values
    pred_label = predictor_label or predictor
    mod_label = moderator_label or moderator
    dv_lab = dv_label or dv

    # Mean-center predictor and covariates (NOT moderator — we probe across its range)
    pred_c = df[predictor].values - df[predictor].mean()
    mod_vals = df[moderator].values
    mod_mean = mod_vals.mean()

    # Build design matrix with raw (centered) values
    X_dict = {'const': np.ones(len(df)), predictor: pred_c, moderator: mod_vals - mod_mean}
    for cov in covariates:
        X_dict[cov] = df[cov].values - df[cov].mean()
    X_dict[f'{predictor}x{moderator}'] = pred_c * (mod_vals - mod_mean)

    X = np.column_stack(list(X_dict.values()))
    col_names = list(X_dict.keys())

    # Fit OLS
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = sm.OLS(y, X).fit()

    # Get indices for key parameters
    b_pred = model.params[col_names.index(predictor)]
    b_int = model.params[col_names.index(f'{predictor}x{moderator}')]
    pred_idx = col_names.index(predictor)
    int_idx = col_names.index(f'{predictor}x{moderator}')

    # Compute simple slopes across moderator range
    mod_range = np.linspace(df[moderator].min() - 1, df[moderator].max() + 1, n_points)
    mod_centered = mod_range - mod_mean

    simple_slopes = b_pred + b_int * mod_centered

    # Compute SE of simple slope at each moderator value
    # Var(b1 + b3*W) = Var(b1) + W²*Var(b3) + 2*W*Cov(b1,b3)
    vcov = model.cov_params()
    var_b1 = vcov[pred_idx, pred_idx]
    var_b3 = vcov[int_idx, int_idx]
    cov_b1b3 = vcov[pred_idx, int_idx]

    se_slopes = np.sqrt(var_b1 + mod_centered**2 * var_b3 + 2 * mod_centered * cov_b1b3)

    # t-critical for 95% CI
    from scipy.stats import t as t_dist
    df_resid = model.df_resid
    t_crit = t_dist.ppf(0.975, df_resid)

    ci_lower = simple_slopes - t_crit * se_slopes
    ci_upper = simple_slopes + t_crit * se_slopes

    # Find Johnson-Neyman points (where CI crosses zero)
    # Solve: simple_slope ± t_crit * se_slope = 0
    # (b1 + b3*w)² = t²(Var(b1) + w²*Var(b3) + 2w*Cov(b1,b3))
    # Rearrange to quadratic in w:
    # (b3² - t²*Var(b3))w² + (2*b1*b3 - 2*t²*Cov(b1,b3))w + (b1² - t²*Var(b1)) = 0
    a_coef = b_int**2 - t_crit**2 * var_b3
    b_coef = 2 * b_pred * b_int - 2 * t_crit**2 * cov_b1b3
    c_coef = b_pred**2 - t_crit**2 * var_b1

    discriminant = b_coef**2 - 4 * a_coef * c_coef
    jn_points_centered = []
    if discriminant >= 0 and abs(a_coef) > 1e-10:
        w1 = (-b_coef + np.sqrt(discriminant)) / (2 * a_coef)
        w2 = (-b_coef - np.sqrt(discriminant)) / (2 * a_coef)
        jn_points_centered = [w1, w2]

    jn_points = [w + mod_mean for w in jn_points_centered]
    # Filter to within data range (±10%)
    mod_min, mod_max = df[moderator].min(), df[moderator].max()
    mod_span = mod_max - mod_min
    jn_points = [p for p in jn_points if mod_min - 0.1*mod_span <= p <= mod_max + 0.1*mod_span]

    # Determine significance regions
    is_significant = (ci_lower > 0) | (ci_upper < 0)

    # Build figure
    fig = go.Figure()

    # Significance shading
    sig_ranges = []
    in_sig = False
    for i, s in enumerate(is_significant):
        if s and not in_sig:
            start_i = i
            in_sig = True
        elif not s and in_sig:
            sig_ranges.append((start_i, i - 1))
            in_sig = False
    if in_sig:
        sig_ranges.append((start_i, len(is_significant) - 1))

    for start_i, end_i in sig_ranges:
        fig.add_vrect(
            x0=mod_range[start_i], x1=mod_range[end_i],
            fillcolor='rgba(46, 125, 50, 0.1)', line_width=0,
            annotation_text='Significant', annotation_position='top left',
            annotation_font=dict(size=10, color='#2E7D32'),
        )

    # Confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([mod_range, mod_range[::-1]]),
        y=np.concatenate([ci_upper, ci_lower[::-1]]),
        fill='toself', fillcolor='rgba(21, 101, 192, 0.15)',
        line=dict(width=0), showlegend=False, hoverinfo='skip',
    ))

    # Simple slope line
    fig.add_trace(go.Scatter(
        x=mod_range, y=simple_slopes,
        mode='lines', line=dict(color='#1565C0', width=2.5),
        name='Simple slope',
        hovertemplate=f'{mod_label}: %{{x:.1f}}<br>Slope: %{{y:.4f}}<extra></extra>',
    ))

    # Zero reference
    fig.add_hline(y=0, line_dash='dot', line_color='gray', line_width=1)

    # J-N points
    for jn in jn_points:
        fig.add_vline(x=jn, line_dash='dash', line_color='#C62828', line_width=1.5)
        fig.add_annotation(
            x=jn, y=max(ci_upper) * 0.9,
            text=f'J-N: {jn:.1f}', showarrow=False,
            font=dict(color='#C62828', size=11),
        )

    # Rug plot of actual moderator values
    for val in df[moderator].values:
        fig.add_trace(go.Scatter(
            x=[val], y=[min(ci_lower) * 1.1],
            mode='markers', marker=dict(symbol='line-ns', size=8,
                                         color='black', line_width=1),
            showlegend=False, hoverinfo='skip',
        ))

    fig.update_layout(
        height=400, width=600,
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY, size=12),
        xaxis_title=mod_label,
        yaxis_title=f'Simple slope of {pred_label} → {dv_lab}',
        title=dict(
            text=f'Johnson-Neyman Plot: {pred_label} × {mod_label} → {dv_lab}'
                 f'<br><sub>Shaded region = 95% CI; green = significant (p < .05)</sub>',
            font=dict(size=13),
        ),
        showlegend=False,
        margin=dict(l=70, r=40, t=80, b=60),
    )
    fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='black')

    filename = f'jn_{predictor}_{moderator}_{dv}'.replace(' ', '_')
    if show_fig:
        fig.show(config=dict(toImageButtonOptions=dict(filename=filename)))

    return fig


def test_h2_1(
    subj_df: pd.DataFrame,
    verbose: bool = True
) -> Dict:
    """
    H2.1: Paired comparisons — tACS vs. Sham.
    
    Tests WSLS and R-W parameters with TOST equivalence testing.
    
    Parameters
    ----------
    subj_df : DataFrame
        Subject-level data with sham_* and active_* columns
    verbose : bool
        If True, print results
    
    Returns
    -------
    dict with results for each parameter
    """
    if verbose:
        print('='*70)
        print('H2.1: Paired comparisons — tACS vs. Sham')
        print('='*70)
        print()
    
    results = {}
    
    # WSLS parameters
    if verbose:
        print('--- WSLS Parameters ---')
    
    results['p_stay_win'] = paired_ttest_with_tost(
        'sham_p_stay_win', 'active_p_stay_win', subj_df, 'p(stay|win)', verbose=verbose)
    
    results['p_shift_lose'] = paired_ttest_with_tost(
        'sham_p_shift_lose', 'active_p_shift_lose', subj_df, 'p(shift|lose)', verbose=verbose)
    
    # R-W parameters
    if verbose:
        print('--- Rescorla-Wagner Parameters ---')
    
    results['alpha'] = paired_ttest_with_tost(
        'sham_alpha', 'active_alpha', subj_df, 'α (learning rate)', verbose=verbose)
    
    results['beta'] = paired_ttest_with_tost(
        'sham_beta', 'active_beta', subj_df, 'β (inv. temperature)', verbose=verbose)
    
    return results


# =============================================================================
# H2.2: Age Moderating tACS Effect
# =============================================================================

def test_h2_2(
    subj_df: pd.DataFrame,
    verbose: bool = True
) -> Dict:
    """
    H2.2: Age as predictor of tACS-induced behavioral change.
    
    Model: delta_* ~ age + global_composite
    
    Parameters
    ----------
    subj_df : DataFrame
        Subject-level data with delta_* change scores
    verbose : bool
        If True, print results
    
    Returns
    -------
    dict with results for each change score
    """
    if verbose:
        print('='*70)
        print('H2.2: Age as predictor of tACS-induced behavioral change')
        print('='*70)
        print()
    
    results = {}
    predictors = ['age', 'global_composite']
    
    dvs = [
        ('delta_p_stay_win', 'Δ p(stay|win)'),
        ('delta_p_shift_lose', 'Δ p(shift|lose)'),
        ('delta_alpha', 'Δ α'),
        ('delta_beta', 'Δ β'),
    ]
    
    for dv, label in dvs:
        if dv in subj_df.columns:
            result = run_ols(dv, predictors, subj_df, verbose=verbose,
                            model_label=f'{label} ~ Age + Global Cog')
            results[dv] = result
    
    return results


# =============================================================================
# Exploratory: Theta Moderation of tACS Effect
# =============================================================================

def test_theta_moderation(
    subj_df: pd.DataFrame,
    verbose: bool = True
) -> Dict:
    """
    Exploratory: Theta reactivity as predictor/moderator of tACS effect.
    
    Models:
    1. Bivariate correlations: theta_p95 with change scores
    2. Multiple regression: delta_* ~ theta_p95 + age + global_composite
    
    Parameters
    ----------
    subj_df : DataFrame
        Subject-level data with theta_p95 and delta_* columns
    verbose : bool
        If True, print results
    
    Returns
    -------
    dict with correlation and regression results
    """
    if verbose:
        print('='*70)
        print('Exploratory: Theta Reactivity as Moderator of tACS Effect')
        print('='*70)
        print()
    
    results = {
        'correlations': {},
        'regressions': {}
    }
    
    if 'theta_p95' not in subj_df.columns:
        if verbose:
            print('  theta_p95 not available in data.')
        return results
    
    # Check N with theta data
    n_theta = subj_df['theta_p95'].notna().sum()
    if verbose:
        print(f'Subjects with theta data: {n_theta}')
        print()
    
    if n_theta < 5:
        if verbose:
            print('  Insufficient subjects with theta data.')
        return results
    
    # 1. Bivariate correlations with change scores
    if verbose:
        print('--- Bivariate Correlations: θ_p95 × Change Scores ---')
    
    change_vars = [
        ('delta_p_stay_win', 'Δ p(stay|win)'),
        ('delta_p_shift_lose', 'Δ p(shift|lose)'),
        ('delta_alpha', 'Δ α'),
        ('delta_beta', 'Δ β'),
    ]
    
    for var, label in change_vars:
        if var not in subj_df.columns:
            continue
        
        df_pair = subj_df[['theta_p95', var]].dropna()
        n = len(df_pair)
        
        if n < 5:
            continue
        
        r, p = stats.pearsonr(df_pair['theta_p95'], df_pair[var])
        
        results['correlations'][var] = {
            'r': r,
            'p': p,
            'n': n
        }
        
        if verbose:
            sig_marker = '*' if p < 0.05 else ''
            print(f'  θ_p95 × {label}: r = {r:.3f}, p = {p:.3f}{sig_marker}, n = {n}')
    
    # 2. Also check theta × baseline performance
    if verbose:
        print('\n--- Bivariate Correlations: θ_p95 × Baseline Performance ---')
    
    baseline_vars = [
        ('sham_p_stay_win', 'p(stay|win)'),
        ('sham_p_shift_lose', 'p(shift|lose)'),
        ('sham_alpha', 'α'),
        ('sham_beta', 'β'),
    ]
    
    for var, label in baseline_vars:
        if var not in subj_df.columns:
            continue
        
        df_pair = subj_df[['theta_p95', var]].dropna()
        n = len(df_pair)
        
        if n < 5:
            continue
        
        r, p = stats.pearsonr(df_pair['theta_p95'], df_pair[var])
        
        results['correlations'][f'baseline_{var}'] = {
            'r': r,
            'p': p,
            'n': n
        }
        
        if verbose:
            sig_marker = '*' if p < 0.05 else ''
            print(f'  θ_p95 × {label}: r = {r:.3f}, p = {p:.3f}{sig_marker}, n = {n}')
    
    # 3. Multiple regression with theta as predictor of change
    if verbose:
        print('\n--- Regression: Change Scores ~ θ_p95 + Age + Global Cog ---')
    
    predictors = ['theta_p95', 'age', 'global_composite']
    
    for var, label in change_vars:
        if var not in subj_df.columns:
            continue
        
        result = run_ols(var, predictors, subj_df, verbose=verbose,
                        model_label=f'{label} ~ θ_p95 + Age + Global')
        if result is not None:
            results['regressions'][var] = result
    
    if verbose:
        print()
    
    return results


def test_theta_baseline_predictors(
    subj_df: pd.DataFrame,
    verbose: bool = True
) -> Dict:
    """
    Exploratory: Theta reactivity predicting baseline learning.
    
    Adds theta_p95 to H1-style models to test whether neural measure
    predicts task performance independent of cognitive composites.
    
    Parameters
    ----------
    subj_df : DataFrame
        Subject-level data
    verbose : bool
        If True, print results
    
    Returns
    -------
    dict with regression results
    """
    if verbose:
        print('='*70)
        print('Exploratory: Theta Predicting Baseline Learning')
        print('='*70)
        print()
    
    results = {}
    
    if 'theta_p95' not in subj_df.columns:
        if verbose:
            print('  theta_p95 not available.')
        return results
    
    # Model: baseline DV ~ theta_p95 + global_composite + age
    predictors = ['theta_p95', 'global_composite', 'age']
    
    dvs = [
        ('sham_p_stay_win', 'p(stay|win)'),
        ('sham_p_shift_lose', 'p(shift|lose)'),
        ('sham_alpha', 'α'),
        ('sham_beta', 'β'),
    ]
    
    for dv, label in dvs:
        if dv in subj_df.columns:
            result = run_ols(dv, predictors, subj_df, verbose=verbose,
                            model_label=f'{label} ~ θ_p95 + Global + Age')
            results[dv] = result
    
    return results

def plot_h1_1_scatter(
    subj_df: pd.DataFrame,
    show_fig: bool = True
) -> go.Figure:
    """Plot H1.1 scatter: WSLS vs. global cognition."""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['p(stay | win) ~ Global Cognition', 'p(shift | lose) ~ Global Cognition'],
        horizontal_spacing=0.12
    )
    
    plot_specs = [
        ('sham_p_stay_win', 'p(stay | win)', [0.0, 1.0]),
        ('sham_p_shift_lose', 'p(shift | lose)', [0.0, 1.0]),
    ]
    
    for col, (dv, ylabel, ylim) in enumerate(plot_specs, start=1):
        df_plot = subj_df[['subject_id', 'global_composite', dv, 'age']].dropna()
        
        if len(df_plot) >= 3:
            # Scatter points
            fig.add_trace(go.Scatter(
                x=df_plot['global_composite'],
                y=df_plot[dv],
                mode='markers',
                marker=dict(size=11, opacity=0.8,
                            color=[_age_to_rgb(a) for a in df_plot['age']],
                            line=dict(width=0.5, color='white')),
                text=[f'sub-{s}<br>Age: {a:.0f}<br>Cog: {c:.2f}<br>{ylabel}: {y:.3f}'
                      for s, a, c, y in zip(df_plot['subject_id'], df_plot['age'],
                                            df_plot['global_composite'], df_plot[dv])],
                hoverinfo='text',
                showlegend=False,
            ), row=1, col=col)
            
            # Regression line
            slope, intercept, r, p, se = stats.linregress(df_plot['global_composite'], df_plot[dv])
            x_line = np.linspace(df_plot['global_composite'].min(), df_plot['global_composite'].max(), 100)
            y_line = intercept + slope * x_line
            
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                line=dict(color='#404040', width=2),
                showlegend=False, hoverinfo='skip',
            ), row=1, col=col)
            
            # Stats annotation
            xref = 'x domain' if col == 1 else f'x{col} domain'
            yref = 'y domain' if col == 1 else f'y{col} domain'
            
            fig.add_annotation(
                x=0.95, y=0.95,
                xref=xref, yref=yref,
                text=f'r = {r:.2f}, p = {p:.3f}',
                showarrow=False,
                font=dict(size=12, color='#404040'),
                xanchor='right', yanchor='top',
                bgcolor='rgba(255,255,255,0.8)'
            )
        
        fig.update_yaxes(title_text=ylabel, range=ylim, row=1, col=col)
        fig.update_xaxes(title_text='Global Cognitive Composite', row=1, col=col)
    
    fig.update_layout(
        height=400, width=850,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=60, r=100, t=60, b=60),
        font=dict(family=FONT_FAMILY, size=13),
    )
    
    fig.update_xaxes(showgrid=False, zeroline=False,
                     showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='black')
    
    # Colorbar
    fig.add_trace(go.Scatter(
        x=[None]*50, y=[None]*50, mode='markers',
        marker=dict(size=0.1, color=np.linspace(AGE_MIN, AGE_MAX, 50),
                    colorscale=AGE_COLORSCALE,
                    cmin=AGE_MIN, cmax=AGE_MAX,
                    colorbar=dict(x=1.08, y=0.5, len=0.6, thickness=10,
                                  title='Age', titleside='right', tickfont=dict(size=11))),
        showlegend=False, hoverinfo='skip',
    ))
    
    if show_fig:
        fig.show(config=dict(toImageButtonOptions=dict(filename='h1_1_cognition_wsls')))
    
    return fig


def plot_h2_1_paired(
    subj_df: pd.DataFrame,
    show_fig: bool = True
) -> go.Figure:
    """Plot H2.1 paired lines: sham vs. active."""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['p(stay | win)', 'p(shift | lose)'],
        horizontal_spacing=0.15
    )
    
    plot_specs = [
        ('sham_p_stay_win', 'active_p_stay_win', 'p(stay | win)', [0.0, 1.0]),
        ('sham_p_shift_lose', 'active_p_shift_lose', 'p(shift | lose)', [0.0, 1.0]),
    ]
    
    for col, (sham_col, active_col, ylabel, ylim) in enumerate(plot_specs, start=1):
        df_plot = subj_df[['subject_id', sham_col, active_col, 'age']].dropna()
        
        if len(df_plot) >= 2:
            # Individual lines
            for _, row in df_plot.iterrows():
                color = _age_to_rgb(row['age'])
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[row[sham_col], row[active_col]],
                    mode='lines+markers',
                    line=dict(color=color, width=1.5),
                    marker=dict(size=8, color=color, line=dict(width=0.5, color='white')),
                    opacity=0.7,
                    text=f'sub-{row["subject_id"]}<br>Age: {row["age"]:.0f}',
                    hoverinfo='text+y',
                    showlegend=False,
                ), row=1, col=col)
            
            # Group means
            for i, (cond_col, cond_label) in enumerate([(sham_col, 'Sham'), (active_col, 'Active')]):
                mean = df_plot[cond_col].mean()
                sem = df_plot[cond_col].sem()
                
                fig.add_trace(go.Scatter(
                    x=[i], y=[mean],
                    mode='markers',
                    marker=dict(size=14, color='#404040', symbol='diamond',
                                line=dict(width=2, color='white')),
                    error_y=dict(type='data', array=[sem], visible=True,
                                color='#404040', thickness=2, width=8),
                    hovertemplate=f'{cond_label}<br>M = {mean:.3f}<br>SEM = {sem:.3f}<extra></extra>',
                    showlegend=False,
                ), row=1, col=col)
            
            # Stats annotation
            diff = df_plot[active_col] - df_plot[sham_col]
            dz = diff.mean() / diff.std() if diff.std() > 0 else 0
            t_stat, p_val = stats.ttest_rel(df_plot[active_col], df_plot[sham_col])
            
            xref = 'x domain' if col == 1 else f'x{col} domain'
            yref = 'y domain' if col == 1 else f'y{col} domain'
            
            fig.add_annotation(
                x=0.95, y=0.95,
                xref=xref, yref=yref,
                text=f'dz = {dz:.2f}, p = {p_val:.3f}<br>N = {len(df_plot)}',
                showarrow=False,
                font=dict(size=12, color='#404040'),
                xanchor='right', yanchor='top',
                bgcolor='rgba(255,255,255,0.8)'
            )
        
        fig.update_xaxes(tickvals=[0, 1], ticktext=['Sham', 'Active'], row=1, col=col)
        fig.update_yaxes(title_text=ylabel, range=ylim, row=1, col=col)
    
    fig.update_layout(
        height=420, width=750,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=60, r=100, t=60, b=50),
        font=dict(family=FONT_FAMILY, size=14),
    )
    
    fig.update_xaxes(showgrid=False, zeroline=False,
                     showline=True, linewidth=1, linecolor='black', tickfont=dict(size=14))
    fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='black')
    
    # Colorbar
    fig.add_trace(go.Scatter(
        x=[None]*50, y=[None]*50, mode='markers',
        marker=dict(size=0.1, color=np.linspace(AGE_MIN, AGE_MAX, 50),
                    colorscale=AGE_COLORSCALE,
                    cmin=AGE_MIN, cmax=AGE_MAX,
                    colorbar=dict(x=1.08, y=0.5, len=0.6, thickness=10,
                                  title='Age', titleside='right', tickfont=dict(size=11))),
        showlegend=False, hoverinfo='skip',
    ))
    
    if show_fig:
        fig.show(config=dict(toImageButtonOptions=dict(filename='h2_1_tacs_vs_sham')))
    
    return fig


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_hypothesis_tests(
    subj_df: pd.DataFrame,
    show_plots: bool = True,
    verbose: bool = True,
    include_theta_exploratory: bool = True
) -> Dict:
    """
    Run all pre-registered hypothesis tests.
    
    Parameters
    ----------
    subj_df : DataFrame
        Subject-level data with all necessary variables
    show_plots : bool
        If True, display visualizations
    verbose : bool
        If True, print summaries
    include_theta_exploratory : bool
        If True, run theta moderation analyses (exploratory)
    
    Returns
    -------
    dict with keys: h1_1, h1_2, h2_1, h2_2, theta_moderation, theta_baseline, and figure objects
    """
    results = {}
    
    # H1.1: Cognitive function → baseline WSLS
    results['h1_1'] = test_h1_1(subj_df, verbose=verbose)
    
    # H1.2: SPSRQ → baseline WSLS
    results['h1_2'] = test_h1_2(subj_df, h1_1_results=results['h1_1'], verbose=verbose)
    
    # Exploratory: Age × H1 predictor interactions
    results['h1_age_interactions'] = test_h1_age_interactions(subj_df, verbose=verbose)
    
    # Johnson-Neyman plots for significant/marginal interactions
    if show_plots and results['h1_age_interactions']:
        jn_figs = {}
        for key, res in results['h1_age_interactions'].items():
            if res is not None and res['interaction_p'] < 0.10:
                # Parse the DV and predictor from the key name
                dv_map = {
                    'cog_x_age_stay': ('sham_p_stay_win', 'global_composite', 'p(stay|win)', 'Global Cognition'),
                    'cog_x_age_shift': ('sham_p_shift_lose', 'global_composite', 'p(shift|lose)', 'Global Cognition'),
                    'sr_x_age_stay': ('sham_p_stay_win', 'spsrq_sr', 'p(stay|win)', 'SPSRQ-SR'),
                    'sp_x_age_shift': ('sham_p_shift_lose', 'spsrq_sp', 'p(shift|lose)', 'SPSRQ-SP'),
                    'cog_x_age_alpha': ('sham_alpha', 'global_composite', 'α (learning rate)', 'Global Cognition'),
                    'cog_x_age_beta': ('sham_beta', 'global_composite', 'β (inv. temperature)', 'Global Cognition'),
                }
                if key in dv_map:
                    dv, pred, dv_label, pred_label = dv_map[key]
                    fig = plot_johnson_neyman(
                        dv=dv, predictor=pred, moderator='age',
                        data=subj_df, covariates=['education_years'],
                        dv_label=dv_label, predictor_label=pred_label,
                        moderator_label='Age',
                        show_fig=True,
                    )
                    if fig is not None:
                        jn_figs[f'jn_{key}'] = fig
        results['jn_figures'] = jn_figs
    
    # H2.1: Paired t-tests
    results['h2_1'] = test_h2_1(subj_df, verbose=verbose)
    
    # H2.2: Age moderation
    results['h2_2'] = test_h2_2(subj_df, verbose=verbose)
    
    # Exploratory: Theta moderation
    if include_theta_exploratory:
        results['theta_moderation'] = test_theta_moderation(subj_df, verbose=verbose)
        results['theta_baseline'] = test_theta_baseline_predictors(subj_df, verbose=verbose)
    
    # Plots
    if show_plots:
        results['h1_1_fig'] = plot_h1_1_scatter(subj_df, show_fig=True)
        results['h2_1_fig'] = plot_h2_1_paired(subj_df, show_fig=True)
    
    return results


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing hypothesis_tests module...")
    print("Functions: test_h1_1, test_h1_2, test_h2_1, test_h2_2")
    print("           paired_ttest_with_tost, run_ols, run_hypothesis_tests")
