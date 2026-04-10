"""
ddm.py — Drift Diffusion Modeling for tACS Bandit study

Implements DDM analysis using PyDDM to decompose choice behavior into:
- Drift rate (v): Rate of evidence accumulation
- Boundary separation (a): Response caution/threshold  
- Non-decision time (t): Encoding + motor execution time

The primary hypothesis is that theta-tACS affects drift rate during
reward-based decisions.

Note: This module requires PyDDM (pip install pyddm).
Future work will migrate to HSSM for hierarchical Bayesian estimation.

References:
- Shinn, M., et al. (2020). A flexible framework for simulating and
  fitting generalized drift-diffusion models. eLife.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, List, Tuple
import warnings

from config import (
    COLOR_SHAM,
    COLOR_ACTIVE,
    AGE_COLORSCALE,
    AGE_MIN,
    AGE_MAX,
    PLOTLY_TEMPLATE,
    FONT_FAMILY,
)

# Try to import PyDDM
try:
    import pyddm
    from pyddm import Model, Fittable, Sample, LossRobustBIC
    from pyddm.models import (
        DriftConstant, NoiseConstant, BoundConstant,
        OverlayNonDecision, OverlayUniformMixture, OverlayChain
    )
    from pyddm.functions import fit_adjust_model
    PYDDM_AVAILABLE = True
except ImportError:
    PYDDM_AVAILABLE = False


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_ddm_data(
    df: pd.DataFrame,
    min_rt: float = 0.2,
    max_rt: float = 5.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Prepare behavioral data for PyDDM analysis.
    
    Parameters
    ----------
    df : DataFrame
        Trial-level data with rt, choice, current_good, stim_condition
    min_rt : float
        Minimum RT in seconds (exclude anticipatory responses)
    max_rt : float
        Maximum RT in seconds (exclude inattentive responses)
    verbose : bool
        If True, print summary
    
    Returns
    -------
    DataFrame ready for DDM fitting
    """
    ddm_data = df.copy()
    
    # Convert RT from ms to seconds if needed
    if ddm_data['rt'].median() > 100:  # Likely in ms
        ddm_data['rt'] = ddm_data['rt'] / 1000.0
    
    # Code accuracy: 1 = chose high-probability option
    ddm_data['correct'] = (ddm_data['choice'] == ddm_data['current_good']).astype(int)
    
    # Filter valid trials
    valid = (
        ddm_data['rt'].notna() &
        ddm_data['correct'].notna() &
        (ddm_data['rt'] >= min_rt) &
        (ddm_data['rt'] <= max_rt) &
        ddm_data['choice'].notna()
    )
    
    ddm_clean = ddm_data.loc[valid].copy()
    
    if verbose:
        n_excluded = (~valid).sum()
        n_retained = valid.sum()
        
        print('DDM data preparation:')
        print(f'  Retained: {n_retained} trials ({n_retained/len(ddm_data):.1%})')
        print(f'  Excluded: {n_excluded} trials')
        print(f'  Participants: {ddm_clean["subject_id"].nunique()}')
        
        # Condition column name varies
        cond_col = 'stim_condition' if 'stim_condition' in ddm_clean.columns else 'condition'
        if cond_col in ddm_clean.columns:
            print(f'  Conditions: {ddm_clean[cond_col].unique().tolist()}')
    
    return ddm_clean


def create_pyddm_sample(df: pd.DataFrame) -> 'Sample':
    """
    Convert DataFrame to PyDDM Sample object.
    
    Parameters
    ----------
    df : DataFrame
        DDM-prepared data with 'rt' and 'correct' columns
    
    Returns
    -------
    pyddm.Sample object
    """
    if not PYDDM_AVAILABLE:
        raise ImportError("PyDDM is required. Install with: pip install pyddm")
    
    correct_rts = df.loc[df['correct'] == 1, 'rt'].values
    error_rts = df.loc[df['correct'] == 0, 'rt'].values
    
    rts = np.concatenate([correct_rts, error_rts])
    choices = np.concatenate([np.ones(len(correct_rts)), np.zeros(len(error_rts))])
    
    return Sample(rts, choices)


# =============================================================================
# Model Specification
# =============================================================================

def create_ddm_model(name: str = 'DDM') -> 'Model':
    """
    Create a standard DDM model with robust outlier handling.
    
    Parameters
    ----------
    name : str
        Model name
    
    Returns
    -------
    pyddm.Model object
    """
    if not PYDDM_AVAILABLE:
        raise ImportError("PyDDM is required. Install with: pip install pyddm")
    
    model = Model(
        name=name,
        drift=DriftConstant(drift=Fittable(minval=0.1, maxval=5.0)),
        noise=NoiseConstant(noise=1.0),  # Fixed for identifiability
        bound=BoundConstant(B=Fittable(minval=0.3, maxval=3.0)),
        overlay=OverlayChain(overlays=[
            OverlayNonDecision(nondectime=Fittable(minval=0.05, maxval=0.5)),
            OverlayUniformMixture(umixturecoef=0.02),  # 2% contaminants
        ]),
        T_dur=5.0,
    )
    
    return model


def extract_ddm_params(fitted_model: 'Model') -> Dict:
    """Extract parameters from a fitted PyDDM model."""
    params = fitted_model.parameters()
    return {
        'v': float(params['drift']['drift']),
        'a': float(params['bound']['B']),
        't': float(params['overlay']['nondectime']),
    }


# =============================================================================
# Model Fitting
# =============================================================================

def fit_ddm(
    df: pd.DataFrame,
    verbose: bool = False
) -> Dict:
    """
    Fit DDM to a dataset.
    
    Parameters
    ----------
    df : DataFrame
        DDM-prepared data
    verbose : bool
        If True, print fitting info
    
    Returns
    -------
    dict with v, a, t, n_trials, bic, fitted_model
    """
    if not PYDDM_AVAILABLE:
        raise ImportError("PyDDM is required. Install with: pip install pyddm")
    
    # Suppress PyDDM warnings
    warnings.filterwarnings('ignore', message='Setting undecided probability')
    warnings.filterwarnings('ignore', message='This variable')
    
    sample = create_pyddm_sample(df)
    model = create_ddm_model()
    
    fitted = fit_adjust_model(
        sample=sample,
        model=model,
        lossfunction=LossRobustBIC,
        verbose=verbose,
    )
    
    params = extract_ddm_params(fitted)
    params['n_trials'] = len(df)
    params['n_correct'] = len(sample.choice_upper)
    params['n_error'] = len(sample.choice_lower)
    params['bic'] = fitted.get_fit_result().value()
    params['fitted_model'] = fitted
    
    return params


def fit_ddm_by_condition(
    ddm_data: pd.DataFrame,
    conditions: List[str] = ['sham', 'active'],
    min_trials: int = 20,
    verbose: bool = True
) -> Dict:
    """
    Fit DDM separately for each condition.
    
    Parameters
    ----------
    ddm_data : DataFrame
        DDM-prepared data
    conditions : list
        Conditions to fit
    min_trials : int
        Minimum trials required
    verbose : bool
        If True, print progress
    
    Returns
    -------
    dict with condition → parameter dict
    """
    if not PYDDM_AVAILABLE:
        print("PyDDM not available. Install with: pip install pyddm")
        return {}
    
    # Determine condition column
    cond_col = 'stim_condition' if 'stim_condition' in ddm_data.columns else 'condition'
    
    condition_params = {}
    
    for cond in conditions:
        cond_df = ddm_data[ddm_data[cond_col] == cond]
        
        if len(cond_df) < min_trials:
            if verbose:
                print(f'  {cond}: Insufficient trials (n={len(cond_df)}), skipping')
            continue
        
        if verbose:
            print(f'  Fitting {cond} condition (n={len(cond_df)} trials)...')
        
        try:
            params = fit_ddm(cond_df, verbose=False)
            condition_params[cond] = params
            
            if verbose:
                print(f'    v = {params["v"]:.3f}, a = {params["a"]:.3f}, t = {params["t"]:.3f}s')
        except Exception as e:
            if verbose:
                print(f'    Fitting failed: {str(e)[:50]}')
    
    return condition_params


def fit_ddm_by_subject(
    ddm_data: pd.DataFrame,
    conditions: List[str] = ['sham', 'active'],
    min_trials: int = 20,
    min_errors: int = 3,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fit DDM per subject × condition.
    
    Parameters
    ----------
    ddm_data : DataFrame
        DDM-prepared data
    conditions : list
        Conditions to fit
    min_trials : int
        Minimum trials required per condition
    min_errors : int
        Minimum error trials required
    verbose : bool
        If True, print progress
    
    Returns
    -------
    DataFrame with DDM parameters per subject
    """
    if not PYDDM_AVAILABLE:
        print("PyDDM not available. Install with: pip install pyddm")
        return pd.DataFrame()
    
    # Suppress warnings
    warnings.filterwarnings('ignore', message='Setting undecided probability')
    warnings.filterwarnings('ignore', message='This variable')
    warnings.filterwarnings('ignore', message='Infinite likelihood')
    
    cond_col = 'stim_condition' if 'stim_condition' in ddm_data.columns else 'condition'
    
    subjects = sorted(ddm_data['subject_id'].unique())
    results = []
    
    for idx, subj in enumerate(subjects):
        if verbose and (idx + 1) % 5 == 0:
            print(f'  Processing subject {idx+1}/{len(subjects)}')
        
        subj_row = {'subject_id': subj}
        
        for cond in conditions:
            cond_df = ddm_data[
                (ddm_data['subject_id'] == subj) &
                (ddm_data[cond_col] == cond)
            ]
            
            # Check minimum trials
            if len(cond_df) < min_trials:
                subj_row[f'ddm_v_{cond}'] = np.nan
                subj_row[f'ddm_a_{cond}'] = np.nan
                subj_row[f'ddm_t_{cond}'] = np.nan
                subj_row[f'ddm_n_{cond}'] = len(cond_df)
                continue
            
            # Check minimum errors
            n_errors = (cond_df['correct'] == 0).sum()
            if n_errors < min_errors:
                subj_row[f'ddm_v_{cond}'] = np.nan
                subj_row[f'ddm_a_{cond}'] = np.nan
                subj_row[f'ddm_t_{cond}'] = np.nan
                subj_row[f'ddm_n_{cond}'] = len(cond_df)
                continue
            
            try:
                params = fit_ddm(cond_df, verbose=False)
                subj_row[f'ddm_v_{cond}'] = params['v']
                subj_row[f'ddm_a_{cond}'] = params['a']
                subj_row[f'ddm_t_{cond}'] = params['t']
                subj_row[f'ddm_n_{cond}'] = params['n_trials']
            except Exception as e:
                subj_row[f'ddm_v_{cond}'] = np.nan
                subj_row[f'ddm_a_{cond}'] = np.nan
                subj_row[f'ddm_t_{cond}'] = np.nan
                subj_row[f'ddm_n_{cond}'] = len(cond_df)
        
        results.append(subj_row)
    
    return pd.DataFrame(results)


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def bootstrap_ddm_difference(
    ddm_data: pd.DataFrame,
    conditions: Tuple[str, str] = ('sham', 'active'),
    n_bootstrap: int = 200,
    param: str = 'v',
    verbose: bool = True
) -> Dict:
    """
    Bootstrap confidence intervals for DDM parameter differences.
    
    Parameters
    ----------
    ddm_data : DataFrame
        DDM-prepared data
    conditions : tuple
        (condition1, condition2) to compare
    n_bootstrap : int
        Number of bootstrap iterations
    param : str
        Parameter to test ('v', 'a', or 't')
    verbose : bool
        If True, print progress
    
    Returns
    -------
    dict with observed_diff, boot_diffs, ci_low, ci_high, p_value
    """
    if not PYDDM_AVAILABLE:
        raise ImportError("PyDDM is required")
    
    warnings.filterwarnings('ignore')
    
    cond_col = 'stim_condition' if 'stim_condition' in ddm_data.columns else 'condition'
    
    cond1, cond2 = conditions
    data1 = ddm_data[ddm_data[cond_col] == cond1]
    data2 = ddm_data[ddm_data[cond_col] == cond2]
    
    # Observed difference
    if verbose:
        print(f'Fitting observed data...')
    
    params1 = fit_ddm(data1, verbose=False)
    params2 = fit_ddm(data2, verbose=False)
    obs_diff = params2[param] - params1[param]
    
    if verbose:
        print(f'Observed Δ{param} ({cond2} - {cond1}): {obs_diff:+.3f}')
        print(f'Running {n_bootstrap} bootstrap iterations...')
    
    # Bootstrap
    boot_diffs = []
    
    for i in range(n_bootstrap):
        if verbose and (i + 1) % 50 == 0:
            print(f'  Iteration {i+1}/{n_bootstrap}')
        
        try:
            # Resample within each condition
            boot1 = data1.sample(n=len(data1), replace=True).reset_index(drop=True)
            boot2 = data2.sample(n=len(data2), replace=True).reset_index(drop=True)
            
            # Check minimum errors
            if (boot1['correct'] == 0).sum() < 5 or (boot2['correct'] == 0).sum() < 5:
                continue
            
            # Fit (separate model objects per condition)
            p1 = fit_ddm(boot1, verbose=False)
            p2 = fit_ddm(boot2, verbose=False)
            
            boot_diffs.append(p2[param] - p1[param])
        except Exception:
            continue
    
    boot_diffs = np.array(boot_diffs)
    
    # 95% CI
    ci_low = np.percentile(boot_diffs, 2.5)
    ci_high = np.percentile(boot_diffs, 97.5)
    
    # Two-tailed p-value
    if obs_diff >= 0:
        p_value = 2 * np.mean(boot_diffs <= 0)
    else:
        p_value = 2 * np.mean(boot_diffs >= 0)
    p_value = min(p_value, 1.0)
    
    if verbose:
        print(f'\nResults:')
        print(f'  Δ{param} = {obs_diff:+.3f}')
        print(f'  95% CI: [{ci_low:+.3f}, {ci_high:+.3f}]')
        print(f'  p = {p_value:.3f}')
        
        if ci_low <= 0 <= ci_high:
            print(f'  → CI includes zero, no significant difference')
        else:
            direction = 'higher' if obs_diff > 0 else 'lower'
            print(f'  → {cond2} shows significantly {direction} {param}')
    
    return {
        'observed_diff': obs_diff,
        'boot_diffs': boot_diffs,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_value': p_value,
        'n_valid_boots': len(boot_diffs),
    }


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_ddm_analysis(
    data_clean: pd.DataFrame,
    conditions: List[str] = ['sham', 'active'],
    run_bootstrap: bool = False,
    n_bootstrap: int = 200,
    verbose: bool = True
) -> Dict:
    """
    Run complete DDM analysis pipeline.
    
    Parameters
    ----------
    data_clean : DataFrame
        Behaviorally cleaned trial-level data
    conditions : list
        Conditions to analyze
    run_bootstrap : bool
        If True, run bootstrap CI for condition comparison
    n_bootstrap : int
        Number of bootstrap iterations
    verbose : bool
        If True, print summaries
    
    Returns
    -------
    dict with keys:
        'ddm_data': prepared DDM data
        'condition_params': condition-level fits
        'subject_params': subject-level fits
        'bootstrap_results': bootstrap CI results (if run)
        'pyddm_available': bool
    """
    results = {'pyddm_available': PYDDM_AVAILABLE}
    
    if not PYDDM_AVAILABLE:
        print("PyDDM not installed. Install with: pip install pyddm")
        return results
    
    # Prepare data
    if verbose:
        print('='*70)
        print('Drift Diffusion Model Analysis')
        print('='*70)
        print()
    
    ddm_data = prepare_ddm_data(data_clean, verbose=verbose)
    results['ddm_data'] = ddm_data
    
    # RT/accuracy summary
    if verbose:
        print(f'\nRT distribution (seconds):')
        print(f'  Mean: {ddm_data["rt"].mean():.3f}, Median: {ddm_data["rt"].median():.3f}')
        print(f'  SD: {ddm_data["rt"].std():.3f}')
        print(f'\nOverall accuracy: {ddm_data["correct"].mean():.1%}')
    
    # Condition-level fits
    if verbose:
        print('\n' + '-'*50)
        print('Condition-Level DDM Fits')
        print('-'*50)
    
    condition_params = fit_ddm_by_condition(ddm_data, conditions, verbose=verbose)
    results['condition_params'] = condition_params
    
    # Condition comparison
    if 'active' in condition_params and 'sham' in condition_params:
        if verbose:
            print('\n' + '-'*50)
            print('Condition Comparison')
            print('-'*50)
            
            for param in ['v', 'a', 't']:
                diff = condition_params['active'][param] - condition_params['sham'][param]
                print(f"  {param}: sham = {condition_params['sham'][param]:.3f}, "
                      f"active = {condition_params['active'][param]:.3f}, Δ = {diff:+.3f}")
    
    # Subject-level fits
    if verbose:
        print('\n' + '-'*50)
        print('Subject-Level DDM Fits')
        print('-'*50)
    
    subject_params = fit_ddm_by_subject(ddm_data, conditions, verbose=verbose)
    results['subject_params'] = subject_params
    
    # Summary of subject-level
    if len(subject_params) > 0:
        complete_mask = (
            subject_params['ddm_v_sham'].notna() & 
            subject_params['ddm_v_active'].notna()
        )
        n_complete = complete_mask.sum()
        
        if verbose and n_complete > 0:
            print(f'\n  Complete cases (both conditions): {n_complete}')
            
            for param in ['v', 'a', 't']:
                sham_vals = subject_params.loc[complete_mask, f'ddm_{param}_sham']
                active_vals = subject_params.loc[complete_mask, f'ddm_{param}_active']
                
                t_stat, p_val = stats.ttest_rel(active_vals, sham_vals)
                diff = active_vals - sham_vals
                dz = diff.mean() / diff.std() if diff.std() > 0 else 0
                
                print(f'\n  {param.upper()}: Δ = {diff.mean():+.3f}, dz = {dz:.2f}, '
                      f't = {t_stat:.2f}, p = {p_val:.3f}')
    
    # Bootstrap CI
    if run_bootstrap and 'active' in condition_params and 'sham' in condition_params:
        if verbose:
            print('\n' + '='*70)
            print('Bootstrap Test: Active vs Sham Drift Rate')
            print('='*70)
        
        boot_results = bootstrap_ddm_difference(
            ddm_data,
            conditions=('sham', 'active'),
            n_bootstrap=n_bootstrap,
            param='v',
            verbose=verbose
        )
        results['bootstrap_results'] = boot_results
    
    return results


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing ddm module...")
    print(f"PyDDM available: {PYDDM_AVAILABLE}")
    if PYDDM_AVAILABLE:
        print(f"PyDDM version: {pyddm.__version__}")
    print("\nFunctions: prepare_ddm_data, fit_ddm, fit_ddm_by_condition,")
    print("           fit_ddm_by_subject, bootstrap_ddm_difference, run_ddm_analysis")
