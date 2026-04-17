"""
rescorla_wagner_extended.py — Extended Rescorla-Wagner models for tACS Bandit study

This module implements more sophisticated RL models to test hypotheses about
confirmation bias, perseveration, and value differentiation in aging and tACS.

MODELS IMPLEMENTED:
==================

1. Asymmetric R-W (α⁺, α⁻, β)
   - Separate learning rates for positive (α⁺) and negative (α⁻) prediction errors
   - α⁺/α⁻ ratio > 1 indicates confirmation bias (learning more from rewards than losses)
   - Theoretical basis: Eppinger et al. (2011), Frank & O'Reilly (2006)

2. Asymmetric R-W + Stickiness (α⁺, α⁻, β, τ)
   - Adds choice stickiness parameter (τ) to capture perseveration
   - τ > 0: tendency to repeat previous choice regardless of value
   - τ < 0: tendency to switch (rare)
   - Separates value-driven staying from habitual perseveration

3. Standard R-W + Stickiness (α, β, τ)
   - Single learning rate with stickiness
   - For model comparison to isolate contribution of asymmetry vs stickiness

KEY HYPOTHESES ADDRESSED:
========================

H-RL1: Older adults show elevated α⁺/α⁻ ratios (confirmation bias)
H-RL2: α⁺/α⁻ correlates with WSLS patterns
H-RL5: β declines with age (reduced value differentiation)
H-RL6: Theta-tACS increases β
H-RL7: Older adults show elevated τ (perseveration)
H-RL8: τ correlates with poorer executive function

References:
-----------
- Eppinger, B., Hämmerer, D., & Li, S.-C. (2011). Neuromodulation of reward-based
  learning and decision making in human aging. NYAS.
- Frank, M. J., & O'Reilly, R. C. (2006). A mechanistic account of striatal
  dopamine function in human cognition. Behavioral Neuroscience.
- Katahira, K. (2015). The relation between reinforcement learning parameters
  and the influence of reinforcement history on choice behavior. J Math Psych.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats
from typing import Optional, Dict, List, Tuple, Union
import warnings

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    SUBJECT_INFO,
    PLOTLY_TEMPLATE,
    FONT_FAMILY,
    COLOR_SHAM,
    COLOR_ACTIVE,
    COLOR_GOLD,
    COLOR_GREEN,
    COLOR_RED,
    AGE_MIN,
    AGE_MAX,
    AGE_COLORSCALE,
    MIN_TRIALS_FOR_FITTING,
)


# =============================================================================
# Constants and Priors
# =============================================================================

# Parameter bounds for optimization
PARAM_BOUNDS = {
    'alpha': (0.001, 0.999),      # Learning rate (standard)
    'alpha_pos': (0.001, 0.999),  # Learning rate for positive PE
    'alpha_neg': (0.001, 0.999),  # Learning rate for negative PE
    'beta': (0.1, 20.0),          # Inverse temperature
    'tau': (-3.0, 3.0),           # Choice stickiness (can be negative)
}

# Priors for MAP estimation (parameters of prior distributions)
PRIORS = {
    # Beta(2,2) for learning rates — mild bias toward 0.5
    'alpha': {'dist': 'beta', 'a': 2, 'b': 2},
    'alpha_pos': {'dist': 'beta', 'a': 2, 'b': 2},
    'alpha_neg': {'dist': 'beta', 'a': 2, 'b': 2},
    # Gamma(2,5) for inverse temperature — mode around 0.25, allows up to ~2
    'beta': {'dist': 'gamma', 'shape': 2, 'scale': 2},
    # Normal(0,1) for stickiness — centered at 0 (no stickiness)
    'tau': {'dist': 'normal', 'mean': 0, 'sd': 1},
}


# =============================================================================
# Utility Functions
# =============================================================================

def softmax(q_values: np.ndarray, beta: float) -> np.ndarray:
    """
    Compute softmax choice probabilities.
    
    Parameters
    ----------
    q_values : array
        Q-values for each option
    beta : float
        Inverse temperature (higher = more deterministic)
    
    Returns
    -------
    array : Choice probabilities
    """
    # Numerical stability: subtract max
    q_scaled = beta * (q_values - np.max(q_values))
    exp_q = np.exp(q_scaled)
    return exp_q / np.sum(exp_q)


def softmax_with_stickiness(
    q_values: np.ndarray, 
    beta: float, 
    tau: float,
    prev_choice: Optional[int]
) -> np.ndarray:
    """
    Compute softmax choice probabilities with choice stickiness.
    
    The stickiness term adds a bonus to the previously chosen option,
    capturing perseveration independent of learned values.
    
    P(a) ∝ exp(β * Q(a) + τ * C(a))
    
    where C(a) = 1 if a was chosen on previous trial, 0 otherwise.
    
    Parameters
    ----------
    q_values : array
        Q-values for each option
    beta : float
        Inverse temperature
    tau : float
        Stickiness parameter (positive = perseveration)
    prev_choice : int or None
        Index of previous choice (None for first trial)
    
    Returns
    -------
    array : Choice probabilities
    """
    # Build stickiness vector
    stickiness = np.zeros_like(q_values)
    if prev_choice is not None:
        stickiness[prev_choice] = 1.0
    
    # Combined value = β*Q + τ*C
    combined = beta * q_values + tau * stickiness
    
    # Softmax with numerical stability
    combined = combined - np.max(combined)
    exp_vals = np.exp(combined)
    return exp_vals / np.sum(exp_vals)


def compute_log_prior(params: Dict, param_names: List[str]) -> float:
    """
    Compute log prior probability for MAP estimation.
    
    Parameters
    ----------
    params : dict
        Parameter values
    param_names : list
        Names of parameters to include
    
    Returns
    -------
    float : Log prior probability
    """
    log_prior = 0.0
    
    for name in param_names:
        value = params[name]
        prior_spec = PRIORS.get(name)
        
        if prior_spec is None:
            continue
        
        if prior_spec['dist'] == 'beta':
            # Beta distribution for bounded [0,1] parameters
            a, b = prior_spec['a'], prior_spec['b']
            if 0 < value < 1:
                log_prior += (a - 1) * np.log(value) + (b - 1) * np.log(1 - value)
            else:
                log_prior += -np.inf
                
        elif prior_spec['dist'] == 'gamma':
            # Gamma distribution for positive parameters
            shape, scale = prior_spec['shape'], prior_spec['scale']
            if value > 0:
                log_prior += (shape - 1) * np.log(value) - value / scale
            else:
                log_prior += -np.inf
                
        elif prior_spec['dist'] == 'normal':
            # Normal distribution for unbounded parameters
            mean, sd = prior_spec['mean'], prior_spec['sd']
            log_prior += -0.5 * ((value - mean) / sd) ** 2
    
    return log_prior


# =============================================================================
# Model 1: Asymmetric Rescorla-Wagner (α⁺, α⁻, β)
# =============================================================================

def asymmetric_rw_nll(
    params: np.ndarray,
    choices: np.ndarray,
    rewards: np.ndarray
) -> float:
    """
    Negative log-likelihood for Asymmetric Rescorla-Wagner model.
    
    Value update rule:
        If δ > 0 (positive PE): Q(a) ← Q(a) + α⁺ * δ
        If δ < 0 (negative PE): Q(a) ← Q(a) + α⁻ * δ
    
    Where δ = r - Q(a) is the prediction error.
    
    Parameters
    ----------
    params : array
        [alpha_pos, alpha_neg, beta]
    choices : array
        Sequence of choices (0 or 1)
    rewards : array
        Sequence of rewards (0 or 1)
    
    Returns
    -------
    float : Negative log-likelihood
    """
    alpha_pos, alpha_neg, beta = params
    
    # Initialize Q-values at 0.5 (unbiased)
    q_values = np.array([0.5, 0.5])
    
    nll = 0.0
    
    for choice, reward in zip(choices, rewards):
        choice = int(choice)
        
        # Choice probability
        p_choice = softmax(q_values, beta)
        
        # Avoid log(0)
        p = np.clip(p_choice[choice], 1e-10, 1.0)
        nll -= np.log(p)
        
        # Prediction error
        delta = reward - q_values[choice]
        
        # Asymmetric update: use α⁺ for positive PE, α⁻ for negative PE
        if delta >= 0:
            q_values[choice] += alpha_pos * delta
        else:
            q_values[choice] += alpha_neg * delta
    
    return nll


def asymmetric_rw_nlp(
    params: np.ndarray,
    choices: np.ndarray,
    rewards: np.ndarray
) -> float:
    """
    Negative log-posterior for Asymmetric R-W (for MAP estimation).
    
    NLP = NLL - log(prior)
    """
    alpha_pos, alpha_neg, beta = params
    
    param_dict = {
        'alpha_pos': alpha_pos,
        'alpha_neg': alpha_neg,
        'beta': beta
    }
    
    nll = asymmetric_rw_nll(params, choices, rewards)
    log_prior = compute_log_prior(param_dict, ['alpha_pos', 'alpha_neg', 'beta'])
    
    return nll - log_prior


def fit_asymmetric_rw(
    choices: np.ndarray,
    rewards: np.ndarray,
    method: str = 'mle',
    n_starts: int = 10
) -> Dict:
    """
    Fit Asymmetric Rescorla-Wagner model.
    
    Parameters
    ----------
    choices : array
        Sequence of choices (0 or 1)
    rewards : array
        Sequence of rewards (0 or 1)
    method : str
        'mle' or 'map'
    n_starts : int
        Number of random starting points
    
    Returns
    -------
    dict with keys:
        alpha_pos, alpha_neg, alpha_ratio, beta, nll, bic, aic,
        converged, at_boundary
    """
    bounds = [
        PARAM_BOUNDS['alpha_pos'],
        PARAM_BOUNDS['alpha_neg'],
        PARAM_BOUNDS['beta']
    ]
    
    objective = asymmetric_rw_nlp if method == 'map' else asymmetric_rw_nll
    
    best_result = None
    best_nll = np.inf
    
    for _ in range(n_starts):
        # Random starting point
        x0 = [
            np.random.uniform(0.1, 0.9),   # alpha_pos
            np.random.uniform(0.1, 0.9),   # alpha_neg
            np.random.uniform(0.5, 5.0),   # beta
        ]
        
        try:
            result = optimize.minimize(
                objective,
                x0,
                args=(choices, rewards),
                method='L-BFGS-B',
                bounds=bounds
            )
            
            # Get actual NLL (not NLP) for model comparison
            nll = asymmetric_rw_nll(result.x, choices, rewards)
            
            if nll < best_nll:
                best_nll = nll
                best_result = result
                
        except Exception:
            continue
    
    if best_result is None:
        return {
            'alpha_pos': np.nan, 'alpha_neg': np.nan, 'alpha_ratio': np.nan,
            'beta': np.nan, 'nll': np.nan, 'bic': np.nan, 'aic': np.nan,
            'converged': False, 'at_boundary': True
        }
    
    alpha_pos, alpha_neg, beta = best_result.x
    n_trials = len(choices)
    n_params = 3
    
    # Check if at boundary
    at_boundary = (
        alpha_pos < 0.01 or alpha_pos > 0.99 or
        alpha_neg < 0.01 or alpha_neg > 0.99 or
        beta < 0.2 or beta > 19.0
    )
    
    # Model fit indices
    bic = n_params * np.log(n_trials) + 2 * best_nll
    aic = 2 * n_params + 2 * best_nll
    
    # Compute alpha ratio (confirmation bias index)
    # Ratio > 1 means learning more from positive than negative PE
    alpha_ratio = alpha_pos / alpha_neg if alpha_neg > 0.001 else np.nan
    
    return {
        'alpha_pos': alpha_pos,
        'alpha_neg': alpha_neg,
        'alpha_ratio': alpha_ratio,
        'beta': beta,
        'nll': best_nll,
        'bic': bic,
        'aic': aic,
        'converged': best_result.success,
        'at_boundary': at_boundary,
        'n_trials': n_trials,
        'model': 'asymmetric_rw'
    }


# =============================================================================
# Model 2: Asymmetric R-W + Stickiness (α⁺, α⁻, β, τ)
# =============================================================================

def asymmetric_rw_sticky_nll(
    params: np.ndarray,
    choices: np.ndarray,
    rewards: np.ndarray
) -> float:
    """
    Negative log-likelihood for Asymmetric R-W with choice stickiness.
    
    Combines asymmetric learning with perseveration tendency.
    
    Parameters
    ----------
    params : array
        [alpha_pos, alpha_neg, beta, tau]
    choices : array
        Sequence of choices (0 or 1)
    rewards : array
        Sequence of rewards (0 or 1)
    
    Returns
    -------
    float : Negative log-likelihood
    """
    alpha_pos, alpha_neg, beta, tau = params
    
    q_values = np.array([0.5, 0.5])
    prev_choice = None
    
    nll = 0.0
    
    for choice, reward in zip(choices, rewards):
        choice = int(choice)
        
        # Choice probability with stickiness
        p_choice = softmax_with_stickiness(q_values, beta, tau, prev_choice)
        
        p = np.clip(p_choice[choice], 1e-10, 1.0)
        nll -= np.log(p)
        
        # Prediction error and asymmetric update
        delta = reward - q_values[choice]
        
        if delta >= 0:
            q_values[choice] += alpha_pos * delta
        else:
            q_values[choice] += alpha_neg * delta
        
        prev_choice = choice
    
    return nll


def asymmetric_rw_sticky_nlp(
    params: np.ndarray,
    choices: np.ndarray,
    rewards: np.ndarray
) -> float:
    """Negative log-posterior for Asymmetric R-W + Stickiness."""
    alpha_pos, alpha_neg, beta, tau = params
    
    param_dict = {
        'alpha_pos': alpha_pos,
        'alpha_neg': alpha_neg,
        'beta': beta,
        'tau': tau
    }
    
    nll = asymmetric_rw_sticky_nll(params, choices, rewards)
    log_prior = compute_log_prior(param_dict, ['alpha_pos', 'alpha_neg', 'beta', 'tau'])
    
    return nll - log_prior


def fit_asymmetric_rw_sticky(
    choices: np.ndarray,
    rewards: np.ndarray,
    method: str = 'mle',
    n_starts: int = 10
) -> Dict:
    """
    Fit Asymmetric R-W + Stickiness model.
    
    Parameters
    ----------
    choices : array
        Sequence of choices (0 or 1)
    rewards : array
        Sequence of rewards (0 or 1)
    method : str
        'mle' or 'map'
    n_starts : int
        Number of random starting points
    
    Returns
    -------
    dict with fitted parameters and fit indices
    """
    bounds = [
        PARAM_BOUNDS['alpha_pos'],
        PARAM_BOUNDS['alpha_neg'],
        PARAM_BOUNDS['beta'],
        PARAM_BOUNDS['tau']
    ]
    
    objective = asymmetric_rw_sticky_nlp if method == 'map' else asymmetric_rw_sticky_nll
    
    best_result = None
    best_nll = np.inf
    
    for _ in range(n_starts):
        x0 = [
            np.random.uniform(0.1, 0.9),   # alpha_pos
            np.random.uniform(0.1, 0.9),   # alpha_neg
            np.random.uniform(0.5, 5.0),   # beta
            np.random.uniform(-0.5, 0.5),  # tau
        ]
        
        try:
            result = optimize.minimize(
                objective,
                x0,
                args=(choices, rewards),
                method='L-BFGS-B',
                bounds=bounds
            )
            
            nll = asymmetric_rw_sticky_nll(result.x, choices, rewards)
            
            if nll < best_nll:
                best_nll = nll
                best_result = result
                
        except Exception:
            continue
    
    if best_result is None:
        return {
            'alpha_pos': np.nan, 'alpha_neg': np.nan, 'alpha_ratio': np.nan,
            'beta': np.nan, 'tau': np.nan, 'nll': np.nan, 'bic': np.nan,
            'aic': np.nan, 'converged': False, 'at_boundary': True
        }
    
    alpha_pos, alpha_neg, beta, tau = best_result.x
    n_trials = len(choices)
    n_params = 4
    
    at_boundary = (
        alpha_pos < 0.01 or alpha_pos > 0.99 or
        alpha_neg < 0.01 or alpha_neg > 0.99 or
        beta < 0.2 or beta > 19.0 or
        tau < -2.9 or tau > 2.9
    )
    
    bic = n_params * np.log(n_trials) + 2 * best_nll
    aic = 2 * n_params + 2 * best_nll
    
    alpha_ratio = alpha_pos / alpha_neg if alpha_neg > 0.001 else np.nan
    
    return {
        'alpha_pos': alpha_pos,
        'alpha_neg': alpha_neg,
        'alpha_ratio': alpha_ratio,
        'beta': beta,
        'tau': tau,
        'nll': best_nll,
        'bic': bic,
        'aic': aic,
        'converged': best_result.success,
        'at_boundary': at_boundary,
        'n_trials': n_trials,
        'model': 'asymmetric_rw_sticky'
    }


# =============================================================================
# Model 3: Standard R-W + Stickiness (α, β, τ)
# =============================================================================

def standard_rw_sticky_nll(
    params: np.ndarray,
    choices: np.ndarray,
    rewards: np.ndarray
) -> float:
    """
    Negative log-likelihood for Standard R-W with stickiness.
    
    Single learning rate (symmetric) but with perseveration.
    For model comparison: isolates contribution of stickiness vs asymmetry.
    
    Parameters
    ----------
    params : array
        [alpha, beta, tau]
    choices : array
        Sequence of choices (0 or 1)
    rewards : array
        Sequence of rewards (0 or 1)
    
    Returns
    -------
    float : Negative log-likelihood
    """
    alpha, beta, tau = params
    
    q_values = np.array([0.5, 0.5])
    prev_choice = None
    
    nll = 0.0
    
    for choice, reward in zip(choices, rewards):
        choice = int(choice)
        
        p_choice = softmax_with_stickiness(q_values, beta, tau, prev_choice)
        
        p = np.clip(p_choice[choice], 1e-10, 1.0)
        nll -= np.log(p)
        
        # Standard (symmetric) update
        delta = reward - q_values[choice]
        q_values[choice] += alpha * delta
        
        prev_choice = choice
    
    return nll


def standard_rw_sticky_nlp(
    params: np.ndarray,
    choices: np.ndarray,
    rewards: np.ndarray
) -> float:
    """Negative log-posterior for Standard R-W + Stickiness."""
    alpha, beta, tau = params
    
    param_dict = {'alpha': alpha, 'beta': beta, 'tau': tau}
    
    nll = standard_rw_sticky_nll(params, choices, rewards)
    log_prior = compute_log_prior(param_dict, ['alpha', 'beta', 'tau'])
    
    return nll - log_prior


def fit_standard_rw_sticky(
    choices: np.ndarray,
    rewards: np.ndarray,
    method: str = 'mle',
    n_starts: int = 10
) -> Dict:
    """
    Fit Standard R-W + Stickiness model.
    
    Parameters
    ----------
    choices : array
        Sequence of choices (0 or 1)
    rewards : array
        Sequence of rewards (0 or 1)
    method : str
        'mle' or 'map'
    n_starts : int
        Number of random starting points
    
    Returns
    -------
    dict with fitted parameters and fit indices
    """
    bounds = [
        PARAM_BOUNDS['alpha'],
        PARAM_BOUNDS['beta'],
        PARAM_BOUNDS['tau']
    ]
    
    objective = standard_rw_sticky_nlp if method == 'map' else standard_rw_sticky_nll
    
    best_result = None
    best_nll = np.inf
    
    for _ in range(n_starts):
        x0 = [
            np.random.uniform(0.1, 0.9),   # alpha
            np.random.uniform(0.5, 5.0),   # beta
            np.random.uniform(-0.5, 0.5),  # tau
        ]
        
        try:
            result = optimize.minimize(
                objective,
                x0,
                args=(choices, rewards),
                method='L-BFGS-B',
                bounds=bounds
            )
            
            nll = standard_rw_sticky_nll(result.x, choices, rewards)
            
            if nll < best_nll:
                best_nll = nll
                best_result = result
                
        except Exception:
            continue
    
    if best_result is None:
        return {
            'alpha': np.nan, 'beta': np.nan, 'tau': np.nan,
            'nll': np.nan, 'bic': np.nan, 'aic': np.nan,
            'converged': False, 'at_boundary': True
        }
    
    alpha, beta, tau = best_result.x
    n_trials = len(choices)
    n_params = 3
    
    at_boundary = (
        alpha < 0.01 or alpha > 0.99 or
        beta < 0.2 or beta > 19.0 or
        tau < -2.9 or tau > 2.9
    )
    
    bic = n_params * np.log(n_trials) + 2 * best_nll
    aic = 2 * n_params + 2 * best_nll
    
    return {
        'alpha': alpha,
        'beta': beta,
        'tau': tau,
        'nll': best_nll,
        'bic': bic,
        'aic': aic,
        'converged': best_result.success,
        'at_boundary': at_boundary,
        'n_trials': n_trials,
        'model': 'standard_rw_sticky'
    }


# =============================================================================
# Batch Fitting: All Models by Condition
# =============================================================================

def fit_all_models_by_condition(
    data: pd.DataFrame,
    conditions: List[str] = ['sham', 'active'],
    method: str = 'mle',
    min_trials: int = None,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Fit all extended models for each subject × condition.
    
    Parameters
    ----------
    data : DataFrame
        Trial-level data with subject_id, condition, choice, reward columns
    conditions : list
        Conditions to fit
    method : str
        'mle' or 'map'
    min_trials : int
        Minimum trials required for fitting
    verbose : bool
        Print progress
    
    Returns
    -------
    dict with DataFrames:
        'asymmetric_rw': Asymmetric R-W fits
        'asymmetric_rw_sticky': Asymmetric R-W + Stickiness fits
        'standard_rw_sticky': Standard R-W + Stickiness fits
    """
    if min_trials is None:
        min_trials = MIN_TRIALS_FOR_FITTING
    
    results = {
        'asymmetric_rw': [],
        'asymmetric_rw_sticky': [],
        'standard_rw_sticky': []
    }
    
    subjects = data['subject_id'].unique()
    
    for subj in subjects:
        for cond in conditions:
            subj_cond_data = data[
                (data['subject_id'] == subj) &
                (data['condition'] == cond) &
                (data['choice'].notna())
            ].sort_values('trial_num')
            
            if len(subj_cond_data) < min_trials:
                continue
            
            choices_raw = subj_cond_data['choice'].values
            rewards_raw = subj_cond_data['reward'].values
            
            # Convert choice to 0/1 (original data uses 1/2)
            # Handle both 0/1 and 1/2 coding
            choices_numeric = np.array(choices_raw, dtype=float)
            if np.nanmin(choices_numeric) >= 1:
                # Choices are 1/2 coded, convert to 0/1
                choices = (choices_numeric - 1).astype(int)
            else:
                # Choices are already 0/1
                choices = choices_numeric.astype(int)
            
            # Convert reward to numeric 0/1
            # Handle: boolean objects, strings ('True'/'False'), or numeric
            if rewards_raw.dtype == object:
                rewards = np.array([
                    1 if (r is True or str(r).upper() == 'TRUE') else 0 
                    for r in rewards_raw
                ])
            elif rewards_raw.dtype == bool:
                rewards = rewards_raw.astype(int)
            else:
                rewards = np.array(rewards_raw, dtype=float)
                rewards = np.nan_to_num(rewards, nan=0).astype(int)
            
            # Fit Model 1: Asymmetric R-W
            fit1 = fit_asymmetric_rw(choices, rewards, method=method)
            fit1['subject_id'] = subj
            fit1['condition'] = cond
            results['asymmetric_rw'].append(fit1)
            
            # Fit Model 2: Asymmetric R-W + Stickiness
            fit2 = fit_asymmetric_rw_sticky(choices, rewards, method=method)
            fit2['subject_id'] = subj
            fit2['condition'] = cond
            results['asymmetric_rw_sticky'].append(fit2)
            
            # Fit Model 3: Standard R-W + Stickiness
            fit3 = fit_standard_rw_sticky(choices, rewards, method=method)
            fit3['subject_id'] = subj
            fit3['condition'] = cond
            results['standard_rw_sticky'].append(fit3)
            
            if verbose:
                print(f"  Sub-{subj} {cond}: α⁺={fit1['alpha_pos']:.3f}, "
                      f"α⁻={fit1['alpha_neg']:.3f}, τ={fit2['tau']:.3f}")
    
    # Convert to DataFrames
    for model_name in results:
        results[model_name] = pd.DataFrame(results[model_name])
    
    return results


# =============================================================================
# Model Comparison
# =============================================================================

def compare_models(
    model_fits: Dict[str, pd.DataFrame],
    include_standard_rw: bool = True,
    standard_rw_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compare models using BIC.
    
    Parameters
    ----------
    model_fits : dict
        Output from fit_all_models_by_condition()
    include_standard_rw : bool
        If True, include standard R-W (requires standard_rw_df)
    standard_rw_df : DataFrame, optional
        Standard R-W fits from original rescorla_wagner.py
    
    Returns
    -------
    DataFrame with model comparison statistics
    """
    comparison_rows = []
    
    # Get all subject × condition combinations
    ref_df = model_fits['asymmetric_rw']
    
    for _, row in ref_df.iterrows():
        subj = row['subject_id']
        cond = row['condition']
        
        comp_row = {
            'subject_id': subj,
            'condition': cond,
        }
        
        # Standard R-W (if provided)
        if include_standard_rw and standard_rw_df is not None:
            std_fit = standard_rw_df[
                (standard_rw_df['subject_id'] == subj) &
                (standard_rw_df['condition'] == cond)
            ]
            if len(std_fit) > 0:
                comp_row['bic_standard_rw'] = std_fit.iloc[0]['bic']
                comp_row['nll_standard_rw'] = std_fit.iloc[0]['nll']
        
        # Asymmetric R-W
        asym_fit = model_fits['asymmetric_rw'][
            (model_fits['asymmetric_rw']['subject_id'] == subj) &
            (model_fits['asymmetric_rw']['condition'] == cond)
        ]
        if len(asym_fit) > 0:
            comp_row['bic_asymmetric_rw'] = asym_fit.iloc[0]['bic']
            comp_row['nll_asymmetric_rw'] = asym_fit.iloc[0]['nll']
        
        # Asymmetric R-W + Stickiness
        asym_sticky_fit = model_fits['asymmetric_rw_sticky'][
            (model_fits['asymmetric_rw_sticky']['subject_id'] == subj) &
            (model_fits['asymmetric_rw_sticky']['condition'] == cond)
        ]
        if len(asym_sticky_fit) > 0:
            comp_row['bic_asymmetric_rw_sticky'] = asym_sticky_fit.iloc[0]['bic']
            comp_row['nll_asymmetric_rw_sticky'] = asym_sticky_fit.iloc[0]['nll']
        
        # Standard R-W + Stickiness
        std_sticky_fit = model_fits['standard_rw_sticky'][
            (model_fits['standard_rw_sticky']['subject_id'] == subj) &
            (model_fits['standard_rw_sticky']['condition'] == cond)
        ]
        if len(std_sticky_fit) > 0:
            comp_row['bic_standard_rw_sticky'] = std_sticky_fit.iloc[0]['bic']
            comp_row['nll_standard_rw_sticky'] = std_sticky_fit.iloc[0]['nll']
        
        comparison_rows.append(comp_row)
    
    comparison_df = pd.DataFrame(comparison_rows)
    
    # Determine best model for each subject × condition
    bic_cols = [c for c in comparison_df.columns if c.startswith('bic_')]
    if bic_cols:
        comparison_df['best_model'] = comparison_df[bic_cols].idxmin(axis=1)
        comparison_df['best_model'] = comparison_df['best_model'].str.replace('bic_', '')
    
    return comparison_df


def summarize_model_comparison(comparison_df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    Summarize model comparison results.
    
    Parameters
    ----------
    comparison_df : DataFrame
        Output from compare_models()
    verbose : bool
        Print summary
    
    Returns
    -------
    dict with summary statistics
    """
    bic_cols = [c for c in comparison_df.columns if c.startswith('bic_')]
    
    # Mean BIC by model
    mean_bic = comparison_df[bic_cols].mean()
    
    # Count of best model wins
    if 'best_model' in comparison_df.columns:
        best_counts = comparison_df['best_model'].value_counts()
    else:
        best_counts = pd.Series()
    
    summary = {
        'mean_bic': mean_bic.to_dict(),
        'best_model_counts': best_counts.to_dict(),
        'n_fits': len(comparison_df)
    }
    
    if verbose:
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        print("\nMean BIC by Model (lower = better):")
        for model, bic in sorted(mean_bic.items(), key=lambda x: x[1]):
            model_name = model.replace('bic_', '')
            print(f"  {model_name}: {bic:.1f}")
        
        print(f"\nBest Model Counts (N = {len(comparison_df)}):")
        for model, count in best_counts.items():
            pct = 100 * count / len(comparison_df)
            print(f"  {model}: {count} ({pct:.1f}%)")
    
    return summary


# =============================================================================
# Outlier Diagnostics and Sensitivity Analysis
# =============================================================================

# Thresholds for identifying boundary/outlier estimates
BOUNDARY_THRESHOLD_ALPHA = 0.01  # α < 0.01 or α > 0.99 considered at boundary
BOUNDARY_THRESHOLD_BETA = 0.5    # β < 0.5 or β > 19.5 considered at boundary
BOUNDARY_THRESHOLD_TAU = 2.9     # |τ| > 2.9 considered at boundary
OUTLIER_RATIO_THRESHOLD = 10.0   # α⁺/α⁻ > 10 flagged as extreme


def flag_boundary_estimates(
    model_fits: Dict[str, pd.DataFrame],
    alpha_thresh: float = BOUNDARY_THRESHOLD_ALPHA,
    beta_thresh: float = BOUNDARY_THRESHOLD_BETA,
    tau_thresh: float = BOUNDARY_THRESHOLD_TAU,
    ratio_thresh: float = OUTLIER_RATIO_THRESHOLD,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Flag parameter estimates at or near optimization boundaries.
    
    Boundary estimates may indicate:
    - Poor model fit for that subject/condition
    - Insufficient data to constrain parameters
    - Local minima in optimization
    
    Parameters
    ----------
    model_fits : dict
        Output from fit_all_models_by_condition()
    alpha_thresh : float
        Threshold for flagging α near 0 or 1
    beta_thresh : float
        Threshold for flagging β near bounds
    tau_thresh : float
        Threshold for flagging |τ| near bounds
    ratio_thresh : float
        Threshold for flagging extreme α⁺/α⁻ ratios
    verbose : bool
        Print diagnostic summary
    
    Returns
    -------
    DataFrame with boundary flags for each fit
    """
    fits = model_fits['asymmetric_rw_sticky'].copy()
    
    # Flag each type of boundary issue
    fits['boundary_alpha_pos'] = (fits['alpha_pos'] < alpha_thresh) | (fits['alpha_pos'] > 1 - alpha_thresh)
    fits['boundary_alpha_neg'] = (fits['alpha_neg'] < alpha_thresh) | (fits['alpha_neg'] > 1 - alpha_thresh)
    fits['boundary_beta'] = (fits['beta'] < beta_thresh) | (fits['beta'] > 20 - beta_thresh)
    fits['boundary_tau'] = fits['tau'].abs() > tau_thresh
    fits['extreme_ratio'] = fits['alpha_ratio'] > ratio_thresh
    
    # Any boundary flag
    fits['any_boundary'] = (
        fits['boundary_alpha_pos'] | 
        fits['boundary_alpha_neg'] | 
        fits['boundary_beta'] | 
        fits['boundary_tau']
    )
    
    # Boundary or extreme ratio
    fits['exclude_sensitivity'] = fits['any_boundary'] | fits['extreme_ratio']
    
    if verbose:
        print("\n" + "="*60)
        print("BOUNDARY ESTIMATE DIAGNOSTICS")
        print("="*60)
        
        n_total = len(fits)
        print(f"\nTotal fits: {n_total}")
        print(f"\nBoundary flags (α < {alpha_thresh} or α > {1-alpha_thresh}):")
        print(f"  α⁺ at boundary: {fits['boundary_alpha_pos'].sum()} ({100*fits['boundary_alpha_pos'].mean():.1f}%)")
        print(f"  α⁻ at boundary: {fits['boundary_alpha_neg'].sum()} ({100*fits['boundary_alpha_neg'].mean():.1f}%)")
        print(f"  β at boundary: {fits['boundary_beta'].sum()} ({100*fits['boundary_beta'].mean():.1f}%)")
        print(f"  |τ| > {tau_thresh}: {fits['boundary_tau'].sum()} ({100*fits['boundary_tau'].mean():.1f}%)")
        print(f"\nExtreme α⁺/α⁻ ratio (> {ratio_thresh}): {fits['extreme_ratio'].sum()} ({100*fits['extreme_ratio'].mean():.1f}%)")
        print(f"\nAny boundary issue: {fits['any_boundary'].sum()} ({100*fits['any_boundary'].mean():.1f}%)")
        print(f"Exclude for sensitivity (boundary OR extreme ratio): {fits['exclude_sensitivity'].sum()} ({100*fits['exclude_sensitivity'].mean():.1f}%)")
        
        # Show which subjects are flagged
        flagged = fits[fits['exclude_sensitivity']][['subject_id', 'condition', 'alpha_pos', 'alpha_neg', 'alpha_ratio', 'tau']]
        if len(flagged) > 0:
            print(f"\nFlagged estimates:")
            print(flagged.to_string(index=False))
    
    return fits


def run_sensitivity_analysis(
    model_fits: Dict[str, pd.DataFrame],
    boundary_flags: pd.DataFrame,
    wsls_df: Optional[pd.DataFrame] = None,
    subj_df: Optional[pd.DataFrame] = None,
    verbose: bool = True
) -> Dict:
    """
    Re-run key hypothesis tests excluding boundary/outlier estimates.
    
    Compares results with vs. without problematic fits to assess
    robustness of conclusions.
    
    Parameters
    ----------
    model_fits : dict
        Output from fit_all_models_by_condition()
    boundary_flags : DataFrame
        Output from flag_boundary_estimates()
    wsls_df : DataFrame, optional
        WSLS parameters
    subj_df : DataFrame, optional
        Subject-level data with age, cognitive composites
    verbose : bool
        Print comparison
    
    Returns
    -------
    dict with sensitivity analysis results
    """
    results = {
        'full_sample': {},
        'clean_sample': {},
        'comparison': {}
    }
    
    # Get sham condition fits
    sham_full = boundary_flags[boundary_flags['condition'] == 'sham'].copy()
    sham_clean = sham_full[~sham_full['exclude_sensitivity']].copy()
    
    # Get age lookup
    if subj_df is not None and 'age' in subj_df.columns:
        age_lookup = subj_df.set_index('subject_id')['age'].to_dict()
        sham_full['age'] = sham_full['subject_id'].map(age_lookup)
        sham_clean['age'] = sham_clean['subject_id'].map(age_lookup)
    
    if verbose:
        print("\n" + "="*60)
        print("SENSITIVITY ANALYSIS: WITH vs. WITHOUT BOUNDARY ESTIMATES")
        print("="*60)
        print(f"\nFull sample (sham): N = {len(sham_full)}")
        print(f"Clean sample (sham): N = {len(sham_clean)} (excluded {len(sham_full) - len(sham_clean)})")
    
    # Test H-RL1: Age × α⁺/α⁻ ratio
    if 'age' in sham_full.columns:
        # Full sample
        df_full = sham_full[['age', 'alpha_ratio']].dropna()
        if len(df_full) >= 5:
            r_full, p_full = stats.pearsonr(df_full['age'], df_full['alpha_ratio'])
            results['full_sample']['H_RL1'] = {'r': r_full, 'p': p_full, 'n': len(df_full)}
        
        # Clean sample
        df_clean = sham_clean[['age', 'alpha_ratio']].dropna()
        if len(df_clean) >= 5:
            r_clean, p_clean = stats.pearsonr(df_clean['age'], df_clean['alpha_ratio'])
            results['clean_sample']['H_RL1'] = {'r': r_clean, 'p': p_clean, 'n': len(df_clean)}
        
        if verbose and 'H_RL1' in results['full_sample'] and 'H_RL1' in results['clean_sample']:
            print(f"\nH-RL1: Age × α⁺/α⁻ ratio")
            print(f"  Full:  r = {r_full:.3f}, p = {p_full:.3f}, n = {len(df_full)}")
            print(f"  Clean: r = {r_clean:.3f}, p = {p_clean:.3f}, n = {len(df_clean)}")
            print(f"  Δr = {r_clean - r_full:+.3f}")
    
    # Test H-RL7: Age × τ
    if 'age' in sham_full.columns:
        df_full = sham_full[['age', 'tau']].dropna()
        if len(df_full) >= 5:
            r_full, p_full = stats.pearsonr(df_full['age'], df_full['tau'])
            results['full_sample']['H_RL7'] = {'r': r_full, 'p': p_full, 'n': len(df_full)}
        
        df_clean = sham_clean[['age', 'tau']].dropna()
        if len(df_clean) >= 5:
            r_clean, p_clean = stats.pearsonr(df_clean['age'], df_clean['tau'])
            results['clean_sample']['H_RL7'] = {'r': r_clean, 'p': p_clean, 'n': len(df_clean)}
        
        if verbose and 'H_RL7' in results['full_sample'] and 'H_RL7' in results['clean_sample']:
            print(f"\nH-RL7: Age × τ")
            print(f"  Full:  r = {r_full:.3f}, p = {p_full:.3f}, n = {len(df_full)}")
            print(f"  Clean: r = {r_clean:.3f}, p = {p_clean:.3f}, n = {len(df_clean)}")
            print(f"  Δr = {r_clean - r_full:+.3f}")
    
    # Test H-RL5: Age × β
    if 'age' in sham_full.columns:
        df_full = sham_full[['age', 'beta']].dropna()
        if len(df_full) >= 5:
            r_full, p_full = stats.pearsonr(df_full['age'], df_full['beta'])
            results['full_sample']['H_RL5'] = {'r': r_full, 'p': p_full, 'n': len(df_full)}
        
        df_clean = sham_clean[['age', 'beta']].dropna()
        if len(df_clean) >= 5:
            r_clean, p_clean = stats.pearsonr(df_clean['age'], df_clean['beta'])
            results['clean_sample']['H_RL5'] = {'r': r_clean, 'p': p_clean, 'n': len(df_clean)}
        
        if verbose and 'H_RL5' in results['full_sample'] and 'H_RL5' in results['clean_sample']:
            print(f"\nH-RL5: Age × β")
            print(f"  Full:  r = {r_full:.3f}, p = {p_full:.3f}, n = {len(df_full)}")
            print(f"  Clean: r = {r_clean:.3f}, p = {p_clean:.3f}, n = {len(df_clean)}")
            print(f"  Δr = {r_clean - r_full:+.3f}")
    
    # Test H-RL2: α⁺/α⁻ × WSLS
    if wsls_df is not None:
        wsls_sham = wsls_df[wsls_df['condition'] == 'sham'] if 'condition' in wsls_df.columns else wsls_df
        
        # Full sample
        merged_full = sham_full.merge(
            wsls_sham[['subject_id', 'p_stay_win', 'p_shift_lose']],
            on='subject_id', how='inner'
        )
        
        # Clean sample
        merged_clean = sham_clean.merge(
            wsls_sham[['subject_id', 'p_stay_win', 'p_shift_lose']],
            on='subject_id', how='inner'
        )
        
        if len(merged_full) >= 5:
            r_stay_full, p_stay_full = stats.pearsonr(merged_full['alpha_ratio'], merged_full['p_stay_win'])
            r_shift_full, p_shift_full = stats.pearsonr(merged_full['alpha_ratio'], merged_full['p_shift_lose'])
            results['full_sample']['H_RL2_stay'] = {'r': r_stay_full, 'p': p_stay_full, 'n': len(merged_full)}
            results['full_sample']['H_RL2_shift'] = {'r': r_shift_full, 'p': p_shift_full, 'n': len(merged_full)}
        
        if len(merged_clean) >= 5:
            r_stay_clean, p_stay_clean = stats.pearsonr(merged_clean['alpha_ratio'], merged_clean['p_stay_win'])
            r_shift_clean, p_shift_clean = stats.pearsonr(merged_clean['alpha_ratio'], merged_clean['p_shift_lose'])
            results['clean_sample']['H_RL2_stay'] = {'r': r_stay_clean, 'p': p_stay_clean, 'n': len(merged_clean)}
            results['clean_sample']['H_RL2_shift'] = {'r': r_shift_clean, 'p': p_shift_clean, 'n': len(merged_clean)}
        
        if verbose and 'H_RL2_stay' in results['full_sample'] and 'H_RL2_stay' in results['clean_sample']:
            print(f"\nH-RL2: α⁺/α⁻ × p(stay|win)")
            print(f"  Full:  r = {r_stay_full:.3f}, p = {p_stay_full:.3f}, n = {len(merged_full)}")
            print(f"  Clean: r = {r_stay_clean:.3f}, p = {p_stay_clean:.3f}, n = {len(merged_clean)}")
            print(f"  Δr = {r_stay_clean - r_stay_full:+.3f}")
            
            print(f"\nH-RL2: α⁺/α⁻ × p(shift|lose)")
            print(f"  Full:  r = {r_shift_full:.3f}, p = {p_shift_full:.3f}, n = {len(merged_full)}")
            print(f"  Clean: r = {r_shift_clean:.3f}, p = {p_shift_clean:.3f}, n = {len(merged_clean)}")
            print(f"  Δr = {r_shift_clean - r_shift_full:+.3f}")
    
    # Stimulation effects sensitivity
    if verbose:
        print(f"\n" + "-"*50)
        print("STIMULATION EFFECTS SENSITIVITY")
        print("-"*50)
    
    active_flags = boundary_flags[boundary_flags['condition'] == 'active'].set_index('subject_id')
    sham_flags = boundary_flags[boundary_flags['condition'] == 'sham'].set_index('subject_id')
    
    common_subjects = active_flags.index.intersection(sham_flags.index)
    
    # Identify clean pairs (neither sham nor active flagged)
    clean_pairs = []
    for subj in common_subjects:
        if not sham_flags.loc[subj, 'exclude_sensitivity'] and not active_flags.loc[subj, 'exclude_sensitivity']:
            clean_pairs.append(subj)
    
    if verbose:
        print(f"\nFull paired sample: N = {len(common_subjects)}")
        print(f"Clean paired sample: N = {len(clean_pairs)} (excluded {len(common_subjects) - len(clean_pairs)})")
    
    if len(clean_pairs) >= 3:
        for param, label in [('alpha_pos', 'α⁺'), ('alpha_neg', 'α⁻'), ('alpha_ratio', 'α⁺/α⁻'), ('tau', 'τ')]:
            # Full sample
            sham_full_vals = sham_flags.loc[common_subjects, param]
            active_full_vals = active_flags.loc[common_subjects, param]
            paired_full = pd.DataFrame({'sham': sham_full_vals, 'active': active_full_vals}).dropna()
            
            # Clean sample
            sham_clean_vals = sham_flags.loc[clean_pairs, param]
            active_clean_vals = active_flags.loc[clean_pairs, param]
            paired_clean = pd.DataFrame({'sham': sham_clean_vals, 'active': active_clean_vals}).dropna()
            
            if len(paired_full) >= 3 and len(paired_clean) >= 3:
                t_full, p_full = stats.ttest_rel(paired_full['sham'], paired_full['active'])
                t_clean, p_clean = stats.ttest_rel(paired_clean['sham'], paired_clean['active'])
                
                diff_full = paired_full['active'] - paired_full['sham']
                diff_clean = paired_clean['active'] - paired_clean['sham']
                
                results['full_sample'][f'stim_{param}'] = {'t': t_full, 'p': p_full, 'n': len(paired_full), 'diff': diff_full.mean()}
                results['clean_sample'][f'stim_{param}'] = {'t': t_clean, 'p': p_clean, 'n': len(paired_clean), 'diff': diff_clean.mean()}
                
                if verbose:
                    sig_full = '*' if p_full < 0.05 else ''
                    sig_clean = '*' if p_clean < 0.05 else ''
                    print(f"\n{label}:")
                    print(f"  Full:  Δ = {diff_full.mean():+.3f}, t = {t_full:.2f}, p = {p_full:.3f}{sig_full}")
                    print(f"  Clean: Δ = {diff_clean.mean():+.3f}, t = {t_clean:.2f}, p = {p_clean:.3f}{sig_clean}")
    
    # Summary
    if verbose:
        print(f"\n" + "="*60)
        print("SENSITIVITY SUMMARY")
        print("="*60)
        
        # Check if any conclusions change
        changes = []
        for test in ['H_RL1', 'H_RL5', 'H_RL7', 'H_RL2_stay', 'H_RL2_shift']:
            if test in results['full_sample'] and test in results['clean_sample']:
                p_full = results['full_sample'][test]['p']
                p_clean = results['clean_sample'][test]['p']
                if (p_full < 0.05) != (p_clean < 0.05):
                    changes.append(test)
        
        for param in ['alpha_pos', 'alpha_neg', 'alpha_ratio', 'tau']:
            key = f'stim_{param}'
            if key in results['full_sample'] and key in results['clean_sample']:
                p_full = results['full_sample'][key]['p']
                p_clean = results['clean_sample'][key]['p']
                if (p_full < 0.05) != (p_clean < 0.05):
                    changes.append(key)
        
        if changes:
            print(f"\n⚠ Conclusions CHANGE for: {', '.join(changes)}")
        else:
            print(f"\n✓ Conclusions ROBUST: No significance changes after excluding boundary estimates")
    
    return results


# =============================================================================
# Statistical Tests for Hypotheses
# =============================================================================

def test_extended_hypotheses(
    model_fits: Dict[str, pd.DataFrame],
    wsls_df: Optional[pd.DataFrame] = None,
    subj_df: Optional[pd.DataFrame] = None,
    verbose: bool = True
) -> Dict:
    """
    Test hypotheses about extended RL parameters.
    
    Hypotheses tested:
    - H-RL1: Age × α⁺/α⁻ ratio (sham)
    - H-RL2: α⁺/α⁻ × WSLS correlations
    - H-RL5: Age × β (sham)
    - H-RL7: Age × τ (sham)
    - H-RL8: EF × τ (sham)
    - Stimulation effects on α⁺, α⁻, α⁺/α⁻, β, τ
    
    Parameters
    ----------
    model_fits : dict
        Output from fit_all_models_by_condition()
    wsls_df : DataFrame, optional
        WSLS parameters (for H-RL2)
    subj_df : DataFrame, optional
        Subject-level data with age, cognitive composites
    verbose : bool
        Print results
    
    Returns
    -------
    dict with test results
    """
    results = {}
    
    # Use asymmetric_rw_sticky as primary model (has all parameters)
    fits = model_fits['asymmetric_rw_sticky']
    
    # Get age lookup
    if subj_df is not None and 'age' in subj_df.columns:
        age_lookup = subj_df.set_index('subject_id')['age'].to_dict()
    else:
        age_lookup = {}
    
    # Get sham condition fits
    sham_fits = fits[fits['condition'] == 'sham'].copy()
    if len(age_lookup) > 0:
        sham_fits['age'] = sham_fits['subject_id'].map(age_lookup)
    
    if verbose:
        print("\n" + "="*70)
        print("EXTENDED RL HYPOTHESIS TESTS")
        print("="*70)
    
    # -------------------------------------------------------------------------
    # H-RL1: Age × α⁺/α⁻ ratio (confirmation bias increases with age?)
    # -------------------------------------------------------------------------
    if 'age' in sham_fits.columns:
        df_rl1 = sham_fits[['age', 'alpha_ratio']].dropna()
        
        if len(df_rl1) >= 5:
            r, p = stats.pearsonr(df_rl1['age'], df_rl1['alpha_ratio'])
            results['H_RL1_age_alpha_ratio'] = {'r': r, 'p': p, 'n': len(df_rl1)}
            
            if verbose:
                sig = '*' if p < 0.05 else ''
                print(f"\nH-RL1: Age × α⁺/α⁻ ratio (confirmation bias)")
                print(f"  r = {r:.3f}, p = {p:.3f}{sig}, n = {len(df_rl1)}")
                if r > 0:
                    print(f"  → Older adults show {'higher' if p < 0.05 else 'trending higher'} confirmation bias")
    
    # -------------------------------------------------------------------------
    # H-RL2: α⁺/α⁻ × WSLS correlations
    # -------------------------------------------------------------------------
    if wsls_df is not None:
        # Merge WSLS with sham fits
        wsls_sham = wsls_df[wsls_df['condition'] == 'sham'] if 'condition' in wsls_df.columns else wsls_df
        
        merged = sham_fits.merge(
            wsls_sham[['subject_id', 'p_stay_win', 'p_shift_lose']],
            on='subject_id',
            how='inner'
        )
        
        if len(merged) >= 5:
            # α⁺/α⁻ × p(stay|win)
            r_stay, p_stay = stats.pearsonr(merged['alpha_ratio'], merged['p_stay_win'])
            results['H_RL2_alpha_ratio_vs_stay_win'] = {'r': r_stay, 'p': p_stay, 'n': len(merged)}
            
            # α⁺/α⁻ × p(shift|lose)
            r_shift, p_shift = stats.pearsonr(merged['alpha_ratio'], merged['p_shift_lose'])
            results['H_RL2_alpha_ratio_vs_shift_lose'] = {'r': r_shift, 'p': p_shift, 'n': len(merged)}
            
            if verbose:
                print(f"\nH-RL2: α⁺/α⁻ × WSLS (linking computational to behavioral)")
                sig_stay = '*' if p_stay < 0.05 else ''
                sig_shift = '*' if p_shift < 0.05 else ''
                print(f"  α⁺/α⁻ × p(stay|win): r = {r_stay:.3f}, p = {p_stay:.3f}{sig_stay}")
                print(f"  α⁺/α⁻ × p(shift|lose): r = {r_shift:.3f}, p = {p_shift:.3f}{sig_shift}")
    
    # -------------------------------------------------------------------------
    # H-RL5: Age × β (value differentiation declines with age?)
    # -------------------------------------------------------------------------
    if 'age' in sham_fits.columns:
        df_rl5 = sham_fits[['age', 'beta']].dropna()
        
        if len(df_rl5) >= 5:
            r, p = stats.pearsonr(df_rl5['age'], df_rl5['beta'])
            results['H_RL5_age_beta'] = {'r': r, 'p': p, 'n': len(df_rl5)}
            
            if verbose:
                sig = '*' if p < 0.05 else ''
                print(f"\nH-RL5: Age × β (value differentiation)")
                print(f"  r = {r:.3f}, p = {p:.3f}{sig}, n = {len(df_rl5)}")
    
    # -------------------------------------------------------------------------
    # H-RL7: Age × τ (perseveration increases with age?)
    # -------------------------------------------------------------------------
    if 'age' in sham_fits.columns:
        df_rl7 = sham_fits[['age', 'tau']].dropna()
        
        if len(df_rl7) >= 5:
            r, p = stats.pearsonr(df_rl7['age'], df_rl7['tau'])
            results['H_RL7_age_tau'] = {'r': r, 'p': p, 'n': len(df_rl7)}
            
            if verbose:
                sig = '*' if p < 0.05 else ''
                print(f"\nH-RL7: Age × τ (perseveration)")
                print(f"  r = {r:.3f}, p = {p:.3f}{sig}, n = {len(df_rl7)}")
    
    # -------------------------------------------------------------------------
    # H-RL8: EF × τ (perseveration linked to executive function?)
    # -------------------------------------------------------------------------
    if subj_df is not None and 'ef_composite' in subj_df.columns:
        ef_lookup = subj_df.set_index('subject_id')['ef_composite'].to_dict()
        sham_fits['ef_composite'] = sham_fits['subject_id'].map(ef_lookup)
        
        df_rl8 = sham_fits[['ef_composite', 'tau']].dropna()
        
        if len(df_rl8) >= 5:
            r, p = stats.pearsonr(df_rl8['ef_composite'], df_rl8['tau'])
            results['H_RL8_ef_tau'] = {'r': r, 'p': p, 'n': len(df_rl8)}
            
            if verbose:
                sig = '*' if p < 0.05 else ''
                print(f"\nH-RL8: Executive Function × τ")
                print(f"  r = {r:.3f}, p = {p:.3f}{sig}, n = {len(df_rl8)}")
                if r < 0 and p < 0.05:
                    print(f"  → Lower EF associated with higher perseveration")
    
    # -------------------------------------------------------------------------
    # Stimulation effects: Paired t-tests for each parameter
    # -------------------------------------------------------------------------
    active_fits = fits[fits['condition'] == 'active'].set_index('subject_id')
    sham_fits_indexed = sham_fits.set_index('subject_id')
    
    common_subjects = active_fits.index.intersection(sham_fits_indexed.index)
    
    if len(common_subjects) >= 3:
        if verbose:
            print(f"\n" + "-"*50)
            print(f"STIMULATION EFFECTS (N = {len(common_subjects)})")
            print("-"*50)
        
        params_to_test = [
            ('alpha_pos', 'α⁺'),
            ('alpha_neg', 'α⁻'),
            ('alpha_ratio', 'α⁺/α⁻'),
            ('beta', 'β'),
            ('tau', 'τ')
        ]
        
        for param, label in params_to_test:
            sham_vals = sham_fits_indexed.loc[common_subjects, param]
            active_vals = active_fits.loc[common_subjects, param]
            
            paired_df = pd.DataFrame({
                'sham': sham_vals,
                'active': active_vals
            }).dropna()
            
            if len(paired_df) >= 3:
                t, p = stats.ttest_rel(paired_df['sham'], paired_df['active'])
                diff = paired_df['active'] - paired_df['sham']
                dz = diff.mean() / diff.std() if diff.std() > 0 else 0
                
                results[f'stim_effect_{param}'] = {
                    'sham_mean': paired_df['sham'].mean(),
                    'active_mean': paired_df['active'].mean(),
                    'diff_mean': diff.mean(),
                    't': t,
                    'p': p,
                    'dz': dz,
                    'n': len(paired_df)
                }
                
                if verbose:
                    sig = '*' if p < 0.05 else ''
                    direction = '↑' if diff.mean() > 0 else '↓'
                    print(f"  {label}: Sham={paired_df['sham'].mean():.3f}, "
                          f"Active={paired_df['active'].mean():.3f}, "
                          f"Δ={diff.mean():+.3f} {direction}, "
                          f"t={t:.2f}, p={p:.3f}{sig}")
    
    # -------------------------------------------------------------------------
    # Age moderation of stimulation effects
    # -------------------------------------------------------------------------
    if len(common_subjects) >= 5 and len(age_lookup) > 0:
        if verbose:
            print(f"\n" + "-"*50)
            print("AGE MODERATION OF STIMULATION EFFECTS")
            print("-"*50)
        
        for param, label in [('alpha_ratio', 'α⁺/α⁻'), ('tau', 'τ'), ('beta', 'β')]:
            sham_vals = sham_fits_indexed.loc[common_subjects, param]
            active_vals = active_fits.loc[common_subjects, param]
            delta = active_vals - sham_vals
            
            age_delta_df = pd.DataFrame({
                'age': [age_lookup.get(s) for s in common_subjects],
                'delta': delta.values
            }).dropna()
            
            if len(age_delta_df) >= 5:
                r, p = stats.pearsonr(age_delta_df['age'], age_delta_df['delta'])
                results[f'age_moderation_delta_{param}'] = {'r': r, 'p': p, 'n': len(age_delta_df)}
                
                if verbose:
                    sig = '*' if p < 0.05 else ''
                    print(f"  Age × Δ{label}: r = {r:.3f}, p = {p:.3f}{sig}")
    
    return results


# =============================================================================
# Visualization Functions
# =============================================================================

def _age_to_rgb(age: float) -> str:
    """Convert age to RGB color string (blue → gold gradient)."""
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list('blue_gold', ['#1565C0', '#FFB300'])
    norm = np.clip((age - AGE_MIN) / (AGE_MAX - AGE_MIN), 0, 1)
    rgba = cmap(norm)
    return f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'


def plot_alpha_distribution(
    model_fits: Dict[str, pd.DataFrame],
    condition: str = 'sham',
    show_fig: bool = True
) -> go.Figure:
    """
    Plot α⁺ vs α⁻ distribution with paired lines.
    
    Shows whether α⁺ > α⁻ (confirmation bias) is consistent across subjects.
    
    Parameters
    ----------
    model_fits : dict
        Output from fit_all_models_by_condition()
    condition : str
        Condition to plot
    show_fig : bool
        Display figure
    
    Returns
    -------
    go.Figure
    """
    fits = model_fits['asymmetric_rw_sticky']
    cond_fits = fits[fits['condition'] == condition].copy()
    
    fig = go.Figure()
    
    # Paired lines connecting α⁺ and α⁻ for each subject
    for _, row in cond_fits.iterrows():
        fig.add_trace(go.Scatter(
            x=['α⁺ (pos PE)', 'α⁻ (neg PE)'],
            y=[row['alpha_pos'], row['alpha_neg']],
            mode='lines+markers',
            line=dict(color='rgba(100,100,100,0.3)', width=1),
            marker=dict(size=8),
            showlegend=False,
            hovertemplate=f"Sub-{row['subject_id']}<br>α⁺={row['alpha_pos']:.3f}<br>α⁻={row['alpha_neg']:.3f}"
        ))
    
    # Add box plots
    fig.add_trace(go.Box(
        x=['α⁺ (pos PE)'] * len(cond_fits),
        y=cond_fits['alpha_pos'],
        name='α⁺',
        marker_color=COLOR_GREEN,
        boxpoints=False,
        showlegend=True
    ))
    
    fig.add_trace(go.Box(
        x=['α⁻ (neg PE)'] * len(cond_fits),
        y=cond_fits['alpha_neg'],
        name='α⁻',
        marker_color=COLOR_RED,
        boxpoints=False,
        showlegend=True
    ))
    
    # Stats annotation
    t, p = stats.ttest_rel(cond_fits['alpha_pos'], cond_fits['alpha_neg'])
    mean_ratio = cond_fits['alpha_ratio'].mean()
    
    fig.add_annotation(
        x=0.5, y=0.95, xref='paper', yref='paper',
        text=f"Mean α⁺/α⁻ = {mean_ratio:.2f}<br>Paired t = {t:.2f}, p = {p:.3f}",
        showarrow=False,
        font=dict(size=12),
        bgcolor='rgba(255,255,255,0.8)'
    )
    
    fig.update_layout(
        title=f'Asymmetric Learning Rates ({condition.title()} Condition)<br>'
              f'<sup>Lines connect same subject; α⁺/α⁻ > 1 indicates confirmation bias</sup>',
        yaxis_title='Learning Rate',
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=500,
        width=500,
        showlegend=True
    )
    
    fig.update_yaxes(range=[0, 1], showgrid=False, linecolor='black', linewidth=1)
    fig.update_xaxes(showgrid=False, linecolor='black', linewidth=1)
    
    if show_fig:
        fig.show()
    
    return fig


def plot_age_vs_alpha_ratio(
    model_fits: Dict[str, pd.DataFrame],
    subj_df: pd.DataFrame,
    condition: str = 'sham',
    show_fig: bool = True
) -> go.Figure:
    """
    Scatter plot: Age vs α⁺/α⁻ ratio (H-RL1).
    
    Tests whether confirmation bias increases with age.
    
    Parameters
    ----------
    model_fits : dict
        Output from fit_all_models_by_condition()
    subj_df : DataFrame
        Subject-level data with age
    condition : str
        Condition to plot
    show_fig : bool
        Display figure
    
    Returns
    -------
    go.Figure
    """
    fits = model_fits['asymmetric_rw_sticky']
    cond_fits = fits[fits['condition'] == condition].copy()
    
    # Merge with age
    age_lookup = subj_df.set_index('subject_id')['age'].to_dict()
    cond_fits['age'] = cond_fits['subject_id'].map(age_lookup)
    
    df = cond_fits[['subject_id', 'age', 'alpha_ratio']].dropna()
    
    if len(df) < 3:
        print("Insufficient data for age × α⁺/α⁻ plot")
        return None
    
    # Correlation
    r, p = stats.pearsonr(df['age'], df['alpha_ratio'])
    
    fig = go.Figure()
    
    # Scatter points colored by age
    fig.add_trace(go.Scatter(
        x=df['age'],
        y=df['alpha_ratio'],
        mode='markers',
        marker=dict(
            size=12,
            color=[_age_to_rgb(a) for a in df['age']],
            line=dict(width=1, color='white')
        ),
        text=[f"Sub-{s}<br>Age: {a:.0f}<br>α⁺/α⁻: {r:.2f}" 
              for s, a, r in zip(df['subject_id'], df['age'], df['alpha_ratio'])],
        hoverinfo='text',
        showlegend=False
    ))
    
    # Regression line
    slope, intercept = np.polyfit(df['age'], df['alpha_ratio'], 1)
    x_line = np.linspace(df['age'].min() - 5, df['age'].max() + 5, 100)
    fig.add_trace(go.Scatter(
        x=x_line,
        y=intercept + slope * x_line,
        mode='lines',
        line=dict(color='#404040', width=2, dash='dash'),
        showlegend=False
    ))
    
    # Reference line at ratio = 1 (no bias)
    fig.add_hline(y=1, line_dash='dot', line_color='gray', 
                  annotation_text='No bias (α⁺=α⁻)', annotation_position='right')
    
    fig.update_layout(
        title=f'H-RL1: Age × Confirmation Bias ({condition.title()})<br>'
              f'<sup>r = {r:.3f}, p = {p:.3f}, N = {len(df)}</sup>',
        xaxis_title='Age (years)',
        yaxis_title='α⁺/α⁻ Ratio (confirmation bias index)',
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=500,
        width=600
    )
    
    fig.update_xaxes(showgrid=False, linecolor='black', linewidth=1)
    fig.update_yaxes(showgrid=False, linecolor='black', linewidth=1)
    
    if show_fig:
        fig.show()
    
    return fig


def plot_alpha_ratio_vs_wsls(
    model_fits: Dict[str, pd.DataFrame],
    wsls_df: pd.DataFrame,
    subj_df: pd.DataFrame,
    condition: str = 'sham',
    show_fig: bool = True
) -> go.Figure:
    """
    2-panel scatter: α⁺/α⁻ vs WSLS measures (H-RL2).
    
    Links computational parameters to model-free behavioral signatures.
    
    Parameters
    ----------
    model_fits : dict
        Output from fit_all_models_by_condition()
    wsls_df : DataFrame
        WSLS parameters
    subj_df : DataFrame
        Subject-level data with age
    condition : str
        Condition to plot
    show_fig : bool
        Display figure
    
    Returns
    -------
    go.Figure
    """
    fits = model_fits['asymmetric_rw_sticky']
    cond_fits = fits[fits['condition'] == condition].copy()
    
    # Get WSLS for same condition
    if 'condition' in wsls_df.columns:
        wsls_cond = wsls_df[wsls_df['condition'] == condition]
    else:
        wsls_cond = wsls_df
    
    # Merge
    merged = cond_fits.merge(
        wsls_cond[['subject_id', 'p_stay_win', 'p_shift_lose']],
        on='subject_id'
    )
    
    # Add age
    age_lookup = subj_df.set_index('subject_id')['age'].to_dict()
    merged['age'] = merged['subject_id'].map(age_lookup)
    
    merged = merged.dropna(subset=['alpha_ratio', 'p_stay_win', 'p_shift_lose', 'age'])
    
    if len(merged) < 5:
        print("Insufficient data for α⁺/α⁻ × WSLS plot")
        return None
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['α⁺/α⁻ × p(stay|win)', 'α⁺/α⁻ × p(shift|lose)'],
        horizontal_spacing=0.12
    )
    
    # Panel A: α⁺/α⁻ vs p(stay|win)
    r1, p1 = stats.pearsonr(merged['alpha_ratio'], merged['p_stay_win'])
    
    fig.add_trace(go.Scatter(
        x=merged['alpha_ratio'],
        y=merged['p_stay_win'],
        mode='markers',
        marker=dict(size=11, color=[_age_to_rgb(a) for a in merged['age']],
                    line=dict(width=0.5, color='white')),
        hovertemplate='Sub-%{text}<br>α⁺/α⁻: %{x:.2f}<br>p(stay|win): %{y:.3f}',
        text=merged['subject_id'],
        showlegend=False
    ), row=1, col=1)
    
    # Regression line
    slope, intercept = np.polyfit(merged['alpha_ratio'], merged['p_stay_win'], 1)
    x_line = np.linspace(merged['alpha_ratio'].min(), merged['alpha_ratio'].max(), 100)
    fig.add_trace(go.Scatter(
        x=x_line, y=intercept + slope * x_line,
        mode='lines', line=dict(color='#404040', width=2),
        showlegend=False
    ), row=1, col=1)
    
    # Panel B: α⁺/α⁻ vs p(shift|lose)
    r2, p2 = stats.pearsonr(merged['alpha_ratio'], merged['p_shift_lose'])
    
    fig.add_trace(go.Scatter(
        x=merged['alpha_ratio'],
        y=merged['p_shift_lose'],
        mode='markers',
        marker=dict(size=11, color=[_age_to_rgb(a) for a in merged['age']],
                    line=dict(width=0.5, color='white')),
        hovertemplate='Sub-%{text}<br>α⁺/α⁻: %{x:.2f}<br>p(shift|lose): %{y:.3f}',
        text=merged['subject_id'],
        showlegend=False
    ), row=1, col=2)
    
    slope, intercept = np.polyfit(merged['alpha_ratio'], merged['p_shift_lose'], 1)
    fig.add_trace(go.Scatter(
        x=x_line, y=intercept + slope * x_line,
        mode='lines', line=dict(color='#404040', width=2),
        showlegend=False
    ), row=1, col=2)
    
    # Update subplot titles with stats
    fig.layout.annotations[0].text = f'α⁺/α⁻ × p(stay|win)<br><sup>r = {r1:.2f}, p = {p1:.3f}</sup>'
    fig.layout.annotations[1].text = f'α⁺/α⁻ × p(shift|lose)<br><sup>r = {r2:.2f}, p = {p2:.3f}</sup>'
    
    fig.update_layout(
        title=f'H-RL2: Confirmation Bias × WSLS ({condition.title()})<br>'
              f'<sup>Linking computational parameters to behavioral signatures</sup>',
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=450,
        width=900
    )
    
    fig.update_xaxes(title_text='α⁺/α⁻ Ratio', showgrid=False, linecolor='black', linewidth=1)
    fig.update_yaxes(showgrid=False, linecolor='black', linewidth=1, row=1, col=1)
    fig.update_yaxes(showgrid=False, linecolor='black', linewidth=1, row=1, col=2)
    
    if show_fig:
        fig.show()
    
    return fig


def plot_age_vs_tau(
    model_fits: Dict[str, pd.DataFrame],
    subj_df: pd.DataFrame,
    condition: str = 'sham',
    show_fig: bool = True
) -> go.Figure:
    """
    Scatter plot: Age vs τ (stickiness/perseveration) (H-RL7).
    
    Parameters
    ----------
    model_fits : dict
        Output from fit_all_models_by_condition()
    subj_df : DataFrame
        Subject-level data with age
    condition : str
        Condition to plot
    show_fig : bool
        Display figure
    
    Returns
    -------
    go.Figure
    """
    fits = model_fits['asymmetric_rw_sticky']
    cond_fits = fits[fits['condition'] == condition].copy()
    
    age_lookup = subj_df.set_index('subject_id')['age'].to_dict()
    cond_fits['age'] = cond_fits['subject_id'].map(age_lookup)
    
    df = cond_fits[['subject_id', 'age', 'tau']].dropna()
    
    if len(df) < 3:
        print("Insufficient data for age × τ plot")
        return None
    
    r, p = stats.pearsonr(df['age'], df['tau'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['age'],
        y=df['tau'],
        mode='markers',
        marker=dict(size=12, color=[_age_to_rgb(a) for a in df['age']],
                    line=dict(width=1, color='white')),
        text=[f"Sub-{s}<br>Age: {a:.0f}<br>τ: {t:.3f}" 
              for s, a, t in zip(df['subject_id'], df['age'], df['tau'])],
        hoverinfo='text',
        showlegend=False
    ))
    
    slope, intercept = np.polyfit(df['age'], df['tau'], 1)
    x_line = np.linspace(df['age'].min() - 5, df['age'].max() + 5, 100)
    fig.add_trace(go.Scatter(
        x=x_line, y=intercept + slope * x_line,
        mode='lines', line=dict(color='#404040', width=2, dash='dash'),
        showlegend=False
    ))
    
    # Reference line at τ = 0 (no stickiness)
    fig.add_hline(y=0, line_dash='dot', line_color='gray',
                  annotation_text='No perseveration', annotation_position='right')
    
    fig.update_layout(
        title=f'H-RL7: Age × Choice Stickiness ({condition.title()})<br>'
              f'<sup>r = {r:.3f}, p = {p:.3f}, N = {len(df)}</sup>',
        xaxis_title='Age (years)',
        yaxis_title='τ (choice stickiness)',
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=500,
        width=600
    )
    
    fig.update_xaxes(showgrid=False, linecolor='black', linewidth=1)
    fig.update_yaxes(showgrid=False, linecolor='black', linewidth=1)
    
    if show_fig:
        fig.show()
    
    return fig


def plot_ef_vs_tau(
    model_fits: Dict[str, pd.DataFrame],
    subj_df: pd.DataFrame,
    condition: str = 'sham',
    show_fig: bool = True
) -> go.Figure:
    """
    Scatter plot: Executive Function vs τ (H-RL8).
    
    Parameters
    ----------
    model_fits : dict
        Output from fit_all_models_by_condition()
    subj_df : DataFrame
        Subject-level data with ef_composite and age
    condition : str
        Condition to plot
    show_fig : bool
        Display figure
    
    Returns
    -------
    go.Figure
    """
    if 'ef_composite' not in subj_df.columns:
        print("ef_composite not available in subj_df")
        return None
    
    fits = model_fits['asymmetric_rw_sticky']
    cond_fits = fits[fits['condition'] == condition].copy()
    
    ef_lookup = subj_df.set_index('subject_id')['ef_composite'].to_dict()
    age_lookup = subj_df.set_index('subject_id')['age'].to_dict()
    
    cond_fits['ef_composite'] = cond_fits['subject_id'].map(ef_lookup)
    cond_fits['age'] = cond_fits['subject_id'].map(age_lookup)
    
    df = cond_fits[['subject_id', 'ef_composite', 'tau', 'age']].dropna()
    
    if len(df) < 5:
        print("Insufficient data for EF × τ plot")
        return None
    
    r, p = stats.pearsonr(df['ef_composite'], df['tau'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['ef_composite'],
        y=df['tau'],
        mode='markers',
        marker=dict(size=12, color=[_age_to_rgb(a) for a in df['age']],
                    line=dict(width=1, color='white')),
        text=[f"Sub-{s}<br>EF: {e:.2f}<br>τ: {t:.3f}" 
              for s, e, t in zip(df['subject_id'], df['ef_composite'], df['tau'])],
        hoverinfo='text',
        showlegend=False
    ))
    
    slope, intercept = np.polyfit(df['ef_composite'], df['tau'], 1)
    x_line = np.linspace(df['ef_composite'].min() - 0.5, df['ef_composite'].max() + 0.5, 100)
    fig.add_trace(go.Scatter(
        x=x_line, y=intercept + slope * x_line,
        mode='lines', line=dict(color='#404040', width=2, dash='dash'),
        showlegend=False
    ))
    
    fig.add_hline(y=0, line_dash='dot', line_color='gray')
    
    fig.update_layout(
        title=f'H-RL8: Executive Function × Choice Stickiness ({condition.title()})<br>'
              f'<sup>r = {r:.3f}, p = {p:.3f}, N = {len(df)}</sup>',
        xaxis_title='Executive Function Composite (z)',
        yaxis_title='τ (choice stickiness)',
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=500,
        width=600
    )
    
    fig.update_xaxes(showgrid=False, linecolor='black', linewidth=1)
    fig.update_yaxes(showgrid=False, linecolor='black', linewidth=1)
    
    if show_fig:
        fig.show()
    
    return fig


def plot_stimulation_effects_extended(
    model_fits: Dict[str, pd.DataFrame],
    subj_df: pd.DataFrame,
    show_fig: bool = True
) -> go.Figure:
    """
    3-panel paired spaghetti plot for stimulation effects on α⁻, τ, β.
    
    Parameters
    ----------
    model_fits : dict
        Output from fit_all_models_by_condition()
    subj_df : DataFrame
        Subject-level data with age
    show_fig : bool
        Display figure
    
    Returns
    -------
    go.Figure
    """
    fits = model_fits['asymmetric_rw_sticky']
    
    sham_fits = fits[fits['condition'] == 'sham'].set_index('subject_id')
    active_fits = fits[fits['condition'] == 'active'].set_index('subject_id')
    
    common = sham_fits.index.intersection(active_fits.index)
    
    if len(common) < 3:
        print("Insufficient paired data for stimulation effects plot")
        return None
    
    age_lookup = subj_df.set_index('subject_id')['age'].to_dict()
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['α⁻ (neg PE learning)', 'τ (perseveration)', 'β (value diff.)'],
        horizontal_spacing=0.08
    )
    
    params = [('alpha_neg', 'α⁻'), ('tau', 'τ'), ('beta', 'β')]
    
    for col, (param, label) in enumerate(params, 1):
        sham_vals = sham_fits.loc[common, param]
        active_vals = active_fits.loc[common, param]
        
        # Individual lines
        for subj in common:
            age = age_lookup.get(subj, 50)
            color = _age_to_rgb(age)
            
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[sham_vals[subj], active_vals[subj]],
                mode='lines+markers',
                line=dict(color=color, width=1.5),
                marker=dict(size=8, color=color, line=dict(width=0.5, color='white')),
                opacity=0.7,
                hovertemplate=f'Sub-{subj}<br>Age: {age:.0f}<br>Sham: {sham_vals[subj]:.3f}<br>Active: {active_vals[subj]:.3f}',
                showlegend=False
            ), row=1, col=col)
        
        # Group means
        for i, (vals, cond_label) in enumerate([(sham_vals, 'Sham'), (active_vals, 'Active')]):
            mean = vals.mean()
            sem = vals.sem()
            fig.add_trace(go.Scatter(
                x=[i], y=[mean],
                mode='markers',
                marker=dict(size=14, color='#404040', symbol='diamond',
                            line=dict(width=2, color='white')),
                error_y=dict(type='data', array=[sem], visible=True,
                            color='#404040', thickness=2, width=8),
                showlegend=False
            ), row=1, col=col)
        
        # Stats
        t, p = stats.ttest_rel(sham_vals, active_vals)
        diff = active_vals - sham_vals
        dz = diff.mean() / diff.std() if diff.std() > 0 else 0
        
        fig.add_annotation(
            x=0.5, y=0.95,
            xref=f'x{col} domain' if col > 1 else 'x domain',
            yref=f'y{col} domain' if col > 1 else 'y domain',
            text=f'dz = {dz:.2f}<br>p = {p:.3f}',
            showarrow=False,
            font=dict(size=11),
            bgcolor='rgba(255,255,255,0.8)'
        )
        
        fig.update_xaxes(tickvals=[0, 1], ticktext=['Sham', 'Active'], row=1, col=col)
    
    fig.update_layout(
        title=f'Stimulation Effects on Extended RL Parameters (N = {len(common)})',
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=450,
        width=900
    )
    
    fig.update_xaxes(showgrid=False, linecolor='black', linewidth=1)
    fig.update_yaxes(showgrid=False, linecolor='black', linewidth=1)
    
    if show_fig:
        fig.show()
    
    return fig


def plot_model_comparison_bic(
    comparison_df: pd.DataFrame,
    show_fig: bool = True
) -> go.Figure:
    """
    Bar plot comparing models by mean BIC.
    
    Parameters
    ----------
    comparison_df : DataFrame
        Output from compare_models()
    show_fig : bool
        Display figure
    
    Returns
    -------
    go.Figure
    """
    bic_cols = [c for c in comparison_df.columns if c.startswith('bic_')]
    
    if not bic_cols:
        print("No BIC columns found")
        return None
    
    mean_bic = comparison_df[bic_cols].mean().sort_values()
    model_names = [c.replace('bic_', '').replace('_', ' ').title() for c in mean_bic.index]
    
    # Color: best model in green, others in blue
    colors = [COLOR_GREEN if i == 0 else COLOR_SHAM for i in range(len(model_names))]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=model_names,
        y=mean_bic.values,
        marker_color=colors,
        text=[f'{v:.1f}' for v in mean_bic.values],
        textposition='outside'
    ))
    
    # Best model count
    if 'best_model' in comparison_df.columns:
        counts = comparison_df['best_model'].value_counts()
        count_text = ' | '.join([f'{m}: {c}' for m, c in counts.items()])
    else:
        count_text = ''
    
    fig.update_layout(
        title=f'Model Comparison (Mean BIC, lower = better)<br><sup>Best model counts: {count_text}</sup>',
        xaxis_title='Model',
        yaxis_title='Mean BIC',
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=450,
        width=700
    )
    
    fig.update_xaxes(showgrid=False, linecolor='black', linewidth=1)
    fig.update_yaxes(showgrid=False, linecolor='black', linewidth=1)
    
    if show_fig:
        fig.show()
    
    return fig


def plot_delta_parameter_summary(
    test_results: Dict,
    show_fig: bool = True
) -> go.Figure:
    """
    Bar plot summarizing stimulation effects (Δ) for all parameters.
    
    Parameters
    ----------
    test_results : dict
        Output from test_extended_hypotheses()
    show_fig : bool
        Display figure
    
    Returns
    -------
    go.Figure
    """
    params = [
        ('stim_effect_alpha_pos', 'α⁺'),
        ('stim_effect_alpha_neg', 'α⁻'),
        ('stim_effect_alpha_ratio', 'α⁺/α⁻'),
        ('stim_effect_beta', 'β'),
        ('stim_effect_tau', 'τ')
    ]
    
    labels = []
    deltas = []
    errors = []
    pvals = []
    
    for key, label in params:
        if key in test_results:
            res = test_results[key]
            labels.append(label)
            deltas.append(res['diff_mean'])
            # Approximate SE from t and n
            if res['t'] != 0:
                se = res['diff_mean'] / res['t'] if res['t'] != 0 else 0
            else:
                se = 0
            errors.append(abs(se) * 1.96)  # 95% CI
            pvals.append(res['p'])
    
    if not labels:
        print("No stimulation effect results to plot")
        return None
    
    # Colors: significant in green/red, non-significant in gray
    colors = []
    for d, p in zip(deltas, pvals):
        if p < 0.05:
            colors.append(COLOR_GREEN if d > 0 else COLOR_RED)
        else:
            colors.append('#999999')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=labels,
        y=deltas,
        error_y=dict(type='data', array=errors, visible=True),
        marker_color=colors,
        text=[f'p={p:.3f}' for p in pvals],
        textposition='outside'
    ))
    
    fig.add_hline(y=0, line_color='black', line_width=1)
    
    fig.update_layout(
        title='Stimulation Effects Summary (Active − Sham)<br>'
              '<sup>Error bars: 95% CI; Green/Red: p < .05</sup>',
        xaxis_title='Parameter',
        yaxis_title='Δ (Active − Sham)',
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=450,
        width=600
    )
    
    fig.update_xaxes(showgrid=False, linecolor='black', linewidth=1)
    fig.update_yaxes(showgrid=False, linecolor='black', linewidth=1)
    
    if show_fig:
        fig.show()
    
    return fig


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_extended_rw_analysis(
    data: pd.DataFrame,
    subj_df: pd.DataFrame,
    wsls_df: Optional[pd.DataFrame] = None,
    standard_rw_df: Optional[pd.DataFrame] = None,
    conditions: List[str] = ['sham', 'active'],
    method: str = 'mle',
    show_plots: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run complete extended Rescorla-Wagner analysis pipeline.
    
    Parameters
    ----------
    data : DataFrame
        Trial-level data (cleaned)
    subj_df : DataFrame
        Subject-level data with age, cognitive composites
    wsls_df : DataFrame, optional
        WSLS parameters for H-RL2
    standard_rw_df : DataFrame, optional
        Standard R-W fits for model comparison
    conditions : list
        Conditions to analyze
    method : str
        'mle' or 'map'
    show_plots : bool
        Display visualizations
    verbose : bool
        Print progress and results
    
    Returns
    -------
    dict with keys:
        'model_fits': fitted parameters for each model
        'comparison': model comparison results
        'hypothesis_tests': statistical test results
        'figures': dict of figure objects
    """
    results = {
        'model_fits': {},
        'comparison': None,
        'hypothesis_tests': {},
        'figures': {}
    }
    
    if verbose:
        print("="*70)
        print("EXTENDED RESCORLA-WAGNER ANALYSIS")
        print("="*70)
        print(f"\nModels: Asymmetric R-W, Asymmetric R-W + Stickiness, Standard R-W + Stickiness")
        print(f"Method: {method.upper()}")
        print(f"Conditions: {conditions}")
        print()
    
    # -------------------------------------------------------------------------
    # 1. Fit all models
    # -------------------------------------------------------------------------
    if verbose:
        print("Fitting extended models...")
    
    model_fits = fit_all_models_by_condition(
        data, conditions=conditions, method=method, verbose=verbose
    )
    results['model_fits'] = model_fits
    
    # -------------------------------------------------------------------------
    # 2. Model comparison
    # -------------------------------------------------------------------------
    if verbose:
        print("\nComparing models...")
    
    comparison_df = compare_models(
        model_fits, 
        include_standard_rw=(standard_rw_df is not None),
        standard_rw_df=standard_rw_df
    )
    results['comparison'] = comparison_df
    
    summary = summarize_model_comparison(comparison_df, verbose=verbose)
    results['comparison_summary'] = summary
    
    # -------------------------------------------------------------------------
    # 3. Hypothesis tests
    # -------------------------------------------------------------------------
    test_results = test_extended_hypotheses(
        model_fits, wsls_df=wsls_df, subj_df=subj_df, verbose=verbose
    )
    results['hypothesis_tests'] = test_results
    
    # -------------------------------------------------------------------------
    # 4. Visualizations
    # -------------------------------------------------------------------------
    if show_plots:
        if verbose:
            print("\n" + "="*70)
            print("GENERATING VISUALIZATIONS")
            print("="*70)
        
        # 1. Alpha distribution (α⁺ vs α⁻)
        results['figures']['alpha_distribution'] = plot_alpha_distribution(
            model_fits, condition='sham', show_fig=True
        )
        
        # 2. Age × α⁺/α⁻ (H-RL1)
        results['figures']['age_vs_alpha_ratio'] = plot_age_vs_alpha_ratio(
            model_fits, subj_df, condition='sham', show_fig=True
        )
        
        # 3. α⁺/α⁻ × WSLS (H-RL2)
        if wsls_df is not None:
            results['figures']['alpha_ratio_vs_wsls'] = plot_alpha_ratio_vs_wsls(
                model_fits, wsls_df, subj_df, condition='sham', show_fig=True
            )
        
        # 4. Age × τ (H-RL7)
        results['figures']['age_vs_tau'] = plot_age_vs_tau(
            model_fits, subj_df, condition='sham', show_fig=True
        )
        
        # 5. EF × τ (H-RL8)
        results['figures']['ef_vs_tau'] = plot_ef_vs_tau(
            model_fits, subj_df, condition='sham', show_fig=True
        )
        
        # 6. Stimulation effects (3-panel)
        results['figures']['stimulation_effects'] = plot_stimulation_effects_extended(
            model_fits, subj_df, show_fig=True
        )
        
        # 7. Model comparison BIC
        results['figures']['model_comparison'] = plot_model_comparison_bic(
            comparison_df, show_fig=True
        )
        
        # 8. Delta parameter summary
        results['figures']['delta_summary'] = plot_delta_parameter_summary(
            test_results, show_fig=True
        )
    
    # -------------------------------------------------------------------------
    # 5. Outlier diagnostics and sensitivity analysis
    # -------------------------------------------------------------------------
    if verbose:
        print("\n")
    
    boundary_flags = flag_boundary_estimates(model_fits, verbose=verbose)
    results['boundary_flags'] = boundary_flags
    
    sensitivity_results = run_sensitivity_analysis(
        model_fits, boundary_flags, 
        wsls_df=wsls_df, subj_df=subj_df, 
        verbose=verbose
    )
    results['sensitivity'] = sensitivity_results
    
    if verbose:
        print("\n" + "="*70)
        print("EXTENDED R-W ANALYSIS COMPLETE")
        print("="*70)
    
    return results


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing rescorla_wagner_extended module...")
    print("\nModels available:")
    print("  1. Asymmetric R-W (α⁺, α⁻, β)")
    print("  2. Asymmetric R-W + Stickiness (α⁺, α⁻, β, τ)")
    print("  3. Standard R-W + Stickiness (α, β, τ)")
    print("\nMain functions:")
    print("  - fit_asymmetric_rw()")
    print("  - fit_asymmetric_rw_sticky()")
    print("  - fit_standard_rw_sticky()")
    print("  - fit_all_models_by_condition()")
    print("  - compare_models()")
    print("  - test_extended_hypotheses()")
    print("  - run_extended_rw_analysis()")
