"""
rescorla_wagner.py — Rescorla-Wagner reinforcement learning model for tACS Bandit study

Implements a simple Rescorla-Wagner (R-W) model with:
- Learning rule: Q_{c,t+1} = Q_{c,t} + α(r_t - Q_{c,t})
- Choice rule (softmax): p(choice=i) = exp(βQ_i) / Σ_j exp(βQ_j)

Free parameters:
- α (learning rate): 0 < α < 1, higher = stronger recency weighting
- β (inverse temperature): β > 0, higher = more deterministic exploitation

Estimation methods:
- MLE: Maximum Likelihood Estimation
- MAP: Maximum A Posteriori with priors α ~ Beta(2,2), β ~ Gamma(2, scale=5)

References:
- Daw, N. D. (2011). Trial-by-trial data analysis using computational models.
  Decision Making, Affect, and Learning, 3-38.
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from typing import Optional, Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    PLOTLY_TEMPLATE,
    FONT_FAMILY,
    MIN_TRIALS_FOR_FITTING,
)


# =============================================================================
# Core Model Functions
# =============================================================================

def rescorla_wagner_nll(
    params: Tuple[float, float],
    choices: np.ndarray,
    rewards: np.ndarray
) -> float:
    """
    Negative log-likelihood for Rescorla-Wagner model.
    
    Parameters
    ----------
    params : tuple (alpha, beta)
        alpha: learning rate (0, 1)
        beta: inverse temperature (> 0)
    choices : array-like
        Choice sequence (1 or 2)
    rewards : array-like
        Reward sequence (bool or 0/1)
    
    Returns
    -------
    float
        Negative log-likelihood (lower = better fit)
    """
    alpha, beta = params
    Q = np.array([0.5, 0.5])
    nll = 0.0
    
    for choice, reward in zip(choices, rewards):
        if np.isnan(choice) or np.isnan(reward):
            continue
        
        # Numerically stable softmax
        exp_vals = np.exp(beta * (Q - np.max(Q)))
        p = exp_vals / exp_vals.sum()
        
        c = int(choice) - 1  # Convert to 0-indexed
        nll -= np.log(np.clip(p[c], 1e-8, 1.0))
        
        # Update Q-value for chosen option
        r = float(reward)
        Q[c] += alpha * (r - Q[c])
    
    return nll


def rescorla_wagner_nlp(
    params: Tuple[float, float],
    choices: np.ndarray,
    rewards: np.ndarray,
    prior_alpha: Tuple[float, float] = (2, 2),
    prior_beta: Tuple[float, float] = (2, 5)
) -> float:
    """
    Negative log-posterior for Rescorla-Wagner model (MAP estimation).
    
    Adds weakly informative priors:
    - α ~ Beta(2, 2): favors intermediate learning rates
    - β ~ Gamma(2, scale=5): favors moderate inverse temperatures
    
    Parameters
    ----------
    params : tuple (alpha, beta)
    choices, rewards : arrays
    prior_alpha : tuple (a, b) for Beta prior on α
    prior_beta : tuple (shape, scale) for Gamma prior on β
    
    Returns
    -------
    float
        Negative log-posterior
    """
    alpha, beta_param = params
    
    # Parameter constraints
    if alpha <= 0 or alpha >= 1 or beta_param <= 0:
        return 1e9
    
    # Likelihood
    nll = rescorla_wagner_nll(params, choices, rewards)
    
    # Priors
    log_prior_alpha = stats.beta.logpdf(alpha, a=prior_alpha[0], b=prior_alpha[1])
    log_prior_beta = stats.gamma.logpdf(beta_param, a=prior_beta[0], scale=prior_beta[1])
    
    # Negative log-posterior = NLL - log(prior)
    nlp = nll - (log_prior_alpha + log_prior_beta)
    
    if np.isinf(nlp) or np.isnan(nlp):
        return 1e9
    
    return nlp


def fit_rescorla_wagner(
    choices: np.ndarray,
    rewards: np.ndarray,
    method: str = 'mle',
    n_starts: int = 10,
    prior_alpha: Tuple[float, float] = (2, 2),
    prior_beta: Tuple[float, float] = (2, 5)
) -> Dict:
    """
    Fit R-W model using MLE or MAP with multiple random restarts.
    
    Parameters
    ----------
    choices : array
        Choice sequence (1 or 2)
    rewards : array
        Reward sequence (bool or 0/1)
    method : str
        'mle' or 'map'
    n_starts : int
        Number of random initialization attempts
    prior_alpha, prior_beta : tuples
        Prior parameters for MAP estimation
    
    Returns
    -------
    dict with keys:
        alpha, beta, nll, bic, aic, at_boundary, converged, method
    """
    n_trials = len(choices)
    bounds = [(0.001, 0.999), (0.01, 50.0)]
    
    # Select objective function
    if method == 'map':
        obj_func = lambda p, c, r: rescorla_wagner_nlp(p, c, r, prior_alpha, prior_beta)
    else:
        obj_func = rescorla_wagner_nll
    
    best_result = None
    best_obj = np.inf
    
    # Multiple random restarts
    for _ in range(n_starts):
        x0 = [np.random.uniform(0.05, 0.95), np.random.uniform(0.5, 15.0)]
        result = optimize.minimize(
            obj_func, x0, args=(choices, rewards),
            bounds=bounds, method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_result = result
    
    alpha, beta = best_result.x
    
    # Get NLL (even if we fitted with MAP)
    nll = rescorla_wagner_nll([alpha, beta], choices, rewards)
    
    # Information criteria
    k = 2  # number of parameters
    bic = k * np.log(n_trials) + 2 * nll
    aic = 2 * k + 2 * nll
    
    # Check if at boundary
    at_boundary = (alpha <= 0.002 or alpha >= 0.998 or 
                   beta <= 0.02 or beta >= 49.9)
    
    return {
        'alpha': alpha,
        'beta': beta,
        'nll': nll,
        'bic': bic,
        'aic': aic,
        'at_boundary': at_boundary,
        'converged': best_result.success,
        'method': method
    }


# =============================================================================
# Simulation for Parameter Recovery
# =============================================================================

def simulate_rw_agent(
    alpha: float,
    beta: float,
    n_trials: int = 50,
    p_reward: Tuple[float, float] = (0.75, 0.25),
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate choice data from an R-W agent for parameter recovery.
    
    Parameters
    ----------
    alpha : float
        True learning rate
    beta : float
        True inverse temperature
    n_trials : int
        Number of trials to simulate
    p_reward : tuple
        (p_reward_option1, p_reward_option2)
    seed : int, optional
        Random seed
    
    Returns
    -------
    tuple of (choices, rewards)
    """
    if seed is not None:
        np.random.seed(seed)
    
    Q = np.array([0.5, 0.5])
    choices, rewards = [], []
    
    for _ in range(n_trials):
        # Softmax choice
        exp_vals = np.exp(beta * (Q - np.max(Q)))
        p = exp_vals / exp_vals.sum()
        choice = np.random.choice([1, 2], p=p)
        
        # Reward (probabilistic)
        reward = int(np.random.random() < p_reward[choice - 1])
        
        choices.append(choice)
        rewards.append(reward)
        
        # Update Q-value
        Q[choice - 1] += alpha * (reward - Q[choice - 1])
    
    return np.array(choices), np.array(rewards)


def run_parameter_recovery(
    n_simulations: int = 100,
    n_trials: int = 100,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run parameter recovery validation.
    
    Parameters
    ----------
    n_simulations : int
        Number of simulated agents
    n_trials : int
        Trials per agent
    verbose : bool
        If True, print summary
    
    Returns
    -------
    DataFrame with true and estimated parameters
    """
    # Sample true parameters
    true_alphas = np.random.uniform(0.1, 0.9, n_simulations)
    true_betas = np.random.uniform(1.0, 15.0, n_simulations)
    
    recovery_results = {
        'true_alpha': [], 'true_beta': [],
        'est_alpha_mle': [], 'est_beta_mle': [],
        'est_alpha_map': [], 'est_beta_map': []
    }
    
    for i in range(n_simulations):
        choices, rewards = simulate_rw_agent(
            true_alphas[i], true_betas[i],
            n_trials=n_trials, seed=i
        )
        
        fit_mle = fit_rescorla_wagner(choices, rewards, method='mle', n_starts=5)
        fit_map = fit_rescorla_wagner(choices, rewards, method='map', n_starts=5)
        
        recovery_results['true_alpha'].append(true_alphas[i])
        recovery_results['true_beta'].append(true_betas[i])
        recovery_results['est_alpha_mle'].append(fit_mle['alpha'])
        recovery_results['est_beta_mle'].append(fit_mle['beta'])
        recovery_results['est_alpha_map'].append(fit_map['alpha'])
        recovery_results['est_beta_map'].append(fit_map['beta'])
    
    recovery_df = pd.DataFrame(recovery_results)
    
    if verbose:
        corr_alpha_mle = np.corrcoef(recovery_df['true_alpha'], recovery_df['est_alpha_mle'])[0, 1]
        corr_beta_mle = np.corrcoef(recovery_df['true_beta'], recovery_df['est_beta_mle'])[0, 1]
        corr_alpha_map = np.corrcoef(recovery_df['true_alpha'], recovery_df['est_alpha_map'])[0, 1]
        corr_beta_map = np.corrcoef(recovery_df['true_beta'], recovery_df['est_beta_map'])[0, 1]
        
        print('='*70)
        print('Parameter Recovery Validation')
        print('='*70)
        print(f'\nSimulations: {n_simulations}, Trials/sim: {n_trials}')
        print(f'\nRecovery correlations:')
        print(f'  MLE: r(true α, est α) = {corr_alpha_mle:.3f}')
        print(f'       r(true β, est β) = {corr_beta_mle:.3f}')
        print(f'  MAP: r(true α, est α) = {corr_alpha_map:.3f}')
        print(f'       r(true β, est β) = {corr_beta_map:.3f}')
        
        if corr_alpha_mle > 0.7 and corr_beta_map > 0.7:
            print('\n✓ Parameter recovery is adequate.')
        else:
            print('\n⚠ Parameter recovery may be weak.')
    
    return recovery_df


# =============================================================================
# Fitting to Empirical Data
# =============================================================================

def fit_rw_by_condition(
    data: pd.DataFrame,
    conditions: List[str] = ['sham', 'active'],
    method: str = 'mle',
    min_trials: int = MIN_TRIALS_FOR_FITTING,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fit R-W model to each subject × condition.
    
    Parameters
    ----------
    data : DataFrame
        Trial-level data with columns: subject_id, condition, choice, reward
    conditions : list
        Conditions to fit
    method : str
        'mle' or 'map'
    min_trials : int
        Minimum valid trials required
    verbose : bool
        If True, print progress
    
    Returns
    -------
    DataFrame with one row per subject × condition
    """
    fit_data = data[data['condition'].isin(conditions)]
    results = []
    
    for (sub_id, condition), group in fit_data.groupby(['subject_id', 'condition']):
        valid = group.dropna(subset=['choice', 'reward'])
        
        if len(valid) < min_trials:
            if verbose:
                print(f'  sub-{sub_id} {condition}: only {len(valid)} valid trials, skipping')
            continue
        
        choices = valid['choice'].values
        rewards = valid['reward'].values
        
        fit = fit_rescorla_wagner(choices, rewards, method=method, n_starts=10)
        fit['subject_id'] = sub_id
        fit['condition'] = condition
        fit['n_trials'] = len(valid)
        results.append(fit)
    
    return pd.DataFrame(results)


def fit_rw_subject_level(
    data: pd.DataFrame,
    conditions: List[str] = ['sham', 'active'],
    method: str = 'mle',
    min_trials: int = MIN_TRIALS_FOR_FITTING,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fit R-W model per subject (pooling across conditions).
    
    For individual differences analyses (correlations with cognitive function).
    
    Parameters
    ----------
    data : DataFrame
        Trial-level data
    conditions : list
        Conditions to include
    method : str
        'mle' or 'map'
    min_trials : int
        Minimum valid trials required
    verbose : bool
        If True, print progress
    
    Returns
    -------
    DataFrame with one row per subject
    """
    fit_data = data[data['condition'].isin(conditions)]
    results = []
    
    for sub_id in fit_data['subject_id'].unique():
        sub_data = fit_data[fit_data['subject_id'] == sub_id].dropna(subset=['choice', 'reward'])
        
        if len(sub_data) < min_trials:
            continue
        
        choices = sub_data['choice'].values
        rewards = sub_data['reward'].values
        
        fit = fit_rescorla_wagner(choices, rewards, method=method, n_starts=10)
        fit['subject_id'] = sub_id
        fit['n_trials'] = len(sub_data)
        fit['n_runs'] = sub_data['run'].nunique()
        results.append(fit)
    
    return pd.DataFrame(results)


# =============================================================================
# Comparison and Statistics
# =============================================================================

def compare_mle_map(rw_mle: pd.DataFrame, rw_map: pd.DataFrame) -> Dict:
    """
    Compare MLE and MAP estimates.
    
    Returns correlations and mean differences.
    """
    compare = rw_mle.merge(
        rw_map[['subject_id', 'condition', 'alpha', 'beta']],
        on=['subject_id', 'condition'],
        suffixes=('_mle', '_map')
    )
    
    corr_alpha = np.corrcoef(compare['alpha_mle'], compare['alpha_map'])[0, 1]
    corr_beta = np.corrcoef(compare['beta_mle'], compare['beta_map'])[0, 1]
    
    mean_diff_alpha = (compare['alpha_map'] - compare['alpha_mle']).mean()
    mean_diff_beta = (compare['beta_map'] - compare['beta_mle']).mean()
    
    return {
        'corr_alpha': corr_alpha,
        'corr_beta': corr_beta,
        'mean_diff_alpha': mean_diff_alpha,
        'mean_diff_beta': mean_diff_beta,
        'comparison_df': compare
    }


def compute_rw_condition_stats(
    rw: pd.DataFrame,
    h2_subjects: Optional[List[str]] = None
) -> Dict:
    """
    Compute summary statistics and paired tests for R-W parameters.
    
    Parameters
    ----------
    rw : DataFrame
        R-W fits by condition
    h2_subjects : list, optional
        List of H2-eligible subjects for paired tests
    
    Returns
    -------
    dict with summary stats and test results
    """
    results = {}
    
    for param in ['alpha', 'beta']:
        param_results = {}
        
        for cond in ['sham', 'active']:
            cond_data = rw[rw['condition'] == cond][param]
            if len(cond_data) > 0:
                param_results[f'{cond}_mean'] = cond_data.mean()
                param_results[f'{cond}_sd'] = cond_data.std()
                param_results[f'{cond}_n'] = len(cond_data)
        
        # Paired test if H2 subjects provided
        if h2_subjects is not None:
            sham_data = rw[rw['condition'] == 'sham'].set_index('subject_id')
            active_data = rw[rw['condition'] == 'active'].set_index('subject_id')
            paired_subjects = (
                sham_data.index
                .intersection(active_data.index)
                .intersection(h2_subjects)
            )
            
            if len(paired_subjects) >= 3:
                sham_vals = sham_data.loc[paired_subjects, param].values
                active_vals = active_data.loc[paired_subjects, param].values
                
                t_stat, p_val = stats.ttest_rel(active_vals, sham_vals)
                diff = active_vals - sham_vals
                dz = diff.mean() / diff.std() if diff.std() > 0 else 0
                
                param_results['paired_n'] = len(paired_subjects)
                param_results['diff_mean'] = diff.mean()
                param_results['t'] = t_stat
                param_results['p'] = p_val
                param_results['dz'] = dz
        
        results[param] = param_results
    
    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_parameter_recovery(
    recovery_df: pd.DataFrame,
    show_fig: bool = True
) -> go.Figure:
    """Plot parameter recovery scatter plots."""
    
    corr_alpha_mle = np.corrcoef(recovery_df['true_alpha'], recovery_df['est_alpha_mle'])[0, 1]
    corr_beta_mle = np.corrcoef(recovery_df['true_beta'], recovery_df['est_beta_mle'])[0, 1]
    corr_alpha_map = np.corrcoef(recovery_df['true_alpha'], recovery_df['est_alpha_map'])[0, 1]
    corr_beta_map = np.corrcoef(recovery_df['true_beta'], recovery_df['est_beta_map'])[0, 1]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'α Recovery (MLE), r = {corr_alpha_mle:.3f}',
            f'β Recovery (MLE), r = {corr_beta_mle:.3f}',
            f'α Recovery (MAP), r = {corr_alpha_map:.3f}',
            f'β Recovery (MAP), r = {corr_beta_map:.3f}'
        ]
    )
    
    max_beta = max(recovery_df['true_beta'].max(), 
                   recovery_df['est_beta_mle'].max(),
                   recovery_df['est_beta_map'].max()) + 1
    
    # MLE α
    fig.add_trace(go.Scatter(
        x=recovery_df['true_alpha'], y=recovery_df['est_alpha_mle'],
        mode='markers', marker=dict(size=6, opacity=0.6),
        showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        line=dict(dash='dash', color='gray'), showlegend=False
    ), row=1, col=1)
    
    # MLE β
    fig.add_trace(go.Scatter(
        x=recovery_df['true_beta'], y=recovery_df['est_beta_mle'],
        mode='markers', marker=dict(size=6, opacity=0.6),
        showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[0, max_beta], y=[0, max_beta], mode='lines',
        line=dict(dash='dash', color='gray'), showlegend=False
    ), row=1, col=2)
    
    # MAP α
    fig.add_trace(go.Scatter(
        x=recovery_df['true_alpha'], y=recovery_df['est_alpha_map'],
        mode='markers', marker=dict(size=6, opacity=0.6, color='#FF7F0E'),
        showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        line=dict(dash='dash', color='gray'), showlegend=False
    ), row=2, col=1)
    
    # MAP β
    fig.add_trace(go.Scatter(
        x=recovery_df['true_beta'], y=recovery_df['est_beta_map'],
        mode='markers', marker=dict(size=6, opacity=0.6, color='#FF7F0E'),
        showlegend=False
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=[0, max_beta], y=[0, max_beta], mode='lines',
        line=dict(dash='dash', color='gray'), showlegend=False
    ), row=2, col=2)
    
    # Axes
    fig.update_xaxes(range=[0, 1], title_text='True α', row=1, col=1)
    fig.update_yaxes(range=[0, 1], title_text='Est α (MLE)', row=1, col=1)
    fig.update_xaxes(range=[0, max_beta], title_text='True β', row=1, col=2)
    fig.update_yaxes(range=[0, max_beta], title_text='Est β (MLE)', row=1, col=2)
    fig.update_xaxes(range=[0, 1], title_text='True α', row=2, col=1)
    fig.update_yaxes(range=[0, 1], title_text='Est α (MAP)', row=2, col=1)
    fig.update_xaxes(range=[0, max_beta], title_text='True β', row=2, col=2)
    fig.update_yaxes(range=[0, max_beta], title_text='Est β (MAP)', row=2, col=2)
    
    fig.update_layout(
        height=600, width=700,
        title_text='Parameter Recovery Validation',
        template=PLOTLY_TEMPLATE
    )
    
    if show_fig:
        fig.show()
    
    return fig


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_rw_analysis(
    data_clean: pd.DataFrame,
    h2_subjects: Optional[List[str]] = None,
    run_recovery: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run complete Rescorla-Wagner analysis pipeline.
    
    Parameters
    ----------
    data_clean : DataFrame
        Behaviorally cleaned trial-level data
    h2_subjects : list, optional
        H2-eligible subject IDs for paired tests
    run_recovery : bool
        If True, run parameter recovery validation
    verbose : bool
        If True, print summaries
    
    Returns
    -------
    dict with keys:
        'rw_mle': condition-level MLE fits
        'rw_map': condition-level MAP fits
        'rw_subj': subject-level fits
        'condition_stats': summary stats and tests
        'recovery_df': parameter recovery results (if run)
    """
    results = {}
    
    # Parameter recovery (optional)
    if run_recovery:
        recovery_df = run_parameter_recovery(n_simulations=100, n_trials=100, verbose=verbose)
        results['recovery_df'] = recovery_df
    
    # Fit by condition (MLE and MAP)
    if verbose:
        print('\n' + '='*70)
        print('Rescorla-Wagner Model Fitting')
        print('='*70)
    
    rw_mle = fit_rw_by_condition(data_clean, method='mle', verbose=verbose)
    rw_map = fit_rw_by_condition(data_clean, method='map', verbose=False)
    
    results['rw_mle'] = rw_mle
    results['rw_map'] = rw_map
    
    if verbose:
        print(f'\nFitted {len(rw_mle)} subject × condition blocks')
        
        # Boundary warnings
        n_boundary = rw_mle['at_boundary'].sum()
        if n_boundary > 0:
            print(f'\n⚠ WARNING: {n_boundary} fits hit parameter boundaries')
        else:
            print('\n✓ No fits hit parameter boundaries')
    
    # MLE vs MAP comparison
    comparison = compare_mle_map(rw_mle, rw_map)
    results['mle_map_comparison'] = comparison
    
    if verbose:
        print(f'\nMLE vs MAP correlation: α = {comparison["corr_alpha"]:.3f}, β = {comparison["corr_beta"]:.3f}')
    
    # Subject-level fits
    rw_subj = fit_rw_subject_level(data_clean, method='mle', verbose=False)
    results['rw_subj'] = rw_subj
    
    if verbose:
        print(f'\nSubject-level fits: {len(rw_subj)} subjects')
        print(f'  α: M = {rw_subj["alpha"].mean():.3f}, SD = {rw_subj["alpha"].std():.3f}')
        print(f'  β: M = {rw_subj["beta"].mean():.2f}, SD = {rw_subj["beta"].std():.2f}')
    
    # Condition statistics
    condition_stats = compute_rw_condition_stats(rw_mle, h2_subjects)
    results['condition_stats'] = condition_stats
    
    if verbose and h2_subjects is not None:
        print('\n' + '-'*50)
        print('Paired Comparison (H2 subjects)')
        print('-'*50)
        for param in ['alpha', 'beta']:
            res = condition_stats[param]
            if 'p' in res:
                print(f"  {param}: Δ = {res['diff_mean']:+.3f}, dz = {res['dz']:.2f}, "
                      f"t({res['paired_n']-1}) = {res['t']:.2f}, p = {res['p']:.4f}")
    
    return results


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing rescorla_wagner module...")
    print("Functions: fit_rescorla_wagner, simulate_rw_agent, run_parameter_recovery")
    print("           fit_rw_by_condition, fit_rw_subject_level, run_rw_analysis")
