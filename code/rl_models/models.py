"""
Hierarchical Bayesian reinforcement learning models for two-armed bandit data.

Models implemented:
    - RW: Standard Rescorla-Wagner with single learning rate
    - RW_dual: Separate learning rates for positive/negative prediction errors
    - PH: Pearce-Hall hybrid with dynamic associability

All models use softmax action selection and support hierarchical (group-level)
priors over individual-level parameters via a non-centered parameterization
to improve MCMC sampling geometry.

References:
    Rescorla & Wagner (1972). A theory of Pavlovian conditioning.
    Pearce & Hall (1980). A model for Pavlovian learning.
    Daw (2011). Trial-by-trial data analysis using computational models.
    Gershman (2016). Empirical priors for reinforcement learning models.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, Sequence

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


class ModelType(str, Enum):
    RW = "RW"
    RW_DUAL = "RW_dual"
    PH = "PH"


@dataclass(frozen=True)
class ModelSpec:
    """Specification for an RL model variant."""

    name: ModelType
    n_params: int
    param_names: tuple[str, ...]
    description: str


MODEL_SPECS: Dict[ModelType, ModelSpec] = {
    ModelType.RW: ModelSpec(
        name=ModelType.RW,
        n_params=2,
        param_names=("alpha", "beta"),
        description="Standard Rescorla-Wagner: single learning rate + inverse temperature",
    ),
    ModelType.RW_DUAL: ModelSpec(
        name=ModelType.RW_DUAL,
        n_params=3,
        param_names=("alpha_pos", "alpha_neg", "beta"),
        description="Dual learning rate RW: separate rates for positive/negative PEs",
    ),
    ModelType.PH: ModelSpec(
        name=ModelType.PH,
        n_params=4,
        param_names=("alpha_init", "eta", "kappa", "beta"),
        description="Pearce-Hall hybrid: dynamic associability modulates learning rate",
    ),
}


# ---------------------------------------------------------------------------
# Core update functions (pure JAX, no NumPyro dependency)
# ---------------------------------------------------------------------------


def _rw_update(
    V: jnp.ndarray,
    choice: int,
    reward: float,
    alpha: float,
) -> jnp.ndarray:
    """Single Rescorla-Wagner update.

    V: shape (2,) value estimates for each option
    choice: 0 or 1
    reward: 0.0 or 1.0
    alpha: learning rate in (0, 1)
    """
    pe = reward - V[choice]
    V = V.at[choice].add(alpha * pe)
    return V


def _rw_dual_update(
    V: jnp.ndarray,
    choice: int,
    reward: float,
    alpha_pos: float,
    alpha_neg: float,
) -> jnp.ndarray:
    """Dual learning-rate RW update."""
    pe = reward - V[choice]
    alpha = jnp.where(pe >= 0, alpha_pos, alpha_neg)
    V = V.at[choice].add(alpha * pe)
    return V


def _ph_update(
    carry: tuple[jnp.ndarray, float],
    choice: int,
    reward: float,
    eta: float,
    kappa: float,
) -> tuple[jnp.ndarray, float]:
    """Pearce-Hall hybrid update.

    carry = (V, associability)
    eta: weight of |PE| in associability update (0 = pure RW, 1 = pure PH)
    kappa: scaling of associability → effective learning rate
    """
    V, assoc = carry
    pe = reward - V[choice]
    alpha_eff = kappa * assoc
    # Clamp effective alpha to [0, 1]
    alpha_eff = jnp.clip(alpha_eff, 0.0, 1.0)
    V = V.at[choice].add(alpha_eff * pe)
    # Update associability: weighted mix of previous assoc and |PE|
    assoc = eta * jnp.abs(pe) + (1.0 - eta) * assoc
    return (V, assoc)


# ---------------------------------------------------------------------------
# Trial-sequence log-likelihood scanners
# ---------------------------------------------------------------------------


def _softmax_log_prob(V: jnp.ndarray, choice: int, beta: float) -> float:
    """Log probability of choice under softmax with inverse temperature beta."""
    logits = beta * V
    log_probs = logits - jax.nn.logsumexp(logits)
    return log_probs[choice]


def rw_loglik_sequence(
    choices: jnp.ndarray,
    rewards: jnp.ndarray,
    alpha: float,
    beta: float,
) -> float:
    """Compute total log-likelihood for a trial sequence under standard RW."""
    V_init = jnp.array([0.5, 0.5])

    def scan_fn(V, trial):
        c, r = trial
        ll = _softmax_log_prob(V, c, beta)
        V = _rw_update(V, c, r, alpha)
        return V, ll

    _, log_liks = jax.lax.scan(scan_fn, V_init, (choices, rewards))
    return jnp.sum(log_liks)


def rw_dual_loglik_sequence(
    choices: jnp.ndarray,
    rewards: jnp.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    beta: float,
) -> float:
    """Log-likelihood under dual learning rate RW."""
    V_init = jnp.array([0.5, 0.5])

    def scan_fn(V, trial):
        c, r = trial
        ll = _softmax_log_prob(V, c, beta)
        V = _rw_dual_update(V, c, r, alpha_pos, alpha_neg)
        return V, ll

    _, log_liks = jax.lax.scan(scan_fn, V_init, (choices, rewards))
    return jnp.sum(log_liks)


def ph_loglik_sequence(
    choices: jnp.ndarray,
    rewards: jnp.ndarray,
    alpha_init: float,
    eta: float,
    kappa: float,
    beta: float,
) -> float:
    """Log-likelihood under Pearce-Hall hybrid."""
    V_init = jnp.array([0.5, 0.5])
    carry_init = (V_init, alpha_init)

    def scan_fn(carry, trial):
        V, assoc = carry
        c, r = trial
        ll = _softmax_log_prob(V, c, beta)
        carry = _ph_update((V, assoc), c, r, eta, kappa)
        return carry, ll

    _, log_liks = jax.lax.scan(scan_fn, carry_init, (choices, rewards))
    return jnp.sum(log_liks)


# ---------------------------------------------------------------------------
# NumPyro hierarchical models
# ---------------------------------------------------------------------------


def _sample_hierarchical_param(
    name: str,
    n_subjects: int,
    *,
    mu_loc: float = 0.0,
    mu_scale: float = 1.5,
    sigma_scale: float = 1.0,
) -> jnp.ndarray:
    """Non-centered hierarchical parameterization for a parameter.

    Samples group-level mean and std, then individual offsets.
    Returns individual-level parameter values on the unconstrained scale.
    """
    mu = numpyro.sample(f"mu_{name}", dist.Normal(mu_loc, mu_scale))
    sigma = numpyro.sample(f"sigma_{name}", dist.HalfNormal(sigma_scale))
    offset = numpyro.sample(f"offset_{name}", dist.Normal(jnp.zeros(n_subjects), 1.0))
    return mu + sigma * offset


def model_rw_hierarchical(
    choices: jnp.ndarray,
    rewards: jnp.ndarray,
    subject_ids: jnp.ndarray,
    n_subjects: int,
    *,
    condition_ids: Optional[jnp.ndarray] = None,
    n_conditions: int = 1,
) -> None:
    """Hierarchical RW model.

    Parameters
    ----------
    choices : (total_trials,) int array — 0 or 1
    rewards : (total_trials,) float array — 0.0 or 1.0
    subject_ids : (total_trials,) int array — subject index per trial
    n_subjects : int
    condition_ids : optional (total_trials,) int array — condition index per trial
        If provided, group-level means are estimated per condition.
    n_conditions : int — number of conditions (only used if condition_ids provided)
    """
    use_conditions = condition_ids is not None and n_conditions > 1

    if use_conditions:
        # Condition-level means (e.g., active vs. sham vs. baseline)
        mu_alpha_c = numpyro.sample(
            "mu_alpha_cond",
            dist.Normal(jnp.zeros(n_conditions), 1.5),
        )
        mu_beta_c = numpyro.sample(
            "mu_beta_cond",
            dist.Normal(jnp.zeros(n_conditions), 1.5),
        )
        sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(1.0))
        sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(1.0))

        offset_alpha = numpyro.sample(
            "offset_alpha", dist.Normal(jnp.zeros(n_subjects), 1.0)
        )
        offset_beta = numpyro.sample(
            "offset_beta", dist.Normal(jnp.zeros(n_subjects), 1.0)
        )
    else:
        alpha_raw = _sample_hierarchical_param("alpha", n_subjects)
        beta_raw = _sample_hierarchical_param("beta", n_subjects, mu_loc=1.0)

    def subject_loglik(subj_idx):
        mask = subject_ids == subj_idx
        c = jnp.where(mask, choices, 0)
        r = jnp.where(mask, rewards, 0.0)
        n_trials = jnp.sum(mask)

        if use_conditions:
            # For condition-level model, use the most common condition for this
            # subject's run. In practice, each subject-run maps to one condition.
            subj_conds = jnp.where(mask, condition_ids, -1)
            cond = jnp.argmax(jnp.bincount(subj_conds.clip(0), length=n_conditions))
            alpha = jax.nn.sigmoid(mu_alpha_c[cond] + sigma_alpha * offset_alpha[subj_idx])
            beta = jax.nn.softplus(mu_beta_c[cond] + sigma_beta * offset_beta[subj_idx])
        else:
            alpha = jax.nn.sigmoid(alpha_raw[subj_idx])
            beta = jax.nn.softplus(beta_raw[subj_idx])

        ll = rw_loglik_sequence(c, r, alpha, beta)
        # Scale by actual trial count to handle padding
        return ll

    # Vectorize over subjects
    total_ll = jax.vmap(subject_loglik)(jnp.arange(n_subjects))
    numpyro.factor("obs", jnp.sum(total_ll))

    # Store transformed parameters for posterior extraction
    if not use_conditions:
        numpyro.deterministic("alpha", jax.nn.sigmoid(alpha_raw))
        numpyro.deterministic("beta", jax.nn.softplus(beta_raw))


def model_rw_dual_hierarchical(
    choices: jnp.ndarray,
    rewards: jnp.ndarray,
    subject_ids: jnp.ndarray,
    n_subjects: int,
) -> None:
    """Hierarchical dual learning rate RW."""
    alpha_pos_raw = _sample_hierarchical_param("alpha_pos", n_subjects)
    alpha_neg_raw = _sample_hierarchical_param("alpha_neg", n_subjects)
    beta_raw = _sample_hierarchical_param("beta", n_subjects, mu_loc=1.0)

    def subject_loglik(subj_idx):
        mask = subject_ids == subj_idx
        c = jnp.where(mask, choices, 0)
        r = jnp.where(mask, rewards, 0.0)
        alpha_pos = jax.nn.sigmoid(alpha_pos_raw[subj_idx])
        alpha_neg = jax.nn.sigmoid(alpha_neg_raw[subj_idx])
        beta = jax.nn.softplus(beta_raw[subj_idx])
        return rw_dual_loglik_sequence(c, r, alpha_pos, alpha_neg, beta)

    total_ll = jax.vmap(subject_loglik)(jnp.arange(n_subjects))
    numpyro.factor("obs", jnp.sum(total_ll))

    numpyro.deterministic("alpha_pos", jax.nn.sigmoid(alpha_pos_raw))
    numpyro.deterministic("alpha_neg", jax.nn.sigmoid(alpha_neg_raw))
    numpyro.deterministic("beta", jax.nn.softplus(beta_raw))


def model_ph_hierarchical(
    choices: jnp.ndarray,
    rewards: jnp.ndarray,
    subject_ids: jnp.ndarray,
    n_subjects: int,
) -> None:
    """Hierarchical Pearce-Hall hybrid model."""
    alpha_init_raw = _sample_hierarchical_param("alpha_init", n_subjects)
    eta_raw = _sample_hierarchical_param("eta", n_subjects)
    kappa_raw = _sample_hierarchical_param("kappa", n_subjects)
    beta_raw = _sample_hierarchical_param("beta", n_subjects, mu_loc=1.0)

    def subject_loglik(subj_idx):
        mask = subject_ids == subj_idx
        c = jnp.where(mask, choices, 0)
        r = jnp.where(mask, rewards, 0.0)
        alpha_init = jax.nn.sigmoid(alpha_init_raw[subj_idx])
        eta = jax.nn.sigmoid(eta_raw[subj_idx])
        kappa = jax.nn.softplus(kappa_raw[subj_idx])
        beta = jax.nn.softplus(beta_raw[subj_idx])
        return ph_loglik_sequence(c, r, alpha_init, eta, kappa, beta)

    total_ll = jax.vmap(subject_loglik)(jnp.arange(n_subjects))
    numpyro.factor("obs", jnp.sum(total_ll))

    numpyro.deterministic("alpha_init", jax.nn.sigmoid(alpha_init_raw))
    numpyro.deterministic("eta", jax.nn.sigmoid(eta_raw))
    numpyro.deterministic("kappa", jax.nn.softplus(kappa_raw))
    numpyro.deterministic("beta", jax.nn.softplus(beta_raw))


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_FUNCTIONS = {
    ModelType.RW: model_rw_hierarchical,
    ModelType.RW_DUAL: model_rw_dual_hierarchical,
    ModelType.PH: model_ph_hierarchical,
}
