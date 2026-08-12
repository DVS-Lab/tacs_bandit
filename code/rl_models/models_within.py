"""
models_within.py — Hierarchical RW with condition varying within subject

The model in models.py takes one condition per subject, so it can only express
a between-subject manipulation. This design is within-subject: every subject
contributes both sham and active runs, and the estimand is the difference
between them.

Parameterization, per run s belonging to subject j:

    alpha_s = sigmoid( mu_a + delta_a * x_cond(s) + eta_a * x_block(s)
                       + sigma_a * z_a[j] + tau_a * w_a[j] * x_cond(s) )

with the same form for beta under a softplus link. Contrasts are centered at
+/-0.5, so mu is the grand mean and delta is the full sham-to-active step.

What each term buys:

  delta_a  group-level tACS effect on learning rate — the paper's estimand,
           with a posterior rather than a t-test on two sets of point estimates
  tau_a    between-subject spread in that effect: how much people differ in
           responsiveness. Given the age-moderation story, arguably the more
           interesting parameter of the two
  eta_a    block-order effect. Within a subject, condition is perfectly
           confounded with early-vs-late in a two-hour session; counterbalance
           breaks that across the group but only if the model is told about it.
           Without eta, practice and fatigue drift can load onto delta
  sigma_a  ordinary between-subject variation in the learning rate itself

Moderators are optional and enter on the condition effect:

    delta_a[j] = delta_a + moderators[j] @ gamma_a

so a moderated analysis is a data change rather than a rewrite, and the
uncertainty in delta propagates into the moderation estimate instead of being
discarded by regressing on point estimates.

Non-centered throughout: subject effects are sampled standard normal and
scaled, which keeps the geometry well conditioned when a variance component is
small — the usual funnel that produces divergences otherwise.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .models import rw_loglik_sequence


# Default prior scales. Kept here rather than inline so a sensitivity analysis
# is an argument change instead of an edit to the model body.
#
# These are weakly informative on the link scale: mu on the logit/softplus
# scale, delta as a plausible condition shift, sigma and tau as between-subject
# spreads. If a posterior for one of the variance terms sits far into its
# prior's tail, the prior is constraining the estimate downward and the run
# should be repeated with a wider scale — see test_prior_sensitivity.py.
DEFAULT_PRIORS = {
    'mu_alpha': 1.5,
    'mu_beta': 1.5,
    'delta': 0.5,     # condition effects (alpha and beta)
    'eta': 0.5,       # block-order effects
    'sigma': 1.0,     # between-subject spread in the parameter
    'tau': 0.5,       # between-subject spread in the condition effect
    'gamma': 0.5,     # moderator coefficients
}


def model_rw_within_subject(
    choices: jnp.ndarray,
    rewards: jnp.ndarray,
    masks: jnp.ndarray,
    seq_subject: jnp.ndarray,
    cond_x: jnp.ndarray,
    block_x: jnp.ndarray,
    n_subjects: int,
    *,
    moderators: Optional[jnp.ndarray] = None,
    include_block: bool = True,
    include_random_slope: bool = True,
    priors: Optional[dict] = None,
) -> None:
    """
    Hierarchical Rescorla-Wagner with a within-subject condition effect.

    Parameters
    ----------
    choices, rewards, masks : (n_sequences, max_trials)
        One row per run. Q-values reset at the start of each row, which is
        correct here — every run is a fresh bandit with new contingencies.
    seq_subject : (n_sequences,) int
        Which subject each run belongs to.
    cond_x : (n_sequences,) float
        Centered condition contrast, -0.5 sham / +0.5 active.
    block_x : (n_sequences,) float
        Centered block contrast, -0.5 first / +0.5 second.
    moderators : optional (n_subjects, n_moderators) float
        Standardized subject-level moderators (age, iTF distance, ...). When
        given, each one gets a coefficient on the condition effect.
    include_block : set False only to check how much the block term matters.
    include_random_slope : set False for a fixed condition effect across
        subjects; mainly useful as a simpler comparison model.
    """
    p = dict(DEFAULT_PRIORS)
    if priors:
        p.update(priors)

    # --- Group means ---
    mu_alpha = numpyro.sample('mu_alpha', dist.Normal(0.0, p['mu_alpha']))
    mu_beta = numpyro.sample('mu_beta', dist.Normal(1.0, p['mu_beta']))

    # --- Condition effects (the estimands) ---
    delta_alpha = numpyro.sample('delta_alpha', dist.Normal(0.0, p['delta']))
    delta_beta = numpyro.sample('delta_beta', dist.Normal(0.0, p['delta']))

    # --- Between-subject variation in the parameters themselves ---
    sigma_alpha = numpyro.sample('sigma_alpha', dist.HalfNormal(p['sigma']))
    sigma_beta = numpyro.sample('sigma_beta', dist.HalfNormal(p['sigma']))
    z_alpha = numpyro.sample('z_alpha', dist.Normal(jnp.zeros(n_subjects), 1.0))
    z_beta = numpyro.sample('z_beta', dist.Normal(jnp.zeros(n_subjects), 1.0))

    # --- Subject-level condition effect ---
    delta_alpha_subj = delta_alpha
    delta_beta_subj = delta_beta

    if moderators is not None:
        n_mod = moderators.shape[1]
        gamma_alpha = numpyro.sample(
            'gamma_alpha', dist.Normal(jnp.zeros(n_mod), p['gamma'])
        )
        gamma_beta = numpyro.sample(
            'gamma_beta', dist.Normal(jnp.zeros(n_mod), p['gamma'])
        )
        delta_alpha_subj = delta_alpha + moderators @ gamma_alpha
        delta_beta_subj = delta_beta + moderators @ gamma_beta

    if include_random_slope:
        tau_alpha = numpyro.sample('tau_alpha', dist.HalfNormal(p['tau']))
        tau_beta = numpyro.sample('tau_beta', dist.HalfNormal(p['tau']))
        w_alpha = numpyro.sample('w_alpha', dist.Normal(jnp.zeros(n_subjects), 1.0))
        w_beta = numpyro.sample('w_beta', dist.Normal(jnp.zeros(n_subjects), 1.0))
        slope_alpha = delta_alpha_subj + tau_alpha * w_alpha
        slope_beta = delta_beta_subj + tau_beta * w_beta
    else:
        slope_alpha = delta_alpha_subj * jnp.ones(n_subjects)
        slope_beta = delta_beta_subj * jnp.ones(n_subjects)

    # --- Block-order nuisance term ---
    if include_block:
        eta_alpha = numpyro.sample('eta_alpha', dist.Normal(0.0, p['eta']))
        eta_beta = numpyro.sample('eta_beta', dist.Normal(0.0, p['eta']))
    else:
        eta_alpha = 0.0
        eta_beta = 0.0

    # --- Per-run parameters ---
    subj = seq_subject
    alpha_lin = (
        mu_alpha
        + slope_alpha[subj] * cond_x
        + eta_alpha * block_x
        + sigma_alpha * z_alpha[subj]
    )
    beta_lin = (
        mu_beta
        + slope_beta[subj] * cond_x
        + eta_beta * block_x
        + sigma_beta * z_beta[subj]
    )

    alpha_seq = jax.nn.sigmoid(alpha_lin)
    beta_seq = jax.nn.softplus(beta_lin)

    numpyro.deterministic('alpha_seq', alpha_seq)
    numpyro.deterministic('beta_seq', beta_seq)

    # Learning rate per subject in each condition, on the natural scale — the
    # quantity that goes into the master CSV and the figures.
    numpyro.deterministic(
        'alpha_sham', jax.nn.sigmoid(mu_alpha - 0.5 * slope_alpha + sigma_alpha * z_alpha)
    )
    numpyro.deterministic(
        'alpha_active', jax.nn.sigmoid(mu_alpha + 0.5 * slope_alpha + sigma_alpha * z_alpha)
    )
    numpyro.deterministic(
        'beta_sham', jax.nn.softplus(mu_beta - 0.5 * slope_beta + sigma_beta * z_beta)
    )
    numpyro.deterministic(
        'beta_active', jax.nn.softplus(mu_beta + 0.5 * slope_beta + sigma_beta * z_beta)
    )

    def seq_loglik(i):
        return rw_loglik_sequence(
            choices[i], rewards[i], alpha_seq[i], beta_seq[i], mask=masks[i]
        )

    total_ll = jax.vmap(seq_loglik)(jnp.arange(choices.shape[0]))
    numpyro.factor('obs', jnp.sum(total_ll))
