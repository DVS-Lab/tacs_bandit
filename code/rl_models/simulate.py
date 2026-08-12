"""
Synthetic data generation for model recovery and parameter recovery tests.

Generates two-armed bandit trial sequences with known RL parameters,
matching the structure of the real task (contingency reversals,
75/25 reward probabilities, Bernoulli reward draws).

Model recovery: simulate from each model, fit all models, verify correct
model is selected by information criteria.

Parameter recovery: simulate with known parameters, fit, check posterior
concentrates near true values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd

from .data_loader import BanditDataset, SubjectData, DEFAULT_CONDITION_MAP


@dataclass
class SimulationConfig:
    """Configuration for synthetic data generation."""

    n_subjects: int = 20
    n_trials_per_subject: int = 200
    win_fraction: float = 0.75
    min_contingency_trials: int = 25
    contingency_jitter: int = 4
    seed: int = 0


def _seed_from_key(rng_key: jax.Array) -> int:
    """Derive a NumPy seed from a JAX PRNG key.

    Both words have to be mixed. jax.random.PRNGKey(n) puts the seed in word 1
    and leaves word 0 as zero, so seeding from word 0 alone gives the same
    seed for every key — every "independent" simulated agent comes out
    byte-identical, which silently removes all between-subject variance from a
    recovery test.
    """
    key = np.asarray(rng_key).astype(np.uint64).ravel()
    mixed = 0
    for word in key:
        mixed = (mixed * np.uint64(6364136223846793005) + np.uint64(word)) % (2**63)
    return int(mixed % (2**31))


def simulate_rw_agent(
    rng_key: jax.Array,
    alpha: float,
    beta: float,
    *,
    n_trials: int = 200,
    win_fraction: float = 0.75,
    min_contingency_trials: int = 25,
    contingency_jitter: int = 4,
) -> dict[str, np.ndarray]:
    """Simulate a single RW agent on a two-armed bandit.

    Returns a dict with choices, rewards, correct, current_good arrays.
    """
    rng = np.random.RandomState(_seed_from_key(rng_key))

    V = np.array([0.5, 0.5])
    current_good = rng.randint(0, 2)  # 0 or 1
    trial_in_contingency = 0
    contingency_trials = min_contingency_trials + rng.randint(0, contingency_jitter + 1)

    choices = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials, dtype=float)
    correct = np.zeros(n_trials, dtype=bool)
    good_option = np.zeros(n_trials, dtype=int)

    for t in range(n_trials):
        good_option[t] = current_good

        # Softmax choice
        logits = beta * V
        p = np.exp(logits) / np.sum(np.exp(logits))
        choice = int(rng.random() > p[0])  # 0 or 1
        choices[t] = choice

        # Reward
        is_correct = choice == current_good
        correct[t] = is_correct
        reward_prob = win_fraction if is_correct else (1.0 - win_fraction)
        reward = float(rng.random() < reward_prob)
        rewards[t] = reward

        # Update
        pe = reward - V[choice]
        V[choice] += alpha * pe

        # Contingency reversal
        trial_in_contingency += 1
        if trial_in_contingency >= contingency_trials:
            current_good = 1 - current_good
            trial_in_contingency = 0
            contingency_trials = min_contingency_trials + rng.randint(
                0, contingency_jitter + 1
            )

    return {
        "choices": choices,
        "rewards": rewards,
        "correct": correct,
        "current_good": good_option,
    }


def simulate_rw_dual_agent(
    rng_key: jax.Array,
    alpha_pos: float,
    alpha_neg: float,
    beta: float,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Simulate a dual learning-rate RW agent."""
    rng = np.random.RandomState(_seed_from_key(rng_key))

    n_trials = kwargs.get("n_trials", 200)
    win_fraction = kwargs.get("win_fraction", 0.75)
    min_contingency_trials = kwargs.get("min_contingency_trials", 25)
    contingency_jitter = kwargs.get("contingency_jitter", 4)

    V = np.array([0.5, 0.5])
    current_good = rng.randint(0, 2)
    trial_in_contingency = 0
    contingency_trials = min_contingency_trials + rng.randint(0, contingency_jitter + 1)

    choices = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials, dtype=float)
    correct = np.zeros(n_trials, dtype=bool)
    good_option = np.zeros(n_trials, dtype=int)

    for t in range(n_trials):
        good_option[t] = current_good
        logits = beta * V
        p = np.exp(logits) / np.sum(np.exp(logits))
        choice = int(rng.random() > p[0])
        choices[t] = choice

        is_correct = choice == current_good
        correct[t] = is_correct
        reward_prob = win_fraction if is_correct else (1.0 - win_fraction)
        reward = float(rng.random() < reward_prob)
        rewards[t] = reward

        pe = reward - V[choice]
        alpha = alpha_pos if pe >= 0 else alpha_neg
        V[choice] += alpha * pe

        trial_in_contingency += 1
        if trial_in_contingency >= contingency_trials:
            current_good = 1 - current_good
            trial_in_contingency = 0
            contingency_trials = min_contingency_trials + rng.randint(
                0, contingency_jitter + 1
            )

    return {
        "choices": choices,
        "rewards": rewards,
        "correct": correct,
        "current_good": good_option,
    }


def generate_synthetic_dataset(
    config: SimulationConfig,
    *,
    true_mu_alpha: float = 0.3,
    true_sigma_alpha: float = 0.1,
    true_mu_beta: float = 3.0,
    true_sigma_beta: float = 1.0,
) -> tuple[BanditDataset, dict[str, np.ndarray]]:
    """Generate a full synthetic dataset with known group-level parameters.

    Draws individual-level parameters from a normal distribution (on the
    transformed scale), generates trial sequences, and packages into a
    BanditDataset.

    Returns
    -------
    dataset : BanditDataset ready for model fitting.
    true_params : dict with arrays of true individual-level parameters.
    """
    rng = np.random.RandomState(config.seed)

    # Draw individual parameters
    # alpha: sample on logit scale, then sigmoid
    logit_alpha = np.log(true_mu_alpha / (1 - true_mu_alpha))
    logit_alphas = rng.normal(logit_alpha, true_sigma_alpha, config.n_subjects)
    alphas = 1.0 / (1.0 + np.exp(-logit_alphas))

    # beta: sample on log scale, then softplus (≈ exp for large values)
    log_beta = np.log(np.exp(true_mu_beta) - 1)  # inverse softplus
    log_betas = rng.normal(log_beta, true_sigma_beta, config.n_subjects)
    betas = np.log(1 + np.exp(log_betas))

    subjects = []

    for i in range(config.n_subjects):
        key = jr.PRNGKey(config.seed + i)
        sim = simulate_rw_agent(
            key,
            alphas[i],
            betas[i],
            n_trials=config.n_trials_per_subject,
            win_fraction=config.win_fraction,
            min_contingency_trials=config.min_contingency_trials,
            contingency_jitter=config.contingency_jitter,
        )

        subj_id = f"sim_{i:03d}"
        subj = SubjectData(
            subject_id=subj_id,
            choices=sim["choices"],
            rewards=sim["rewards"],
            n_trials=config.n_trials_per_subject,
            run_ids=np.ones(config.n_trials_per_subject, dtype=int),
            conditions=np.zeros(config.n_trials_per_subject, dtype=int),
            correct=sim["correct"],
            rts=np.full(config.n_trials_per_subject, np.nan),
            raw_df=pd.DataFrame(),
        )
        subjects.append(subj)

    dataset = BanditDataset(
        subjects=subjects,
        choices=jnp.array(
            np.stack([s.choices for s in subjects]), dtype=jnp.int32
        ),
        rewards=jnp.array(
            np.stack([s.rewards for s in subjects]), dtype=jnp.float32
        ),
        masks=jnp.ones(
            (config.n_subjects, config.n_trials_per_subject), dtype=jnp.float32
        ),
        condition_ids=jnp.zeros(config.n_subjects, dtype=jnp.int32),
        n_subjects=config.n_subjects,
        n_conditions=1,
        max_trials=config.n_trials_per_subject,
        subject_id_map={f"sim_{i:03d}": i for i in range(config.n_subjects)},
        condition_map=DEFAULT_CONDITION_MAP,
    )

    true_params = {
        "alpha": alphas,
        "beta": betas,
        "mu_alpha": true_mu_alpha,
        "sigma_alpha": true_sigma_alpha,
        "mu_beta": true_mu_beta,
        "sigma_beta": true_sigma_beta,
    }

    return dataset, true_params
