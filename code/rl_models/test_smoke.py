"""Minimal smoke test — validates pipeline runs end-to-end."""

import jax
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np
from .models import ModelType, rw_loglik_sequence
from .simulate import SimulationConfig, generate_synthetic_dataset, simulate_rw_agent
from .fitting import fit_model, print_diagnostics, extract_subject_posteriors
from .data_loader import SubjectData
import jax.random as jr


def test_rw_loglik():
    """Verify RW log-likelihood produces finite values."""
    choices = jnp.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0])
    rewards = jnp.array([1., 0., 1., 1., 0., 1., 0., 1., 1., 0.])
    ll = rw_loglik_sequence(choices, rewards, alpha=0.3, beta=3.0)
    assert jnp.isfinite(ll), f"Log-likelihood is not finite: {ll}"
    assert ll < 0, f"Log-likelihood should be negative: {ll}"
    print(f"  RW loglik: {float(ll):.4f} — OK")


def test_simulation():
    """Verify simulation produces valid data."""
    key = jr.PRNGKey(0)
    sim = simulate_rw_agent(key, alpha=0.3, beta=3.0, n_trials=100)
    assert sim["choices"].shape == (100,)
    assert set(np.unique(sim["choices"])).issubset({0, 1})
    assert set(np.unique(sim["rewards"])).issubset({0.0, 1.0})
    acc = np.mean(sim["correct"])
    print(f"  Simulation: 100 trials, accuracy={acc:.1%} — OK")


def test_dataset_construction():
    """Verify synthetic dataset builds correctly."""
    config = SimulationConfig(n_subjects=3, n_trials_per_subject=50, seed=0)
    dataset, true_params = generate_synthetic_dataset(config)
    assert dataset.n_subjects == 3
    assert dataset.choices.shape == (3, 50)
    assert dataset.masks.shape == (3, 50)
    assert dataset.choices.dtype == jnp.int32
    assert dataset.rewards.dtype == jnp.float32
    assert float(jnp.sum(dataset.masks)) == 150
    print(f"  Dataset: {dataset.n_subjects} subjects, shape={dataset.choices.shape} — OK")


def test_mcmc_runs():
    """Verify MCMC executes without error (minimal samples)."""
    config = SimulationConfig(n_subjects=3, n_trials_per_subject=50, seed=42)
    dataset, _ = generate_synthetic_dataset(config)

    result = fit_model(
        dataset,
        model_type=ModelType.RW,
        num_warmup=50,
        num_samples=50,
        num_chains=1,
        seed=0,
    )
    print_diagnostics(result)

    # Check we can extract posteriors
    alpha_post = extract_subject_posteriors(result, "alpha")
    assert len(alpha_post) == 3
    for subj_id, samples in alpha_post.items():
        assert len(samples) == 50  # num_samples * num_chains
        assert np.all(np.isfinite(samples))
        assert np.all((samples >= 0) & (samples <= 1))  # sigmoid bounded
    print("  MCMC + posterior extraction — OK")


if __name__ == "__main__":
    print("Running smoke tests...\n")

    test_rw_loglik()
    test_simulation()
    test_dataset_construction()
    test_mcmc_runs()

    print("\nAll smoke tests passed.")
