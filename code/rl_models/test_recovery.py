"""
Parameter recovery test for the hierarchical RW model.

Generates synthetic data with known parameters, fits the model,
and checks that posteriors recover the true values. This validates
the entire pipeline before running on real data.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# Force CPU to keep things simple
jax.config.update("jax_platform_name", "cpu")

from .models import ModelType
from .simulate import SimulationConfig, generate_synthetic_dataset
from .fitting import fit_model, print_diagnostics, extract_subject_posteriors


def main():
    # --- Generate synthetic data ---
    print("=" * 60)
    print("PARAMETER RECOVERY TEST")
    print("=" * 60)

    true_mu_alpha = 0.35
    true_sigma_alpha = 0.15
    true_mu_beta = 3.5
    true_sigma_beta = 0.8

    config = SimulationConfig(
        n_subjects=15,          # Reasonable for testing; match your real N
        n_trials_per_subject=150,  # ~match your per-run trial count
        seed=42,
    )

    print(f"\nGenerating synthetic data: {config.n_subjects} subjects, "
          f"{config.n_trials_per_subject} trials each")
    print(f"True parameters:")
    print(f"  mu_alpha  = {true_mu_alpha}")
    print(f"  sigma_alpha = {true_sigma_alpha}")
    print(f"  mu_beta   = {true_mu_beta}")
    print(f"  sigma_beta  = {true_sigma_beta}")

    dataset, true_params = generate_synthetic_dataset(
        config,
        true_mu_alpha=true_mu_alpha,
        true_sigma_alpha=true_sigma_alpha,
        true_mu_beta=true_mu_beta,
        true_sigma_beta=true_sigma_beta,
    )

    print(f"\n{dataset.summary()}")

    # --- Fit the model ---
    result = fit_model(
        dataset,
        model_type=ModelType.RW,
        num_warmup=500,     # Reduced for testing speed
        num_samples=1000,
        num_chains=2,       # 2 chains for testing; use 4 for real data
        seed=0,
    )

    # --- Check diagnostics ---
    print_diagnostics(result)

    # --- Check parameter recovery ---
    print("\n--- Parameter Recovery ---")

    # Group-level means
    post = result.idata.posterior

    # mu_alpha is on unconstrained scale; transform to check
    mu_alpha_samples = jax.nn.sigmoid(post["mu_alpha"].values.flatten())
    mu_alpha_mean = float(np.mean(mu_alpha_samples))
    mu_alpha_hdi = np.percentile(mu_alpha_samples, [2.5, 97.5])

    print(f"  mu_alpha:  true={true_mu_alpha:.3f}, "
          f"recovered={mu_alpha_mean:.3f}, "
          f"95% CI=[{mu_alpha_hdi[0]:.3f}, {mu_alpha_hdi[1]:.3f}]")
    recovered_alpha = mu_alpha_hdi[0] <= true_mu_alpha <= mu_alpha_hdi[1]
    print(f"    True value in 95% CI: {recovered_alpha}")

    # For beta, use the mean of individual-level transformed posteriors
    # as a proxy for the group mean (more robust than transforming mu_beta)
    beta_posteriors = extract_subject_posteriors(result, "beta")
    all_beta_means = [float(np.mean(s)) for s in beta_posteriors.values()]
    grand_mean_beta = np.mean(all_beta_means)
    grand_std_beta = np.std(all_beta_means)

    print(f"  group beta: true={true_mu_beta:.3f}, "
          f"recovered grand mean={grand_mean_beta:.3f} (SD={grand_std_beta:.3f})")

    # Individual-level recovery
    alpha_posteriors = extract_subject_posteriors(result, "alpha")
    true_alphas = true_params["alpha"]
    true_betas = true_params["beta"]

    n_alpha_recovered = 0
    for i, (subj_id, samples) in enumerate(alpha_posteriors.items()):
        ci = np.percentile(samples, [2.5, 97.5])
        if ci[0] <= true_alphas[i] <= ci[1]:
            n_alpha_recovered += 1

    n_beta_recovered = 0
    for i, (subj_id, samples) in enumerate(beta_posteriors.items()):
        ci = np.percentile(samples, [2.5, 97.5])
        if ci[0] <= true_betas[i] <= ci[1]:
            n_beta_recovered += 1

    print(f"\n  Individual alpha recovery: {n_alpha_recovered}/{config.n_subjects} "
          f"({100*n_alpha_recovered/config.n_subjects:.0f}%) within 95% CI")
    print(f"  Individual beta recovery:  {n_beta_recovered}/{config.n_subjects} "
          f"({100*n_beta_recovered/config.n_subjects:.0f}%) within 95% CI")

    # --- Summary ---
    print("\n" + "=" * 60)
    alpha_ok = n_alpha_recovered >= 0.8 * config.n_subjects
    beta_ok = n_beta_recovered >= 0.8 * config.n_subjects
    if recovered_alpha and alpha_ok and beta_ok:
        print("PASS: Parameter recovery looks good. Pipeline is validated.")
    else:
        issues = []
        if not recovered_alpha:
            issues.append("group mu_alpha not in 95% CI")
        if not alpha_ok:
            issues.append(f"individual alpha recovery {n_alpha_recovered}/{config.n_subjects}")
        if not beta_ok:
            issues.append(f"individual beta recovery {n_beta_recovered}/{config.n_subjects}")
        print(f"CHECK: {'; '.join(issues)}")
        print("       May need more samples, subjects, or investigation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
