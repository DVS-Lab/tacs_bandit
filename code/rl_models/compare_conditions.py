#!/usr/bin/env python3
"""
Compare posterior distributions across tACS conditions.

Loads the per-condition pickle files (RW_active_idata.pkl, RW_sham_idata.pkl,
RW_baseline_idata.pkl), transforms group-level parameters to interpretable
scales, computes posterior probabilities of condition differences, and
generates publication-quality figures.

Usage:
    python -m rl_models.compare_conditions --results-dir results
    python -m rl_models.compare_conditions --results-dir results --output-dir figures
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.log1p(np.exp(x))


def load_condition_posteriors(
    results_dir: Path,
    conditions: tuple[str, ...] = ("active", "sham", "baseline"),
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load posterior samples for each condition.

    Returns a dict like:
        {
            "active": {"mu_alpha_raw": array, "mu_alpha": array, "mu_beta_raw": ..., ...},
            "sham": {...},
            "baseline": {...},
        }
    """
    posteriors = {}
    for cond in conditions:
        pkl_path = results_dir / f"RW_{cond}_idata.pkl"
        if not pkl_path.exists():
            print(f"  Warning: {pkl_path} not found, skipping {cond}")
            continue

        with open(pkl_path, "rb") as f:
            idata = pickle.load(f)

        post = idata.posterior

        # Extract group-level means — flatten across chains
        mu_alpha_raw = post["mu_alpha"].values.flatten()
        mu_beta_raw = post["mu_beta"].values.flatten()
        sigma_alpha_raw = post["sigma_alpha"].values.flatten()
        sigma_beta_raw = post["sigma_beta"].values.flatten()

        # Transform to interpretable scale
        posteriors[cond] = {
            "mu_alpha_raw": mu_alpha_raw,
            "mu_alpha": sigmoid(mu_alpha_raw),
            "mu_beta_raw": mu_beta_raw,
            "mu_beta": softplus(mu_beta_raw),
            "sigma_alpha_raw": sigma_alpha_raw,
            "sigma_beta_raw": sigma_beta_raw,
            # Individual-level parameters
            "alpha": sigmoid(post["alpha"].values.reshape(-1, post["alpha"].shape[-1])),
            "beta": softplus(post["beta"].values.reshape(-1, post["beta"].shape[-1])),
        }

    return posteriors


def compute_posterior_comparisons(
    posteriors: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict]:
    """Compute posterior probabilities of pairwise condition differences.

    For each pair (A, B), computes:
        P(mu_alpha_A > mu_alpha_B)
        P(mu_beta_A > mu_beta_B)
        Mean difference and 95% HDI of the difference

    This is the Bayesian analog of a hypothesis test: instead of asking
    "is there a significant difference?", we ask "what is the probability
    that the active condition has a higher learning rate than sham?"
    """
    comparisons = {}
    pairs = [
        ("active", "sham"),
        ("active", "baseline"),
        ("sham", "baseline"),
    ]

    for cond_a, cond_b in pairs:
        if cond_a not in posteriors or cond_b not in posteriors:
            continue

        key = f"{cond_a}_vs_{cond_b}"
        comp = {}

        for param, param_raw in [("alpha", "mu_alpha"), ("beta", "mu_beta")]:
            samples_a = posteriors[cond_a][param_raw]
            samples_b = posteriors[cond_b][param_raw]

            # Ensure same length (subsample the longer one if needed)
            n = min(len(samples_a), len(samples_b))
            diff = samples_a[:n] - samples_b[:n]

            comp[f"P({cond_a}_{param} > {cond_b}_{param})"] = float(np.mean(diff > 0))
            comp[f"diff_{param}_mean"] = float(np.mean(diff))
            comp[f"diff_{param}_sd"] = float(np.std(diff))
            comp[f"diff_{param}_hdi_2.5"] = float(np.percentile(diff, 2.5))
            comp[f"diff_{param}_hdi_97.5"] = float(np.percentile(diff, 97.5))

        comparisons[key] = comp

    return comparisons


def print_comparison_summary(comparisons: Dict[str, Dict]) -> None:
    """Print a human-readable summary."""
    print("\n" + "=" * 70)
    print("POSTERIOR CONDITION COMPARISONS")
    print("=" * 70)

    for pair_key, comp in comparisons.items():
        cond_a, _, cond_b = pair_key.partition("_vs_")
        print(f"\n--- {cond_a.upper()} vs {cond_b.upper()} ---")

        for param in ["alpha", "beta"]:
            p_key = f"P({cond_a}_{param} > {cond_b}_{param})"
            p_val = comp[p_key]
            diff_mean = comp[f"diff_{param}_mean"]
            diff_lo = comp[f"diff_{param}_hdi_2.5"]
            diff_hi = comp[f"diff_{param}_hdi_97.5"]

            # Transform the raw differences for alpha (approximate)
            print(f"  {p_key}: {p_val:.3f}")
            print(f"    Raw diff: {diff_mean:.3f} [{diff_lo:.3f}, {diff_hi:.3f}]")

            # Interpretation
            if p_val > 0.95:
                print(f"    --> Strong evidence {cond_a} > {cond_b}")
            elif p_val > 0.90:
                print(f"    --> Moderate evidence {cond_a} > {cond_b}")
            elif p_val < 0.05:
                print(f"    --> Strong evidence {cond_b} > {cond_a}")
            elif p_val < 0.10:
                print(f"    --> Moderate evidence {cond_b} > {cond_a}")
            else:
                print(f"    --> No clear directional evidence")

    print()


def generate_figures(
    posteriors: Dict[str, Dict[str, np.ndarray]],
    comparisons: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """Generate publication-quality matplotlib figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    cond_colors = {
        "active": "tab:purple",
        "sham": "tab:green",
        "baseline": "tab:gray",
    }
    cond_labels = {
        "active": "Active (theta tACS)",
        "sham": "Sham",
        "baseline": "Baseline (no stim)",
    }
    conditions = [c for c in ["active", "sham", "baseline"] if c in posteriors]

    # ---- Figure 1: Group-level posterior density plots ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for cond in conditions:
        samples = posteriors[cond]["mu_alpha"]
        bins = np.linspace(0, 1, 80)
        ax.hist(
            samples, bins=bins, density=True, alpha=0.35,
            color=cond_colors[cond], label=cond_labels[cond],
        )
        # KDE-like line via histogram with finer bins
        counts, edges = np.histogram(samples, bins=200, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, counts, color=cond_colors[cond], linewidth=1.5)
    ax.set_xlabel("Group mean learning rate (α)")
    ax.set_ylabel("Posterior density")
    ax.set_title("Group mean learning rate by condition")
    ax.legend(fontsize=9)

    ax = axes[1]
    for cond in conditions:
        samples = posteriors[cond]["mu_beta"]
        bins = np.linspace(0, 6, 80)
        ax.hist(
            samples, bins=bins, density=True, alpha=0.35,
            color=cond_colors[cond], label=cond_labels[cond],
        )
        counts, edges = np.histogram(samples, bins=200, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, counts, color=cond_colors[cond], linewidth=1.5)
    ax.set_xlabel("Group mean inverse temperature (β)")
    ax.set_ylabel("Posterior density")
    ax.set_title("Group mean inverse temperature by condition")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "group_posterior_densities.png", dpi=150)
    plt.close(fig)
    print(f"  Saved group_posterior_densities.png")

    # ---- Figure 2: Posterior difference distributions (active - sham) ----
    if "active" in posteriors and "sham" in posteriors:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        n = min(len(posteriors["active"]["mu_alpha"]),
                len(posteriors["sham"]["mu_alpha"]))

        # Alpha difference (on transformed scale)
        alpha_diff = posteriors["active"]["mu_alpha"][:n] - posteriors["sham"]["mu_alpha"][:n]
        ax = axes[0]
        ax.hist(alpha_diff, bins=80, density=True, color="tab:purple", alpha=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(np.mean(alpha_diff), color="tab:red", linestyle="-", linewidth=1.5,
                   label=f"Mean = {np.mean(alpha_diff):.3f}")
        p_gt = np.mean(alpha_diff > 0)
        ax.set_xlabel("Δα (active − sham)")
        ax.set_ylabel("Posterior density")
        ax.set_title(f"Learning rate difference\nP(active > sham) = {p_gt:.3f}")
        ax.legend(fontsize=9)

        # Beta difference
        beta_diff = posteriors["active"]["mu_beta"][:n] - posteriors["sham"]["mu_beta"][:n]
        ax = axes[1]
        ax.hist(beta_diff, bins=80, density=True, color="tab:purple", alpha=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(np.mean(beta_diff), color="tab:red", linestyle="-", linewidth=1.5,
                   label=f"Mean = {np.mean(beta_diff):.3f}")
        p_gt_beta = np.mean(beta_diff > 0)
        ax.set_xlabel("Δβ (active − sham)")
        ax.set_ylabel("Posterior density")
        ax.set_title(f"Inverse temperature difference\nP(active > sham) = {p_gt_beta:.3f}")
        ax.legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(output_dir / "active_vs_sham_difference.png", dpi=150)
        plt.close(fig)
        print(f"  Saved active_vs_sham_difference.png")

    # ---- Figure 3: Individual-level parameter scatter by condition ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (param, label, xlim) in enumerate([
        ("alpha", "Learning rate (α)", (0, 1)),
        ("beta", "Inverse temperature (β)", (0, 8)),
    ]):
        ax = axes[idx]
        for cond in conditions:
            # Posterior means per subject
            subj_means = np.mean(posteriors[cond][param], axis=0)
            subj_sds = np.std(posteriors[cond][param], axis=0)
            n_subj = len(subj_means)
            y_jitter = np.random.RandomState(42).normal(0, 0.08, n_subj)
            cond_y = {"active": 2, "sham": 1, "baseline": 0}[cond]

            ax.errorbar(
                subj_means, cond_y + y_jitter,
                xerr=subj_sds,
                fmt="o", markersize=3.5, alpha=0.5,
                color=cond_colors[cond], elinewidth=0.5, capsize=0,
            )
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Baseline", "Sham", "Active"])
        ax.set_xlabel(label)
        ax.set_xlim(xlim)
        ax.set_title(f"Individual {label} by condition")

    fig.tight_layout()
    fig.savefig(output_dir / "individual_params_by_condition.png", dpi=150)
    plt.close(fig)
    print(f"  Saved individual_params_by_condition.png")

    # ---- Figure 4: Condition summary bar chart with error bars ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    x = np.arange(len(conditions))
    width = 0.5

    # Alpha
    ax = axes[0]
    means_alpha = [np.mean(posteriors[c]["mu_alpha"]) for c in conditions]
    ci_lo_alpha = [np.percentile(posteriors[c]["mu_alpha"], 2.5) for c in conditions]
    ci_hi_alpha = [np.percentile(posteriors[c]["mu_alpha"], 97.5) for c in conditions]
    err_lo = [m - lo for m, lo in zip(means_alpha, ci_lo_alpha)]
    err_hi = [hi - m for m, hi in zip(means_alpha, ci_hi_alpha)]
    bars = ax.bar(
        x, means_alpha, width,
        color=[cond_colors[c] for c in conditions],
        alpha=0.6, edgecolor=[cond_colors[c] for c in conditions],
    )
    ax.errorbar(
        x, means_alpha, yerr=[err_lo, err_hi],
        fmt="none", ecolor="black", capsize=5, linewidth=1.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([cond_labels[c] for c in conditions], fontsize=9)
    ax.set_ylabel("Group mean α")
    ax.set_title("Learning rate by condition")
    ax.set_ylim(0, 1)

    # Beta
    ax = axes[1]
    means_beta = [np.mean(posteriors[c]["mu_beta"]) for c in conditions]
    ci_lo_beta = [np.percentile(posteriors[c]["mu_beta"], 2.5) for c in conditions]
    ci_hi_beta = [np.percentile(posteriors[c]["mu_beta"], 97.5) for c in conditions]
    err_lo = [m - lo for m, lo in zip(means_beta, ci_lo_beta)]
    err_hi = [hi - m for m, hi in zip(means_beta, ci_hi_beta)]
    ax.bar(
        x, means_beta, width,
        color=[cond_colors[c] for c in conditions],
        alpha=0.6, edgecolor=[cond_colors[c] for c in conditions],
    )
    ax.errorbar(
        x, means_beta, yerr=[err_lo, err_hi],
        fmt="none", ecolor="black", capsize=5, linewidth=1.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([cond_labels[c] for c in conditions], fontsize=9)
    ax.set_ylabel("Group mean β")
    ax.set_title("Inverse temperature by condition")
    ax.set_ylim(0, 4)

    fig.tight_layout()
    fig.savefig(output_dir / "condition_summary_bars.png", dpi=150)
    plt.close(fig)
    print(f"  Saved condition_summary_bars.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare posterior distributions across tACS conditions.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing RW_{condition}_idata.pkl files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for output figures (default: figures/)",
    )
    return parser


def main():
    args = build_parser().parse_args()

    print("Loading posterior samples...")
    posteriors = load_condition_posteriors(args.results_dir)

    if len(posteriors) < 2:
        print("Need at least 2 conditions to compare. Check your results directory.")
        return

    # Print summary statistics
    print("\n--- Group-level parameter estimates (transformed) ---")
    for cond, post in posteriors.items():
        mu_a = post["mu_alpha"]
        mu_b = post["mu_beta"]
        print(f"  {cond:>10s}:  α = {np.mean(mu_a):.3f} "
              f"[{np.percentile(mu_a, 2.5):.3f}, {np.percentile(mu_a, 97.5):.3f}]"
              f"  β = {np.mean(mu_b):.3f} "
              f"[{np.percentile(mu_b, 2.5):.3f}, {np.percentile(mu_b, 97.5):.3f}]")

    # Compute and print comparisons
    comparisons = compute_posterior_comparisons(posteriors)
    print_comparison_summary(comparisons)

    # Save comparisons as JSON
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comp_path = args.output_dir / "posterior_comparisons.json"
    with open(comp_path, "w") as f:
        json.dump(comparisons, f, indent=2)
    print(f"  Saved posterior comparisons to {comp_path}")

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(posteriors, comparisons, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
