#!/usr/bin/env python3
"""
Posterior predictive checks for hierarchical RW model.

Loads fitted posteriors, simulates synthetic trial sequences using posterior
parameter draws, and compares simulated summary statistics to observed data.
This validates whether the fitted model captures the structure of the real data.

Key checks:
    1. Overall accuracy: does the model reproduce each subject's accuracy?
    2. Post-reversal recovery: does the model capture how quickly subjects
       re-learn after contingency reversals?
    3. Win-stay / lose-shift: does the model reproduce choice persistence
       patterns?
    4. Accuracy by run half: does the model capture within-run learning curves?

Usage:
    python -m rl_models.ppc --data-dir ../data/bandit --results-dir results
    python -m rl_models.ppc --data-dir ../data/bandit --results-dir results --output-dir figures
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_loader import load_dataset_from_directory, BanditDataset, SubjectData


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.log1p(np.exp(x))


def simulate_from_posterior(
    subject_data: SubjectData,
    alpha: float,
    beta: float,
    *,
    alpha_neg: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Simulate one trial sequence using the subject's actual reward schedule
    but generating choices from the RW model with given parameters.

    If alpha_neg is provided, uses dual learning rates (alpha for positive PEs,
    alpha_neg for negative PEs). Otherwise uses a single learning rate.
    """
    n_trials = subject_data.n_trials
    choices_sim = np.zeros(n_trials, dtype=int)
    correct_sim = np.zeros(n_trials, dtype=bool)
    rewards_sim = np.zeros(n_trials, dtype=float)

    V = np.array([0.5, 0.5])

    for t in range(n_trials):
        # Softmax choice from model
        logits = beta * V
        logits -= np.max(logits)  # numerical stability
        p = np.exp(logits) / np.sum(np.exp(logits))
        choice = int(np.random.random() > p[0])
        choices_sim[t] = choice

        # Use the ACTUAL contingency to determine correctness
        current_good = subject_data.raw_df.iloc[t]["current_good"]
        if isinstance(current_good, (int, float, np.integer, np.floating)):
            good_idx = int(current_good) - 1
        else:
            good_idx = 0
        correct_sim[t] = (choice == good_idx)

        # Generate reward based on correctness
        reward_prob = 0.75 if correct_sim[t] else 0.25
        reward = float(np.random.random() < reward_prob)
        rewards_sim[t] = reward

        # Update values
        pe = reward - V[choice]
        if alpha_neg is not None:
            lr = alpha if pe >= 0 else alpha_neg
        else:
            lr = alpha
        V[choice] += lr * pe

    return {
        "choices": choices_sim,
        "correct": correct_sim,
        "rewards": rewards_sim,
    }


def compute_summary_stats(
    subject_data: SubjectData,
    correct: Optional[np.ndarray] = None,
    choices: Optional[np.ndarray] = None,
    rewards: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute behavioral summary statistics for one subject/simulation."""
    if correct is None:
        correct = subject_data.correct
    if choices is None:
        choices = subject_data.choices
    if rewards is None:
        rewards = subject_data.rewards

    stats = {}

    # Overall accuracy
    stats["accuracy"] = float(np.mean(correct))

    # Win-stay rate: P(same choice | previous trial was rewarded)
    win_stay = 0
    win_count = 0
    lose_shift = 0
    lose_count = 0
    for t in range(1, len(choices)):
        if rewards[t - 1] > 0.5:
            win_count += 1
            if choices[t] == choices[t - 1]:
                win_stay += 1
        else:
            lose_count += 1
            if choices[t] != choices[t - 1]:
                lose_shift += 1

    stats["win_stay"] = win_stay / max(win_count, 1)
    stats["lose_shift"] = lose_shift / max(lose_count, 1)

    # First-half vs second-half accuracy (within-run learning)
    half = len(correct) // 2
    stats["accuracy_first_half"] = float(np.mean(correct[:half]))
    stats["accuracy_second_half"] = float(np.mean(correct[half:]))

    return stats


def compute_post_reversal_accuracy(
    subject_data: SubjectData,
    correct: Optional[np.ndarray] = None,
    window: int = 10,
) -> np.ndarray:
    """Compute accuracy in sliding windows after contingency reversals.

    Returns an array of length `window` where element i is the average
    accuracy on trial i after a reversal.
    """
    if correct is None:
        correct = subject_data.correct

    df = subject_data.raw_df
    if "trial_in_contingency" not in df.columns:
        return np.full(window, np.nan)

    trial_in_cont = df["trial_in_contingency"].values

    # Find reversal points (where trial_in_contingency resets to 0 or 1)
    reversal_indices = []
    for t in range(1, len(trial_in_cont)):
        if trial_in_cont[t] <= 1 and trial_in_cont[t - 1] > 1:
            reversal_indices.append(t)

    if len(reversal_indices) == 0:
        return np.full(window, np.nan)

    # Collect accuracy at each post-reversal position
    post_rev = np.full((len(reversal_indices), window), np.nan)
    for i, rev_idx in enumerate(reversal_indices):
        for j in range(window):
            idx = rev_idx + j
            if idx < len(correct):
                post_rev[i, j] = float(correct[idx])

    return np.nanmean(post_rev, axis=0)


def run_ppc(
    dataset: BanditDataset,
    idata_path: Path,
    n_simulations: int = 200,
) -> Dict[str, any]:
    """Run posterior predictive checks.

    For each subject, draws n_simulations parameter sets from the posterior,
    simulates trial sequences, and computes summary statistics.
    """
    with open(idata_path, "rb") as f:
        idata = pickle.load(f)

    # Detect model type from posterior variable names
    post_vars = set(idata.posterior.data_vars)
    if "alpha_pos" in post_vars and "alpha_neg" in post_vars:
        model_type = "RW_dual"
        alpha_pos_post = sigmoid(idata.posterior["alpha_pos"].values.reshape(
            -1, idata.posterior["alpha_pos"].shape[-1]
        ))
        alpha_neg_post = sigmoid(idata.posterior["alpha_neg"].values.reshape(
            -1, idata.posterior["alpha_neg"].shape[-1]
        ))
        beta_post = softplus(idata.posterior["beta"].values.reshape(
            -1, idata.posterior["beta"].shape[-1]
        ))
        n_total_samples = alpha_pos_post.shape[0]
        n_subjects_posterior = alpha_pos_post.shape[1]
    else:
        model_type = "RW"
        alpha_post = sigmoid(idata.posterior["alpha"].values.reshape(
            -1, idata.posterior["alpha"].shape[-1]
        ))
        beta_post = softplus(idata.posterior["beta"].values.reshape(
            -1, idata.posterior["beta"].shape[-1]
        ))
        n_total_samples = alpha_post.shape[0]
        n_subjects_posterior = alpha_post.shape[1]

    n_subjects_data = len(dataset.subjects)

    if n_subjects_data != n_subjects_posterior:
        print(f"  Warning: dataset has {n_subjects_data} subjects but posterior "
              f"has {n_subjects_posterior}. Using min({n_subjects_data}, {n_subjects_posterior}).")
        print(f"  Make sure you use the same glob/filter for both fitting and PPC.")
    
    n_subjects = min(n_subjects_data, n_subjects_posterior)
    print(f"  Model type: {model_type}")

    results = {
        "observed": [],
        "simulated": [],
        "post_reversal_observed": [],
        "post_reversal_simulated": [],
    }

    for subj_idx, subj in enumerate(dataset.subjects[:n_subjects]):
        # Observed statistics
        obs_stats = compute_summary_stats(subj)
        obs_post_rev = compute_post_reversal_accuracy(subj)
        results["observed"].append(obs_stats)
        results["post_reversal_observed"].append(obs_post_rev)

        # Simulated statistics
        sim_stats_list = []
        sim_post_rev_list = []
        sample_indices = np.random.choice(n_total_samples, n_simulations, replace=True)

        for sim_idx in sample_indices:
            if model_type == "RW_dual":
                alpha_pos = float(alpha_pos_post[sim_idx, subj_idx])
                alpha_neg = float(alpha_neg_post[sim_idx, subj_idx])
                beta = float(beta_post[sim_idx, subj_idx])
                sim = simulate_from_posterior(subj, alpha_pos, beta,
                                             alpha_neg=alpha_neg)
            else:
                alpha = float(alpha_post[sim_idx, subj_idx])
                beta = float(beta_post[sim_idx, subj_idx])
                sim = simulate_from_posterior(subj, alpha, beta)
            sim_stats = compute_summary_stats(
                subj,
                correct=sim["correct"],
                choices=sim["choices"],
                rewards=sim["rewards"],
            )
            sim_stats_list.append(sim_stats)

            sim_post_rev = compute_post_reversal_accuracy(
                subj, correct=sim["correct"]
            )
            sim_post_rev_list.append(sim_post_rev)

        results["simulated"].append(sim_stats_list)
        results["post_reversal_simulated"].append(sim_post_rev_list)

    return results


def generate_ppc_figures(
    results: Dict,
    dataset: BanditDataset,
    output_dir: Path,
) -> None:
    """Generate posterior predictive check figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_subjects = len(results["observed"])

    # ---- Figure 1: Observed vs simulated accuracy scatter ----
    fig, ax = plt.subplots(figsize=(6, 6))
    obs_acc = [r["accuracy"] for r in results["observed"]]
    sim_acc_means = [
        np.mean([s["accuracy"] for s in results["simulated"][i]])
        for i in range(n_subjects)
    ]
    sim_acc_lo = [
        np.percentile([s["accuracy"] for s in results["simulated"][i]], 2.5)
        for i in range(n_subjects)
    ]
    sim_acc_hi = [
        np.percentile([s["accuracy"] for s in results["simulated"][i]], 97.5)
        for i in range(n_subjects)
    ]

    ax.errorbar(
        obs_acc, sim_acc_means,
        yerr=[
            [m - lo for m, lo in zip(sim_acc_means, sim_acc_lo)],
            [hi - m for m, hi in zip(sim_acc_means, sim_acc_hi)],
        ],
        fmt="o", markersize=5, alpha=0.7, color="tab:blue",
        elinewidth=0.8, capsize=0,
    )
    lim = [0.35, 0.95]
    ax.plot(lim, lim, "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("Observed accuracy")
    ax.set_ylabel("Simulated accuracy (posterior mean)")
    ax.set_title("Posterior predictive check: overall accuracy")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")

    # Annotate correlation
    r = np.corrcoef(obs_acc, sim_acc_means)[0, 1]
    ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
            fontsize=11, verticalalignment="top")

    fig.tight_layout()
    fig.savefig(output_dir / "ppc_accuracy_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  Saved ppc_accuracy_scatter.png")

    # ---- Figure 2: Win-stay / lose-shift comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for idx, stat_name, label in [
        (0, "win_stay", "Win-stay rate"),
        (1, "lose_shift", "Lose-shift rate"),
    ]:
        ax = axes[idx]
        obs = [r[stat_name] for r in results["observed"]]
        sim_means = [
            np.mean([s[stat_name] for s in results["simulated"][i]])
            for i in range(n_subjects)
        ]
        ax.scatter(obs, sim_means, alpha=0.6, s=25, color="tab:blue")
        lim = [0, 1]
        ax.plot(lim, lim, "k--", alpha=0.4, linewidth=1)
        ax.set_xlabel(f"Observed {label.lower()}")
        ax.set_ylabel(f"Simulated {label.lower()}")
        ax.set_title(label)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        r = np.corrcoef(obs, sim_means)[0, 1]
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                fontsize=11, verticalalignment="top")

    fig.tight_layout()
    fig.savefig(output_dir / "ppc_winstay_loseshift.png", dpi=150)
    plt.close(fig)
    print(f"  Saved ppc_winstay_loseshift.png")

    # ---- Figure 3: Post-reversal recovery curves ----
    fig, ax = plt.subplots(figsize=(8, 5))

    # Average observed post-reversal curve
    obs_curves = np.array(results["post_reversal_observed"])
    valid = ~np.any(np.isnan(obs_curves), axis=1)
    if np.sum(valid) > 0:
        obs_mean = np.nanmean(obs_curves[valid], axis=0)
        obs_se = np.nanstd(obs_curves[valid], axis=0) / np.sqrt(np.sum(valid))
        window = len(obs_mean)
        x = np.arange(1, window + 1)

        ax.plot(x, obs_mean, "o-", color="tab:blue", linewidth=2,
                markersize=5, label="Observed", zorder=3)
        ax.fill_between(x, obs_mean - 1.96 * obs_se, obs_mean + 1.96 * obs_se,
                        alpha=0.2, color="tab:blue")

        # Average simulated post-reversal curves
        all_sim_curves = []
        for i in range(n_subjects):
            if valid[i]:
                subj_sim_curves = np.array(results["post_reversal_simulated"][i])
                valid_sims = ~np.any(np.isnan(subj_sim_curves), axis=1)
                if np.sum(valid_sims) > 0:
                    all_sim_curves.append(np.nanmean(subj_sim_curves[valid_sims], axis=0))

        if len(all_sim_curves) > 0:
            sim_curves = np.array(all_sim_curves)
            sim_mean = np.mean(sim_curves, axis=0)
            sim_lo = np.percentile(sim_curves, 2.5, axis=0)
            sim_hi = np.percentile(sim_curves, 97.5, axis=0)

            ax.plot(x, sim_mean, "s--", color="tab:red", linewidth=2,
                    markersize=5, label="Simulated (posterior)", zorder=2)
            ax.fill_between(x, sim_lo, sim_hi, alpha=0.15, color="tab:red")

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Trials after contingency reversal")
    ax.set_ylabel("Accuracy")
    ax.set_title("Post-reversal recovery")
    ax.legend()
    ax.set_ylim(0.2, 0.9)

    fig.tight_layout()
    fig.savefig(output_dir / "ppc_post_reversal_recovery.png", dpi=150)
    plt.close(fig)
    print(f"  Saved ppc_post_reversal_recovery.png")

    # ---- Figure 4: Accuracy distribution (observed vs simulated) ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    obs_acc = [r["accuracy"] for r in results["observed"]]
    # Pool all simulated accuracies
    all_sim_acc = []
    for i in range(n_subjects):
        for s in results["simulated"][i]:
            all_sim_acc.append(s["accuracy"])

    bins = np.linspace(0.4, 0.9, 30)
    ax.hist(obs_acc, bins=bins, density=True, alpha=0.5, color="tab:blue",
            label="Observed")
    ax.hist(all_sim_acc, bins=bins, density=True, alpha=0.3, color="tab:red",
            label="Simulated")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Density")
    ax.set_title("Accuracy distributions: observed vs simulated")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "ppc_accuracy_distributions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved ppc_accuracy_distributions.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Posterior predictive checks for hierarchical RW model.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing subject bandit CSVs",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing RW_idata.pkl",
    )
    parser.add_argument(
        "--pkl-name",
        type=str,
        default="RW_idata.pkl",
        help="Name of the pickle file to load (default: RW_idata.pkl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for output figures",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=200,
        help="Number of posterior simulations per subject (default: 200)",
    )
    parser.add_argument(
        "--glob",
        default="**/sub-*_*_task-bandit_*.csv",
        help="Glob pattern for CSV files",
    )
    return parser


def main():
    args = build_parser().parse_args()

    print("Loading observed data...")
    dataset = load_dataset_from_directory(
        args.data_dir,
        glob_pattern=args.glob,
    )

    idata_path = args.results_dir / args.pkl_name
    if not idata_path.exists():
        print(f"Error: {idata_path} not found")
        return

    print(f"Running posterior predictive checks ({args.n_simulations} simulations per subject)...")
    results = run_ppc(dataset, idata_path, n_simulations=args.n_simulations)

    # Print summary
    n_subjects = len(results["observed"])
    obs_acc = [r["accuracy"] for r in results["observed"]]
    sim_acc = [
        np.mean([s["accuracy"] for s in results["simulated"][i]])
        for i in range(n_subjects)
    ]
    r = np.corrcoef(obs_acc, sim_acc)[0, 1]

    print(f"\n--- PPC Summary ---")
    print(f"  Subjects: {n_subjects}")
    print(f"  Observed accuracy:  {np.mean(obs_acc):.3f} (SD={np.std(obs_acc):.3f})")
    print(f"  Simulated accuracy: {np.mean(sim_acc):.3f} (SD={np.std(sim_acc):.3f})")
    print(f"  Obs-Sim correlation: r = {r:.3f}")

    # How many subjects' observed accuracy falls within the simulated 95% CI?
    n_covered = 0
    for i in range(n_subjects):
        sim_accs = [s["accuracy"] for s in results["simulated"][i]]
        lo, hi = np.percentile(sim_accs, [2.5, 97.5])
        if lo <= obs_acc[i] <= hi:
            n_covered += 1
    print(f"  Coverage: {n_covered}/{n_subjects} subjects' observed accuracy "
          f"within simulated 95% CI ({100*n_covered/n_subjects:.0f}%)")

    print("\nGenerating figures...")
    generate_ppc_figures(results, dataset, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
