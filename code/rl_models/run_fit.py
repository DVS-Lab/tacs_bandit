#!/usr/bin/env python3
"""
Fit hierarchical Bayesian RL models to bandit task data.

Usage:
    # Fit standard RW to all subjects
    python run_fit.py --data-dir ../data --model RW

    # Fit with condition-level group means (active vs. sham vs. baseline)
    python run_fit.py --data-dir ../data --model RW --conditions

    # Fit all models for comparison
    python run_fit.py --data-dir ../data --model all

    # Parameter recovery on synthetic data
    python run_fit.py --simulate --model RW

    # Filter to specific runs
    python run_fit.py --data-dir ../data --model RW --runs 2 3 6 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
jax.config.update("jax_platform_name", "cpu")

import arviz as az
import numpy as np

from .models import ModelType, MODEL_SPECS
from .data_loader import load_dataset_from_directory
from .fitting import fit_model, print_diagnostics, extract_subject_posteriors, compare_models
from .simulate import SimulationConfig, generate_synthetic_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit hierarchical Bayesian RL models to bandit task data.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Root directory containing subject data folders (sub-*/)",
    )
    parser.add_argument(
        "--model",
        choices=["RW", "RW_dual", "PH", "all"],
        default="RW",
        help="Which model to fit (default: RW)",
    )
    parser.add_argument(
        "--conditions",
        action="store_true",
        help="Estimate condition-level group means (active/sham/baseline)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        help="Only include trials from these run numbers",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=1000,
        help="MCMC warmup iterations (default: 1000)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="MCMC sampling iterations per chain (default: 2000)",
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=4,
        help="Number of MCMC chains (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for output files (default: results/)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run parameter recovery on synthetic data instead of real data",
    )
    parser.add_argument(
        "--sim-n-subjects",
        type=int,
        default=20,
        help="Number of simulated subjects (default: 20)",
    )
    parser.add_argument(
        "--glob",
        default="**/sub-*_ses-*_task-bandit_*.csv",
        help="Glob pattern for finding CSV files (default: **/sub-*_ses-*_task-bandit_*.csv)",
    )
    return parser


def main():
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load or simulate data ---
    if args.simulate:
        print("=== PARAMETER RECOVERY MODE ===\n")
        config = SimulationConfig(
            n_subjects=args.sim_n_subjects,
            n_trials_per_subject=200,
            seed=args.seed,
        )
        dataset, true_params = generate_synthetic_dataset(config)
        print(dataset.summary())
    else:
        if args.data_dir is None:
            print("Error: --data-dir is required when not using --simulate")
            sys.exit(1)
        print(f"Loading data from: {args.data_dir}")
        dataset = load_dataset_from_directory(
            args.data_dir,
            glob_pattern=args.glob,
            runs=args.runs,
        )
        print(dataset.summary())
        true_params = None

    # --- Determine models to fit ---
    if args.model == "all":
        model_types = [ModelType.RW, ModelType.RW_DUAL, ModelType.PH]
    else:
        model_types = [ModelType(args.model)]

    # --- Fit models ---
    results = {}
    for mt in model_types:
        result = fit_model(
            dataset,
            model_type=mt,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            seed=args.seed,
            use_conditions=args.conditions and mt == ModelType.RW,
        )
        print_diagnostics(result)
        results[mt.value] = result

        # Save ArviZ InferenceData
        netcdf_path = args.output_dir / f"{mt.value}_idata.nc"
        result.idata.to_netcdf(str(netcdf_path))
        print(f"  Saved InferenceData to {netcdf_path}")

        # Save group-level summary
        summary = result.summary()
        summary_path = args.output_dir / f"{mt.value}_summary.csv"
        summary.to_csv(str(summary_path))
        print(f"  Saved summary to {summary_path}")

        # Save individual-level posteriors
        spec = MODEL_SPECS[mt]
        indiv_posteriors = {}
        for param in spec.param_names:
            try:
                posteriors = extract_subject_posteriors(result, param)
                for subj_id, samples in posteriors.items():
                    if subj_id not in indiv_posteriors:
                        indiv_posteriors[subj_id] = {}
                    indiv_posteriors[subj_id][param] = {
                        "mean": float(np.mean(samples)),
                        "std": float(np.std(samples)),
                        "hdi_2.5": float(np.percentile(samples, 2.5)),
                        "hdi_97.5": float(np.percentile(samples, 97.5)),
                    }
            except KeyError:
                pass

        indiv_path = args.output_dir / f"{mt.value}_individual_params.json"
        with open(indiv_path, "w") as f:
            json.dump(indiv_posteriors, f, indent=2)
        print(f"  Saved individual posteriors to {indiv_path}")

    # --- Model comparison ---
    if len(results) > 1:
        print("\n=== MODEL COMPARISON ===\n")
        try:
            comparison = compare_models(results, ic="waic")
            print(comparison)
            comparison.to_csv(str(args.output_dir / "model_comparison.csv"))
        except Exception as e:
            print(f"  Model comparison failed (may need pointwise log-lik): {e}")
            print("  This is expected — we need to add pointwise log-likelihood")
            print("  computation for WAIC/LOO. Adding this is a next step.")

    # --- Parameter recovery check ---
    if true_params is not None:
        print("\n=== PARAMETER RECOVERY ===\n")
        post = results["RW"].idata.posterior
        mu_alpha_samples = jax.nn.sigmoid(post["mu_alpha"].values.flatten())
        mu_beta_samples = jax.nn.softplus(post["mu_beta"].values.flatten())

        for name, true_val, samples in [
            ("mu_alpha", true_params["mu_alpha"], mu_alpha_samples),
            ("mu_beta", true_params["mu_beta"], mu_beta_samples),
        ]:
            ci = np.percentile(samples, [2.5, 97.5])
            recovered = ci[0] <= true_val <= ci[1]
            print(f"  {name}: true={true_val:.3f}, "
                  f"mean={np.mean(samples):.3f}, "
                  f"95% CI=[{ci[0]:.3f}, {ci[1]:.3f}] "
                  f"{'OK' if recovered else 'MISS'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
