"""
run_within_fit.py — Fit the within-subject hierarchical RW model to real data

Stage 1 of the Bayesian workflow: the unconditional model, no moderators.
Produces the group-level tACS effect (delta_alpha), the between-subject spread
in that effect (tau_alpha), and subject-level posteriors for each condition.

Moderated models come later by passing a standardized moderator matrix; the
model already accepts one, so that is a data change rather than a rewrite.

Recovery must pass before these numbers mean anything:

    python -m rl_models.test_recovery_within

Usage
-----
    python -m rl_models.run_within_fit
    python -m rl_models.run_within_fit --sample dissertation --draws 2000
"""

from __future__ import annotations

import os

# Must precede the JAX import or the chains run sequentially.
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=4')

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import arviz as az
import numpy as np
import pandas as pd

from config import REPO_ROOT
from .data_prep import build_run_dataset
from .fit_within import fit_within_model

OUTPUT_DIR = REPO_ROOT / 'derivatives' / 'rl_models'


def subject_table(idata: az.InferenceData, subject_ids: List[str]) -> pd.DataFrame:
    """
    Posterior means per subject, on the natural parameter scale.

    Column names match what the master CSV already uses for the MLE fits, so
    downstream code consuming sham_alpha / active_alpha keeps working when the
    provenance changes from point estimates to posterior means.
    """
    post = idata.posterior
    rows = {'subject_id': subject_ids}

    for var, col in [('alpha_sham', 'sham_alpha'), ('alpha_active', 'active_alpha'),
                     ('beta_sham', 'sham_beta'), ('beta_active', 'active_beta')]:
        if var in post:
            draws = post[var].values.reshape(-1, len(subject_ids))
            rows[col] = draws.mean(axis=0)
            rows[f'{col}_sd'] = draws.std(axis=0)

    df = pd.DataFrame(rows)
    if 'sham_alpha' in df and 'active_alpha' in df:
        df['delta_alpha'] = df['active_alpha'] - df['sham_alpha']
    if 'sham_beta' in df and 'active_beta' in df:
        df['delta_beta'] = df['active_beta'] - df['sham_beta']
    return df


def group_table(idata: az.InferenceData) -> pd.DataFrame:
    """Group-level parameters with HDIs — the paper's headline numbers."""
    names = ['mu_alpha', 'delta_alpha', 'eta_alpha', 'sigma_alpha', 'tau_alpha',
             'mu_beta', 'delta_beta', 'eta_beta', 'sigma_beta', 'tau_beta']
    available = [n for n in names if n in idata.posterior]
    summary = az.summary(idata, var_names=available, hdi_prob=0.94)

    # Probability of direction: how much of the posterior sits on one side of
    # zero. More directly interpretable than a p-value for this audience.
    pd_vals = {}
    for name in available:
        draws = idata.posterior[name].values.reshape(-1)
        pd_vals[name] = float(max((draws > 0).mean(), (draws < 0).mean()))
    summary['p_direction'] = pd.Series(pd_vals)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--sample', default='all',
                        choices=['all', 'dissertation', 'new'])
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-block', action='store_true',
                        help='drop the block-order term (sensitivity check)')
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR))
    args = parser.parse_args(argv)

    print('=' * 70)
    print(f'Within-subject hierarchical RW — sample={args.sample!r}')
    print('=' * 70)

    dataset = build_run_dataset(sample=args.sample, verbose=True)

    fit = fit_within_model(
        dataset,
        include_block=not args.no_block,
        num_warmup=args.warmup,
        num_samples=args.draws,
        num_chains=args.chains,
        seed=args.seed,
        progress_bar=False,
    )

    print('\n' + '=' * 70)
    print('Group-level parameters')
    print('=' * 70)
    group = group_table(fit.idata)
    print(group.round(3).to_string())

    subjects = subject_table(fit.idata, dataset.subject_ids)
    print('\n' + '=' * 70)
    print(f'Subject-level posterior means ({len(subjects)} subjects)')
    print('=' * 70)
    print(subjects.describe().round(3).to_string())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'rw_within_{args.sample}'

    fit.save(out_dir / f'{tag}_idata.pkl')
    group.to_csv(out_dir / f'{tag}_group.csv')
    subjects.to_csv(out_dir / f'{tag}_subjects.csv', index=False)
    dataset.run_index.to_csv(out_dir / f'{tag}_runs.csv', index=False)

    print(f'\nWrote {out_dir / f"{tag}_idata.pkl"}')
    print(f'Wrote {out_dir / f"{tag}_group.csv"}')
    print(f'Wrote {out_dir / f"{tag}_subjects.csv"}')

    if not fit.diagnostics['converged']:
        print('\nWARNING: diagnostics did not clear the convergence thresholds; '
              'inspect before using these numbers.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
