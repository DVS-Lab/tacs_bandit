"""
test_prior_sensitivity.py — Are the variance estimates data-driven or prior-driven?

In the first real fit the variance posteriors sat far into their priors' tails:
sigma_alpha came back at 2.80 under a HalfNormal(1.0) prior, and tau_alpha at
1.41 under a HalfNormal(0.5). When a posterior is out past the prior's bulk,
the prior is pulling the estimate *down*, and the reported value is a lower
bound rather than an estimate.

That matters because tau_alpha — between-subject variability in the tACS effect
— is a headline number. If it is prior-constrained it should be reported with a
wider prior, or at least with the sensitivity shown.

This refits at several prior widths and reports how much each parameter moves.
A parameter that barely shifts as the prior triples is identified by the data;
one that tracks the prior is not.

Usage
-----
    python -m rl_models.test_prior_sensitivity
    python -m rl_models.test_prior_sensitivity --sample dissertation
"""

from __future__ import annotations

import os

os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=4')

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import arviz as az
import numpy as np
import pandas as pd

from config import REPO_ROOT
from .data_prep import build_run_dataset
from .fit_within import fit_within_model

OUTPUT_DIR = REPO_ROOT / 'derivatives' / 'rl_models'

# Default, then progressively wider. If the estimates are data-driven they
# should be stable across all three.
PRIOR_SETTINGS = {
    'default':  {'sigma': 1.0, 'tau': 0.5, 'delta': 0.5},
    'wide':     {'sigma': 3.0, 'tau': 2.0, 'delta': 1.0},
    'very_wide': {'sigma': 5.0, 'tau': 5.0, 'delta': 2.0},
}

TRACKED = ['mu_alpha', 'delta_alpha', 'eta_alpha', 'sigma_alpha', 'tau_alpha',
           'mu_beta', 'delta_beta', 'sigma_beta', 'tau_beta']


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--sample', default='all',
                        choices=['all', 'dissertation', 'new'])
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR))
    args = parser.parse_args(argv)

    dataset = build_run_dataset(sample=args.sample, verbose=True)

    rows = []
    for label, priors in PRIOR_SETTINGS.items():
        print(f'\n{"=" * 66}')
        print(f'Priors: {label}  {priors}')
        print(f'{"=" * 66}')

        fit = fit_within_model(
            dataset, priors=priors, num_warmup=args.warmup,
            num_samples=args.draws, num_chains=args.chains,
            progress_bar=False,
        )

        available = [v for v in TRACKED if v in fit.idata.posterior]
        summary = az.summary(fit.idata, var_names=available, hdi_prob=0.94)
        for name in available:
            rows.append({
                'priors': label,
                'prior_sigma': priors['sigma'],
                'prior_tau': priors['tau'],
                'parameter': name,
                'mean': summary.loc[name, 'mean'],
                'hdi_lo': summary.loc[name, 'hdi_3%'],
                'hdi_hi': summary.loc[name, 'hdi_97%'],
                'r_hat': summary.loc[name, 'r_hat'],
                'divergences': fit.diagnostics['n_divergences'],
            })

        # The same contrast expressed on the natural scale. delta_alpha lives
        # on the logit, where a subject whose alpha sits near 0 or 1 has an
        # effectively unbounded coordinate — so the variance terms can inflate
        # without the likelihood objecting, and the prior ends up setting the
        # scale. alpha itself is bounded in (0, 1), so the mean active-minus-
        # sham difference should be identified even when the logit-scale
        # parameters are not. If this is stable while delta_alpha drifts, the
        # estimand to report is this one.
        post = fit.idata.posterior
        if 'alpha_active' in post and 'alpha_sham' in post:
            n_subj = post['alpha_active'].shape[-1]
            act = post['alpha_active'].values.reshape(-1, n_subj)
            sham = post['alpha_sham'].values.reshape(-1, n_subj)
            contrast = (act - sham).mean(axis=1)     # per posterior draw
            lo, hi = az.hdi(contrast, hdi_prob=0.94)
            rows.append({
                'priors': label,
                'prior_sigma': priors['sigma'],
                'prior_tau': priors['tau'],
                'parameter': 'alpha_contrast_natural',
                'mean': float(contrast.mean()),
                'hdi_lo': float(lo),
                'hdi_hi': float(hi),
                'r_hat': np.nan,
                'divergences': fit.diagnostics['n_divergences'],
            })

    df = pd.DataFrame(rows)
    wide = df.pivot(index='parameter', columns='priors', values='mean')
    wide = wide[[c for c in PRIOR_SETTINGS if c in wide.columns]]

    print(f'\n{"=" * 66}')
    print('Posterior means across prior widths')
    print(f'{"=" * 66}')
    print(wide.round(3).to_string())

    # Two things matter, and they are not the same question.
    #
    # 1. Does the estimate move relative to its own uncertainty? Comparing the
    #    movement to the parameter's magnitude instead is misleading for
    #    near-zero quantities: a contrast that shifts 0.017 with an HDI 0.10
    #    wide is stable, even though 0.017 is a large fraction of 0.04.
    # 2. Does the conclusion change? If every prior puts zero inside the
    #    interval, the inference is prior-robust whatever the point estimate does.
    span = wide.max(axis=1) - wide.min(axis=1)
    hdi_width = (df.groupby('parameter')
                   .apply(lambda g: (g['hdi_hi'] - g['hdi_lo']).mean(),
                          include_groups=False))
    rel_unc = (span / hdi_width.replace(0, np.nan)).fillna(0)

    includes_zero = (df.assign(z=(df['hdi_lo'] <= 0) & (df['hdi_hi'] >= 0))
                       .groupby('parameter')['z'])
    zero_consistent = includes_zero.nunique() == 1

    print('\nMovement across prior settings:')
    for name in wide.index:
        drift = 'drifts' if rel_unc[name] > 0.5 else 'stable'
        concl = 'conclusion stable' if zero_consistent.get(name, True) \
            else 'CONCLUSION CHANGES'
        print(f'  {name:22s}: range {span[name]:.3f} '
              f'({100 * rel_unc[name]:.0f}% of HDI width)  -> {drift}, {concl}')

    rel = rel_unc

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'prior_sensitivity_{args.sample}.csv'
    df.to_csv(path, index=False)
    print(f'\nWrote {path}')

    sensitive = [n for n in wide.index if rel[n] > 0.15]
    if sensitive:
        print(f'\nPrior-sensitive parameters: {sensitive}')
        print('Report these with the wider prior, or state the sensitivity.')
    else:
        print('\nAll tracked parameters are stable across prior widths.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
