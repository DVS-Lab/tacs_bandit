"""
test_recovery_within.py — Can the model recover the tACS effect?

test_recovery.py checks that alpha and beta come back. That is necessary but
not sufficient here: the estimand for the paper is delta_alpha, the condition
effect, and a model can recover its subject-level parameters perfectly while
being unable to say anything useful about the difference between conditions.
This simulates data with a known delta_alpha at the real design's dimensions
and asks whether the posterior finds it.

Two cases matter:

  signal — a true non-zero delta_alpha should land inside the posterior
           interval, and the interval should exclude zero often enough for the
           design to be worth running
  null   — a true delta_alpha of zero should produce an interval covering
           zero, otherwise the model manufactures effects

Usage
-----
    python -m rl_models.test_recovery_within
    python -m rl_models.test_recovery_within --delta 0.0 0.4 --draws 1000
"""

from __future__ import annotations

import os

# NUTS chains run sequentially unless XLA is told there are several devices,
# and this has to be set before JAX initializes its backend.
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=4')

import argparse
import sys
from typing import Dict, List, Optional

import arviz as az
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd

from .data_prep import RunDataset
from .fit_within import fit_within_model
from .simulate import simulate_rw_agent

# Matched to the real dataset: 55 subjects, ~4 runs each (2 per condition),
# ~73 trials per run.
DEFAULT_N_SUBJECTS = 55
DEFAULT_RUNS_PER_CONDITION = 2
DEFAULT_N_TRIALS = 73


def simulate_within_dataset(
    *,
    mu_alpha: float = 0.0,
    delta_alpha: float = 0.4,
    sigma_alpha: float = 0.8,
    tau_alpha: float = 0.3,
    mu_beta: float = 1.2,
    delta_beta: float = 0.0,
    sigma_beta: float = 0.5,
    tau_beta: float = 0.2,
    eta_alpha: float = 0.0,
    n_subjects: int = DEFAULT_N_SUBJECTS,
    runs_per_condition: int = DEFAULT_RUNS_PER_CONDITION,
    n_trials: int = DEFAULT_N_TRIALS,
    seed: int = 0,
) -> tuple[RunDataset, Dict[str, float]]:
    """
    Generate run-level sequences from the same generative form the model assumes.

    Parameters are on the linear (pre-link) scale, matching the model: alpha
    passes through a sigmoid, beta through a softplus.

    Block is assigned so that half the subjects get active first and half get
    sham first, mirroring the counterbalance. Without that, condition and block
    would be perfectly confounded and the block term unidentifiable.
    """
    rng = np.random.RandomState(seed)

    z_alpha = rng.normal(size=n_subjects)
    z_beta = rng.normal(size=n_subjects)
    w_alpha = rng.normal(size=n_subjects)
    w_beta = rng.normal(size=n_subjects)

    slope_alpha = delta_alpha + tau_alpha * w_alpha
    slope_beta = delta_beta + tau_beta * w_beta

    rows: List[Dict] = []
    key_counter = 0

    for j in range(n_subjects):
        # Half the subjects are "counterbalance A": active in the first block.
        active_first = (j % 2 == 0)

        for cond_code, cond_name in [(0, 'sham'), (1, 'active')]:
            cond_x = cond_code - 0.5
            if cond_name == 'active':
                block = 0 if active_first else 1
            else:
                block = 1 if active_first else 0
            block_x = block - 0.5

            alpha_lin = (mu_alpha + slope_alpha[j] * cond_x
                         + eta_alpha * block_x + sigma_alpha * z_alpha[j])
            beta_lin = (mu_beta + slope_beta[j] * cond_x
                        + sigma_beta * z_beta[j])

            alpha = float(1.0 / (1.0 + np.exp(-alpha_lin)))
            beta = float(np.log1p(np.exp(beta_lin)))

            for _ in range(runs_per_condition):
                key_counter += 1
                sim = simulate_rw_agent(
                    jr.PRNGKey(seed * 100003 + key_counter),
                    alpha, beta, n_trials=n_trials,
                )
                rows.append({
                    'subject': j,
                    'condition': cond_code,
                    'block': block,
                    'choices': np.asarray(sim['choices'], dtype=np.int32),
                    'rewards': np.asarray(sim['rewards'], dtype=np.float32),
                })

    n_seq = len(rows)
    choices = np.zeros((n_seq, n_trials), dtype=np.int32)
    rewards = np.zeros((n_seq, n_trials), dtype=np.float32)
    masks = np.ones((n_seq, n_trials), dtype=np.float32)
    seq_subject = np.array([r['subject'] for r in rows], dtype=np.int32)
    seq_condition = np.array([r['condition'] for r in rows], dtype=np.int32)
    seq_block = np.array([r['block'] for r in rows], dtype=np.int32)

    for i, r in enumerate(rows):
        choices[i] = r['choices']
        rewards[i] = r['rewards']

    run_index = pd.DataFrame([
        {'subject_id': str(r['subject']), 'run': -1,
         'condition': 'active' if r['condition'] else 'sham',
         'block': r['block'], 'n_trials': n_trials}
        for r in rows
    ])

    dataset = RunDataset(
        choices=jnp.array(choices),
        rewards=jnp.array(rewards),
        masks=jnp.array(masks),
        seq_subject=jnp.array(seq_subject),
        seq_condition=jnp.array(seq_condition),
        seq_block=jnp.array(seq_block),
        subject_ids=[str(j) for j in range(n_subjects)],
        run_index=run_index,
    )

    truth = {
        'mu_alpha': mu_alpha, 'delta_alpha': delta_alpha,
        'sigma_alpha': sigma_alpha, 'tau_alpha': tau_alpha,
        'mu_beta': mu_beta, 'delta_beta': delta_beta,
        'eta_alpha': eta_alpha,
    }
    return dataset, truth


def _interval(idata: az.InferenceData, name: str, hdi_prob: float = 0.94):
    draws = idata.posterior[name].values.reshape(-1)
    lo, hi = az.hdi(draws, hdi_prob=hdi_prob)
    return float(draws.mean()), float(lo), float(hi)


def run_recovery(
    delta_alpha: float,
    *,
    seed: int = 0,
    draws: int = 1000,
    warmup: int = 1000,
    chains: int = 4,
    verbose: bool = True,
) -> Dict:
    """Simulate at a known delta_alpha, fit, and report whether it came back."""
    dataset, truth = simulate_within_dataset(delta_alpha=delta_alpha, seed=seed)

    if verbose:
        print(f'\n{"=" * 66}')
        print(f'Recovery at true delta_alpha = {delta_alpha:+.2f}')
        print(f'{"=" * 66}')
        print(dataset.summary())

    fit = fit_within_model(
        dataset, num_samples=draws, num_warmup=warmup, num_chains=chains,
        seed=seed + 1, progress_bar=False, verbose=verbose,
    )

    result = {'true_delta_alpha': delta_alpha, **fit.diagnostics}
    for name in ['mu_alpha', 'delta_alpha', 'sigma_alpha', 'tau_alpha', 'delta_beta']:
        if name not in fit.idata.posterior:
            continue
        mean, lo, hi = _interval(fit.idata, name)
        true = truth.get(name, np.nan)
        result[f'{name}_mean'] = mean
        result[f'{name}_hdi_lo'] = lo
        result[f'{name}_hdi_hi'] = hi
        result[f'{name}_true'] = true
        result[f'{name}_covered'] = bool(lo <= true <= hi) if np.isfinite(true) else None

    d_lo, d_hi = result['delta_alpha_hdi_lo'], result['delta_alpha_hdi_hi']
    result['excludes_zero'] = bool(d_lo > 0 or d_hi < 0)

    if verbose:
        print(f'\n  delta_alpha: true {delta_alpha:+.3f}  '
              f'posterior {result["delta_alpha_mean"]:+.3f} '
              f'[{d_lo:+.3f}, {d_hi:+.3f}]')
        print(f'    true value inside 94% HDI: {result["delta_alpha_covered"]}')
        print(f'    HDI excludes zero:         {result["excludes_zero"]}')
        for name in ['mu_alpha', 'sigma_alpha', 'tau_alpha']:
            if f'{name}_mean' in result:
                print(f'  {name:12s}: true {result[f"{name}_true"]:+.3f}  '
                      f'posterior {result[f"{name}_mean"]:+.3f} '
                      f'[{result[f"{name}_hdi_lo"]:+.3f}, {result[f"{name}_hdi_hi"]:+.3f}]  '
                      f'covered={result[f"{name}_covered"]}')

    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--delta', type=float, nargs='+', default=[0.0, 0.4],
                        help='true delta_alpha values to test')
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)

    # Offset the seed per condition. Reusing one seed makes every condition
    # draw identical subject-level random effects, so a single unlucky draw
    # shows up in all of them and reads as systematic bias rather than as the
    # sampling variation it is.
    results = [
        run_recovery(d, seed=args.seed + 100 * i, draws=args.draws,
                     warmup=args.warmup, chains=args.chains)
        for i, d in enumerate(args.delta)
    ]

    df = pd.DataFrame(results)
    print(f'\n{"=" * 66}')
    print('Recovery summary')
    print(f'{"=" * 66}')
    cols = ['true_delta_alpha', 'delta_alpha_mean', 'delta_alpha_hdi_lo',
            'delta_alpha_hdi_hi', 'delta_alpha_covered', 'excludes_zero',
            'max_r_hat', 'n_divergences', 'converged']
    print(df[[c for c in cols if c in df.columns]].round(3).to_string(index=False))

    ok = bool(df['delta_alpha_covered'].all()) and bool(df['converged'].all())
    print(f'\n{"PASS" if ok else "FAIL"}: '
          f'{"delta_alpha recovered and chains converged in every case" if ok else "see table above"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
