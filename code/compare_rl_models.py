"""
compare_rl_models.py — Single vs dual learning rate, by cross-validation

The posterior predictive check on the single-rate model (`ppc_within.py`)
reproduces accuracy, post-reversal recovery and win-stay closely but predicts
lose-shift near 0.5 for everyone, while participants range from 0.1 to 1.0
(r = .49 between observed and predicted, 61% inside the 95% interval). One
learning rate cannot respond differently to better- and worse-than-expected
outcomes, which is what that gap looks like. The dual model adds exactly that
asymmetry and nothing else, so a comparison isolates it.

Comparison is **leave-one-run-out** PSIS-LOO. A run is the natural unit: each
is an independent bandit with its own values, and the model is fit on runs.
Leave-one-*trial*-out would be wrong here, since trials within a run are
sequentially dependent by construction.

The models are fit with `numpyro.factor`, which stores no per-observation
log-likelihood, so it is recomputed here from the posterior draws using each
model's own likelihood function -- the same code the sampler used.

Reports elpd for each model, the difference with its standard error, and the
Pareto-k diagnostic. A difference smaller than about twice its SE is not
evidence for either model.

Usage
-----
    python compare_rl_models.py
    python compare_rl_models.py --draws 1000
"""

from __future__ import annotations

import argparse
import pickle
import sys

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from config import REPO_ROOT
from rl_models.models import rw_loglik_sequence, rw_dual_loglik_sequence
from rl_models.data_prep import build_run_dataset

RL_DIR = REPO_ROOT / 'derivatives' / 'rl_models'
OUT = REPO_ROOT / 'derivatives' / 'rl_model_comparison.csv'


def load(model: str):
    with open(RL_DIR / f'{model}_within_all_idata.pkl', 'rb') as f:
        blob = pickle.load(f)
    return (blob['idata'] if isinstance(blob, dict) else blob,
            blob.get('diagnostics', {}) if isinstance(blob, dict) else {})


def run_loglik(model: str, idata, ds, draw_idx) -> np.ndarray:
    """(draws, runs) log-likelihood, recomputed with the model's own likelihood."""
    post = idata.posterior
    beta = jnp.asarray(post['beta_seq'].values.reshape(-1, ds.n_sequences)[draw_idx])
    if model == 'rw_dual':
        a_pos = jnp.asarray(post['alpha_pos_seq'].values.reshape(-1, ds.n_sequences)[draw_idx])
        a_neg = jnp.asarray(post['alpha_neg_seq'].values.reshape(-1, ds.n_sequences)[draw_idx])

        def one(i, ap, an, b):
            return rw_dual_loglik_sequence(ds.choices[i], ds.rewards[i], ap, an, b,
                                           mask=ds.masks[i])
        per_draw = jax.vmap(lambda ap, an, b: jax.vmap(one)(jnp.arange(ds.n_sequences), ap, an, b))
        return np.asarray(per_draw(a_pos, a_neg, beta))

    alpha = jnp.asarray(post['alpha_seq'].values.reshape(-1, ds.n_sequences)[draw_idx])

    def one(i, a, b):
        return rw_loglik_sequence(ds.choices[i], ds.rewards[i], a, b, mask=ds.masks[i])
    per_draw = jax.vmap(lambda a, b: jax.vmap(one)(jnp.arange(ds.n_sequences), a, b))
    return np.asarray(per_draw(alpha, beta))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20260922)
    args = ap.parse_args(argv)

    ds = build_run_dataset(sample='all', verbose=True)
    rng = np.random.default_rng(args.seed)

    loos, rows = {}, []
    for model in ('rw', 'rw_dual'):
        idata, diag = load(model)
        n_draws = idata.posterior['beta_seq'].values.reshape(-1, ds.n_sequences).shape[0]
        idx = rng.choice(n_draws, size=min(args.draws, n_draws), replace=False)
        ll = run_loglik(model, idata, ds, idx)                      # (draws, runs)
        print(f'{model}: max R-hat {diag.get("max_r_hat", float("nan")):.3f}, '
              f'{diag.get("n_divergences", "?")} divergences, '
              f'{len(idx)} draws x {ds.n_sequences} runs')
        # One chain of the subsampled draws is enough for PSIS-LOO weights.
        d = az.from_dict(posterior={'dummy': np.zeros((1, len(idx)))},
                         log_likelihood={'run': ll[None, :, :]})
        loo = az.loo(d, pointwise=True)
        loos[model] = loo
        k = np.asarray(loo.pareto_k)
        rows.append(dict(model=model, elpd_loo=float(loo.elpd_loo), se=float(loo.se),
                         p_loo=float(loo.p_loo), max_pareto_k=float(k.max()),
                         n_k_above_0_7=int((k > 0.7).sum())))

    out = pd.DataFrame(rows)
    print()
    print(out.round(2).to_string(index=False))

    # PSIS-LOO is not trustworthy when many Pareto k exceed 0.7, and here it
    # is structural rather than incidental: every run carries its own alpha and
    # beta, so leaving a run out is close to leaving out the parameters that
    # explain it, and the importance weights blow up. p_loo above the number of
    # observations says the same thing. The elpd difference below is therefore
    # reported *with* this warning, and the posterior predictive check
    # (ppc_within.py) is the comparison to lean on.
    bad = out.n_k_above_0_7.max()
    if bad > 0.1 * ds.n_sequences:
        print(f'\n  WARNING: {bad} of {ds.n_sequences} runs have Pareto k > 0.7, and '
              f'p_loo ({out.p_loo.max():.0f}) exceeds the number of runs.\n'
              f'  PSIS-LOO is unreliable for this model; treat the difference below as '
              f'indicative only\n  and judge the models on the posterior predictive '
              f'checks instead.')

    a, b = loos['rw'], loos['rw_dual']
    # Paired SE of the difference, computed from the pointwise values.
    diff_points = np.asarray(b.loo_i) - np.asarray(a.loo_i)
    d_elpd = float(diff_points.sum())
    d_se = float(np.sqrt(len(diff_points)) * diff_points.std(ddof=1))
    print(f'\nelpd difference (dual - single): {d_elpd:+.1f} +/- {d_se:.1f} (SE)')
    if abs(d_elpd) < 2 * d_se:
        verdict = ('The models fit about equally well: the difference is smaller than '
                   'twice its standard error. Asymmetric learning rates do not buy '
                   'predictive accuracy here.')
    elif d_elpd > 0:
        verdict = 'The dual model predicts held-out runs better.'
    else:
        verdict = 'The single-rate model predicts held-out runs better.'
    print(verdict)

    out['elpd_diff_dual_minus_single'] = d_elpd
    out['elpd_diff_se'] = d_se
    out['verdict'] = verdict
    out.to_csv(OUT, index=False)
    print(f'\nwrote {OUT.relative_to(REPO_ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
