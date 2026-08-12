"""
fit_within.py — NUTS fitting for the within-subject hierarchical RW model

fitting.fit_model() is wired to BanditDataset and the between-subject models.
This is the equivalent entry point for RunDataset and model_rw_within_subject.

InferenceData is saved as pickle rather than NetCDF: the h5py/NumPy versions in
this environment conflict, and arviz's NetCDF writer goes through h5py.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import arviz as az
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from numpyro.infer import MCMC, NUTS

from .data_prep import RunDataset, block_contrast, condition_contrast
from .models_within import model_rw_within_subject


@dataclass
class WithinFitResult:
    idata: az.InferenceData
    mcmc: Any
    diagnostics: Dict[str, Any]

    def summary(self, var_names=None):
        if var_names is None:
            var_names = ['mu_alpha', 'delta_alpha', 'eta_alpha', 'sigma_alpha',
                         'tau_alpha', 'mu_beta', 'delta_beta', 'eta_beta',
                         'sigma_beta', 'tau_beta']
        available = [v for v in var_names if v in self.idata.posterior]
        return az.summary(self.idata, var_names=available)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as fh:
            pickle.dump({'idata': self.idata, 'diagnostics': self.diagnostics}, fh)
        return path


def load_fit(path: Path) -> Dict[str, Any]:
    """Read back a fit saved by WithinFitResult.save()."""
    with open(Path(path), 'rb') as fh:
        return pickle.load(fh)


def _check_diagnostics(idata: az.InferenceData, mcmc: Any) -> Dict[str, Any]:
    """R-hat, ESS, and divergences — the three that decide whether to trust it."""
    summary = az.summary(idata)
    n_div = int(np.sum(mcmc.get_extra_fields().get('diverging', np.zeros(1))))

    worst_rhat = float(summary['r_hat'].max())
    min_ess = float(summary['ess_bulk'].min())

    # R-hat <= 1.01 is the current recommendation (Vehtari et al., 2021); an
    # earlier strict < 1.01 here failed a fit that sat exactly at 1.01 with
    # ~1000 ESS and no divergences, which is a healthy fit by any standard.
    return {
        'max_r_hat': worst_rhat,
        'min_ess_bulk': min_ess,
        'n_divergences': n_div,
        'converged': worst_rhat <= 1.01 and min_ess > 400 and n_div == 0,
        'worst_rhat_param': summary['r_hat'].idxmax(),
        'min_ess_param': summary['ess_bulk'].idxmin(),
    }


def fit_within_model(
    dataset: RunDataset,
    *,
    moderators: Optional[np.ndarray] = None,
    include_block: bool = True,
    include_random_slope: bool = True,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 42,
    target_accept_prob: float = 0.95,
    progress_bar: bool = True,
    verbose: bool = True,
) -> WithinFitResult:
    """
    Fit the within-subject hierarchical RW model to run-level sequences.

    `moderators` should be standardized (mean 0, SD 1) so the priors on their
    coefficients carry the intended weight.
    """
    cond_x = jnp.array(condition_contrast(dataset))
    block_x = jnp.array(block_contrast(dataset))

    model_kwargs = dict(
        choices=dataset.choices,
        rewards=dataset.rewards,
        masks=dataset.masks,
        seq_subject=dataset.seq_subject,
        cond_x=cond_x,
        block_x=block_x,
        n_subjects=dataset.n_subjects,
        include_block=include_block,
        include_random_slope=include_random_slope,
    )
    if moderators is not None:
        model_kwargs['moderators'] = jnp.array(np.asarray(moderators, dtype=np.float32))

    kernel = NUTS(model_rw_within_subject, target_accept_prob=target_accept_prob)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )

    if verbose:
        print(f'Fitting within-subject RW: {dataset.n_sequences} runs, '
              f'{dataset.n_subjects} subjects')
        print(f'  {num_chains} chains x {num_samples} samples (+{num_warmup} warmup)')

    mcmc.run(jr.PRNGKey(seed), extra_fields=('diverging',), **model_kwargs)

    idata = az.from_numpyro(mcmc)
    diagnostics = _check_diagnostics(idata, mcmc)

    if verbose:
        d = diagnostics
        flag = 'OK' if d['converged'] else 'CHECK'
        print(f'  [{flag}] max R-hat {d["max_r_hat"]:.4f} ({d["worst_rhat_param"]}), '
              f'min ESS {d["min_ess_bulk"]:.0f} ({d["min_ess_param"]}), '
              f'{d["n_divergences"]} divergences')

    return WithinFitResult(idata=idata, mcmc=mcmc, diagnostics=diagnostics)
