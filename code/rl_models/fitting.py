"""
Model fitting (MCMC) and comparison (WAIC, LOO) for hierarchical RL models.

Provides a clean interface for running NUTS inference, extracting posteriors,
computing diagnostics, and comparing models via information criteria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import arviz as az
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, log_likelihood, Predictive

from .models import ModelType, MODEL_FUNCTIONS, MODEL_SPECS
from .data_loader import BanditDataset


@dataclass
class FitResult:
    """Container for MCMC results + diagnostics."""

    model_type: ModelType
    mcmc: MCMC
    idata: az.InferenceData
    dataset: BanditDataset
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def posterior(self) -> az.InferenceData:
        return self.idata.posterior

    def summary(self, var_names: Optional[list[str]] = None) -> Any:
        """ArviZ summary table for key parameters."""
        if var_names is None:
            spec = MODEL_SPECS[self.model_type]
            # Group-level params
            var_names = [f"mu_{p}" for p in spec.param_names] + [
                f"sigma_{p}" for p in spec.param_names
            ]
        return az.summary(self.idata, var_names=var_names)


def fit_model(
    dataset: BanditDataset,
    model_type: ModelType = ModelType.RW,
    *,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    seed: int = 42,
    target_accept_prob: float = 0.9,
    use_conditions: bool = False,
) -> FitResult:
    """Run NUTS MCMC for a hierarchical RL model.

    Parameters
    ----------
    dataset : BanditDataset from data_loader.
    model_type : which RL model variant to fit.
    num_warmup, num_samples, num_chains : MCMC settings.
    seed : PRNG seed.
    target_accept_prob : NUTS target acceptance rate. 0.9 is conservative
        and appropriate for hierarchical models with funnel geometry.
    use_conditions : if True and model supports it, estimate condition-level
        group means (e.g., separate mu_alpha for active vs. sham).

    Returns
    -------
    FitResult with MCMC object, ArviZ InferenceData, and diagnostics.
    """
    model_fn = MODEL_FUNCTIONS[model_type]
    rng_key = jr.PRNGKey(seed)

    # Build model kwargs
    model_kwargs = dict(
        choices=dataset.choices,
        rewards=dataset.rewards,
        subject_ids=dataset.subject_ids,
        n_subjects=dataset.n_subjects,
    )

    # Only the RW model currently supports condition-level estimation
    if use_conditions and model_type == ModelType.RW:
        model_kwargs["condition_ids"] = dataset.condition_ids
        model_kwargs["n_conditions"] = dataset.n_conditions

    kernel = NUTS(model_fn, target_accept_prob=target_accept_prob)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )

    print(f"\nFitting {model_type.value} model...")
    print(f"  {dataset.n_subjects} subjects, {len(dataset.choices)} total trials")
    print(f"  {num_chains} chains × {num_samples} samples (+ {num_warmup} warmup)")
    print()

    mcmc.run(rng_key, **model_kwargs)

    # Convert to ArviZ InferenceData
    idata = az.from_numpyro(mcmc)

    # Compute diagnostics
    diagnostics = _compute_diagnostics(mcmc, idata)

    return FitResult(
        model_type=model_type,
        mcmc=mcmc,
        idata=idata,
        dataset=dataset,
        diagnostics=diagnostics,
    )


def _compute_diagnostics(mcmc: MCMC, idata: az.InferenceData) -> Dict[str, Any]:
    """Compute standard MCMC diagnostics."""
    diag = {}

    # Divergences
    if hasattr(mcmc, "get_extra_fields"):
        extra = mcmc.get_extra_fields()
        if "diverging" in extra:
            n_div = int(np.sum(extra["diverging"]))
            diag["n_divergences"] = n_div

    # R-hat and ESS from ArviZ
    try:
        rhat = az.rhat(idata)
        ess = az.ess(idata)

        # Get worst R-hat across all parameters
        rhat_vals = []
        for var in rhat.data_vars:
            vals = rhat[var].values
            if np.isfinite(vals).any():
                rhat_vals.append(np.nanmax(vals))
        diag["max_rhat"] = float(max(rhat_vals)) if rhat_vals else float("nan")

        # Get minimum ESS
        ess_vals = []
        for var in ess.data_vars:
            vals = ess[var].values
            if np.isfinite(vals).any():
                ess_vals.append(np.nanmin(vals))
        diag["min_ess"] = float(min(ess_vals)) if ess_vals else float("nan")
    except Exception:
        diag["max_rhat"] = float("nan")
        diag["min_ess"] = float("nan")

    return diag


def print_diagnostics(result: FitResult) -> None:
    """Print a human-readable diagnostics summary."""
    d = result.diagnostics
    print(f"\n--- Diagnostics for {result.model_type.value} ---")
    print(f"  Divergences: {d.get('n_divergences', '?')}")
    print(f"  Max R-hat:   {d.get('max_rhat', '?'):.4f}")
    print(f"  Min ESS:     {d.get('min_ess', '?'):.0f}")

    rhat_ok = d.get("max_rhat", 2.0) < 1.05
    ess_ok = d.get("min_ess", 0) > 400
    div_ok = d.get("n_divergences", 999) == 0

    if rhat_ok and ess_ok and div_ok:
        print("  Status:      CONVERGED")
    else:
        issues = []
        if not rhat_ok:
            issues.append("R-hat > 1.05")
        if not ess_ok:
            issues.append("ESS < 400")
        if not div_ok:
            issues.append(f"{d.get('n_divergences', '?')} divergences")
        print(f"  Status:      WARNING — {', '.join(issues)}")
    print()


def compare_models(
    results: Dict[str, FitResult],
    *,
    ic: str = "waic",
) -> az.InferenceData:
    """Compare fitted models using information criteria.

    Parameters
    ----------
    results : dict mapping model names to FitResults.
    ic : "waic" or "loo" (PSIS-LOO-CV).

    Returns
    -------
    ArviZ comparison DataFrame.
    """
    idata_dict = {name: r.idata for name, r in results.items()}

    if ic == "waic":
        comparison = az.compare(idata_dict, ic="waic")
    else:
        comparison = az.compare(idata_dict, ic="loo")

    return comparison


def extract_subject_posteriors(
    result: FitResult,
    param_name: str,
) -> Dict[str, np.ndarray]:
    """Extract posterior samples for a parameter, keyed by subject ID.

    Parameters
    ----------
    result : FitResult from fit_model.
    param_name : name of a transformed parameter (e.g., "alpha", "beta").

    Returns
    -------
    Dict mapping subject ID strings to (n_samples,) arrays of posterior draws.
    """
    # Get posterior array: shape (n_chains, n_samples, n_subjects)
    post = result.idata.posterior[param_name].values
    # Flatten chains: (n_chains * n_samples, n_subjects)
    flat = post.reshape(-1, post.shape[-1])

    inv_map = {v: k for k, v in result.dataset.subject_id_map.items()}
    return {inv_map[i]: flat[:, i] for i in range(flat.shape[1])}
