# Hierarchical Bayesian RL Models for Two-Armed Bandit Data

Fits reinforcement learning models to trial-level choice data from a probabilistic reversal learning (two-armed bandit) task with tACS stimulation conditions. Uses NumPyro (JAX) for Hamiltonian Monte Carlo inference with non-centered hierarchical parameterization.

## Models

| Model | Parameters | Description |
|-------|-----------|-------------|
| **RW** | α, β | Standard Rescorla-Wagner: single learning rate + softmax inverse temperature |
| **RW_dual** | α⁺, α⁻, β | Separate learning rates for positive and negative prediction errors |
| **PH** | α₀, η, κ, β | Pearce-Hall hybrid: dynamic associability modulates effective learning rate |

All models support hierarchical estimation with group-level priors (μ, σ) over individual-level parameters. The RW model additionally supports condition-level group means for testing tACS effects (e.g., μ_α for active vs. sham vs. baseline).

## Quick Start

```bash
# Install dependencies
pip install numpyro jax jaxlib arviz pandas

# Run from the code/ directory (parent of rl_models/)
cd code

# Run smoke tests
python -m rl_models.test_smoke

# Run parameter recovery test
python -m rl_models.test_recovery

# Fit RW model to real data
python -m rl_models.run_fit --data-dir ../data --model RW --num-chains 4

# Fit with condition-level group means (active vs. sham)
python -m rl_models.run_fit --data-dir ../data --model RW --conditions

# Fit all models for comparison
python -m rl_models.run_fit --data-dir ../data --model all

# Only fit to stimulation runs (2, 3, 6, 7)
python -m rl_models.run_fit --data-dir ../data --model RW --runs 2 3 6 7

# Parameter recovery on synthetic data
python -m rl_models.run_fit --simulate --model RW
```

## Data Format

Expects CSV files produced by `bandit_main.py` or `bandit_main_theta.py` with at minimum:

- `choice`: 1 or 2 (slot machine chosen)
- `reward`: True/False or 1/0
- `trial_num`: trial index within run
- `run`: run number (1-8)
- `stim_condition`: "active", "sham", or "baseline" (optional)

Files are discovered via glob pattern under the data directory: `**/sub-*_ses-*_task-bandit_*.csv`

## Architecture

```
code/
├── bandit_main.py
├── bandit_main_theta.py
├── ...
└── rl_models/
    ├── __init__.py
    ├── models.py          # Model definitions (JAX likelihood functions + NumPyro models)
    ├── data_loader.py     # CSV loading, preprocessing, choice remapping (1/2 → 0/1)
    ├── fitting.py         # MCMC runner, diagnostics, model comparison
    ├── simulate.py        # Synthetic data generation for recovery tests
    ├── run_fit.py         # CLI entry point (python -m rl_models.run_fit)
    ├── test_smoke.py      # Smoke tests (python -m rl_models.test_smoke)
    └── test_recovery.py   # Full parameter recovery test
```

## Output

Results are saved to `results/` (configurable via `--output-dir`):

- `{model}_idata.nc` — Full ArviZ InferenceData (posteriors, diagnostics)
- `{model}_summary.csv` — Group-level parameter summary table
- `{model}_individual_params.json` — Per-subject posterior statistics
- `model_comparison.csv` — WAIC/LOO comparison (when fitting multiple models)

## Technical Notes

**Non-centered parameterization**: Individual parameters are sampled as `μ + σ * offset` where offset ~ N(0,1). This avoids the funnel geometry that causes divergences in centered hierarchical models.

**Parameter transforms**: Learning rates use sigmoid (bounded to [0,1]), inverse temperature uses softplus (bounded to [0,∞)). Group-level means (μ) are estimated on the unconstrained scale; transform before interpretation.

**Condition-level model**: When `--conditions` is used with the RW model, separate group means (μ_α, μ_β) are estimated per condition. Individual parameters are still partially pooled toward their condition-specific group mean.

## Next Steps

- [ ] Pointwise log-likelihood for WAIC/LOO-CV model comparison
- [ ] Posterior predictive checks (simulate data from posterior, compare to observed)
- [ ] Trial-level value trajectories from posterior (for neural correlates)
- [ ] Visualization module (trace plots, posterior densities, recovery scatter)
- [ ] Integration with tACS condition analysis (contrast posteriors across conditions)
