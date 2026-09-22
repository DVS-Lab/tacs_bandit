"""
ppc_within.py — Posterior predictive checks for the within-subject hierarchical RW

Does the fitted model actually generate behaviour like the participants'? The
question is not decorative here: even under the hierarchical fit, 28% of
subjects have alpha >= .95, meaning they replace their value estimate with the
last outcome. That is win-stay/lose-shift rather than incremental learning, and
if RW is the wrong model for a quarter of the sample it should show up as a
failure to reproduce their behaviour.

Method. For each of `--draws` posterior draws, every run is simulated with that
draw's own run-level parameters (`alpha_seq`, `beta_seq`, the same quantities
the likelihood used), starting from V = [.5, .5] and applying the model's own
update rule. Choices are sampled from the softmax; rewards come from the task's
real contingencies (P(reward | good option) = .75, P(reward | bad) = .25, both
measured from the data below) against the *actual* schedule of which option was
good on each trial. So a simulated agent that chooses differently from the
participant gets the reward the task would have given it -- the counterfactual
that makes this a real predictive check rather than a replay.

Four statistics, chosen because they are what the model is used to argue about:

  accuracy          proportion of trials on the currently-good option
  p(stay | win)     choice persistence after reward
  p(shift | lose)   switching after non-reward -- the H1 measure
  trials to criterion   trials after a reversal to 3 consecutive correct

Two ways of reading the result, both reported:

  group level    where the observed group mean falls in the predictive
                 distribution (a posterior predictive p near 0 or 1 means the
                 model cannot produce the group's behaviour)
  subject level  the share of subject x condition cells whose observed value
                 falls inside the 95% predictive interval. 95% is the target;
                 well below it means the model misses individual differences

`rl_models/ppc.py` is not this: it targets the older between-subject model and
its own loader, which applied no exclusions.

Usage
-----
    python ppc_within.py
    python ppc_within.py --draws 1000 --seed 7
"""

from __future__ import annotations

import argparse
import pickle
import sys
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import REPO_ROOT
from paper_style import (WIDTH_2COL, FONT_AXIS_TITLE, FONT_TICK,
                         FONT_PANEL_LABEL, REGRESSION_COLOR, NEUTRAL_GRAY,
                         SHAM_COLOR, ACTIVE_COLOR)

RL_DIR = REPO_ROOT / 'derivatives' / 'rl_models'
FIG_DIR = REPO_ROOT / 'data' / 'figures' / 'qc'
CRITERION = 3          # consecutive correct, as in trials-to-criterion elsewhere

STATS = [('accuracy', 'Accuracy'), ('p_stay_win', 'p(stay | win)'),
         ('p_shift_lose', 'p(shift | lose)'), ('ttc', 'Trials to criterion')]


def load_runs():
    """
    The same run sequences the fit used, plus which option was good per trial.

    Selection mirrors rl_models.data_prep.build_run_dataset exactly: the H2
    dataset after exclusions, sorted by (subject_id, run), trials with a choice
    and a reward, runs of at least 10 trials. Sequence order must match, since
    alpha_seq / beta_seq are indexed by it.
    """
    from data_loading import load_all_subjects
    from exclusions import apply_all_exclusions
    from rl_models.data_prep import CONDITION_CODES, FIRST_BLOCK_RUNS, SECOND_BLOCK_RUNS

    data = apply_all_exclusions(load_all_subjects(sample='all', verbose=False),
                                verbose=False)['data_h2']
    df = data[data['condition'].isin(('sham', 'active'))].dropna(
        subset=['choice', 'reward', 'current_good'])

    runs = []
    for (sub_id, run), g in df.groupby(['subject_id', 'run'], sort=True):
        if len(g) < 10 or str(g['condition'].iloc[0]) not in CONDITION_CODES:
            continue
        run = int(run)
        if run not in set(FIRST_BLOCK_RUNS) | set(SECOND_BLOCK_RUNS):
            continue
        runs.append(dict(
            subject_id=str(sub_id), run=run, condition=str(g['condition'].iloc[0]),
            choices=(g['choice'].to_numpy() - 1).astype(np.int8),
            rewards=g['reward'].to_numpy().astype(np.int8),
            good=(g['current_good'].to_numpy() - 1).astype(np.int8)))
    return runs


def pack(runs):
    """Runs into padded (n_runs, max_trials) arrays."""
    n, T = len(runs), max(len(r['choices']) for r in runs)
    ch = np.zeros((n, T), np.int8); rw = np.zeros((n, T), np.int8)
    gd = np.zeros((n, T), np.int8); mk = np.zeros((n, T), bool)
    for i, r in enumerate(runs):
        t = len(r['choices'])
        ch[i, :t], rw[i, :t], gd[i, :t], mk[i, :t] = r['choices'], r['rewards'], r['good'], True
    return ch, rw, gd, mk


def statistics(ch, rw, gd, mk) -> Dict[str, np.ndarray]:
    """The four summary statistics, per run."""
    n, T = ch.shape
    correct = (ch == gd) & mk
    acc = correct.sum(1) / mk.sum(1)

    prev_ok = mk[:, 1:] & mk[:, :-1]
    stay = (ch[:, 1:] == ch[:, :-1]) & prev_ok
    won, lost = (rw[:, :-1] == 1) & prev_ok, (rw[:, :-1] == 0) & prev_ok
    with np.errstate(invalid='ignore', divide='ignore'):
        p_stay = np.where(won.sum(1) > 0, (stay & won).sum(1) / np.maximum(won.sum(1), 1), np.nan)
        p_shift = np.where(lost.sum(1) > 0, (~stay & lost).sum(1) / np.maximum(lost.sum(1), 1), np.nan)

    # Trials to criterion: after each reversal, how many trials until CRITERION
    # consecutive correct. Runs that never reach it contribute the trials that
    # remained, which is conservative and identical for observed and simulated.
    ttc = np.full(n, np.nan)
    for i in range(n):
        t_end = int(mk[i].sum())
        rev = [t for t in range(1, t_end) if gd[i, t] != gd[i, t - 1]]
        vals = []
        for start in rev:
            run_len, hit = 0, None
            for t in range(start, t_end):
                run_len = run_len + 1 if correct[i, t] else 0
                if run_len >= CRITERION:
                    hit = t - start - CRITERION + 2
                    break
            vals.append(hit if hit is not None else t_end - start)
        if vals:
            ttc[i] = float(np.mean(vals))
    return dict(accuracy=acc, p_stay_win=p_stay, p_shift_lose=p_shift, ttc=ttc)


def simulate(alpha, beta, gd, mk, p_good, p_bad, rng, alpha_neg=None):
    """
    One posterior draw: choices and rewards for every run.

    `alpha_neg` given makes this the dual model: `alpha` applies to
    better-than-expected outcomes and `alpha_neg` to worse-than-expected ones,
    matching rl_models.models._rw_dual_update.
    """
    n, T = gd.shape
    V = np.full((n, 2), 0.5)
    ch = np.zeros((n, T), np.int8); rw = np.zeros((n, T), np.int8)
    idx = np.arange(n)
    for t in range(T):
        p1 = 1.0 / (1.0 + np.exp(-beta * (V[:, 1] - V[:, 0])))
        c = (rng.random(n) < p1).astype(np.int8)
        good_choice = c == gd[:, t]
        r = (rng.random(n) < np.where(good_choice, p_good, p_bad)).astype(np.int8)
        live = mk[:, t]
        ch[:, t], rw[:, t] = np.where(live, c, 0), np.where(live, r, 0)
        pe = r - V[idx, c]
        rate = alpha if alpha_neg is None else np.where(pe >= 0, alpha, alpha_neg)
        V[idx, c] += np.where(live, rate, 0.0) * pe
    return ch, rw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--model', default='rw', choices=['rw', 'rw_dual'])
    ap.add_argument('--draws', type=int, default=500)
    ap.add_argument('--seed', type=int, default=20260922)
    args = ap.parse_args(argv)

    runs = load_runs()
    ch, rw, gd, mk = pack(runs)
    meta = pd.DataFrame([{k: r[k] for k in ('subject_id', 'run', 'condition')} for r in runs])
    print(f'{len(runs)} runs from {meta.subject_id.nunique()} subjects')

    # The task's contingencies, measured rather than assumed.
    valid = mk
    good_choice = (ch == gd) & valid
    p_good = rw[good_choice].mean()
    p_bad = rw[valid & ~good_choice].mean()
    print(f'reward probabilities from the data: good {p_good:.3f}, bad {p_bad:.3f}')

    idata_path = RL_DIR / f'{args.model}_within_all_idata.pkl'
    out_csv = REPO_ROOT / 'derivatives' / f'ppc_within_{args.model}_summary.csv'
    with open(idata_path, 'rb') as f:
        blob = pickle.load(f)
    # run_within_fit pickles {'idata': InferenceData, 'diagnostics': {...}}.
    idata = blob['idata'] if isinstance(blob, dict) else blob
    diag = blob.get('diagnostics', {}) if isinstance(blob, dict) else {}
    if diag:
        print(f"fit diagnostics: max R-hat {diag.get('max_r_hat'):.3f}, "
              f"min ESS {diag.get('min_ess_bulk'):.0f}, "
              f"{diag.get('n_divergences')} divergences")
    post = idata.posterior
    dual = args.model == 'rw_dual'
    alpha = post['alpha_pos_seq' if dual else 'alpha_seq'].values.reshape(-1, len(runs))
    alpha_neg = post['alpha_neg_seq'].values.reshape(-1, len(runs)) if dual else None
    beta = post['beta_seq'].values.reshape(-1, len(runs))
    if alpha.shape[1] != len(runs):
        raise SystemExit(f'posterior has {alpha.shape[1]} sequences, data has {len(runs)}: '
                         'the fit and this dataset disagree. Refit or check the sample.')
    rng = np.random.default_rng(args.seed)
    draw_idx = rng.choice(alpha.shape[0], size=min(args.draws, alpha.shape[0]), replace=False)
    print(f'{alpha.shape[0]} posterior draws available; simulating {len(draw_idx)}')

    observed = statistics(ch, rw, gd, mk)
    sim = {k: np.empty((len(draw_idx), len(runs))) for k, _ in STATS}
    for j, d in enumerate(draw_idx):
        sch, srw = simulate(alpha[d], beta[d], gd, mk, p_good, p_bad, rng,
                            alpha_neg=alpha_neg[d] if dual else None)
        s = statistics(sch, srw, gd, mk)
        for k, _ in STATS:
            sim[k][j] = s[k]

    # Aggregate runs to subject x condition, for observed and each draw.
    key = meta.subject_id + '|' + meta.condition
    cells = pd.Index(sorted(key.unique()))
    G = np.stack([(key == c).values for c in cells])          # (n_cells, n_runs)

    def by_cell(v):
        w = G * np.isfinite(v)[None, :]
        return np.where(w.sum(1) > 0, (w * np.nan_to_num(v)[None, :]).sum(1) / np.maximum(w.sum(1), 1), np.nan)

    rows = []
    print(f"\n{'statistic':22}{'observed':>10}{'predicted (95% PI)':>26}{'post. pred. p':>15}"
          f"{'in PI':>11}{'r(obs,pred)':>10}")
    for k, label in STATS:
        obs_cell = by_cell(observed[k])
        sim_cell = np.stack([by_cell(sim[k][j]) for j in range(len(draw_idx))])
        lo, hi = np.nanpercentile(sim_cell, [2.5, 97.5], axis=0)
        inside = np.nanmean((obs_cell >= lo) & (obs_cell <= hi))
        obs_mean = np.nanmean(obs_cell)
        sim_mean = np.nanmean(sim_cell, axis=1)
        p = float(np.mean(sim_mean >= obs_mean))
        p_two = 2 * min(p, 1 - p)
        # How well individual differences are tracked, not just the group mean:
        # a model can hit the average and still be blind to who does what.
        pred_med = np.nanmedian(sim_cell, axis=0)
        ok = np.isfinite(obs_cell) & np.isfinite(pred_med)
        r_ind = float(np.corrcoef(obs_cell[ok], pred_med[ok])[0, 1])
        flag = '   <-- model misses this' if (p_two < .05 or inside < 0.80 or r_ind < 0.5) else ''
        print(f'  {label:20}{obs_mean:>10.3f}'
              f'{f"{np.mean(sim_mean):.3f} [{np.percentile(sim_mean, 2.5):.3f}, {np.percentile(sim_mean, 97.5):.3f}]":>26}'
              f'{p_two:>15.3f}{inside:>11.0%}{r_ind:>10.2f}{flag}')
        for i, c in enumerate(cells):
            sid, cond = c.split('|')
            rows.append(dict(statistic=k, subject_id=sid, condition=cond,
                             observed=obs_cell[i], pred_median=np.nanmedian(sim_cell[:, i]),
                             pred_lo=lo[i], pred_hi=hi[i],
                             inside=bool(lo[i] <= obs_cell[i] <= hi[i])))

    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    # --- figure: observed vs predicted per subject x condition ---------------
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH_2COL, WIDTH_2COL * 0.30))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.86, bottom=0.17, wspace=0.30)
    for ax, (k, label) in zip(axes, STATS):
        d = out[out.statistic == k]
        for cond, colour in [('sham', SHAM_COLOR), ('active', ACTIVE_COLOR)]:
            s = d[d.condition == cond]
            ax.vlines(s.observed, s.pred_lo, s.pred_hi, color=colour, alpha=0.35, lw=0.7)
            ax.scatter(s.observed, s.pred_median, s=9, color=colour, alpha=0.85,
                       edgecolors='white', linewidths=0.3, label=cond, zorder=3)
        lim = [np.nanmin([d.observed.min(), d.pred_lo.min()]),
               np.nanmax([d.observed.max(), d.pred_hi.max()])]
        pad = 0.04 * (lim[1] - lim[0])
        ax.plot([lim[0] - pad, lim[1] + pad], [lim[0] - pad, lim[1] + pad],
                color=REGRESSION_COLOR, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(lim[0] - pad, lim[1] + pad); ax.set_ylim(lim[0] - pad, lim[1] + pad)
        ok = d.observed.notna() & d.pred_median.notna()
        r_ind = np.corrcoef(d.observed[ok], d.pred_median[ok])[0, 1]
        ax.set_title(f'{label}\n{d.inside.mean():.0%} within 95% PI   r = {r_ind:.2f}',
                     fontsize=FONT_AXIS_TITLE, pad=3)
        ax.set_xlabel('observed', fontsize=FONT_AXIS_TITLE, labelpad=1.5)
        ax.tick_params(labelsize=FONT_TICK, pad=1.2)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel('predicted (median, 95% PI)', fontsize=FONT_AXIS_TITLE, labelpad=2)
    axes[0].legend(fontsize=FONT_TICK - 1, frameon=False, loc='upper left')
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f'qc_ppc_within_{args.model}.png'
    fig.savefig(png, dpi=300, facecolor='white')
    plt.close(fig)
    print(f'\nwrote {out_csv.relative_to(REPO_ROOT)}\nwrote {png.relative_to(REPO_ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
