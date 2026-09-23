"""
qc_paper_variables.py — Distributions, correlations, collinearity and reliability

A last look at every variable the manuscript uses, in one place, so nothing
odd reaches a reviewer first. Four checks, each answering a question that
could change how a result is read:

1. **Distributions** — is anything bimodal, floored, ceilinged or driven by a
   handful of points? Prints n, mean, SD, skew and a Shapiro-Wilk test, and
   draws every variable.

2. **Baseline correlations** — what is entangled with what, before any model.
   Age is the paper's organising variable, so its correlates matter: a
   "cognition" effect that is really age, or vice versa, would show here.

3. **Collinearity** — variance inflation for the covariate sets the
   preregistered models actually use. Age and global cognition correlate
   strongly in an ageing sample, and if VIF is high the separate coefficients
   are not interpretable however small their p-values are.

4. **Reliability** — split-half (odd vs even runs within the sham session,
   Spearman-Brown corrected) for each behavioural measure. This bounds
   everything correlational: a measure with a reliability of .5 cannot
   correlate with anything above about .7, so a null result on an unreliable
   measure says little about the construct. It is also the natural check on
   the posterior predictive finding that lose-shift is poorly reproduced.

Usage
-----
    python qc_paper_variables.py
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import (REPO_ROOT, COG_COMPOSITE, rl,
                    AGE_BAND_YOUNG_MAX as BAND_YOUNG,
                    AGE_BAND_OLD_MIN as BAND_OLD)
from paper_style import (WIDTH_2COL, FONT_AXIS_TITLE, FONT_TICK, ACCENT_RED,
                         NEUTRAL_GRAY, REGRESSION_COLOR)
import samples

FIG_DIR = REPO_ROOT / 'data' / 'figures' / 'qc'
OUT = REPO_ROOT / 'derivatives' / 'qc_paper_variables.csv'

GROUPS: Dict[str, List[Tuple[str, str]]] = {
    'Demographics': [('age', 'Age'), ('education_years', 'Education (y)')],
    'Cognition': [(COG_COMPOSITE, 'Global cognition'), ('ef_reduced', 'Executive function'),
                  ('memory_reduced', 'Memory'), ('speed_reduced', 'Speed'),
                  ('kbit_iq', 'KBIT IQ'), ('scd_q', 'SCD-Q')],
    'Baseline behaviour': [('sham_accuracy', 'Accuracy'), ('sham_win_rate', 'Win rate'),
                           ('sham_p_stay_win', 'p(stay|win)'),
                           ('sham_p_shift_lose', 'p(shift|lose)'),
                           ('sham_ttc', 'Trials to criterion'),
                           (rl('sham_alpha'), 'alpha'), (rl('sham_beta'), 'beta')],
    'Stimulation change': [('delta_accuracy', 'd accuracy'), ('delta_win_rate', 'd win rate'),
                           ('delta_p_stay_win', 'd p(stay|win)'),
                           ('delta_p_shift_lose', 'd p(shift|lose)'),
                           ('delta_ttc', 'd TTC'), (rl('delta_alpha'), 'd alpha'),
                           (rl('delta_beta'), 'd beta')],
    'Surveys': [('spsrq_sr', 'SPSRQ reward'), ('spsrq_sp', 'SPSRQ punishment'),
                ('bpsqi_global', 'Sleep (B-PSQI)'), ('ffmq_total', 'FFMQ'),
                ('crt_total', 'CRT')],
    'Physiology and anatomy': [('theta_p95', 'Theta reactivity'), ('mean_magnE', '|E| DLPFC'),
                               ('dist_pial_dlpfc_p1', 'Scalp-cortex distance'),
                               ('layer_skull', 'Skull thickness'), ('csf_charm', 'CSF volume'),
                               ('lh_dlpfc_thickness', 'DLPFC thickness')],
}

# Correlation matrix and collinearity: the variables that enter models.
CORR_VARS = [('age', 'Age'), ('education_years', 'Education'),
             (COG_COMPOSITE, 'Global cog'), ('ef_reduced', 'EF'), ('memory_reduced', 'Memory'),
             ('speed_reduced', 'Speed'), ('kbit_iq', 'KBIT'), ('spsrq_sr', 'SPSRQ-R'),
             ('spsrq_sp', 'SPSRQ-P'), ('theta_p95', 'Theta'),
             ('sham_accuracy', 'Accuracy'), ('sham_p_stay_win', 'p(stay|win)'),
             ('sham_p_shift_lose', 'p(shift|lose)'), (rl('sham_alpha'), 'alpha'),
             (rl('sham_beta'), 'beta'), ('mean_magnE', '|E|')]

MODELS = {
    'H1.1 / H1.1.2 covariates': [COG_COMPOSITE, 'age', 'education_years'],
    'H1.1.1 (adds beta)': [rl('sham_beta'), COG_COMPOSITE, 'age', 'education_years'],
    'H1.2 (adds SPSRQ)': ['spsrq_sr', 'spsrq_sp', COG_COMPOSITE, 'age', 'education_years'],
    'H2.2': ['age', COG_COMPOSITE],
    'theta models': ['theta_p95', 'age', COG_COMPOSITE],
}


def vif(d: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Variance inflation factor per predictor."""
    import statsmodels.api as sm
    m = d[cols].apply(pd.to_numeric, errors='coerce').dropna()
    out = {}
    for c in cols:
        others = [x for x in cols if x != c]
        r2 = sm.OLS(m[c], sm.add_constant(m[others])).fit().rsquared
        out[c] = 1.0 / max(1e-9, 1.0 - r2)
    return pd.Series(out), len(m)


def split_half_reliability(verbose=True) -> pd.DataFrame:
    """
    Odd vs even sham runs, Spearman-Brown corrected.

    Each subject's sham session has two runs under counterbalance, so the
    split is run-level rather than trial-level: it asks whether the measure
    is stable across occasions, which is what a between-subject correlation
    depends on.
    """
    from data_loading import load_all_subjects
    from exclusions import apply_all_exclusions
    from wsls import compute_wsls_h1_h2  # noqa: F401  (kept for parity of definitions)

    data = apply_all_exclusions(load_all_subjects(sample='all', verbose=False),
                                verbose=False)['data_clean']
    sham = data[data.condition == 'sham'].dropna(subset=['choice', 'reward', 'current_good'])
    rows = []
    for sid, g in sham.groupby('subject_id'):
        runs = sorted(g['run'].unique())
        if len(runs) < 2:
            continue
        for half, rs in (('a', runs[0::2]), ('b', runs[1::2])):
            h = g[g.run.isin(rs)]
            if len(h) < 20:
                continue
            correct = (h.choice == h.current_good)
            prev = h.groupby('run').apply(
                lambda x: pd.DataFrame({'stay': x.choice.values[1:] == x.choice.values[:-1],
                                        'won': x.reward.values[:-1] == 1}),
                include_groups=False).reset_index(drop=True)
            rows.append(dict(subject_id=str(sid), half=half,
                             accuracy=correct.mean(),
                             p_stay_win=prev.stay[prev.won].mean() if prev.won.any() else np.nan,
                             p_shift_lose=(~prev.stay[~prev.won]).mean() if (~prev.won).any() else np.nan))
    d = pd.DataFrame(rows).pivot(index='subject_id', columns='half')
    out = []
    for m in ('accuracy', 'p_stay_win', 'p_shift_lose'):
        x = d[m].dropna()
        if len(x) < 10:
            continue
        r = stats.pearsonr(x['a'], x['b'])[0]
        sb = 2 * r / (1 + r)
        out.append(dict(measure=m, n=len(x), split_half_r=r, spearman_brown=sb,
                        max_observable_r=np.sqrt(max(sb, 0))))
    return pd.DataFrame(out)


def main() -> int:
    d = samples.load()
    h1 = samples.frame('H1', d)
    print(f'H1 sample N = {len(h1)}\n')

    # ---- 1. distributions -------------------------------------------------
    rows = []
    panels = [(c, lab, grp) for grp, items in GROUPS.items() for c, lab in items
              if c in h1.columns]
    print(f"{'variable':26}{'n':>4}{'mean':>12}{'SD':>12}{'skew':>8}{'Shapiro p':>11}  flag")
    for c, lab, grp in panels:
        x = pd.to_numeric(h1[c], errors='coerce').dropna()
        if len(x) < 4:
            continue
        sk = float(stats.skew(x)); w, p = stats.shapiro(x)
        med, mad = x.median(), (x - x.median()).abs().median() * 1.4826
        n_out = int((((x - med) / mad).abs() > 3.5).sum()) if mad > 0 else 0
        flags = []
        if abs(sk) > 1: flags.append('skewed')
        if p < .001: flags.append('non-normal')
        if n_out: flags.append(f'{n_out} outlier(s)')
        print(f'  {lab:24}{len(x):>4}{x.mean():>12.4g}{x.std():>12.4g}{sk:>8.2f}{p:>11.3f}  '
              f'{", ".join(flags)}')
        rows.append(dict(group=grp, variable=c, label=lab, n=len(x), mean=x.mean(),
                         sd=x.std(), skew=sk, shapiro_p=p, n_outliers=n_out))

    ncol = 6
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(WIDTH_2COL, WIDTH_2COL * nrow / ncol * 0.85))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.94, bottom=0.05, hspace=0.75, wspace=0.25)
    for ax, (c, lab, grp) in zip(axes.ravel(), panels):
        x = pd.to_numeric(h1[c], errors='coerce').dropna()
        ax.hist(x, bins=14, color=NEUTRAL_GRAY, alpha=0.8, edgecolor='white', linewidth=0.4)
        ax.axvline(x.median(), color=REGRESSION_COLOR, lw=0.9, ls='--')
        sk = float(stats.skew(x))
        ax.set_title(f'{lab}\nn={len(x)}, skew {sk:+.1f}', fontsize=FONT_TICK,
                     color=ACCENT_RED if abs(sk) > 1 else 'black', pad=2)
        ax.tick_params(labelsize=FONT_TICK - 1.5, pad=1)
        ax.set_yticks([])
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    fig.suptitle('Distributions of every variable used in the manuscript (H1 sample)',
                 fontsize=FONT_AXIS_TITLE + 1, y=0.985)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / 'qc_distributions.png', dpi=300, facecolor='white')
    plt.close(fig)

    # Age is the paper's organising variable and it is not normal: check
    # whether that is bimodality from recruiting two age bands, which would
    # make "age" effects partly a group contrast.
    a = pd.to_numeric(h1['age'], errors='coerce').dropna().sort_values()
    gaps = a.diff().dropna()
    big = gaps[gaps > 4]
    print(f'\nAge distribution: range {a.min():.0f}-{a.max():.0f}, median {a.median():.0f}; '
          f'{(a < BAND_YOUNG).sum()} under {BAND_YOUNG}, '
          f'{((a >= BAND_YOUNG) & (a < BAND_OLD)).sum()} {BAND_YOUNG}-{BAND_OLD - 1}, '
          f'{(a >= BAND_OLD).sum()} {BAND_OLD}+')
    if len(big):
        for i, g in big.items():
            lo = a.loc[:i].iloc[-2]
            print(f'  gap of {g:.0f} years between {lo:.0f} and {a.loc[i]:.0f} '
                  f'-- recruitment bands, so age is not uniformly sampled')

    # ---- 2. baseline correlations ----------------------------------------
    cols = [(c, lab) for c, lab in CORR_VARS if c in h1.columns]
    M = h1[[c for c, _ in cols]].apply(pd.to_numeric, errors='coerce')
    C = M.corr()
    fig, ax = plt.subplots(figsize=(WIDTH_2COL * 0.72, WIDTH_2COL * 0.68))
    im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
    labs = [lab for _, lab in cols]
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=45, ha='right', fontsize=FONT_TICK - 1)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=FONT_TICK - 1)
    for i in range(len(labs)):
        for j in range(len(labs)):
            v = C.iloc[i, j]
            if i != j and abs(v) >= 0.3:
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=FONT_TICK - 2.5,
                        color='white' if abs(v) > 0.6 else 'black')
    ax.set_title('Baseline correlations (|r| >= .30 labelled)', fontsize=FONT_AXIS_TITLE, pad=4)
    fig.colorbar(im, ax=ax, fraction=0.045).ax.tick_params(labelsize=FONT_TICK - 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'qc_correlations.png', dpi=300, facecolor='white')
    plt.close(fig)

    print('\nStrongest baseline correlations (|r| >= .40):')
    seen = set()
    for a in C.columns:
        for b in C.columns:
            if a == b or (b, a) in seen:
                continue
            seen.add((a, b))
            r = C.loc[a, b]
            if abs(r) >= 0.40:
                la = dict(cols)[a]; lb = dict(cols)[b]
                n = M[[a, b]].dropna().shape[0]
                p = stats.pearsonr(*M[[a, b]].dropna().T.values)[1]
                print(f'  {la:22} x {lb:22} r = {r:+.2f}  p = {p:.4f}  n = {n}')

    # ---- 2b. is each age effect a gradient or a group difference? ---------
    # Age was recruited in two bands (see AGE_BAND_* in config.py),
    # so a linear age correlation is close to a two-group contrast. An effect
    # that also appears *within* a band is a gradient; one that appears only
    # between bands is a group difference and should be described as such.
    print('\nAge effects: overall, within band, and between bands')
    print(f"  {'measure':26}{'overall r':>11}{'young r':>10}{'old r':>9}"
          f"{'between-band d':>16}")
    young_b, old_b = d[d.age < BAND_YOUNG], d[d.age >= BAND_OLD]
    for col, lab in [(COG_COMPOSITE, 'Global cognition'), ('ef_reduced', 'Executive function'),
                     ('mean_magnE', '|E| DLPFC'), ('dist_pial_dlpfc_p1', 'Scalp-cortex dist'),
                     ('layer_skull', 'Skull thickness'), ('sham_ttc', 'Trials to criterion')]:
        if col not in d.columns:
            continue
        use = d
        if col in ('mean_magnE', 'dist_pial_dlpfc_p1', 'layer_skull'):
            use = d[d['t1_only'].eq(False)]          # the Dose rule
        def r_of(frame):
            m = frame[['age', col]].apply(pd.to_numeric, errors='coerce').dropna()
            return stats.pearsonr(m.age, m[col])[0] if len(m) > 5 else np.nan
        yb = use[use.age < BAND_YOUNG][col].pipe(pd.to_numeric, errors='coerce').dropna()
        ob = use[use.age >= BAND_OLD][col].pipe(pd.to_numeric, errors='coerce').dropna()
        pooled = np.sqrt((yb.var(ddof=1) + ob.var(ddof=1)) / 2)
        dd = (ob.mean() - yb.mean()) / pooled if pooled > 0 else np.nan
        print(f'  {lab:24}{r_of(use):>11.2f}{r_of(use[use.age < BAND_YOUNG]):>10.2f}'
              f'{r_of(use[use.age >= BAND_OLD]):>9.2f}{dd:>16.2f}')
        rows.append(dict(group='age_bands', variable=col, label=lab,
                         n=len(yb) + len(ob), mean=dd))

    # ---- 3. collinearity --------------------------------------------------
    print('\nVariance inflation in the models actually fitted (VIF > 5 is a problem):')
    for name, cols_ in MODELS.items():
        have = [c for c in cols_ if c in h1.columns]
        if len(have) < 2:
            continue
        v, n = vif(h1, have)
        worst = v.idxmax()
        flag = '   <-- collinear' if v.max() > 5 else ''
        print(f'  {name:26} n = {n:2d}  max VIF {v.max():.2f} ({worst}){flag}')
        for c in have:
            rows.append(dict(group='VIF', variable=c, label=name, n=n, mean=v[c]))

    # ---- 4. reliability ---------------------------------------------------
    print('\nSplit-half reliability of the baseline behavioural measures')
    print('  (odd vs even sham runs, Spearman-Brown corrected; the last column is')
    print('   the largest correlation the measure could show with anything else)')
    rel = split_half_reliability()
    for r in rel.itertuples():
        flag = '   <-- unreliable' if r.spearman_brown < 0.5 else ''
        print(f'  {r.measure:16} n = {r.n:2d}  r = {r.split_half_r:+.2f}  '
              f'SB = {r.spearman_brown:+.2f}  max r = {r.max_observable_r:.2f}{flag}')
        rows.append(dict(group='reliability', variable=r.measure, label='split-half',
                         n=r.n, mean=r.spearman_brown))

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'\nwrote {OUT.relative_to(REPO_ROOT)}')
    print(f'wrote {(FIG_DIR / "qc_distributions.png").relative_to(REPO_ROOT)}')
    print(f'wrote {(FIG_DIR / "qc_correlations.png").relative_to(REPO_ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
