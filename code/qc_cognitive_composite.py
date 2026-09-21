"""
qc_cognitive_composite.py — Is `global_reduced` a defensible measure?

A validation report for the cognitive composite, not a manuscript figure. Two
pages, written to data/figures/qc/:

  page 1  what the measure is made of -- distributions of every constituent,
          the three domain scores, the composite itself, and exactly which
          subject is missing which input
  page 2  whether it behaves -- age relationships, internal structure,
          convergent validity against an independent IQ measure, and the
          comparison against the legacy composite it replaces

Why this exists. The legacy `global_composite` averaged whatever measures a
subject happened to have. Digit Span, BVMT and Trails were administered only
from roughly age 56 (24-28 of the 29 subjects aged 56+, and 1 of the 36 below
it), so an older subject's domain score was a mean of two z-scores where a
younger subject's was a single z-score. A mean of two z-scores is mechanically
less variable than one, and that difference tracked age at r = +.90 for
memory. `global_reduced` uses the same six measures for everyone.

Usage
-----
    python qc_cognitive_composite.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import REPO_ROOT
from paper_style import (WIDTH_2COL, FONT_AXIS_TITLE, FONT_TICK,
                         FONT_PANEL_LABEL, AGE_YOUNG, AGE_OLD,
                         REGRESSION_COLOR, NEUTRAL_GRAY, ACCENT_RED)

FIG_DIR = REPO_ROOT / 'data' / 'figures' / 'qc'

RAW = ['flanker_score', 'running_dots_score', 'set_shifting_score',
       'hvlt_total', 'salthouse_letter', 'salthouse_pattern']
RAW_LABEL = {'flanker_score': 'Flanker', 'running_dots_score': 'Running Dots',
             'set_shifting_score': 'Set Shifting', 'hvlt_total': 'HVLT',
             'salthouse_letter': 'Salthouse Letter',
             'salthouse_pattern': 'Salthouse Pattern'}
DOMAIN_OF = {'flanker_score': 'Attention', 'running_dots_score': 'Attention',
             'set_shifting_score': 'Attention', 'hvlt_total': 'Memory',
             'salthouse_letter': 'Speed', 'salthouse_pattern': 'Speed'}
COMP = ['attention_reduced', 'memory_reduced', 'speed_reduced', 'global_reduced']
COMP_LABEL = {'attention_reduced': 'Attention', 'memory_reduced': 'Memory',
              'speed_reduced': 'Speed', 'global_reduced': 'Global composite'}
DOMAIN_COLOR = {'Attention': '#1565C0', 'Memory': '#2E7D32', 'Speed': '#E64A19',
                'Global composite': '#4A148C'}


def load() -> pd.DataFrame:
    d = pd.read_csv(REPO_ROOT / 'data' / 'master_subject_data.csv',
                    dtype={'subject_id': str})
    for c in ['age', 'education_years', 'kbit_iq'] + RAW + COMP:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    return d


def _clean(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=FONT_TICK, pad=1.5)


def hist(ax, vals, title, color, n_total):
    v = vals.dropna()
    ax.hist(v, bins=12, color=color, alpha=0.75, edgecolor='white', linewidth=0.5)
    ax.axvline(v.mean(), color=REGRESSION_COLOR, lw=1.0, ls='--')
    ax.set_title(f'{title}  (n={len(v)}/{n_total})', fontsize=FONT_AXIS_TITLE, pad=2.5)
    # Shapiro-Wilk: a composite with a badly skewed input is worth knowing about
    # before it goes into a linear model as a covariate.
    if len(v) > 3:
        w, p = stats.shapiro(v)
        txt = f'W={w:.2f}, p={p:.3f}' if p >= .001 else f'W={w:.2f}, p<.001'
        ax.text(0.97, 0.94, txt, transform=ax.transAxes, ha='right', va='top',
                fontsize=FONT_TICK - 1.5,
                color=ACCENT_RED if p < .05 else NEUTRAL_GRAY,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.8))
    _clean(ax)
    ax.set_yticks([])


def age_scatter(ax, x, y, ylabel, color):
    m = pd.concat([x, y], axis=1).dropna()
    if len(m) < 4:
        ax.set_visible(False)
        return
    xs, ys = m.iloc[:, 0].values, m.iloc[:, 1].values
    ax.scatter(xs, ys, s=11, c=color, alpha=0.7, edgecolors='white', linewidths=0.3)
    sl, ic, r, p, _ = stats.linregress(xs, ys)
    g = np.linspace(xs.min(), xs.max(), 100)
    ax.plot(g, ic + sl * g, color=REGRESSION_COLOR, lw=1.0,
            ls='-' if p < .05 else (0, (3, 2)))
    ax.text(0.03, 0.05, f'r={r:+.2f}, p={p:.3f}' if p >= .001 else f'r={r:+.2f}, p<.001',
            transform=ax.transAxes, fontsize=FONT_TICK - 0.5,
            color=REGRESSION_COLOR if p < .05 else NEUTRAL_GRAY)
    ax.set_ylabel(ylabel, fontsize=FONT_AXIS_TITLE, labelpad=1.5)
    ax.set_xlabel('Age (years)', fontsize=FONT_AXIS_TITLE, labelpad=1.5)
    _clean(ax)


def page1(d: pd.DataFrame) -> Path:
    n = len(d)
    fig = plt.figure(figsize=(WIDTH_2COL, WIDTH_2COL * 1.05))
    fig.suptitle('Cognitive composite: what it is made of', y=0.982,
                 fontsize=FONT_PANEL_LABEL, fontweight='bold')

    for i, c in enumerate(RAW):
        ax = fig.add_axes([0.075 + (i % 3) * 0.315, 0.818 - (i // 3) * 0.158,
                           0.235, 0.100])
        hist(ax, d[c], RAW_LABEL[c], DOMAIN_COLOR[DOMAIN_OF[c]], n)

    for i, c in enumerate(COMP):
        ax = fig.add_axes([0.075 + i * 0.235, 0.492, 0.175, 0.100])
        hist(ax, d[c], COMP_LABEL[c], DOMAIN_COLOR[COMP_LABEL[c]], n)

    # Completeness map: one row per subject, ordered by age, so a gap that
    # tracks age is visible rather than inferred from a coverage count.
    ax = fig.add_axes([0.150, 0.080, 0.520, 0.330])
    o = d.sort_values('age')
    M = (~o[RAW].isna()).astype(int).values
    ax.imshow(M.T, aspect='auto', cmap=matplotlib.colors.ListedColormap(['#D32F2F', '#E8EAF6']),
              interpolation='nearest', vmin=0, vmax=1)
    ax.set_yticks(range(len(RAW)))
    ax.set_yticklabels([RAW_LABEL[c] for c in RAW], fontsize=FONT_TICK)
    ax.set_xlabel('Subjects, ordered by age  →', fontsize=FONT_AXIS_TITLE, labelpad=2)
    ax.set_xticks([])
    ax.set_title('Completeness (red = missing)', fontsize=FONT_AXIS_TITLE, pad=3)
    for s in ax.spines.values():
        s.set_visible(False)

    ax = fig.add_axes([0.760, 0.080, 0.205, 0.330])
    k = d['global_reduced_n'].value_counts().sort_index()
    ax.bar(k.index.astype(int), k.values, color='#4A148C', alpha=0.8,
           edgecolor='white', linewidth=0.6)
    for xi, yi in zip(k.index, k.values):
        ax.text(xi, yi + 0.6, str(yi), ha='center', fontsize=FONT_TICK - 0.5)
    ax.set_xlabel('Inputs contributing (of 6)', fontsize=FONT_AXIS_TITLE, labelpad=2)
    ax.set_ylabel('Subjects', fontsize=FONT_AXIS_TITLE, labelpad=2)
    ax.set_title('Composite completeness', fontsize=FONT_AXIS_TITLE, pad=3)
    _clean(ax)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / 'qc_cognitive_composite_p1.png'
    fig.savefig(p, dpi=300, facecolor='white')
    plt.close(fig)
    return p


def page2(d: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(WIDTH_2COL, WIDTH_2COL * 1.05))
    fig.suptitle('Cognitive composite: does it behave?', y=0.982,
                 fontsize=FONT_PANEL_LABEL, fontweight='bold')

    for i, c in enumerate(RAW):
        ax = fig.add_axes([0.085 + (i % 3) * 0.315, 0.812 - (i // 3) * 0.162,
                           0.225, 0.104])
        age_scatter(ax, d['age'], d[c], RAW_LABEL[c], DOMAIN_COLOR[DOMAIN_OF[c]])

    for i, c in enumerate(COMP):
        ax = fig.add_axes([0.085 + i * 0.235, 0.487, 0.165, 0.104])
        age_scatter(ax, d['age'], d[c], COMP_LABEL[c], DOMAIN_COLOR[COMP_LABEL[c]])

    # Internal structure: do measures within a domain cohere more than across?
    ax = fig.add_axes([0.105, 0.085, 0.225, 0.275])
    C = d[RAW].corr()
    im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(RAW)))
    ax.set_xticklabels([RAW_LABEL[c] for c in RAW], rotation=45, ha='right',
                       fontsize=FONT_TICK - 1)
    ax.set_yticks(range(len(RAW)))
    ax.set_yticklabels([RAW_LABEL[c] for c in RAW], fontsize=FONT_TICK - 1)
    for i in range(len(RAW)):
        for j in range(len(RAW)):
            ax.text(j, i, f'{C.iloc[i, j]:.2f}', ha='center', va='center',
                    fontsize=FONT_TICK - 1.5,
                    color='white' if abs(C.iloc[i, j]) > 0.55 else 'black')
    ax.set_title('Inter-measure correlations', fontsize=FONT_AXIS_TITLE, pad=3)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).ax.tick_params(labelsize=FONT_TICK - 1)

    # Convergent validity against KBIT: an independent IQ estimate that is not
    # part of the composite. If the composite measures general cognition it
    # should track this; if it does not, the label is wrong.
    ax = fig.add_axes([0.475, 0.085, 0.200, 0.275])
    age_scatter(ax, d['kbit_iq'], d['global_reduced'], 'Global composite (z)', '#4A148C')
    ax.set_xlabel('KBIT IQ (independent)', fontsize=FONT_AXIS_TITLE, labelpad=1.5)
    ax.set_title('Convergent validity', fontsize=FONT_AXIS_TITLE, pad=3)

    # The measure this replaces. Points far off the diagonal are subjects whose
    # score changed because the legacy version averaged a different test set.
    ax = fig.add_axes([0.770, 0.085, 0.200, 0.275])
    m = d[['global_composite', 'global_reduced', 'age']].dropna()
    norm = plt.Normalize(m.age.min(), m.age.max())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('ba', [AGE_YOUNG, AGE_OLD])
    ax.scatter(m.global_composite, m.global_reduced, s=13, c=cmap(norm(m.age)),
               edgecolors='white', linewidths=0.35)
    lim = [min(m.global_composite.min(), m.global_reduced.min()) - 0.1,
           max(m.global_composite.max(), m.global_reduced.max()) + 0.1]
    ax.plot(lim, lim, color=NEUTRAL_GRAY, lw=0.8, ls=':')
    r, p = stats.pearsonr(m.global_composite, m.global_reduced)
    ax.text(0.03, 0.95, f'r = {r:.3f}', transform=ax.transAxes, va='top',
            fontsize=FONT_TICK, color=REGRESSION_COLOR)
    ax.set_xlabel('Legacy global_composite', fontsize=FONT_AXIS_TITLE, labelpad=1.5)
    ax.set_ylabel('New global_reduced', fontsize=FONT_AXIS_TITLE, labelpad=1.5)
    ax.set_title('vs. the measure it replaces (colour = age)',
                 fontsize=FONT_AXIS_TITLE, pad=3)
    _clean(ax)

    p = FIG_DIR / 'qc_cognitive_composite_p2.png'
    fig.savefig(p, dpi=300, facecolor='white')
    plt.close(fig)
    return p


def text_report(d: pd.DataFrame) -> None:
    print('=' * 70)
    print('COGNITIVE COMPOSITE VALIDATION')
    print('=' * 70)

    print('\nConstituents')
    for c in RAW:
        v = d[c].dropna()
        print(f'  {RAW_LABEL[c]:20} {DOMAIN_OF[c]:10} n={len(v):2d}  '
              f'mean={v.mean():7.2f}  sd={v.std():6.2f}  '
              f'skew={stats.skew(v):+.2f}')

    print('\nInternal structure (Pearson)')
    C = d[RAW].corr()
    within, across = [], []
    for i, a in enumerate(RAW):
        for b in RAW[i + 1:]:
            (within if DOMAIN_OF[a] == DOMAIN_OF[b] else across).append(C.loc[a, b])
    print(f'  mean within-domain r : {np.mean(within):+.3f}  (n={len(within)} pairs)')
    print(f'  mean across-domain r : {np.mean(across):+.3f}  (n={len(across)} pairs)')
    print('  within > across means the domains carry real structure; if they are'
          '\n  equal the three-domain split is decorative and the composite is'
          '\n  simply a general factor.')

    # Cronbach's alpha on the z-scored constituents, complete cases only.
    Z = d[RAW].apply(lambda s: (s - s.mean()) / s.std()).dropna()
    k = Z.shape[1]
    alpha = (k / (k - 1)) * (1 - Z.var(ddof=1).sum() / Z.sum(axis=1).var(ddof=1))
    print(f"\n  Cronbach's alpha: {alpha:.3f}  (complete cases, n={len(Z)})")

    print('\nAge relationships')
    for c in RAW + COMP:
        m = d[[c, 'age']].dropna()
        r, p = stats.pearsonr(m[c], m.age)
        lab = RAW_LABEL.get(c, COMP_LABEL.get(c, c))
        print(f'  {lab:20} r={r:+.3f}  p={p:.4f}  n={len(m)}')

    print('\nConvergent / discriminant validity')
    for other, label in [('kbit_iq', 'KBIT IQ'), ('education_years', 'Education (yrs)')]:
        if other in d.columns:
            m = d[['global_reduced', other]].dropna()
            r, p = stats.pearsonr(m.global_reduced, m[other])
            print(f'  global_reduced x {label:16} r={r:+.3f}  p={p:.4f}  n={len(m)}')

    print('\nAgainst the legacy composite')
    m = d[['global_composite', 'global_reduced']].dropna()
    r, _ = stats.pearsonr(m.global_composite, m.global_reduced)
    print(f'  r = {r:.3f}  (n={len(m)})')
    for c, lab in [('global_n_domains', 'legacy  n_domains'),
                   ('global_reduced_n', 'new     n_inputs')]:
        x = d[[c, 'age']].dropna()
        rr, pp = stats.pearsonr(x[c], x.age)
        print(f'  {lab} x age: r={rr:+.3f} (p={pp:.3f})')
    for c, lab in [('memory_n_measures', 'legacy  memory n'),
                   ('memory_reduced_n', 'new     memory n')]:
        if c in d.columns and d[c].nunique() > 1:
            x = d[[c, 'age']].dropna()
            rr, pp = stats.pearsonr(x[c], x.age)
            print(f'  {lab} x age: r={rr:+.3f} (p={pp:.3f})')
        else:
            print(f'  {lab} x age: constant (every subject has the same input)')


def main() -> int:
    d = load()
    text_report(d)
    p1, p2 = page1(d), page2(d)
    print(f'\nwrote {p1}\nwrote {p2}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
