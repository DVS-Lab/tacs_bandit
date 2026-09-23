"""
table1.py — Sample characteristics for the manuscript

Describes the **H1 sample** (passed the preregistered exclusions, has age;
N = 61), which is the largest set every baseline analysis uses. Rows whose
measure covers fewer people report their own n, so the table never implies
more data than exists -- the cognitive battery, EEG and head models each have
their own coverage, for reasons given in `VARIABLES.md`.

Split by age at the median, because age organises the paper's results. The
comparison column is a test of that split, not a randomisation check: this is
a within-subject design, so the groups are not expected to match. Welch's t
for continuous measures (unequal variances, which age groups generally have),
chi-square for categorical, Fisher's exact when an expected count is small.

Learning rate and inverse temperature are the hierarchical estimates
(`RL_ESTIMATES`), so they cover the 57 subjects with both sessions.

Usage
-----
    python table1.py
    python table1.py --sample H2
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from config import REPO_ROOT, COG_COMPOSITE, rl
import samples

OUT_CSV = REPO_ROOT / 'derivatives' / 'table1.csv'
OUT_MD = REPO_ROOT / 'derivatives' / 'table1.md'

# (column, label, kind). 'n_pct' rows are categorical.
ROWS: List[Tuple[str, str, str]] = [
    ('age', 'Age (years)', 'cont'),
    ('gender', 'Sex', 'n_pct'),
    ('race', 'Race', 'n_pct'),
    ('ethnicity', 'Ethnicity', 'n_pct'),
    ('education_years', 'Education (years)', 'cont'),
    ('counterbalance', 'Counterbalance (A / B)', 'n_pct'),
    ('__sep__Cognition', '', 'sep'),
    (COG_COMPOSITE, 'Global cognition (z)', 'cont'),
    ('ef_reduced', 'Executive function (z)', 'cont'),
    ('memory_reduced', 'Memory (z)', 'cont'),
    ('speed_reduced', 'Processing speed (z)', 'cont'),
    ('kbit_iq', 'KBIT IQ', 'cont'),
    ('scd_q', 'Subjective cognitive decline', 'cont'),
    ('__sep__Baseline behaviour (sham)', '', 'sep'),
    ('sham_accuracy', 'Accuracy', 'cont'),
    ('sham_win_rate', 'Win rate', 'cont'),
    ('sham_p_stay_win', 'p(stay | win)', 'cont'),
    ('sham_p_shift_lose', 'p(shift | lose)', 'cont'),
    ('sham_ttc', 'Trials to criterion', 'cont'),
    ('__RL_ALPHA__', 'Learning rate alpha', 'cont'),
    ('__RL_BETA__', 'Inverse temperature beta', 'cont'),
    ('__sep__Reward sensitivity', '', 'sep'),
    ('spsrq_sr', 'SPSRQ reward sensitivity', 'cont'),
    ('spsrq_sp', 'SPSRQ punishment sensitivity', 'cont'),
    ('__sep__Physiology and anatomy', '', 'sep'),
    ('theta_p95', 'Theta bursting (p95)', 'cont'),
    # 'dose' rows drop the seven T1-only head models, as every E-field
    # analysis does. Reporting them over the whole sample would describe a
    # sample no result uses.
    ('mean_magnE', 'Modelled |E| in DLPFC (V/m)', 'dose'),
    ('dist_pial_dlpfc_p1', 'Scalp-to-cortex distance (mm)', 'dose'),
    ('layer_skull', 'Skull thickness at F3 (mm)', 'dose'),
]


def fmt_cont(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors='coerce').dropna()
    if len(x) == 0:
        return '—'
    return f'{x.mean():.2f} ({x.std():.2f})'


def cont_test(a: pd.Series, b: pd.Series):
    a = pd.to_numeric(a, errors='coerce').dropna()
    b = pd.to_numeric(b, errors='coerce').dropna()
    if len(a) < 3 or len(b) < 3:
        return '—', np.nan
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return f't({stats.ttest_ind(a, b, equal_var=False).df:.0f}) = {t:+.2f}', p


def cat_test(a: pd.Series, b: pd.Series):
    tab = pd.crosstab(pd.concat([a, b]), ['younger'] * len(a) + ['older'] * len(b))
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return '—', np.nan
    chi2, p, _, expected = stats.chi2_contingency(tab)
    if (expected < 5).any() and tab.shape == (2, 2):
        _, p = stats.fisher_exact(tab.values)
        return "Fisher's exact", p
    return f'chi2({int((tab.shape[0]-1)*(tab.shape[1]-1))}) = {chi2:.2f}', p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--sample', default='H1', choices=list(samples.SAMPLES))
    args = ap.parse_args(argv)

    from config import SUBJECT_INFO
    d = samples.frame(args.sample)
    d['counterbalance'] = d.subject_id.map(
        lambda s: SUBJECT_INFO.get(s, {}).get('counterbalance', '?'))
    d['age'] = pd.to_numeric(d['age'], errors='coerce')
    median_age = d['age'].median()
    young, old = d[d.age < median_age], d[d.age >= median_age]
    print(f'{args.sample} sample: N = {len(d)}; median age {median_age:.1f} '
          f'(younger {len(young)}, older {len(old)})')

    rows = []
    for col, label, kind in ROWS:
        if kind == 'sep':
            rows.append(dict(section=col.replace('__sep__', ''), measure='', n='',
                             overall='', younger='', older='', test='', p=''))
            continue
        real = {'__RL_ALPHA__': rl('sham_alpha'), '__RL_BETA__': rl('sham_beta')}.get(col, col)
        if real not in d.columns:
            continue
        if kind == 'dose':
            keep = d['t1_only'].eq(False)
            dd, yy, oo = d[keep], young[young['t1_only'].eq(False)], old[old['t1_only'].eq(False)]
            label = label + ' [FLAIR head models]'
        else:
            dd, yy, oo = d, young, old
        n = int(dd[real].notna().sum())
        if kind in ('cont', 'dose'):
            stat, p = cont_test(yy[real], oo[real])
            rows.append(dict(section='', measure=label, n=n, overall=fmt_cont(dd[real]),
                             younger=fmt_cont(yy[real]), older=fmt_cont(oo[real]),
                             test=stat, p='' if np.isnan(p) else f'{p:.3f}'))
        else:
            counts = d[real].value_counts(dropna=False)
            stat, p = cat_test(young[real].dropna(), old[real].dropna())
            first = True
            for level, k in counts.items():
                lvl = 'missing' if pd.isna(level) else str(level)
                rows.append(dict(
                    section='', measure=f'{label}: {lvl}' if not first else f'{label}: {lvl}',
                    n=n, overall=f'{k} ({100 * k / len(d):.0f}%)',
                    younger=f'{(young[real] == level).sum()}',
                    older=f'{(old[real] == level).sum()}',
                    test=stat if first else '', p=('' if np.isnan(p) else f'{p:.3f}') if first else ''))
                first = False

    t = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT_CSV, index=False)

    # Markdown, for pasting into the manuscript
    lines = [f'**Table 1.** Sample characteristics ({args.sample} sample, N = {len(d)}). '
             f'Mean (SD) unless noted; n per row is the number with that measure. '
             f'Age split at the median ({median_age:.1f} years): younger n = {len(young)}, '
             f'older n = {len(old)}. Welch t or chi-square/Fisher for the split.\n',
             '| Measure | n | Overall | Younger | Older | Test | p |',
             '|---|---|---|---|---|---|---|']
    for r in t.itertuples():
        if r.section:
            lines.append(f'| **{r.section}** | | | | | | |')
            continue
        measure = str(r.measure).replace('|', r'\|')   # p(stay | win), |E| ...
        lines.append(f'| {measure} | {r.n} | {r.overall} | {r.younger} | {r.older} '
                     f'| {r.test} | {r.p} |')
    OUT_MD.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nwrote {OUT_CSV.relative_to(REPO_ROOT)}\nwrote {OUT_MD.relative_to(REPO_ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
