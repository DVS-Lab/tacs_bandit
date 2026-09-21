"""
compare_composites.py — Does the composite switch change any conclusion?

Runs the preregistered models under both cognitive composites and reports
where they disagree.

The legacy `global_composite` averages whatever measures a subject has.
Digit Span, BVMT and Trails were administered only from roughly age 56, so an
older subject's domain score is a mean of two z-scores where a younger
subject's is a single one -- mechanically less variable, and that difference
tracks age at r = +.90 for memory. `global_reduced` uses the same six measures
for everyone. The two correlate at r = .96, so conclusions are expected to
hold; this script demonstrates that rather than assuming it.

Every model here is specified exactly as in hypothesis_tests.py, with only the
composite column swapped. A row is flagged when the two versions disagree on
sign or cross p = .05 -- those are the only differences that change what the
paper claims.

Usage
-----
    python compare_composites.py
    python compare_composites.py --csv out.csv
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import REPO_ROOT, COG_COMPOSITE, COG_COMPOSITE_LEGACY

# (label, dependent variable, predictors). COG is substituted per run.
MODELS: List[Tuple[str, str, List[str]]] = [
    ('H1.1a  p(stay|win)',      'sham_p_stay_win',    ['COG', 'age', 'education_years']),
    ('H1.1b  p(shift|lose)',    'sham_p_shift_lose',  ['COG', 'age', 'education_years']),
    ('H1.2a  + SPSRQ-SR',       'sham_p_stay_win',    ['spsrq_sr', 'COG', 'age', 'education_years']),
    ('H1.2b  + SPSRQ-SP',       'sham_p_shift_lose',  ['spsrq_sp', 'COG', 'age', 'education_years']),
    ('H1.1.1 beta -> loseshift','sham_p_shift_lose',  ['sham_beta', 'COG', 'age', 'education_years']),
    ('H2.2   d accuracy',       'delta_accuracy',     ['age', 'COG']),
    ('H2.2   d win rate',       'delta_win_rate',     ['age', 'COG']),
    ('H2.2   d alpha',          'delta_alpha',        ['age', 'COG']),
    ('H2.2   d beta',           'delta_beta',         ['age', 'COG']),
    ('H2.2   d p(stay|win)',    'delta_p_stay_win',   ['age', 'COG']),
    ('H2.2   d p(shift|lose)',  'delta_p_shift_lose', ['age', 'COG']),
    ('theta  d accuracy',       'delta_accuracy',     ['theta_p95', 'age', 'COG']),
    ('theta  d win rate',       'delta_win_rate',     ['theta_p95', 'age', 'COG']),
    # The hierarchical learning rates, which the MLE columns above do not use.
    ('H2.2   d alpha (hb)',     'delta_alpha_hb',     ['age', 'COG']),
    ('H2.2   d beta  (hb)',     'delta_beta_hb',      ['age', 'COG']),
]

# Moderation: composite x age on the baseline measures.
MODERATIONS = [('sham_p_stay_win', 'p(stay|win)'), ('sham_p_shift_lose', 'p(shift|lose)'),
               ('sham_alpha', 'alpha'), ('sham_beta', 'beta')]


def fit(d: pd.DataFrame, dv: str, preds: List[str], cog: str):
    """OLS with the composite substituted in. Returns (b, p, n) for the composite."""
    cols = [dv] + [cog if c == 'COG' else c for c in preds]
    if any(c not in d.columns for c in cols):
        return None
    m = d[cols].apply(pd.to_numeric, errors='coerce').dropna()
    if len(m) < len(cols) + 3:
        return None
    X = sm.add_constant(m[[c for c in cols[1:]]].astype(float))
    res = sm.OLS(m[dv].astype(float), X).fit()
    return res.params[cog], res.pvalues[cog], len(m)


def fit_moderation(d: pd.DataFrame, dv: str, cog: str):
    """DV ~ cog * age, centred. Returns (b, p, n) for the interaction."""
    cols = [dv, cog, 'age']
    if any(c not in d.columns for c in cols):
        return None
    m = d[cols].apply(pd.to_numeric, errors='coerce').dropna()
    if len(m) < 12:
        return None
    c1 = m[cog] - m[cog].mean()
    c2 = m['age'] - m['age'].mean()
    X = sm.add_constant(pd.DataFrame({'cog': c1, 'age': c2, 'cog_x_age': c1 * c2}))
    res = sm.OLS(m[dv].astype(float), X).fit()
    return res.params['cog_x_age'], res.pvalues['cog_x_age'], len(m)


def verdict(a, b) -> str:
    """
    How the two runs differ, in the terms that matter for a claim.

    A sign flip only counts when at least one version is distinguishable from
    zero. Two null coefficients that happen to sit on opposite sides of zero
    -- b = -0.0004 (p = .99) becoming b = +0.0041 (p = .89) -- describe the
    same result, and flagging them buries the one comparison that does change.
    """
    if a is None or b is None:
        return 'n/a'
    (ba, pa, _), (bb, pb, _) = a, b
    if (pa < .05) != (pb < .05):
        return '** SIGNIFICANCE FLIPS **'
    if np.sign(ba) != np.sign(bb):
        if min(pa, pb) < .10:
            return '** SIGN FLIPS **'
        return 'same (both null)'
    return 'same'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--csv', default=None, help='also write the table here')
    args = ap.parse_args(argv)

    d = pd.read_csv(REPO_ROOT / 'data' / 'master_subject_data.csv',
                    dtype={'subject_id': str})
    legacy, new = COG_COMPOSITE_LEGACY, COG_COMPOSITE
    print(f'Comparing  {legacy}  vs  {new}\n')

    r = d[[legacy, new]].dropna()
    print(f'  the two composites correlate at r = {r[legacy].corr(r[new]):.3f} '
          f'(n = {len(r)})\n')

    rows = []
    hdr = f"{'model':30}{'legacy b':>10}{'p':>8}{'new b':>10}{'p':>8}{'n':>5}   verdict"
    print(hdr); print('-' * len(hdr))
    for label, dv, preds in MODELS:
        a = fit(d, dv, preds, legacy)
        b = fit(d, dv, preds, new)
        v = verdict(a, b)
        if a is None or b is None:
            print(f'  {label:28} — unavailable (missing column)')
            continue
        print(f'  {label:28}{a[0]:>10.4f}{a[1]:>8.3f}{b[0]:>10.4f}{b[1]:>8.3f}'
              f'{b[2]:>5}   {v}')
        rows.append(dict(model=label, dv=dv, legacy_b=a[0], legacy_p=a[1],
                         new_b=b[0], new_p=b[1], n=b[2], verdict=v))

    print(f"\n{'moderation (composite x age)':30}"
          f"{'legacy b':>10}{'p':>8}{'new b':>10}{'p':>8}{'n':>5}   verdict")
    print('-' * len(hdr))
    for dv, label in MODERATIONS:
        a = fit_moderation(d, dv, legacy)
        b = fit_moderation(d, dv, new)
        v = verdict(a, b)
        if a is None or b is None:
            print(f'  {label:28} — unavailable')
            continue
        print(f'  {label:28}{a[0]:>10.4f}{a[1]:>8.3f}{b[0]:>10.4f}{b[1]:>8.3f}'
              f'{b[2]:>5}   {v}')
        rows.append(dict(model=f'moderation {label}', dv=dv, legacy_b=a[0],
                         legacy_p=a[1], new_b=b[0], new_p=b[1], n=b[2], verdict=v))

    changed = [r for r in rows if r['verdict'].startswith('**')]
    print('\n' + '=' * 70)
    if changed:
        print(f'{len(changed)} of {len(rows)} models change conclusion:')
        for c in changed:
            print(f"    {c['model']}: legacy p={c['legacy_p']:.3f} -> new p={c['new_p']:.3f}")
        print('\nThese are the results where the composite choice matters. Report\n'
              'the preregistered (legacy) version as primary and this as a\n'
              'sensitivity analysis, or say explicitly which was used.')
    else:
        print(f'All {len(rows)} models agree on sign and significance.')
        print('The composite switch changes no conclusion, which is the robustness\n'
              'statement worth reporting.')

    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f'\nwrote {args.csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
