"""
compare_rl_estimates.py — Every learning-parameter test, under both estimators

The hierarchical fit is primary (config.RL_ESTIMATES = 'hb'); MLE is what the
defended analyses used. Both are reported for every test of alpha or beta,
because three borderline results depend on which one is used:

  H1.1.1  beta -> lose-shifting       MLE p = .056   hierarchical p = .26
  H2.2    age x delta alpha           MLE p = .054   hierarchical p = .82
  H1.1.2  cognition x age -> alpha    MLE p = .18    hierarchical p = .045

The MLE trends come from fits pinned at a bound (alpha = 0 or 1, beta = 50),
which pooling pulls toward the group. None of the three is robust to the
method, and they should be reported as such rather than as findings.

Each test is run three ways, so the source of any difference is visible:

  mle_full   MLE on the test's own sample (H1 = 61, H2 = 57)
  mle_57     MLE restricted to the subjects the hierarchical fit covers
  hb         hierarchical posterior means (57)

mle_full vs mle_57 isolates the sample change; mle_57 vs hb isolates the
estimator. Specifications match the notebook (hypothesis_tests.py for the
covariates).

Usage
-----
    python compare_rl_estimates.py
    python compare_rl_estimates.py --csv ../derivatives/rl_estimate_comparison.csv
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from config import COG_COMPOSITE, rl
import samples


def _num(df, cols):
    return df[cols].apply(pd.to_numeric, errors='coerce').dropna()


def ols_term(df, y, xs, term):
    m = _num(df, [y] + xs)
    r = sm.OLS(m[y], sm.add_constant(m[xs])).fit()
    return r.params[term], r.pvalues[term], len(m)


def interaction(df, y, x, z, covariates=()):
    """y ~ x_c + z_c + covariates + x_c*z_c, the notebook's 2.5 specification."""
    m = _num(df, [y, x, z, *covariates])
    c = lambda s: s - s.mean()
    X = pd.DataFrame({'x': c(m[x]), 'z': c(m[z])})
    for cv in covariates:
        X[cv] = m[cv]
    X['xz'] = X.x * X.z
    r = sm.OLS(m[y], sm.add_constant(X)).fit()
    return r.params['xz'], r.pvalues['xz'], len(m)


def corr(df, a, b):
    m = _num(df, [a, b])
    r, p = stats.pearsonr(m[a], m[b])
    return r, p, len(m)


def one_sample_dz(df, c):
    x = _num(df, [c])[c]
    return x.mean() / x.std(ddof=1), stats.ttest_1samp(x, 0).pvalue, len(x)


def paired_dz(df, a, b):
    m = _num(df, [a, b])
    d = m[b] - m[a]
    return d.mean() / d.std(ddof=1), stats.ttest_rel(m[b], m[a]).pvalue, len(m)


# (label, sample, function(frame, estimator) -> (stat, p, n))
TESTS: List[Tuple[str, str, Callable]] = []
for p in ('alpha', 'beta'):
    TESTS += [
        (f'H1.1.1  sham {p} -> p(shift|lose) | cog, age, edu', 'H1',
         lambda d, e, p=p: ols_term(d, 'sham_p_shift_lose',
                                    [rl(f'sham_{p}', e), COG_COMPOSITE, 'age', 'education_years'],
                                    rl(f'sham_{p}', e))),
        (f'H1.1.2  cognition x age -> sham {p} | edu', 'H1',
         lambda d, e, p=p: interaction(d, rl(f'sham_{p}', e), COG_COMPOSITE, 'age',
                                       covariates=('education_years',))),
        (f'H1      age x sham {p} (r)', 'H1',
         lambda d, e, p=p: corr(d, 'age', rl(f'sham_{p}', e))),
        (f'H2.1    active vs sham {p} (dz)', 'H2',
         lambda d, e, p=p: paired_dz(d, rl(f'sham_{p}', e), rl(f'active_{p}', e))),
        (f'H2.2    age x delta {p} (r)', 'H2',
         lambda d, e, p=p: corr(d, 'age', rl(f'delta_{p}', e))),
        (f'theta   theta_p95 x delta {p} (r)', 'H2',
         lambda d, e, p=p: corr(d, 'theta_p95', rl(f'delta_{p}', e))),
        (f'field   |E| x delta {p} (r)', 'Dose_H2',
         lambda d, e, p=p: corr(d, 'mean_magnE', rl(f'delta_{p}', e))),
    ]


def verdict(a, b) -> str:
    """Same rule as compare_composites: only differences that change a claim."""
    (ba, pa, _), (bb, pb, _) = a, b
    if (pa < .05) != (pb < .05):
        return '** SIGNIFICANCE DIFFERS **'
    if np.sign(ba) != np.sign(bb) and min(pa, pb) < .10:
        return '** SIGN DIFFERS **'
    if min(pa, pb) < .10 <= max(pa, pb):
        return 'trend in one only'
    return 'same'


def run(verbose: bool = True) -> pd.DataFrame:
    d = samples.load()
    hb_ids = samples.ids('H1_RL', d)
    rows = []
    for label, smp, f in TESTS:
        full = d[d.subject_id.isin(samples.ids(smp, d))]
        cov = full[full.subject_id.isin(hb_ids)]
        a, b, c = f(full, 'mle'), f(cov, 'mle'), f(cov, 'hb')
        rows.append(dict(test=label, sample=smp,
                         mle_full_stat=a[0], mle_full_p=a[1], mle_full_n=a[2],
                         mle_57_stat=b[0], mle_57_p=b[1], mle_57_n=b[2],
                         hb_stat=c[0], hb_p=c[1], hb_n=c[2],
                         verdict=verdict(a, c)))
    out = pd.DataFrame(rows)
    if verbose:
        fmt = lambda s, p, n: f'{s:+.3f} p={p:.3f} n={int(n)}'
        print(f"{'test':46}{'MLE (own sample)':>24}{'MLE (hb subjects)':>24}"
              f"{'hierarchical':>24}   verdict")
        for r in out.itertuples():
            print(f'  {r.test:44}{fmt(r.mle_full_stat, r.mle_full_p, r.mle_full_n):>24}'
                  f'{fmt(r.mle_57_stat, r.mle_57_p, r.mle_57_n):>24}'
                  f'{fmt(r.hb_stat, r.hb_p, r.hb_n):>24}   {r.verdict}')
        flagged = out[out.verdict != 'same']
        print(f'\n{len(flagged)} of {len(out)} tests differ between MLE and the '
              f'hierarchical fit in a way that changes what could be claimed:')
        for r in flagged.itertuples():
            print(f'  {r.test.strip()}: MLE p = {r.mle_full_p:.3f}, '
                  f'hierarchical p = {r.hb_p:.3f}')
        print('\nReport these as method-dependent, not as findings.')
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--csv', default=None)
    args = ap.parse_args(argv)
    out = run()
    if args.csv:
        out.to_csv(args.csv, index=False)
        print(f'wrote {args.csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
