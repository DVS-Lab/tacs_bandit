"""
skull_circularity.py — Does skull thickness explain the field, or just feed it?

The age -> skull -> |E| story has a circularity in its middle link. |E| is
simulated by an FEM on charm's tissue segmentation, so charm's skull is an
*input* to the model that produced the field. A thicker charm skull lowers
the simulated |E| by construction, and a strong charm-skull x |E| correlation
says as much about the simulator as about the head.

The check is to measure the same span without charm: skull thickness along
the F3 ray read directly from raw T1 intensity, with no tissue labels
(`skull_t1_span`, from extract_skull_t1.py in the SimNIBS repo). If both
measures show the age effect, the anatomy is real. If only the charm measure
predicts |E|, the dose link rests on the segmentation that generated it.

Reported in ANALYSIS_HANDOFF.md as:

    charm skull     x age +.337   x |E| -.778
    T1 span         x age +.359   x |E| -.231 (p = .081)
    agreement r = +.40;  mediation 89% -> 18% (n.s.)

Those numbers were first computed inline and existed only as text. This
script is now their source.

Mediation is product-of-coefficients: a = age -> skull, b = skull -> |E|
controlling for age, c = total age -> |E|, proportion mediated = a*b / c.
The indirect effect gets a percentile bootstrap CI (fixed seed), which is
what "n.s." refers to.

Sample: FLAIR head models only (T1-only excluded, as everywhere), with age,
both skull measures and |E| present. Every statistic uses that one sample, so
the rows are comparable.

Usage
-----
    python skull_circularity.py
    python skull_circularity.py --n-boot 10000
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from config import REPO_ROOT, EFIELD_CSV_PATH

FIELD = 'mean_magnE'
MEASURES = {
    'layer_skull': 'charm skull (an FEM input)',
    'skull_t1_span': 'T1 intensity span (external)',
}
OUT = REPO_ROOT / 'derivatives' / 'skull_circularity.csv'
SEED = 20260921


def load() -> pd.DataFrame:
    e = pd.read_csv(EFIELD_CSV_PATH, dtype={'subject_id': str})
    m = pd.read_csv(REPO_ROOT / 'data' / 'master_subject_data.csv',
                    dtype={'subject_id': str})[['subject_id', 'age']]
    d = e.merge(m, on='subject_id', how='left')
    d['age'] = pd.to_numeric(d['age'], errors='coerce')
    d = d[~d['t1_only'].astype(bool)]
    return d.dropna(subset=['age', FIELD, *MEASURES]).reset_index(drop=True)


def paths(x: pd.DataFrame, m: str):
    """a, b, c for age -> m -> |E|."""
    c = sm.OLS(x[FIELD], sm.add_constant(x[['age']])).fit().params['age']
    a = sm.OLS(x[m], sm.add_constant(x[['age']])).fit().params['age']
    b = sm.OLS(x[FIELD], sm.add_constant(x[['age', m]])).fit().params[m]
    return a, b, c


def mediation(d: pd.DataFrame, m: str, n_boot: int, rng) -> dict:
    a, b, c = paths(d, m)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        s = d.iloc[rng.integers(0, len(d), len(d))]
        ba, bb, _ = paths(s, m)
        boot[i] = ba * bb
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(a=a, b=b, c=c, indirect=a * b, prop_mediated=a * b / c,
                indirect_ci_lo=lo, indirect_ci_hi=hi,
                indirect_sig=bool(lo > 0 or hi < 0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--n-boot', type=int, default=5000)
    args = ap.parse_args(argv)

    d = load()
    rng = np.random.default_rng(SEED)
    print(f'N = {len(d)} (FLAIR head models with age, both skull measures, and |E|)\n')

    rows = []
    print(f"{'':32}{'x age':>9}{'x |E|':>9}{'p':>8}{'mediated':>10}   indirect 95% CI")
    for m, label in MEASURES.items():
        r_age, p_age = stats.pearsonr(d['age'], d[m])
        r_e, p_e = stats.pearsonr(d[m], d[FIELD])
        med = mediation(d, m, args.n_boot, rng)
        sig = '' if med['indirect_sig'] else '  (n.s.)'
        print(f"  {label:30}{r_age:>+9.3f}{r_e:>+9.3f}{p_e:>8.3f}"
              f"{med['prop_mediated']:>9.0%}   [{med['indirect_ci_lo']:+.5f}, "
              f"{med['indirect_ci_hi']:+.5f}]{sig}")
        rows.append(dict(measure=m, label=label, n=len(d), r_age=r_age, p_age=p_age,
                         r_field=r_e, p_field=p_e, **med))

    r_agree, p_agree = stats.pearsonr(d['layer_skull'], d['skull_t1_span'])
    print(f'\n  agreement between the two measures: r = {r_agree:+.3f} (p = {p_agree:.3f})')
    r_c, _ = stats.pearsonr(d['layer_skull'], d[FIELD])
    r_t, _ = stats.pearsonr(d['skull_t1_span'], d[FIELD])
    print(f'  the field relationship falls from r = {r_c:+.3f} to {r_t:+.3f} once the\n'
          f'  measure no longer comes from the segmentation the field was computed on.')

    out = pd.DataFrame(rows)
    out['agreement_r'] = r_agree
    out['n_boot'] = args.n_boot
    out['seed'] = SEED
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f'\nwrote {OUT.relative_to(REPO_ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
