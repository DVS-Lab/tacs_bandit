"""
sweep_moderators.py — Every subject-level measure against age, and against
the stimulation response

Two sweeps, one machine, because they share the variable-selection rule and
the multiplicity correction:

    --target age    each measure against age (H1 sample)
    --target stim   each measure against each change score (H2 sample)

**Why this exists.** The notebook's exploratory sweep (section 6.2) covered 17
moderators chosen by hand. The master table holds ~86 usable subject-level
measures, so roughly 80% of the questionnaire battery had never been looked at
-- including instruments with 56-61 coverage. "We found nothing" is only worth
saying if the search was actually exhaustive, and a hand-picked list cannot
support it. This sweep is the exhaustive version.

It is exploratory and is reported as such: Benjamini-Hochberg FDR across all
tests in the sweep, and the count of uncorrected hits printed next to the
number expected by chance. A result that does not clear FDR here is a
hypothesis for another dataset, not a finding in this one.

**Two things the sweep checks that a bare correlation does not.**

1. *Gradient or group difference* (`--target age`). Age was recruited in two
   bands (see AGE_BAND_* in config.py), so a correlation across the whole
   range is close to a two-group contrast. Every age result is reported with
   the within-band correlations and the between-band d, and in this sample
   essentially all of them are pure group differences.

2. *Baseline dependency* (`--target stim`). A change score correlates with its
   own baseline by construction, so any measure that tracks baseline
   performance will appear to moderate the response. The sweep reports, for
   each hit, the partial correlation holding the sham value constant. This is
   not a nuisance control: for the reach-rate change score it is the whole
   effect. Older and lower-scoring subjects start lower, have more room to
   improve, and so appear to benefit. (The correlation with the delivered
   field is negative, r = -.24. That sign is *not* disqualifying on its own --
   if tACS disrupts the process, more field means more disruption, which is
   this sign. What rules it out is that the sign is inconsistent across DVs
   once they are oriented so positive means better, and that |E| drops to
   p = .96 once age and the sham baseline are held constant.)

Usage
-----
    python sweep_moderators.py --target age
    python sweep_moderators.py --target stim
    python sweep_moderators.py --target age --top 40
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from config import (REPO_ROOT, rl, ttc,
                    AGE_BAND_YOUNG_MAX as BAND_YOUNG,
                    AGE_BAND_OLD_MIN as BAND_OLD)
import samples

OUT = REPO_ROOT / 'derivatives' / 'moderator_sweep_{target}.csv'

MIN_N = 30          # below this a correlation is not worth the multiplicity
MIN_UNIQUE = 5      # drops flags and near-constant columns

# Anatomy, per-condition behaviour and bookkeeping columns are not moderators:
# anatomy has its own script (efield_results.py), the sham_/active_ columns are
# the outcomes themselves, and the rest are identifiers or item counts.
SKIP_PREFIX = (
    'sham_', 'active_', 'delta_', 'lh_', 'rh_', 'dist_', 'layer_', 'charm_',
    'dlpfc_', 'f3_', 'skull_', 'slice_', 'parcel_', 'n_gm', 'roi_', 'std_magnE',
    'etiv', 'icv_', 'csf_', 'brain_', 'cortex_', 'total_', 'white_', 'subcort_',
    'ventricle_', 'supratentorial', 'brainseg', 'mask_', 'surface_', 'fs_',
    't1_only',
)
SKIP_SUFFIX = (
    '_n_items', '_n_measures', '_n_domains', '_source', '_reduced_n', '_sd_hb',
    '_n_vert', '_n_parcels', '_n_rays',
)
SKIP_EXACT = {'subject_id', 'gender', 'race', 'ethnicity'}


def moderators(d: pd.DataFrame, drop: List[str] = ()) -> List[str]:
    out = []
    for c in d.columns:
        if c in SKIP_EXACT or c in drop:
            continue
        if c.startswith(SKIP_PREFIX) or c.endswith(SKIP_SUFFIX):
            continue
        s = pd.to_numeric(d[c], errors='coerce')
        if s.notna().sum() >= MIN_N and s.nunique() >= MIN_UNIQUE:
            out.append(c)
    return out


def bands(d: pd.DataFrame, col: str) -> dict:
    """Within-band gradients and the between-band standardised difference."""
    m = d[['age', col]].apply(pd.to_numeric, errors='coerce').dropna()
    y, o = m[m.age < BAND_YOUNG], m[m.age >= BAND_OLD]
    if len(y) < 6 or len(o) < 6:
        return dict(r_young=np.nan, r_old=np.nan, band_d=np.nan, band_p=np.nan)
    pooled = np.sqrt((y[col].var(ddof=1) + o[col].var(ddof=1)) / 2)
    return dict(
        r_young=stats.pearsonr(y.age, y[col])[0],
        r_old=stats.pearsonr(o.age, o[col])[0],
        band_d=(o[col].mean() - y[col].mean()) / pooled if pooled > 0 else np.nan,
        band_p=stats.ttest_ind(y[col], o[col], equal_var=False)[1])


def sweep_age(d: pd.DataFrame) -> pd.DataFrame:
    # Baseline behaviour belongs in this sweep: `sham_*` columns are skipped
    # as *moderators* (they are outcomes, not predictors) but they are
    # perfectly good targets for age. Leaving them out is how the age effect on
    # response time went unnoticed until RT was derived by hand.
    # The non-primary TTC variant is dropped: under TTC_CENSORING = 'censored'
    # the raw `sham_ttc` is the survivorship-biased one, and it clears FDR here
    # (q = .02) purely on that bias. Reporting both would put a known artifact
    # in a list of survivors.
    ttc_drop = {c for c in ('sham_ttc', 'sham_ttc_censored') if c != ttc('sham_ttc')}
    behaviour = [c for c in d.columns if c.startswith('sham_') and c not in ttc_drop
                 and pd.to_numeric(d[c], errors='coerce').notna().sum() >= MIN_N
                 and pd.to_numeric(d[c], errors='coerce').nunique() >= MIN_UNIQUE]
    rows = []
    for c in moderators(d, drop=['age']) + behaviour:
        m = d[['age', c]].apply(pd.to_numeric, errors='coerce').dropna()
        if len(m) < MIN_N:
            continue
        r, p = stats.pearsonr(m.age, m[c])
        rows.append(dict(variable=c, n=len(m), r=r, p=p, **bands(d, c)))
    return pd.DataFrame(rows)


def sweep_stim(d: pd.DataFrame, dvs: List[str]) -> pd.DataFrame:
    rows = []
    for c in moderators(d):
        s = pd.to_numeric(d[c], errors='coerce')
        for dv in dvs:
            m = pd.concat([s, pd.to_numeric(d[dv], errors='coerce')], axis=1).dropna()
            m.columns = ['mod', 'dv']
            if len(m) < MIN_N:
                continue
            r, p = stats.pearsonr(m['mod'], m['dv'])
            # Partial correlation holding the sham value constant, so a
            # "moderator" that is really tracking baseline performance is
            # visible as such rather than being read as a response to
            # stimulation.
            base = f"sham_{dv.replace('delta_', '', 1)}"
            r_adj, p_adj = np.nan, np.nan
            if base in d.columns:
                t = pd.concat([s.rename('mod'),
                               pd.to_numeric(d[dv], errors='coerce').rename('dv'),
                               pd.to_numeric(d[base], errors='coerce').rename('base')],
                              axis=1).dropna()
                if len(t) >= MIN_N:
                    f = sm.OLS(t['dv'], sm.add_constant(t[['mod', 'base']])).fit()
                    r_adj, p_adj = f.params['mod'], f.pvalues['mod']
            rows.append(dict(variable=c, dv=dv, n=len(m), r=r, p=p,
                             b_adj_for_baseline=r_adj, p_adj_for_baseline=p_adj))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--target', choices=['age', 'stim'], required=True)
    ap.add_argument('--top', type=int, default=20)
    args = ap.parse_args(argv)

    if args.target == 'age':
        d = samples.frame('H1')
        d['age'] = pd.to_numeric(d['age'], errors='coerce')
        t = sweep_age(d)
        label = f'age (H1 sample, N = {len(d)})'
        cols = ['variable', 'n', 'r', 'p', 'q', 'r_young', 'r_old', 'band_d', 'band_p']
    else:
        d = samples.frame('H2')
        d['age'] = pd.to_numeric(d['age'], errors='coerce')
        # The seven preregistered DVs, plus the reach rate and the derived
        # behaviour measures (RT and choice dynamics). Widening the DV set
        # widens the multiplicity correction too, which is the honest trade:
        # looking in more places should make each hit harder to believe.
        dvs = [c for c in ['delta_p_stay_win', 'delta_p_shift_lose',
                           rl('delta_alpha'), rl('delta_beta'), 'delta_accuracy',
                           'delta_win_rate', ttc('delta_ttc'), 'delta_ttc_reach_rate',
                           'delta_rt_mean', 'delta_rt_cv', 'delta_rt_post_error',
                           'delta_rt_post_loss', 'delta_switch_rate',
                           'delta_persev_errors', 'delta_asymptotic_acc']
               if c in d.columns]
        t = sweep_stim(d, dvs)
        label = f'the stimulation response (H2 sample, N = {len(d)}; {len(dvs)} DVs)'
        cols = ['variable', 'dv', 'n', 'r', 'p', 'q', 'p_adj_for_baseline']

    t = t.sort_values('p').reset_index(drop=True)
    t['q'] = multipletests(t.p, method='fdr_bh')[1]

    n_var = t.variable.nunique()
    print(f'{len(t)} tests against {label}')
    print(f'{n_var} measures swept\n')
    print(t.head(args.top)[cols].to_string(index=False,
                                           float_format=lambda v: f'{v:.4f}'))
    hits, chance, surv = (t.p < .05).sum(), 0.05 * len(t), (t.q < .05).sum()
    print(f'\nuncorrected p < .05: {hits} (expected by chance {chance:.1f})')
    print(f'survive FDR (q < .05): {surv}')
    if surv == 0:
        print('  -> nothing here survives correction. Treat every row above as '
              'a hypothesis,\n     not a result.')
    elif args.target == 'age':
        print('  -> check `band_d` against `r_young`/`r_old`: a survivor with '
              'near-zero within-band\n     correlations is a group difference, '
              'not an age gradient.')

    out = str(OUT).format(target=args.target)
    t.to_csv(out, index=False)
    print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
