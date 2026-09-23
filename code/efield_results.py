"""
efield_results.py — Every E-field and anatomy number quoted in the handoff

ANALYSIS_HANDOFF.md §7 (E-field through FreeSurfer) quotes several dozen
statistics. Some came from figure scripts; many were computed inline and
existed only as text, so a change of sample -- adding 11542 once their age was
recovered -- could not be propagated without recomputing them by hand. This
script is now their source. Each block below corresponds to a handoff
subsection and prints its numbers with N.

Sample, unless a block says otherwise: FLAIR head models (T1-only excluded)
with age. The T1-only blocks use all head models with age.

The `age_bands` block exists because age was recruited in two bands rather
than sampled uniformly, so a linear age correlation is close to a two-group
contrast. It reports, for each anatomy measure, the gradient within each band
and the standardised difference between them.

Mediation is product-of-coefficients (a*b / c) with a percentile bootstrap CI
on the indirect effect, fixed seed. Commonality partition and the ICV-
normalised atrophy measures reuse fig_atrophy_vs_geometry so the figure and
the text cannot disagree.

`--exclude` drops subjects, which is how this script was checked: excluding
11542 reproduces the pre-recovery (N = 58) values in the handoff.

Usage
-----
    python efield_results.py
    python efield_results.py --exclude 11542
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from config import (REPO_ROOT, FREESURFER_PARCELS_PATH,
                    AGE_BAND_YOUNG_MAX, AGE_BAND_OLD_MIN)
import fig_atrophy_vs_geometry as fav
from samples import assert_sample

FIELD = 'mean_magnE'
DIST = 'dist_pial_dlpfc_p1'
OUT = REPO_ROOT / 'derivatives' / 'efield_results.csv'
SEED = 20260921
N_BOOT = 5000

ROWS: List[dict] = []


def rec(block, key, value, n, note=''):
    ROWS.append(dict(block=block, key=key, value=value, n=n, note=note))
    v = f'{value:+.3f}' if isinstance(value, float) and abs(value) < 100 else f'{value}'
    print(f'  {key:46} {v:>12}   n={n}{"   " + note if note else ""}')


def r_p(x, y):
    m = pd.concat([x, y], axis=1).dropna()
    r, p = stats.pearsonr(m.iloc[:, 0], m.iloc[:, 1])
    return r, p, len(m)


def ols(d, y, xs):
    m = d[[y] + xs].dropna()
    return sm.OLS(m[y], sm.add_constant(m[xs])).fit(), len(m)


def mediation(d, x, m, y, rng, n_boot=N_BOOT):
    s = d[[x, m, y]].dropna()

    def paths(t):
        c = sm.OLS(t[y], sm.add_constant(t[[x]])).fit().params[x]
        a = sm.OLS(t[m], sm.add_constant(t[[x]])).fit().params[x]
        full = sm.OLS(t[y], sm.add_constant(t[[x, m]])).fit()
        return a, full.params[m], c, full.pvalues[x]

    a, b, c, p_direct = paths(s)
    boot = [np.prod(paths(s.iloc[rng.integers(0, len(s), len(s))])[:2])
            for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(prop=a * b / c, lo=lo, hi=hi, p_direct=p_direct, n=len(s))


def load(exclude):
    # fav.load drops T1-only; the T1-only blocks need them, so load both ways.
    import fig_atrophy_vs_geometry as f
    e = pd.read_csv(f.EFIELD_CSV_PATH, dtype={'subject_id': str})
    fs = pd.read_csv(f.FREESURFER_MORPH_PATH, dtype={'subject_id': str})
    m = pd.read_csv(REPO_ROOT / 'data' / 'master_subject_data.csv',
                    dtype={'subject_id': str})
    allh = (e.merge(fs, on='subject_id', how='left')
             .merge(m[['subject_id', 'age', 'gender']], on='subject_id', how='left'))
    allh['age'] = pd.to_numeric(allh['age'], errors='coerce')
    allh = allh.dropna(subset=['age'])
    allh = allh[~allh.subject_id.isin(exclude)].reset_index(drop=True)
    flair = f.load()
    flair = flair[~flair.subject_id.isin(exclude)].reset_index(drop=True)
    if not exclude:
        # The primary sample is defined in samples.py; stop if this one differs.
        assert_sample(flair['subject_id'], 'Dose')
    return flair, allh


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--exclude', nargs='*', default=[])
    args = ap.parse_args(argv)
    d, allh = load(set(args.exclude))
    rng = np.random.default_rng(SEED)
    if args.exclude:
        print(f'excluding: {args.exclude}')
    print(f'FLAIR sample N = {len(d)}; all head models N = {len(allh)}')

    # ---- The one notable positive --------------------------------------
    print('\nAge x E-field')
    B = 'age_x_efield'
    for col, lab in [(FIELD, 'mean'), ('p95_magnE', 'p95'),
                     ('peak_magnE', 'peak'), ('median_magnE', 'median')]:
        r, p, n = r_p(d.age, d[col])
        rec(B, f'r age x {lab} |E|', r, n, f'p = {p:.4f}')
    r, p, n = r_p(allh.age, allh[FIELD])
    rec(B, 'r age x |E|, T1-only included', r, n, f'p = {p:.4f}')
    fit, n = ols(allh.assign(t1=allh.t1_only.astype(float)), FIELD, ['age', 't1'])
    flair_mean = allh.loc[~allh.t1_only.astype(bool), FIELD].mean()
    t1_mean = allh.loc[allh.t1_only.astype(bool), FIELD].mean()
    rec(B, 'T1-only |E| vs FLAIR, raw means, %', 100 * (t1_mean / flair_mean - 1), n)
    rec(B, 'T1-only |E| difference, age-adjusted, %', 100 * fit.params['t1'] / flair_mean, n,
        f'p = {fit.pvalues["t1"]:.4f} controlling for age')

    # ---- Gradient or group difference? ---------------------------------
    # Age was recruited in two bands, not sampled uniformly: almost everyone is
    # under 40 or over 55, and only a handful fall in between (the band n are
    # printed below; they differ by sample). A Pearson r over that distribution
    # is arithmetically close to a two-group contrast, so "|E| declines with
    # age" claims a gradient the design cannot show. This block separates the
    # two: the correlation *within* each band is the gradient, the standardised
    # difference *between* bands is the group effect. An effect carried entirely by the between-band term should be
    # reported as a group difference (Welch t, Cohen's d) with the correlation
    # alongside it, not instead of it.
    print('\nAge bands: gradient within vs difference between')
    B = 'age_bands'
    YOUNG, OLD = AGE_BAND_YOUNG_MAX, AGE_BAND_OLD_MIN
    yb_all, ob_all = d[d.age < YOUNG], d[d.age >= OLD]
    rec(B, f'band n, younger than {YOUNG}', len(yb_all), len(d))
    rec(B, f'band n, {OLD} and over', len(ob_all), len(d))
    rec(B, f'band n, {YOUNG}-{OLD - 1} (excluded from the contrast)',
        int(((d.age >= YOUNG) & (d.age < OLD)).sum()), len(d))
    for col, lab in [(FIELD, '|E| DLPFC'), (DIST, 'scalp-cortex dist'),
                     ('layer_skull', 'skull thickness'),
                     ('lh_dlpfc_thickness', 'DLPFC thickness'),
                     ('csf_charm', 'whole-head CSF')]:
        if col not in d.columns:
            continue
        r_o, p_o, n_o = r_p(d.age, d[col])
        rec(B, f'{lab}: r x age, whole sample', r_o, n_o, f'p = {p_o:.4f}')
        for band, blab in [(yb_all, 'younger band'), (ob_all, 'older band')]:
            m = band[['age', col]].dropna()
            if len(m) > 5:
                r_b, p_b = stats.pearsonr(m.age, m[col])
                rec(B, f'{lab}: r x age within {blab}', r_b, len(m), f'p = {p_b:.3f}')
        y, o = yb_all[col].dropna(), ob_all[col].dropna()
        if len(y) > 2 and len(o) > 2:
            t, p_t = stats.ttest_ind(y, o, equal_var=False)
            pooled = np.sqrt((y.var(ddof=1) + o.var(ddof=1)) / 2)
            rec(B, f'{lab}: between-band d (old - young)',
                (o.mean() - y.mean()) / pooled, len(y) + len(o),
                f't({stats.ttest_ind(y, o, equal_var=False).df:.0f}) = {t:+.2f}, p = {p_t:.4f}')

    print('\nROI coverage')
    B = 'roi_coverage'
    cov = 'n_gm_elements_roi'
    rec(B, 'GM elements in ROI, min', float(d[cov].min()), len(d))
    rec(B, 'GM elements in ROI, max', float(d[cov].max()), len(d))
    r, p, n = r_p(d.age, d[cov]); rec(B, 'r coverage x age', r, n, f'p = {p:.4f}')
    r, p, n = r_p(d[cov], d[FIELD]); rec(B, 'r coverage x |E|', r, n, f'p = {p:.4f}')
    fit, n = ols(d, FIELD, ['age', cov])
    rec(B, 'age effect on |E| controlling coverage, p', fit.pvalues['age'], n)
    r, p, n = r_p(d.age, d['parcel_mean_magnE'])
    rec(B, 'r age x parcel |E|', r, n, f'p = {p:.4f}')
    r, p, n = r_p(d['parcel_mean_magnE'], d[FIELD]); rec(B, 'r parcel x sphere |E|', r, n)

    # ---- Mechanism -----------------------------------------------------
    print('\nMechanism: geometry, not thinning')
    B = 'mechanism'
    r, p, n = r_p(d[DIST], d[FIELD]); rec(B, 'r distance x |E|', r, n, f'p = {p:.2g}')
    for col, lab in [('charm_dlpfc_thickness', 'charm'), ('lh_dlpfc_thickness', 'FreeSurfer')]:
        r, p, n = r_p(d[col], d[FIELD]); rec(B, f'r DLPFC thickness x |E| ({lab})', r, n, f'p = {p:.3f}')
    med = mediation(d, 'age', DIST, FIELD, rng)
    rec(B, 'distance mediation, % mediated', 100 * med['prop'], med['n'],
        f"indirect CI [{med['lo']:+.5f}, {med['hi']:+.5f}], direct p = {med['p_direct']:.3f}")
    for v in ['dist_pial_dlpfc_min', 'dist_central_dlpfc_min', 'dist_central_dlpfc_p1',
              'dist_pial_p1', 'dist_central_p1']:
        mv = mediation(d, 'age', v, FIELD, rng, n_boot=1000)
        sig = 'sig' if (mv['hi'] < 0 or mv['lo'] > 0) else 'n.s.'
        rec(B, f'  variant {v}, % mediated', 100 * mv['prop'], mv['n'], sig)
    r, p, n = r_p(d.age, d[DIST]); rec(B, 'a-path: r age x distance', r, n, f'p = {p:.4f}')

    print('\nFreeSurfer vs charm')
    B = 'two_pipelines'
    # Pipeline agreement needs no age and is not affected by the T1-only
    # issue (FreeSurfer is T1-only for everyone), so it uses every head model.
    r, _, n = r_p(allh['charm_mean_thickness'], allh['lh_mean_thickness'])
    rec(B, 'r charm x FS thickness, whole hemisphere', r, n, 'all head models')
    r, _, n = r_p(allh['charm_dlpfc_thickness'], allh['lh_dlpfc_thickness'])
    rec(B, 'r charm x FS thickness, DLPFC', r, n, 'all head models')
    for col, lab in [('charm_dlpfc_thickness', 'charm'), ('lh_dlpfc_thickness', 'FS')]:
        r, p, n = r_p(d.age, d[col]); rec(B, f'r age x DLPFC thickness ({lab})', r, n, f'p = {p:.2g}')
        r, p, n = r_p(d[col], d[DIST]); rec(B, f'r thickness x distance ({lab})', r, n, f'p = {p:.3f}')
        fit, n = ols(d, DIST, ['age', col])
        rec(B, f'thickness explains age->distance, p ({lab})', fit.pvalues[col], n)
        fit, n = ols(d, FIELD, [DIST, col])
        rec(B, f'horse race: thickness | distance, p ({lab})', fit.pvalues[col], n)

    print('\nAtrophy vs geometry')
    B = 'atrophy'
    for col, lab in [('gm_n', 'total GM / ICV'), ('vent_n', 'ventricles / ICV'),
                     ('lh_dlpfc_thickness', 'DLPFC thickness'), ('csf_n', 'whole-head CSF / ICV'),
                     ('cortex_n', 'cortical GM / ICV'), ('brain_n', 'brain / ICV'),
                     ('lh_mean_thickness', 'mean thickness')]:
        ra, pa, n = r_p(d.age, d[col]); rf, pf, _ = r_p(d[col], d[FIELD])
        rec(B, f'{lab}: r x age', ra, n, f'p = {pa:.2g}')
        rec(B, f'{lab}: r x |E|', rf, n, f'p = {pf:.3f}')
    x = d[[FIELD] + fav.GEO_BLOCK + fav.ATR_BLOCK].dropna()
    fit = lambda c: sm.OLS(x[FIELD], sm.add_constant(x[c])).fit()
    geo, atr, full = fit(fav.GEO_BLOCK), fit(fav.ATR_BLOCK), fit(fav.GEO_BLOCK + fav.ATR_BLOCK)
    ug, ua = full.rsquared - atr.rsquared, full.rsquared - geo.rsquared

    def incr_p(base, k):
        f_ = ((base.ssr - full.ssr) / k) / (full.ssr / full.df_resid)
        return 1 - stats.f.cdf(f_, k, full.df_resid)
    rec(B, 'commonality: geometry unique R2', ug, len(x), f'p = {incr_p(atr, len(fav.GEO_BLOCK)):.2g}')
    rec(B, 'commonality: atrophy unique R2', ua, len(x), f'p = {incr_p(geo, len(fav.ATR_BLOCK)):.3f}')
    rec(B, 'commonality: shared R2', full.rsquared - ug - ua, len(x))
    rec(B, 'commonality: full R2', full.rsquared, len(x))
    r, p, n = r_p(d.age, d['etiv'])
    sl = sm.OLS(d['etiv'], sm.add_constant(d[['age']])).fit().params['age']
    rec(B, 'r eTIV x age', r, n, f'p = {p:.3f}; slope x 40 y = {100 * sl * 40 / d.etiv.mean():+.1f}% of mean eTIV')
    r, p, n = r_p(d.age, d['icv_charm']); rec(B, 'r measured ICV x age', r, n, f'p = {p:.3f}')

    print('\nSex')
    B = 'sex'
    s = d.dropna(subset=['layer_skull', 'gender']).copy()
    s['female'] = (s.gender == 'Female').astype(float)
    s['age_x_f'] = s.age * s.female
    fit = sm.OLS(s.layer_skull, sm.add_constant(s[['age', 'female', 'age_x_f']])).fit()
    rec(B, 'age x sex on skull, p', fit.pvalues['age_x_f'], len(s))
    rec(B, 'age main effect with interaction, p', fit.pvalues['age'], len(s))
    lo_t, hi_t = s.age.quantile([1 / 3, 2 / 3])
    for g in ['Female', 'Male']:
        t = s[s.gender == g]
        r, p, n = r_p(t.age, t.layer_skull); rec(B, f'{g}: r skull x age', r, n, f'p = {p:.4f}')
        diff = t[t.age >= hi_t].layer_skull.mean() - t[t.age <= lo_t].layer_skull.mean()
        rec(B, f'{g}: older - younger tertile skull (mm)', diff, n)
    rec(B, 'women in the older tertile', int(((s.age >= hi_t) & (s.gender == 'Female')).sum()), len(s))

    # The whole dose mechanism is sex-moderated, so report the chain by sex
    # rather than only the skull link. In women every step is present; in men
    # the age->distance step is absent (r = .00) and the mediated path is zero.
    #
    # What this does NOT support: a menopause-timed onset. Women in this sample
    # are 23-32 and 56-76 with two people in between, so a "step at 50" is the
    # recruitment gap, and no within-band gradient is resolvable (older women
    # n = 11, all within-band p > .17). The defensible claim is a group one:
    # the age-related skull and distance difference is present in women and
    # absent in men.
    print('\nDose mechanism by sex')
    B = 'sex_mechanism'
    sm_ = d.dropna(subset=['gender']).copy()
    for g in ['Female', 'Male']:
        t = sm_[sm_.gender == g]
        for col, lab in [(DIST, 'age x distance'), (FIELD, 'age x |E|'),
                         ('layer_skull', 'age x skull')]:
            r, p, n = r_p(t.age, t[col])
            rec(B, f'{g}: {lab}', r, n, f'p = {p:.4f}')
        mv = mediation(t, 'age', DIST, FIELD, rng, n_boot=2000)
        sig = 'CI excludes 0' if (mv['hi'] < 0 or mv['lo'] > 0) else 'CI includes 0'
        rec(B, f'{g}: % of age->|E| mediated by distance', 100 * mv['prop'], mv['n'],
            f"[{mv['lo']:+.5f}, {mv['hi']:+.5f}] {sig}")
    t = sm_[['age', 'layer_skull', 'icv_charm']].join(
        (sm_.gender == 'Female').astype(float).rename('female')).dropna()
    t['age_x_f'] = t.age * t.female
    fit = sm.OLS(t.layer_skull, sm.add_constant(t[['age', 'female', 'age_x_f', 'icv_charm']])).fit()
    rec(B, 'age x sex on skull, controlling ICV, p', fit.pvalues['age_x_f'], len(t),
        'women have smaller heads (ICV d = -1.33), so ICV is controlled')
    rec(B, 'older women (55+) in the FLAIR sample',
        int(((sm_.age >= OLD) & (sm_.gender == 'Female')).sum()), len(sm_),
        'the whole female age effect rests on these')

    print('\nT1-only head models, controlling for age and sex')
    B = 't1_only'
    t = allh.dropna(subset=['gender']).copy()
    t['female'] = (t.gender == 'Female').astype(float)
    t['t1'] = t.t1_only.astype(float)
    for col in ['layer_csf', 'csf_charm', FIELD, 'layer_skull', 'layer_diploe', 'layer_inner_table']:
        fit, n = ols(t, col, ['age', 'female', 't1'])
        rec(B, f'{col}: T1-only effect, p', fit.pvalues['t1'], n)

    print('\nGlobal vs DLPFC-specific atrophy (FreeSurfer)')
    B = 'global_atrophy'
    fit, n = ols(d, 'lh_dlpfc_thickness', ['age', 'lh_mean_thickness'])
    rec(B, 'DLPFC thickness ~ age | mean thickness: b', fit.params['age'], n,
        f'p = {fit.pvalues["age"]:.3f}')
    lp = pd.read_csv(FREESURFER_PARCELS_PATH, dtype={'subject_id': str})
    lp = lp[(lp.hemi == 'lh') & lp.subject_id.isin(d.subject_id)]
    wide = lp.pivot_table(index='subject_id', columns='parcel', values='thick_avg')
    ages = d.set_index('subject_id').age
    rs = {p_: r_p(ages.reindex(wide.index), wide[p_])[0] for p_ in wide.columns}
    rs = pd.Series(rs).sort_values()
    for p_ in ['rostralmiddlefrontal', 'caudalmiddlefrontal']:
        rec(B, f'{p_}: r x age (rank of {len(rs)})', float(rs[p_]), len(wide),
            f'rank {int((rs < rs[p_]).sum()) + 1}')
    rec(B, 'median parcel r x age', float(rs.median()), len(wide))

    print('\nFreeSurfer batch effect (delivery | age)')
    B = 'batch'
    b = allh.copy()
    b['set2'] = b.fs_delivery.str.contains('set2').astype(float)
    ps = []
    for col in ['lh_dlpfc_thickness', 'lh_mean_thickness', 'cortex_vol', 'etiv', 'surface_holes']:
        fit, n = ols(b, col, ['age', 'set2']); ps.append(fit.pvalues['set2'])
        rec(B, f'{col}: delivery effect, p', fit.pvalues['set2'], n)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ROWS).assign(excluded=' '.join(args.exclude)).to_csv(OUT, index=False)
    print(f'\nwrote {OUT.relative_to(REPO_ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
