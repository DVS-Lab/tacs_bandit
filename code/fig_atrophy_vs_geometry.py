"""
fig_atrophy_vs_geometry.py — Atrophy is real, and it is not why dose falls

The intuitive account of the age-related dose reduction is atrophy: the brain
shrinks, so the cortex sits further from the electrode. This figure tests that
account properly, with FreeSurfer and charm morphometry, and it fails.

  a   atrophy is real and plainly visible -- ventricular CSF at the same
      anatomical level in a younger and an older participant
  b   every measure placed by its two correlations, with age and with the
      delivered field. The atrophy measures track age hard and sit flat
      against the field; the geometric measures do the reverse.
  c   the formal version: entered together, the atrophy block adds nothing
      over geometry, while geometry adds almost everything over atrophy.

Atrophy in this sample is not subtle -- total gray matter falls at r = -.77,
ventricles rise at +.65, cortex thins at -.64. The point is not that the brain
is unchanged. It is that none of that reaches the field: no atrophy measure
correlates with |E| at p < .05 except intracranial CSF (-.33), which then
drops out entirely once geometry is in the model.

**Volumes are normalised by a directly measured intracranial volume, not by
eTIV.** FreeSurfer derives eTIV from the determinant of the Talairach
registration rather than measuring it, and in this sample it rises with age at
r = +.47 (+12.5% over 40 years), which is not anatomically possible --
intracranial volume is fixed in adulthood. Counting labelled voxels instead
gives an ICV that is flat with age (r = -.011). The two agree on absolute size
(r = +.70), so only the age trend is spurious, and the conclusions are
unchanged either way (total GM x age: -.772 normalised directly vs -.764 by
eTIV). It is a correctness fix, not a result-changing one.

CAVEAT. |E| is modelled, and skull geometry is an input to that model, so the
geometry-to-field relationship partly reports which input the FEM is most
sensitive to. That does not rescue the atrophy account -- brain volume is an
input too, and it predicts nothing -- but the phrasing should stay
"contributor to variation in simulated dose".

Usage
-----
    python fig_atrophy_vs_geometry.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.ndimage import distance_transform_edt

from config import REPO_ROOT, EFIELD_CSV_PATH, FREESURFER_MORPH_PATH
from paper_style import (WIDTH_2COL, FONT_AXIS_TITLE, FONT_TICK,
                         FONT_PANEL_LABEL, REGRESSION_COLOR)
from fig_distance_anatomy import M2M_DIR

FIG_DIR = REPO_ROOT / 'data' / 'figures' / 'paper'
FIELD = 'mean_magnE'
ATROPHY_COLOR, GEOMETRY_COLOR = '#7E57C2', '#00897B'
CSF_TINT = '#4FC3F7'

# (column, label, family). Atrophy measures are normalised by the directly
# measured ICV; geometry measures are millimetres along the electrode-to-cortex
# ray and need no normalisation.
MEASURES = [
    ('gm_n',                'Total GM',        'atrophy'),
    ('cortex_n',            'Cortical GM',       'atrophy'),
    ('brain_n',             'Brain volume',      'atrophy'),
    ('vent_n',              'Ventricles',        'atrophy'),
    ('csf_n',               'CSF (whole head)', 'atrophy'),
    ('lh_dlpfc_thickness',  'DLPFC thickness',   'atrophy'),
    ('lh_mean_thickness',   'Mean thickness',    'atrophy'),
    ('layer_csf',           'CSF (local)',      'geometry'),
    ('layer_diploe',        'Diploe',            'geometry'),
    ('layer_skull',         'Skull',             'geometry'),
    ('dist_pial_dlpfc_p1',  'Scalp-to-cortex',  'geometry'),
]
GEO_BLOCK = ['dist_pial_dlpfc_p1', 'layer_skull']
ATR_BLOCK = ['gm_n', 'vent_n', 'lh_dlpfc_thickness', 'csf_n']


def load():
    e = pd.read_csv(EFIELD_CSV_PATH, dtype={'subject_id': str})
    fs = pd.read_csv(FREESURFER_MORPH_PATH, dtype={'subject_id': str})
    m = pd.read_csv(REPO_ROOT / 'data' / 'master_subject_data.csv',
                    dtype={'subject_id': str})
    d = (e.merge(fs, on='subject_id', how='left')
          .merge(m[['subject_id', 'age', 'gender']], on='subject_id', how='left'))
    d['age'] = pd.to_numeric(d['age'], errors='coerce')
    d = d.dropna(subset=['age'])
    if 't1_only' in d.columns:
        d = d[~d['t1_only'].astype(bool)]
    for src, out in [('total_gray_vol', 'gm_n'), ('cortex_vol', 'cortex_n'),
                     ('brainseg_not_vent', 'brain_n'),
                     ('ventricle_choroid_vol', 'vent_n'), ('csf_charm', 'csf_n')]:
        d[out] = d[src] / d['icv_charm']
    return d.reset_index(drop=True)


def ventricular_slice(sub_id):
    """
    Axial section at the level of maximum ventricular CSF.

    "Ventricular" is defined as CSF more than 18 mm inside the brain envelope.
    Taking the slice with the most CSF of any kind instead lands at the vertex,
    where sulcal and interhemispheric CSF dominate and the ventricles are not
    in view at all.
    """
    t = nib.as_closest_canonical(
        nib.load(str(M2M_DIR / f'm2m_sub-{sub_id}' / 'final_tissues.nii.gz')))
    t1 = nib.as_closest_canonical(
        nib.load(str(M2M_DIR / f'm2m_sub-{sub_id}' / 'T1.nii.gz')))
    lab = np.squeeze(t.get_fdata()).astype(int)
    brain = np.isin(lab, [1, 2, 3])
    depth = distance_transform_edt(brain, sampling=t.header.get_zooms()[:3])
    deep = (lab == 3) & (depth > 18)

    zs = np.where(brain.any(axis=(0, 1)))[0]
    z = zs[int(np.argmax([deep[:, :, k].sum() for k in zs]))]

    img, csf = t1.get_fdata()[:, :, z].T, (lab[:, :, z] == 3).T
    keep = brain[:, :, z].T
    rows, cols = np.where(keep)
    pad = 6
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad, img.shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad, img.shape[1])
    return img[r0:r1, c0:c1], csf[r0:r1, c0:c1]


def panel_slices(axes, d):
    """
    Axial sections at a common anatomical plane in a younger and an older
    participant.

    Both are taken at the subject-space z corresponding to MNI z = +18, the
    level of the lateral ventricular bodies, carried through each subject's own
    nonlinear warp. Letting each subject supply its own slice -- say the one
    with the most ventricular CSF -- would guarantee the most flattering view
    of the ventricles in each and put the two panels at different anatomical
    levels, so part of any visible difference would be a difference in where we
    cut rather than in the anatomy.
    """
    lo, hi = d['age'].quantile([1 / 3, 2 / 3])
    picks = []
    for band in ('young', 'old'):
        g = d[d.age <= lo] if band == 'young' else d[d.age >= hi]
        g = g.dropna(subset=['slice_z_mni18'])
        picks.append(g.iloc[(g.csf_n - g.csf_n.mean()).abs().argsort().iloc[0]])

    for ax, r, band in zip(axes, picks, ('Younger', 'Older')):
        m2m = M2M_DIR / f'm2m_sub-{r.subject_id}'
        t1 = nib.as_closest_canonical(nib.load(str(m2m / 'T1.nii.gz')))
        tis = nib.as_closest_canonical(nib.load(str(m2m / 'final_tissues.nii.gz')))
        lab = np.squeeze(tis.get_fdata()).astype(int)

        inv = np.linalg.inv(t1.affine)
        k = int(round((inv @ np.array([0, 0, float(r.slice_z_mni18), 1.0]))[2]))
        k = int(np.clip(k, 0, lab.shape[2] - 1))

        img, csf = t1.get_fdata()[:, :, k].T, (lab[:, :, k] == 3).T
        brain = np.isin(lab[:, :, k], [1, 2, 3]).T
        rows, cols = np.where(brain)
        pad = 6
        sl = (slice(max(rows.min() - pad, 0), min(rows.max() + pad, img.shape[0])),
              slice(max(cols.min() - pad, 0), min(cols.max() + pad, img.shape[1])))
        img, csf = img[sl], csf[sl]

        vmax = np.percentile(img[img > 0], 99.5) if (img > 0).any() else 1.0
        ax.imshow(img, cmap='gray', origin='lower', vmin=0, vmax=vmax,
                  interpolation='bilinear', aspect='equal')
        tint = np.zeros((*csf.shape, 4))
        tint[csf] = matplotlib.colors.to_rgba(CSF_TINT, 0.55)
        ax.imshow(tint, origin='lower', interpolation='nearest', aspect='equal')
        ax.set_title(f'{band} (age {r.age:.0f})\nCSF {100 * r.csf_n:.0f}% of ICV',
                     fontsize=FONT_TICK, pad=1.5, linespacing=1.35)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    return picks


def panel_dissociation(ax, d):
    """
    Every measure at (r with age, r with field).

    Both axes are shaded inside their significance band, because the claim is
    about both: a measure only explains the age effect on dose if it is clear
    of both. Every label uses the same leader-line treatment -- the shrinkage
    measures are too close together to label in place, and mixing styles across
    the panel reads as carelessness rather than as a considered choice.
    """
    n = len(d)
    crit = stats.t.ppf(0.975, n - 2)
    r_crit = crit / np.sqrt(n - 2 + crit ** 2)
    ax.axhspan(-r_crit, r_crit, color='#F2F2F2', zorder=0)
    ax.axvspan(-r_crit, r_crit, color='#F2F2F2', zorder=0)
    ax.axhline(0, color='#C4C4C4', lw=0.6, zorder=1)
    ax.axvline(0, color='#C4C4C4', lw=0.6, zorder=1)

    # Label anchors, ordered so leader lines never cross. Left column serves
    # the shrinkage cluster; right column serves everything else.
    ANCHOR = {'Brain volume': (-0.40, 0.360), 'Total GM': (-0.40, 0.268),
              'DLPFC thickness': (-0.40, 0.176), 'Mean thickness': (-0.40, 0.084),
              'Cortical GM': (-0.40, -0.008),
              'Ventricles': (0.95, -0.150), 'CSF (local)': (0.95, -0.268),
              'CSF (whole head)': (0.95, -0.386), 'Diploe': (0.95, -0.504),
              'Skull': (0.95, -0.760), 'Scalp-to-cortex': (0.95, -0.890)}

    for col, lab, fam in MEASURES:
        x = d[[col, 'age', FIELD]].dropna()
        ra = stats.pearsonr(x.age, x[col])[0]
        rf = stats.pearsonr(x[col], x[FIELD])[0]
        c = ATROPHY_COLOR if fam == 'atrophy' else GEOMETRY_COLOR
        ax.scatter(ra, rf, s=38, color=c, zorder=4, edgecolors='white',
                   linewidths=0.6)
        ax.annotate(lab, (ra, rf), xytext=ANCHOR[lab], textcoords='data',
                    fontsize=FONT_TICK - 1.5, color=c, va='center', ha='left',
                    arrowprops=dict(arrowstyle='-', lw=0.5, color=c,
                                    alpha=0.5, shrinkA=1, shrinkB=3))

    ax.set_xlabel('$r$ with age', fontsize=FONT_AXIS_TITLE, labelpad=1.5)
    ax.set_ylabel('$r$ with delivered field', fontsize=FONT_AXIS_TITLE,
                  labelpad=1.5)
    ax.set_xlim(-0.95, 1.62); ax.set_ylim(-1.02, 0.44)
    ax.set_xticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75])
    ax.tick_params(labelsize=FONT_TICK, pad=1.2)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.text(-0.90, -0.97, f'$N$ = {n}   shaded: $p$ > .05',
            fontsize=FONT_TICK - 1.5, color='#9E9E9E', va='bottom')
    for lab, c, y in [('Atrophy', ATROPHY_COLOR, -0.70),
                      ('Geometry', GEOMETRY_COLOR, -0.80)]:
        ax.scatter(-0.86, y, s=34, color=c, zorder=5)
        ax.text(-0.78, y, lab, fontsize=FONT_TICK, color=c,
                fontweight='bold', va='center')


def panel_partition(ax, d):
    """
    The variance in |E| partitioned into what each block explains uniquely and
    what they share.

    An earlier version showed each block's R2 with the other block's increment
    beside it, which read as though the significance test applied to the whole
    bar rather than to the increment. A commonality partition avoids that: the
    three segments sum to the full model and each is unambiguous.
    """
    x = d[[FIELD] + GEO_BLOCK + ATR_BLOCK].dropna()

    def fit(cols):
        return sm.OLS(x[FIELD], sm.add_constant(x[cols])).fit()

    geo, atr, full = fit(GEO_BLOCK), fit(ATR_BLOCK), fit(GEO_BLOCK + ATR_BLOCK)
    uniq_geo = full.rsquared - atr.rsquared
    uniq_atr = full.rsquared - geo.rsquared
    shared = full.rsquared - uniq_geo - uniq_atr

    def incr_p(base, added):
        f = ((base.ssr - full.ssr) / len(added)) / (full.ssr / full.df_resid)
        return 1 - stats.f.cdf(f, len(added), full.df_resid)

    p_geo, p_atr = incr_p(atr, GEO_BLOCK), incr_p(geo, ATR_BLOCK)

    segs = [(uniq_geo, GEOMETRY_COLOR, 1.0, 'Geometry only'),
            (shared, '#B0BEC5', 1.0, 'Shared'),
            (uniq_atr, ATROPHY_COLOR, 1.0, 'Atrophy only')]
    left = 0.0
    for val, colour, alpha, _ in segs:
        ax.barh(0, val, left=left, height=0.42, color=colour, alpha=alpha,
                edgecolor='white', linewidth=0.8, zorder=3)
        left += val

    ax.set_xlim(0, 0.90); ax.set_ylim(-0.95, 0.55)
    ax.set_yticks([])
    ax.set_xlabel('Variance in |E| explained ($R^2$)',
                  fontsize=FONT_AXIS_TITLE, labelpad=1.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75])
    ax.tick_params(labelsize=FONT_TICK, pad=1.2)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)

    star = lambda p: '***' if p < .001 else ('**' if p < .01 else
                                             ('*' if p < .05 else 'n.s.'))
    rows = [(f'Geometry only   {uniq_geo:.2f} {star(p_geo)}', GEOMETRY_COLOR),
            (f'Shared          {shared:.2f}', '#78909C'),
            (f'Atrophy only    {uniq_atr:.2f} {star(p_atr)}', ATROPHY_COLOR)]
    for i, (txt, c) in enumerate(rows):
        y = -0.32 - i * 0.20
        ax.add_patch(plt.Rectangle((0.01, y - 0.045), 0.045, 0.09,
                                   color=segs[i][1], clip_on=False, zorder=4))
        ax.text(0.075, y, txt, fontsize=FONT_TICK - 0.5, color=c, va='center',
                family='monospace')
    ax.text(0.90, 0.42, f'full model $R^2$ = {full.rsquared:.2f}',
            fontsize=FONT_TICK - 1, color=REGRESSION_COLOR, ha='right',
            va='bottom')
    return full.rsquared


def build(out_name: str = 'fig_atrophy_vs_geometry') -> Path:
    d = load()
    print(f'N = {len(d)} (T1-only excluded)')

    FIG_W, FIG_H = WIDTH_2COL, WIDTH_2COL * 0.46
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_a1 = fig.add_axes([0.010, 0.530, 0.180, 0.390])
    ax_a2 = fig.add_axes([0.010, 0.075, 0.180, 0.390])
    ax_b = fig.add_axes([0.268, 0.130, 0.430, 0.790])
    ax_c = fig.add_axes([0.748, 0.395, 0.246, 0.255])

    picks = panel_slices([ax_a1, ax_a2], d)
    panel_dissociation(ax_b, d)
    r2_full = panel_partition(ax_c, d)
    print(f'  sections: sub-{picks[0].subject_id} (age {picks[0].age:.0f}) and '
          f'sub-{picks[1].subject_id} (age {picks[1].age:.0f}); full model R2 = {r2_full:.3f}')

    LABEL_KW = dict(fontsize=FONT_PANEL_LABEL, fontweight='bold', va='top')
    for x, y, k in [(0.004, 0.992, 'a'), (0.200, 0.992, 'b'),
                    (0.706, 0.992, 'c')]:
        fig.text(x, y, k, **LABEL_KW)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f'{out_name}.png'
    fig.savefig(png, dpi=400, facecolor='white')
    fig.savefig(FIG_DIR / f'{out_name}.svg', dpi=400, facecolor='white')
    plt.close(fig)
    print(f'wrote {png}')
    return png


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--name', default='fig_atrophy_vs_geometry')
    args = p.parse_args(argv)
    build(out_name=args.name)
    return 0


if __name__ == '__main__':
    sys.exit(main())
