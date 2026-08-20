"""
fig_skull_layers.py — Skull thickening under the electrode is a female effect

The skull vault is a sandwich rather than a solid plate, so a ray from the
electrode to cortex crosses five layers in a fixed order:

    scalp -> outer table -> diploe -> inner table -> CSF -> cortex

Outer and inner table are the same material, compact bone; they are the outer
and inner faces of the sandwich, separated by the marrow-filled diploe.

Four panels:

  a   the layers themselves, on one participant's scan, resampled so the
      electrode-to-cortex ray runs vertically and each tissue becomes a
      labelled horizontal band -- an anatomical key for the rest of the figure
  b   layer thickness by sex and age tertile
  c   the interaction: skull thickness against age, fit separately by sex
  d   who is actually in the sample, which is what limits panel c

**The central result is an interaction, not a main effect.** Pooled, skull
thickness rises with age (r = +.34). Split by sex, that is carried entirely by
women (r = +.67, p < .001) with nothing in men (r = -.04, p = .83); the
age x sex interaction is p = .003 for total skull and p < .001 for diploe, and
the main effect of age disappears once it is in the model (p = .81). Older
women's skulls are 2.63 mm thicker than younger women's, against 0.19 mm in
men. This matches the hyperostosis frontalis interna literature -- frontal
inner-table and diploic thickening, strongly female-predominant after
menopause -- and F3 sits over frontal bone.

CSF runs the other way: it increases with age in both sexes (women +.41, men
+.57). So CSF is the general ageing effect here and skull is the sex-specific
one, which is the reverse of the earlier reading.

**What replicates independently, and what does not.** Measuring the same span
from raw T1 intensity, with no segmentation labels, reproduces the *age* effect
(all r = +.36 p = .006; women +.49 p = .007) -- so skull thickening with age is
not charm inventing a boundary. But two things do not survive the independent
measure:

  - the age x sex interaction reaches only p = .18 (men +.17 rather than -.04)
  - the mediation of age -> |E| drops from 89% to 18%, and is not significant

The reason is circularity, and it is worth stating plainly. |E| is *computed
from* charm's segmentation, so charm's skull is an input to the model that
produced the field:

    charm skull        x |E| = -.778
    T1 intensity span  x |E| = -.231 (p = .081)

The two skull measures agree with each other at only r = +.40. So the very
strong skull-to-field relationship is substantially a statement about which
input the FEM is most sensitive to, not about how much current a thicker skull
actually blocks. The anatomical claim (women's skulls thicken with age under
the electrode) stands on its own; the causal claim about dose does not, on this
evidence.

**Do not interpret the layer split.** QC images (fig_qc_skull) show the diploe
segmented as scattered islands inside compact bone rather than a continuous
stratum, in FLAIR and T1-only subjects alike. Total skull has clean, anatomically
plausible boundaries; the outer/diploe/inner decomposition does not, and its
per-subject values are noisier than the correlations imply. Report total skull.

**The T1-only exclusion is not about skull.** Controlling for age and sex, the
seven T1-only head models do not differ on any skull measure (total p = .58,
diploe p = .45, inner table p = .90). They differ in CSF (p = .025) and in |E|
(p = .018). The exclusion is justified -- their fields are systematically lower,
in the direction of the hypothesis -- but the earlier stated rationale, that
skull segmentation degrades without a FLAIR, is not supported here.

CAVEAT ON INTERPRETATION. Skull geometry is an input to the FEM that produces
the simulated field, so a skull-to-|E| relationship partly reports which input
the model is most sensitive to. The fitted per-millimetre coefficients
reproduce SimNIBS's assumed conductivity ratio almost exactly (3.13 assumed,
3.14 fitted). The defensible phrasing is "anatomical contributor to variation
in simulated dose", not delivered current.

Usage
-----
    python fig_skull_layers.py
"""

from __future__ import annotations

import argparse
import sys
from itertools import groupby
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import map_coordinates

from config import REPO_ROOT, EFIELD_CSV_PATH
from paper_style import (WIDTH_2COL, FONT_AXIS_TITLE, FONT_TICK,
                         FONT_PANEL_LABEL, REGRESSION_COLOR, AGE_YOUNG, AGE_OLD)
from fig_distance_anatomy import (DIST, ELECTRODE_COLOR, M2M_DIR,
                                  load_subjects, representative)

FIG_DIR = REPO_ROOT / 'data' / 'figures' / 'paper'
FIELD = 'mean_magnE'

# Ordered outward-in. Colours follow SimNIBS's final_tissues LUT; the two
# compact-bone layers get two shades of one yellow rather than two distinct
# colours, because they are the same tissue separated by position along the
# ray, not by segmentation.
LAYERS = [
    ('layer_scalp',       'Scalp',       '#F2A488'),
    ('layer_outer_table', 'Outer table', '#FFE9A8'),
    ('layer_diploe',      'Diploe',      '#F58B3C'),
    ('layer_inner_table', 'Inner table', '#E6C97A'),
    ('layer_csf',         'CSF',         '#5B9BD5'),
]
SKULL_COLOR = '#B36A16'
SEX_COLORS = {'Female': '#C2185B', 'Male': '#00796B'}
SCALP, CSF_L, COMPACT, SPONGY = 5, 3, 7, 8


def load():
    e = pd.read_csv(EFIELD_CSV_PATH, dtype={'subject_id': str})
    m = pd.read_csv(REPO_ROOT / 'data' / 'master_subject_data.csv',
                    dtype={'subject_id': str})
    d = e.merge(m[['subject_id', 'age', 'gender']], on='subject_id', how='left')
    d['age'] = pd.to_numeric(d['age'], errors='coerce')
    d = d.dropna(subset=['age', 'gender'] + [c for c, _, _ in LAYERS])
    if 't1_only' in d.columns:
        d = d[~d['t1_only'].astype(bool)]
    return d.reset_index(drop=True)


def ray_aligned_section(sub_id, f3, half_w_mm=21.0, half_h_mm=21.0, step=0.15):
    """
    A section resampled so the electrode-to-cortex ray runs vertically.

    fig_distance_anatomy orients its sections to the superior axis, which is
    right for showing where the measurement sits in the head. Here the point is
    the layered structure the ray passes through, so the ray becomes the
    vertical axis and each tissue becomes a horizontal band that can be
    labelled -- scalp at the top, cortex at the bottom.

    The field of view is wide enough to read as an anatomical image rather than
    a strip: at a few millimetres the layers are unambiguous but the panel
    stops looking like a brain, which defeats the point of showing one.
    """
    m2m = M2M_DIR / f'm2m_sub-{sub_id}'
    t1 = nib.as_closest_canonical(nib.load(str(m2m / 'T1.nii.gz')))
    tis = nib.as_closest_canonical(nib.load(str(m2m / 'final_tissues.nii.gz')))
    lab_vol = np.squeeze(tis.get_fdata()).astype(np.int16)

    pial = nib.load(str(m2m / 'surfaces' / 'lh.pial.gii')).agg_data()[0]
    near = pial[np.argmin(np.linalg.norm(pial - f3, axis=1))]

    u = near - f3
    path = float(np.linalg.norm(u))
    u = u / path
    z = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(u, z)
    e1 = e1 / np.linalg.norm(e1)          # horizontal, perpendicular to the ray
    e2 = -u                                # up in the image = back toward scalp

    hw, hh = int(half_w_mm / step), int(half_h_mm / step)
    # Offset toward cortex rather than sitting on the ray midpoint: half the
    # frame would otherwise be air above the scalp, which is what made this
    # panel read as a strip instead of an anatomical image.
    centre = (f3 + near) / 2.0 + u * (0.30 * half_h_mm)
    a = (np.arange(-hw, hw) + 0.5) * step
    b = (np.arange(-hh, hh) + 0.5) * step
    A, B = np.meshgrid(a, b, indexing='xy')
    world = (centre[None, None, :] + A[..., None] * e1[None, None, :]
             + B[..., None] * e2[None, None, :])

    inv = np.linalg.inv(t1.affine)
    vox = (world @ inv[:3, :3].T) + inv[:3, 3]
    coords = np.stack([vox[..., 0], vox[..., 1], vox[..., 2]])
    img = map_coordinates(t1.get_fdata(), coords, order=1, mode='constant')
    lab = map_coordinates(lab_vol, coords, order=0, mode='constant')

    # Endpoint rows by projecting the actual points onto e2, rather than
    # assuming they sit symmetrically about the centre row. They do not: the
    # view is deliberately offset toward cortex, and hard-coding the midpoint
    # relationship drew the ray several millimetres too deep -- putting the
    # electrode marker inside the skull and dragging every label with it.
    def to_row(pt):
        return float(np.dot(pt - centre, e2)) / step + hh

    f3_row, cortex_row = to_row(f3), to_row(near)
    return img, lab, path, f3_row, cortex_row, step, (2 * hh, 2 * hw)


def panel_section(ax, row, half_w_mm, half_h_mm):
    img, lab, path, f3_row, cortex_row, step, shape = ray_aligned_section(
        row.subject_id, np.array([row.f3_x, row.f3_y, row.f3_z], float),
        half_w_mm, half_h_mm)

    vmax = np.percentile(img[img > 0], 99.5) if (img > 0).any() else 1.0
    ax.imshow(img, cmap='gray', origin='lower', vmin=0, vmax=vmax,
              interpolation='bilinear')

    codes = {SCALP: '#F2A488', COMPACT: '#FFE9A8', SPONGY: '#F58B3C',
             CSF_L: '#5B9BD5'}
    tint = np.zeros((*lab.shape, 4))
    for code, hexcol in codes.items():
        tint[lab == code] = matplotlib.colors.to_rgba(hexcol, 0.55)
    ax.imshow(tint, origin='lower', interpolation='nearest')

    H, W = shape
    # Labels come from this subject's own median layer thicknesses -- the same
    # numbers panel b plots -- rather than from run-lengths down the centre
    # column, which can hold stray extra runs of compact bone and produced a
    # duplicated "Inner table".
    at = 0.0
    for col, label, _ in LAYERS:
        L = float(row[col])
        if L >= 0.6:
            r = f3_row - (at + L / 2) / step
            ax.annotate(label, xy=(W * 0.54, r), xytext=(W * 1.04, r),
                        fontsize=FONT_TICK - 0.5, va='center', ha='left',
                        color=REGRESSION_COLOR, annotation_clip=False,
                        arrowprops=dict(arrowstyle='-', lw=0.6,
                                        color='#9E9E9E', shrinkA=0, shrinkB=1))
        at += L

    ax.plot([W / 2, W / 2], [f3_row, cortex_row], color='white', lw=0.9,
            ls=(0, (2.5, 1.5)), zorder=4)
    ax.plot(W / 2, cortex_row, marker='o', color='white', markersize=2.2, zorder=5)
    ax.plot(W / 2, f3_row, marker='v', color=ELECTRODE_COLOR, markersize=5,
            markeredgecolor='white', markeredgewidth=0.6, zorder=5)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def panel_composition(ax, d):
    """Layer thickness by sex and age tertile, oriented to match panel a."""
    lo, hi = d['age'].quantile([1 / 3, 2 / 3])
    groups = [('Younger', 'Female'), ('Older', 'Female'),
              ('Younger', 'Male'), ('Older', 'Male')]
    xs = [0, 1, 2.5, 3.5]
    labels = []

    for x, (band, sex) in zip(xs, groups):
        g = d[(d.gender == sex) &
              ((d.age <= lo) if band == 'Younger' else (d.age >= hi))]
        at = 0.0
        for col, _, colour in LAYERS:
            w = g[col].mean()
            ax.bar(x, w, bottom=at, width=0.78, color=colour,
                   edgecolor='white', linewidth=0.6)
            at += w
        ax.text(x, -0.7, f'{at:.1f}', ha='center', va='bottom',
                fontsize=FONT_TICK - 0.5, color=REGRESSION_COLOR,
                fontweight='bold')
        # n goes in the tick label rather than floating beside the bar, where
        # it collided with the group headings.
        labels.append(f'{band}\n($n$={len(g)})')

    # Group headings above the totals, so nothing sits under the bars.
    for x, sex, key in [(0.5, 'Women', 'Female'), (3.0, 'Men', 'Male')]:
        ax.text(x, -2.6, sex, ha='center', va='bottom',
                fontsize=FONT_AXIS_TITLE, color=SEX_COLORS[key],
                fontweight='bold')

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=FONT_TICK - 1)
    ax.set_ylabel('Depth from scalp surface (mm)', fontsize=FONT_AXIS_TITLE,
                  labelpad=1.5)
    ax.set_ylim(17.6, -4.2)
    ax.set_xlim(-0.75, 4.25)
    ax.set_yticks([0, 5, 10, 15])
    ax.tick_params(labelsize=FONT_TICK, pad=1.2)
    ax.tick_params(axis='x', length=0)
    for sp in ('top', 'right', 'bottom'):
        ax.spines[sp].set_visible(False)


def panel_interaction(ax, d):
    """Skull thickness against age, fit separately within each sex."""
    for sex in ('Female', 'Male'):
        g = d[d.gender == sex]
        c = SEX_COLORS[sex]
        ax.scatter(g.age, g.layer_skull, s=16, color=c, alpha=0.75,
                   edgecolors='white', linewidths=0.4, zorder=3)
        sl, ic, r, p, _ = stats.linregress(g.age, g.layer_skull)
        xs = np.linspace(g.age.min(), g.age.max(), 100)
        ax.plot(xs, ic + sl * xs, color=c, lw=1.4, zorder=4,
                ls='-' if p < .05 else (0, (3, 2)))
        lbl = 'Women' if sex == 'Female' else 'Men'
        txt = (f'{lbl}  $r$ = {r:.2f}, $p$ < .001' if p < .001
               else f'{lbl}  $r$ = {r:.2f}, $p$ = {p:.2f}')
        pos = dict(x=0.03, y=0.97, ha='left', va='top') if sex == 'Female' \
            else dict(x=0.97, y=0.04, ha='right', va='bottom')
        ax.text(pos['x'], pos['y'], txt, transform=ax.transAxes,
                ha=pos['ha'], va=pos['va'], fontsize=FONT_TICK - 0.5, color=c,
                fontweight='bold' if p < .05 else 'normal')

    ax.set_xlabel('Age (years)', fontsize=FONT_AXIS_TITLE, labelpad=1.5)
    ax.set_ylabel('Skull thickness (mm)', fontsize=FONT_AXIS_TITLE, labelpad=1.5)
    ax.tick_params(labelsize=FONT_TICK, pad=1.2)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def panel_sample(ax, d):
    """
    Who is actually in the sample, as densities over a strip of individuals.

    Panel c rests on this: the female slope is estimated from few older women,
    because women here skew young and men old. The density curves make the
    shapes comparable at a glance -- both sexes are n = 29, and both curves are
    scaled by the same constant, so the areas under them are equal and the
    difference is entirely in where each sex sits along the axis.

    The imbalance runs *against* the effect rather than creating it: the sex
    whose skull changes with age is the one thinnest on the ground at older
    ages, which dilutes the pooled estimate relative to the within-women one.
    """
    lo, hi = d['age'].quantile([1 / 3, 2 / 3])
    grid = np.linspace(d.age.min() - 4, d.age.max() + 4, 300)

    # One scale for both curves so the areas stay comparable.
    # Narrower bandwidth than Scott's default, which at n = 29 over a 56-year
    # span smooths both distributions almost flat and hides the very skew this
    # panel exists to show.
    kdes = {sex: stats.gaussian_kde(d[d.gender == sex].age.values,
                                    bw_method=0.30)(grid)
            for sex in ('Female', 'Male')}
    scale = 0.78 / max(k.max() for k in kdes.values())

    for base, sex, sign in [(1.0, 'Female', +1), (0.0, 'Male', -1)]:
        c = SEX_COLORS[sex]
        g = d[d.gender == sex]
        ax.fill_between(grid, base, base + sign * kdes[sex] * scale,
                        color=c, alpha=0.17, linewidth=0, zorder=1)
        ax.plot(grid, base + sign * kdes[sex] * scale, color=c, lw=1.0,
                alpha=0.75, zorder=2)
        ax.plot(grid, np.full_like(grid, base), color='#D8D8D8', lw=0.6, zorder=1)

        rng = np.random.default_rng(0)
        ax.scatter(g.age, base + sign * rng.uniform(0.05, 0.20, len(g)),
                   s=13, color=c, alpha=0.85, edgecolors='white',
                   linewidths=0.35, zorder=4)
        ax.text(d.age.max() + 3.2, base + sign * 0.30,
                'Women' if sex == 'Female' else 'Men', fontsize=FONT_TICK,
                color=c, fontweight='bold', va='center', ha='right')

        for band, side in [(g.age <= lo, 'Y'), (g.age >= hi, 'O')]:
            xpos = (d.age.min() + lo) / 2 if side == 'Y' else (hi + d.age.max()) / 2
            ax.text(xpos, base + sign * 0.42, f'$n$={int(band.sum())}',
                    ha='center', va='center', fontsize=FONT_TICK - 0.5,
                    color=c, fontweight='bold')

    for b in (lo, hi):
        ax.axvline(b, color='#BDBDBD', ls=':', lw=0.8, zorder=0)
    ax.text(lo - 1.2, -0.96, 'younger', ha='right', va='bottom',
            fontsize=FONT_TICK - 1, color='#9E9E9E')
    ax.text(hi + 1.2, -0.96, 'older', ha='left', va='bottom',
            fontsize=FONT_TICK - 1, color='#9E9E9E')

    ax.set_xlabel('Age (years)', fontsize=FONT_AXIS_TITLE, labelpad=1.5)
    ax.set_yticks([])
    ax.set_ylim(-1.0, 1.95)
    ax.set_xlim(d.age.min() - 4, d.age.max() + 4)
    ax.tick_params(labelsize=FONT_TICK, pad=1.2)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)


def build(out_name: str = 'fig_skull_layers') -> Path:
    d = load()
    hi_a = d['age'].quantile(2 / 3)
    exemplar = representative(d[d.age >= hi_a])
    print(f'N = {len(d)} (T1-only excluded); section from sub-{exemplar.subject_id}')

    FIG_W, FIG_H = WIDTH_2COL, WIDTH_2COL * 0.50
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    A_X, A_W = 0.012, 0.235
    TOP_Y, BOT_Y, H = 0.620, 0.135, 0.350
    ax_a = fig.add_axes([A_X, TOP_Y - 0.035, A_W, H + 0.07])
    ax_b = fig.add_axes([0.365, TOP_Y, 0.255, H])
    ax_c = fig.add_axes([0.735, TOP_Y, 0.250, H])
    ax_d = fig.add_axes([0.075, BOT_Y, 0.910, 0.320])

    half_h = 21.0
    half_w = half_h * (A_W * FIG_W) / ((H + 0.07) * FIG_H)
    panel_section(ax_a, exemplar, half_w, half_h)
    panel_composition(ax_b, d)
    panel_interaction(ax_c, d)
    panel_sample(ax_d, d)

    LABEL_KW = dict(fontsize=FONT_PANEL_LABEL, fontweight='bold', va='top')
    for x, y, k in [(0.004, 0.992, 'a'), (0.300, 0.992, 'b'),
                    (0.672, 0.992, 'c'), (0.004, 0.505, 'd')]:
        fig.text(x, y, k, **LABEL_KW)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f'{out_name}.png'
    fig.savefig(png, dpi=400, facecolor='white')
    fig.savefig(FIG_DIR / f'{out_name}.svg', dpi=400, facecolor='white')
    plt.close(fig)

    import statsmodels.api as sm
    x = d[['layer_skull', 'age', 'gender']].dropna().copy()
    x['female'] = (x.gender == 'Female').astype(int)
    x['age_c'] = x.age - x.age.mean(); x['ix'] = x.age_c * x.female
    mo = sm.OLS(x.layer_skull, sm.add_constant(x[['age_c', 'female', 'ix']])).fit()
    print(f"  age x sex on skull: b = {mo.params['ix']:+.4f}, p = {mo.pvalues['ix']:.4f}")
    for sex in ('Female', 'Male'):
        g = d[d.gender == sex]
        r, pv = stats.pearsonr(g.age, g.layer_skull)
        print(f'    {sex:7} n={len(g)}  skull x age r = {r:+.3f}, p = {pv:.4f}')
    print(f'wrote {png}')
    return png


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--name', default='fig_skull_layers')
    args = p.parse_args(argv)
    build(out_name=args.name)
    return 0


if __name__ == '__main__':
    sys.exit(main())
