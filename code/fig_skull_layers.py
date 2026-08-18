"""
fig_skull_layers.py — What the distance is made of, and which part matters

The skull vault is a sandwich rather than a solid plate, so a ray from the
electrode to cortex crosses five layers in a fixed order:

    scalp -> outer table -> diploe -> inner table -> CSF -> cortex

Outer and inner table are the same material, compact bone; they are the outer
and inner faces of the sandwich. The diploe between them is spongy,
marrow-filled bone.

Three panels:

  a   the layers themselves, on one participant's own scan, resampled so the
      electrode-to-cortex ray runs vertically -- which turns the layers into
      labelled horizontal bands and lets the panel be read as an anatomical
      key for the rest of the figure
  b   how much of each layer, younger versus older tertile
  c   each layer's two correlations: with age, and with the delivered field

Panel c is the argument. A layer can only carry the age effect on dose if it
is related to *both*, and only skull is. CSF has the strongest age
relationship of any layer (r = +.50) but the weakest link to the field
(r = -.25) and mediates nothing detectable; skull has a weaker age
relationship (+.34), the strongest field relationship (-.78), and mediates
89%. Cortical thinning, tested in fig_mechanism, is unrelated to either.

Skull thickness is already known to drive inter-individual variability in
tDCS/tACS field strength. The narrower contribution here is that the *age*
effect on delivered dose runs through skull, not through atrophy or CSF.

CAVEAT. This rests on charm's skull segmentation, the output known to degrade
without a FLAIR -- the reason the seven T1-only head models are excluded
throughout. At 1 mm voxels the inner table is one to two voxels thick, so the
outer/diploe/inner split is descriptive; total skull is the number to trust.

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
SCALP, CSF_L, COMPACT, SPONGY = 5, 3, 7, 8


def load():
    e = pd.read_csv(EFIELD_CSV_PATH, dtype={'subject_id': str})
    m = pd.read_csv(REPO_ROOT / 'data' / 'master_subject_data.csv',
                    dtype={'subject_id': str})
    d = e.merge(m[['subject_id', 'age']], on='subject_id', how='left')
    d['age'] = pd.to_numeric(d['age'], errors='coerce')
    d = d.dropna(subset=['age'] + [c for c, _, _ in LAYERS])
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
    """
    Layer thickness as vertical stacks, depth increasing downward.

    Oriented to match panel a -- scalp at the top, CSF at the bottom -- so the
    labels there apply here too and the panel needs no legend of its own.
    """
    lo, hi = d['age'].quantile([1 / 3, 2 / 3])
    groups = [('Younger', d[d.age <= lo]), ('Older', d[d.age >= hi])]

    for x, (name, g) in enumerate(groups):
        at = 0.0
        for col, label, colour in LAYERS:
            w = g[col].mean()
            ax.bar(x, w, bottom=at, width=0.66, color=colour,
                   edgecolor='white', linewidth=0.7)
            if w > 1.4:
                ax.text(x, at + w / 2, f'{w:.1f}', ha='center', va='center',
                        fontsize=FONT_TICK - 1.5, color='#3A3A3A')
            at += w
        ax.text(x, -0.5, f'{at:.1f}', ha='center', va='bottom',
                fontsize=FONT_TICK, color=REGRESSION_COLOR, fontweight='bold')

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f'{n}\n($n$ = {len(g)})' for n, g in groups],
                       fontsize=FONT_TICK)
    ax.set_ylabel('Depth from scalp surface (mm)', fontsize=FONT_AXIS_TITLE,
                  labelpad=1.5)
    ax.set_ylim(16.9, -2.1)     # inverted: depth grows downward, as in panel a
    ax.set_xlim(-0.7, len(groups) - 0.3)
    ax.tick_params(labelsize=FONT_TICK, pad=1.2)
    ax.set_yticks([0, 5, 10, 15])
    for sp in ('top', 'right', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='x', length=0)


def panel_correlations(ax, d):
    """
    Each layer at (r with age, r with field). Mediators sit off both axes.

    A layer can only carry the age effect on dose if it is related to both, so
    the plot is the mediation logic rather than two rankings. Shaded bands mark
    p > .05 on each axis.
    """
    n = len(d)
    crit = stats.t.ppf(0.975, n - 2)
    r_crit = crit / np.sqrt(n - 2 + crit ** 2)

    ax.axhspan(-r_crit, r_crit, color='#F0F0F0', zorder=0)
    ax.axvspan(-r_crit, r_crit, color='#F0F0F0', zorder=0)
    ax.axhline(0, color='#BDBDBD', lw=0.6, zorder=1)
    ax.axvline(0, color='#BDBDBD', lw=0.6, zorder=1)

    offsets = {'Scalp': (7, -9), 'Outer table': (7, 5), 'Diploe': (8, -3),
               'Inner table': (-17, -12), 'CSF': (6, 4),
               'Skull (total)': (9, -2)}
    items = [(lab, col, c) for col, lab, c in LAYERS]
    items.append(('Skull (total)', 'layer_skull', SKULL_COLOR))

    for label, col, colour in items:
        ra = stats.pearsonr(d.age, d[col])[0]
        rf = stats.pearsonr(d[col], d[FIELD])[0]
        big = label == 'Skull (total)'
        ax.scatter(ra, rf, s=54 if big else 34, color=colour, zorder=4,
                   edgecolors=REGRESSION_COLOR if big else 'white',
                   linewidths=1.0 if big else 0.5)
        ax.annotate(label, (ra, rf), textcoords='offset points',
                    xytext=offsets.get(label, (5, 5)), fontsize=FONT_TICK - 1,
                    fontweight='bold' if big else 'normal',
                    color=REGRESSION_COLOR)

    ax.set_xlabel('$r$ with age', fontsize=FONT_AXIS_TITLE, labelpad=1.5)
    ax.set_ylabel('$r$ with delivered field', fontsize=FONT_AXIS_TITLE,
                  labelpad=1.5)
    ax.tick_params(labelsize=FONT_TICK, pad=1.2)
    ax.set_xlim(-0.35, 0.75)
    ax.set_ylim(-0.93, 0.20)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.text(0.98, 0.03, 'shaded: $p$ > .05', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=FONT_TICK - 1.5, color='#9E9E9E')


def build(out_name: str = 'fig_skull_layers') -> Path:
    d = load()
    hi_a = d['age'].quantile(2 / 3)
    exemplar = representative(d[d.age >= hi_a])
    print(f'N = {len(d)} (T1-only excluded); section from sub-{exemplar.subject_id}')

    FIG_W, FIG_H = WIDTH_2COL, WIDTH_2COL * 0.42
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    A_X, A_W, Y, H = 0.012, 0.290, 0.185, 0.700
    ax_a = fig.add_axes([A_X, Y, A_W, H])
    ax_b = fig.add_axes([0.430, Y, 0.165, H])
    ax_c = fig.add_axes([0.715, Y, 0.270, H])

    # Section cropped to the panel's own aspect so imshow does not letterbox it.
    half_h = 21.0
    half_w = half_h * (A_W * FIG_W) / (H * FIG_H)
    panel_section(ax_a, exemplar, half_w, half_h)
    panel_composition(ax_b, d)
    panel_correlations(ax_c, d)

    LABEL_KW = dict(fontsize=FONT_PANEL_LABEL, fontweight='bold', va='top')
    for x, k in [(0.004, 'a'), (0.372, 'b'), (0.648, 'c')]:
        fig.text(x, 0.985, k, **LABEL_KW)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f'{out_name}.png'
    fig.savefig(png, dpi=400, facecolor='white')
    fig.savefig(FIG_DIR / f'{out_name}.svg', dpi=400, facecolor='white')
    plt.close(fig)
    for col, label, _ in LAYERS + [('layer_skull', 'Skull (total)', '')]:
        print(f'  {label:14s} age r = {stats.pearsonr(d.age, d[col])[0]:+.3f}   '
              f'|E| r = {stats.pearsonr(d[col], d[FIELD])[0]:+.3f}')
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
