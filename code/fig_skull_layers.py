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


def ray_aligned_section(sub_id, f3, half_w_mm=3.5, pad_mm=2.5, step=0.1):
    """
    A section resampled so the electrode-to-cortex ray runs vertically.

    fig_distance_anatomy orients its sections to the superior axis, which is
    right for showing where the measurement sits in the head. Here the point
    is the layered structure the ray passes through, so the ray itself becomes
    the vertical axis and each tissue becomes a horizontal band that can be
    labelled. Scalp ends up at the top, cortex at the bottom.
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

    hw = int(half_w_mm / step)
    n_up, n_dn = int(pad_mm / step), int((path + pad_mm) / step)
    a = (np.arange(-hw, hw) + 0.5) * step
    b = (np.arange(-n_dn, n_up) + 0.5) * step
    A, B = np.meshgrid(a, b, indexing='xy')
    world = (f3[None, None, :] + A[..., None] * e1[None, None, :]
             + B[..., None] * e2[None, None, :])

    inv = np.linalg.inv(t1.affine)
    vox = (world @ inv[:3, :3].T) + inv[:3, 3]
    coords = np.stack([vox[..., 0], vox[..., 1], vox[..., 2]])
    img = map_coordinates(t1.get_fdata(), coords, order=1, mode='constant')
    lab = map_coordinates(lab_vol, coords, order=0, mode='constant')

    return img, lab, path, (n_dn + n_up, 2 * hw), step, n_dn


def panel_section(ax, row):
    img, lab, path, shape, step, n_dn = ray_aligned_section(
        row.subject_id, np.array([row.f3_x, row.f3_y, row.f3_z], float))

    vmax = np.percentile(img[img > 0], 99.5) if (img > 0).any() else 1.0
    ax.imshow(img, cmap='gray', origin='lower', vmin=0, vmax=vmax,
              interpolation='bilinear', aspect='auto')

    codes = {SCALP: '#F2A488', COMPACT: '#FFE9A8', SPONGY: '#F58B3C',
             CSF_L: '#5B9BD5'}
    tint = np.zeros((*lab.shape, 4))
    for code, hexcol in codes.items():
        tint[lab == code] = matplotlib.colors.to_rgba(hexcol, 0.55)
    ax.imshow(tint, origin='lower', interpolation='nearest', aspect='auto')

    H, W = shape
    # The electrode sits where b = 0, which is row n_dn -- NOT the top of the
    # image, which is pad_mm of air above it. Anchoring depths to H - 1 shifts
    # every label by that padding.
    top = n_dn

    # Labels are placed from this subject's own median layer thicknesses --
    # the same numbers panel b plots -- rather than from run-lengths down the
    # single centre column. That column can contain stray extra runs of
    # compact bone, which produced a second, spurious "Inner table" label.
    at = 0.0
    for col, label, _ in LAYERS:
        L = float(row[col])
        if L >= 0.6:
            mid_row = top - (at + L / 2) / step
            ax.annotate(label, xy=(W * 0.62, mid_row), xytext=(W * 1.20, mid_row),
                        fontsize=FONT_TICK - 0.5, va='center', ha='left',
                        color=REGRESSION_COLOR,
                        arrowprops=dict(arrowstyle='-', lw=0.6,
                                        color='#9E9E9E', shrinkA=0, shrinkB=1))
        at += L

    ax.plot([W / 2, W / 2], [top, top - path / step],
            color='white', lw=0.9, ls=(0, (2.5, 1.5)), zorder=4)
    ax.plot(W / 2, top, marker='v', color=ELECTRODE_COLOR, markersize=5,
            markeredgecolor='white', markeredgewidth=0.6, zorder=5, clip_on=False)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def panel_composition(ax, d):
    lo, hi = d['age'].quantile([1 / 3, 2 / 3])
    groups = [('Older', d[d.age >= hi]), ('Younger', d[d.age <= lo])]
    for y, (name, g) in enumerate(groups):
        left = 0.0
        for col, label, colour in LAYERS:
            w = g[col].mean()
            ax.barh(y, w, left=left, height=0.5, color=colour,
                    edgecolor='white', linewidth=0.7,
                    label=label if y == 0 else None)
            if w > 1.4:
                ax.text(left + w / 2, y, f'{w:.1f}', ha='center', va='center',
                        fontsize=FONT_TICK - 1.5, color='#3A3A3A')
            left += w
        ax.text(left + 0.3, y, f'{left:.1f}', va='center',
                fontsize=FONT_TICK, color=REGRESSION_COLOR, fontweight='bold')

    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f'{n}\n($n$ = {len(g)})' for n, g in groups],
                       fontsize=FONT_TICK)
    ax.set_xlabel('Thickness along the ray (mm)', fontsize=FONT_AXIS_TITLE,
                  labelpad=1.5)
    ax.tick_params(labelsize=FONT_TICK, pad=1.2)
    ax.set_xlim(0, 18.5); ax.set_ylim(-0.6, 1.9)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=FONT_TICK - 1, frameon=False, ncol=3, loc='upper center',
              bbox_to_anchor=(0.48, 1.30), handlelength=0.9,
              columnspacing=0.9, handletextpad=0.35)


def panel_correlations(ax, d):
    """Each layer's correlation with age and with the field, on one axis."""
    n = len(d)
    crit = stats.t.ppf(0.975, n - 2)
    r_crit = crit / np.sqrt(n - 2 + crit ** 2)

    items = [(lab, col) for col, lab, _ in LAYERS]
    items.append(('Skull (total)', 'layer_skull'))
    ys = np.arange(len(items))[::-1]

    ax.axvspan(-r_crit, r_crit, color='#F2F2F2', zorder=0)
    ax.axvline(0, color='#BDBDBD', lw=0.6, zorder=1)

    for y, (label, col) in zip(ys, items):
        ra = stats.pearsonr(d.age, d[col])[0]
        rf = stats.pearsonr(d[col], d[FIELD])[0]
        ax.plot([ra, rf], [y, y], color='#D0D0D0', lw=0.8, zorder=2)
        ax.scatter(ra, y, s=30, color=AGE_OLD, edgecolors='white',
                   linewidths=0.5, zorder=4, label='with age' if y == ys[0] else None)
        ax.scatter(rf, y, s=30, color=AGE_YOUNG, marker='D', edgecolors='white',
                   linewidths=0.5, zorder=4,
                   label='with field' if y == ys[0] else None)

    ax.set_yticks(ys)
    ax.set_yticklabels([lab for lab, _ in items], fontsize=FONT_TICK)
    ax.get_yticklabels()[-1].set_fontweight('bold')
    ax.set_xlabel('Correlation ($r$)', fontsize=FONT_AXIS_TITLE, labelpad=1.5)
    ax.set_xlim(-0.95, 0.72)
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.tick_params(labelsize=FONT_TICK, pad=1.2)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=FONT_TICK - 1, frameon=False, ncol=2, loc='upper center',
              bbox_to_anchor=(0.5, 1.20), handletextpad=0.2, columnspacing=0.9)
    ax.text(0.0, -0.62, 'shaded: $p$ > .05', ha='center', va='center',
            fontsize=FONT_TICK - 1.5, color='#9E9E9E')


def build(out_name: str = 'fig_skull_layers') -> Path:
    d = load()
    hi_a = d['age'].quantile(2 / 3)
    exemplar = representative(d[d.age >= hi_a])
    print(f'N = {len(d)} (T1-only excluded); section from sub-{exemplar.subject_id}')

    FIG_W, FIG_H = WIDTH_2COL, WIDTH_2COL * 0.36
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_a = fig.add_axes([0.012, 0.075, 0.105, 0.845])
    ax_b = fig.add_axes([0.315, 0.215, 0.290, 0.560])
    ax_c = fig.add_axes([0.760, 0.215, 0.225, 0.560])

    panel_section(ax_a, exemplar)
    panel_composition(ax_b, d)
    panel_correlations(ax_c, d)

    LABEL_KW = dict(fontsize=FONT_PANEL_LABEL, fontweight='bold', va='top')
    for x, k in [(0.004, 'a'), (0.235, 'b'), (0.655, 'c')]:
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
