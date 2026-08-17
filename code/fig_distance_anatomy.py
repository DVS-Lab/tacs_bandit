"""
fig_distance_anatomy.py — What "scalp-to-cortex distance" actually looks like

Coronal sections through the F3 electrode in a younger and an older
participant, with CSF tinted so the gap between skull and cortex is visible.
The measure driving the whole dose result is abstract on a scatter plot; here
it is just the space between the electrode and the brain.

**The two subjects are representative by construction.** Each is the one whose
scalp-to-cortex distance sits closest to its age tertile's *mean* -- not the
most extreme, and not chosen by eye. Picking exemplars freely would let the
figure show any difference the author wanted, which is exactly the objection
that made the E-field figure use group averages instead of example brains.
Their measured distances are printed on the panels so the reader can check
them against the group means in the caption.

Sections are oblique, not coronal: each is taken through the plane containing
both the electrode and the nearest cortical point. A straight coronal slice
puts that point up to 12 mm out of plane, so the annotated line would stop in
empty space short of the brain. Here both endpoints lie in the image and the
line is the measured distance at true length.

Usage
-----
    python fig_distance_anatomy.py
    python fig_distance_anatomy.py --subjects 10606 11622
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from config import REPO_ROOT, EFIELD_CSV_PATH
from paper_style import (WIDTH_2COL, FONT_AXIS_TITLE, FONT_TICK,
                         FONT_PANEL_LABEL)

# One colour for the electrode in both panels. Colouring it by age group would
# imply the marker encodes something about the participant; it does not -- it
# is the same electrode in the same place, and the group is already in the
# panel title. Amber reads clearly against both the grayscale and the CSF tint.
ELECTRODE_COLOR = '#FFC107'

M2M_DIR = Path.home() / 'Desktop' / 'projects' / 'tacs_bandit' / 'simNIBS' / 'm2m'
FIG_DIR = REPO_ROOT / 'data' / 'figures' / 'paper'

DIST = 'dist_pial_dlpfc_p1'
CSF_LABEL = 3          # final_tissues_LUT.txt
HALF_WIDTH_MM = 34     # crop tight: centred on the gap, not on the electrode


def load_subjects() -> Tuple[pd.DataFrame, float, float]:
    e = pd.read_csv(EFIELD_CSV_PATH, dtype={'subject_id': str})
    m = pd.read_csv(REPO_ROOT / 'data' / 'master_subject_data.csv',
                    dtype={'subject_id': str})
    d = e.merge(m[['subject_id', 'age']], on='subject_id', how='left')
    d['age'] = pd.to_numeric(d['age'], errors='coerce')
    d = d.dropna(subset=['age', DIST])
    if 't1_only' in d.columns:
        d = d[~d['t1_only'].astype(bool)]
    lo, hi = d['age'].quantile([1 / 3, 2 / 3])
    return d.reset_index(drop=True), float(lo), float(hi)


def representative(group: pd.DataFrame) -> pd.Series:
    """The subject closest to the group's mean distance."""
    target = group[DIST].mean()
    return group.iloc[(group[DIST] - target).abs().argsort().iloc[0]]


def oblique_section(sub_id: str, f3_world: np.ndarray,
                    half_w_mm: float = HALF_WIDTH_MM,
                    half_h_mm: float = HALF_WIDTH_MM):
    """
    T1 and CSF resampled onto the plane that actually contains the measurement.

    A plain coronal slice through F3 does not work: the nearest cortical point
    lies up to 12 mm anterior or posterior to the electrode, so its projection
    into that plane lands in empty space and the annotated line stops short of
    the brain. Here the section is taken through the plane spanned by the
    F3-to-cortex vector and the superior axis, so both endpoints lie in the
    image and the drawn line is the measured distance at true length.
    """
    m2m = M2M_DIR / f'm2m_sub-{sub_id}'
    t1 = nib.as_closest_canonical(nib.load(str(m2m / 'T1.nii.gz')))
    tis = nib.as_closest_canonical(nib.load(str(m2m / 'final_tissues.nii.gz')))

    pial = nib.load(str(m2m / 'surfaces' / 'lh.pial.gii')).agg_data()[0]
    near = pial[np.argmin(np.linalg.norm(pial - f3_world, axis=1))]

    u = near - f3_world
    u = u / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    n = np.cross(u, z)
    if np.linalg.norm(n) < 1e-6:          # u parallel to z: pick another axis
        n = np.cross(u, np.array([0.0, 1.0, 0.0]))
    n = n / np.linalg.norm(n)
    e2 = z - np.dot(z, n) * n             # superior, projected into the plane
    e2 = e2 / np.linalg.norm(e2)
    e1 = np.cross(e2, n)
    e1 = e1 / np.linalg.norm(e1)

    # The crop is sampled at the caller's aspect ratio rather than square:
    # imshow keeps pixels square, so a section that does not match its panel
    # gets letterboxed, which is exactly the dead space this figure avoids.
    step = 0.4                            # mm per pixel; finer than the 1 mm voxels
    hw, hh = int(half_w_mm / step), int(half_h_mm / step)
    centre = (f3_world + near) / 2.0
    a = (np.arange(-hw, hw) + 0.5) * step
    b = (np.arange(-hh, hh) + 0.5) * step
    A, B = np.meshgrid(a, b, indexing='xy')
    world = (centre[None, None, :] + A[..., None] * e1[None, None, :]
             + B[..., None] * e2[None, None, :])

    inv = np.linalg.inv(t1.affine)
    vox = (world @ inv[:3, :3].T) + inv[:3, 3]
    coords = np.stack([vox[..., 0], vox[..., 1], vox[..., 2]])

    img = map_coordinates(t1.get_fdata(), coords, order=1, mode='constant')
    lab = map_coordinates(np.squeeze(tis.get_fdata()).astype(np.int16), coords,
                          order=0, mode='constant')

    def to_px(p):
        d = p - centre
        return (np.dot(d, e1) / step + hw, np.dot(d, e2) / step + hh)

    return img, (lab == CSF_LABEL), to_px(f3_world), to_px(near), step


def draw_panel(ax, sub_id, row, colour, title, half_w_mm=HALF_WIDTH_MM,
               half_h_mm=HALF_WIDTH_MM, title_pad=2):
    img, csf, (ei, ek), (ni, nk), _ = oblique_section(
        sub_id, np.array([row.f3_x, row.f3_y, row.f3_z], dtype=float),
        half_w_mm, half_h_mm)

    vmax = np.percentile(img[img > 0], 99.5) if (img > 0).any() else 1.0
    ax.imshow(img, cmap='gray', origin='lower', vmin=0, vmax=vmax,
              interpolation='bilinear')

    # CSF tinted rather than outlined: the point is the *volume* of the gap.
    tint = np.zeros((*csf.shape, 4))
    tint[csf] = matplotlib.colors.to_rgba('#4FC3F7', 0.55)
    ax.imshow(tint, origin='lower', interpolation='nearest')

    # Both endpoints lie in this plane by construction, so the line is the
    # measured distance drawn at true length.
    ax.plot([ei, ni], [ek, nk], color='white', lw=0.9, ls=(0, (2.5, 1.5)),
            zorder=4)
    ax.plot(ni, nk, marker='o', color='white', markersize=2.2, zorder=5)
    ax.plot(ei, ek, marker='v', color=ELECTRODE_COLOR, markersize=5,
            markeredgecolor='white', markeredgewidth=0.6, zorder=5)
    ax.annotate('F3', (ei, ek), textcoords='offset points', xytext=(0, 5),
                ha='center', fontsize=FONT_TICK, color='white', zorder=6)
    ax.text((ei + ni) / 2, (ek + nk) / 2, f'  {row[DIST]:.1f} mm',
            ha='left', va='center', fontsize=FONT_TICK, color='white',
            zorder=6)
    if title:
        ax.set_title(title, fontsize=FONT_AXIS_TITLE, pad=title_pad)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def build(out_name: str = 'fig_distance_anatomy',
          subjects: Optional[Tuple[str, str]] = None) -> Path:
    d, lo, hi = load_subjects()
    young_g, old_g = d[d.age <= lo], d[d.age >= hi]
    if subjects:
        y = d[d.subject_id == subjects[0]].iloc[0]
        o = d[d.subject_id == subjects[1]].iloc[0]
    else:
        y, o = representative(young_g), representative(old_g)

    print(f'younger tertile n={len(young_g)}, mean distance '
          f'{young_g[DIST].mean():.1f} mm -> sub-{y.subject_id} '
          f'(age {y.age:.0f}, {y[DIST]:.1f} mm)')
    print(f'older   tertile n={len(old_g)}, mean distance '
          f'{old_g[DIST].mean():.1f} mm -> sub-{o.subject_id} '
          f'(age {o.age:.0f}, {o[DIST]:.1f} mm)')

    FIG_W = WIDTH_2COL * 0.62
    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.56), facecolor='white')
    W, H, Y = 0.482, 0.885, 0.015
    ax_a = fig.add_axes([0.010, Y, W, H])
    ax_b = fig.add_axes([0.505, Y, W, H])

    draw_panel(ax_a, y.subject_id, y, ELECTRODE_COLOR,
               f'Younger (age {y.age:.0f})')
    draw_panel(ax_b, o.subject_id, o, ELECTRODE_COLOR,
               f'Older (age {o.age:.0f})')

    LABEL_KW = dict(fontsize=FONT_PANEL_LABEL, fontweight='bold', va='top')
    fig.text(0.004, 0.995, 'a', **LABEL_KW)
    fig.text(0.497, 0.995, 'b', **LABEL_KW)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f'{out_name}.png'
    fig.savefig(png, dpi=400, facecolor='white')
    fig.savefig(FIG_DIR / f'{out_name}.svg', dpi=400, facecolor='white')
    plt.close(fig)
    print(f'wrote {png}')
    return png


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--subjects', nargs=2, default=None,
                   metavar=('YOUNG', 'OLD'))
    p.add_argument('--name', default='fig_distance_anatomy')
    args = p.parse_args(argv)
    build(out_name=args.name, subjects=tuple(args.subjects) if args.subjects else None)
    return 0


if __name__ == '__main__':
    sys.exit(main())
