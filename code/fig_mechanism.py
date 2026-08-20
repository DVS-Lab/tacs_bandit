"""
fig_mechanism.py — The dose deficit is geometry, not cortical thinning

Six panels on a 3 x 2 grid, read row by row:

  a, b, c   what the measurement is, and that it grows with age. Oblique
            sections through the electrode in a younger and an older
            participant, then age against scalp-to-cortex distance.
  d, e, f   cortex also thins with age, but thinning does not reach the
            field -- distance does.

The sections are here because "scalp-to-cortex distance" is an abstraction on
a scatter plot. Panel f's x-axis means nothing to a reader until they have
seen the gap it measures.

**The two participants are representative by construction**: each is the one
whose distance sits closest to its age tertile's mean, not the most extreme
and not chosen by eye. Their values are printed on the sections so this can be
checked against the group means.

Layout notes, since this grid mixes images with scatters:

- Every panel occupies an identical rectangle, so panel labels and plot areas
  align across all six by construction rather than by adjustment.
- The sections are resampled at the panel's own aspect ratio. imshow keeps
  pixels square, so a square section in a non-square panel would be
  letterboxed and reintroduce the whitespace this layout exists to avoid.

Usage
-----
    python fig_mechanism.py
    python fig_mechanism.py --thickness charm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import REPO_ROOT
from paper_style import (WIDTH_2COL, FONT_AXIS_TITLE, FONT_TICK,
                         FONT_PANEL_LABEL)
from fig_distance_anatomy import (DIST, ELECTRODE_COLOR, draw_panel,
                                  load_subjects, representative)
from fig_thickness_distance import FIELD, THICKNESS, load as load_scatter_data

FIG_DIR = REPO_ROOT / 'data' / 'figures' / 'paper'


def scatter(ax, x, y, age, xlabel, ylabel, annot_loc='upper right'):
    """Age-coloured scatter with fit and 95% CI. Returns (r, p)."""
    from paper_style import AGE_YOUNG, AGE_OLD, REGRESSION_COLOR
    norm = plt.Normalize(age.min(), age.max())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'blue_gold', [AGE_YOUNG, AGE_OLD])
    ax.scatter(x, y, c=cmap(norm(age)), s=13, edgecolors='white',
               linewidths=0.35, zorder=3)

    slope, intercept, r, p, _ = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 200)
    ys = intercept + slope * xs
    se = np.sqrt(np.sum((y - (intercept + slope * x)) ** 2) / (len(x) - 2))
    ci = stats.t.ppf(0.975, len(x) - 2) * se * np.sqrt(
        1 / len(x) + (xs - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    ax.fill_between(xs, ys - ci, ys + ci, color='gray', alpha=0.18, linewidth=0)
    # Dashed where the relationship is not significant, so the eye does not
    # read a trend into panel e.
    ax.plot(xs, ys, color=REGRESSION_COLOR, lw=1.1, zorder=2,
            ls='-' if p < .05 else (0, (3, 2)))

    ax.set_xlabel(xlabel, fontsize=FONT_AXIS_TITLE, labelpad=1.5)
    ax.set_ylabel(ylabel, fontsize=FONT_AXIS_TITLE, labelpad=1.5)
    ax.tick_params(labelsize=FONT_TICK - 0.5, pad=1.2)
    ax.locator_params(nbins=5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    ha, xa = ('right', 0.97) if 'right' in annot_loc else ('left', 0.03)
    va, ya = ('bottom', 0.04) if 'lower' in annot_loc else ('top', 0.96)
    txt = (f'$r$ = {r:.2f}, $p$ = {p:.3f}' if p >= .001
           else f'$r$ = {r:.2f}, $p$ < .001')
    ax.text(xa, ya, txt, transform=ax.transAxes, ha=ha, va=va,
            fontsize=FONT_TICK - 0.5,
            color=REGRESSION_COLOR if p < .05 else '#8A8A8A',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.85))
    return r, p


def build(out_name: str = 'fig_mechanism',
          thickness: str = 'freesurfer') -> Path:
    d, tcol = load_scatter_data(thickness)
    anat, lo, hi = load_subjects()
    y = representative(anat[anat.age <= lo])
    o = representative(anat[anat.age >= hi])
    src = 'FreeSurfer' if thickness == 'freesurfer' else 'charm'
    print(f'N = {len(d)} (T1-only excluded), thickness from {src}')
    print(f'  sections: sub-{y.subject_id} ({y.age:.0f} y, {y[DIST]:.1f} mm) and '
          f'sub-{o.subject_id} ({o.age:.0f} y, {o[DIST]:.1f} mm)')

    FIG_W, FIG_H = WIDTH_2COL, WIDTH_2COL * 0.64
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # One rectangle shape for all six panels. Everything else -- labels, plot
    # areas, the section crops -- is derived from it, so alignment holds
    # without per-panel tweaking.
    W, H = 0.272, 0.375
    XS = [0.058, 0.378, 0.698]
    Y_TOP, Y_BOT = 0.575, 0.095
    ax = {}
    for i, x in enumerate(XS):
        ax[i] = fig.add_axes([x, Y_TOP, W, H])
        ax[i + 3] = fig.add_axes([x, Y_BOT, W, H])

    # Sections cropped to the panel's own aspect so nothing is letterboxed.
    half_h = 30.0
    half_w = half_h * (W * FIG_W) / (H * FIG_H)
    draw_panel(ax[0], y.subject_id, y, ELECTRODE_COLOR,
               f'Younger (age {y.age:.0f})', half_w, half_h, title_pad=1.5)
    draw_panel(ax[1], o.subject_id, o, ELECTRODE_COLOR,
               f'Older (age {o.age:.0f})', half_w, half_h, title_pad=1.5)

    age = d['age'].values
    t_lab, d_lab = 'DLPFC thickness (mm)', 'Scalp-to-cortex distance (mm)'
    f_lab = 'Mean |E| in DLPFC (V/m)'

    r = {}
    r['c'] = scatter(ax[2], age, d[DIST].values, age, 'Age (years)', d_lab,
                     annot_loc='lower right')
    r['d'] = scatter(ax[3], age, d[tcol].values, age, 'Age (years)', t_lab)
    r['e'] = scatter(ax[4], d[tcol].values, d[FIELD].values, age, t_lab, f_lab,
                     annot_loc='upper left')
    r['f'] = scatter(ax[5], d[DIST].values, d[FIELD].values, age, d_lab, f_lab)

    # Panel labels: one y per row and one x per column, both in figure
    # coordinates. Panels a and b carry titles and the scatters carry y-axis
    # labels, so anchoring to anything panel-relative would put the six labels
    # at six slightly different places.
    LABEL_KW = dict(fontsize=FONT_PANEL_LABEL, fontweight='bold', va='top')
    label_y = {0: 0.997, 1: Y_BOT + H + 0.038}
    for i, key in enumerate('abcdef'):
        fig.text(XS[i % 3] - 0.050, label_y[i // 3], key, **LABEL_KW)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f'{out_name}.png'
    fig.savefig(png, dpi=400, facecolor='white')
    fig.savefig(FIG_DIR / f'{out_name}.svg', dpi=400, facecolor='white')
    plt.close(fig)

    for k in 'cdef':
        print(f'  {k}: r = {r[k][0]:+.3f}, p = {r[k][1]:.4f}')
    print(f'wrote {png}')
    return png


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--thickness', default='freesurfer',
                   choices=['freesurfer', 'charm'])
    p.add_argument('--name', default='fig_mechanism')
    args = p.parse_args(argv)
    build(out_name=args.name, thickness=args.thickness)
    return 0


if __name__ == '__main__':
    sys.exit(main())
