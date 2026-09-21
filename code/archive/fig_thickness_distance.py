"""
fig_thickness_distance.py — Figure: the dose deficit is geometry, not thinning

Four panels, read as two rows:

  a, b   age is related to both structural measures -- cortex thins, and it
         also sits further from the scalp
  c, d   only one of them touches the delivered field

That contrast is the whole point. "Older brains get less current because they
are atrophied" is the intuitive story and it is wrong in its middle link:
cortical thickness and scalp-to-cortex distance are essentially unrelated to
each other here, thickness adds nothing to |E| once distance is in the model,
and thickness does not explain why distance grows with age. Age drives the two
independently, and only distance reaches the field.

Both morphometry pipelines agree on this (FreeSurfer 7.3.2 and charm), which
matters because the obvious objection is that the thickness measure is simply
poor. It is not -- the same measure recovers age x thickness at r = -.64.

The seven T1-only head models are excluded, as everywhere else: their fields
are systematically ~28% lower because skull segmentation degrades without a
FLAIR, in the same direction as the hypothesis.

Usage
-----
    python fig_thickness_distance.py
    python fig_thickness_distance.py --thickness charm
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

from config import (REPO_ROOT, EFIELD_CSV_PATH, FREESURFER_MORPH_PATH)
from paper_style import (WIDTH_2COL, FONT_AXIS_TITLE, FONT_TICK,
                         FONT_PANEL_LABEL, AGE_YOUNG, AGE_OLD,
                         REGRESSION_COLOR)

FIG_DIR = REPO_ROOT / 'data' / 'figures' / 'paper'

# Primary measures. Distance is measured to the pial surface within the DLPFC
# parcel, robust percentile -- see extract_scalp_cortex_distance.py for why
# the DLPFC-restricted variants are the meaningful ones.
DIST = 'dist_pial_dlpfc_p1'
FIELD = 'mean_magnE'
THICKNESS = {'freesurfer': 'lh_dlpfc_thickness',
             'charm': 'charm_dlpfc_thickness'}


def load(thickness: str = 'freesurfer') -> Tuple[pd.DataFrame, str]:
    e = pd.read_csv(EFIELD_CSV_PATH, dtype={'subject_id': str})
    fs = pd.read_csv(FREESURFER_MORPH_PATH, dtype={'subject_id': str})
    m = pd.read_csv(REPO_ROOT / 'data' / 'master_subject_data.csv',
                    dtype={'subject_id': str})
    d = (e.merge(fs, on='subject_id', how='left')
          .merge(m[['subject_id', 'age']], on='subject_id', how='left'))
    d['age'] = pd.to_numeric(d['age'], errors='coerce')
    if 't1_only' in d.columns:
        d = d[~d['t1_only'].astype(bool)]
    col = THICKNESS[thickness]
    return d.dropna(subset=['age', col, DIST, FIELD]).reset_index(drop=True), col


def scatter(ax, x, y, age, xlabel, ylabel, annot_loc='upper right'):
    """Age-coloured scatter with OLS fit and 95% CI. Returns (r, p)."""
    norm = plt.Normalize(age.min(), age.max())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'blue_gold', [AGE_YOUNG, AGE_OLD])
    ax.scatter(x, y, c=cmap(norm(age)), s=18, edgecolors='white',
               linewidths=0.4, zorder=3)

    slope, intercept, r, p, _ = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 200)
    ys = intercept + slope * xs
    se = np.sqrt(np.sum((y - (intercept + slope * x)) ** 2) / (len(x) - 2))
    ci = stats.t.ppf(0.975, len(x) - 2) * se * np.sqrt(
        1 / len(x) + (xs - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    ax.fill_between(xs, ys - ci, ys + ci, color='gray', alpha=0.18, linewidth=0)
    # A non-significant relationship gets a dashed line, so the eye does not
    # read a trend into panels that do not have one.
    ax.plot(xs, ys, color=REGRESSION_COLOR, lw=1.2, zorder=2,
            ls='-' if p < .05 else (0, (3, 2)))

    ax.set_xlabel(xlabel, fontsize=FONT_AXIS_TITLE, labelpad=1.5)
    ax.set_ylabel(ylabel, fontsize=FONT_AXIS_TITLE, labelpad=1.5)
    ax.tick_params(labelsize=FONT_TICK, pad=1.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    ha, xa = ('right', 0.98) if 'right' in annot_loc else ('left', 0.02)
    va, ya = ('bottom', 0.03) if 'lower' in annot_loc else ('top', 0.97)
    txt = f'$r$ = {r:.3f}, $p$ = {p:.3f}' if p >= .001 else \
          f'$r$ = {r:.3f}, $p$ < .001'
    ax.text(xa, ya, txt, transform=ax.transAxes, ha=ha, va=va,
            fontsize=FONT_TICK,
            color=REGRESSION_COLOR if p < .05 else '#8A8A8A',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='none', alpha=0.85))
    return r, p


def build(out_name: str = 'fig_thickness_distance',
          thickness: str = 'freesurfer') -> Path:
    d, tcol = load(thickness)
    src = 'FreeSurfer' if thickness == 'freesurfer' else 'charm'
    print(f'N = {len(d)} (T1-only excluded), thickness from {src}')

    FIG_W, FIG_H = WIDTH_2COL, WIDTH_2COL * 0.58
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # Explicit rectangles rather than subplots: the default spacing leaves a
    # wide gutter between rows that nothing occupies.
    W, H = 0.400, 0.345
    L, R = 0.068, 0.578
    TOP, BOT = 0.600, 0.100
    axes = {k: fig.add_axes(rect) for k, rect in {
        'a': [L, TOP, W, H], 'b': [R, TOP, W, H],
        'c': [L, BOT, W, H], 'd': [R, BOT, W, H]}.items()}

    age = d['age'].values
    t_lab = 'DLPFC cortical thickness (mm)'
    d_lab = 'Scalp-to-cortex distance (mm)'
    f_lab = 'Mean |E| in DLPFC (V/m)'

    r_at, _ = scatter(axes['a'], age, d[tcol].values, age,
                      'Age (years)', t_lab)
    r_ad, _ = scatter(axes['b'], age, d[DIST].values, age,
                      'Age (years)', d_lab, annot_loc='lower right')
    r_tf, p_tf = scatter(axes['c'], d[tcol].values, d[FIELD].values, age,
                         t_lab, f_lab, annot_loc='upper left')
    r_df, _ = scatter(axes['d'], d[DIST].values, d[FIELD].values, age,
                      d_lab, f_lab)

    # Row captions carry the argument, so the reader does not have to
    # reconstruct it from four correlation coefficients.
    mid = (L + R + W) / 2
    fig.text(mid, 0.998, 'Age is related to both structural measures',
             ha='center', va='top', fontsize=FONT_AXIS_TITLE)
    fig.text(mid, 0.497, 'Only distance is related to the delivered field',
             ha='center', va='top', fontsize=FONT_AXIS_TITLE)

    LABEL_KW = dict(fontsize=FONT_PANEL_LABEL, fontweight='bold', va='top')
    fig.text(0.004, 0.978, 'a', **LABEL_KW)
    fig.text(0.514, 0.978, 'b', **LABEL_KW)
    fig.text(0.004, 0.477, 'c', **LABEL_KW)
    fig.text(0.514, 0.477, 'd', **LABEL_KW)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f'{out_name}.png'
    fig.savefig(png, dpi=400, facecolor='white')
    fig.savefig(FIG_DIR / f'{out_name}.svg', dpi=400, facecolor='white')
    plt.close(fig)

    r_td, p_td = stats.pearsonr(d[tcol], d[DIST])
    print(f'  age x thickness   r = {r_at:+.3f}')
    print(f'  age x distance    r = {r_ad:+.3f}')
    print(f'  thickness x |E|   r = {r_tf:+.3f} (p = {p_tf:.3f})')
    print(f'  distance x |E|    r = {r_df:+.3f}')
    print(f'  thickness x dist  r = {r_td:+.3f} (p = {p_td:.3f})  <- the dissociation')
    print(f'wrote {png}')
    return png


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--thickness', default='freesurfer',
                   choices=['freesurfer', 'charm'])
    p.add_argument('--name', default=None)
    args = p.parse_args(argv)
    build(out_name=args.name or ('fig_thickness_distance' if
                                 args.thickness == 'freesurfer' else
                                 'fig_thickness_distance_charm'),
          thickness=args.thickness)
    return 0


if __name__ == '__main__':
    sys.exit(main())
