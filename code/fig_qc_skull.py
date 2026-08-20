"""
fig_qc_skull.py — Does charm's skull segmentation look right?

The skull result in fig_skull_layers rests on charm's tissue labels, and skull
is the class charm segments least reliably -- it is the stated reason the seven
T1-only head models are excluded. Two things therefore need checking by eye
rather than by statistic: whether the labelled layers correspond to visible
anatomy at all, and whether they look worse in the subjects scanned without a
FLAIR.

Each panel shows the electrode-to-cortex ray with charm's labels tinted over
the participant's own T1, and prints the skull thickness the pipeline derived
from it. The point is to be able to check the number against the picture.

Top row: participants with T1 + FLAIR, spanning the age range.
Bottom row: every T1-only participant.

What the numbers already say, for context:

- T1-only skull measures do NOT differ from the rest once age and sex are
  controlled (skull total p = .58, diploe p = .45, inner table p = .90), so
  whatever is wrong with those head models, it does not show up as a systematic
  skull-thickness offset.
- Their CSF is higher (p = .025) and their |E| lower (p = .018), which is the
  difference the exclusion was actually protecting against.
- charm's skull agrees with the label-free intensity measure at only r = +.40
  overall, and charm reads 1.28 mm thicker on average. That is moderate
  agreement between two measures of the same quantity, and it is the reason
  this figure exists.

Usage
-----
    python fig_qc_skull.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import REPO_ROOT, EFIELD_CSV_PATH
from paper_style import (WIDTH_2COL, FONT_AXIS_TITLE, FONT_TICK,
                         FONT_PANEL_LABEL, REGRESSION_COLOR)
from fig_distance_anatomy import ELECTRODE_COLOR
from fig_skull_layers import ray_aligned_section, LAYERS, SCALP, CSF_L, COMPACT, SPONGY

FIG_DIR = REPO_ROOT / 'data' / 'figures' / 'paper'
N_COLS = 6
TISSUE = {SCALP: '#F2A488', COMPACT: '#FFE9A8', SPONGY: '#F58B3C', CSF_L: '#5B9BD5'}


def load():
    e = pd.read_csv(EFIELD_CSV_PATH, dtype={'subject_id': str})
    m = pd.read_csv(REPO_ROOT / 'data' / 'master_subject_data.csv',
                    dtype={'subject_id': str})
    d = e.merge(m[['subject_id', 'age', 'gender']], on='subject_id', how='left')
    d['age'] = pd.to_numeric(d['age'], errors='coerce')
    return d.dropna(subset=['age', 'layer_skull']).reset_index(drop=True)


def draw(ax, row, half_w, half_h):
    img, lab, path, f3_row, cortex_row, step, shape = ray_aligned_section(
        row.subject_id, np.array([row.f3_x, row.f3_y, row.f3_z], float),
        half_w, half_h)
    vmax = np.percentile(img[img > 0], 99.5) if (img > 0).any() else 1.0
    ax.imshow(img, cmap='gray', origin='lower', vmin=0, vmax=vmax,
              interpolation='bilinear')
    tint = np.zeros((*lab.shape, 4))
    for code, hexcol in TISSUE.items():
        tint[lab == code] = matplotlib.colors.to_rgba(hexcol, 0.48)
    ax.imshow(tint, origin='lower', interpolation='nearest')

    H, W = shape
    ax.plot([W / 2, W / 2], [f3_row, cortex_row], color='white', lw=0.7,
            ls=(0, (2, 1.4)), zorder=4)
    ax.plot(W / 2, f3_row, marker='v', color=ELECTRODE_COLOR, markersize=3.4,
            markeredgecolor='white', markeredgewidth=0.4, zorder=5)
    sex = 'F' if row.gender == 'Female' else 'M'
    ax.set_title(f'{row.subject_id}  {row.age:.0f}{sex}\nskull {row.layer_skull:.1f} mm',
                 fontsize=FONT_TICK - 1.5, pad=1.2, linespacing=1.3)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def build(out_name: str = 'fig_qc_skull') -> Path:
    d = load()
    t1 = d[d.t1_only.astype(bool)].sort_values('age')
    fl = d[~d.t1_only.astype(bool)].sort_values('age')
    # Span the age range rather than sampling at random, so the row covers the
    # conditions the skull claim is made across.
    idx = np.linspace(0, len(fl) - 1, N_COLS).round().astype(int)
    rows = [('T1 + FLAIR', fl.iloc[idx]), ('T1 only', t1)]
    print(f'FLAIR shown: {list(fl.iloc[idx].subject_id)}')
    print(f'T1-only    : {list(t1.subject_id)}')

    FIG_W = WIDTH_2COL
    cell = FIG_W / (N_COLS + 0.6)
    FIG_H = 2 * cell + 0.85
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    pw = 0.955 / N_COLS
    ph = (cell / FIG_H) * 0.97

    for r, (label, group) in enumerate(rows):
        y = 0.505 - r * (ph + 0.115)
        for c in range(N_COLS):
            if c >= len(group):
                break
            ax = fig.add_axes([0.032 + c * pw, y, pw * 0.90, ph])
            draw(ax, group.iloc[c], half_w=15.0,
                 half_h=15.0 * (FIG_H * ph) / (FIG_W * pw * 0.90))
        fig.text(0.004, y + ph + 0.075, label, fontsize=FONT_AXIS_TITLE,
                 fontweight='bold', color=REGRESSION_COLOR, va='top')

    handles = [plt.Line2D([], [], marker='s', ls='', markersize=4.5,
                          color=col, label=lab)
               for lab, col in [('Scalp', '#F2A488'), ('Compact bone', '#FFE9A8'),
                                ('Diploe', '#F58B3C'), ('CSF', '#5B9BD5')]]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               fontsize=FONT_TICK - 1, bbox_to_anchor=(0.5, 0.005),
               handletextpad=0.3, columnspacing=1.4)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f'{out_name}.png'
    fig.savefig(png, dpi=400, facecolor='white')
    fig.savefig(FIG_DIR / f'{out_name}.svg', dpi=400, facecolor='white')
    plt.close(fig)
    print(f'wrote {png}')
    return png


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--name', default='fig_qc_skull')
    args = p.parse_args(argv)
    build(out_name=args.name)
    return 0


if __name__ == '__main__':
    sys.exit(main())
