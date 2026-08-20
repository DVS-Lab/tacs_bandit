"""
plot_stim_verification.py — Look at the 6 Hz stimulation artifact run by run

Renders what stim_verification.py decides numerically, so the counterbalance
assignment can be eyeballed rather than taken on faith.

Two views:

  heatmap  — every subject as a row, runs 1-8 as columns, coloured by sustained
             6 Hz power. Counterbalance A subjects light up on runs 2-3 and
             B subjects on runs 6-7, so a wrong assignment shows up as a row
             whose bright cells sit in the wrong half.

  panels   — one small plot per subject with early and middle 6 Hz power across
             all 8 runs. This is the view that separates active from sham:
               active   — both early and middle high (stimulation runs the
                          whole run)
               sham     — early high, middle at floor (ramps up, ramps down)
               baseline — both at floor

Reads derivatives/eeg/stim_verification_runs.csv, so run stim_verification.py
first.

Usage
-----
    python plot_stim_verification.py
    python plot_stim_verification.py --subjects 10641 10810
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from config import (
    REPO_ROOT, SUBJECT_INFO, DISSERTATION_SUBJECTS,
    COLOR_ACTIVE, COLOR_SHAM, COLOR_BASELINE, get_condition_map,
)

DERIV_DIR = REPO_ROOT / 'derivatives' / 'eeg'
RUNS_CSV = DERIV_DIR / 'stim_verification_runs.csv'

ALL_RUNS = [1, 2, 3, 4, 5, 6, 7, 8]
DETECTED_COLORS = {'ACTIVE': COLOR_ACTIVE, 'SHAM': COLOR_SHAM, 'NONE': COLOR_BASELINE}


def load_runs(path: Optional[Path] = None) -> pd.DataFrame:
    path = Path(path or RUNS_CSV)
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found — run stim_verification.py first.'
        )
    return pd.read_csv(path, dtype={'subject_id': str})


def _subject_order(runs: pd.DataFrame) -> List[str]:
    """Group subjects by counterbalance so the expected pattern is contiguous."""
    subs = sorted(runs['subject_id'].unique())
    return sorted(subs, key=lambda s: (SUBJECT_INFO.get(s, {}).get('counterbalance', '?'), s))


def plot_heatmap(runs: pd.DataFrame, out_path: Path) -> Path:
    """Sustained 6 Hz power for every subject x run."""
    order = _subject_order(runs)
    grid = (runs.pivot_table(index='subject_id', columns='run', values='middle_db')
                .reindex(index=order, columns=ALL_RUNS))

    fig, ax = plt.subplots(figsize=(9, max(8, 0.22 * len(order))))
    im = ax.imshow(grid.values, aspect='auto', cmap='magma',
                   interpolation='nearest', vmin=60, vmax=170)

    ax.set_xticks(range(len(ALL_RUNS)))
    ax.set_xticklabels(ALL_RUNS)
    ax.set_xlabel('Run')
    ax.set_yticks(range(len(order)))
    labels = []
    for s in order:
        cb = SUBJECT_INFO.get(s, {}).get('counterbalance', '?')
        star = '*' if s in DISSERTATION_SUBJECTS else ' '
        labels.append(f'{s} [{cb}]{star}')
    ax.set_yticklabels(labels, fontsize=6)

    # Boundaries around the two stimulation blocks
    for edge in (0.5, 2.5, 4.5, 6.5):
        ax.axvline(edge, color='white', lw=0.6, alpha=0.5)

    # Separator between the A and B groups
    cbs = [SUBJECT_INFO.get(s, {}).get('counterbalance', '?') for s in order]
    for i in range(1, len(cbs)):
        if cbs[i] != cbs[i - 1]:
            ax.axhline(i - 0.5, color='cyan', lw=1.4)

    ax.set_title('Sustained 6 Hz power by run\n'
                 'A: bright on runs 2-3   B: bright on runs 6-7   (* = defense sample)',
                 fontsize=10)
    fig.colorbar(im, ax=ax, label='middle-window 6 Hz power (dB)', shrink=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_panels(runs: pd.DataFrame, out_path: Path,
                subjects: Optional[List[str]] = None) -> Path:
    """One panel per subject: early vs middle 6 Hz power across runs."""
    order = subjects if subjects else _subject_order(runs)
    order = [s for s in order if s in set(runs['subject_id'])]

    n = len(order)
    ncols = 4 if n > 4 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.5 * nrows),
                             squeeze=False)

    for idx, sid in enumerate(order):
        ax = axes[idx // ncols][idx % ncols]
        sub = runs[runs['subject_id'] == sid].set_index('run').reindex(ALL_RUNS)
        cb = SUBJECT_INFO.get(sid, {}).get('counterbalance', '?')
        cmap = get_condition_map(cb)

        # Shade the two stimulation blocks by what the counterbalance claims
        for lo, hi in [(1.5, 3.5), (5.5, 7.5)]:
            claimed = cmap.get(int(lo + 1), 'unknown')
            ax.axvspan(lo, hi, color=DETECTED_COLORS.get(claimed.upper(), '#EEEEEE'),
                       alpha=0.13, zorder=0)

        ax.plot(ALL_RUNS, sub['early_db'], 'o--', color='#888888', ms=3.5, lw=1,
                label='early (0-45 s)', zorder=2)
        ax.plot(ALL_RUNS, sub['middle_db'], 'o-', color='#222222', ms=4, lw=1.4,
                label='middle (sustained)', zorder=3)

        for run in ALL_RUNS:
            det = sub.loc[run, 'detected'] if run in sub.index else None
            if isinstance(det, str) and det in DETECTED_COLORS:
                ax.scatter([run], [sub.loc[run, 'middle_db']], s=58, zorder=4,
                           color=DETECTED_COLORS[det], edgecolors='white', linewidths=0.7)

        star = ' *' if sid in DISSERTATION_SUBJECTS else ''
        ax.set_title(f'sub-{sid}  [CB {cb}]{star}', fontsize=9)
        ax.set_xticks(ALL_RUNS)
        ax.tick_params(labelsize=7)
        ax.set_ylim(55, 175)
        ax.grid(alpha=0.2, lw=0.5)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')

    handles = [
        plt.Line2D([], [], color='#222222', marker='o', lw=1.4, label='middle (sustained)'),
        plt.Line2D([], [], color='#888888', marker='o', ls='--', lw=1, label='early (0-45 s)'),
        Patch(color=COLOR_ACTIVE, label='detected ACTIVE'),
        Patch(color=COLOR_SHAM, label='detected SHAM'),
        Patch(color=COLOR_BASELINE, label='detected NONE'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle('6 Hz power across all 8 runs — shading shows the condition '
                 'the counterbalance claims', y=1.015, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--subjects', nargs='+', default=None)
    parser.add_argument('--output-dir', default=str(DERIV_DIR))
    args = parser.parse_args(argv)

    runs = load_runs()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.subjects:
        p = plot_panels(runs, out_dir / 'stim_check_panels_subset.png', args.subjects)
        print(f'Wrote {p}')
    else:
        print(f'Wrote {plot_heatmap(runs, out_dir / "stim_check_heatmap.png")}')
        print(f'Wrote {plot_panels(runs, out_dir / "stim_check_panels.png")}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
