"""
fig_efield_age.py — Figure: simulated E-field declines with age

Four panels:

  A, B   group-average field on the fsaverage cortical surface, young vs old
         age tertile, on a shared colour scale
  C      the young-minus-old difference, so the spatial extent of the deficit
         is visible rather than inferred
  D      the individual-level relationship: age against mean |E| in the F3 ROI

**Why group averages rather than two example brains.** The obvious version of
this figure puts one young and one old subject side by side. That is not
defensible here. The correlation is r ~ -.34, about 11% of variance, and across
all 400 possible young/old pairings in this sample the apparent difference
ranges from -0.070 to +0.111 V/m against a true group difference of +0.026 —
and in 24% of pairings the *older* subject has the higher field. Which pair you
pick determines what the reader concludes. Averaging 20 subjects per group
shows the effect that actually exists.

**The seven T1-only subjects are excluded throughout.** Their head models were
built without a FLAIR, and their field estimates are systematically ~28% lower
(p = .013 controlling for age) because skull segmentation degrades without it.
Including them would bias the maps in the same direction as the hypothesis.

Requires the fsaverage overlays produced by the simNIBS project:
    simnibs_python scripts/to_fsaverage.py

Usage
-----
    python fig_efield_age.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

from config import REPO_ROOT, EFIELD_CSV_PATH

SIMNIBS_DIR = Path.home() / 'Desktop' / 'projects' / 'tacs_bandit' / 'simNIBS'
OVERLAY_DIR = SIMNIBS_DIR / 'fsavg_overlays'
FIG_DIR = REPO_ROOT / 'data' / 'figures' / 'paper'

# --- Figure style, matching the paper notebook ---
CM_TO_IN = 1 / 2.54
WIDTH_2COL = 17.5 * CM_TO_IN
FONT_AXIS = 7
FONT_TICK = 6.5
FONT_PANEL = 9
AGE_YOUNG, AGE_OLD = '#1565C0', '#FFB300'
REGRESSION_COLOR = '#404040'


def load_subject_data() -> pd.DataFrame:
    """E-field summary joined to age, T1-only subjects dropped."""
    efield = pd.read_csv(EFIELD_CSV_PATH, dtype={'subject_id': str})
    master = pd.read_csv(REPO_ROOT / 'data' / 'master_subject_data.csv',
                         dtype={'subject_id': str})
    d = efield.merge(master[['subject_id', 'age']], on='subject_id', how='left')
    d['age'] = pd.to_numeric(d['age'], errors='coerce')
    d = d.dropna(subset=['age', 'mean_magnE'])
    if 't1_only' in d.columns:
        d = d[~d['t1_only'].astype(bool)]
    return d.reset_index(drop=True)


def load_overlay(sub_id: str, hemi: str = 'lh') -> Optional[np.ndarray]:
    """Read one subject's fsaverage-space field magnitude for a hemisphere."""
    folder = OVERLAY_DIR / f'fsavg_sub-{sub_id}'
    hits = sorted(folder.glob(f'{hemi}.*.magn'))
    if not hits:
        return None
    return nib.freesurfer.read_morph_data(str(hits[0]))


def group_mean(sub_ids, hemi: str = 'lh') -> Tuple[Optional[np.ndarray], int]:
    """Vertexwise mean across subjects, skipping any without an overlay."""
    stack = [v for v in (load_overlay(s, hemi) for s in sub_ids) if v is not None]
    if not stack:
        return None, 0
    return np.mean(np.vstack(stack), axis=0), len(stack)


def build(out_name: str = 'fig_efield_age', hemi: str = 'lh') -> Path:
    from nilearn import datasets, plotting

    d = load_subject_data()
    lo, hi = d['age'].quantile([1 / 3, 2 / 3])
    young = d[d['age'] <= lo]
    old = d[d['age'] >= hi]

    print(f'N = {len(d)} (T1-only excluded)')
    print(f'  young tertile: n={len(young)}, age <= {lo:.0f}, '
          f'mean |E| = {young.mean_magnE.mean():.4f}')
    print(f'  old tertile  : n={len(old)}, age >= {hi:.0f}, '
          f'mean |E| = {old.mean_magnE.mean():.4f}')

    young_map, n_y = group_mean(young['subject_id'], hemi)
    old_map, n_o = group_mean(old['subject_id'], hemi)
    if young_map is None or old_map is None:
        raise SystemExit(
            f'No fsaverage overlays found in {OVERLAY_DIR}. Run:\n'
            f'  simnibs_python {SIMNIBS_DIR}/scripts/to_fsaverage.py')
    print(f'  overlays used: {n_y} young, {n_o} old')

    fsavg = datasets.fetch_surf_fsaverage('fsaverage')
    surf = fsavg[f'infl_{hemi}']
    bg = fsavg[f'sulc_{hemi}']

    fig = plt.figure(figsize=(WIDTH_2COL, WIDTH_2COL * 0.62))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.05, wspace=0.02)

    # Shared scale across A and B so the two are actually comparable.
    vmax = float(np.percentile(np.concatenate([young_map, old_map]), 99))

    for col, (data, label, n) in enumerate([
        (young_map, f'Younger (age $\\leq$ {lo:.0f})', n_y),
        (old_map, f'Older (age $\\geq$ {hi:.0f})', n_o),
    ]):
        ax = fig.add_subplot(gs[0, col], projection='3d')
        plotting.plot_surf_stat_map(
            surf, data, hemi=hemi, view='lateral', bg_map=bg,
            colorbar=False, cmap='inferno', vmax=vmax, threshold=vmax * 0.05,
            axes=ax, figure=fig, darkness=0.6)
        ax.set_title(f'{label}\n$n$ = {n}', fontsize=FONT_AXIS, pad=0)
        ax.text2D(0.02, 0.95, 'AB'[col], transform=ax.transAxes,
                  fontsize=FONT_PANEL, fontweight='bold')

    # --- C: difference map -------------------------------------------------
    diff = young_map - old_map
    dmax = float(np.percentile(np.abs(diff), 99))
    ax = fig.add_subplot(gs[0, 2], projection='3d')
    plotting.plot_surf_stat_map(
        surf, diff, hemi=hemi, view='lateral', bg_map=bg,
        colorbar=False, cmap='coolwarm', vmax=dmax, threshold=dmax * 0.1,
        axes=ax, figure=fig, darkness=0.6)
    ax.set_title('Younger $-$ older', fontsize=FONT_AXIS, pad=0)
    ax.text2D(0.02, 0.95, 'C', transform=ax.transAxes,
              fontsize=FONT_PANEL, fontweight='bold')

    # Colourbars, placed manually so the surfaces stay flush.
    cax1 = fig.add_axes([0.13, 0.545, 0.30, 0.014])
    plt.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(0, vmax), cmap='inferno'),
        cax=cax1, orientation='horizontal')
    cax1.set_xlabel('|E| (V/m)', fontsize=FONT_TICK, labelpad=1)
    cax1.tick_params(labelsize=FONT_TICK - 1, pad=1)

    cax2 = fig.add_axes([0.70, 0.545, 0.20, 0.014])
    plt.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(-dmax, dmax), cmap='coolwarm'),
        cax=cax2, orientation='horizontal')
    cax2.set_xlabel('$\\Delta$ |E| (V/m)', fontsize=FONT_TICK, labelpad=1)
    cax2.tick_params(labelsize=FONT_TICK - 1, pad=1)

    # --- D: the individual-level relationship ------------------------------
    ax = fig.add_subplot(gs[1, :])
    x = d['age'].values
    y = d['mean_magnE'].values

    norm = plt.Normalize(x.min(), x.max())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'blue_gold', [AGE_YOUNG, AGE_OLD])
    ax.scatter(x, y, c=cmap(norm(x)), s=22, edgecolors='white',
               linewidths=0.5, zorder=3)

    slope, intercept, r, p, _ = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 200)
    ys = intercept + slope * xs
    resid_se = np.sqrt(np.sum((y - (intercept + slope * x)) ** 2) / (len(x) - 2))
    ci = stats.t.ppf(0.975, len(x) - 2) * resid_se * np.sqrt(
        1 / len(x) + (xs - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    ax.fill_between(xs, ys - ci, ys + ci, color='gray', alpha=0.18, linewidth=0)
    ax.plot(xs, ys, color=REGRESSION_COLOR, lw=1.3, zorder=2)

    # Mark the tertile boundaries that define panels A and B.
    for bound in (lo, hi):
        ax.axvline(bound, color='#BDBDBD', ls=':', lw=0.7, zorder=0)

    ax.set_xlabel('Age (years)', fontsize=FONT_AXIS)
    ax.set_ylabel('Mean |E| in DLPFC ROI (V/m)', fontsize=FONT_AXIS)
    ax.tick_params(labelsize=FONT_TICK)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.text(0.985, 0.94,
            f'$r$ = {r:.3f}, $p$ = {p:.3f}, $N$ = {len(x)}',
            transform=ax.transAxes, ha='right', va='top', fontsize=FONT_TICK,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='none', alpha=0.85))
    ax.text(0.005, 0.94, 'D', transform=ax.transAxes,
            fontsize=FONT_PANEL, fontweight='bold')

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f'{out_name}.png'
    fig.savefig(png, dpi=400, bbox_inches='tight', facecolor='white')
    fig.savefig(FIG_DIR / f'{out_name}.svg', bbox_inches='tight',
                facecolor='white')
    plt.close(fig)

    print(f'\nage x |E|: r = {r:.3f}, p = {p:.4f}, N = {len(x)}')
    print(f'group difference: {young.mean_magnE.mean() - old.mean_magnE.mean():+.4f} V/m')
    print(f'wrote {png}')
    return png


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--hemi', default='lh', choices=['lh', 'rh'],
                   help='left by default: the montage targets left DLPFC')
    p.add_argument('--name', default='fig_efield_age')
    args = p.parse_args(argv)
    build(out_name=args.name, hemi=args.hemi)
    return 0


if __name__ == '__main__':
    sys.exit(main())
