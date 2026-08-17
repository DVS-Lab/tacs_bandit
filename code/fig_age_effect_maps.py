"""
fig_age_effect_maps.py — Both age effects are diffuse, not focal

Two vertexwise maps of the *same* subjects on the *same* scale: the
correlation between age and cortical thickness, and between age and delivered
field. Expressing both in correlation units rather than mm and V/m is the
point — it is what makes the shapes comparable.

**This figure was built expecting a contrast that is not there.** The
young-minus-old difference map in fig_efield_age looks strikingly frontal, so
the field deficit appeared regionally specific while thinning looked global.
That impression is an artefact of absolute units: the difference is largest
where the field is largest, and the field is largest under the electrode. In
relative terms the two are near-identical in extent —

    age x thickness   26.1% of vertices suprathreshold, median r = -.332
    age x field       35.7% of vertices suprathreshold, median r = -.330

Both effects are diffuse. That is the correct reading, and it fits the
mechanism better than the focal one did: distance from scalp to cortex is a
whole-brain property, so a geometric account predicts the field should fall
roughly proportionally everywhere, which is what these maps show.

Do not describe the E-field deficit as frontally specific on the basis of the
difference map.

Correlation maps rather than young-minus-old difference maps. A tertile split
discards two thirds of the sample at every vertex; the correlation uses all of
it, and the underlying claim is about a continuous relationship anyway.

Maps are thresholded at the critical r for p < .05, **uncorrected**. These are
descriptive maps of where an effect already established in the ROI analyses
sits, not a whole-brain search, and no vertexwise claim should be made from
them.

Requires the FreeSurfer deliveries and the fsaverage E-field overlays.
Resampled thickness is cached in data/cache/ since it takes a couple of
minutes to rebuild.

Usage
-----
    python fig_global_vs_focal.py
    python fig_global_vs_focal.py --refresh
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
from scipy import stats
from scipy.spatial import cKDTree

from config import (REPO_ROOT, EFIELD_CSV_PATH, FREESURFER_ROOT)
from paper_style import (WIDTH_2COL, FONT_AXIS_TITLE, FONT_TICK,
                         FONT_PANEL_LABEL)
from fig_efield_age import (OVERLAY_DIR, brain_axes, make_surface_drawer,
                            roi_mask, load_overlay)

FIG_DIR = REPO_ROOT / 'data' / 'figures' / 'paper'
CACHE = REPO_ROOT / 'data' / 'cache' / 'thickness_fsaverage.npz'
HEMI, SIDE = 'lh', 'left'


def resample_thickness(refresh: bool = False) -> Tuple[np.ndarray, list]:
    """
    Each subject's cortical thickness on the fsaverage surface.

    recon-all writes thickness in native space, where vertex counts differ
    between subjects, so nothing can be averaged until it is resampled.
    `?h.fsaverage.sphere.reg` places every native vertex on the fsaverage
    sphere, and a nearest-neighbour lookup there carries the value across --
    the same operation mri_surf2surf performs, done here so the figure does
    not depend on a local FreeSurfer install matching the one that produced
    the data.
    """
    if CACHE.exists() and not refresh:
        z = np.load(CACHE, allow_pickle=True)
        return z['thickness'], list(z['subject_ids'])

    from nilearn import datasets, surface
    fsavg = datasets.fetch_surf_fsaverage('fsaverage')
    sphere, _ = surface.load_surf_mesh(fsavg[f'sphere_{SIDE}'])
    unit = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    target = unit(sphere)

    rows, ids = [], []
    subs = sorted(FREESURFER_ROOT.glob('*/sub-*'))
    print(f'Resampling thickness for {len(subs)} subjects (cached afterwards)')
    for d in subs:
        reg, th = d / 'surf' / f'{HEMI}.fsaverage.sphere.reg', d / 'surf' / f'{HEMI}.thickness'
        if not (reg.exists() and th.exists()):
            continue
        native, _ = nib.freesurfer.read_geometry(str(reg))
        vals = nib.freesurfer.read_morph_data(str(th))
        _, idx = cKDTree(unit(native)).query(target, k=1)
        rows.append(vals[idx])
        ids.append(d.name.split('_')[0].replace('sub-', ''))

    arr = np.vstack(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, thickness=arr, subject_ids=np.array(ids))
    print(f'  cached {arr.shape} -> {CACHE}')
    return arr, ids


def vertexwise_r(data: np.ndarray, age: np.ndarray) -> np.ndarray:
    """Pearson r between age and each column, done in one pass."""
    a = age - age.mean()
    d = data - data.mean(axis=0, keepdims=True)
    denom = np.sqrt((a ** 2).sum() * (d ** 2).sum(axis=0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom > 0, (a @ d) / denom, 0.0)


def build(out_name: str = 'fig_global_vs_focal', refresh: bool = False) -> Path:
    from nilearn import datasets, plotting

    thick, thick_ids = resample_thickness(refresh)
    thick = pd.DataFrame(thick, index=thick_ids)

    e = pd.read_csv(EFIELD_CSV_PATH, dtype={'subject_id': str})
    m = pd.read_csv(REPO_ROOT / 'data' / 'master_subject_data.csv',
                    dtype={'subject_id': str})
    d = e.merge(m[['subject_id', 'age']], on='subject_id', how='left')
    d['age'] = pd.to_numeric(d['age'], errors='coerce')
    d = d.dropna(subset=['age'])
    if 't1_only' in d.columns:
        d = d[~d['t1_only'].astype(bool)]

    # One sample for both maps: any subject missing either modality is dropped
    # from both, so the two panels cannot be computed on different people.
    field = {s: v for s in d.subject_id
             if (v := load_overlay(s, HEMI)) is not None}
    keep = [s for s in d.subject_id if s in field and s in thick.index]
    d = d[d.subject_id.isin(keep)].reset_index(drop=True)
    age = d['age'].values
    print(f'N = {len(d)} with age, thickness, and field (T1-only excluded)')

    r_thick = vertexwise_r(thick.loc[keep].values, age)
    r_field = vertexwise_r(np.vstack([field[s] for s in keep]), age)

    crit = stats.t.ppf(0.975, len(age) - 2)
    r_crit = crit / np.sqrt(len(age) - 2 + crit ** 2)
    print(f'  threshold |r| > {r_crit:.3f}  (p < .05, uncorrected)')
    for lbl, r in [('thickness', r_thick), ('field', r_field)]:
        supra = np.abs(r) > r_crit
        print(f'  age x {lbl:9s}: {100 * supra.mean():5.1f}% of vertices, '
              f'median r where suprathreshold = {np.median(r[supra]):+.3f}')

    fsavg = datasets.fetch_surf_fsaverage('fsaverage')
    FIG_W, FIG_H = WIDTH_2COL * 0.78, WIDTH_2COL * 0.78 * 0.46
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    draw = make_surface_drawer(fig, plotting, fsavg[f'infl_{SIDE}'],
                               fsavg[f'sulc_{SIDE}'], roi_mask(fsavg, SIDE), SIDE)

    # Effects top out near |r| = .5, so a -1..1 range renders them nearly white.
    RMAX = 0.6
    BW, top = 0.415, 0.845
    ax_a, bh = brain_axes(fig, 0.010, top, BW, FIG_W, FIG_H)
    draw(ax_a, r_thick, 'coolwarm', RMAX, r_crit)
    ax_b, _ = brain_axes(fig, 0.445, top, BW, FIG_W, FIG_H)
    draw(ax_b, r_field, 'coolwarm', RMAX, r_crit)

    TITLE = dict(fontsize=FONT_AXIS_TITLE, ha='center', va='top')
    fig.text(0.010 + BW / 2, 0.995, 'Age $\\times$ cortical thickness', **TITLE)
    fig.text(0.445 + BW / 2, 0.995, 'Age $\\times$ delivered field', **TITLE)

    cax = fig.add_axes([0.885, top - bh * 0.80, 0.016, bh * 0.60])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-RMAX, RMAX),
                                            cmap='coolwarm'), cax=cax)
    cb.set_label('$r$ with age', fontsize=FONT_TICK, labelpad=2)
    cb.set_ticks([-RMAX, -0.3, 0.0, 0.3, RMAX])
    cax.tick_params(labelsize=FONT_TICK - 1, pad=1)

    LABEL_KW = dict(fontsize=FONT_PANEL_LABEL, fontweight='bold', va='top')
    fig.text(0.004, 0.995, 'a', **LABEL_KW)
    fig.text(0.439, 0.995, 'b', **LABEL_KW)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f'{out_name}.png'
    fig.savefig(png, dpi=400, facecolor='white')
    fig.savefig(FIG_DIR / f'{out_name}.svg', dpi=400, facecolor='white')
    plt.close(fig)
    print(f'wrote {png}')
    return png


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--refresh', action='store_true',
                   help='rebuild the resampled-thickness cache')
    p.add_argument('--name', default='fig_global_vs_focal')
    args = p.parse_args(argv)
    build(out_name=args.name, refresh=args.refresh)
    return 0


if __name__ == '__main__':
    sys.exit(main())
