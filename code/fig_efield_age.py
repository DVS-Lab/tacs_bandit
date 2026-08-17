"""
fig_efield_age.py — Figure: simulated E-field declines with age

Two versions, from the same data and the same statistics.

Full (default), four panels:

  A, B   group-average field on the fsaverage cortical surface, young vs old
         age tertile, on a shared colour scale
  C      the young-minus-old difference, so the spatial extent of the deficit
         is visible rather than inferred
  D      the individual-level relationship: age against mean |E| in the F3 ROI

Compact (--compact), two panels: C and D relabelled A and B. The group maps
are dropped because they show where the field lands under this montage, which
is a property of the electrodes rather than of the result and is the same in
both groups. Their group descriptives move into the difference map's header,
so no information is lost.

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
    python fig_efield_age.py --compact
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

# Widths, fonts, and colours come from paper_style, which also applies the
# matching rcParams on import. They used to be duplicated here, and the copy
# drifted: this script never set font.family, so it rendered in DejaVu Sans
# while every other figure used Arial — silently, for as long as it existed.
from paper_style import (
    WIDTH_2COL, FONT_AXIS_TITLE, FONT_TICK, FONT_PANEL_LABEL,
    AGE_YOUNG, AGE_OLD, REGRESSION_COLOR,
)


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


# Group-mean F3 electrode position in MNI space, obtained by transforming each
# subject's F3 through subject2mni_coords. F3 is an EEG 10-10 landmark derived
# from head geometry, so after normalization it lands in essentially the same
# place for everyone (SD 0.1 mm across subjects).
F3_MNI = np.array([-52.0, 44.6, 44.2])
ROI_RADIUS_MM = 20.0


def roi_mask(fsavg, side: str, centre=F3_MNI, radius=ROI_RADIUS_MM) -> np.ndarray:
    """
    Vertices of the fsaverage surface within `radius` of the F3 centre.

    Distances are computed on the *pial* surface, then displayed on the
    inflated one. Two reasons for pial specifically:

    - The inflated surface has no meaningful metric, so measuring on it would
      give a mask of the wrong shape and size.
    - The ROI is a sphere centred on the F3 *scalp electrode*, which sits
      roughly 18 mm above cortex, and it captures gray-matter volume near the
      pial boundary — the gyral crowns closest to the skull. Measuring to the
      white surface instead puts the nearest vertex 21.6 mm away, so a 20 mm
      sphere would intersect nothing at all.

    fsaverage is in MNI305 rather than MNI152, which differ by a small affine.
    For a schematic showing roughly where the measurement was taken that is
    immaterial, but it means the outline should be read as indicative rather
    than as the exact per-subject ROI, which was defined in each individual's
    own space.
    """
    from nilearn import surface
    coords, _ = surface.load_surf_mesh(fsavg[f'pial_{side}'])
    return (np.linalg.norm(coords - centre, axis=1) <= radius)


def outline_from_mask(fsavg, side: str, mask: np.ndarray) -> np.ndarray:
    """
    Boundary vertices of a mask: inside the mask but adjacent to something
    outside it. Drawing only the boundary keeps the field visible underneath,
    which a filled overlay would hide.
    """
    from nilearn import surface
    _, faces = surface.load_surf_mesh(fsavg[f'infl_{side}'])
    edge = np.zeros(mask.shape, dtype=bool)
    for tri in faces:
        vals = mask[tri]
        if vals.any() and not vals.all():
            edge[tri[vals]] = True
    return edge


# A matplotlib 3D axes inscribes its render in a cube bounding box, so the
# brain never fills its rectangle and reports an extent far larger than the
# ink. Laying panels out against the rectangle is what leaves surface figures
# full of dead space. These constants were measured directly off the rendered
# canvas by thresholding non-white pixels at a range of zoom levels. Zoom 1.50
# is the working point: past roughly 1.55 the occipital pole is cut off by the
# rectangle edge, which shows up as a flat vertical crop on the back of the
# brain. Together they give the real ink box, which everything else is laid
# out against.
SURF_ZOOM = 1.50
SURF_FILL = 0.953      # brain width as a fraction of the rectangle's width
SURF_ASPECT = 1.378    # brain width / brain height
SURF_YOFF = 0.0097     # brain centre above rectangle centre, in rect heights


def brain_axes(fig, x_left, top, brain_w, fig_w, fig_h):
    """
    Add a 3D axes positioned so the *rendered brain* occupies exactly
    (x_left, top) to (x_left + brain_w, top - brain_h).

    Returns (axes, brain_h). The underlying rectangle is deliberately larger
    than the brain and may overrun the figure edge; it is transparent, so
    neighbouring panels can overlap it without any visible effect.
    """
    rect_w = brain_w / SURF_FILL
    rect_h = rect_w * fig_w / fig_h          # square in inches
    brain_h = (rect_w * fig_w / SURF_ASPECT) / fig_h
    cx = x_left + brain_w / 2
    cy = top - brain_h / 2 - SURF_YOFF * rect_h
    ax = fig.add_axes([cx - rect_w / 2, cy - rect_h / 2, rect_w, rect_h],
                      projection='3d')
    return ax, brain_h


def scatter_panel(ax, d: pd.DataFrame, lo: float, hi: float) -> Tuple[float, float]:
    """
    Age against mean |E| in the F3 ROI, with fit and 95% CI. Returns (r, p).

    Shared by both figure versions so the statistics reported in each are
    computed once, from the same code, and cannot drift apart.
    """
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

    # Tertile boundaries, so the reader can see which subjects contribute to
    # the group maps.
    for bound in (lo, hi):
        ax.axvline(bound, color='#BDBDBD', ls=':', lw=0.7, zorder=0)

    ax.set_xlabel('Age (years)', fontsize=FONT_AXIS_TITLE, labelpad=2)
    ax.set_ylabel('Mean |E| in DLPFC ROI (V/m)', fontsize=FONT_AXIS_TITLE, labelpad=2)
    ax.tick_params(labelsize=FONT_TICK, pad=1.5)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.text(0.985, 0.96,
            f'$r$ = {r:.3f}, $p$ = {p:.3f}\n$N$ = {len(x)}',
            transform=ax.transAxes, ha='right', va='top', fontsize=FONT_TICK,
            linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='none', alpha=0.85))
    return r, p


def make_surface_drawer(fig, plotting, surf, bg, roi, side: str):
    """
    Returns a function that renders one field map onto a 3D axes.
    """
    def draw(ax, data, cmap, vmax, thresh, show_roi: bool = True):
        plotting.plot_surf_stat_map(
            surf, data, hemi=side, view='lateral', bg_map=bg,
            colorbar=False, cmap=cmap, vmax=vmax, threshold=thresh,
            axes=ax, figure=fig, darkness=0.6)
        if show_roi:
            # plot_surf_contours draws onto the existing surface axes; a second
            # plot_surf call would re-render and wipe out the field underneath.
            plotting.plot_surf_contours(
                surf, roi.astype(int), levels=[1], colors=['#00E5FF'],
                axes=ax, figure=fig, linewidths=1.1)
        # The surface is ~300k triangles. Left as vectors it exports to a
        # >150 MB SVG that no journal system will take, so the mesh is
        # rasterized at the figure dpi while text, the scatter, and the
        # colourbars stay as vectors.
        for coll in ax.collections:
            coll.set_rasterized(True)
        ax.set_box_aspect(None, zoom=SURF_ZOOM)
    return draw


def _prepare(hemi: str):
    """Load data, compute the group maps, and fetch the surfaces they render on."""
    from nilearn import datasets

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

    # nilearn names these 'infl_left'/'infl_right', not by the lh/rh suffix the
    # SimNIBS overlays use.
    fsavg = datasets.fetch_surf_fsaverage('fsaverage')
    side = 'left' if hemi == 'lh' else 'right'
    roi = roi_mask(fsavg, side)
    print(f'  ROI outline: {roi.sum()} vertices within {ROI_RADIUS_MM:.0f} mm of F3')

    return dict(d=d, lo=lo, hi=hi, side=side, young=young, old=old,
                young_map=young_map, old_map=old_map, n_y=n_y, n_o=n_o,
                surf=fsavg[f'infl_{side}'], bg=fsavg[f'sulc_{side}'], roi=roi)


def _save(fig, out_name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f'{out_name}.png'
    fig.savefig(png, dpi=400, facecolor='white')
    fig.savefig(FIG_DIR / f'{out_name}.svg', dpi=400, facecolor='white')
    plt.close(fig)
    return png


def build(out_name: str = 'fig_efield_age', hemi: str = 'lh') -> Path:
    """Full four-panel version: group maps, difference map, and scatter."""
    from nilearn import plotting

    c = _prepare(hemi)

    # Everything below is placed against the brains' measured ink boxes rather
    # than through a gridspec. The panels mix 3D surfaces with a 2D scatter,
    # which size themselves differently inside a grid cell and leave labels and
    # colourbars drifting out of alignment.
    FIG_W, FIG_H = WIDTH_2COL, WIDTH_2COL * 0.70
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    draw = make_surface_drawer(fig, plotting, c['surf'], c['bg'], c['roi'],
                               c['side'])

    BW = 0.36                       # brain width, figure fraction
    A_X, B_X = 0.015, 0.395
    TITLE_KW = dict(fontsize=FONT_AXIS_TITLE, ha='center', va='top', linespacing=1.5)
    LABEL_KW = dict(fontsize=FONT_PANEL_LABEL, fontweight='bold', va='top')

    # --- A and B: group averages on a shared vertical colourbar ------------
    vmax = float(np.percentile(np.concatenate([c['young_map'], c['old_map']]), 99))
    top1 = 0.930

    ax_a, bh = brain_axes(fig, A_X, top1, BW, FIG_W, FIG_H)
    draw(ax_a, c['young_map'], 'inferno', vmax, vmax * 0.05)
    ax_b, _ = brain_axes(fig, B_X, top1, BW, FIG_W, FIG_H)
    draw(ax_b, c['old_map'], 'inferno', vmax, vmax * 0.05)

    # Titles as figure text, not ax.set_title, which anchors to the oversized
    # rectangle and floats the caption well above the brain.
    title1_y = 0.995
    fig.text(A_X + BW / 2, title1_y,
             f"Younger (age $\\leq$ {c['lo']:.0f})\n$n$ = {c['n_y']}", **TITLE_KW)
    fig.text(B_X + BW / 2, title1_y,
             f"Older (age $\\geq$ {c['hi']:.0f})\n$n$ = {c['n_o']}", **TITLE_KW)

    cax = fig.add_axes([0.790, top1 - bh * 0.78, 0.014, bh * 0.58])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax),
                                            cmap='inferno'), cax=cax)
    cb.set_label('|E| (V/m)', fontsize=FONT_TICK, labelpad=2)
    cax.tick_params(labelsize=FONT_TICK - 1, pad=1)

    # --- C: difference map, centred over its own colourbar -----------------
    title2_y = top1 - bh - 0.035
    top2 = title2_y - 0.030
    diff = c['young_map'] - c['old_map']
    dmax = float(np.percentile(np.abs(diff), 99))

    ax_c, _ = brain_axes(fig, A_X, top2, BW, FIG_W, FIG_H)
    draw(ax_c, diff, 'coolwarm', dmax, dmax * 0.1)
    fig.text(A_X + BW / 2, title2_y, 'Younger $-$ older', **TITLE_KW)

    cbar_w = 0.19
    cax2 = fig.add_axes([A_X + (BW - cbar_w) / 2, top2 - bh - 0.020,
                         cbar_w, 0.018])
    cb2 = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-dmax, dmax),
                                             cmap='coolwarm'), cax=cax2,
                       orientation='horizontal')
    cb2.set_label('$\\Delta$ |E| (V/m)', fontsize=FONT_TICK, labelpad=1)
    cax2.tick_params(labelsize=FONT_TICK - 1, pad=1)

    # --- D: the individual-level relationship ------------------------------
    # Bottom edge sits level with C's brain, top a little below C's title so
    # the two panels read as one row.
    ax_d = fig.add_axes([0.575, top2 - bh + 0.055, 0.395, bh - 0.055])
    r, p = scatter_panel(ax_d, c['d'], c['lo'], c['hi'])

    # Panel labels go on the figure, not the axes, so C and D share an exact
    # y — which axes-relative placement cannot guarantee when one panel is a
    # 3D surface and the other a 2D scatter.
    fig.text(0.004, title1_y, 'a', **LABEL_KW)
    fig.text(0.384, title1_y, 'b', **LABEL_KW)
    fig.text(0.004, title2_y, 'c', **LABEL_KW)
    fig.text(0.501, title2_y, 'd', **LABEL_KW)

    png = _save(fig, out_name)
    print(f"\nage x |E|: r = {r:.3f}, p = {p:.4f}, N = {len(c['d'])}")
    print(f"group difference: "
          f"{c['young'].mean_magnE.mean() - c['old'].mean_magnE.mean():+.4f} V/m")
    print(f'wrote {png}')
    return png


def build_compact(out_name: str = 'fig_efield_age_compact',
                  hemi: str = 'lh') -> Path:
    """
    Two-panel version: the difference map and the scatter.

    The two group maps carry nothing the difference map does not. What they
    show — where the field lands under this montage — is a property of the
    electrode placement, not of the result, and it is the same in both. The
    one thing they add is the group descriptives in their headers, so those
    move into panel A's title here and the panel stays self-contained.
    """
    from nilearn import plotting

    c = _prepare(hemi)

    FIG_W, FIG_H = WIDTH_2COL, WIDTH_2COL * 0.40
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    draw = make_surface_drawer(fig, plotting, c['surf'], c['bg'], c['roi'],
                               c['side'])

    BW, A_X = 0.375, 0.010
    top = 0.845

    diff = c['young_map'] - c['old_map']
    dmax = float(np.percentile(np.abs(diff), 99))
    ax_a, bh = brain_axes(fig, A_X, top, BW, FIG_W, FIG_H)
    draw(ax_a, diff, 'coolwarm', dmax, dmax * 0.1)

    y_rng = (c['young']['age'].min(), c['young']['age'].max())
    o_rng = (c['old']['age'].min(), c['old']['age'].max())
    title_y = 0.985
    fig.text(A_X + BW / 2, title_y,
             f"Younger $-$ older\n"
             f"$n$ = {c['n_y']}, {y_rng[0]:.0f}$-${y_rng[1]:.0f} y   vs   "
             f"$n$ = {c['n_o']}, {o_rng[0]:.0f}$-${o_rng[1]:.0f} y",
             fontsize=FONT_AXIS_TITLE, ha='center', va='top', linespacing=1.6)

    cbar_w = 0.19
    cax = fig.add_axes([A_X + (BW - cbar_w) / 2, top - bh - 0.030,
                        cbar_w, 0.030])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-dmax, dmax),
                                            cmap='coolwarm'), cax=cax,
                      orientation='horizontal')
    cb.set_label('$\\Delta$ |E| (V/m)', fontsize=FONT_TICK, labelpad=1)
    cax.tick_params(labelsize=FONT_TICK - 1, pad=1)

    # Pulled in close to the brain: the gap only needs to clear the y-axis
    # label, and the panels read as a pair rather than two separate figures.
    ax_b = fig.add_axes([0.470, top - bh + 0.085, 0.500, bh - 0.085])
    r, p = scatter_panel(ax_b, c['d'], c['lo'], c['hi'])

    LABEL_KW = dict(fontsize=FONT_PANEL_LABEL, fontweight='bold', va='top')
    fig.text(0.004, title_y, 'a', **LABEL_KW)
    fig.text(0.395, title_y, 'b', **LABEL_KW)

    png = _save(fig, out_name)
    print(f"\nage x |E|: r = {r:.3f}, p = {p:.4f}, N = {len(c['d'])}")
    print(f'wrote {png}')
    return png


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--hemi', default='lh', choices=['lh', 'rh'],
                   help='left by default: the montage targets left DLPFC')
    p.add_argument('--name', default=None)
    p.add_argument('--compact', action='store_true',
                   help='two-panel version: difference map + scatter only')
    args = p.parse_args(argv)
    if args.compact:
        build_compact(out_name=args.name or 'fig_efield_age_compact', hemi=args.hemi)
    else:
        build(out_name=args.name or 'fig_efield_age', hemi=args.hemi)
    return 0


if __name__ == '__main__':
    sys.exit(main())
