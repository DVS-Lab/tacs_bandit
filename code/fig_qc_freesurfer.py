"""
fig_qc_freesurfer.py — Visual check of the FreeSurfer reconstructions flagged in the audit

The Stage 2 variable audit flagged four subjects whose morphometry sits far
from the rest of the sample:

  11461   thinnest cortex in the sample (1.85 mm) and 41 surface defects
  10866   ventricles 97k mm3, twice the median for over-70s
  11472   45 surface defects (median 13)
  10661   38 surface defects

Numbers alone cannot say whether these are real anatomy or reconstruction
failures -- a pial surface that leaks into dura or skull, a white surface that
misses a gyrus. This figure shows the surfaces on the T1, the standard way of
answering that.

Two reference subjects are shown for comparison, **chosen by rule, not by
eye**: among subjects not flagged and with no more than the median number of
defects, the one whose mean cortical thickness is closest to the median of its
age band (70+, and 50-60).

Each row is one subject; the three panels are

  coronal through dorsolateral frontal cortex (40 mm behind the frontal pole)
  coronal through the lateral ventricles (their centroid)
  axial through the lateral ventricles (their centroid)

with the white surface in yellow, the pial surface in red, and the lateral
ventricles tinted blue from aseg. Slice positions are anchored to each
subject's own anatomy, not to fixed coordinates, so the rows are comparable.

Usage
-----
    python fig_qc_freesurfer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from config import REPO_ROOT, FREESURFER_ROOT, FREESURFER_MORPH_PATH
from paper_style import WIDTH_2COL, FONT_AXIS_TITLE, FONT_TICK, FONT_PANEL_LABEL

FIG_DIR = REPO_ROOT / 'data' / 'figures' / 'qc'
FLAGGED = {
    '11461': 'thinnest cortex; 41 defects',
    '10866': 'ventricles 2x age median',
    '11472': '45 surface defects',
    '10661': '38 surface defects',
}
LAT_VENT = (4, 43)            # aseg: left / right lateral ventricle
HALF = 80.0                   # mm, half-width of each panel
STEP = 0.5                    # mm per pixel


def subject_dir(sid: str) -> Path:
    hits = sorted(FREESURFER_ROOT.glob(f'*/sub-{sid}_*'))
    if not hits:
        raise FileNotFoundError(f'no recon-all output for {sid}')
    return hits[0]


def references(fs: pd.DataFrame, age: pd.Series) -> dict:
    """One typical subject per age band, by rule."""
    d = fs.assign(age=fs.subject_id.map(age))
    d = d[~d.subject_id.isin(FLAGGED) & (d.surface_holes <= d.surface_holes.median())]
    out = {}
    for label, lo, hi in [('reference, 70+', 70, 200), ('reference, 50-60', 50, 60)]:
        band = d[(d.age >= lo) & (d.age <= hi)]
        target = band.lh_mean_thickness.median()
        pick = band.iloc[(band.lh_mean_thickness - target).abs().argsort().iloc[0]]
        out[pick.subject_id] = label
    return out


def plane_segments(verts, faces, axis, level, keep):
    """Line segments where a surface crosses the plane `axis = level`."""
    v = verts[faces]                                   # (F, 3, 3)
    s = v[..., axis] - level                           # signed distance
    cross = (s.min(axis=1) < 0) & (s.max(axis=1) > 0)
    v, s = v[cross], s[cross]
    segs = []
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        on = (s[:, a] * s[:, b]) < 0
        t = s[on, a] / (s[on, a] - s[on, b])
        p = v[on, a] + t[:, None] * (v[on, b] - v[on, a])
        segs.append((np.where(on)[0], p))
    # Each crossing triangle has exactly two crossing edges; pair them up.
    idx = np.concatenate([i for i, _ in segs])
    pts = np.concatenate([p for _, p in segs])
    order = np.argsort(idx, kind='stable')
    idx, pts = idx[order], pts[order]
    first = np.r_[True, idx[1:] != idx[:-1]]
    a, b = pts[first], pts[~first]
    n = min(len(a), len(b))
    return np.stack([a[:n][:, keep], b[:n][:, keep]], axis=1)


def resample(img, inv, axis, level, centre):
    """The volume on a regular plane in tkr-RAS, HALF mm either side of centre."""
    keep = [i for i in range(3) if i != axis]
    g = np.arange(-HALF, HALF, STEP)
    U, V = np.meshgrid(g + centre[keep[0]], g + centre[keep[1]], indexing='xy')
    world = np.zeros(U.shape + (3,))
    world[..., keep[0]], world[..., keep[1]], world[..., axis] = U, V, level
    vox = world @ inv[:3, :3].T + inv[:3, 3]
    return vox, keep


def draw(ax, sid, label, row, axis, level_of, title):
    d = subject_dir(sid)
    t1 = nib.load(str(d / 'mri' / 'T1.mgz'))
    aseg = nib.load(str(d / 'mri' / 'aseg.mgz'))
    tkr = t1.header.get_vox2ras_tkr()
    inv = np.linalg.inv(tkr)
    a = np.asarray(aseg.dataobj)

    # Anatomical anchors, in tkr-RAS.
    def ras(mask):
        ijk = np.argwhere(mask)
        return ijk @ tkr[:3, :3].T + tkr[:3, 3]
    brain = ras(a > 0)
    vent = ras(np.isin(a, LAT_VENT))
    centre = np.r_[brain[:, 0].mean(), vent[:, 1].mean(), vent[:, 2].mean()]
    level = level_of(brain, vent)

    vox, keep = resample(t1, inv, axis, level, centre)
    coords = np.moveaxis(vox, -1, 0)
    img = map_coordinates(np.asarray(t1.dataobj, dtype=float), coords, order=1)
    lab = map_coordinates(a, coords, order=0)
    # T1.mgz is intensity-normalised (white matter = 110), so a fixed window
    # keeps white matter grey rather than saturated and makes the grey/white
    # boundary -- the thing the yellow line should follow -- visible.
    ax.imshow(img, cmap='gray', origin='lower', vmin=0, vmax=165,
              extent=[-HALF, HALF, -HALF, HALF], interpolation='bilinear')
    tint = np.zeros(lab.shape + (4,))
    tint[np.isin(lab, LAT_VENT)] = matplotlib.colors.to_rgba('#4FC3F7', 0.45)
    ax.imshow(tint, origin='lower', extent=[-HALF, HALF, -HALF, HALF])

    for surf, colour in [('white', '#FFD600'), ('pial', '#FF3D00')]:
        for h in ('lh', 'rh'):
            verts, faces = nib.freesurfer.read_geometry(str(d / 'surf' / f'{h}.{surf}'))
            seg = plane_segments(verts, faces, axis, level, keep)
            seg = seg - centre[keep][None, None, :]
            ax.add_collection(LineCollection(seg, colors=colour, linewidths=0.5))
    ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=FONT_AXIS_TITLE, pad=3)


PANELS = [
    ('coronal, DLPFC', 1, lambda b, v: b[:, 1].max() - 40.0),
    ('coronal, ventricles', 1, lambda b, v: v[:, 1].mean()),
    ('axial, ventricles', 2, lambda b, v: v[:, 2].mean()),
]


def build() -> Path:
    fs = pd.read_csv(FREESURFER_MORPH_PATH, dtype={'subject_id': str})
    m = pd.read_csv(REPO_ROOT / 'data' / 'master_subject_data.csv',
                    dtype={'subject_id': str})
    age = pd.to_numeric(m.set_index('subject_id')['age'], errors='coerce')
    rows = {**FLAGGED, **references(fs, age)}
    info = fs.set_index('subject_id')

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(WIDTH_2COL, WIDTH_2COL * n / 3 * 1.02),
                             facecolor='black')
    fig.subplots_adjust(left=0.20, right=0.995, top=0.965, bottom=0.005,
                        wspace=0.02, hspace=0.04)
    for r, (sid, why) in enumerate(rows.items()):
        i = info.loc[sid]
        for c, (title, axis, level_of) in enumerate(PANELS):
            draw(axes[r, c], sid, why, i, axis, level_of, title if r == 0 else None)
            if r == 0:
                axes[r, c].title.set_color('white')
        colour = '#FFB74D' if sid in FLAGGED else '#81C784'
        axes[r, 0].text(-0.04, 0.5,
                        f'sub-{sid}\n{why}\nage {age[sid]:.0f}\n'
                        f'thickness {i.lh_mean_thickness:.2f} mm\n'
                        f'defects {i.surface_holes:.0f}\n'
                        f'ventricles {i.ventricle_choroid_vol / 1e3:.0f}k mm$^3$',
                        transform=axes[r, 0].transAxes, ha='right', va='center',
                        fontsize=FONT_TICK - 0.5, color=colour, linespacing=1.3)
        print(f'  {sid}: {why}')

    fig.text(0.005, 0.995, 'white surface = yellow   pial = red   lateral ventricles = blue',
             fontsize=FONT_TICK - 0.5, color='white', va='top')
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / 'qc_freesurfer_flagged.png'
    fig.savefig(out, dpi=300, facecolor='black')
    plt.close(fig)
    print(f'wrote {out}')
    return out


if __name__ == '__main__':
    build()
    sys.exit(0)
