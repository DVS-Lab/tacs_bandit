# How every brain image was made

Written so that any panel can be explained, defended, or reproduced without
reading the plotting code. Current as of August 2026.

Every brain image in this project is one of **three kinds**. Knowing which kind
a panel is tells you almost everything about how it was made.

| Kind | What it is | Panels |
|---|---|---|
| **A. Surface map** | a value per vertex, painted on an inflated template brain | E-field maps, thickness maps |
| **B. Ray section** | a 2-D slice through the plane containing the electrode-to-cortex line | distance sections, skull layer sections, QC grid |
| **C. Landmark section** | an ordinary axial slice, but at matched anatomy across subjects | atrophy sections |

---

## The two source pipelines

Nothing is drawn from raw scans directly. Two pipelines run first, and every
image is downstream of one or both.

**SimNIBS `charm`** takes each participant's T1 (plus FLAIR where available)
and produces a head model in `~/Desktop/projects/tacs_bandit/simNIBS/m2m/m2m_sub-XXXXX/`:

- `T1.nii.gz` — the participant's scan. For our subjects this is byte-identical
  to the original (they were already 1 mm isotropic, so charm copied rather than
  resampled). Verified: max voxel difference 6e-05.
- `final_tissues.nii.gz` — a label per voxel: 1 WM, 2 GM, 3 CSF, 5 scalp,
  7 compact bone, 8 spongy bone (diploe). Key given in `final_tissues_LUT.txt`.
- `surfaces/lh.pial.gii`, `lh.white.gii`, `lh.central.gii` — cortical surfaces
  in the participant's own space, **vertex-corresponded** to each other.
- `surfaces/lh.sphere.reg.gii` — where each of that subject's vertices sits on
  the fsaverage sphere. This is what makes cross-subject averaging possible.
- `toMNI/` — nonlinear warps between subject space and MNI.

The FEM solve then produces `simulations/sub-XXXXX/..._TDCS_1_scalar.msh`, the
electric field on the head mesh.

**FreeSurfer `recon-all` 7.3.2** (run by the lab, delivered in two batches)
produces the conventional morphometry: `?h.thickness` in native surface space,
`?h.aparc.stats` and `aseg.stats`, and `?h.fsaverage.sphere.reg` — FreeSurfer's
own registration to fsaverage, independent of charm's.

---

## Kind A — surface maps on fsaverage

**The problem.** Every brain has a different number of vertices in a different
arrangement, so nothing can be averaged across people in native space. The
answer is to resample everyone onto one shared template mesh: **fsaverage**,
which has exactly **163,842 vertices per hemisphere**. Vertex 40,000 then means
the same anatomical location in every subject, and averaging becomes elementwise
arithmetic.

Note that fsaverage lives in **MNI305** space, which differs from the more
commonly quoted MNI152 by a small affine. That is immaterial for showing where
something is, but it means coordinates read off these surfaces are indicative
rather than exact.

### A1. E-field maps (`fig_efield_age.py`)

1. **Sample the field on cortex.** `to_fsaverage.py` runs SimNIBS's
   `middle_gm_interpolation` on the existing mesh — no re-solving. The field is
   read **halfway between the white and pial surfaces** (`depth=0.5`), the usual
   convention: the pial boundary inflates estimates through partial volume with
   CSF, the white boundary deflates them.
2. **Resample to fsaverage.** Same call, via `out_fsaverage`. Output is
   `fsavg_overlays/fsavg_sub-XXXXX/lh.*.magn`, one value per fsaverage vertex.
3. **Average.** `group_mean()` stacks subjects and takes the vertexwise mean.
   The difference map is `young_map - old_map`.
4. **Draw.** nilearn's `plot_surf_stat_map` on `fsaverage.infl_left` — the
   *inflated* surface, so sulcal depths are visible rather than hidden in folds.

**The ROI outline** (cyan) is a 20 mm sphere around F3 at MNI (−52, 45, 44).
Two things about it that look wrong but are not:

- Distances are computed on the **pial** surface, then drawn on the *inflated*
  one. The inflated surface has no meaningful metric, so measuring on it would
  give a mask of the wrong size and shape.
- Pial specifically, not white: the sphere is centred on the *scalp* electrode,
  ~18 mm above cortex, so it clips the gyral crowns. Measured to the white
  surface the nearest vertex is 21.6 mm away and a 20 mm sphere would contain
  nothing at all.

It is drawn with `plot_surf_contours`, not a second `plot_surf` call — the
latter re-renders and erases the field underneath.

### A2. Thickness maps (`fig_age_effect_maps.py`)

`recon-all` writes thickness in **native** surface space, where vertex counts
differ between subjects (146k, 124k, 142k in three of ours), so it cannot be
averaged as-is.

`lh.fsaverage.sphere.reg` gives each native vertex a position on the fsaverage
sphere. For each fsaverage vertex we find the nearest native vertex on that
sphere (after normalising both to unit vectors, so the nominal radius does not
matter) and take its thickness. This is what FreeSurfer's `mri_surf2surf` does;
doing it in Python means the figure does not depend on a local FreeSurfer
install matching the one that produced the data. Cached in
`data/cache/thickness_fsaverage.npz`.

Both panels of that figure then show a **vertexwise correlation with age**, not
a group difference. A tertile split discards two thirds of the sample at every
vertex; the correlation uses all of it. Thresholded at the critical r for
p < .05, **uncorrected** — these are descriptive maps of an effect established
in the ROI analyses, not a whole-brain search.

### The four constants that control how a brain fills its panel

A matplotlib 3-D axes inscribes its render in a cube, so the brain never fills
its rectangle and the axes reports an extent far larger than the visible ink.
Laying panels out against the rectangle is what leaves surface figures full of
dead space. These were measured directly off the rendered canvas by
thresholding non-white pixels at a range of zoom levels:

```
SURF_ZOOM   = 1.50    working point; past ~1.55 the occipital pole is clipped
SURF_FILL   = 0.953   brain width as a fraction of the rectangle width
SURF_ASPECT = 1.378   brain width / brain height
SURF_YOFF   = 0.0097  brain centre above rectangle centre, in rect heights
```

`brain_axes()` inverts them: you give it the box the *brain* should occupy and
it back-computes the oversized rectangle that puts it there. That is why titles,
colourbars and panel labels sit flush against the brains rather than floating.

---

## Kind B — ray sections

These show the tissue between the electrode and the cortex. All are built by
`oblique_section()` (`fig_distance_anatomy.py`) or `ray_aligned_section()`
(`fig_skull_layers.py`), which differ only in orientation.

**Why not a plain coronal slice.** The nearest cortical point sits up to 12 mm
anterior or posterior to F3. On a coronal slice through the electrode, that
point projects into empty space and the annotated distance line stops short of
the brain. Both functions therefore build an **oblique plane that contains the
measurement**:

1. `u` = unit vector from F3 to the nearest pial vertex.
2. Pick a second in-plane axis. `oblique_section` uses the superior direction
   projected into the plane, so the image looks anatomically normal.
   `ray_aligned_section` instead sets the vertical axis to `-u`, so the ray runs
   straight down the image and each tissue becomes a horizontal band that can be
   labelled.
3. Sample a regular grid on that plane, convert world → voxel with the inverse
   affine, and interpolate: `map_coordinates` with **order=1** (trilinear) for
   the T1, **order=0** (nearest) for the labels — interpolating labels would
   invent tissue classes that do not exist.
4. Sampling step is **0.4 mm** for the anatomy figures and **0.15 mm** for the
   layer figure, both finer than the 1 mm voxels so the oblique plane is not the
   limiting factor.

Both volumes are put through `nib.as_closest_canonical` first, so orientation is
consistent regardless of how each scan was stored.

**Endpoint positions are projected, never assumed.** An earlier version computed
them as `centre ± path/2`, which is only correct when the view is centred on the
ray midpoint. When the view was offset toward cortex to show more brain, that
put the electrode marker several millimetres inside the skull and dragged every
tissue label with it. They are now obtained by projecting the actual 3-D points
onto the section's axes.

**Layer labels come from the measured values, not from the picture.** In
`fig_skull_layers` panel a, the label depths are the subject's own median layer
thicknesses — the same numbers panel b plots. Reading run-lengths down the
centre column instead produced a second, spurious "inner table" label, because
that column happened to contain three runs of compact bone.

### The measurements these sections illustrate

The pictures are illustrations; the numbers come from
`extract_skull_layers.py`, which casts **50 rays** from F3 to the nearest DLPFC
pial vertices, samples the labels at **0.1 mm**, run-length encodes each ray,
and takes the median across rays. Because the traversal order is fixed
(scalp → outer table → diploe → inner table → CSF), the first run of compact
bone is the outer table and compact after the diploe is the inner table.

A per-ray assertion checks that the layer thicknesses reconstruct the ray length
exactly. **Medians are not additive**, so summing the median layers does not
equal the median path — that slack is reported separately (max 0.213 mm, 1.6%).

---

## Kind C — landmark sections

Used for the atrophy panel in `fig_atrophy_vs_geometry.py`.

**The trap this avoids.** The first version took, for each subject, the axial
slice containing the most ventricular CSF *in that subject*. That guarantees the
most flattering view of the ventricles in everyone, and puts the two panels at
different anatomical levels — 69% of brain height in one, 73% in the other. Part
of what the reader saw was where we cut.

Now `extract_slice_landmark.py` maps a grid of MNI points at **z = +18** — the
level of the lateral ventricular bodies — through each subject's own nonlinear
warp (`mni2subject_coords`) and records the median subject-space z. The figure
takes the axial slice there. Same anatomy in everyone, chosen independently of
the data.

The MNI plane maps to a slightly curved surface in subject space, spanning a
**median IQR of 4.5 mm**, so "same plane" is approximate — but it is an
approximation to the same anatomy, which per-subject selection is not.

CSF is tinted from `final_tissues` label 3. Participants shown are those whose
CSF fraction is closest to their age tertile's mean.

---

## Panel-by-panel index

| Figure | Panel | Kind | Source | Notes |
|---|---|---|---|---|
| `fig_efield_age_compact` | a | A | fsavg E-field overlays | younger − older, n=20/20; ROI outline from pial, drawn on inflated |
| | b | — | `efield_roi_summary.csv` | scatter |
| `fig_atrophy_vs_geometry` | a | C | T1 + `final_tissues` | axial at MNI z=+18 via per-subject warp |
| | b, c | — | merged CSV | scatter, variance partition |
| `fig_skull_layers` | a | B | T1 + `final_tissues` | ray vertical; labels from measured thicknesses |
| | b, c, d | — | merged CSV | composition, interaction, sample |
| `fig_qc_skull` | all | B | T1 + `final_tissues` | one panel per subject, thickness printed |
| `fig_age_effect_maps` | a | A | FreeSurfer thickness → fsaverage | vertexwise r with age |
| | b | A | fsavg E-field overlays | vertexwise r with age |
| `fig_distance_anatomy` | a, b | B | T1 + `final_tissues` | superior-oriented oblique |
| `fig_mechanism` | a, b | B | as above | same sections, panel-matched crop |

---

## What is reproducible where

**Inside the repo:** every scatter, every statistic, both surface-map figures
(the fsaverage overlays and the thickness cache are small enough to keep).

**Needs the SimNIBS project** (~86 GB, outside git, identifiable — see
`DVS-Lab/tacs_bandit_simnibs`): all ray and landmark sections, since they read
each participant's T1 and tissue labels.

**Needs a FreeSurfer delivery:** the thickness map, if the cache is rebuilt.

Column extraction scripts live in the SimNIBS repo and all merge into the single
canonical `data/efield_roi_summary.csv`:
`extract_efield_roi`, `extract_efield_parcel`, `extract_scalp_cortex_distance`,
`extract_charm_thickness`, `extract_skull_layers`, `extract_skull_t1`,
`extract_intracranial`, `extract_slice_landmark`.

---

## Limitations worth stating before anyone asks

**|E| is modelled, not measured.** Skull geometry is an *input* to the FEM, so a
strong skull-to-field relationship partly reports which input the model is most
sensitive to. Measured externally from T1 intensity with no labels, the same
skull span relates to |E| at only r = −.23 (p = .081) against charm's −.78.

**Diploe segmentation is not trustworthy at 1 mm.** The QC grid shows it as
scattered islands inside compact bone rather than a continuous stratum, in FLAIR
and T1-only subjects alike. Total skull has clean boundaries; the
outer/diploe/inner split should not be interpreted.

**Surface maps are uncorrected.** Thresholded at p < .05 per vertex, descriptive
only.

**eTIV is not used.** FreeSurfer derives it from the Talairach determinant
rather than measuring it, and it rises with age here (r = +.47, +12.5% over 40
years), which is not possible. Volumes are normalised by an ICV counted directly
from the segmentation, which is flat with age (r = −.011).
