# Archived analysis scripts

Superseded code kept for provenance. Nothing here is imported by current
analyses, and nothing here should be re-run to produce a manuscript figure.

## `fig_thickness_distance.py`, `fig_mechanism.py` (archived 2026-08-20)

Two earlier versions of the geometry argument, superseded by
`fig_atrophy_vs_geometry.py` and `fig_skull_layers.py`.

They were retired for two reasons:

1. **They test the atrophy account with cortical thickness alone.** At the time
   that was the only morphometry available, because FreeSurfer covered 28 of 66
   subjects. With the full delivery the account can be tested against gray
   matter volume, ventricles, whole-head CSF and brain volume as well -- and it
   fails against all of them, which is a much stronger statement than thickness
   alone supports.

2. **Their stated exclusion rationale is wrong.** Both say the seven T1-only
   head models are excluded "because skull segmentation degrades without a
   FLAIR". QC (`fig_qc_skull.py`) showed those subjects do not differ on any
   skull measure once age and sex are controlled (total p = .58); they differ on
   CSF (p = .025) and delivered field (p = .018). The exclusion holds, the
   reason did not.

They also predate the finding that skull thickening with age is female-specific,
so their framing of the result as a general ageing effect is superseded.

`fig_mechanism.py` imports from `fig_thickness_distance.py`; both were moved
together and nothing else depended on either.
