# Archived data artifacts

Superseded files kept for provenance. Nothing here is read by current code.

## `efield_roi_summary_dissertation_n35_peak-to-peak.csv`

The E-field extract as it stood for the dissertation defence: 35 of the 39
dissertation subjects (the rest had no usable anatomical at the time).

**Its values are exactly 2x the current ones.** It was produced when the
montage was simulated at 1500/500 uA, which is the *peak-to-peak* amplitude;
the pipeline now simulates 750/250 uA peak, per SimNIBS guidance that the
field should be computed at peak current. Because the quasi-static solve is
linear in current, this is a pure scale factor -- every correlation and
p-value is identical, only the absolute V/m differ.

It also predates the `t1_only` flag, so it cannot support the exclusion the
current analyses apply.

Current data: `data/efield_roi_summary.csv` (66 subjects), pointed to by
`config.EFIELD_CSV_PATH`. Regenerate with `scripts/extract_efield_roi.py`
in `DVS-Lab/tacs_bandit_simnibs`.
