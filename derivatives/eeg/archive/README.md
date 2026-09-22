# Archived EEG outputs

Superseded outputs from an earlier EEG pipeline (March–April 2026). Nothing in
`code/` reads them, and none should be used for analysis. They are kept only
for provenance.

- `group_eeg_runs*.csv`, `group_eeg_summary*.csv` (v1, v3, v4, v5): run- and
  subject-level summaries from successive versions of the original EEG script.
  Replaced by `theta_run_metrics.csv` / `theta_subject_metrics.csv`
  (`code/run_theta_metrics.py`) and `individual_theta_frequency.csv`
  (`code/individual_theta.py`).
- `all_detected_peaks_v5.csv`, `group_eeg_summary_v5.csv`: **invalid.** The v5
  spectral-parameterisation fit passed log10 power to specparam, which expects
  linear power (`sm.fit(freqs, np.log10(psd))`), so the fitted peaks are
  meaningless (median "peak" 28 Hz).

Archived 2026-09-22 during the variable audit (Stage 2).
`group_stim_detection_v4.csv` stays in `derivatives/eeg/`: the stimulation
verification thresholds were set from it (see `code/stim_verification.py`).
