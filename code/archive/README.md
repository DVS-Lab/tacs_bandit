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

---

## Notebooks that live only in git history (recorded 2026-09-24)

Three Jupyter autosaves in `code/.ipynb_checkpoints/` had no live counterpart
when that directory was added to `.gitignore`. They are **not in the working
tree** and are not files anyone should open by habit, but two of them are the
only copy of real work, so they are recorded here rather than lost to a
directory nobody looks in.

Retrieve either with:

```bash
git show e6639bd:code/.ipynb_checkpoints/<name>-checkpoint.ipynb > /tmp/<name>.ipynb
```

(`e6639bd` is the last commit in which they were tracked; `92274e0` is the
commit that untracked them, when `.ipynb_checkpoints/` was added to
`.gitignore`. Any commit at or before `e6639bd` also has them. They were first
added in `de0a41c` "first pass at analyses" and `75a31ea` "develop theta
reactivity pipeline" respectively.)

### `tacs_bandit_analyses_corrected-checkpoint.ipynb` (11 March 2026)

The **entire analysis before the refactor into modules**: 47 cells, 109 KB of
source, 10 cells carrying saved output. It runs data loading, trial
preprocessing, WSLS, Rescorla-Wagner fitting, demographics, survey
integration, reversal-locked analyses, a correlation heatmap, EEG quality
checks, hypothesis testing and exploratory analyses in one file.

It is **not** a stale copy of `code/main_analyses.ipynb` — the two share zero
code cells verbatim, and 31 of its cells appear nowhere else.

Kept out of the tree for two reasons. It is 4.2 MB, most of it March outputs.
And it is the lineage in which several of the bugs in ANALYSIS_HANDOFF.md
section 6 were found — the cognitive composite that averaged whatever tests a
subject happened to have, the unseeded maximum-likelihood restarts, the
trials-to-criterion mean over solved reversals only. That makes it valuable for
checking the bug record and actively misleading for anything else. Read it as a
historical document, never as a method.

### `feedback_theta_exploration-checkpoint.ipynb` (13 March 2026)

A 30-cell single-subject walkthrough (sub-11318, run 1): load the `.easy` file,
inspect the raw trace and power spectrum, load behaviour, compute feedback
event times, define a `theta_power` helper. No saved output.

Worth knowing about because it is almost certainly where the **unused epoching
machinery in `eeg_theta.py` came from** — `extract_epochs`,
`align_timestamps`, `EPOCH_PRE`/`EPOCH_POST`, `BASELINE_WIN`. A
feedback-locked theta measure was started here and never finished, and the
leftover constants later made `theta_p95` look like an event-locked
"reactivity" measure when it is nothing of the kind (see the theta entry in
section 9 of the handoff). The notebook is the record of the intent that was
abandoned.

The commit that introduced it is called **"develop theta reactivity
pipeline"** (`75a31ea`), which is where the word "reactivity" entered the
codebase. It survived into function names (`compute_theta_reactivity_run`),
column descriptions and draft manuscript text for six months, describing a
measure that was never built. The names were corrected on 2026-09-22; the
functions keep their old names so callers do not break, with the discrepancy
stated in the module docstring.

### `Feedback-Locked_Theta_Exploration-checkpoint.ipynb` — deleted

72 bytes, zero cells. A file Jupyter created and never wrote to. Deleted
2026-09-24; it remains in history at the commits above if anyone wants to
confirm it was empty.
