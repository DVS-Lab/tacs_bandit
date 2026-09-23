# Figure captions — draft text

Draft captions for every figure in `data/figures/paper/`. Each entry names the
producing script and the numbers quoted, so a caption can be re-checked by
re-running one command. **Prose: nothing regenerates this file.** If a figure
is rebuilt and a number moves, edit here by hand.

Figure numbering below is a proposal, not a decision — the main-text set is
five figures, with the rest offered as supplementary.

Companions: `MANUSCRIPT_METHODS.md`, `ANALYSIS_HANDOFF.md`.

---

## Main text

### Figure 1 — Modelled field declines with age
`fig_efield_age_compact.png` · `python fig_efield_age.py --compact`

> **Figure 1. Older adults receive a weaker modelled electric field at
> identical stimulation current.** **(a)** Group-average difference in field
> magnitude on the fsaverage cortical surface, younger minus older age tertile
> (n = 20 per group, 23–30 y vs 59–79 y), with the F3 region of interest
> outlined. Warm colours indicate a stronger field in younger participants.
> **(b)** Mean field magnitude in the DLPFC region of interest against age
> (N = 59; r = −0.300, p = 0.021). Shading marks the 40–55 y recruitment gap,
> which contains 5 participants: age was sampled in two bands, so the
> association is better described as a group difference (Cohen's d = −0.77
> between bands, t(49) = 2.84, p = 0.007) than as a continuous gradient. Points
> are coloured by age. Dotted lines mark the tertile boundaries defining the
> groups in (a). Seven head models built without a FLAIR image are excluded
> throughout.

*Note.* Group averages rather than two example brains: across all 400 possible
young/old pairings the apparent difference ranges from −0.070 to +0.111 V/m
against a true group difference of +0.026, and in 24% of pairings the older
participant has the higher field. Consider stating this in the caption or
Methods if a reviewer is likely to ask why exemplars were not used.

### Figure 2 — The mechanism is geometry, not atrophy
`fig_atrophy_vs_geometry.png` · `python fig_atrophy_vs_geometry.py`

> **Figure 2. The age-related reduction in delivered field is explained by
> head geometry rather than by cortical atrophy.** **(a)** Axial slices with
> CSF segmentation overlaid for a younger (24 y, CSF 19% of intracranial
> volume) and an older participant (76 y, 25%), illustrating the atrophy that
> the analysis tests as a candidate mechanism. **(b)** Each anatomical measure
> positioned by its correlation with age (x) and with delivered field (y),
> coloured by whether it indexes atrophy (purple) or geometry (teal); shading
> marks the region where an association does not reach p < .05. Atrophy
> measures correlate strongly with age but weakly with delivered field;
> geometry measures — scalp-to-cortex distance and skull thickness — correlate
> with both, and most strongly with field. **(c)** Commonality partition of
> variance in field magnitude: geometry uniquely explains 0.70 (p < .001),
> atrophy uniquely 0.02 (p = 0.244), with 0.11 shared, for a full model
> R² = 0.83. N = 59.

### Figure 3 — Skull thickening with age is female-specific
`fig_skull_layers.png` · `python fig_skull_layers.py`

> **Figure 3. Age-related skull thickening beneath the stimulating electrode
> is present in women and absent in men.** **(a)** Tissue segmentation along
> the measurement ray dropped from the F3 electrode (triangle) through scalp,
> outer table, diploe, inner table and CSF. **(b)** Mean layer profile by sex
> and age tertile: total depth to CSF rises from 13.3 to 16.1 mm in women
> (n = 14 younger, 8 older) but is essentially flat in men (14.5 to 15.1 mm;
> n = 6, 12), and the increase is carried by the diploe. **(c)** Skull
> thickness against age by sex: women r = 0.67, p < .001 (n = 29); men
> r = −0.01, p = 0.96 (n = 30). The age × sex interaction is significant
> (p = 0.005) and survives adjustment for intracranial volume (p = 0.003),
> which differs between sexes (d = −1.33). **(d)** Age distribution by sex,
> with the tertile boundaries used in (b); the sampling gap between roughly 32
> and 56 years in women means (c) is a contrast between younger and older women
> rather than a gradient, and no onset age can be localised. The difference
> between sexes in the indirect age → scalp-cortex distance → field pathway is
> also reliable (bootstrapped difference in indirect effects, 95% CI
> [−0.0016, −0.0002]).

### Figure 4 — Stimulation had no effect on any preregistered outcome
`fig_h2_effect_sizes.png` · notebook §4.2

> **Figure 4. Active stimulation did not differ from sham on any preregistered
> outcome, and all effects are statistically equivalent to zero.** Standardised
> within-participant effect sizes (dz, active − sham) with 95% confidence
> intervals for the seven preregistered outcomes (N = 57). Shading marks the
> smallest effect size of interest (dz = ±0.5). Every interval lies inside that
> region; by two one-sided tests all seven outcomes are equivalent at dz = 0.5,
> and win-stay, lose-shift and trials-to-criterion are equivalent at the
> stricter dz = 0.3. Participants could not identify their condition better
> than chance (69% of active runs judged "stimulated" vs 64% of sham runs,
> dz = 0.12, p = 0.38).

### Figure 5 — Learning dynamics are unchanged by stimulation
`fig_reversal_locked_accuracy.png` · notebook §3.3

> **Figure 5. Reversal-locked accuracy does not differ between active and sham
> stimulation.** Probability of choosing the currently-rewarded option, aligned
> to each reversal, for sham (blue) and active (orange) sessions. Accuracy
> falls to approximately 0.30 at the reversal and recovers over the following
> ten trials, with the two conditions overlapping throughout. Trials were
> averaged within participant before averaging across participants; shading is
> the standard error across participants (60–61 per point, from 63
> participants and 939 reversals).

---

## Supplementary

### Figure S1 — Baseline cognition does not predict choice strategy
`fig_h1_cognition_wsls.png`

> **Figure S1.** Win-stay (left) and lose-shift (right) probability during the
> sham session against the global cognitive composite, coloured by age
> (N = 61). Neither association is significant (r = 0.047, p = 0.72; r = −0.163,
> p = 0.21). The preregistered models including age and education are likewise
> null.

### Figure S2 — Inverse temperature and lose-shifting
`fig_h1_beta_loseshift.png`

> **Figure S2.** Lose-shift probability against the softmax inverse temperature
> estimated from the sham session (hierarchical estimates; r = −0.163,
> p = 0.23, N = 57). Under maximum-likelihood estimates the same association
> reaches p = 0.089; the difference is carried by fits lying on a parameter
> boundary, and neither is treated as a finding.

### Figure S3 — Age does not moderate the stimulation response
`fig_h2_age_moderation.png`

> **Figure S3.** Change in accuracy (active − sham) against age (r = 0.188,
> p = 0.16, N = 57). Accuracy is the preregistered primary outcome and is
> plotted for that reason rather than because it is the strongest association
> in the family; no change score is reliably moderated by age.

### Figure S4 — Subject-level learning-rate effects
`fig_hb_subject_effects.png`

> **Figure S4.** **(a)** Distribution of participant-level learning-rate
> differences (active − sham) from the hierarchical model (M = +0.026,
> SD = 0.142, N = 57); the dashed line marks zero. **(b)** The same differences
> against age (r = −0.031, p = 0.82). Individual effects are distributed
> symmetrically about zero with no systematic shift and no age dependence.

### Figure S5 — Theta bursting and the stimulation response
`fig_theta_x_delta_ttc.png`

> **Figure S5.** Change in trials-to-criterion (active − sham) against baseline
> theta bursting, the 95th percentile of the 4–8 Hz envelope normalised to each
> run's mean (r = 0.150, p = 0.28, N = 53). The measure indexes how bursty
> theta power is relative to a participant's own average; it is not locked to
> task events and is not regionally specific (see Methods).

### Figure S6 — Individualised theta frequency
`fig_itf_distribution.png`

> **Figure S6.** Distribution of individual alpha frequency (M = 9.94 Hz,
> SD = 0.98, n = 59) and the derived individualised theta frequency (M = 5.04
> Hz), with distance from the delivered 6.0 Hz (M = 1.11 Hz). Alpha peaks were
> obtained by spectral-model fitting with a minimum peak bandwidth of 1 Hz; a
> largest-value rule returns a narrowband instrumental artifact at 9.767 Hz for
> most participants (see Methods). Distance from the stimulation frequency does
> not predict any outcome.

### Figure S7 — Scalp-to-cortex distance
`fig_distance_anatomy.png` · `python fig_distance_anatomy.py`

> **Figure S7.** Scalp-to-cortex distance beneath the F3 electrode, the single
> strongest predictor of delivered field magnitude (r = −0.899, p < 10⁻²¹,
> N = 59), and the mediator of 86.6% of the age–field association (percentile
> bootstrap CI on the indirect effect excludes zero; direct path p = 0.51).

### Figure S8 — Cortical age effects are not specific to the stimulation site
`fig_age_effect_maps.png`, `fig_global_vs_focal.png` · `python fig_age_effect_maps.py`

> **Figure S8.** Vertexwise correlation between age and cortical thickness
> across the surface, and the DLPFC parcels' position within the distribution
> of all parcels. Rostral and caudal middle frontal rank 9th and 2nd of 34
> parcels for age sensitivity (r = −0.58, −0.66), against a median parcel
> correlation of −0.46 — the stimulation site thins with age, but not
> distinctively so.

### Figure S9 — Skull segmentation quality
`fig_qc_skull.png` · `python fig_qc_skull.py`

> **Figure S9.** Skull and scalp segmentation at the F3 measurement ray for
> twelve head models — six built from T1 and FLAIR images (top) and six from T1
> alone (bottom) — spanning the age range, each labelled with participant age,
> sex and measured skull thickness. Controlling for age and sex, the two groups
> do not differ on any skull measure; they differ on CSF (p = 0.022) and on
> delivered field (p = 0.019), which is the basis for excluding T1-only models
> from field analyses.

*Note.* The diploe segmentation looks visibly patchier in several T1-only
panels, which is the a priori concern that motivated the exclusion. The
statistics do not support it — the groups do not differ on any skull measure —
so the caption should not imply the images demonstrate a skull-segmentation
failure. The exclusion rests on the CSF and field differences.

---

## Housekeeping

**Removed 2026-09-23:** `fig_mechanism` and `fig_thickness_distance` (both
`.png` and `.svg`). They were leftover output from two scripts retired on
2026-08-20 as superseded by Figures 2 and 3 — see `code/archive/README.md`.
The scripts remain archived and the images remain in git history.

**`fig_age_x_efield.png`** is the notebook's own version of Figure 1b, kept for
the executed notebook. Do not submit both it and `fig_efield_age_compact.png`.

**Every figure exists as both `.png` (300–400 dpi) and `.svg`.** Submit the SVG
where the journal accepts vector art.
