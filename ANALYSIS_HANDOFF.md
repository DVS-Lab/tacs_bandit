# tACS Bandit — Analysis Handoff

Written to be self-contained: everything needed to understand the pipeline, the
decisions behind it, and what remains uncertain. Current as of August 2026.

---

## 1. The study

Participants (ages ~22–79) performed a two-armed probabilistic reversal-learning
bandit task across 8 runs while receiving transcranial alternating current
stimulation (tACS) and EEG recording.

**Run structure**, identical for everyone:

| Runs | Type |
|---|---|
| 1, 5 | baseline (no stimulation) |
| 2, 3 | stimulation block 1 |
| 4, 8 | post-stimulation |
| 6, 7 | stimulation block 2 |

**Counterbalance** determines which block is active and which is sham:

- `A` — runs 2–3 active, runs 6–7 sham
- `B` — runs 2–3 sham, runs 6–7 active

Everyone was stimulated at **6.0 Hz**, verified across all 237 stimulation runs.
No participant received an individualized frequency, despite the acquisition
code containing machinery for it (`select_theta_frequency.py`, and
`run_theta_estimation.py` which imports a `theta_estimator` module that does not
exist — dead code).

**Design consequence that shapes the modelling.** Within any single subject,
active vs sham is perfectly confounded with early vs late in a ~2-hour session.
Counterbalancing breaks this at the group level only. Any model of the condition
effect should carry a block-order term or it will absorb practice and fatigue.

---

## 2. Data layout

`data/` is **read-only**. The single sanctioned exception is
`data/master_subject_data.csv`, which is generated output.

```
data/
  bandit/sub-XXXXX/          behavioral CSVs, one per run
  nic/raw/                   EEG + stimulation recordings (.easy/.info/.txt)
  TACSBandit-...2026-07-14_1714.csv    tACS REDCap export (current)
  TACSBandit-...2026-04-21_1210.csv    superseded (only 40 subjects)
  RF1SocialRewardProce*.csv            RF1 study REDCap
  TabCATStudyData*.csv                 cognitive battery
  efield_roi_summary.csv               modelled E-field, 66 subjects
  roi-analyses-final.csv               fMRI ROI, 231 subjects
  master_subject_data.csv              GENERATED — 66 x 145
derivatives/
  eeg/                       stim verification, theta, iTF
  rl_models/                 hierarchical fits
```

### Behavioral CSV columns worth knowing

`choice` (1/2, NaN if missed), `reward` (bool), `correct` (bool — chose the
currently-good option), `run`, `current_good`, `stim_guess`, `stim_confidence`,
`age`, `gender`.

**`stim_condition` in these files is unusable.** It comes from a subject-ID
parity fallback in `bandit_main.py` (line ~738): even subject number → active
first, odd → sham first. It matches the verified counterbalance for only about
half the sample. Nothing in the analysis pipeline reads it — `data_loading.py`
derives condition from `config.SUBJECT_INFO` — but it is a live trap for any new
code, and it *was* being read by `rl_models/data_loader.py` before that path was
replaced.

### NIC filename conventions

At least four, accumulated over the study:

```
sub-10369_Run 1.easy        (most common)
Bandit-10804_Run 2.easy
10559 TACS_Run 1.easy
sub-1432_A_run-1_Run 1.easy (pilot)
```

`nic_files.py` is the single place that knows these. Code globbing only the
first pattern silently reports "no EEG" for the others — this caused a real
data loss (see §6).

### Channel availability

The montage is F3, Fp1, FCz, FT7, F4, P4, P3, EXT — but usable EEG channels
depend on protocol:

| Subjects | Runs 1/4/5/8 | Runs 2/3/6/7 |
|---|---|---|
| Dissertation-era (39) | 3 (F4, P4, P3) | 3 |
| Post-defense (22) | **7–8** (incl. FCz) | 3–4 |

Four channels stimulate during stimulation runs. **FCz** — the canonical
frontal-midline theta site — is only available for the later subjects, and only
on non-stimulation runs. P3/P4 record for everyone, which is why the alpha-based
theta estimator covers the whole sample while the direct one does not.

---

## 3. Samples and how they are selected

`config.DISSERTATION_SUBJECTS` is a frozen list of the 39 subjects in the
defended dissertation. **Never modify it.** `config.SUBJECT_INFO` holds 66.

### From registration to analysis (full sample, as of 2026-09-21)

| stage | N | who drops out, and why |
|---|---|---|
| registered | 66 | |
| behavioural files load | 64 | `10961`, `11066`: no behavioural CSVs |
| after exclusions (`data_clean`) | 63 | `11492`: side bias on every run (97–100% one location) |
| **H1** | 61 | `11433`: no usable sham runs; `11440`: 7 of 8 runs fail stimulus bias |
| **H2** | 57 | subjects without usable runs in both sessions |

`11439` and `11472` were once listed here as lost (no behavioural files). Their
files have since been recovered and both are in H1 and H2. `10961` still has no
behavioural files, though its MRI is in the FreeSurfer delivery.

`load_all_subjects(sample=...)` loads 38 of 39 for `'dissertation'`, 64 of 66 for
`'all'` and 26 for `'new'`. The default stays `'dissertation'`, so old code
that passes no argument still produces the defended sample.

### Analysis samples

**Every analysis uses one of these by name, from `code/samples.py`.** Each is the
largest set with the data its question needs, rather than one "has everything"
sample. That intersection would be 35 (below the defended 39) and would drop
subjects from the preregistered tests for reasons unrelated to them.

| sample | N | rule | used for |
|---|---|---|---|
| **H1** | 61 | passed the preregistered H1 exclusions (sham-session behaviour); has age | H1.1, H1.1.1, H1.2; baseline behaviour; exploratory baseline sweep |
| **H2** | 57 | H2-eligible: passed the preregistered exclusions in both sessions | H2.1 paired tests and equivalence; H2.2 age moderation |
| **Dose** | 59 | FLAIR head model (T1-only excluded); has age | E-field x age; geometry, skull and atrophy analyses; Figs 1-3 |
| **Dose_H2** | 50 | H2 and Dose | E-field as a moderator of the stimulation response |

Moderator analyses (theta, iTF, fMRI, surveys) run on H1 or H2 intersected with
that measure and report their own N.

`python code/samples.py` checks that each sample is at its declared N and that
every subject in it has the variables its analyses need. It exits non-zero
otherwise. The figure scripts, `efield_results.py`, `skull_circularity.py` and
the paper notebook all assert that their working set *is* the named sample, so
a filter that drifts stops the script and names the subjects. A deliberate
change of sample means editing `expected_n` in `samples.py`, never a side
effect.

---

## 4. Module map

### Core

| File | Role |
|---|---|
| `config.py` | subject registry, paths, thresholds, condition maps, plotting constants |
| `data_loading.py` | trial CSVs → DataFrame, condition labels from counterbalance, WSLS columns |
| `exclusions.py` | pre-registered criteria → `data_h1`, `data_h2`, `data_clean`, `h2_eligible` |
| `cognitive_merge.py` | assembles `master_subject_data.csv` from REDCap/TabCAT/behavioral frames |
| `wsls.py`, `accuracy_analysis.py`, `rescorla_wagner.py` | behavioral measures |
| `reversal_analysis.py` | reversal identification, trials-to-criterion |
| `blinding_analysis.py` | d′ from `stim_guess` |

### Written during this work

| File | Role |
|---|---|
| `nic_files.py` | canonical NIC file discovery across all naming conventions |
| `stim_verification.py` | counterbalance validation from the 6 Hz artifact |
| `plot_stim_verification.py` | visual check of the above |
| `build_master_data.py` | end-to-end regeneration of the master CSV |
| `run_theta_metrics.py` | driver for `eeg_theta.run_theta_analysis` |
| `individual_theta.py` | IAF and individualized theta frequency |
| `build_results_paper_nb.py` | generates `results_paper.ipynb` |
| `rl_models/data_prep.py` | run-level dataset from the verified pipeline |
| `rl_models/models_within.py` | within-subject hierarchical RW |
| `rl_models/fit_within.py` | NUTS runner with diagnostics |
| `rl_models/run_within_fit.py` | fit driver |
| `rl_models/test_recovery_within.py` | δ_α recovery validation |
| `rl_models/test_prior_sensitivity.py` | prior robustness |

### Notebooks

- `results_figures_v2_withVS.ipynb` — **dissertation artifact, do not modify**
- `results_paper.ipynb` — the paper analyses; **generated**, edit
  `build_results_paper_nb.py` instead
- `individualized_freq_estimation_v4.ipynb` — exploratory per-subject frequency
  work on 4 subjects; contains a Klimesch implementation predating
  `individual_theta.py`
- `archive/tacs_bandit_eeg_v4.ipynb` — origin of the stim-detection logic

---

## 5. Key decisions and their reasoning

**Counterbalance is verified from EEG, not from records.** Active tACS produces
a sustained 6 Hz artifact; sham ramps up and down within the first ~45 s. So
active vs sham is distinguished by *sustained* mid-run power, not by presence of
power — both conditions show elevated early power by design. Detection is blind
to the config value; counterbalance is inferred from the pattern, then compared.

Outcome: **14 subjects corrected**, 13 new plus `10641`, which is inside the
defended sample and had its active/sham labels swapped. After correction: 59
confirmed, 1 unresolvable (`11773`, both stim runs delivered as sham — a known
error already in `STIM_EXCLUSIONS`), 6 without EEG.

`10810` is the instructive case: config said `A`, REDCap said `B`, EEG said `A`.
REDCap is a good but imperfect record — it agreed with the EEG on 12 of 14
corrections but was wrong here — so the EEG arbitrates and REDCap is fallback
only.

**Unverified counterbalance (5 subjects).** `11066` and `11316` have no EEG.
`11433`, `11542`, `11861` are set to `A` from REDCap by inference: every case
where config said `B`, REDCap said `A`, and EEG could adjudicate resolved to
`A` — 12 for 12. This is inference, not verification, and is labelled as such in
`config.py`.

**RL modelling is run-level, not subject-level.** Each run is a fresh bandit, so
Q-values reset per run and condition varies within subject. The previous
implementation concatenated all runs per subject (Q carried across boundaries)
and took one condition per subject by modal label — which for this design is
usually "baseline".

**Report natural-scale quantities for α.** On the logit scale, subjects whose α
sits near 0 or 1 have effectively unbounded coordinates. 38% of subjects sit at
a boundary, so σ_α and τ_α inflate with the prior. δ_α and the natural-scale
contrast are prior-robust; **τ_α's magnitude is not** and should be reported
qualitatively.

**Individualized theta is a moderator, not a delivered parameter.** Since
everyone got 6 Hz, the question is whether people whose endogenous theta sat
near 6 Hz responded better. Estimated via the Klimesch anchor (IAF − 5 Hz)
because alpha is detectable at P3/P4 for everyone. Run 1 only: run 5 follows the
first stimulation block, which is active for counterbalance A and sham for B, so
including it introduces an asymmetry between groups.

---

## 6. Bugs found, and what each cost

Every one had the same signature: **no error raised, data silently degraded.**

| Bug | Location | Effect |
|---|---|---|
| Seeded NumPy from `PRNGKey[0]`, which is always 0 | `rl_models/simulate.py` | every simulated agent byte-identical; all prior parameter-recovery evidence was on N copies of one subject |
| Globbed one NIC filename convention | `eeg_theta.find_eeg_run` | subjects recorded under other conventions produced no theta; defended theta coverage was 34, now 59 |
| `reversal_id` initialized from `np.nan` → float64, then assigned strings | `reversal_analysis.identify_reversals` | crashed under pandas 2.x; worked around by a 15KB inline reimplementation in the notebook |
| No high-pass before amplitude thresholding; no re-reference for no-earclip subjects | `individual_theta.py` (mine) | 15 subjects appeared to lack an alpha rhythm; actually 96% of their samples were being rejected. Coverage 44 → 55 of 59 |
| MLE restarts drawn from unseeded `np.random` | `rescorla_wagner.fit_rescorla_wagner` | each rebuild gave different learning rates for ~5 subjects, flipping between α bounds; the cognition × age moderation of α ranged p = .005–.125 across builds. Replaced by a 40×40 grid search over the full bounds, then L-BFGS-B from the 5 best grid points plus seeded random starts. Verified deterministic and never worse than an independent 100×100 grid (0 of 121 fits; 2026-09-21) |
| A fixed start grid (interim fix) missed flat optima | same | reproducible but not global: 11622 sham returned the chance corner (NLL 98.427) when α 0.999, β 0.15 fits better (98.160). Superseded by the grid search above |
| Age read only from the self-report field | `cognitive_merge.extract_demographics` | 11542 had no age and dropped from every age analysis. REDCap computes age as (Today's Date − birthdate)/365.25, exact for all 152 records; the same formula on the record's `Participant Date of Birth` recovers it (62.72, `age_source = dob_fallback`). FLAIR sample 58 → 59 |
| Composite averaged whatever tests a subject had | `cognitive_merge.compute_cognitive_composites` | Digit Span / BVMT / Trails were given only from ~age 56, so the memory composite's input count tracked age at r = +.90. `global_reduced` uses the same six measures for everyone; `COG_COMPOSITE` in config.py switches between them |
| Set Shifting parsed, then dropped at the merge | same | a 63/66-coverage attention measure was never used |
| BPAQ mapped from columns that are empty at source | `cognitive_merge.load_rf1_extended` | five all-NaN columns; the instrument was never administered. Removed |
| CTQ total is a REDCap calculated field returning 999 | `cognitive_merge.load_rf1_substance_mood` | read 66/66 complete while every value was the missing code 999; only 8 of ours answered any item. Removed |
| SOGS total is a broken REDCap calculation | `cognitive_merge.load_rf1_extended` | 0 for all 347 records in the RF1 project, although item answers are not all zero. Blanked, pending scoring from items |
| Mach-IV total returns 20 for an empty questionnaire | `cognitive_merge.load_rf1_exploratory` | 55 of 66 "scores" were the empty-form floor. 20 treated as missing; 11 real scores remain |
| BBS scored from a *labels* export | `cognitive_merge.score_bbs` | the labels export blanks every 2–6 response, so scores (-6/-1/0) used endpoint answers only. Scorer now refuses a labels export; needs a raw export |
| Education ranked TabCAT above the island screener | `cognitive_merge.build_subject_df` | TabCAT agrees with RF1 at r = .35, the island screener at .88. Reordered; 2 values changed; `education_source` added |
| Notebook sweep used legacy domain composites | `build_results_paper_nb.py` §6.2, `survey_exploration.py` | the age-confounded `attention/memory/speed_composite`. Swapped for `*_reduced` |

Also corrected: `dz` and TOST used the population SD (`ddof=0`), making the
equivalence test anti-conservative; H2 comparisons mixed samples across DVs;
`rl_models` applied no exclusions at all.

**Only the simulator bug is confined to synthetic data.** The others touched
real results.

---

## 7. Current results

### Reproduction against the defense

Running `results_paper.ipynb` with `SAMPLE='dissertation'` reproduces the
defended values for every subject except two, both expected:

- `10641` — counterbalance corrected, so its active/sham measures changed
- `10804` — its defended MLE fit sat at both parameter bounds with a likelihood
  *exactly* at chance (NLL 99.813 = 144·ln2). The current fit is better by 4.23
  and stable across refits.

### Stimulation effects: null, with equivalence established

*From `results_paper.ipynb` executed 2026-09-21 on the final master CSV
(deterministic RL fit; 63 subjects with behaviour, 57 H2-eligible).*

Every paired comparison null (|dz| ≤ 0.19, N = 57). At SESOI dz = 0.5, **all 7
DVs statistically equivalent** (p ≤ 0.010); at dz = 0.3, 3 of 7 (α, β, accuracy
and win rate inconclusive). α and β are hierarchical estimates (below); under
MLE they were also equivalent at dz = 0.3. The hierarchical
model agrees: δ_α = +0.227 [−0.350, +0.775] (N = 57, refit 2026-09-21). The
posterior mean moves with prior width (+0.23 to +0.48) but the interval spans
zero under every prior, so the conclusion is prior-robust.

Three methods with different assumptions converge on the same answer.

### Heterogeneity without an explanation

τ_α is credibly greater than zero — people differ in response. No candidate
moderator explains it: age, cognition, theta power, iTF distance, and striatal
reactivity are all null. The exploratory sweep found 7 of 102 baseline
correlations at p < .05 (5.1 expected by chance), **none surviving FDR**.

### The one notable positive

*Every number in this section through "FreeSurfer status" is produced by
`python code/efield_results.py` (sample: FLAIR head models with age, N = 59,
after 11542's age was recovered from their date of birth), except the
circularity table, which comes from `code/skull_circularity.py`.*

**Age × E-field r = −0.300, p = .021** (N = 59): older participants receive a
weaker modelled field at identical current. A dose-delivery finding, not a
rescue of the null — E-field does not predict behavioral change.

Held up across the sample expansion (it was r = −0.395, N = 34 at the
dissertation) and across the choice of dose metric (mean −.300, p95 −.289,
peak −.301, median −.312). Including the seven T1-only head models gives
r = −0.344, p = .005, N = 66; they are excluded from the primary analysis
because their fields are systematically lower -- 28% in raw means, 26%
age-adjusted (p = .013) -- in the same direction as the hypothesis.

**ROI coverage — checked and cleared.** The sphere is centred on the F3
*scalp* electrode, ~18 mm above cortex, so the gray matter falling inside it
ranges from 20 to 4757 elements across subjects, correlating with age at
r = −.42 and with mean |E| at r = +.85. Controlling for coverage zeroes the
age effect, which raised the worry that the metric was partly measuring how
much cortex happens to sit near the electrode.

It is not. Averaging instead over a fixed DK40 parcel (rostral + caudal middle
frontal), where **every subject contributes the identical 10,979 vertices**,
gives r = −0.325, p = .012 — the effect survives with sampling volume held
constant. The two measures agree at r = .952. Sphere stays primary (defined by
electrode position, so it cannot be circular); the parcel is the robustness
check. Columns `parcel_*` in the E-field CSV.

### Mechanism: it is geometry, and it is *not* cortical thinning

Scalp-to-cortex distance, measured from the charm surfaces to each subject's
F3 electrode, is available for all 66 subjects and is by far the strongest
predictor of delivered dose:

| | r with mean \|E\| |
|---|---|
| scalp-to-cortex distance | **−.899** |
| DLPFC cortical thickness | +.221 (ns) |

Distance mediates the age effect: indirect effect 95% CI [−.00078, −.00009],
**87% mediated**, direct path falls to p = .51. This holds for every
DLPFC-restricted distance variant (79–97% mediated) and fails only for the two
whole-hemisphere variants, which average in cortex nowhere near the electrode.

**Do not describe this as an atrophy effect.** Cortical thickness and
scalp-to-cortex distance are essentially unrelated here (r = −.215, p = .10),
thickness adds nothing to |E| once distance is in the model (p = .62), and
thickness does not explain the age→distance link (p = .84). Age drives both
independently. Whatever pushes cortex away from the skull with age — CSF
expansion, sulcal widening, skull change — it is not the thinning of cortex
itself.

Note also that the b-path (distance → field) is close to a physical identity:
a quasi-static field decays with distance from the source. The empirical
content is the a-path, age → distance, which is real but modest (r = +.293,
p = .024). The defensible claim is that the age effect on dose is *geometric*,
not that atrophy causes it.

Primary distance measure: `dist_pial_dlpfc_p1`. Robustness:
`dist_pial_dlpfc_min` (identical to `dist_pial_min`, confirming the nearest
cortex to F3 is DLPFC).

### What the geometry finding does and does not support

Age -> |E| is carried by the tissue between electrode and cortex, not by
atrophy. That much is solid, and is now tested properly rather than inferred:

- Every conventional atrophy measure tracks age hard (total GM -.78, ventricles
  +.67, cortical thickness -.64) and **none reaches the field** (all p > .05
  except whole-head CSF at -.34 and its near-complement, brain/ICV, at +.36
  -- the same signal, since brain and CSF together make up ICV).
- Commonality analysis on |E|: geometry uniquely explains **.70** (p < .001),
  atrophy uniquely **.02** (p = .24), shared .11, full model R2 = .83.
- Volumes are normalised by ICV counted from the segmentation, not by
  FreeSurfer eTIV. eTIV is derived from the Talairach determinant and rises
  with age here (r = +.48; the fitted slope over 40 years is +18% of mean
  eTIV), which is impossible; directly measured ICV is flat (r = +.001). Conclusions unchanged either way.

**Within the geometry, skull thickening with age is female-specific.** Women
r = +.67 (p < .001), men r = -.01 (p = .96), age x sex p = .005; the age main
effect vanishes once the interaction is included (p = .96). Older women's
skulls are 2.63 mm thicker than younger women's, against 0.36 mm in men. This
matches the hyperostosis frontalis interna literature, and F3 sits over frontal
bone. Caveat: only 8 women fall in the older tertile, though the imbalance runs
*against* the effect rather than producing it.

**The causal step is weaker than the headline numbers suggest.** |E| is
computed from charm's segmentation, so charm's skull is an input to the model
that produced the field. Measuring the same span from raw T1 intensity, with no
labels:

| | x age | x \|E\| |
|---|---|---|
| charm skull (an FEM input) | +.350 | **-.781** |
| T1 intensity span (external) | +.372 | **-.247 (p = .059)** |

The two agree at only r = +.42. The *age* effect replicates independently; the
*field* effect largely does not, and mediation falls from 90% to 19% (n.s.).
So the anatomical claim stands on its own and the dose claim does not. Report
this as an anatomical contributor to variation in *simulated* dose.

Reproduce with `python code/skull_circularity.py` (N = 59; bootstrap CI on the
indirect effect, fixed seed; writes `derivatives/skull_circularity.csv`).

**Do not interpret the outer/diploe/inner split.** QC images (`fig_qc_skull.py`)
show diploe segmented as scattered islands rather than a continuous stratum, in
FLAIR and T1-only subjects alike. Total skull is the trustworthy number.

**The T1-only exclusion is not about skull.** Controlling for age and sex the
seven differ on CSF (p = .022) and |E| (p = .019) but on no skull measure
(total p = .63). The exclusion holds; the stated rationale needed correcting.

Figures: `fig_efield_age.py --compact`, `fig_atrophy_vs_geometry.py`,
`fig_skull_layers.py`, `fig_qc_skull.py`.

### FreeSurfer status — complete

All 66 delivered across two batches (`tacs_bandit_freesurfer_20260814`, n=28;
`freesurfer_set2_20260814`, n=38), both FreeSurfer 7.3.2, all T1-only with no
FLAIR/T2pial flags — so the T1-only confound affecting the E-field does *not*
touch the morphometry. **No batch effect**: delivery does not predict DLPFC
thickness, mean thickness, cortical volume, eTIV, or surface holes once age is
controlled (all p > .28).

`code/freesurfer_morph.py` walks the whole `FREESURFER_ROOT` parent rather than
one batch folder, records `fs_delivery` and `fs_version` per subject, and
**raises on mixed FreeSurfer versions** — that would otherwise be a batch
effect perfectly confounded with batch membership and invisible after merging.
Outputs `data/freesurfer_morph.csv` and `data/freesurfer_parcels_long.csv`.

**Two independent pipelines agree.** charm's vertex-corresponded surfaces give
thickness as ‖pial − white‖ for all 66 without recon-all, correlating with
FreeSurfer at r = .892 whole-hemisphere and r = .820 in DLPFC. charm runs
~0.6 mm thicker in absolute terms (pial placement), so charm values should not
be compared to FreeSurfer norms — but every conclusion below replicates across
both, which is a stronger result than either alone:

| | charm | FreeSurfer |
|---|---|---|
| age × DLPFC thickness | −.664 | −.641 |
| thickness × distance | −.215 (ns) | −.180 (ns) |
| thickness × \|E\| | +.221 (ns) | +.207 (ns) |
| thickness explains age→distance | p = .84 | p = .94 |
| horse race: thickness \| distance | p = .62 | p = .44 |

Atrophy is **global, not clearly DLPFC-specific**: controlling for
whole-hemisphere thickness, the age→DLPFC-thickness effect is not significant
(b = −0.0010, p = .099, N = 59). Parcel by parcel, caudal middle frontal ranks
2nd of 34 left-hemisphere parcels for age-related thinning (r = −.66) and
rostral middle frontal 9th (r = −.58), against a median of −.46: DLPFC is among
the most affected regions, but not detectably more than the hemisphere as a
whole. (An earlier "6th of 34, r = −.546" could not be reproduced and is
superseded.)

Note `10961`, previously written off as lost, is present in set 2.

Figure: `code/fig_efield_age.py` (`--compact` for the two-panel version).
Pipeline: `DVS-Lab/tacs_bandit_simnibs` — `extract_efield_parcel.py`,
`extract_scalp_cortex_distance.py`, `extract_charm_thickness.py`, each merging
its columns into the single `data/efield_roi_summary.csv`.

---

## 8. Reproducing

```bash
cd code

# 1. verify counterbalance (~15 min)
python stim_verification.py --sample all
python plot_stim_verification.py

# 2. theta (~20 min)
python run_theta_metrics.py
python individual_theta.py --runs 1

# 3. master CSV
python build_master_data.py --sample all

# 4. hierarchical RL — recovery first, it gates the rest
python -m rl_models.test_recovery_within
python -m rl_models.run_within_fit --sample all
python -m rl_models.test_prior_sensitivity --sample all

# 5. paper notebook
python build_results_paper_nb.py --execute
python build_results_paper_nb.py --sample dissertation --execute   # reproduction check
```

Environment: `/opt/anaconda3/bin/python`. numpyro 0.20, jax 0.9, arviz 0.23,
specparam 2.0.0rc6. No MNE — EEG code is custom scipy.

InferenceData is pickled, not NetCDF (h5py/NumPy conflict in this environment).

---

## 9. Open items

**Decided 2026-09-22: hierarchical estimates are primary**
(`config.RL_ESTIMATES = 'hb'`; every analysis reads α/β through `rl()`). Under
MLE, α sits on a bound for 17 of 61 sham fits and 19 of 57 active fits, and β
reaches its ceiling of 50. The hierarchical fit has no value on a bound. It
needs both sessions, so H1 analyses of α/β run on 57 (sample `H1_RL`). MLE is
reported alongside for every test (`compare_rl_estimates.py`, notebook §7.5e).

**Three results depend on the estimator, and should be reported that way, not
as findings:**

| test | MLE | hierarchical |
|---|---|---|
| H1.1.1 β → lose-shifting | p = .056 (n = 58) | p = .26 (n = 54) |
| H2.2 age × Δα | r = −.26, p = .054 | r = −.03, p = .82 |
| H1.1.2 cognition × age → α | p = .18 | p = .044 (n = 54) |

The two MLE trends are carried by fits pinned at a bound. The H1.1.2 α effect is
one of three H1.1.2 moderation tests and would not survive correction for them.

**Not started.** Posterior predictive checks; the RW_dual comparison motivated
by the α pile-up; the moderated hierarchical model (the model already accepts a
moderator matrix, so this is a data change); `best_model` and the extended-RW
columns (17 columns, but only 3/39 populated at defense).

**Data that will not be recovered.** Behavioral CSVs for `10961` and `11066`
(`11439` and `11472` were once listed here and have since been recovered). EEG
for `11030`, `11066`, `11316`, `11433`, `11542`, `11861`.

**Counterbalance rests on REDCap for six subjects**, because they have no EEG to
verify it: `11030`, `11066`, `11316`, `11433`, `11542`, `11861`. REDCap agreed
with EEG for 51 of 52 subjects where both exist (the miss was `10810`), so it is
a sound fallback. Four of the six are in H2 (`11030`, `11316`, `11542`,
`11861`). Removing them changes no H2 conclusion: every change score stays null,
and age × Δα is p = .054 with or without them (2026-09-22). `11030` was
undocumented until the Stage 2 audit.

**iTF (exploratory, §7.6) has method limits worth stating or fixing.** The IAF
uses 2 s Welch windows, so 0.5 Hz resolution and 26 of 55 subjects at exactly
10.0 Hz. iTF = IAF − 5 is clipped to 4–8 Hz, which puts the 5 subjects with
IAF < 9 Hz at exactly 4.0. Four IAFs of 12–13 Hz may not be alpha.

**Recoverable, needs a REDCap export (Stage 4).** BBS needs a *raw* (numeric)
export of the tACS BBS fields; the labels export in use has lost every 2–6
response. Separately, 7 of its 14 bias items match no column label. SOGS can be
scored from item responses already on disk (64 of ours) once a scoring key is
agreed on. Both are kept as empty columns marked `unavailable` in the spec, so
they come back in place once scored.

**Verify in REDCap.** Education for `11885` is 2 years (RF1 and island agree;
TabCAT says 18), and for `10590` it's 19 (RF1) against 7 (TabCAT and island).

**Construct decision: EF vs attention.** Legacy `ef_composite` (Running Dots,
Flanker, Trails B−A) has the age confound (input count × age r = +.46; Trails
only from ~56), and correlates .91 with `attention_reduced`. The uniform
"attention" composite's three tasks — Flanker (inhibition), Set Shifting
(shifting), Running Dots (updating) — are the textbook components of executive
function (Miyake et al., 2000), so it may be better *named* EF. `ef_composite`
is still the preregistered secondary moderator in notebook §6.1.

**35 survey variables no analysis uses** (TEIQue, AQ, PANAS, IOS, Mach-IV,
Planfulness, present bias, gullibility, GTI, PNR, RF1 CRT). Coverage 61–66 for
most, so they could join the exploratory FDR sweep, or stay unanalysed by design.

**FreeSurfer QC of the four outliers: all pass** (`code/fig_qc_freesurfer.py`,
2026-09-22; two reference subjects chosen by rule for comparison). `11461` (76 y,
thinnest cortex, 41 defects): surfaces sit on the tissue boundaries; the thin
cortex is genuine atrophy (wide sulci, enlarged ventricles, widened Sylvian
fissure). `10866`: the ventricles really are ~2× the age median, and if anything
slightly under-segmented at the posterior horns. Surrounding cortex and sulci are
ordinary for age. **This disproportion may be worth routing through the lab's
incidental-findings procedure; it is not ours to interpret clinically.** `11472`
and `10661` (45, 38 defects): defects are counted before topology correction, the
final surfaces follow the boundaries, and thickness is typical. Caveat: three
slices per subject, not the whole surface.

**Untested assumption worth naming.** 38% of subjects have α ≈ 1, meaning they
replace their value estimate entirely with the last outcome — win-stay/lose-shift
with no integration. That may be real for a fast-reversing task, or it may mean
RW is the wrong model for a third of the sample. Nothing here has tested which.
