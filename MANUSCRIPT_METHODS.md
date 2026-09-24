# Methods and Limitations — draft text

Draft manuscript prose for the analysis decisions made during the September
2026 audit. Every number here is produced by a script in `code/`; the source is
named in square brackets so a reviewer's question can be answered by re-running
one command rather than by searching. **If a number changes, re-run the named script and update this file** — it is
prose, so nothing regenerates it. `python code/check_manuscript_numbers.py`
re-verifies every load-bearing number here against the data and fails if one
has drifted or if a guarded sentence was rewritten; run it before circulating
a draft.

Companion documents: `ANALYSIS_HANDOFF.md` (full reasoning and the bug record),
`VARIABLES.md` (provenance for every variable).

---

## Methods

### Participants and recruitment

Sixty-six adults were registered. **Recruitment targeted two age bands rather
than sampling the age range uniformly**: in the primary analysis sample
(N = 61), 27 participants were under 40 and 30 were 55 or over, with 4 falling
in between. Ages ranged from 23 to 79 (median 53.3); 30 were female and 31
male. [`code/table1.py`, `code/qc_paper_variables.py`]

This design decision governs how age effects are reported throughout. A
correlation computed across a bimodal distribution is arithmetically close to a
two-group contrast, so every age effect is reported as a **between-band
standardised difference alongside the correlation**, and within-band gradients
are reported where the data can speak to them. Band edges are 40 and 55
(`AGE_BAND_YOUNG_MAX`, `AGE_BAND_OLD_MIN` in `code/config.py`); participants
between them are excluded from the contrast.

### Task and stimulation

Participants completed a two-armed probabilistic reversal-learning task across
8 runs (~70–80 trials each) while receiving transcranial alternating current
stimulation and EEG. Runs 1 and 5 were baseline, 4 and 8 post-stimulation, and
2–3 and 6–7 the two stimulation blocks. Counterbalance assignment determined
which block delivered active stimulation and which delivered sham.

**All participants were stimulated at 6.0 Hz**, verified across all 237
stimulation runs [`code/stim_verification.py`]. No participant received an
individualised frequency.

Counterbalance was verified against the EEG recordings for 52 participants and
agreed with the REDCap record in 51 of 52. For six participants with no usable
EEG, counterbalance rests on the REDCap record alone.

### Exclusions and analysis samples

Preregistered exclusions were applied at the run level (missed trials > 20%,
side bias > 95%, stimulus bias > 95%, responses < 200 ms, feedback invariance),
plus three runs excluded for documented experimenter error. Of 66 registered
participants, 64 had behavioural files, 63 survived cleaning, 61 formed the H1
sample and 57 the H2 sample.

Rather than a single "complete data" sample — whose intersection is 35, below
the N of the earlier defended analysis, and which would drop participants from
preregistered tests for reasons unrelated to those tests — **each question uses
the largest sample with the data it requires**, defined once in
`code/samples.py`:

| sample | N | rule |
|---|---|---|
| H1 | 61 | passed H1 exclusions (sham-session behaviour); has age |
| H1_RL | 57 | H1 with a hierarchical RL fit (requires both sessions) |
| H2 | 57 | passed exclusions in both sessions |
| Dose | 59 | FLAIR-based head model; has age |
| Dose_H2 | 50 | H2 ∩ Dose |

Each analysis asserts that its working set *is* the named sample, so a filter
that drifts halts the analysis rather than silently changing N.

### Behavioural measures

Win-stay and lose-shift probabilities, accuracy and win rate were computed per
participant and condition.

**Trials-to-criterion** is the mean number of trials after a reversal until
three consecutive correct choices, within a 15-trial post-reversal window.
Reversals on which criterion was never reached return no value, so averaging
only solved reversals conditions on success and biases the mean downward for
participants who solve fewer — **survivorship bias**. The primary measure
therefore fills unsolved reversals with the number of trials actually available
(a conservative lower bound: the participant had at least that many trials and
did not reach criterion). The proportion of reversals solved is retained as a
separate measure. Criterion was reached on 84.2% of 939 reversals.
[`code/reversal_analysis.py`, switch `TTC_CENSORING`]

**Response time and choice dynamics** were derived from trial-level records
that had not previously been analysed: mean RT, RT variability, post-error and
post-loss slowing (within run), switch rate, lapse rate, perseverative errors
(first five trials after a reversal) and asymptotic accuracy (final five trials
before one). RTs outside 200–1990 ms were excluded as anticipations and
deadline misses. [`code/derived_behaviour.py`]

**Blinding** was assessed from the post-run question asking whether
participants believed they had been stimulated, collapsed to one response per
run.

### Cognitive measures

The cognitive composite averages three domain scores — executive function
(Flanker, Set Shifting, Running Dots, following the unity/diversity model),
memory (HVLT) and processing speed (Salthouse letter and pattern comparison) —
each computed from z-scores. **The same measures contribute for every
participant.** An earlier composite averaged whatever tests each participant
had completed; because several instruments were administered only from about
age 56, the number of contributing tests correlated with age at r = +.90,
making the composite partly an index of age itself. [`code/cognitive_merge.py`,
switch `COG_COMPOSITE`]

Education was coalesced from three sources (RF1 record, island screener,
TabCAT) in that order of priority, arbitrated against the highest level
completed where sources disagreed.

Several questionnaires were excluded after inspection: BPAQ (mapped to columns
empty at source — never administered), CTQ (total was a calculated field
returning the missing code for all records), Mach-IV (the recorded total is the
value an empty form produces for 55 of 66), and SOGS (calculated total returned
zero for all records in the source project). BBS requires a numeric export
that was unavailable; the labels export blanks intermediate responses. These
are retained as empty columns marked unavailable rather than silently dropped.

### Computational modelling

Choice behaviour was modelled with Rescorla-Wagner learning and a softmax
choice rule. **Hierarchical Bayesian estimates are primary**; maximum-likelihood
estimates are reported alongside for every test. Under maximum likelihood the
learning rate lands exactly on a bound of the search space (0.001 or 0.999) for
17 of 61 sham fits and 19 of 57 active fits, and the inverse temperature
reaches its ceiling in one fit per condition; the hierarchical fit places no
value on a bound. [switch `RL_ESTIMATES`; `code/compare_rl_estimates.py`]

Hierarchical models were fitted in NumPyro with NUTS (4 chains, 1000 warmup and
2000 sampling iterations each, target acceptance 0.95), using a non-centred
parameterisation. Parameter recovery was verified before fitting, and prior
sensitivity was checked.

Maximum-likelihood fits use a deterministic 40 × 40 grid search over the full
parameter bounds followed by L-BFGS-B from the five best grid points plus
seeded random restarts. Unseeded restarts previously produced different
estimates on each rebuild for several participants.

**Model adequacy** was assessed by posterior predictive checks simulating from
the posterior with counterfactual reward schedules. The single-rate model
reproduces accuracy (94.7% of participant-conditions inside the 95% predictive
interval), win-stay (95.6%) and trials-to-criterion (93.0%), but reproduces
lose-shift poorly (60.5% inside, r = .49 between observed and predicted). A
dual-learning-rate model improves this (71.1%, r = .63) without resolving it.
Split-half reliability of lose-shift is .82, so the shortfall is not
measurement noise. Leave-one-run-out PSIS-LOO was computed but is not
interpretable here — each run carries its own parameters, so 87–104 of 222 runs
exceed a Pareto-k of 0.7 and p_loo exceeds the number of observations.
[`code/ppc_within.py`, `code/compare_rl_models.py`]

### EEG

Recordings were made with a Starstim system at 500 Hz, with no acquisition
filtering. Four of eight channels recorded EEG (F4, P4, P3 and an external
channel); the remaining four were stimulation electrodes.

**Theta measure.** The primary EEG measure is the 95th percentile of the 4–8 Hz
Hilbert envelope, normalised by each run's own mean and averaged over
non-stimulation runs. It indexes how **bursty** a participant's theta power is
relative to their own average; it is not locked to feedback or any other event,
and is reported as a global rather than a regional measure because the three
recording channels correlate at median r = .998. Every run carries a
**phase-randomised surrogate control** — a signal with the same power spectrum
and its temporal structure destroyed — scored by the identical pipeline. Real
values exceed surrogates (260.6% vs 191.9%; paired t(118) = 9.24, p = 1 × 10⁻¹⁵;
positive in 92% of runs), and surrogate values correlate with real values at
only r = −.08, establishing that between-participant differences reflect genuine
bursting rather than spectral shape. Run-to-run reliability is r = .60
(Spearman-Brown .75). [`code/eeg_theta.py`]

**A stationary instrumental artifact is present in all recordings**: a harmonic
series with a fundamental at 9.767 Hz and components at 2×, 3×, 4× and 6×,
showing no measurable frequency drift within a recording, identical across
participants, and present on the non-scalp external channel. It is attenuated
by 33 dB by the 4–8 Hz filter and does not affect the theta measure. It does
lie within the alpha band.

**Individualised theta frequency** was estimated as individual alpha frequency
from posterior channels minus a fixed 5 Hz offset, bounded to 4–8 Hz, averaged
over all four non-stimulation runs. Alpha peaks were obtained by **fitting a
spectral model** (`specparam`: an aperiodic background plus Gaussian peaks with
a minimum bandwidth of 1 Hz) rather than by taking the largest value in the
band. This distinction is necessary rather than cosmetic: a largest-value rule
returns the instrumental artifact, which is narrower than any physiological
rhythm but larger than the alpha peak, for most participants. The fitted
estimates recover the expected decline of alpha frequency with age
(r = −.287, p = .032), which the largest-value estimates do not (r = −.195,
p = .17). [`code/individual_theta.py`]

### MRI and electric-field modelling

Individual head models were constructed with SimNIBS `charm` and electric
fields simulated with the finite-element solver for the delivered montage.
Field magnitude was averaged over a 20 mm spherical region centred on the F3
scalp electrode. Because the sphere is centred on the scalp, the gray matter it
contains varies between participants; results were confirmed against a fixed
anatomical parcel (rostral + caudal middle frontal), where every participant
contributes identical vertices, giving r = −.325 (p = .012) against r = −.300
for the sphere, with the two measures correlating at r = .952.

**Seven head models built without a FLAIR image are excluded** from all
field analyses. Their field estimates are systematically lower (28% in raw
means, 26% adjusted for age, p = .013), in the same direction as the
hypothesis. Controlling for age and sex they do not differ on any skull
measure, but do differ on CSF and on delivered field. Results including them
are reported as a sensitivity check. Note that FLAIR availability is perfectly
confounded with scan conversion status in this sample, so the exclusion cannot
be separated from whatever else distinguishes those acquisitions.

Cortical morphometry came from FreeSurfer 7.3.2 `recon-all`. Volumes were
normalised by **measured** intracranial volume rather than the FreeSurfer
estimate, because the estimate correlates with age (r = .479) while the
measured value does not (r = .001).

### Statistical approach

Preregistered tests are reported as specified. Where a null result is claimed,
**equivalence testing** (two one-sided tests) is reported against smallest
effect sizes of interest of dz = 0.3 and dz = 0.5.

Exploratory analyses are labelled as such and corrected with the
Benjamini-Hochberg false discovery rate across all tests in the relevant family,
with the number of uncorrected hits reported against the number expected by
chance. [`code/sweep_moderators.py`]

Where a claim concerns a mediated pathway differing between groups, the
**index of moderated mediation** is bootstrapped rather than comparing
subgroup correlations, since a difference in significance between subgroups is
not a test of a difference between subgroups.

Analytical choices with a defensible alternative are implemented as switches
(`COG_COMPOSITE`, `RL_ESTIMATES`, `TTC_CENSORING`) and results are reported
under both settings.

---

## Limitations

**Age was sampled in two bands, not continuously.** With 27 participants under
40, 30 over 55 and 4 in between, this design estimates a group contrast well
and a continuous age gradient poorly. Within-band tests have 32–36% power to
detect a correlation of .30 and require .50 for conventional power, so where
within-band gradients are absent we report that the design cannot resolve them
rather than that they do not exist. Age effects should be read as contrasts
between a younger and an older group.

**The sex-moderation analyses rest on small subgroups.** The finding that
age-related skull thickening and the age → scalp-cortex distance → delivered
field pathway are stronger in women than men (bootstrapped difference in
indirect effects, 95% CI excluding zero) is based on 29 women and 30 men, of
whom only 11 women are 55 or older. The recruitment structure means this is a
two-group comparison within each sex. A menopause-related account is consistent
with the pattern but **cannot be tested here**: women in this sample are
23–32 and 56–76, with two participants in between, so no onset can be
localised.

**EEG recordings are dominated by a common signal.** The three recording
channels correlate at median r = .998, and software re-referencing was applied
only to the participants recorded without an ear-clip reference. Regional
interpretation of the EEG measures is therefore not supported; they are treated
as global. The theta measure's validity rests on within-run normalisation and
the surrogate control rather than on recording quality.

**The instrumental artifact at 9.767 Hz lies within the alpha band.** It does
not affect the theta measure, and spectral-model fitting recovers alpha peaks
that behave as expected with age, but the individualised theta frequency
estimate remains noisier than published values (test-retest r = .63 against a
literature norm above .8). The associated null for frequency matching is
therefore evidence against a large effect, not against a subtle one.

**Within participants, active and sham are confounded with time in session.**
Counterbalancing breaks this at the group level only; any within-participant
interpretation of the condition effect carries practice and fatigue with it.

**Change scores are correlated with their own baseline by construction.** Any
variable tracking baseline performance will appear to moderate the stimulation
response. Where a moderator is reported, the effect holding the sham value
constant is reported alongside it.

**The exploratory moderator search can only detect large effects.** Across
1,290 tests at a median n of 57, surviving false-discovery-rate correction
requires a correlation above .51; power at that threshold is 3% for a true
correlation of .30. The absence of surviving moderators is strong evidence
against a large moderator and weak evidence against a moderate one.

**Counterbalance assignment for six participants rests on the study record
alone**, without EEG verification. The record agreed with the EEG in 51 of 52
participants where both exist. Removing these six does not change any H2
conclusion.

**Two questionnaires could not be scored.** BBS requires a numeric data export
that was unavailable, and SOGS requires item-level scoring not yet performed.
Both are marked unavailable rather than reported as missing data.
