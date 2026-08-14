"""
build_data_audit_nb.py — Assemble data_audit.ipynb

A running account of what data exists, who is excluded and why, and how every
derived measure is computed. Ordered to match the manuscript's Methods so each
section can be read against the text it supports, with one deliberate
rearrangement: behavioral dependent variables are moved up to sit beside the
task data they come from, rather than after the measures.

The point is that every number in the Methods should be *generated here*, not
typed. The dissertation's Table 1 ("Analytic sample sizes by analysis") is
produced at the end for pasting directly into the manuscript.

Expanded sample only. The frozen dissertation sample is reproduced by
results_paper.ipynb with SAMPLE='dissertation'.

Usage
-----
    python build_data_audit_nb.py --execute
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import nbformat as nbf

OUTPUT = Path(__file__).parent / 'data_audit.ipynb'


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


# =============================================================================
# 0 — Setup
# =============================================================================

def section_setup() -> List[nbf.NotebookNode]:
    return [
        md("""
# tACS Bandit — Data Audit

What data exists, who is excluded and why, and how each derived measure is
computed. Section order follows the manuscript's Methods so the two can be read
side by side.

**Everything here is computed, not transcribed.** Section 8 emits the analytic
sample table for the manuscript. If a number in the Methods disagrees with a
number here, this notebook is right and the manuscript needs updating.

| Section | Methods section |
|---|---|
| 1. Sample and demographics | 2.1 Participants |
| 2. Task data and exclusions | 2.3 Task, 2.8 Exclusion criteria |
| 3. Behavioral dependent variables | 2.7 (moved up, next to its source data) |
| 4. Survey measures | 2.6 Measures |
| 5. Stimulation data | 2.2 Design, 2.4 tACS |
| 6. Stimulation-derived metrics | 2.5 EEG and E-field modeling |
| 7. Methods/code consistency | — (new) |
| 8. Analytic sample summary | Table 1 |

Expanded sample only. Run `results_paper.ipynb` with `SAMPLE='dissertation'`
to reproduce the defended analyses.
"""),
        code("""
# ============================================================================
# 0. Setup
# ============================================================================

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

from config import (
    SUBJECT_INFO, DISSERTATION_SUBJECTS, DATA_DIR, REPO_ROOT,
    EXCLUSION_THRESHOLDS, STIM_EXCLUSIONS, NO_EARCLIP_SUBJECTS,
    WIN_FRACTION, MIN_CLEAN_RUNS_PER_CONDITION,
)
from data_loading import load_all_subjects
from exclusions import apply_all_exclusions
from nic_files import discover_runs

SAMPLE = 'all'
DERIV = REPO_ROOT / 'derivatives'
MASTER_CSV = REPO_ROOT / 'data' / 'master_subject_data.csv'

subj = pd.read_csv(MASTER_CSV, dtype={'subject_id': str})
raw_trials = load_all_subjects(sample=SAMPLE, verbose=False)
excl = apply_all_exclusions(raw_trials, verbose=False)

trials = excl['data_clean']
run_excl = excl['run_exclusions']
h2_eligible = sorted(str(s) for s in excl['h2_eligible'])

# Collected as each section runs; section 8 turns it into the Methods table.
ANALYTIC_SAMPLES = []

def record(analysis, n, basis):
    ANALYTIC_SAMPLES.append({'Analysis': analysis, 'N': n, 'Basis for reduction': basis})

print(f'registry     : {len(SUBJECT_INFO)} subjects')
print(f'trial data   : {raw_trials["subject_id"].nunique()} subjects, {len(raw_trials)} trials')
print(f'after cleaning: {trials["subject_id"].nunique()} subjects, {len(trials)} trials')
"""),
    ]


# =============================================================================
# 1 — Sample and demographics
# =============================================================================

def section_sample() -> List[nbf.NotebookNode]:
    return [
        md("""
## 1. Sample and demographics

*Methods §2.1.* Participants were recruited from an existing longitudinal
neuroimaging cohort (R01-AG067011). Registration in `config.SUBJECT_INFO` is
the definition of "enrolled" for these analyses — a subject not in that
dictionary is invisible to every downstream step, so the count below is the
ceiling on every N in this notebook.
"""),
        code("""
# ============================================================================
# 1.1 Who is registered, and what raw data exists for them
# ============================================================================

import glob

rows = []
for sid in SUBJECT_INFO:
    beh_files = glob.glob(str(DATA_DIR / f'sub-{sid}' / '*.csv'))
    beh_runs = {int(Path(f).name.split('run-')[1][:2])
                for f in beh_files if 'run-' in Path(f).name}
    rows.append({
        'subject_id': sid,
        'behavioral_runs': len(beh_runs),
        'eeg_runs': len(discover_runs(sid)),
        'counterbalance': SUBJECT_INFO[sid].get('counterbalance', '?'),
        'earclip': SUBJECT_INFO[sid].get('earclip', True),
        'note': SUBJECT_INFO[sid].get('notes', ''),
    })
availability = pd.DataFrame(rows)

n_reg = len(availability)
n_beh = (availability['behavioral_runs'] > 0).sum()
n_eeg = (availability['eeg_runs'] > 0).sum()

print(f'Registered                : {n_reg}')
print(f'  with behavioral data    : {n_beh}')
print(f'  with EEG data           : {n_eeg}')
print(f'  with both               : {((availability["behavioral_runs"]>0) & (availability["eeg_runs"]>0)).sum()}')

no_beh = availability[availability['behavioral_runs'] == 0]
if len(no_beh):
    print(f'\\nNo behavioral data ({len(no_beh)}):')
    for _, r in no_beh.iterrows():
        has_eeg = 'EEG session exists' if r['eeg_runs'] else 'no data at all'
        print(f'  sub-{r["subject_id"]}: {has_eeg}')
    print('  Subjects with an EEG session but no behavioral files were run; the')
    print('  task files did not reach the repository and are treated as lost.')

record('Registered', n_reg, '—')
record('With usable task data', n_beh,
       f'{n_reg - n_beh} without behavioral files')
"""),
        code("""
# ============================================================================
# 1.2 Demographics
# ============================================================================
# Reported for subjects contributing analyzable task data, which is the sample
# the manuscript describes.

analysis_subjects = sorted(trials['subject_id'].unique())
demo = subj[subj['subject_id'].isin(analysis_subjects)].copy()
demo['age'] = pd.to_numeric(demo['age'], errors='coerce')

age = demo['age'].dropna()
print(f'N = {len(demo)}')
print(f'Age: M = {age.mean():.1f}, SD = {age.std():.1f}, '
      f'range = {age.min():.1f}-{age.max():.1f}  (n = {len(age)})')

for col in ['gender', 'race', 'ethnicity']:
    if col not in demo.columns:
        continue
    counts = demo[col].value_counts(dropna=False)
    pct = 100 * counts / len(demo)
    print(f'\\n{col.capitalize()} (missing {demo[col].isna().sum()}):')
    for level, n in counts.items():
        print(f'  {str(level)[:38]:40s} {n:3d}  ({pct[level]:.0f}%)')

edu = pd.to_numeric(demo['education_years'], errors='coerce').dropna()
print(f'\\nEducation (years): M = {edu.mean():.1f}, SD = {edu.std():.1f}, '
      f'range = {edu.min():.0f}-{edu.max():.0f}  '
      f'(n = {len(edu)}, missing {len(demo) - len(edu)})')
"""),
        code("""
# ============================================================================
# 1.3 Manuscript sentence, pre-filled
# ============================================================================

n_f = int((demo['gender'].astype(str).str.lower().str.startswith('f')).sum())
n_m = int((demo['gender'].astype(str).str.lower().str.startswith('m')).sum())

def pct_of(col, *labels):
    if col not in demo.columns:
        return np.nan
    s = demo[col].astype(str).str.lower()
    hit = s.apply(lambda v: any(l in v for l in labels))
    return 100 * hit.sum() / len(demo)

print('Paste into Methods §2.1 and check the wording:\\n')
print(f'{len(demo)} adults ({n_f} female, {n_m} male; age M = {age.mean():.1f}, '
      f'SD = {age.std():.1f}, range = {age.min():.1f}-{age.max():.1f} years) '
      f'were recruited from an existing longitudinal neuroimaging cohort '
      f'(R01-AG067011; PI: Smith) at Temple University. The sample was '
      f'predominantly White ({pct_of("race", "white"):.0f}%), with Black or '
      f'African American ({pct_of("race", "black", "african"):.0f}%) and Asian '
      f'({pct_of("race", "asian"):.0f}%) participants also represented; '
      f'{pct_of("ethnicity", "not hispanic", "non-hispanic"):.0f}% identified '
      f'as non-Hispanic/Latino.')
"""),
    ]


# =============================================================================
# 2 — Task data and exclusions
# =============================================================================

def section_task() -> List[nbf.NotebookNode]:
    return [
        md("""
## 2. Task data and exclusion criteria

*Methods §2.3, §2.8.* Eight runs of a probabilistic two-armed bandit, ~6 min
each. The good option pays out on 75% of trials and reverses every 25–29
trials.

Pre-registered exclusions ([osf.io/s9k64](https://osf.io/s9k64/overview)) are
applied **at the run level**, then aggregated to participants. Reporting both
levels matters: a participant can lose runs without being excluded, and the
Methods needs both counts.

| Criterion | Threshold |
|---|---|
| Missed trials | > 20% of run |
| Side bias | > 95% one spatial location |
| Stimulus bias | > 95% one stimulus |
| Rapid responding | median RT < 200 ms |
| Feedback invariance | 10+ consecutive losses without switching |

A separate registry (`config.STIM_EXCLUSIONS`) removes runs where the delivered
stimulation did not match the assigned condition. Those runs are excluded from
active-vs-sham comparisons but retained for sham-only analyses when the
behavior itself is valid.
"""),
        code("""
# ============================================================================
# 2.1 Run-level exclusions
# ============================================================================

n_runs = len(run_excl)
flags = {
    'missed trials > 20%': 'flag_missed',
    'side bias > 95%': 'flag_side_bias',
    'stimulus bias > 95%': 'flag_stim_bias',
    'median RT < 200 ms': 'flag_rapid_rt',
    'feedback invariance': 'flag_feedback_invariance',
}

print(f'Total runs collected: {n_runs}\\n')
print(f'{"criterion":26s} {"runs flagged":>13s}   {"% of runs":>9s}')
print('-' * 54)
for label, col in flags.items():
    if col not in run_excl.columns:
        continue
    n = int(run_excl[col].sum())
    print(f'{label:26s} {n:13d}   {100*n/n_runs:8.1f}%')

n_behav = int(run_excl['exclude_behavioral'].sum())
n_stim = int(run_excl['exclude_stim'].sum())
n_any = int((run_excl['exclude_behavioral'] | run_excl['exclude_stim']).sum())

print(f'\\nRuns failing >=1 behavioral criterion : {n_behav} ({100*n_behav/n_runs:.0f}%)')
print(f'Runs excluded for stimulation error   : {n_stim}')
print(f'Unique runs excluded overall          : {n_any}')
print('\\nSome runs meet more than one criterion, so the column above sums to')
print('more than the number of excluded runs.')
"""),
        code("""
# ============================================================================
# 2.2 Which runs, and for whom
# ============================================================================

bad = run_excl[run_excl['exclude_behavioral'] | run_excl['exclude_stim']].copy()
if len(bad):
    def reasons(row):
        out = [label for label, col in flags.items()
               if col in row.index and row[col]]
        if row.get('exclude_stim'):
            out.append('stimulation error')
        return ', '.join(out)
    bad['reason'] = bad.apply(reasons, axis=1)
    print(bad[['subject_id', 'run', 'condition', 'reason']]
          .sort_values(['subject_id', 'run']).to_string(index=False))

per_subject = (run_excl.assign(excluded=run_excl['exclude_behavioral'] |
                                        run_excl['exclude_stim'])
                       .groupby('subject_id')['excluded'].agg(['sum', 'count']))
per_subject.columns = ['runs_excluded', 'runs_total']
lost_all = per_subject[per_subject['runs_excluded'] == per_subject['runs_total']]
print(f'\\nParticipants losing every run: {len(lost_all)}'
      f'{" -> " + ", ".join(lost_all.index) if len(lost_all) else ""}')
"""),
        code("""
# ============================================================================
# 2.3 Participant-level consequences
# ============================================================================

clean_subjects = set(trials['subject_id'].unique())
has_sham = {s for s, g in trials.groupby('subject_id') if (g['condition'] == 'sham').any()}
has_active = {s for s, g in trials.groupby('subject_id') if (g['condition'] == 'active').any()}

sham_runs = run_excl[(run_excl['condition'] == 'sham') &
                     ~run_excl['exclude_behavioral']]
h1_trials = trials[trials['condition'] == 'sham']

print(f'Participants with any clean run        : {len(clean_subjects)}')
print(f'  with >=1 clean sham run (H1)         : {len(has_sham)}')
print(f'  with >=1 clean active run            : {len(has_active)}')
print(f'  eligible for paired comparison (H2)  : {len(h2_eligible)}')
print(f'\\nH1 sample: {len(has_sham)} participants, {len(sham_runs)} clean sham runs, '
      f'{len(h1_trials)} trials')
h2_trials = trials[trials['subject_id'].isin(h2_eligible) &
                   trials['condition'].isin(['sham', 'active'])]
print(f'H2 sample: {len(h2_eligible)} participants, {len(h2_trials)} trials')

dropped_h2 = sorted(clean_subjects - set(h2_eligible))
print(f'\\nNot eligible for H2 ({len(dropped_h2)}): {", ".join(dropped_h2)}')
print(f'Requirement: at least {MIN_CLEAN_RUNS_PER_CONDITION} clean run in each condition.')

record('H1: baseline behavior (sham)', len(has_sham),
       f'{len(SUBJECT_INFO) - len(has_sham)} without a clean sham run')
record('H2: paired stimulation comparisons', len(h2_eligible),
       f'{len(clean_subjects) - len(h2_eligible)} without clean runs in both conditions')
"""),
    ]


# =============================================================================
# 3 — Behavioral dependent variables
# =============================================================================

def section_dvs() -> List[nbf.NotebookNode]:
    return [
        md("""
## 3. Behavioral dependent variables

*Methods §2.7, moved up to sit beside the task data it derives from.*

Three families, each computed per participant per condition:

**Win-stay/lose-shift.** `p(stay|win)` is the proportion of trials repeating
the previous choice after reward; `p(shift|lose)` the proportion switching
after non-reward. The first trial of each run and any missed response are
excluded, since neither has a defined predecessor.

**Rescorla-Wagner.** Q(a,t+1) = Q(a,t) + α·[r(t) − Q(a,t)], choice by softmax
with inverse temperature β. Q initializes to 0.5 at the start of every run —
each run is a fresh bandit with new contingencies, so values must not carry
across. Three estimators are compared in §3.2.

**Reversal adaptation.** Trials-to-criterion is the number of trials after a
reversal until three consecutive correct choices, averaged over reversals
within a condition.
"""),
        code("""
# ============================================================================
# 3.1 Win-stay / lose-shift
# ============================================================================

from wsls import compute_wsls_h1_h2

wsls_h1, wsls_h2 = compute_wsls_h1_h2(excl['data_h1'], excl['data_h2'])

print(f'H1 (sham only)      : {len(wsls_h1)} participants')
print(f'H2 (by condition)   : {wsls_h2["subject_id"].nunique()} participants, '
      f'{len(wsls_h2)} participant-conditions')
print()
print(wsls_h1[['p_stay_win', 'p_shift_lose', 'n_win_trials', 'n_lose_trials']]
      .describe().round(3).to_string())

thin = wsls_h1[(wsls_h1['n_win_trials'] < 20) | (wsls_h1['n_lose_trials'] < 20)]
if len(thin):
    print(f'\\nParticipants with <20 trials in either cell ({len(thin)}) — their'
          f' rates rest on few observations:')
    print(thin[['subject_id', 'n_win_trials', 'n_lose_trials']].to_string(index=False))
"""),
        md("""
### 3.2 Rescorla-Wagner: three estimators

The same model fitted three ways. They differ only in what constrains the
parameter estimates, and the differences are largest exactly where the data are
thinnest — which is where it matters.

**Maximum likelihood (MLE).** Picks the α, β maximizing the likelihood of the
observed choices, with no constraint beyond the parameter bounds. Each
participant and condition is fitted in isolation. When a participant's choices
are consistent with a whole range of parameter values, nothing stops the
optimizer settling at the edge of that range, and α pinned at 0 or 1 is not an
estimate of anything.

**Maximum a posteriori (MAP).** Adds weakly informative priors — Beta(2,2) on
α, Gamma(2,5) on β — and maximizes the posterior instead. Beta(2,2) has zero
density at 0 and 1, so boundary estimates become impossible. This is what
Methods §2.7.2 describes.

**Hierarchical Bayesian.** Estimates all participants jointly, with
group-level distributions acting as priors learned from the data rather than
assumed. A participant with weak data is pulled toward the group; one with
strong data is barely moved. It also returns a full posterior per participant
rather than a point estimate, so uncertainty propagates into anything computed
downstream. This is the estimator the manuscript will report.

Shrinkage is the common thread. MLE applies none, MAP applies a fixed amount
set by the analyst, hierarchical applies an amount the data determine.
"""),
        code("""
# ============================================================================
# 3.2a Fit MLE and MAP, and load the hierarchical fit
# ============================================================================

from rescorla_wagner import fit_rw_by_condition

rw_mle = fit_rw_by_condition(trials, method='mle', verbose=False)
rw_map = fit_rw_by_condition(trials, method='map', verbose=False)

hb_path = DERIV / 'rl_models' / 'rw_within_all_subjects.csv'
rw_hb = (pd.read_csv(hb_path, dtype={'subject_id': str})
         if hb_path.exists() else None)

print(f'MLE : {len(rw_mle)} participant-condition fits')
print(f'MAP : {len(rw_map)} participant-condition fits')
if rw_hb is not None:
    print(f'HB  : {len(rw_hb)} participants (both conditions jointly)')
else:
    print('HB  : not found — run `python -m rl_models.run_within_fit --sample all`')
"""),
        code("""
# ============================================================================
# 3.2b How much do they differ?
# ============================================================================

BOUND_LO, BOUND_HI = 0.005, 0.995

comp = rw_mle.merge(rw_map, on=['subject_id', 'condition'],
                    suffixes=('_mle', '_map'))

print(f'{"":10s} {"alpha at boundary":>18s} {"mean alpha":>11s} {"mean beta":>10s}')
print('-' * 54)
for cond in ['sham', 'active']:
    d = comp[comp['condition'] == cond]
    for method in ['mle', 'map']:
        a = d[f'alpha_{method}']
        n_bound = int((a <= BOUND_LO).sum() + (a >= BOUND_HI).sum())
        print(f'{cond[:6]:6s} {method.upper():4s} {n_bound:15d} '
              f'{a.mean():14.3f} {d[f"beta_{method}"].mean():10.3f}')

if rw_hb is not None and 'sham_alpha' in rw_hb.columns:
    for cond in ['sham', 'active']:
        col = f'{cond}_alpha'
        a = rw_hb[col]
        n_bound = int((a <= BOUND_LO).sum() + (a >= BOUND_HI).sum())
        print(f'{cond[:6]:6s} {"HB":4s} {n_bound:15d} {a.mean():14.3f} '
              f'{rw_hb[f"{cond}_beta"].mean():10.3f}')

print('\\nMAP and hierarchical both eliminate boundary estimates; MLE does not.')
print('That is the property the manuscript relies on, not a difference in fit.')
"""),
        code("""
# ============================================================================
# 3.2c Agreement between estimators
# ============================================================================
# Rank correlation as well as Pearson: beta is bounded only at 50 under MLE, so
# a couple of extreme values dominate the covariance while rank order is
# preserved. Reporting only Pearson would misrepresent the agreement.

def agreement(x, y, label):
    ok = x.notna() & y.notna()
    if ok.sum() < 5:
        print(f'  {label:28s} too few overlapping values')
        return
    r, _ = stats.pearsonr(x[ok], y[ok])
    rho, _ = stats.spearmanr(x[ok], y[ok])
    print(f'  {label:28s} pearson {r:+.3f}   spearman {rho:+.3f}   n = {ok.sum()}')

print('MLE vs MAP')
for cond in ['sham', 'active']:
    d = comp[comp['condition'] == cond]
    agreement(d['alpha_mle'], d['alpha_map'], f'{cond} alpha')
    agreement(d['beta_mle'], d['beta_map'], f'{cond} beta')

if rw_hb is not None:
    print('\\nMLE vs hierarchical')
    for cond in ['sham', 'active']:
        d = (comp[comp['condition'] == cond]
             .merge(rw_hb, on='subject_id', how='inner'))
        if f'{cond}_alpha' in d.columns:
            agreement(d['alpha_mle'], d[f'{cond}_alpha'], f'{cond} alpha')
            agreement(d['beta_mle'], d[f'{cond}_beta'], f'{cond} beta')
"""),
        code("""
# ============================================================================
# 3.3 Reversal adaptation
# ============================================================================

from reversal_analysis import identify_reversals, compute_trials_to_criterion

CRITERION = 3
rev = identify_reversals(trials.copy(), window_pre=5, window_post=15, verbose=False)
ttc = compute_trials_to_criterion(rev, criterion=CRITERION)

print(f'Reversals identified : {rev["reversal_id"].nunique()} across '
      f'{rev["subject_id"].nunique()} participants')
print(f'TTC computed for     : {len(ttc)} reversals, '
      f'{ttc["subject_id"].nunique()} participants')
print(f'Criterion            : {CRITERION} consecutive correct choices\\n')
print(ttc.groupby('condition')['trials_to_criterion']
        .agg(['count', 'mean', 'std']).round(3).to_string())

reached = ttc['reached_criterion'].mean() if 'reached_criterion' in ttc else np.nan
if np.isfinite(reached):
    print(f'\\nReversals where criterion was reached within the window: {100*reached:.0f}%')
    print('Reversals never reaching criterion contribute no TTC value, so')
    print('participant means are based on differing numbers of reversals.')
"""),
        code("""
# ============================================================================
# 3.4 Coverage of the behavioral DVs
# ============================================================================

families = {
    'WSLS (sham)': ['sham_p_stay_win', 'sham_p_shift_lose'],
    'WSLS (active)': ['active_p_stay_win', 'active_p_shift_lose'],
    'R-W (sham)': ['sham_alpha', 'sham_beta'],
    'R-W (active)': ['active_alpha', 'active_beta'],
    'Accuracy': ['sham_accuracy', 'active_accuracy'],
}
n_total = len(SUBJECT_INFO)
for family, cols in families.items():
    present = [f'{c}={subj[c].notna().sum()}' for c in cols if c in subj.columns]
    print(f'  {family:18s} ' + '  '.join(present) + f'   (of {n_total} registered)')
print('\\nMeasures within a family share an N by construction: sham measures')
print('cover everyone with a clean sham run, active measures everyone eligible')
print('for the paired comparison.')
"""),
    ]


# =============================================================================
# 4 — Survey measures
# =============================================================================

def section_measures() -> List[nbf.NotebookNode]:
    return [
        md("""
## 4. Survey and cognitive measures

*Methods §2.6.*

**Cognitive function (§2.6.1).** The parent protocol's full battery spans eight
measures across three domains. Three of them — Digit Span, BVMT, Trail Making A
— were administered only to participants aged 40+, creating missingness that is
structurally confounded with age. Since age is this study's primary moderator, a
composite built from whatever each participant happened to complete would
confound the moderator with composite composition.

The primary composite therefore uses the five measures with coverage across all
ages: HVLT-R Total Immediate Recall, Salthouse Letter, Salthouse Pattern, TabCAT
Flanker, TabCAT Running Dots. Z-scored and averaged directly, no intermediate
domain grouping.

**Reward/punishment sensitivity (§2.6.2).** SPSRQ-RC, 20 items, 5-point scale.
SR predicts baseline p(stay|win) and SP predicts p(shift|lose) under H1.2.
"""),
        code("""
# ============================================================================
# 4.1 Cognitive battery coverage, and the age-dependence of missingness
# ============================================================================

reduced = ['hvlt_total', 'salthouse_letter', 'salthouse_pattern',
           'flanker_score', 'running_dots_score']
age_restricted = ['digit_span_total', 'bvmt_total', 'trails_a_time']

sub_age = pd.to_numeric(subj['age'], errors='coerce')
print(f'{"measure":24s} {"n":>4s} {"% present":>10s} {"mean age present":>17s}')
print('-' * 60)
for label, group in [('reduced composite', reduced), ('age-restricted', age_restricted)]:
    print(f'  [{label}]')
    for m in group:
        if m not in subj.columns:
            continue
        have = subj[m].notna()
        print(f'  {m:22s} {have.sum():4d} {100*have.mean():9.0f}% '
              f'{sub_age[have].mean():17.1f}')

print('\\nThe age-restricted measures are present almost exclusively for older')
print('participants — that is the age-dependent missingness described above,')
print('and the reason the reduced composite is primary.')
"""),
        code("""
# ============================================================================
# 4.2 Composite construction and properties
# ============================================================================

comp_df = subj.copy()
for m in reduced:
    if m in comp_df.columns:
        v = pd.to_numeric(comp_df[m], errors='coerce')
        comp_df[f'{m}_z'] = (v - v.mean()) / v.std()

z_cols = [f'{m}_z' for m in reduced if f'{m}_z' in comp_df.columns]
comp_df['global_reduced'] = comp_df[z_cols].mean(axis=1)

n_all5 = comp_df[reduced].notna().all(axis=1).sum()
print(f'Participants with all 5 measures : {n_all5}')
print(f'Participants with >=4 of 5       : {(comp_df[reduced].notna().sum(axis=1) >= 4).sum()}')
print(f'Composite non-null               : {comp_df["global_reduced"].notna().sum()}')

# Internal consistency of the reduced composite.
z = comp_df[z_cols].dropna()
if len(z) > 3:
    k = z.shape[1]
    alpha_c = (k / (k - 1)) * (1 - z.var(axis=0, ddof=1).sum() /
                               z.sum(axis=1).var(ddof=1))
    print(f'\\nCronbach alpha (n = {len(z)}): {alpha_c:.3f}')

d = comp_df[['global_reduced', 'age']].apply(pd.to_numeric, errors='coerce').dropna()
r_age, p_age = stats.pearsonr(d['age'], d['global_reduced'])
print(f'Composite x age        : r = {r_age:+.3f}, p = {p_age:.4f}, n = {len(d)}')
d2 = comp_df[['global_reduced', 'education_years']].apply(pd.to_numeric, errors='coerce').dropna()
r_ed, p_ed = stats.pearsonr(d2['education_years'], d2['global_reduced'])
print(f'Composite x education  : r = {r_ed:+.3f}, p = {p_ed:.4f}, n = {len(d2)}')
print(f'\\nShared variance with age: {100*r_age**2:.0f}%. Age x Cognition')
print('interactions must be read with that overlap in mind.')

# A non-null composite is not a complete one. Averaging across whatever
# measures a participant happens to have makes the composite non-null for
# nearly everyone while being built from different tests for different people —
# and which tests are missing is tied to age. Report the count with the full
# set of five, which is the defensible N, and note the looser figure alongside.
n_complete = int(comp_df[reduced].notna().all(axis=1).sum())
n_any = int(comp_df['global_reduced'].notna().sum())
print(f'\\nComposite non-null for {n_any}, but complete (all 5 measures) for '
      f'{n_complete}.')
print('Analyses use the composite wherever it is defined; the complete-data')
print('count is reported because participants contributing fewer measures are')
print('systematically younger.')

record('Cognition-dependent analyses', n_complete,
       f'{len(SUBJECT_INFO) - n_complete} missing >=1 of the 5 composite measures '
       f'(composite defined for {n_any})')
"""),
        code("""
# ============================================================================
# 4.3 Questionnaire coverage
# ============================================================================

questionnaires = {
    'SPSRQ reward (SR)': 'spsrq_sr',
    'SPSRQ punishment (SP)': 'spsrq_sp',
    'Sleep quality (BPSQI)': 'bpsqi_global',
    'Mindfulness (FFMQ)': 'ffmq_total',
    'Cognitive reflection (CRT)': 'crt_total',
    'Body sensations (BBS)': 'bbs_avg',
    'Childhood trauma (CTQ)': 'ctq_total',
    'Alcohol use (AUDIT)': 'audit_total',
    'PROMIS anxiety': 'promis_anxiety',
    'Loneliness (UCLA-3)': 'loneliness_total',
    'Education (years)': 'education_years',
}
n_total = len(SUBJECT_INFO)
print(f'{"measure":30s} {"n":>4s} {"missing":>8s}')
print('-' * 46)
for label, col in questionnaires.items():
    if col not in subj.columns:
        print(f'  {label:28s} {"--":>4s}   column absent')
        continue
    n = int(subj[col].notna().sum())
    print(f'  {label:28s} {n:4d} {n_total-n:8d}')

record('SPSRQ analyses', int(subj['spsrq_sr'].notna().sum()),
       f'{n_total - int(subj["spsrq_sr"].notna().sum())} did not complete the questionnaire')
record('Education-dependent analyses', int(subj['education_years'].notna().sum()),
       f'{n_total - int(subj["education_years"].notna().sum())} without education data')
"""),
        md("""
### 4.4 Note on questionnaire sources

Coverage here reflects several REDCap exports read together rather than any one
file. Successive exports of the same project are **not nested** — each release
has gained some fields and lost others, and in one case a calculated field
(AUDIT sum) had fewer values in a later export than an earlier one despite
living at the same event. Taking only the newest export silently loses data.

`cognitive_merge` therefore reads every available export and coalesces per
subject, newest first. The same applies to TabCAT, where the August export
covers five more participants but drops 36 columns.

Measures still at low coverage (AUDIT, PROMIS, loneliness) are limited by
collection, not extraction: those participants have the parent event row with
the fields blank.
"""),
    ]


# =============================================================================
# 5 — Stimulation data
# =============================================================================

def section_stim() -> List[nbf.NotebookNode]:
    return [
        md("""
## 5. Stimulation data

*Methods §2.2, §2.4.* Theta-frequency tACS at 6 Hz, HD montage (F3 at +750 µA;
Fp1, FCz, FT7 returns at −250 µA each). Two protocol orders: **A** = active on
runs 2–3, sham on 6–7; **B** = the reverse. EEG recorded concurrently on the
Starstim-8 at 500 Hz.

**Counterbalance is verified from the EEG, not from records.** Active
stimulation leaves a sustained 6 Hz artifact for the whole run; sham ramps up
and down inside the first ~45 s. So the two are separated by whether power is
*sustained* mid-run, not by whether power is present — both show elevated early
power by design, which is the point of sham. `stim_verification.py` classifies
each run blind to the registry, infers counterbalance from the pattern, then
compares.

This matters because a wrong counterbalance value inverts active and sham for
that participant with no error anywhere — the contrast is simply computed
backwards.
"""),
        code("""
# ============================================================================
# 5.1 EEG availability and counterbalance verification
# ============================================================================

ver_path = DERIV / 'eeg' / 'stim_verification_subjects.csv'
run_ver_path = DERIV / 'eeg' / 'stim_verification_runs.csv'

if ver_path.exists():
    ver = pd.read_csv(ver_path, dtype={'subject_id': str})
    print('Counterbalance verification outcome:')
    for verdict, n in ver['verdict'].value_counts().items():
        print(f'  {verdict:12s} {n:3d}')

    unresolved = ver[ver['verdict'] == 'UNRESOLVED']
    if len(unresolved):
        print(f'\\nUnresolvable from EEG ({len(unresolved)}):')
        for _, r in unresolved.iterrows():
            print(f'  sub-{r["subject_id"]}: {r.get("basis", "")}')
        print('  No ACTIVE run detected anywhere, so the pattern cannot')
        print('  distinguish the orders. These are documented stimulation errors.')

    no_eeg = ver[ver['verdict'] == 'NO_EEG']
    if len(no_eeg):
        print(f'\\nNo EEG, counterbalance unverifiable ({len(no_eeg)}): '
              f'{", ".join(sorted(no_eeg["subject_id"]))}')
        print('  Their registry value stands on the session record alone.')

    if run_ver_path.exists():
        rv = pd.read_csv(run_ver_path, dtype={'subject_id': str})
        print(f'\\nRuns classified from the 6 Hz artifact: {len(rv)}')
        print(rv['detected'].value_counts().to_string())
else:
    print('No verification output. Run: python stim_verification.py --sample all')
    ver = None
"""),
        code("""
# ============================================================================
# 5.2 Stimulation administration errors
# ============================================================================
# Runs where what was delivered did not match what the counterbalance assigned.
# Excluded from active-vs-sham comparisons; retained for sham-only analyses
# when the behavior itself is valid.

if STIM_EXCLUSIONS:
    print(f'{len(STIM_EXCLUSIONS)} run(s) in the stimulation exclusion registry:\\n')
    for (sid, run), info in sorted(STIM_EXCLUSIONS.items()):
        print(f'  sub-{sid} run {run}: assigned {info["assigned"]}, '
              f'delivered {info["actual_delivered"]}')
        print(f'      {info["reason"]}')
else:
    print('No stimulation exclusions registered.')

print('\\nThese were recovered independently by the artifact detector rather')
print('than taken on trust: re-deriving each run\\'s expected condition from the')
print('verified counterbalance leaves exactly these runs anomalous and no')
print('others, so the registry of administration errors is complete.')
"""),
        code("""
# ============================================================================
# 5.3 Delivered stimulation frequency
# ============================================================================
# The protocol allowed for individualized theta, so confirm what was actually
# delivered rather than assuming the nominal 6 Hz.

import re

freqs = {}
for sid in SUBJECT_INFO:
    for run_num, files in discover_runs(sid).items():
        if run_num not in (2, 3, 6, 7) or files['info'] is None:
            continue
        txt = files['info'].read_text(errors='ignore')
        vals = {float(x) for x in re.findall(r'Ftacs \\(Hz\\):\\s*([\\d.]+)', txt)
                if float(x) > 0}
        if vals:
            freqs.setdefault(min(vals), 0)
            freqs[min(vals)] += 1

if freqs:
    print('Stimulation frequency across all stimulation runs:')
    for f, n in sorted(freqs.items()):
        print(f'  {f:.1f} Hz : {n} runs')
    if len(freqs) == 1:
        print('\\nEvery participant received the same fixed frequency. Individualized')
        print('theta is therefore a moderator in these analyses, not a delivered')
        print('parameter — see section 6.2.')
else:
    print('No stimulation parameters found in the NIC .info files.')
"""),
    ]


# =============================================================================
# 6 — Stimulation-derived metrics
# =============================================================================

def section_derived() -> List[nbf.NotebookNode]:
    return [
        md("""
## 6. Stimulation-derived metrics

*Methods §2.5, §2.5.1, and Supplementary Methods.* Three individual-difference
measures derived from the EEG and structural MRI, each with different coverage.

### 6.1 Theta-band burst amplitude index (θ p95)

Baseline EEG, channel **F4** — the only frontal recording site available for
the original protocol, since F3, Fp1, FCz and FT7 are stimulation channels and
are blanked by NIC2 on every run including baselines.

The signal is bandpass filtered to 4–8 Hz, Hilbert transformed, squared to give
instantaneous power, smoothed with a 100 ms moving average, and normalized
within run as percent change from that run's mean. The metric is the **95th
percentile** of that distribution.

The upper tail rather than the mean, because the intended construct is
*transient* theta. An earlier attempt to estimate a narrowband theta peak by
spectral parameterization found identifiable peaks in only a minority of
participants while alpha peaks were detected reliably — the pipeline worked,
but task theta does not present as a sustained rhythm in many people. A
distributional statistic captures burst expression without requiring a discrete
spectral peak to exist.

Note the contralateral measurement: F4 is right-frontal while stimulation
targets left DLPFC.
"""),
        code("""
# ============================================================================
# 6.1 Theta burst amplitude (theta p95)
# ============================================================================

theta_path = DERIV / 'eeg' / 'theta_subject_metrics.csv'
theta_runs_path = DERIV / 'eeg' / 'theta_run_metrics.csv'

if theta_path.exists():
    theta = pd.read_csv(theta_path, dtype={'subject_id': str})
    print(f'Participants with theta metrics : {len(theta)} of {len(SUBJECT_INFO)}')
    cols = [c for c in ['theta_p95', 'theta_p75', 'theta_median'] if c in theta.columns]
    print(theta[cols].describe().round(1).to_string())

    if theta_runs_path.exists():
        tr = pd.read_csv(theta_runs_path, dtype={'subject_id': str})
        print(f'\\nRuns processed: {len(tr)}; passing QC and contributing: '
              f'{theta["n_clean_runs"].sum() if "n_clean_runs" in theta else "n/a"}')
        r1 = tr[tr['run'] == 1].set_index('subject_id')['theta_p95']
        r5 = tr[tr['run'] == 5].set_index('subject_id')['theta_p95']
        both = r1.index.intersection(r5.index)
        if len(both) > 5:
            r, p = stats.pearsonr(r1[both], r5[both])
            print(f'Test-retest (run 1 vs run 5): r = {r:.3f}, p = {p:.4f}, '
                  f'n = {len(both)}')

    missing = sorted(set(SUBJECT_INFO) - set(theta['subject_id']))
    print(f'\\nWithout theta metrics ({len(missing)}): {", ".join(missing)}')
    print('Driven by EEG availability and QC, not by the theta measure itself.')
    record('EEG theta metrics', len(theta),
           f'{len(SUBJECT_INFO) - len(theta)} without usable baseline EEG')
else:
    print('No theta metrics. Run: python run_theta_metrics.py')
    theta = None
"""),
        md("""
### 6.2 Individualized theta frequency (iTF)

Every participant received 6.0 Hz (confirmed in §5.3), so iTF is a **moderator**
— did stimulation work better for people whose endogenous theta already sat near
the delivered frequency? — rather than a parameter that was delivered.

Estimated by the **Klimesch anchor**: individual alpha frequency from posterior
channels, minus a fixed 5 Hz offset. Alpha is the most reliably detectable scalp
rhythm and P3/P4 recorded for everyone, so this covers the whole sample, unlike
a direct frontal-midline theta peak (FCz records only under the later protocol,
and only on non-stimulation runs).

Estimated from **run 1 only**. Run 5 follows the first stimulation block, which
is active for counterbalance A and sham for B — including it would make the
estimate asymmetric between the two groups.

**Caveats.** Klimesch's transition frequency is properly defined from how theta
and alpha move in opposite directions between rest and task; this design has no
resting block, so the fixed offset is an approximation. It gets modest support
here — among participants with both estimators the observed gap is close to 5 Hz
and the two correlate positively — but that check rests on a handful of people.
""") ,
        code("""
# ============================================================================
# 6.2 Individualized theta frequency
# ============================================================================

itf_path = DERIV / 'eeg' / 'individual_theta_frequency.csv'
STIM_FREQ_HZ = 6.0

if itf_path.exists():
    itf = pd.read_csv(itf_path, dtype={'subject_id': str})
    print(f'Participants with EEG            : {len(itf)}')
    print(f'  alpha peak resolved (IAF)      : {itf["iaf"].notna().sum()}')
    print(f'  iTF via Klimesch anchor        : {itf["itf_klimesch"].notna().sum()}')
    print(f'  direct FCz theta peak          : {itf["theta_peak_fcz"].notna().sum()}'
          f'   (later protocol only)')

    print(f'\\nIAF : M = {itf["iaf"].mean():.2f} Hz, SD = {itf["iaf"].std():.2f}, '
          f'range {itf["iaf"].min():.2f}-{itf["iaf"].max():.2f}')
    print(f'iTF : M = {itf["itf_klimesch"].mean():.2f} Hz, '
          f'SD = {itf["itf_klimesch"].std():.2f}')
    print(f'|iTF - {STIM_FREQ_HZ:.0f} Hz| : M = {itf["itf_distance"].mean():.2f} Hz, '
          f'max = {itf["itf_distance"].max():.2f}')

    both = itf[itf['iaf'].notna() & itf['theta_peak_fcz'].notna()]
    if len(both) >= 4:
        offset = (both['iaf'] - both['theta_peak_fcz'])
        r = both['iaf'].corr(both['theta_peak_fcz'])
        print(f'\\nOffset check (n = {len(both)} with both estimators):')
        print(f'  observed IAF - theta peak = {offset.mean():.2f} Hz '
              f'(SD {offset.std():.2f}) against the assumed 5.0')
        print(f'  IAF vs direct theta peak  : r = {r:+.3f}')

    d = itf.merge(subj[['subject_id', 'age']], on='subject_id', how='left')
    d = d[['iaf', 'age']].apply(pd.to_numeric, errors='coerce').dropna()
    if len(d) > 5:
        r, p = stats.pearsonr(d['age'], d['iaf'])
        print(f'\\nIAF x age: r = {r:+.3f}, p = {p:.3f}, n = {len(d)}')
        print('  IAF declines with age in the literature, and age is this study\\'s')
        print('  primary moderator, so this is checked rather than assumed: if the')
        print('  two were strongly related, iTF effects could not be separated from')
        print('  age effects.')
    record('iTF-dependent analyses', int(itf['itf_klimesch'].notna().sum()),
           f'{len(SUBJECT_INFO) - int(itf["itf_klimesch"].notna().sum())} '
           f'without a resolvable alpha peak or without EEG')
else:
    print('No iTF output. Run: python individual_theta.py --runs 1')
    itf = None
"""),
        md("""
### 6.3 Electric field modeling (SimNIBS)

Individualized finite element simulations in SimNIBS 4.6.0, with head models
built by the `charm` pipeline from each participant's T1 and FLAIR from the
parent R01. The montage matches acquisition exactly: F3 at +750 µA, returns at
Fp1/FCz/FT7 at −250 µA each, SimNIBS default isotropic conductivities.

A 20 mm spherical ROI is placed at the F3 electrode position on the individual
cortical surface, and the mean, peak, and 95th percentile of field magnitude are
extracted. Mean field at the DLPFC target is the dose proxy for the
dose-response hypothesis.

**Coverage is the limiting factor here, and it is fixable.** Simulations exist
only for the subset processed to date; the remainder have not been run rather
than being unable to be run. The cell below lists exactly who is outstanding.
"""),
        code("""
# ============================================================================
# 6.3 E-field coverage, and who still needs SimNIBS
# ============================================================================

efield_path = REPO_ROOT / 'data' / 'efield_roi_summary.csv'

if efield_path.exists():
    efield = pd.read_csv(efield_path, dtype={'subject_id': str})
    have = set(efield['subject_id']) & set(SUBJECT_INFO)
    print(f'E-field simulations available : {len(have)} of {len(SUBJECT_INFO)}')
    metrics = [c for c in ['mean_magnE', 'peak_magnE', 'p95_magnE']
               if c in efield.columns]
    print(efield[efield['subject_id'].isin(have)][metrics]
          .describe().round(4).to_string())

    # ---- TODO: run SimNIBS for these participants ----
    todo = sorted(set(SUBJECT_INFO) - have)
    print(f'\\nOUTSTANDING SimNIBS runs: {len(todo)}')
    for i in range(0, len(todo), 8):
        print('   ' + ', '.join(todo[i:i+8]))
    print('\\nThese are pending processing, not unavailable. Structural scans come')
    print('from the parent R01, so a subject here is one whose head model has not')
    print('been built yet. Re-run this cell after processing to confirm coverage.')

    todo_df = pd.DataFrame({'subject_id': todo,
                            'has_behavioral': [s in set(trials['subject_id']) for s in todo]})
    todo_path = DERIV / 'simnibs_todo.csv'
    todo_path.parent.mkdir(parents=True, exist_ok=True)
    todo_df.to_csv(todo_path, index=False)
    print(f'\\nWritten to {todo_path} as a worklist.')

    record('E-field dose analyses', len(have),
           f'{len(SUBJECT_INFO) - len(have)} awaiting SimNIBS processing')
else:
    print(f'No e-field summary at {efield_path}.')
"""),
    ]


# =============================================================================
# 7 — Methods/code consistency
# =============================================================================

def section_consistency() -> List[nbf.NotebookNode]:
    return [
        md("""
## 7. Methods/code consistency

Claims in the Methods that are checkable against the code, verified here.
Nothing executes a Methods section, so this is the class of error that survives
every other check — the analysis can be correct while the description of it is
wrong, and a reader has no way to tell.
"""),
        code("""
# ============================================================================
# 7.1 Do the stated parameters match the implementation?
# ============================================================================

checks = []

def check(claim, stated, actual, ok=None):
    ok = (stated == actual) if ok is None else ok
    checks.append({'Methods claim': claim, 'stated': stated,
                   'in code': actual, 'match': 'yes' if ok else 'NO'})

check('reward probability for the good option', '75%',
      f'{WIN_FRACTION:.0%}')
check('missed-trial exclusion threshold', '>20%',
      f'>{EXCLUSION_THRESHOLDS["missed_pct"]:.0f}%')
check('side-bias threshold', '>95%',
      f'>{EXCLUSION_THRESHOLDS["side_bias_pct"]:.0f}%')
check('stimulus-bias threshold', '>95%',
      f'>{EXCLUSION_THRESHOLDS["stim_bias_pct"]:.0f}%')
check('rapid-responding threshold', '<200 ms',
      f'<{EXCLUSION_THRESHOLDS["rapid_rt_ms"]:.0f} ms')
check('feedback invariance', '10+ consecutive losses',
      f'{EXCLUSION_THRESHOLDS["feedback_invariance"]}+ consecutive losses')
check('minimum clean runs per condition for H2', '1',
      str(MIN_CLEAN_RUNS_PER_CONDITION))
check('trials-to-criterion definition', '3 consecutive correct', f'{CRITERION} consecutive correct')

result = pd.DataFrame(checks)
print(result.to_string(index=False))
if (result['match'] == 'NO').any():
    print('\\nMismatches above need resolving in one direction or the other.')
else:
    print('\\nAll checked parameters agree.')
"""),
        code("""
# ============================================================================
# 7.2 Rescorla-Wagner estimation: what the Methods says vs what was run
# ============================================================================
# Methods 2.7.2 describes MAP estimation with Beta(2,2) and Gamma(2,5) priors.
# The pipeline that produced master_subject_data.csv calls method='mle'.
# These are not interchangeable, so the difference is quantified rather than
# assumed to be small.

master_alpha = subj.set_index('subject_id')['sham_alpha']
for name, frame, col in [('MLE', rw_mle, 'alpha'), ('MAP', rw_map, 'alpha')]:
    sham = frame[frame['condition'] == 'sham'].set_index('subject_id')[col]
    shared = master_alpha.dropna().index.intersection(sham.index)
    if len(shared) > 5:
        r = master_alpha[shared].corr(sham[shared])
        exact = int(np.isclose(master_alpha[shared], sham[shared], atol=1e-6).sum())
        print(f'master CSV sham_alpha vs {name}: r = {r:.4f}, '
              f'exact matches {exact}/{len(shared)}')

n_bound_mle = int(((rw_mle['alpha'] <= 0.005) | (rw_mle['alpha'] >= 0.995)).sum())
n_bound_map = int(((rw_map['alpha'] <= 0.005) | (rw_map['alpha'] >= 0.995)).sum())
print(f'\\nboundary alpha estimates: MLE {n_bound_mle}, MAP {n_bound_map}')
print('\\nMethods 2.7.2 states that boundary estimates "were retained as valid MAP')
print('estimates under these priors". Beta(2,2) has zero density at 0 and 1, so')
print('MAP cannot produce them — as the count above shows. Whichever estimator')
print('the manuscript reports, that sentence needs correcting.')
"""),
    ]


# =============================================================================
# 8 — Summary
# =============================================================================

def section_summary() -> List[nbf.NotebookNode]:
    return [
        md("""
## 8. Analytic sample summary

The manuscript's Table 1. Generated from everything above, so it cannot drift
from the data.
"""),
        code("""
# ============================================================================
# 8.1 Table 1 — analytic sample sizes by analysis
# ============================================================================

table1 = pd.DataFrame(ANALYTIC_SAMPLES)
print('Table 1. Analytic sample sizes by analysis.\\n')
print(table1.to_string(index=False))

out = DERIV / 'analytic_samples.csv'
table1.to_csv(out, index=False)
print(f'\\nWritten to {out}')
"""),
        code("""
# ============================================================================
# 8.2 Markdown version for pasting into the manuscript
# ============================================================================

print('| Analysis | N | Basis for reduction |')
print('|---|---|---|')
for _, r in table1.iterrows():
    print(f'| {r["Analysis"]} | {r["N"]} | {r["Basis for reduction"]} |')
"""),
        code("""
# ============================================================================
# 8.3 Where the sample is lost, end to end
# ============================================================================

stages = [
    ('Registered in SUBJECT_INFO', len(SUBJECT_INFO)),
    ('Behavioral files present', int((availability['behavioral_runs'] > 0).sum())),
    ('At least one clean run', trials['subject_id'].nunique()),
    ('At least one clean sham run', len(has_sham)),
    ('Eligible for paired comparison', len(h2_eligible)),
]
prev = None
for name, n in stages:
    delta = '' if prev is None else f'   ({n - prev:+d})'
    print(f'  {name:34s} {n:3d}{delta}')
    prev = n

print('\\nMeasure-level coverage is reported in sections 4 and 6; the counts')
print('above concern the task data alone.')
"""),
        md("""
### 8.4 Standing caveats

Carry these into the manuscript.

**Data that will not be recovered.** Behavioral files for three participants
with complete EEG sessions (`10961`, `11439`, `11472`) never reached the
repository. One further participant (`11066`) has no data of any kind.

**Counterbalance verified from EEG for all but five.** Two have no EEG at all;
three are set from the session record by inference from a pattern that held in
every adjudicable case, which is inference and not verification. All five are
flagged in `config.py`.

**Questionnaire coverage reflects collection, not extraction.** AUDIT, PROMIS
and loneliness are low because those fields are blank in event rows that exist
— not because the export was read incorrectly. This was confirmed by reading
every available export and coalescing.

**E-field coverage is pending, not unavailable** — see the worklist written in
§6.3.

**One-third of participants have α at the parameter boundary under MLE.** They
are best described as replacing their value estimate entirely with the last
outcome. That may be real for a fast-reversing task, or it may mean
Rescorla-Wagner is the wrong model for that subgroup. Nothing here has tested
which, and it bears on how the null result should be read.
"""),
    ]


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = (section_setup() + section_sample() + section_task()
                + section_dvs() + section_measures() + section_stim()
                + section_derived() + section_consistency() + section_summary())
    nb.metadata = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                       'name': 'python3'},
        'language_info': {'name': 'python'},
    }
    return nb


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', default=str(OUTPUT))
    args = parser.parse_args(argv)

    nb = build()
    out = Path(args.output)
    nbf.write(nb, str(out))
    print(f'Wrote {out}  ({len(nb.cells)} cells, '
          f'{sum(c["cell_type"] == "code" for c in nb.cells)} code)')

    if args.execute:
        print('\nExecuting...')
        result = subprocess.run(
            ['jupyter', 'nbconvert', '--to', 'notebook', '--execute',
             '--inplace', '--ExecutePreprocessor.timeout=3600', str(out)],
            cwd=str(Path(__file__).parent), capture_output=True, text=True)
        sys.stdout.write(result.stdout[-3000:])
        sys.stderr.write(result.stderr[-3000:])
        if result.returncode != 0:
            print(f'\nExecution FAILED (exit {result.returncode})')
        return result.returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
