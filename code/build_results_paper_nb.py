"""
build_results_paper_nb.py — Assemble results_paper.ipynb

The paper notebook is generated rather than hand-edited so the porting from
results_figures_v2_withVS.ipynb stays reviewable: each section is a function
here, and regenerating is a diff rather than a merge of notebook JSON.

results_figures_v2_withVS.ipynb is the dissertation artifact and is not
modified. This builds the paper version alongside it.

Usage
-----
    python build_results_paper_nb.py            # write results_paper.ipynb
    python build_results_paper_nb.py --execute  # write, then run it end to end
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import nbformat as nbf

OUTPUT = Path(__file__).parent / 'results_paper.ipynb'


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


# =============================================================================
# Section 0 — Setup
# =============================================================================

def section_setup() -> List[nbf.NotebookNode]:
    return [
        md("""
# tACS Bandit — Results for the Empirical Paper

Analyses for the manuscript, built on the expanded sample. Ported from
`results_figures_v2_withVS.ipynb` (the dissertation artifact, left untouched).

**Sample switch.** Everything below keys off the `SAMPLE` constant in the next
cell. `'all'` runs the expanded sample; `'dissertation'` reproduces the frozen
N=39. Both the subject-level table and the trial-level data follow it, so they
can never disagree — a split between them was a live bug in the previous
notebook, which read a 66-subject CSV against 38 subjects of trial data.

**What changed in the data since the defense.** Counterbalance was verified
against the 6 Hz stimulation artifact in the EEG for every subject
(`stim_verification.py`); 14 subjects were corrected, one of them (10641)
inside the defended sample. Running in `'dissertation'` mode therefore
reproduces the defended numbers for every subject *except* 10641, whose
active/sham labels were previously swapped, and 10804, whose defended
Rescorla-Wagner fit was pinned at the parameter bounds with a likelihood
exactly at chance. Section 0.4 checks precisely that.
"""),
        code("""
# ============================================================================
# 0.1 Sample selection
# ============================================================================
# 'all'          — every registered subject with usable data (expanded sample)
# 'dissertation' — the frozen N=39 defended sample, for reproduction checks

SAMPLE = 'all'

# Guard against a typo silently selecting a different analysis.
assert SAMPLE in ('all', 'dissertation', 'new'), f'unexpected SAMPLE: {SAMPLE!r}'
print(f'SAMPLE = {SAMPLE!r}')
"""),
        code("""
# ============================================================================
# 0.2 Imports and paths
# ============================================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import pearsonr, ttest_rel, ttest_ind

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

warnings.filterwarnings('ignore')

from config import (
    DATA_DIR, DISSERTATION_SUBJECTS, SUBJECT_INFO,
    NO_EARCLIP_SUBJECTS,
)
from data_loading import load_all_subjects
from exclusions import apply_all_exclusions

MASTER_CSV = DATA_DIR.parent / 'master_subject_data.csv'
FIG_DIR = DATA_DIR.parent / 'figures' / 'paper'
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f'Master CSV: {MASTER_CSV}')
print(f'Figures:    {FIG_DIR}')
"""),
        code("""
# ============================================================================
# 0.3 Publication figure style (JNeurosci)
# ============================================================================

CM_TO_IN = 1 / 2.54
WIDTH_1COL = 8.5 * CM_TO_IN       # 8.5 cm single column
WIDTH_1_5COL = 11.5 * CM_TO_IN    # 11.5 cm
WIDTH_2COL = 17.5 * CM_TO_IN      # 17.5 cm full width

FONT_FAMILY = 'Arial'
FONT_AXIS_TITLE = 7
FONT_TICK = 6.5
FONT_PANEL_LABEL = 10
FONT_LEGEND = 6
FONT_ANNOTATION = 6
FONT_STATS_BOX = 6.5

SHAM_COLOR = '#1565C0'
ACTIVE_COLOR = '#E64A19'
ACCENT_GREEN = '#2E7D32'
ACCENT_RED = '#C62828'
NEUTRAL_GRAY = '#757575'
REGRESSION_COLOR = '#404040'
CI_COLOR = 'gray'
CI_ALPHA = 0.18
SESOI_COLOR = '#BDBDBD'

AGE_MIN, AGE_MAX = 20, 80
AGE_YOUNG, AGE_OLD = '#1565C0', '#FFB300'
AGE_CMAP = mcolors.LinearSegmentedColormap.from_list('blue_gold', [AGE_YOUNG, AGE_OLD])


def age_to_hex(age):
    norm = np.clip((age - AGE_MIN) / (AGE_MAX - AGE_MIN), 0, 1)
    r, g, b, _ = AGE_CMAP(norm)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


plt.rcParams.update({
    'font.family': FONT_FAMILY,
    'axes.labelsize': FONT_AXIS_TITLE,
    'xtick.labelsize': FONT_TICK,
    'ytick.labelsize': FONT_TICK,
    'legend.fontsize': FONT_LEGEND,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 300,
    'figure.dpi': 110,
})


def style_ax(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_AXIS_TITLE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS_TITLE)
    if title:
        ax.set_title(title, fontsize=FONT_AXIS_TITLE)
    ax.tick_params(labelsize=FONT_TICK)
    return ax


print('Figure style set.')
"""),
    ]


# =============================================================================
# Section 0.4 — Data loading and the reproduction check
# =============================================================================

def section_data() -> List[nbf.NotebookNode]:
    return [
        md("""
## 0.4 Load data

Both tables are filtered by `SAMPLE`, and the cell below asserts they describe
the same subjects. In the previous notebook these were selected independently,
so the subject-level table could describe 66 people while the trial-level data
described 38.
"""),
        code("""
# ============================================================================
# 0.4 Load subject-level and trial-level data (both keyed to SAMPLE)
# ============================================================================

subj = pd.read_csv(MASTER_CSV, dtype={'subject_id': str})

trials_all = load_all_subjects(sample=SAMPLE, verbose=False)
exclusion_results = apply_all_exclusions(trials_all, verbose=False)

trials = exclusion_results['data_clean']
h2_eligible = [str(s) for s in exclusion_results['h2_eligible']]
trials_sham = trials[trials['condition'] == 'sham'].copy()

# Restrict the subject-level table to whoever actually has trial data in this
# sample, so every N reported below refers to the same set of people.
subjects_with_data = sorted(trials['subject_id'].unique())
subj = subj[subj['subject_id'].isin(subjects_with_data)].reset_index(drop=True)

assert set(subj['subject_id']) == set(subjects_with_data), (
    'subject-level and trial-level data describe different subjects'
)

print(f'SAMPLE = {SAMPLE!r}')
print(f'  subject-level : {len(subj)} subjects x {len(subj.columns)} variables')
print(f'  trial-level   : {len(trials)} clean trials, '
      f'{trials["subject_id"].nunique()} subjects')
print(f'  H2-eligible   : {len(h2_eligible)} subjects')
"""),
        code("""
# ============================================================================
# 0.5 Reproduction check against the defended results
# ============================================================================
# Run with SAMPLE='dissertation' to verify the port reproduces the defended
# numbers. Two subjects are expected to differ, for documented reasons:
#   10641 — counterbalance corrected from B to A against the EEG, so its
#           active/sham measures were previously computed with swapped labels
#   10804 — its defended R-W fit sat at both parameter bounds with a
#           likelihood exactly at chance; the current fit escapes that optimum
# Any other difference means the port changed something it should not have.

EXPECTED_DIFFERENCES = {'10641', '10804'}

if SAMPLE == 'dissertation':
    import io, subprocess

    defended_raw = subprocess.run(
        ['git', 'show', 'dissertation-defense-v1:data/master_subject_data.csv'],
        capture_output=True, text=True, cwd=str(DATA_DIR.parent.parent),
    ).stdout

    if defended_raw:
        defended = pd.read_csv(io.StringIO(defended_raw), dtype={'subject_id': str})
        merged = defended.merge(subj, on='subject_id', suffixes=('_old', '_new'))

        check_cols = ['sham_p_stay_win', 'sham_p_shift_lose', 'sham_accuracy',
                      'sham_alpha', 'sham_beta', 'theta_p95']
        unexpected = set()
        for col in check_cols:
            a, b = merged.get(f'{col}_old'), merged.get(f'{col}_new')
            if a is None or b is None:
                continue
            both = a.notna() & b.notna()
            differing = merged.loc[both & ((a - b).abs() > 0.01), 'subject_id']
            unexpected |= set(differing) - EXPECTED_DIFFERENCES
            print(f'  {col:20s}: n={both.sum():2d}, '
                  f'differing={sorted(set(differing))}')

        if unexpected:
            print(f'\\n  UNEXPECTED differences: {sorted(unexpected)}')
        else:
            print('\\n  Reproduction OK — only the documented subjects differ.')
    else:
        print('  Could not read the defended CSV from the git tag; skipping.')
else:
    print(f"Reproduction check runs only when SAMPLE='dissertation' "
          f"(currently {SAMPLE!r}).")
"""),
        code("""
# ============================================================================
# 0.6 Derived variables
# ============================================================================

# --- Reduced cognitive composite -------------------------------------------
# Digit Span, BVMT and Trails A were collected only for participants aged 40+,
# so a composite including them is unavailable for younger subjects. The
# primary composite uses the five measures with near-complete coverage.
reduced_measures = {
    'Attention': ['flanker_score', 'running_dots_score'],
    'Memory': ['hvlt_total'],
    'Speed': ['salthouse_letter', 'salthouse_pattern'],
}

for domain, measures in reduced_measures.items():
    for m in measures:
        if m in subj.columns:
            vals = subj[m].dropna()
            if len(vals) > 1:
                subj[f'{m}_z_reduced'] = (subj[m] - vals.mean()) / vals.std()

subj['attention_reduced'] = subj[['flanker_score_z_reduced',
                                  'running_dots_score_z_reduced']].mean(axis=1)
subj['memory_reduced'] = subj[['hvlt_total_z_reduced']].mean(axis=1)
subj['speed_reduced'] = subj[['salthouse_letter_z_reduced',
                              'salthouse_pattern_z_reduced']].mean(axis=1)
subj['global_reduced'] = subj[['attention_reduced', 'memory_reduced',
                               'speed_reduced']].mean(axis=1)

COG_COMPOSITE = 'global_reduced'

# --- Age split, computed on the H2-eligible subset --------------------------
h2_subj = subj[subj['subject_id'].isin(h2_eligible)].copy()
h2_subj['age'] = pd.to_numeric(h2_subj['age'], errors='coerce')
MEDIAN_AGE = h2_subj['age'].median()

subj['age_group'] = np.where(
    pd.to_numeric(subj['age'], errors='coerce') < MEDIAN_AGE, 'Younger', 'Older')

# --- Earclip status ---------------------------------------------------------
# Read from the registry rather than a hardcoded list, so it cannot drift out
# of sync with config.py as the sample grows.
subj['has_earclip'] = ~subj['subject_id'].isin(NO_EARCLIP_SUBJECTS)

print(f'Cognitive composite: {COG_COMPOSITE} '
      f'({subj[COG_COMPOSITE].notna().sum()}/{len(subj)} with data)')
print(f'Median age (H2 subset): {MEDIAN_AGE:.1f}')
print(f"  Younger: N = {(subj['age_group'] == 'Younger').sum()}, "
      f"Older: N = {(subj['age_group'] == 'Older').sum()}")
print(f"Earclip: {subj['has_earclip'].sum()}/{len(subj)}")
"""),
    ]


# =============================================================================
# Section 1 — Sample descriptives and blinding
# =============================================================================

def section_descriptives() -> List[nbf.NotebookNode]:
    return [
        md('## 1. Sample descriptives and blinding integrity'),
        code("""
# ============================================================================
# 1.1 Sample descriptives
# ============================================================================

demo_vars = ['age', 'education_years', COG_COMPOSITE]
available = [v for v in demo_vars if v in subj.columns]
desc = subj[available].apply(pd.to_numeric, errors='coerce').describe().round(2)
print('Demographics:')
print(desc.to_string())

if 'gender' in subj.columns:
    print('\\nGender:')
    print(subj['gender'].value_counts(dropna=False).to_string())

print('\\nCounterbalance:')
print(subj['subject_id'].map(
    lambda s: SUBJECT_INFO.get(s, {}).get('counterbalance', '?')
).value_counts().to_string())
"""),
        code("""
# ============================================================================
# 1.2 Blinding integrity
# ============================================================================
# Participants guessed active vs sham after each stimulation run; d' tests
# whether they could discriminate.

from blinding_analysis import run_blinding_analysis

blinding = run_blinding_analysis(trials, verbose=False)
m = blinding['metrics']

if m:
    print(f"n observations : {m['n_obs']} from {m['n_subjects']} subjects")
    print(f"d'             : {m['dprime']:.3f}")
    print(f"criterion      : {m['criterion']:.3f}")
    print(f"accuracy       : {m['accuracy']:.1%} "
          f"({m['n_correct']}/{m['n_obs']})")
    print(f"binomial p     : {m['binom_p']:.3f}")
    print(f"chi-square     : chi2 = {m['chi2']:.2f}, p = {m['chi_p']:.3f}")
    print(f"\\nblinding intact: {blinding['blinding_intact']}")
else:
    print('No blinding data available for this sample.')
"""),
        code("""
# ============================================================================
# 1.3 Baseline (sham) descriptives
# ============================================================================

sham_vars = ['sham_accuracy', 'sham_win_rate', 'sham_p_stay_win',
             'sham_p_shift_lose', 'sham_alpha', 'sham_beta']
available = [v for v in sham_vars if v in subj.columns]
print('Baseline (sham) performance:')
print(subj[available].describe().round(3).to_string())

df = subj[['age', 'sham_accuracy']].dropna()
if len(df) > 2:
    r, p = stats.pearsonr(df['age'].astype(float), df['sham_accuracy'].astype(float))
    print(f'\\nAge x sham accuracy: r = {r:.3f}, p = {p:.3f}, N = {len(df)}')
"""),
    ]


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = (
        section_setup()
        + section_data()
        + section_descriptives()
    )
    nb.metadata = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                       'name': 'python3'},
        'language_info': {'name': 'python'},
    }
    return nb


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--execute', action='store_true',
                        help='run the notebook after writing it')
    parser.add_argument('--output', default=str(OUTPUT))
    args = parser.parse_args(argv)

    nb = build()
    out = Path(args.output)
    nbf.write(nb, str(out))
    n_code = sum(c['cell_type'] == 'code' for c in nb.cells)
    print(f'Wrote {out}  ({len(nb.cells)} cells, {n_code} code)')

    if args.execute:
        print('\nExecuting...')
        result = subprocess.run(
            ['jupyter', 'nbconvert', '--to', 'notebook', '--execute',
             '--inplace', '--ExecutePreprocessor.timeout=1800', str(out)],
            cwd=str(out.parent), capture_output=True, text=True,
        )
        sys.stdout.write(result.stdout[-4000:])
        sys.stderr.write(result.stderr[-4000:])
        return result.returncode

    return 0


if __name__ == '__main__':
    sys.exit(main())
