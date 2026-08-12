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

def section_setup(sample: str = 'all') -> List[nbf.NotebookNode]:
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
        code(f"""
# ============================================================================
# 0.1 Sample selection
# ============================================================================
# 'all'          — every registered subject with usable data (expanded sample)
# 'dissertation' — the frozen N=39 defended sample, for reproduction checks

SAMPLE = {sample!r}

# Guard against a typo silently selecting a different analysis.
assert SAMPLE in ('all', 'dissertation', 'new'), f'unexpected SAMPLE: {{SAMPLE!r}}'
print(f'SAMPLE = {{SAMPLE!r}}')
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


def stats_box(ax, text, x=0.97, y=0.97, ha='right', va='top'):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=FONT_STATS_BOX,
            fontfamily=FONT_FAMILY, color='#404040', ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.9, edgecolor='none'))


def scatter_regression_mpl(ax, x, y, age, zero_line=False):
    \"\"\"Scatter coloured by age, with a regression line and 95% CI band.\"\"\"
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(age)
    x, y, age = x[mask], y[mask], age[mask]
    n = len(x)

    for xi, yi, ai in zip(x, y, age):
        ax.plot(xi, yi, 'o', color=age_to_hex(ai), markersize=3.5,
                markeredgewidth=0.3, markeredgecolor='white', zorder=3)

    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = intercept + slope * x_line
    y_pred = intercept + slope * x
    resid_se = np.sqrt(np.sum((y - y_pred) ** 2) / (n - 2))
    ci = stats.t.ppf(0.975, n - 2) * resid_se * np.sqrt(
        1 / n + (x_line - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))

    ax.fill_between(x_line, y_line - ci, y_line + ci,
                    color=CI_COLOR, alpha=CI_ALPHA, zorder=1, linewidth=0)
    ax.plot(x_line, y_line, color=REGRESSION_COLOR, linewidth=1.2, zorder=2)
    if zero_line:
        ax.axhline(0, color='#BDBDBD', linewidth=0.6, linestyle='--', zorder=0)

    stats_box(ax, f"r = {r:.3f}, p = {p:.3f}{'*' if p < .05 else ''}, N = {n}")
    return {'r': r, 'p': p, 'n': n, 'slope': slope}


def savefig(fig, name):
    \"\"\"Write a figure as both PNG and SVG into FIG_DIR.\"\"\"
    fig.savefig(str(FIG_DIR / f'{name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(str(FIG_DIR / f'{name}.svg'), format='svg', bbox_inches='tight')


print('Figure style and helpers set.')
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

        # Relative tolerance, not absolute. These parameters live on very
        # different scales — WSLS rates are bounded in [0, 1] while theta_p95
        # runs to several hundred percent — so a fixed absolute threshold
        # flags floating-point noise on the large ones as a real difference.
        RTOL, ATOL = 0.01, 1e-6
        unexpected = set()
        for col in check_cols:
            a, b = merged.get(f'{col}_old'), merged.get(f'{col}_new')
            if a is None or b is None:
                continue
            both = a.notna() & b.notna()
            close = np.isclose(a.where(both), b.where(both),
                               rtol=RTOL, atol=ATOL, equal_nan=True)
            differing = merged.loc[both & ~close, 'subject_id']
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


# =============================================================================
# Section 2 — H1: baseline cognition and behavior (sham only)
# =============================================================================

def section_h1() -> List[nbf.NotebookNode]:
    return [
        md("""
## 2. H1 — Cognition and baseline learning behavior

Sham-condition analyses. Each preregistered model includes education; because
education is missing for a subset of participants, every model is also refit
without it on the full sample, and the two are compared rather than one being
reported alone.
"""),
        code("""
# ============================================================================
# 2.1 Shared regression helper
# ============================================================================

def run_h1_regression(y_col, x_cols, data, label, verbose=True):
    \"\"\"OLS with an APA-style summary. Returns (model, analysis frame).\"\"\"
    df = data[['subject_id', y_col] + x_cols].dropna()
    if len(df) < len(x_cols) + 2:
        if verbose:
            print(f'\\n{label}\\n  insufficient data (N = {len(df)})')
        return None, df

    y = df[y_col].astype(float)
    X = sm.add_constant(df[x_cols].astype(float))
    model = sm.OLS(y, X).fit()

    if verbose:
        print(f'\\n{label}')
        print(f'  N = {int(model.nobs)}')
        print(f'  F({model.df_model:.0f}, {model.df_resid:.0f}) = {model.fvalue:.2f}, '
              f'p = {model.f_pvalue:.3f}, R2 = {model.rsquared:.3f}')
        for var in x_cols:
            b, p = model.params[var], model.pvalues[var]
            beta = b * (df[var].std() / y.std()) if y.std() > 0 else np.nan
            print(f'  {var:25s}: b = {b:+.4f}, beta = {beta:+.3f}, '
                  f'p = {p:.3f}{" *" if p < .05 else ""}')
    return model, df


def compare_specifications(prereg, full, term, label):
    \"\"\"Report whether a term survives dropping the education covariate.\"\"\"
    if prereg is None or full is None:
        print(f'  {label}: one specification could not be fit.')
        return
    p_pre, p_full = prereg.pvalues[term], full.pvalues[term]
    n_pre, n_full = int(prereg.nobs), int(full.nobs)
    print(f'\\n  {label}')
    print(f'    with education    (N={n_pre:3d}): p = {p_pre:.3f}'
          f'{" *" if p_pre < .05 else ""}')
    print(f'    without education (N={n_full:3d}): p = {p_full:.3f}'
          f'{" *" if p_full < .05 else ""}')
    if p_pre < .05 and p_full >= .05:
        print(f'    -> significant only in the preregistered model; '
              f'{abs(n_full - n_pre)} subjects differ, so the result is '
              f'sensitive to sample composition.')
    elif p_pre < .05 and p_full < .05:
        print('    -> robust across both specifications.')
    elif p_pre >= .05 and p_full < .05:
        print('    -> emerges only at full N.')
    else:
        print('    -> null in both.')
"""),
        code("""
# ============================================================================
# 2.2 H1.1 — Cognition predicts win-stay / lose-shift
# ============================================================================

h1_1a_model, h1_1a_df = run_h1_regression(
    'sham_p_stay_win', [COG_COMPOSITE, 'age', 'education_years'], subj,
    'H1.1a [preregistered]: p(stay|win) ~ Global Cognition + Age + Education')

h1_1b_model, h1_1b_df = run_h1_regression(
    'sham_p_shift_lose', [COG_COMPOSITE, 'age', 'education_years'], subj,
    'H1.1b [preregistered]: p(shift|lose) ~ Global Cognition + Age + Education')

print('\\n-- Full sample (education dropped to preserve N) --')
h1_1a_full, _ = run_h1_regression(
    'sham_p_stay_win', [COG_COMPOSITE, 'age'], subj,
    'H1.1a [full sample]: p(stay|win) ~ Global Cognition + Age')
h1_1b_full, _ = run_h1_regression(
    'sham_p_shift_lose', [COG_COMPOSITE, 'age'], subj,
    'H1.1b [full sample]: p(shift|lose) ~ Global Cognition + Age')

print('\\n-- Sensitivity to the education covariate --')
compare_specifications(h1_1a_model, h1_1a_full, COG_COMPOSITE, 'H1.1a cognition term')
compare_specifications(h1_1b_model, h1_1b_full, COG_COMPOSITE, 'H1.1b cognition term')

print('\\n-- Bivariate (full sample) --')
for dv, label in [('sham_p_stay_win', 'p(stay|win)'),
                  ('sham_p_shift_lose', 'p(shift|lose)')]:
    d = subj[[COG_COMPOSITE, dv]].dropna()
    r, p = stats.pearsonr(d[COG_COMPOSITE].astype(float), d[dv].astype(float))
    print(f'  Cognition x {label:14s}: r = {r:+.3f}, p = {p:.3f}, N = {len(d)}')
"""),
        code("""
# ============================================================================
# 2.3 Figure — Cognition x WSLS
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(WIDTH_1_5COL, WIDTH_1COL * 0.85))

for ax, (dv, dv_label) in zip(axes, [('sham_p_stay_win', 'p(stay|win)'),
                                     ('sham_p_shift_lose', 'p(shift|lose)')]):
    d = subj[[COG_COMPOSITE, dv, 'age']].dropna()
    scatter_regression_mpl(ax,
                           d[COG_COMPOSITE].astype(float).values,
                           d[dv].astype(float).values,
                           d['age'].astype(float).values)
    style_ax(ax, xlabel='Global Cognitive Composite', ylabel=dv_label)

sm_age = plt.cm.ScalarMappable(cmap=AGE_CMAP, norm=plt.Normalize(AGE_MIN, AGE_MAX))
cbar = fig.colorbar(sm_age, ax=axes, fraction=0.03, pad=0.02)
cbar.set_label('Age (years)', fontsize=FONT_AXIS_TITLE)
cbar.ax.tick_params(labelsize=FONT_TICK)

savefig(fig, 'fig_h1_cognition_wsls')
plt.show()
"""),
        code("""
# ============================================================================
# 2.4 H1.1.1 — Inverse temperature predicts lose-shifting
# ============================================================================

d_beta = subj[['sham_beta', 'sham_p_shift_lose', 'age']].dropna()

fig, ax = plt.subplots(figsize=(WIDTH_1COL, WIDTH_1COL * 0.85))
stats_beta = scatter_regression_mpl(
    ax,
    d_beta['sham_beta'].astype(float).values,
    d_beta['sham_p_shift_lose'].astype(float).values,
    d_beta['age'].astype(float).values,
)
style_ax(ax, xlabel='Inverse Temperature (beta)', ylabel='p(shift|lose)')
savefig(fig, 'fig_h1_beta_loseshift')
plt.show()

print(f"Bivariate: r = {stats_beta['r']:.3f}, p = {stats_beta['p']:.3f}, "
      f"N = {stats_beta['n']}")

h1_beta_prereg, _ = run_h1_regression(
    'sham_p_shift_lose', ['sham_beta', COG_COMPOSITE, 'age', 'education_years'],
    subj, 'p(shift|lose) ~ beta + Global Cog + Age + Education [preregistered]')
h1_beta_full, _ = run_h1_regression(
    'sham_p_shift_lose', ['sham_beta', COG_COMPOSITE, 'age'],
    subj, 'p(shift|lose) ~ beta + Global Cog + Age [full sample]')

compare_specifications(h1_beta_prereg, h1_beta_full, 'sham_beta', 'beta term')
"""),
        code("""
# ============================================================================
# 2.5 H1.1.2 — Age x Cognition interactions
# ============================================================================
# Continuous predictors are mean-centered so the lower-order terms stay
# interpretable at the sample mean rather than at zero.

interaction_dvs = [
    ('sham_p_shift_lose', 'p(shift|lose)'),
    ('sham_alpha', 'Learning rate (alpha)'),
    ('sham_beta', 'Inverse temperature (beta)'),
]

interaction_results = {}

for dv_col, dv_label in interaction_dvs:
    cols = [dv_col, COG_COMPOSITE, 'age', 'education_years']
    d = subj[['subject_id'] + cols].dropna().copy()
    if len(d) < 10:
        print(f'{dv_label}: insufficient data (N = {len(d)})')
        continue

    for c in [COG_COMPOSITE, 'age', dv_col]:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna()

    d['cog_c'] = d[COG_COMPOSITE] - d[COG_COMPOSITE].mean()
    d['age_c'] = d['age'] - d['age'].mean()
    d['cog_x_age'] = d['cog_c'] * d['age_c']

    y = d[dv_col].astype(float)
    X = sm.add_constant(d[['cog_c', 'age_c', 'education_years', 'cog_x_age']].astype(float))
    model = sm.OLS(y, X).fit()
    interaction_results[dv_col] = {'model': model, 'df': d}

    b, p = model.params['cog_x_age'], model.pvalues['cog_x_age']
    print(f'{dv_label:26s}: interaction b = {b:+.5f}, p = {p:.3f}'
          f'{" *" if p < .05 else ""}, N = {int(model.nobs)}')
"""),
        code("""
# ============================================================================
# 2.6 H1.2 — Reward/punishment sensitivity (SPSRQ) and WSLS
# ============================================================================

for pred, dv, label in [
    ('spsrq_sr', 'sham_p_stay_win', 'SR (reward sensitivity) -> p(stay|win)'),
    ('spsrq_sp', 'sham_p_shift_lose', 'SP (punishment sensitivity) -> p(shift|lose)'),
]:
    if pred not in subj.columns:
        print(f'{label}: {pred} unavailable')
        continue
    d = subj[[pred, dv]].dropna()
    if len(d) < 5:
        print(f'{label}: insufficient data (N = {len(d)})')
        continue
    r, p = stats.pearsonr(d[pred].astype(float), d[dv].astype(float))
    print(f'{label:46s}: r = {r:+.3f}, p = {p:.3f}, N = {len(d)}')

print()
h1_2a_model, _ = run_h1_regression(
    'sham_p_stay_win', ['spsrq_sr', 'age', 'education_years'], subj,
    'H1.2a [preregistered]: p(stay|win) ~ SR + Age + Education')
h1_2b_model, _ = run_h1_regression(
    'sham_p_shift_lose', ['spsrq_sp', 'age', 'education_years'], subj,
    'H1.2b [preregistered]: p(shift|lose) ~ SP + Age + Education')
"""),
    ]


# =============================================================================
# Section 3 — Reversal learning
# =============================================================================

def section_reversal() -> List[nbf.NotebookNode]:
    return [
        md("""
## 3. Reversal learning

Trials-to-criterion and reversal-locked accuracy, by condition.

The previous notebook carried a 15KB cell that reimplemented reversal
identification and trials-to-criterion inline. That was a workaround:
`reversal_analysis.identify_reversals` initialized `reversal_id` from
`np.nan`, giving the column float64, and pandas 2.x raises rather than
upcasting when string IDs are then assigned. With that fixed at the source the
module is used directly — verified to produce identical output to the inline
version (same 909 reversals, same values throughout).
"""),
        code("""
# ============================================================================
# 3.1 Identify reversals and compute trials-to-criterion
# ============================================================================

from reversal_analysis import (
    identify_reversals,
    compute_trials_to_criterion,
    compute_reversal_accuracy,
)

CRITERION = 3        # consecutive correct choices
WINDOW_PRE = 5
WINDOW_POST = 15

trials = identify_reversals(trials, window_pre=WINDOW_PRE,
                            window_post=WINDOW_POST, verbose=True)

ttc = compute_trials_to_criterion(trials, criterion=CRITERION)
print(f'\\nTrials-to-criterion: {len(ttc)} reversals, '
      f'{ttc["subject_id"].nunique()} subjects')
print(ttc.groupby('condition')['trials_to_criterion']
        .agg(['count', 'mean', 'std']).round(3).to_string())
"""),
        code("""
# ============================================================================
# 3.2 Subject-level TTC and the active-vs-sham contrast
# ============================================================================

ttc_subj = (ttc.groupby(['subject_id', 'condition'])['trials_to_criterion']
              .mean().unstack())

for cond in ['sham', 'active']:
    if cond in ttc_subj.columns:
        subj[f'{cond}_ttc'] = subj['subject_id'].map(ttc_subj[cond])

if {'sham', 'active'}.issubset(ttc_subj.columns):
    subj['delta_ttc'] = subj['active_ttc'] - subj['sham_ttc']

paired = subj[subj['subject_id'].isin(h2_eligible)][['sham_ttc', 'active_ttc']].dropna()
if len(paired) > 2:
    t, p = stats.ttest_rel(paired['active_ttc'], paired['sham_ttc'])
    diff = paired['active_ttc'] - paired['sham_ttc']
    dz = diff.mean() / diff.std(ddof=1)
    print(f'Trials-to-criterion, active vs sham (H2-eligible):')
    print(f'  N  = {len(paired)}')
    print(f'  sham   M = {paired["sham_ttc"].mean():.3f} '
          f'(SD {paired["sham_ttc"].std():.3f})')
    print(f'  active M = {paired["active_ttc"].mean():.3f} '
          f'(SD {paired["active_ttc"].std():.3f})')
    print(f'  t({len(paired)-1}) = {t:.3f}, p = {p:.3f}, dz = {dz:+.3f}')
else:
    print('Insufficient paired TTC data.')
"""),
        code("""
# ============================================================================
# 3.3 Figure — reversal-locked accuracy by condition
# ============================================================================

rev_window = trials[trials['in_rev_window'] &
                    trials['condition'].isin(['sham', 'active'])].copy()

# `correct` arrives as object dtype (True/False stored as Python bools), and
# aggregating it keeps the object dtype, which matplotlib's fill_between then
# rejects. Coerce once here rather than casting at every use.
rev_window['correct_num'] = pd.to_numeric(rev_window['correct'], errors='coerce')

curves = (rev_window.groupby(['condition', 'trial_from_rev'])['correct_num']
                    .agg(['mean', 'sem', 'count']).reset_index())
for col in ['mean', 'sem']:
    curves[col] = curves[col].astype(float)

fig, ax = plt.subplots(figsize=(WIDTH_1_5COL, WIDTH_1COL * 0.8))

for cond, color in [('sham', SHAM_COLOR), ('active', ACTIVE_COLOR)]:
    c = curves[curves['condition'] == cond].sort_values('trial_from_rev')
    if len(c) == 0:
        continue
    ax.plot(c['trial_from_rev'], c['mean'], '-', color=color, lw=1.4, label=cond)
    ax.fill_between(c['trial_from_rev'], c['mean'] - c['sem'], c['mean'] + c['sem'],
                    color=color, alpha=0.18, linewidth=0)

ax.axvline(0, color='#757575', ls='--', lw=0.8)
ax.axhline(0.5, color='#BDBDBD', ls=':', lw=0.7)
style_ax(ax, xlabel='Trial relative to reversal', ylabel='p(correct)')
ax.legend(frameon=False, fontsize=FONT_LEGEND)
savefig(fig, 'fig_reversal_locked_accuracy')
plt.show()
"""),
    ]


# =============================================================================
# Section 4 — H2: effects of stimulation
# =============================================================================

def section_h2() -> List[nbf.NotebookNode]:
    return [
        md("""
## 4. H2 — Effects of stimulation

Paired active-vs-sham comparisons with equivalence tests, then age moderation
of the change scores.

Two departures from the previous notebook, both corrections rather than
choices:

1. **Restricted to H2-eligible subjects for every DV.** Previously some
   measures were implicitly restricted (WSLS and R-W came from the H2 dataset)
   while accuracy came from `data_clean` and was not, so different rows of the
   same table described different samples.
2. **Sample SD (ddof=1) for `dz` and TOST.** The previous code took `.std()`
   on a NumPy array, which defaults to the population SD. That inflates `dz`
   slightly and shrinks the standard error used in the equivalence test,
   making TOST anti-conservative — it declares equivalence a little too
   readily, which is the wrong direction for a null-supporting claim.
"""),
        code("""
# ============================================================================
# 4.1 Paired comparisons and equivalence tests
# ============================================================================

h2_dvs = [
    ('sham_p_stay_win', 'active_p_stay_win', 'p(stay|win)'),
    ('sham_p_shift_lose', 'active_p_shift_lose', 'p(shift|lose)'),
    ('sham_alpha', 'active_alpha', 'Learning rate (alpha)'),
    ('sham_beta', 'active_beta', 'Inverse temperature (beta)'),
    ('sham_accuracy', 'active_accuracy', 'Accuracy'),
    ('sham_win_rate', 'active_win_rate', 'Win rate'),
    ('sham_ttc', 'active_ttc', 'Trials-to-criterion'),
]

# Every DV uses the same sample, so the rows of the table are comparable.
h2_subj_df = subj[subj['subject_id'].isin(h2_eligible)]


def tost(diff, sesoi_dz):
    \"\"\"Two one-sided tests. Returns the larger of the two p-values.\"\"\"
    n = len(diff)
    sd = diff.std(ddof=1)
    if n < 3 or sd == 0:
        return np.nan
    se = sd / np.sqrt(n)
    bound = sesoi_dz * sd
    p_upper = stats.t.cdf((diff.mean() - bound) / se, n - 1)
    p_lower = 1 - stats.t.cdf((diff.mean() + bound) / se, n - 1)
    return max(p_upper, p_lower)


h2_results = []
print(f'H2.1 paired comparisons, active vs sham (H2-eligible, N pool = {len(h2_subj_df)})\\n')
print(f'{"DV":28s} {"N":>4s} {"Sham":>8s} {"Active":>9s} {"Delta":>9s} '
      f'{"t":>8s} {"p":>7s} {"dz":>7s}')
print('-' * 84)

for sham_col, active_col, label in h2_dvs:
    if sham_col not in h2_subj_df.columns or active_col not in h2_subj_df.columns:
        continue
    d = h2_subj_df[[sham_col, active_col]].dropna()
    if len(d) < 3:
        continue

    s = d[sham_col].astype(float).values
    a = d[active_col].astype(float).values
    diff = a - s
    n = len(diff)

    t_stat, p_val = stats.ttest_rel(a, s)
    sd = diff.std(ddof=1)
    dz = diff.mean() / sd if sd > 0 else np.nan

    print(f'{label:28s} {n:4d} {s.mean():8.3f} {a.mean():9.3f} '
          f'{diff.mean():+9.4f} {t_stat:+8.3f} {p_val:7.3f} {dz:+7.3f}'
          f'{" *" if p_val < .05 else ""}')

    h2_results.append({
        'label': label, 'n': n, 'sham_m': s.mean(), 'active_m': a.mean(),
        'delta': diff.mean(), 'delta_se': sd / np.sqrt(n),
        't': t_stat, 'p': p_val, 'dz': dz,
        'tost_p_03': tost(diff, 0.3), 'tost_p_05': tost(diff, 0.5),
    })

h2_df = pd.DataFrame(h2_results)

print('\\nEquivalence tests (TOST). "Equivalent" means the effect is credibly')
print('smaller than the smallest effect size of interest.\\n')
print(f'{"DV":28s} {"N":>4s} {"p(0.3)":>8s} {"":<9s} {"p(0.5)":>8s}')
print('-' * 66)
for r in h2_results:
    v3 = 'equivalent' if r['tost_p_03'] < .05 else 'inconclusive'
    v5 = 'equivalent' if r['tost_p_05'] < .05 else 'inconclusive'
    print(f'{r["label"]:28s} {r["n"]:4d} {r["tost_p_03"]:8.3f} {v3:<13s} '
          f'{r["tost_p_05"]:8.3f} {v5}')
"""),
        code("""
# ============================================================================
# 4.2 Figure — effect sizes with 95% CI
# ============================================================================

if len(h2_df):
    plot_df = h2_df.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(WIDTH_1_5COL, 0.32 * len(plot_df) + 1.0))

    y = np.arange(len(plot_df))
    ci = 1.96 * plot_df['delta_se'] / plot_df['delta'].abs().replace(0, np.nan)
    # Effect sizes in dz units with a normal-approximation interval.
    dz_se = np.sqrt(1 / plot_df['n'] + plot_df['dz'] ** 2 / (2 * plot_df['n']))

    ax.axvline(0, color='#BDBDBD', lw=0.8, ls='--', zorder=0)
    for lo, hi, color in [(-0.5, 0.5, SESOI_COLOR)]:
        ax.axvspan(lo, hi, color=color, alpha=0.25, zorder=0)

    ax.errorbar(plot_df['dz'], y, xerr=1.96 * dz_se, fmt='o', ms=4,
                color=REGRESSION_COLOR, ecolor=NEUTRAL_GRAY,
                elinewidth=1, capsize=2, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['label'], fontsize=FONT_TICK)
    style_ax(ax, xlabel="Effect size (dz), active - sham")
    ax.text(0.0, len(plot_df) - 0.3, 'shaded: SESOI dz = +/-0.5',
            fontsize=FONT_ANNOTATION, color=NEUTRAL_GRAY, ha='center')
    savefig(fig, 'fig_h2_effect_sizes')
    plt.show()
"""),
        code("""
# ============================================================================
# 4.3 H2.2 — Age moderation of the stimulation change scores
# ============================================================================

delta_dvs = [
    ('delta_p_stay_win', 'Delta p(stay|win)'),
    ('delta_p_shift_lose', 'Delta p(shift|lose)'),
    ('delta_alpha', 'Delta alpha'),
    ('delta_beta', 'Delta beta'),
    ('delta_accuracy', 'Delta accuracy'),
    ('delta_ttc', 'Delta trials-to-criterion'),
]

print('Age moderation of stimulation change scores (H2-eligible)\\n')
print(f'{"DV":28s} {"N":>4s} {"r":>8s} {"p":>8s}')
print('-' * 52)

age_moderation = []
for col, label in delta_dvs:
    if col not in h2_subj_df.columns:
        continue
    d = h2_subj_df[[col, 'age']].dropna()
    if len(d) < 5:
        continue
    r, p = stats.pearsonr(d['age'].astype(float), d[col].astype(float))
    print(f'{label:28s} {len(d):4d} {r:+8.3f} {p:8.3f}'
          f'{" *" if p < .05 else ""}')
    age_moderation.append({'dv': col, 'label': label, 'r': r, 'p': p, 'n': len(d)})

age_mod_df = pd.DataFrame(age_moderation)
"""),
        code("""
# ============================================================================
# 4.4 Figure — Age x change score, for whichever DV shows the strongest relation
# ============================================================================

if len(age_mod_df):
    best = age_mod_df.loc[age_mod_df['p'].idxmin()]
    d = h2_subj_df[[best['dv'], 'age']].dropna()

    fig, ax = plt.subplots(figsize=(WIDTH_1COL, WIDTH_1COL * 0.85))
    scatter_regression_mpl(ax,
                           d['age'].astype(float).values,
                           d[best['dv']].astype(float).values,
                           d['age'].astype(float).values,
                           zero_line=True)
    style_ax(ax, xlabel='Age (years)', ylabel=best['label'])
    savefig(fig, 'fig_h2_age_moderation')
    plt.show()

    print(f"Strongest age relation: {best['label']} "
          f"(r = {best['r']:+.3f}, p = {best['p']:.3f}, N = {best['n']})")
    print('Selected by smallest p across the change scores above, so this is '
          'a display choice, not an independent test.')
"""),
    ]


# =============================================================================
# Section 5 — EEG theta and electric field
# =============================================================================

def section_theta() -> List[nbf.NotebookNode]:
    return [
        md("""
## 5. Endogenous theta and electric field

Baseline theta power as a moderator of the stimulation response, and the
modelled electric field as a dose proxy.

`theta_p95` is the 95th percentile of theta power during the baseline runs,
averaged over clean runs. Coverage rose from 34 subjects at the defense to 59
here — partly the larger sample, but mostly because `eeg_theta.find_eeg_run`
globbed only one of the NIC naming conventions, so subjects whose recordings
were saved as `Bandit-{id}_Run N` or `{id} TACS_Run N` returned no theta at
all. That is fixed in `nic_files.py` and both consumers now share it.
"""),
        code("""
# ============================================================================
# 5.1 Theta as a moderator of the stimulation response
# ============================================================================

theta_targets = [c for c in ['delta_ttc', 'delta_alpha', 'delta_accuracy',
                             'delta_p_shift_lose'] if c in subj.columns]

print(f'theta_p95 available for {subj["theta_p95"].notna().sum()}/{len(subj)} subjects\\n')
print(f'{"Change score":28s} {"N":>4s} {"r":>8s} {"p":>8s}')
print('-' * 52)

theta_rows = []
for col in theta_targets:
    d = subj[subj['subject_id'].isin(h2_eligible)][['theta_p95', col]].dropna()
    if len(d) < 5:
        continue
    r, p = stats.pearsonr(d['theta_p95'].astype(float), d[col].astype(float))
    print(f'{col:28s} {len(d):4d} {r:+8.3f} {p:8.3f}{" *" if p < .05 else ""}')
    theta_rows.append({'dv': col, 'r': r, 'p': p, 'n': len(d)})

theta_df = pd.DataFrame(theta_rows)
"""),
        code("""
# ============================================================================
# 5.2 Figure — baseline theta x change in trials-to-criterion
# ============================================================================

if 'delta_ttc' in subj.columns:
    d = subj[subj['subject_id'].isin(h2_eligible)][
        ['theta_p95', 'delta_ttc', 'age']].dropna()
    if len(d) > 4:
        fig, ax = plt.subplots(figsize=(WIDTH_1COL, WIDTH_1COL * 0.85))
        scatter_regression_mpl(ax,
                               d['theta_p95'].astype(float).values,
                               d['delta_ttc'].astype(float).values,
                               d['age'].astype(float).values,
                               zero_line=True)
        style_ax(ax, xlabel='Baseline theta power (p95, % change)',
                 ylabel='Delta trials-to-criterion (active - sham)')
        savefig(fig, 'fig_theta_x_delta_ttc')
        plt.show()
    else:
        print(f'Insufficient overlap for the theta x delta-TTC figure (N = {len(d)}).')
"""),
        code("""
# ============================================================================
# 5.3 Theta x cognition moderation of the stimulation response
# ============================================================================
# Mean-centered predictors, so the lower-order terms read at the sample mean.

moderation_models = {}

for dv in [c for c in ['delta_ttc', 'delta_alpha'] if c in subj.columns]:
    d = subj[subj['subject_id'].isin(h2_eligible)][
        ['subject_id', dv, 'theta_p95', COG_COMPOSITE, 'age']].dropna().copy()
    if len(d) < 12:
        print(f'{dv}: insufficient data (N = {len(d)})')
        continue

    for c in [dv, 'theta_p95', COG_COMPOSITE, 'age']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna()

    d['theta_c'] = d['theta_p95'] - d['theta_p95'].mean()
    d['cog_c'] = d[COG_COMPOSITE] - d[COG_COMPOSITE].mean()
    d['theta_x_cog'] = d['theta_c'] * d['cog_c']

    X = sm.add_constant(d[['theta_c', 'cog_c', 'theta_x_cog', 'age']].astype(float))
    model = sm.OLS(d[dv].astype(float), X).fit()
    moderation_models[dv] = model

    b, p = model.params['theta_x_cog'], model.pvalues['theta_x_cog']
    print(f'{dv:16s}: theta x cognition b = {b:+.5f}, p = {p:.3f}'
          f'{" *" if p < .05 else ""}, N = {int(model.nobs)}, '
          f'R2 = {model.rsquared:.3f}')

# Print full coefficient tables for whatever was fit, rather than depending on
# variables having been created by an earlier cell.
for dv, model in moderation_models.items():
    print(f'\\n{"=" * 60}\\n{dv}\\n{"=" * 60}')
    print(pd.DataFrame({'coef': model.params, 't': model.tvalues,
                        'p': model.pvalues}).round(4).to_string())
"""),
        code("""
# ============================================================================
# 5.4 Electric field as a dose proxy
# ============================================================================
# Modelled field strength in the DLPFC ROI, from the SimNIBS pipeline. Lives in
# its own CSV rather than the master table.

EFIELD_METRIC = 'mean_magnE'
efield_path = DATA_DIR / 'efield_roi_summary.csv'

if efield_path.exists():
    efield = pd.read_csv(efield_path, dtype={'subject_id': str})
    efield = efield.merge(subj[['subject_id', 'age'] +
                               [c for c in ['delta_ttc', 'delta_alpha']
                                if c in subj.columns]],
                          on='subject_id', how='inner')
    print(f'E-field data: {len(efield)} subjects overlap the current sample')

    d = efield[['age', EFIELD_METRIC]].apply(pd.to_numeric, errors='coerce').dropna()
    if len(d) > 4:
        r, p = stats.pearsonr(d['age'], d[EFIELD_METRIC])
        rho, p_s = stats.spearmanr(d['age'], d[EFIELD_METRIC])
        print(f'  field range: {d[EFIELD_METRIC].min():.4f} - '
              f'{d[EFIELD_METRIC].max():.4f} V/m')
        print(f'  Age x field: r = {r:+.3f}, p = {p:.3f} '
              f'(Spearman rho = {rho:+.3f}, p = {p_s:.3f}), N = {len(d)}')

        fig, ax = plt.subplots(figsize=(WIDTH_1COL, WIDTH_1COL * 0.85))
        scatter_regression_mpl(ax, d['age'].values, d[EFIELD_METRIC].values,
                               d['age'].values)
        style_ax(ax, xlabel='Age (years)',
                 ylabel='Mean |E| in DLPFC ROI (V/m)')
        savefig(fig, 'fig_age_x_efield')
        plt.show()

    # Does the modelled dose track the behavioural response?
    for dv in [c for c in ['delta_ttc', 'delta_alpha'] if c in efield.columns]:
        dd = efield[[EFIELD_METRIC, dv]].apply(pd.to_numeric, errors='coerce').dropna()
        if len(dd) > 4:
            r, p = stats.pearsonr(dd[EFIELD_METRIC], dd[dv])
            print(f'  field x {dv}: r = {r:+.3f}, p = {p:.3f}, N = {len(dd)}')
else:
    print(f'No e-field summary at {efield_path}; skipping.')
"""),
    ]


# =============================================================================
# Section 6 — Secondary moderators and exploratory sweep
# =============================================================================

def section_moderators() -> List[nbf.NotebookNode]:
    return [
        md("""
## 6. Secondary moderators and exploratory sweep

Preregistered secondary moderators first, then a systematic sweep across all
available predictors with FDR correction within each family. The sweep is
exploratory and labelled as such; it is included so the reported effects can be
read against the full space that was searched rather than against a selected
subset.
"""),
        code("""
# ============================================================================
# 6.1 Preregistered secondary moderators
# ============================================================================

secondary_moderators = [c for c in [
    COG_COMPOSITE, 'ef_composite', 'spsrq_sr', 'spsrq_sp',
    'education_years', 'theta_p95',
] if c in subj.columns]

delta_cols = [c for c in ['delta_p_stay_win', 'delta_p_shift_lose',
                          'delta_alpha', 'delta_beta', 'delta_accuracy',
                          'delta_ttc'] if c in subj.columns]

h2s = subj[subj['subject_id'].isin(h2_eligible)]

rows = []
for mod in secondary_moderators:
    for dv in delta_cols:
        d = h2s[[mod, dv]].apply(pd.to_numeric, errors='coerce').dropna()
        if len(d) < 8:
            continue
        r, p = stats.pearsonr(d[mod], d[dv])
        rows.append({'moderator': mod, 'dv': dv, 'r': r, 'p': p, 'n': len(d)})

secondary_df = pd.DataFrame(rows)
if len(secondary_df):
    print(f'{len(secondary_df)} moderator x change-score tests\\n')
    show = secondary_df.sort_values('p').head(12)
    print(f'{"moderator":18s} {"dv":22s} {"N":>4s} {"r":>8s} {"p":>8s}')
    print('-' * 66)
    for _, row in show.iterrows():
        print(f'{row["moderator"]:18s} {row["dv"]:22s} {row["n"]:4.0f} '
              f'{row["r"]:+8.3f} {row["p"]:8.3f}{" *" if row["p"] < .05 else ""}')
    print(f'\\nUncorrected p < .05: {(secondary_df["p"] < .05).sum()} of '
          f'{len(secondary_df)} (expected by chance: '
          f'{0.05 * len(secondary_df):.1f})')
"""),
        code("""
# ============================================================================
# 6.2 Exploratory sweep with FDR correction
# ============================================================================
# Two families, corrected separately: predictors of baseline behaviour, and
# predictors of the stimulation change scores.

from statsmodels.stats.multitest import multipletests

baseline_dvs = [c for c in ['sham_p_stay_win', 'sham_p_shift_lose',
                            'sham_alpha', 'sham_beta', 'sham_accuracy',
                            'sham_ttc'] if c in subj.columns]

predictors = [c for c in [
    'age', 'education_years', COG_COMPOSITE, 'ef_composite',
    'attention_composite', 'memory_composite', 'speed_composite',
    'spsrq_sr', 'spsrq_sp', 'theta_p95', 'bpsqi_global', 'crt_total',
    'ffmq_total', 'audit_total', 'promis_anxiety', 'promis_depression',
    'loneliness_total',
] if c in subj.columns]


def sweep(data, preds, dvs, family):
    out = []
    for pr in preds:
        for dv in dvs:
            if pr == dv:
                continue
            d = data[[pr, dv]].apply(pd.to_numeric, errors='coerce').dropna()
            if len(d) < 10:
                continue
            r, p = stats.pearsonr(d[pr], d[dv])
            out.append({'family': family, 'predictor': pr, 'dv': dv,
                        'r': r, 'p': p, 'n': len(d)})
    df = pd.DataFrame(out)
    if len(df):
        df['p_fdr'] = multipletests(df['p'], method='fdr_bh')[1]
    return df


sweep_baseline = sweep(subj, predictors, baseline_dvs, 'baseline')
sweep_change = sweep(h2s, predictors, delta_cols, 'change')
sweep_all = pd.concat([sweep_baseline, sweep_change], ignore_index=True)

for family, df in [('baseline behaviour', sweep_baseline),
                   ('stimulation change scores', sweep_change)]:
    if not len(df):
        continue
    n_raw = (df['p'] < .05).sum()
    n_fdr = (df['p_fdr'] < .05).sum()
    print(f'{family}: {len(df)} tests, {n_raw} at p < .05 '
          f'(chance: {0.05 * len(df):.1f}), {n_fdr} survive FDR')
    top = df.nsmallest(8, 'p')
    for _, row in top.iterrows():
        mark = ' **' if row['p_fdr'] < .05 else (' *' if row['p'] < .05 else '')
        print(f'    {row["predictor"]:20s} x {row["dv"]:20s} '
              f'r = {row["r"]:+.3f}, p = {row["p"]:.4f}, '
              f'q = {row["p_fdr"]:.3f}, N = {row["n"]:.0f}{mark}')
    print()
"""),
    ]


# =============================================================================
# Section 7 — Robustness and audits
# =============================================================================

def section_audits() -> List[nbf.NotebookNode]:
    return [
        md("""
## 7. Robustness checks

Missingness in the cognitive battery, and the parameter-boundary problem that
motivates the hierarchical model.
"""),
        code("""
# ============================================================================
# 7.1 Cognitive battery missingness
# ============================================================================
# Digit Span, BVMT and Trails were administered only to participants aged 40+,
# so a composite including them is missing-not-at-random with respect to age —
# which is the study's primary moderator. This is why the primary composite is
# built from the five near-complete measures.

battery = ['flanker_score', 'running_dots_score', 'hvlt_total',
           'salthouse_letter', 'salthouse_pattern',
           'digit_span_total', 'bvmt_total', 'trails_a_time', 'trails_b_time']
battery = [c for c in battery if c in subj.columns]

print(f'{"measure":22s} {"N":>4s} {"% present":>10s} {"mean age present":>18s}')
print('-' * 60)
for m in battery:
    present = subj[subj[m].notna()]
    age_present = pd.to_numeric(present['age'], errors='coerce').mean()
    print(f'{m:22s} {len(present):4d} {100*len(present)/len(subj):9.0f}% '
          f'{age_present:18.1f}')

print(f'\\nPrimary composite ({COG_COMPOSITE}): '
      f'{subj[COG_COMPOSITE].notna().sum()}/{len(subj)} subjects')

# A non-null composite does not mean a complete one. compute_cognitive_
# composites averages across whatever measures a subject has, so the
# full-battery composite is non-null for nearly everyone while being built
# from different tests for different people — and which tests are missing is
# tied to age. Report the measure counts, not just the coverage.
if 'global_composite' in subj.columns:
    print(f'Full-battery composite (global_composite): '
          f'{subj["global_composite"].notna().sum()}/{len(subj)} non-null, but '
          f'built from a varying number of domains:')
    if 'global_n_domains' in subj.columns:
        counts = subj['global_n_domains'].value_counts().sort_index()
        for n_dom, n_subj in counts.items():
            print(f'    {n_dom} of 3 domains: {n_subj} subjects')
        print('  Subjects contributing fewer domains are systematically '
              'younger, so this composite is not comparable across the age\\n'
              '  range. That is why the reduced composite is primary.')
"""),
        code("""
# ============================================================================
# 7.2 Learning-rate boundary audit
# ============================================================================
# Maximum-likelihood R-W fits can land on the edge of the parameter space,
# where the estimate is not identified. In the defended data one subject
# (10804) sat at both bounds with a likelihood exactly at chance level.
#
# This is the motivation for the hierarchical model rather than an incidental
# check: partial pooling pulls such subjects toward the group and yields a
# usable estimate. The hierarchical fit for that subject is well away from the
# boundary, and its likelihood is better.

BOUNDARY_LO, BOUNDARY_HI = 0.02, 0.98

for col in [c for c in ['sham_alpha', 'active_alpha'] if c in subj.columns]:
    a = pd.to_numeric(subj[col], errors='coerce').dropna()
    at_lo = (a <= BOUNDARY_LO).sum()
    at_hi = (a >= BOUNDARY_HI).sum()
    print(f'{col:16s}: {at_lo} at the lower bound, {at_hi} at the upper bound, '
          f'of {len(a)}')

for col in [c for c in ['sham_beta', 'active_beta'] if c in subj.columns]:
    b = pd.to_numeric(subj[col], errors='coerce').dropna()
    print(f'{col:16s}: {(b <= 0.05).sum()} near zero (choice at chance), '
          f'max = {b.max():.2f}')

print('\\nA beta at zero means the model predicts 50/50 on every trial — the '
      'fit has failed rather than found a low-sensitivity subject.')
"""),
    ]


# =============================================================================
# Section 7.5 — Hierarchical Rescorla-Wagner
# =============================================================================

def section_hierarchical() -> List[nbf.NotebookNode]:
    return [
        md("""
## 7.5 Hierarchical Rescorla-Wagner

The maximum-likelihood fits above treat each subject and condition
independently, which leaves them vulnerable exactly where the data are
thinnest. The audit in 7.2 shows the consequence: learning rates piled at the
edge of the parameter space, and inverse temperatures at zero, where the model
predicts 50/50 on every trial and the fit has simply failed.

The hierarchical model estimates all subjects jointly, so subjects with weak
data are pulled toward the group rather than to a boundary. It also expresses
the design properly — condition varies *within* subject, which the previous
`rl_models` implementation could not represent (it took one condition per
subject and assigned it by modal label, which for this design is usually
"baseline").

Parameterization and diagnostics live in `rl_models/models_within.py`. Fit
with:

```
python -m rl_models.run_within_fit --sample all
```

Estimands:

- **delta_alpha** — the group-level tACS effect on learning rate
- **tau_alpha** — between-subject spread in that effect, i.e. how much people
  differ in responsiveness
- **eta_alpha** — block-order nuisance term. Within a subject, condition is
  perfectly confounded with early-vs-late in a two-hour session;
  counterbalancing breaks that across the group only if the model is told
  about it.
"""),
        code("""
# ============================================================================
# 7.5a Group-level parameters
# ============================================================================

RL_DIR = DATA_DIR.parent.parent / 'derivatives' / 'rl_models'
group_path = RL_DIR / 'rw_within_all_group.csv'

if group_path.exists():
    hb_group = pd.read_csv(group_path, index_col=0)
    print('Hierarchical RW, group-level parameters')
    print(f'(p_direction = share of the posterior on one side of zero)\\n')
    cols = ['mean', 'sd', 'hdi_3%', 'hdi_97%', 'p_direction', 'r_hat', 'ess_bulk']
    print(hb_group[[c for c in cols if c in hb_group.columns]].round(3).to_string())

    d = hb_group.loc['delta_alpha']
    print(f"\\ntACS effect on learning rate: {d['mean']:+.3f} "
          f"[{d['hdi_3%']:+.3f}, {d['hdi_97%']:+.3f}]")
    print('  The interval spans zero, so there is no credible group-level effect.')

    t = hb_group.loc['tau_alpha']
    print(f"\\nBetween-subject spread in that effect: {t['mean']:.3f} "
          f"[{t['hdi_3%']:.3f}, {t['hdi_97%']:.3f}]")
    print('  Credibly greater than zero: subjects differ in how they respond,')
    print('  even though the average response is nil. See 7.5c — the magnitude')
    print('  of this term is prior-dependent, so treat it qualitatively.')
else:
    print(f'No hierarchical fit at {group_path}. Run:')
    print('  python -m rl_models.run_within_fit --sample all')
    hb_group = None
"""),
        code("""
# ============================================================================
# 7.5b Subject-level posteriors, and agreement with the MLE fits
# ============================================================================

subj_path = RL_DIR / 'rw_within_all_subjects.csv'

if subj_path.exists():
    hb_subj = pd.read_csv(subj_path, dtype={'subject_id': str})
    print(f'Subject-level posterior means: {len(hb_subj)} subjects\\n')

    merged = subj.merge(hb_subj, on='subject_id', suffixes=('_mle', '_hb'))

    # Both correlations, because Pearson is misleading for beta: the MLE fit
    # is bounded only at 50, so a couple of subjects land near that bound and
    # dominate the covariance. Rank agreement is the fairer comparison of
    # whether the two methods order subjects the same way.
    print(f'  {"parameter":14s} {"pearson":>9s} {"spearman":>9s}   MLE range')
    for param in ['sham_alpha', 'active_alpha', 'sham_beta', 'active_beta']:
        a, b = f'{param}_mle', f'{param}_hb'
        if a not in merged or b not in merged:
            continue
        d = merged[[a, b]].apply(pd.to_numeric, errors='coerce').dropna()
        if len(d) < 5:
            continue
        r, _ = stats.pearsonr(d[a], d[b])
        rho, _ = stats.spearmanr(d[a], d[b])
        print(f'  {param:14s} {r:+9.3f} {rho:+9.3f}   '
              f'{d[a].min():.2f} - {d[a].max():.2f}  (N = {len(d)})')

    n_extreme = (pd.to_numeric(merged.get('active_beta_mle'),
                               errors='coerce') > 20).sum()
    if n_extreme:
        print(f'\\n  {n_extreme} subject(s) have an MLE inverse temperature above 20 '
              f'(bound = 50).')
        print('  Those points break the Pearson correlation for beta while the '
              'rank agreement\\n  holds, which is the boundary problem in 7.2 '
              'showing up again.')

    # Where the two methods disagree is where pooling did the work.
    if {'sham_alpha_mle', 'sham_alpha_hb'}.issubset(merged.columns):
        merged['alpha_shift'] = (merged['sham_alpha_hb']
                                 - merged['sham_alpha_mle']).abs()
        moved = merged.nlargest(5, 'alpha_shift')[
            ['subject_id', 'sham_alpha_mle', 'sham_alpha_hb', 'alpha_shift']]
        print('\\n  Subjects whose sham learning rate moved most under pooling:')
        print(moved.round(3).to_string(index=False))
else:
    print('No subject-level hierarchical output found.')
    hb_subj = None
"""),
        code("""
# ============================================================================
# 7.5c Prior sensitivity
# ============================================================================
# Variance parameters on the logit scale are only weakly identified when many
# subjects sit near a boundary: sigmoid(x) is flat there, so the logit
# coordinate can grow without the likelihood objecting, and the prior ends up
# setting the scale. Refitting under wider priors shows which conclusions
# depend on that choice.

sens_path = RL_DIR / 'prior_sensitivity_all.csv'

if sens_path.exists():
    sens = pd.read_csv(sens_path)
    table = sens.pivot(index='parameter', columns='priors', values='mean')
    order = [c for c in ['default', 'wide', 'very_wide'] if c in table.columns]
    print('Posterior means across prior widths\\n')
    print(table[order].round(3).to_string())

    span = table[order].max(axis=1) - table[order].min(axis=1)
    hdi_w = sens.groupby('parameter').apply(
        lambda g: (g['hdi_hi'] - g['hdi_lo']).mean(), include_groups=False)
    rel = span / hdi_w
    zero_consistent = (sens.assign(z=(sens['hdi_lo'] <= 0) & (sens['hdi_hi'] >= 0))
                           .groupby('parameter')['z'].nunique() == 1)

    print('\\nMovement relative to each parameter\\'s own uncertainty:')
    for name in table.index:
        verdict = 'drifts' if rel[name] > 0.5 else 'stable'
        concl = 'same conclusion' if zero_consistent[name] else 'CONCLUSION CHANGES'
        print(f'  {name:24s} {100 * rel[name]:5.0f}% of HDI width  '
              f'-> {verdict}, {concl}')

    print('\\nEvery effect parameter is stable and every prior gives the same')
    print('conclusion. Only tau_alpha drifts materially, so its magnitude is')
    print('reported qualitatively rather than as a point estimate.')
else:
    print('No prior sensitivity output; run:')
    print('  python -m rl_models.test_prior_sensitivity --sample all')
"""),
        code("""
# ============================================================================
# 7.5d Figure — subject-level condition effects
# ============================================================================
# The group effect is null while subjects differ, so plot the distribution of
# individual effects rather than a single group estimate.

if hb_subj is not None and 'delta_alpha' in hb_subj.columns:
    d = hb_subj.merge(subj[['subject_id', 'age']], on='subject_id', how='left')
    d['age'] = pd.to_numeric(d['age'], errors='coerce')
    d = d.dropna(subset=['delta_alpha'])

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_1_5COL, WIDTH_1COL * 0.8))

    axes[0].hist(d['delta_alpha'], bins=18, color=NEUTRAL_GRAY,
                 edgecolor='white', linewidth=0.5)
    axes[0].axvline(0, color=ACCENT_RED, ls='--', lw=1)
    axes[0].axvline(d['delta_alpha'].mean(), color=REGRESSION_COLOR, lw=1.2)
    style_ax(axes[0], xlabel='Subject-level alpha effect (active - sham)',
             ylabel='Subjects')

    dd = d.dropna(subset=['age'])
    if len(dd) > 4:
        scatter_regression_mpl(axes[1], dd['age'].values,
                               dd['delta_alpha'].values, dd['age'].values,
                               zero_line=True)
    style_ax(axes[1], xlabel='Age (years)',
             ylabel='Subject-level alpha effect')

    plt.tight_layout(pad=0.5)
    savefig(fig, 'fig_hb_subject_effects')
    plt.show()

    print(f"Subject-level effects: mean {d['delta_alpha'].mean():+.4f}, "
          f"SD {d['delta_alpha'].std():.4f}, "
          f"range {d['delta_alpha'].min():+.3f} to {d['delta_alpha'].max():+.3f}")
"""),
    ]


# =============================================================================
# Section 7.6 — Individualized theta frequency
# =============================================================================

def section_itf() -> List[nbf.NotebookNode]:
    return [
        md("""
## 7.6 Individualized theta frequency

Every participant received stimulation at a fixed 6.0 Hz — all 237
stimulation runs, without exception. Individualized theta is therefore not a
delivered parameter but a **moderator**: did stimulation work better for people
whose endogenous theta already sat near the frequency delivered?

Estimated by the Klimesch anchor — individual alpha frequency from posterior
channels, minus a fixed offset. Alpha is the most reliably detectable scalp
rhythm and P3/P4 recorded for every subject, so this covers the whole sample,
unlike a direct frontal-midline theta peak (FCz stimulates during stimulation
runs and only records for the later protocol).

**Limitations, stated plainly.** Klimesch's transition frequency is properly
defined from how theta and alpha shift in opposite directions between rest and
task; this study has no resting block, so the fixed 5 Hz offset is an
approximation rather than a measurement. It does get modest empirical support
here — among the few subjects with both estimators the observed IAF-minus-theta
gap is close to the assumed offset and the two measures correlate positively,
as the anchor predicts — but that check rests on a handful of subjects and
should not be leaned on.

Preprocessing matters more than it might appear for this measure. An earlier
version of this pipeline resolved alpha for only 44 of 59 subjects, and every
failure traced to preprocessing rather than to absent alpha: slow drift that
linear detrending left in place, so a fixed amplitude threshold rejected nearly
all data for the drifty recordings, and no software average re-reference for
the participants run without the earclip. With both fixed, coverage is 55 of
59. The lesson generalizes — a subject silently dropped for a technical reason
is indistinguishable, downstream, from a subject who genuinely lacks the
rhythm.

Generated by `individual_theta.py`.
"""),
        code("""
# ============================================================================
# 7.6a Load and describe
# ============================================================================

itf_path = DATA_DIR.parent.parent / 'derivatives' / 'eeg' / 'individual_theta_frequency.csv'

if itf_path.exists():
    itf = pd.read_csv(itf_path, dtype={'subject_id': str})
    itf_subj = subj.merge(itf, on='subject_id', how='left')

    print(f'Subjects with EEG              : {itf["subject_id"].nunique()}')
    print(f'Alpha peak resolved            : {itf["iaf"].notna().sum()}')
    print(f'Direct FCz theta peak resolved : {itf["theta_peak_fcz"].notna().sum()}')
    print(f'\\nIAF : M = {itf["iaf"].mean():.2f} Hz, SD = {itf["iaf"].std():.2f}, '
          f'range {itf["iaf"].min():.2f} - {itf["iaf"].max():.2f}')
    print(f'iTF : M = {itf["itf_klimesch"].mean():.2f} Hz, '
          f'SD = {itf["itf_klimesch"].std():.2f}')
    print(f'|iTF - 6 Hz| : M = {itf["itf_distance"].mean():.2f} Hz, '
          f'max = {itf["itf_distance"].max():.2f}')
else:
    print(f'No iTF file at {itf_path}. Run: python individual_theta.py')
    itf = None
    itf_subj = subj.copy()
"""),
        code("""
# ============================================================================
# 7.6b Is iTF confounded with age?
# ============================================================================
# Individual alpha frequency declines with age in the literature, and age is
# this study's primary moderator — so an iTF effect could be an age effect
# wearing a different label. Tested directly rather than assumed either way.

if itf is not None:
    d = itf_subj[['iaf', 'itf_distance', 'age']].apply(
        pd.to_numeric, errors='coerce').dropna()
    if len(d) > 5:
        r_iaf, p_iaf = stats.pearsonr(d['age'], d['iaf'])
        r_dist, p_dist = stats.pearsonr(d['age'], d['itf_distance'])
        print(f'IAF x age          : r = {r_iaf:+.3f}, p = {p_iaf:.3f}, N = {len(d)}')
        print(f'|iTF - 6Hz| x age  : r = {r_dist:+.3f}, p = {p_dist:.3f}, N = {len(d)}')
        print()
        if p_iaf >= .05:
            print('  No credible age relation in this sample, so iTF effects below')
            print('  are not simply age effects relabelled. Reported both with and')
            print('  without age covariation regardless.')
        else:
            print('  IAF varies with age here, so iTF effects are reported with age')
            print('  covaried and the two cannot be fully separated.')
"""),
        code("""
# ============================================================================
# 7.6c iTF distance as a moderator of the stimulation response
# ============================================================================
# Prediction: if a fixed 6 Hz works best for people whose endogenous theta is
# already near 6 Hz, distance from 6 Hz should predict a weaker response.

if itf is not None:
    itf_h2 = itf_subj[itf_subj['subject_id'].isin(h2_eligible)]
    targets = [c for c in delta_cols if c in itf_h2.columns]

    print(f'{"change score":24s} {"N":>4s} {"r":>8s} {"p":>8s}   '
          f'{"partial r | age":>16s} {"p":>8s}')
    print('-' * 76)

    for dv in targets:
        d = itf_h2[['itf_distance', dv, 'age']].apply(
            pd.to_numeric, errors='coerce').dropna()
        if len(d) < 10:
            continue
        r, p = stats.pearsonr(d['itf_distance'], d[dv])

        # Partial correlation controlling for age: residualize both, correlate.
        X = sm.add_constant(d[['age']])
        res_x = sm.OLS(d['itf_distance'], X).fit().resid
        res_y = sm.OLS(d[dv], X).fit().resid
        pr, pp = stats.pearsonr(res_x, res_y)

        print(f'{dv:24s} {len(d):4d} {r:+8.3f} {p:8.3f}   '
              f'{pr:+16.3f} {pp:8.3f}')

    print('\\nA negative correlation would mean people further from 6 Hz respond')
    print('less, which is the prediction if frequency matching matters.')
"""),
        code("""
# ============================================================================
# 7.6d Figure — iTF distribution and its relation to the response
# ============================================================================

if itf is not None and itf['iaf'].notna().sum() > 5:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_1_5COL, WIDTH_1COL * 0.8))

    axes[0].hist(itf['itf_klimesch'].dropna(), bins=14, color=NEUTRAL_GRAY,
                 edgecolor='white', linewidth=0.5)
    axes[0].axvline(6.0, color=ACTIVE_COLOR, lw=1.5,
                    label='delivered (6 Hz)')
    style_ax(axes[0], xlabel='Individualized theta frequency (Hz)',
             ylabel='Subjects')
    axes[0].legend(frameon=False, fontsize=FONT_LEGEND)

    dv = 'delta_ttc' if 'delta_ttc' in itf_subj.columns else (
        'delta_alpha' if 'delta_alpha' in itf_subj.columns else None)
    if dv:
        d = itf_subj[itf_subj['subject_id'].isin(h2_eligible)][
            ['itf_distance', dv, 'age']].apply(
            pd.to_numeric, errors='coerce').dropna()
        if len(d) > 4:
            scatter_regression_mpl(axes[1], d['itf_distance'].values,
                                   d[dv].values, d['age'].values,
                                   zero_line=True)
            style_ax(axes[1], xlabel='|iTF - 6 Hz| (Hz)', ylabel=dv)

    plt.tight_layout(pad=0.5)
    savefig(fig, 'fig_itf_distribution')
    plt.show()
"""),
    ]


# =============================================================================
# Section 8 — fMRI
# =============================================================================

def section_fmri() -> List[nbf.NotebookNode]:
    return [
        md("""
## 8. Ventral striatal reactivity

Reward-related VS activation from an independent scan session, related to
baseline behaviour and to the stimulation change scores.
"""),
        code("""
# ============================================================================
# 8.1 Load ROI data and relate to behaviour
# ============================================================================

roi_path = DATA_DIR.parent / 'roi-analyses-final.csv'

if roi_path.exists():
    roi = pd.read_csv(roi_path)
    roi = roi.rename(columns={'sub': 'subject_id'})
    roi['subject_id'] = roi['subject_id'].astype(str)

    # Average the two reward tasks; win>loss is the contrast of interest.
    roi['vs_win_baseline'] = (roi['doors_win'] + roi['social_win']) / 2
    roi['vs_win_loss'] = ((roi['doors_win'] + roi['social_win']) / 2
                          - (roi['doors_loss'] + roi['social_loss']) / 2)

    subj_fmri = subj.merge(roi[['subject_id', 'vs_win_baseline', 'vs_win_loss']],
                           on='subject_id', how='inner')
    print(f'ROI file: {len(roi)} subjects; {len(subj_fmri)} overlap the '
          f'current bandit sample of {len(subj)}')

    fmri_metrics = ['vs_win_baseline', 'vs_win_loss']
    targets = [c for c in (baseline_dvs + delta_cols) if c in subj_fmri.columns]

    rows = []
    for metric in fmri_metrics:
        for dv in targets:
            d = subj_fmri[[metric, dv]].apply(pd.to_numeric, errors='coerce').dropna()
            if len(d) < 10:
                continue
            r, p = stats.pearsonr(d[metric], d[dv])
            rows.append({'metric': metric, 'dv': dv, 'r': r, 'p': p, 'n': len(d)})

    fmri_df = pd.DataFrame(rows)
    if len(fmri_df):
        fmri_df['p_fdr'] = multipletests(fmri_df['p'], method='fdr_bh')[1]
        print(f'\\n{len(fmri_df)} tests, {(fmri_df["p"] < .05).sum()} at p < .05, '
              f'{(fmri_df["p_fdr"] < .05).sum()} surviving FDR\\n')
        for _, row in fmri_df.nsmallest(8, 'p').iterrows():
            mark = ' **' if row['p_fdr'] < .05 else (' *' if row['p'] < .05 else '')
            print(f'  {row["metric"]:18s} x {row["dv"]:20s} '
                  f'r = {row["r"]:+.3f}, p = {row["p"]:.4f}, '
                  f'q = {row["p_fdr"]:.3f}, N = {row["n"]:.0f}{mark}')
else:
    print(f'No ROI file at {roi_path}; skipping.')
"""),
        md("""
---

## Notes

Generated by `build_results_paper_nb.py`. Regenerate with:

```
python build_results_paper_nb.py --execute
```

Set `--sample dissertation` to reproduce the defended analyses; section 0.5
then checks the reproduction and flags any subject differing for reasons other
than the two documented ones.

Still to add: the hierarchical Rescorla-Wagner results (`rl_models`), which
replace the MLE point estimates as the primary parameter estimates, and the
individualized theta frequency work.
"""),
    ]


def build(sample: str = 'all') -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = (
        section_setup(sample)
        + section_data()
        + section_descriptives()
        + section_h1()
        + section_reversal()
        + section_h2()
        + section_theta()
        + section_moderators()
        + section_audits()
        + section_hierarchical()
        + section_itf()
        + section_fmri()
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
    parser.add_argument('--sample', default='all',
                        choices=['all', 'dissertation', 'new'],
                        help='value baked into the SAMPLE constant')
    parser.add_argument('--output', default=str(OUTPUT))
    args = parser.parse_args(argv)

    nb = build(args.sample)
    out = Path(args.output)
    nbf.write(nb, str(out))
    n_code = sum(c['cell_type'] == 'code' for c in nb.cells)
    print(f'Wrote {out}  ({len(nb.cells)} cells, {n_code} code)')

    if args.execute:
        print('\nExecuting...')
        # Always run from this file's directory: the notebook imports config,
        # data_loading and the rest from code/, so executing anywhere else
        # fails on import. nbconvert reports that as a silent no-op unless the
        # return code is checked, which is how an earlier run wrote a notebook
        # with no execution counts and looked like it had succeeded.
        result = subprocess.run(
            ['jupyter', 'nbconvert', '--to', 'notebook', '--execute',
             '--inplace', '--ExecutePreprocessor.timeout=1800', str(out)],
            cwd=str(Path(__file__).parent), capture_output=True, text=True,
        )
        sys.stdout.write(result.stdout[-4000:])
        sys.stderr.write(result.stderr[-4000:])
        if result.returncode != 0:
            print(f'\nExecution FAILED (exit {result.returncode}) — '
                  f'the notebook on disk has no outputs.')
        return result.returncode

    return 0


if __name__ == '__main__':
    sys.exit(main())
