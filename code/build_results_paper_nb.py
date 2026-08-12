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


def build(sample: str = 'all') -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = (
        section_setup(sample)
        + section_data()
        + section_descriptives()
        + section_h1()
        + section_reversal()
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
