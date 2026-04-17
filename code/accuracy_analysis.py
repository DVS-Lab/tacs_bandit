"""
accuracy_analysis.py — Accuracy and win rate analysis for tACS Bandit study

Computes and visualizes accuracy and win rate metrics at multiple levels:
- Run-level performance (for QC and exclusion decisions)
- Condition-level comparisons (sham vs active)
- Learning curves (within-run and across contingencies)
- Reversal-locked performance
- Age-related effects

Key outputs:
- Run-level accuracy/win rate summaries
- Counterbalance effects (order A vs B)
- Learning curves by contingency phase
- Stimulation effect tests on accuracy metrics
- Age correlations with performance

Terminology:
- Accuracy: % trials choosing the higher-probability option (correct == True)
- Win rate: % trials receiving reward (reward == True)
- Optimal: theoretical max given 80/20 contingencies (~80%)
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Optional, Dict, List, Tuple

from config import (
    SUBJECT_INFO,
    COLOR_SHAM,
    COLOR_ACTIVE,
    COLOR_GOLD,
    COLOR_GREEN,
    COLOR_RED,
    AGE_MIN,
    AGE_MAX,
    AGE_COLORSCALE,
    PLOTLY_TEMPLATE,
    FONT_FAMILY,
)


# =============================================================================
# Constants
# =============================================================================

# Theoretical optimal accuracy given 80/20 reward contingencies
OPTIMAL_ACCURACY = 0.80

# Chance-level accuracy
CHANCE_ACCURACY = 0.50

# Run structure
RUNS_PER_SESSION = 8
SHAM_RUNS_A = [1, 2, 3, 4]  # Counterbalance A: sham first
ACTIVE_RUNS_A = [5, 6, 7, 8]
SHAM_RUNS_B = [5, 6, 7, 8]  # Counterbalance B: active first
ACTIVE_RUNS_B = [1, 2, 3, 4]


# =============================================================================
# Age Color Utility
# =============================================================================

def _get_age_colormap():
    """Get matplotlib colormap for age gradient (blue → gold)."""
    import matplotlib.colors as mcolors
    return mcolors.LinearSegmentedColormap.from_list('blue_gold', ['#1565C0', '#FFB300'])


def _age_to_rgb(age: float, age_min: float = AGE_MIN, age_max: float = AGE_MAX) -> str:
    """Convert age to RGB color string for Plotly."""
    cmap = _get_age_colormap()
    norm = np.clip((age - age_min) / (age_max - age_min), 0, 1)
    rgba = cmap(norm)
    return f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'


# =============================================================================
# Data Preparation Utilities
# =============================================================================

def _convert_bool_column(series: pd.Series) -> pd.Series:
    """
    Convert boolean-like column to numeric 0/1.
    
    Handles: True/False objects, 'True'/'False' strings, 1/0 numeric, np.bool_
    """
    if series.dtype == object:
        return series.apply(lambda x: 1 if (x is True or str(x).upper() == 'TRUE') else 0)
    elif series.dtype == bool:
        return series.astype(int)
    else:
        return pd.to_numeric(series, errors='coerce').fillna(0).astype(int)


# =============================================================================
# Run-Level Accuracy Computation
# =============================================================================

def compute_run_level_accuracy(
    data: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute accuracy and win rate for each subject × run.
    
    Parameters
    ----------
    data : DataFrame
        Trial-level data with subject_id, run, correct, reward columns
    verbose : bool
        Print summary statistics
    
    Returns
    -------
    DataFrame with columns:
        subject_id, run, condition, counterbalance, age,
        n_trials, n_valid, accuracy, win_rate, 
        above_chance, near_optimal
    """
    df = data.copy()
    
    # Convert boolean columns to numeric
    df['correct_num'] = _convert_bool_column(df['correct'])
    df['reward_num'] = _convert_bool_column(df['reward'])
    
    # Get counterbalance and age lookups
    cb_lookup = {s: info['counterbalance'] for s, info in SUBJECT_INFO.items()}
    age_lookup = df.groupby('subject_id')['age'].first().to_dict()
    
    # Compute run-level stats
    run_stats = []
    
    for (subj, run), group in df.groupby(['subject_id', 'run']):
        # Filter to valid trials (choice made)
        valid = group[group['choice'].notna()]
        
        n_trials = len(group)
        n_valid = len(valid)
        
        if n_valid == 0:
            accuracy = np.nan
            win_rate = np.nan
        else:
            accuracy = valid['correct_num'].mean()
            win_rate = valid['reward_num'].mean()
        
        # Get condition from data
        condition = group['condition'].iloc[0] if 'condition' in group.columns else 'unknown'
        
        run_stats.append({
            'subject_id': subj,
            'run': run,
            'condition': condition,
            'counterbalance': cb_lookup.get(subj, 'unknown'),
            'age': age_lookup.get(subj, np.nan),
            'n_trials': n_trials,
            'n_valid': n_valid,
            'accuracy': accuracy,
            'win_rate': win_rate,
            'above_chance': accuracy > CHANCE_ACCURACY if not np.isnan(accuracy) else False,
            'near_optimal': accuracy >= (OPTIMAL_ACCURACY - 0.1) if not np.isnan(accuracy) else False,
        })
    
    run_df = pd.DataFrame(run_stats)
    
    if verbose:
        print("="*60)
        print("RUN-LEVEL ACCURACY SUMMARY")
        print("="*60)
        print(f"\nTotal runs: {len(run_df)}")
        print(f"Subjects: {run_df['subject_id'].nunique()}")
        print(f"\nOverall accuracy: M = {run_df['accuracy'].mean():.3f}, SD = {run_df['accuracy'].std():.3f}")
        print(f"Overall win rate: M = {run_df['win_rate'].mean():.3f}, SD = {run_df['win_rate'].std():.3f}")
        print(f"\nRuns above chance (>{CHANCE_ACCURACY:.0%}): {run_df['above_chance'].sum()} / {len(run_df)} ({100*run_df['above_chance'].mean():.1f}%)")
        print(f"Runs near optimal (>={OPTIMAL_ACCURACY-0.1:.0%}): {run_df['near_optimal'].sum()} / {len(run_df)} ({100*run_df['near_optimal'].mean():.1f}%)")
    
    return run_df


def compute_condition_level_accuracy(
    data: pd.DataFrame,
    conditions: List[str] = ['sham', 'active'],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute accuracy and win rate for each subject × condition.
    
    Aggregates across runs within each condition.
    
    Parameters
    ----------
    data : DataFrame
        Trial-level data
    conditions : list
        Conditions to include
    verbose : bool
        Print summary
    
    Returns
    -------
    DataFrame with subject × condition accuracy/win_rate
    """
    df = data[data['condition'].isin(conditions)].copy()
    
    df['correct_num'] = _convert_bool_column(df['correct'])
    df['reward_num'] = _convert_bool_column(df['reward'])
    
    age_lookup = df.groupby('subject_id')['age'].first().to_dict()
    cb_lookup = {s: info['counterbalance'] for s, info in SUBJECT_INFO.items()}
    
    cond_stats = []
    
    for (subj, cond), group in df.groupby(['subject_id', 'condition']):
        valid = group[group['choice'].notna()]
        
        if len(valid) == 0:
            continue
        
        cond_stats.append({
            'subject_id': subj,
            'condition': cond,
            'counterbalance': cb_lookup.get(subj, 'unknown'),
            'age': age_lookup.get(subj, np.nan),
            'n_trials': len(valid),
            'n_runs': group['run'].nunique(),
            'accuracy': valid['correct_num'].mean(),
            'win_rate': valid['reward_num'].mean(),
        })
    
    cond_df = pd.DataFrame(cond_stats)
    
    if verbose:
        print("\n" + "="*60)
        print("CONDITION-LEVEL ACCURACY SUMMARY")
        print("="*60)
        
        for cond in conditions:
            cond_data = cond_df[cond_df['condition'] == cond]
            print(f"\n{cond.upper()}:")
            print(f"  N = {len(cond_data)}")
            print(f"  Accuracy: M = {cond_data['accuracy'].mean():.3f}, SD = {cond_data['accuracy'].std():.3f}")
            print(f"  Win rate: M = {cond_data['win_rate'].mean():.3f}, SD = {cond_data['win_rate'].std():.3f}")
    
    return cond_df


# =============================================================================
# Counterbalance Analysis
# =============================================================================

def analyze_counterbalance_effects(
    run_df: pd.DataFrame,
    verbose: bool = True
) -> Dict:
    """
    Test whether counterbalance order (A vs B) affects accuracy.
    
    Counterbalance A: sham runs 1-4, active runs 5-8
    Counterbalance B: active runs 1-4, sham runs 5-8
    
    Tests:
    - Overall accuracy difference between counterbalance groups
    - Run × counterbalance interaction (practice effects)
    - First-half vs second-half accuracy by counterbalance
    
    Parameters
    ----------
    run_df : DataFrame
        Output from compute_run_level_accuracy()
    verbose : bool
        Print results
    
    Returns
    -------
    dict with test results
    """
    results = {}
    
    # Overall accuracy by counterbalance
    cb_means = run_df.groupby('counterbalance')['accuracy'].mean()
    
    # Subject-level means by counterbalance
    subj_cb = run_df.groupby(['subject_id', 'counterbalance'])['accuracy'].mean().reset_index()
    
    cb_a = subj_cb[subj_cb['counterbalance'] == 'A']['accuracy']
    cb_b = subj_cb[subj_cb['counterbalance'] == 'B']['accuracy']
    
    if len(cb_a) >= 3 and len(cb_b) >= 3:
        t_cb, p_cb = stats.ttest_ind(cb_a, cb_b)
        d_cb = (cb_a.mean() - cb_b.mean()) / np.sqrt((cb_a.var() + cb_b.var()) / 2)
        
        results['overall'] = {
            'A_mean': cb_a.mean(),
            'A_sd': cb_a.std(),
            'A_n': len(cb_a),
            'B_mean': cb_b.mean(),
            'B_sd': cb_b.std(),
            'B_n': len(cb_b),
            't': t_cb,
            'p': p_cb,
            'd': d_cb,
        }
    
    # First half (runs 1-4) vs second half (runs 5-8)
    run_df = run_df.copy()
    run_df['half'] = run_df['run'].apply(lambda x: 'first' if x <= 4 else 'second')
    
    half_stats = run_df.groupby(['subject_id', 'counterbalance', 'half'])['accuracy'].mean().reset_index()
    
    # For counterbalance A: first half = sham, second half = active
    # For counterbalance B: first half = active, second half = sham
    # Test whether the "first half" advantage differs by counterbalance
    
    first_half = half_stats[half_stats['half'] == 'first']
    second_half = half_stats[half_stats['half'] == 'second']
    
    merged = first_half.merge(second_half, on=['subject_id', 'counterbalance'], suffixes=('_first', '_second'))
    merged['improvement'] = merged['accuracy_second'] - merged['accuracy_first']
    
    # Does improvement differ by counterbalance?
    improve_a = merged[merged['counterbalance'] == 'A']['improvement']
    improve_b = merged[merged['counterbalance'] == 'B']['improvement']
    
    if len(improve_a) >= 3 and len(improve_b) >= 3:
        t_improve, p_improve = stats.ttest_ind(improve_a, improve_b)
        
        results['practice_by_counterbalance'] = {
            'A_improvement': improve_a.mean(),
            'B_improvement': improve_b.mean(),
            't': t_improve,
            'p': p_improve,
        }
    
    # Condition-specific: does sham accuracy differ based on when it occurred?
    # Counterbalance A: sham is runs 1-4 (first)
    # Counterbalance B: sham is runs 5-8 (second)
    sham_runs = run_df[run_df['condition'] == 'sham']
    sham_by_subj_cb = sham_runs.groupby(['subject_id', 'counterbalance'])['accuracy'].mean().reset_index()
    
    sham_first = sham_by_subj_cb[sham_by_subj_cb['counterbalance'] == 'A']['accuracy']  # sham first
    sham_second = sham_by_subj_cb[sham_by_subj_cb['counterbalance'] == 'B']['accuracy']  # sham second
    
    if len(sham_first) >= 3 and len(sham_second) >= 3:
        t_sham, p_sham = stats.ttest_ind(sham_first, sham_second)
        
        results['sham_order_effect'] = {
            'sham_first_mean': sham_first.mean(),
            'sham_second_mean': sham_second.mean(),
            't': t_sham,
            'p': p_sham,
        }
    
    # Same for active
    active_runs = run_df[run_df['condition'] == 'active']
    active_by_subj_cb = active_runs.groupby(['subject_id', 'counterbalance'])['accuracy'].mean().reset_index()
    
    active_first = active_by_subj_cb[active_by_subj_cb['counterbalance'] == 'B']['accuracy']  # active first
    active_second = active_by_subj_cb[active_by_subj_cb['counterbalance'] == 'A']['accuracy']  # active second
    
    if len(active_first) >= 3 and len(active_second) >= 3:
        t_active, p_active = stats.ttest_ind(active_first, active_second)
        
        results['active_order_effect'] = {
            'active_first_mean': active_first.mean(),
            'active_second_mean': active_second.mean(),
            't': t_active,
            'p': p_active,
        }
    
    if verbose:
        print("\n" + "="*60)
        print("COUNTERBALANCE EFFECTS ANALYSIS")
        print("="*60)
        
        if 'overall' in results:
            r = results['overall']
            sig = '*' if r['p'] < 0.05 else ''
            print(f"\nOverall Accuracy by Counterbalance:")
            print(f"  A (sham first): M = {r['A_mean']:.3f}, SD = {r['A_sd']:.3f}, n = {r['A_n']}")
            print(f"  B (active first): M = {r['B_mean']:.3f}, SD = {r['B_sd']:.3f}, n = {r['B_n']}")
            print(f"  t = {r['t']:.2f}, p = {r['p']:.3f}{sig}, d = {r['d']:.2f}")
        
        if 'practice_by_counterbalance' in results:
            r = results['practice_by_counterbalance']
            sig = '*' if r['p'] < 0.05 else ''
            print(f"\nPractice Effect (2nd half - 1st half) by Counterbalance:")
            print(f"  A: Δ = {r['A_improvement']:+.3f}")
            print(f"  B: Δ = {r['B_improvement']:+.3f}")
            print(f"  t = {r['t']:.2f}, p = {r['p']:.3f}{sig}")
        
        if 'sham_order_effect' in results:
            r = results['sham_order_effect']
            sig = '*' if r['p'] < 0.05 else ''
            print(f"\nSham Accuracy by Order:")
            print(f"  Sham first (CB=A): M = {r['sham_first_mean']:.3f}")
            print(f"  Sham second (CB=B): M = {r['sham_second_mean']:.3f}")
            print(f"  t = {r['t']:.2f}, p = {r['p']:.3f}{sig}")
        
        if 'active_order_effect' in results:
            r = results['active_order_effect']
            sig = '*' if r['p'] < 0.05 else ''
            print(f"\nActive Accuracy by Order:")
            print(f"  Active first (CB=B): M = {r['active_first_mean']:.3f}")
            print(f"  Active second (CB=A): M = {r['active_second_mean']:.3f}")
            print(f"  t = {r['t']:.2f}, p = {r['p']:.3f}{sig}")
    
    return results


# =============================================================================
# Stimulation Effects on Accuracy
# =============================================================================

def test_stimulation_effects_accuracy(
    cond_df: pd.DataFrame,
    verbose: bool = True
) -> Dict:
    """
    Test whether theta-tACS affects accuracy and win rate.
    
    Uses paired t-tests comparing sham vs active within subjects.
    
    Parameters
    ----------
    cond_df : DataFrame
        Output from compute_condition_level_accuracy()
    verbose : bool
        Print results
    
    Returns
    -------
    dict with test results for accuracy and win_rate
    """
    results = {}
    
    # Pivot to wide format for pairing
    for metric in ['accuracy', 'win_rate']:
        pivot = cond_df.pivot(index='subject_id', columns='condition', values=metric)
        
        if 'sham' not in pivot.columns or 'active' not in pivot.columns:
            continue
        
        paired = pivot[['sham', 'active']].dropna()
        
        if len(paired) < 3:
            continue
        
        t, p = stats.ttest_rel(paired['sham'], paired['active'])
        diff = paired['active'] - paired['sham']
        dz = diff.mean() / diff.std() if diff.std() > 0 else 0
        
        results[metric] = {
            'sham_mean': paired['sham'].mean(),
            'sham_sd': paired['sham'].std(),
            'active_mean': paired['active'].mean(),
            'active_sd': paired['active'].std(),
            'diff_mean': diff.mean(),
            'diff_sd': diff.std(),
            't': t,
            'p': p,
            'dz': dz,
            'n': len(paired),
        }
    
    if verbose:
        print("\n" + "="*60)
        print("STIMULATION EFFECTS ON ACCURACY/WIN RATE")
        print("="*60)
        
        for metric, r in results.items():
            sig = '*' if r['p'] < 0.05 else ''
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Sham: M = {r['sham_mean']:.3f}, SD = {r['sham_sd']:.3f}")
            print(f"  Active: M = {r['active_mean']:.3f}, SD = {r['active_sd']:.3f}")
            print(f"  Δ = {r['diff_mean']:+.3f}")
            print(f"  t({r['n']-1}) = {r['t']:.2f}, p = {r['p']:.3f}{sig}, dz = {r['dz']:.2f}")
    
    return results


# =============================================================================
# Learning Curves
# =============================================================================

def compute_learning_curves(
    data: pd.DataFrame,
    trial_bins: int = 10,
    conditions: List[str] = ['sham', 'active'],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute accuracy across trial bins within runs.
    
    Bins trials within each run to show learning trajectories.
    
    Parameters
    ----------
    data : DataFrame
        Trial-level data
    trial_bins : int
        Number of bins per run
    conditions : list
        Conditions to include
    verbose : bool
        Print summary
    
    Returns
    -------
    DataFrame with trial_bin × condition accuracy
    """
    df = data[data['condition'].isin(conditions)].copy()
    df['correct_num'] = _convert_bool_column(df['correct'])
    
    # Bin trials within each run
    # Assuming ~72 trials per run, bin into groups
    def assign_bin(trial_in_run, n_bins=trial_bins):
        return min(int(trial_in_run * n_bins / 72), n_bins - 1)
    
    # Create trial-in-run counter
    df['trial_in_run'] = df.groupby(['subject_id', 'run']).cumcount()
    df['trial_bin'] = df['trial_in_run'].apply(lambda x: assign_bin(x, trial_bins))
    
    # Compute accuracy by subject × condition × bin
    curves = df.groupby(['subject_id', 'condition', 'trial_bin'])['correct_num'].mean().reset_index()
    curves.rename(columns={'correct_num': 'accuracy'}, inplace=True)
    
    # Group-level curves
    group_curves = curves.groupby(['condition', 'trial_bin'])['accuracy'].agg(['mean', 'sem', 'count']).reset_index()
    
    if verbose:
        print("\n" + "="*60)
        print("LEARNING CURVES (WITHIN-RUN)")
        print("="*60)
        print(f"\nTrial bins: {trial_bins}")
        print(f"Conditions: {conditions}")
        
        for cond in conditions:
            cond_data = group_curves[group_curves['condition'] == cond]
            first_bin = cond_data[cond_data['trial_bin'] == 0]['mean'].values[0]
            last_bin = cond_data[cond_data['trial_bin'] == trial_bins - 1]['mean'].values[0]
            print(f"\n{cond.upper()}:")
            print(f"  First bin: {first_bin:.3f}")
            print(f"  Last bin: {last_bin:.3f}")
            print(f"  Improvement: {last_bin - first_bin:+.3f}")
    
    return curves, group_curves


def compute_contingency_learning_curves(
    data: pd.DataFrame,
    conditions: List[str] = ['sham', 'active'],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute accuracy across trials within each contingency phase.
    
    Each run has 4 contingency phases (separated by reversals).
    This shows learning within each phase and resets at reversals.
    
    Parameters
    ----------
    data : DataFrame
        Trial-level data with trial_in_contingency column
    conditions : list
        Conditions to include
    verbose : bool
        Print summary
    
    Returns
    -------
    DataFrame with trial_in_contingency × condition accuracy
    """
    df = data[data['condition'].isin(conditions)].copy()
    
    if 'trial_in_contingency' not in df.columns:
        print("Warning: trial_in_contingency column not found. Using trial_num instead.")
        return None, None
    
    df['correct_num'] = _convert_bool_column(df['correct'])
    
    # Compute accuracy by subject × condition × trial_in_contingency
    curves = df.groupby(['subject_id', 'condition', 'trial_in_contingency'])['correct_num'].mean().reset_index()
    curves.rename(columns={'correct_num': 'accuracy'}, inplace=True)
    
    # Group-level curves
    group_curves = curves.groupby(['condition', 'trial_in_contingency'])['accuracy'].agg(['mean', 'sem', 'count']).reset_index()
    
    # Limit to first 18 trials (typical contingency length)
    group_curves = group_curves[group_curves['trial_in_contingency'] <= 18]
    
    if verbose:
        print("\n" + "="*60)
        print("CONTINGENCY-PHASE LEARNING CURVES")
        print("="*60)
        print("(Accuracy by trial within each contingency phase)")
        
        for cond in conditions:
            cond_data = group_curves[group_curves['condition'] == cond]
            if len(cond_data) > 0:
                first_trial = cond_data[cond_data['trial_in_contingency'] == 1]['mean'].values
                late_trial = cond_data[cond_data['trial_in_contingency'] >= 10]['mean'].mean()
                
                if len(first_trial) > 0:
                    print(f"\n{cond.upper()}:")
                    print(f"  Trial 1: {first_trial[0]:.3f}")
                    print(f"  Trials 10+: {late_trial:.3f}")
    
    return curves, group_curves


# =============================================================================
# Age Effects
# =============================================================================

def test_age_effects_accuracy(
    cond_df: pd.DataFrame,
    metric: str = 'accuracy',
    verbose: bool = True
) -> Dict:
    """
    Test whether age predicts accuracy/win rate.
    
    Parameters
    ----------
    cond_df : DataFrame
        Output from compute_condition_level_accuracy()
    metric : str
        'accuracy' or 'win_rate'
    verbose : bool
        Print results
    
    Returns
    -------
    dict with correlation results by condition
    """
    results = {}
    
    for cond in cond_df['condition'].unique():
        cond_data = cond_df[cond_df['condition'] == cond][['age', metric]].dropna()
        
        if len(cond_data) >= 5:
            r, p = stats.pearsonr(cond_data['age'], cond_data[metric])
            
            results[cond] = {
                'r': r,
                'p': p,
                'n': len(cond_data),
                'age_range': (cond_data['age'].min(), cond_data['age'].max()),
            }
    
    # Also test age × stimulation effect
    pivot = cond_df.pivot(index='subject_id', columns='condition', values=metric)
    if 'sham' in pivot.columns and 'active' in pivot.columns:
        paired = pivot[['sham', 'active']].dropna()
        paired['delta'] = paired['active'] - paired['sham']
        
        # Add age
        age_lookup = cond_df.groupby('subject_id')['age'].first()
        paired['age'] = paired.index.map(age_lookup)
        paired = paired.dropna()
        
        if len(paired) >= 5:
            r, p = stats.pearsonr(paired['age'], paired['delta'])
            results['age_moderation'] = {
                'r': r,
                'p': p,
                'n': len(paired),
            }
    
    if verbose:
        print("\n" + "="*60)
        print(f"AGE EFFECTS ON {metric.upper()}")
        print("="*60)
        
        for cond, r in results.items():
            if cond == 'age_moderation':
                continue
            sig = '*' if r['p'] < 0.05 else ''
            print(f"\n{cond.upper()}:")
            print(f"  Age × {metric}: r = {r['r']:.3f}, p = {r['p']:.3f}{sig}, n = {r['n']}")
        
        if 'age_moderation' in results:
            r = results['age_moderation']
            sig = '*' if r['p'] < 0.05 else ''
            print(f"\nAge × Stimulation Effect (Δ{metric}):")
            print(f"  r = {r['r']:.3f}, p = {r['p']:.3f}{sig}, n = {r['n']}")
    
    return results


# =============================================================================
# Visualizations
# =============================================================================

def plot_run_level_accuracy(
    run_df: pd.DataFrame,
    show_fig: bool = True
) -> go.Figure:
    """
    Plot accuracy by run with counterbalance coloring.
    
    Shows individual subject trajectories across runs.
    
    Parameters
    ----------
    run_df : DataFrame
        Output from compute_run_level_accuracy()
    show_fig : bool
        Display figure
    
    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    
    # Plot individual subject lines
    for subj in run_df['subject_id'].unique():
        subj_data = run_df[run_df['subject_id'] == subj].sort_values('run')
        age = subj_data['age'].iloc[0]
        cb = subj_data['counterbalance'].iloc[0]
        
        color = _age_to_rgb(age)
        dash = 'solid' if cb == 'A' else 'dash'
        
        fig.add_trace(go.Scatter(
            x=subj_data['run'],
            y=subj_data['accuracy'],
            mode='lines+markers',
            line=dict(color=color, width=1.5, dash=dash),
            marker=dict(size=6, color=color),
            opacity=0.6,
            name=f'Sub-{subj}',
            showlegend=False,
            hovertemplate=f'Sub-{subj}<br>Age: {age:.0f}<br>CB: {cb}<br>Run: %{{x}}<br>Accuracy: %{{y:.3f}}'
        ))
    
    # Group means by run
    run_means = run_df.groupby('run')['accuracy'].agg(['mean', 'sem']).reset_index()
    
    fig.add_trace(go.Scatter(
        x=run_means['run'],
        y=run_means['mean'],
        mode='lines+markers',
        line=dict(color='black', width=3),
        marker=dict(size=10, color='black'),
        error_y=dict(type='data', array=run_means['sem'], visible=True, color='black'),
        name='Group Mean',
        showlegend=True
    ))
    
    # Add reference lines
    fig.add_hline(y=CHANCE_ACCURACY, line_dash='dot', line_color='gray',
                  annotation_text='Chance', annotation_position='right')
    fig.add_hline(y=OPTIMAL_ACCURACY, line_dash='dot', line_color='green',
                  annotation_text='Optimal', annotation_position='right')
    
    # Add vertical line between sessions (run 4-5)
    fig.add_vline(x=4.5, line_dash='dash', line_color='gray', opacity=0.5)
    fig.add_annotation(x=2.5, y=1.02, text='Session 1', showarrow=False, font=dict(size=11))
    fig.add_annotation(x=6.5, y=1.02, text='Session 2', showarrow=False, font=dict(size=11))
    
    fig.update_layout(
        title='Accuracy by Run<br><sup>Solid = CB A (sham first), Dashed = CB B (active first)</sup>',
        xaxis_title='Run',
        yaxis_title='Accuracy',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        yaxis=dict(range=[0.3, 1.02]),
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=500,
        width=700,
        legend=dict(x=0.02, y=0.02, xanchor='left', yanchor='bottom')
    )
    
    if show_fig:
        fig.show()
    
    return fig


def plot_condition_accuracy_paired(
    cond_df: pd.DataFrame,
    metric: str = 'accuracy',
    show_fig: bool = True
) -> go.Figure:
    """
    Paired spaghetti plot for accuracy by condition.
    
    Parameters
    ----------
    cond_df : DataFrame
        Output from compute_condition_level_accuracy()
    metric : str
        'accuracy' or 'win_rate'
    show_fig : bool
        Display figure
    
    Returns
    -------
    go.Figure
    """
    pivot = cond_df.pivot(index='subject_id', columns='condition', values=metric)
    
    if 'sham' not in pivot.columns or 'active' not in pivot.columns:
        print("Need both sham and active conditions")
        return None
    
    paired = pivot[['sham', 'active']].dropna()
    
    # Get age for coloring
    age_lookup = cond_df.groupby('subject_id')['age'].first().to_dict()
    
    fig = go.Figure()
    
    # Individual lines
    for subj in paired.index:
        age = age_lookup.get(subj, 50)
        color = _age_to_rgb(age)
        
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[paired.loc[subj, 'sham'], paired.loc[subj, 'active']],
            mode='lines+markers',
            line=dict(color=color, width=1.5),
            marker=dict(size=8, color=color, line=dict(width=0.5, color='white')),
            opacity=0.7,
            showlegend=False,
            hovertemplate=f'Sub-{subj}<br>Age: {age:.0f}<br>Sham: {paired.loc[subj, "sham"]:.3f}<br>Active: {paired.loc[subj, "active"]:.3f}'
        ))
    
    # Group means
    for i, cond in enumerate(['sham', 'active']):
        mean = paired[cond].mean()
        sem = paired[cond].sem()
        
        fig.add_trace(go.Scatter(
            x=[i], y=[mean],
            mode='markers',
            marker=dict(size=14, color='#404040', symbol='diamond',
                        line=dict(width=2, color='white')),
            error_y=dict(type='data', array=[sem], visible=True, color='#404040', thickness=2),
            showlegend=False
        ))
    
    # Stats annotation
    t, p = stats.ttest_rel(paired['sham'], paired['active'])
    diff = paired['active'] - paired['sham']
    dz = diff.mean() / diff.std() if diff.std() > 0 else 0
    
    fig.add_annotation(
        x=0.5, y=0.98, xref='paper', yref='paper',
        text=f'Δ = {diff.mean():+.3f}<br>t = {t:.2f}, p = {p:.3f}<br>dz = {dz:.2f}',
        showarrow=False,
        font=dict(size=11),
        bgcolor='rgba(255,255,255,0.8)'
    )
    
    fig.update_layout(
        title=f'{metric.replace("_", " ").title()} by Condition (N = {len(paired)})',
        xaxis=dict(tickvals=[0, 1], ticktext=['Sham', 'Active']),
        yaxis_title=metric.replace('_', ' ').title(),
        yaxis=dict(range=[0.4, 1.0]),
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=450,
        width=400
    )
    
    if show_fig:
        fig.show()
    
    return fig


def plot_learning_curves(
    group_curves: pd.DataFrame,
    conditions: List[str] = ['sham', 'active'],
    show_fig: bool = True
) -> go.Figure:
    """
    Plot learning curves (accuracy by trial bin within runs).
    
    Parameters
    ----------
    group_curves : DataFrame
        Output from compute_learning_curves()
    conditions : list
        Conditions to plot
    show_fig : bool
        Display figure
    
    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    
    colors = {'sham': COLOR_SHAM, 'active': COLOR_ACTIVE}
    
    for cond in conditions:
        cond_data = group_curves[group_curves['condition'] == cond].sort_values('trial_bin')
        
        fig.add_trace(go.Scatter(
            x=cond_data['trial_bin'],
            y=cond_data['mean'],
            mode='lines+markers',
            line=dict(color=colors.get(cond, 'gray'), width=2),
            marker=dict(size=8, color=colors.get(cond, 'gray')),
            error_y=dict(type='data', array=cond_data['sem'], visible=True),
            name=cond.capitalize()
        ))
    
    # Reference lines
    fig.add_hline(y=CHANCE_ACCURACY, line_dash='dot', line_color='gray')
    fig.add_hline(y=OPTIMAL_ACCURACY, line_dash='dot', line_color='green', opacity=0.5)
    
    fig.update_layout(
        title='Learning Curves (Within-Run)<br><sup>Accuracy by trial bin</sup>',
        xaxis_title='Trial Bin (within run)',
        yaxis_title='Accuracy',
        yaxis=dict(range=[0.4, 0.95]),
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=450,
        width=600,
        legend=dict(x=0.02, y=0.98)
    )
    
    if show_fig:
        fig.show()
    
    return fig


def plot_contingency_learning(
    group_curves: pd.DataFrame,
    conditions: List[str] = ['sham', 'active'],
    show_fig: bool = True
) -> go.Figure:
    """
    Plot accuracy by trial within contingency phase.
    
    Shows learning after each reversal (resets at trial 1).
    
    Parameters
    ----------
    group_curves : DataFrame
        Output from compute_contingency_learning_curves()
    conditions : list
        Conditions to plot
    show_fig : bool
        Display figure
    
    Returns
    -------
    go.Figure
    """
    if group_curves is None:
        return None
    
    fig = go.Figure()
    
    colors = {'sham': COLOR_SHAM, 'active': COLOR_ACTIVE}
    
    for cond in conditions:
        cond_data = group_curves[group_curves['condition'] == cond].sort_values('trial_in_contingency')
        
        fig.add_trace(go.Scatter(
            x=cond_data['trial_in_contingency'],
            y=cond_data['mean'],
            mode='lines+markers',
            line=dict(color=colors.get(cond, 'gray'), width=2),
            marker=dict(size=6, color=colors.get(cond, 'gray')),
            error_y=dict(type='data', array=cond_data['sem'], visible=True),
            name=cond.capitalize()
        ))
    
    # Reference lines
    fig.add_hline(y=CHANCE_ACCURACY, line_dash='dot', line_color='gray')
    fig.add_hline(y=OPTIMAL_ACCURACY, line_dash='dot', line_color='green', opacity=0.5)
    
    # Mark typical reversal point
    fig.add_vline(x=18, line_dash='dash', line_color='red', opacity=0.3,
                  annotation_text='~Reversal', annotation_position='top')
    
    fig.update_layout(
        title='Contingency-Phase Learning<br><sup>Accuracy by trial within contingency (resets at reversals)</sup>',
        xaxis_title='Trial in Contingency Phase',
        yaxis_title='Accuracy',
        yaxis=dict(range=[0.3, 0.95]),
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=450,
        width=600,
        legend=dict(x=0.98, y=0.02, xanchor='right', yanchor='bottom')
    )
    
    if show_fig:
        fig.show()
    
    return fig


def plot_age_accuracy_scatter(
    cond_df: pd.DataFrame,
    metric: str = 'accuracy',
    condition: str = 'sham',
    show_fig: bool = True
) -> go.Figure:
    """
    Scatter plot: age vs accuracy for a given condition.
    
    Parameters
    ----------
    cond_df : DataFrame
        Output from compute_condition_level_accuracy()
    metric : str
        'accuracy' or 'win_rate'
    condition : str
        Condition to plot
    show_fig : bool
        Display figure
    
    Returns
    -------
    go.Figure
    """
    df = cond_df[cond_df['condition'] == condition][['subject_id', 'age', metric]].dropna()
    
    if len(df) < 5:
        print(f"Insufficient data for age × {metric} plot")
        return None
    
    r, p = stats.pearsonr(df['age'], df[metric])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['age'],
        y=df[metric],
        mode='markers',
        marker=dict(size=12, color=[_age_to_rgb(a) for a in df['age']],
                    line=dict(width=1, color='white')),
        text=[f"Sub-{s}<br>Age: {a:.0f}<br>{metric}: {v:.3f}" 
              for s, a, v in zip(df['subject_id'], df['age'], df[metric])],
        hoverinfo='text',
        showlegend=False
    ))
    
    # Regression line
    slope, intercept = np.polyfit(df['age'], df[metric], 1)
    x_line = np.linspace(df['age'].min() - 5, df['age'].max() + 5, 100)
    fig.add_trace(go.Scatter(
        x=x_line, y=intercept + slope * x_line,
        mode='lines', line=dict(color='#404040', width=2, dash='dash'),
        showlegend=False
    ))
    
    # Reference lines
    fig.add_hline(y=CHANCE_ACCURACY, line_dash='dot', line_color='gray')
    fig.add_hline(y=OPTIMAL_ACCURACY, line_dash='dot', line_color='green', opacity=0.5)
    
    fig.update_layout(
        title=f'Age × {metric.replace("_", " ").title()} ({condition.title()})<br>'
              f'<sup>r = {r:.3f}, p = {p:.3f}, N = {len(df)}</sup>',
        xaxis_title='Age (years)',
        yaxis_title=metric.replace('_', ' ').title(),
        yaxis=dict(range=[0.4, 1.0]),
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=450,
        width=550
    )
    
    if show_fig:
        fig.show()
    
    return fig


def plot_counterbalance_comparison(
    run_df: pd.DataFrame,
    show_fig: bool = True
) -> go.Figure:
    """
    Compare accuracy trajectories by counterbalance.
    
    Parameters
    ----------
    run_df : DataFrame
        Output from compute_run_level_accuracy()
    show_fig : bool
        Display figure
    
    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    
    colors = {'A': COLOR_SHAM, 'B': COLOR_ACTIVE}
    labels = {'A': 'CB A (sham first)', 'B': 'CB B (active first)'}
    
    for cb in ['A', 'B']:
        cb_data = run_df[run_df['counterbalance'] == cb]
        cb_means = cb_data.groupby('run')['accuracy'].agg(['mean', 'sem']).reset_index()
        
        fig.add_trace(go.Scatter(
            x=cb_means['run'],
            y=cb_means['mean'],
            mode='lines+markers',
            line=dict(color=colors[cb], width=2.5),
            marker=dict(size=9, color=colors[cb]),
            error_y=dict(type='data', array=cb_means['sem'], visible=True, color=colors[cb]),
            name=labels[cb]
        ))
    
    # Session divider
    fig.add_vline(x=4.5, line_dash='dash', line_color='gray', opacity=0.5)
    
    # Reference lines
    fig.add_hline(y=CHANCE_ACCURACY, line_dash='dot', line_color='gray')
    
    fig.update_layout(
        title='Accuracy by Run and Counterbalance',
        xaxis_title='Run',
        yaxis_title='Accuracy',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        yaxis=dict(range=[0.5, 0.9]),
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=450,
        width=650,
        legend=dict(x=0.02, y=0.02, xanchor='left', yanchor='bottom')
    )
    
    if show_fig:
        fig.show()
    
    return fig


# =============================================================================
# Baseline Performance Moderates Stimulation Effects
# =============================================================================

def compute_baseline_moderation(
    run_df: pd.DataFrame,
    cond_df: pd.DataFrame,
    baseline_metric: str = 'run1_accuracy',
    verbose: bool = True
) -> Dict:
    """
    Test whether baseline (run-1) accuracy predicts stimulation efficacy.
    
    Logic: If tACS works better for some individuals than others, baseline
    task competence might moderate effects. Run-1 is used because it's
    uncontaminated by any active stimulation (regardless of counterbalance,
    it's their first exposure to the task).
    
    Parameters
    ----------
    run_df : DataFrame
        Output from compute_run_level_accuracy()
    cond_df : DataFrame
        Output from compute_condition_level_accuracy()
    baseline_metric : str
        Which baseline to use ('run1_accuracy' or 'sham_accuracy')
    verbose : bool
        Print results
    
    Returns
    -------
    dict with moderation results for accuracy and win_rate
    """
    results = {}
    
    # Get run-1 accuracy per subject
    run1 = run_df[run_df['run'] == 1][['subject_id', 'accuracy', 'age']].copy()
    run1 = run1.rename(columns={'accuracy': 'run1_accuracy'})
    run1 = run1.set_index('subject_id')
    
    # Get condition-level accuracy/win_rate and compute deltas
    for metric in ['accuracy', 'win_rate']:
        pivot = cond_df.pivot(index='subject_id', columns='condition', values=metric)
        
        if 'sham' not in pivot.columns or 'active' not in pivot.columns:
            continue
        
        paired = pivot[['sham', 'active']].dropna()
        paired['delta'] = paired['active'] - paired['sham']
        
        # Merge with run-1 baseline
        merged = paired.join(run1[['run1_accuracy', 'age']], how='inner')
        merged = merged.dropna()
        
        if len(merged) < 5:
            continue
        
        # Test: run1_accuracy × delta
        r, p = stats.pearsonr(merged['run1_accuracy'], merged['delta'])
        
        # Also test with age partialed out (if we want to be thorough)
        # For now, just report the simple correlation
        
        results[metric] = {
            'r': r,
            'p': p,
            'n': len(merged),
            'baseline_mean': merged['run1_accuracy'].mean(),
            'baseline_sd': merged['run1_accuracy'].std(),
            'delta_mean': merged['delta'].mean(),
            'delta_sd': merged['delta'].std(),
            'data': merged,  # For plotting
        }
        
        # Direction interpretation
        if r > 0:
            interpretation = "Higher baseline → larger tACS benefit"
        else:
            interpretation = "Lower baseline → larger tACS benefit"
        results[metric]['interpretation'] = interpretation
    
    if verbose:
        print("\n" + "="*60)
        print("BASELINE MODERATION OF STIMULATION EFFECTS")
        print("="*60)
        print("\nQuestion: Does run-1 accuracy predict tACS efficacy (active - sham)?")
        
        for metric, r in results.items():
            sig = '*' if r['p'] < 0.05 else ''
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Run-1 accuracy: M = {r['baseline_mean']:.3f}, SD = {r['baseline_sd']:.3f}")
            print(f"  Δ (active - sham): M = {r['delta_mean']:+.3f}, SD = {r['delta_sd']:.3f}")
            print(f"  Correlation: r = {r['r']:.3f}, p = {r['p']:.3f}{sig}, n = {r['n']}")
            print(f"  Interpretation: {r['interpretation']}")
    
    return results


def plot_baseline_moderation(
    moderation_results: Dict,
    metric: str = 'accuracy',
    show_fig: bool = True
) -> go.Figure:
    """
    Scatter plot: Run-1 accuracy (x) vs stimulation effect (y).
    
    Parameters
    ----------
    moderation_results : dict
        Output from compute_baseline_moderation()
    metric : str
        'accuracy' or 'win_rate'
    show_fig : bool
        Display figure
    
    Returns
    -------
    go.Figure
    """
    if metric not in moderation_results:
        print(f"No moderation results for {metric}")
        return None
    
    res = moderation_results[metric]
    data = res['data']
    
    fig = go.Figure()
    
    # Scatter points colored by age
    for subj in data.index:
        age = data.loc[subj, 'age']
        color = _age_to_rgb(age)
        
        fig.add_trace(go.Scatter(
            x=[data.loc[subj, 'run1_accuracy']],
            y=[data.loc[subj, 'delta']],
            mode='markers',
            marker=dict(size=12, color=color, line=dict(width=1, color='white')),
            showlegend=False,
            hovertemplate=f"Sub-{subj}<br>Age: {age:.0f}<br>"
                          f"Run-1 acc: {data.loc[subj, 'run1_accuracy']:.3f}<br>"
                          f"Δ{metric}: {data.loc[subj, 'delta']:+.3f}"
        ))
    
    # Regression line
    x = data['run1_accuracy']
    y = data['delta']
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min() - 0.02, x.max() + 0.02, 100)
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=intercept + slope * x_line,
        mode='lines',
        line=dict(color='#404040', width=2, dash='dash'),
        showlegend=False
    ))
    
    # Reference lines
    fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.7)
    fig.add_vline(x=0.5, line_dash='dot', line_color='gray', opacity=0.5,
                  annotation_text='Chance', annotation_position='bottom')
    
    # Annotation with stats
    sig = '*' if res['p'] < 0.05 else ''
    fig.add_annotation(
        x=0.98, y=0.98, xref='paper', yref='paper',
        text=f"r = {res['r']:.3f}, p = {res['p']:.3f}{sig}<br>N = {res['n']}",
        showarrow=False,
        font=dict(size=12),
        bgcolor='rgba(255,255,255,0.8)',
        xanchor='right', yanchor='top'
    )
    
    # Labels
    metric_label = metric.replace('_', ' ').title()
    
    fig.update_layout(
        title=f"Baseline Performance Moderates tACS Effect<br>"
              f"<sup>Run-1 {metric_label} vs. Stimulation Effect (Active − Sham)</sup>",
        xaxis_title=f"Run-1 {metric_label}",
        yaxis_title=f"Δ {metric_label} (Active − Sham)",
        xaxis=dict(range=[0.35, 0.95]),
        yaxis=dict(zeroline=True),
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=500,
        width=550
    )
    
    if show_fig:
        fig.show()
    
    return fig


def test_baseline_moderates_other_dvs(
    run_df: pd.DataFrame,
    wsls_h2: pd.DataFrame = None,
    rw_df: pd.DataFrame = None,
    ddm_df: pd.DataFrame = None,
    verbose: bool = True
) -> Dict:
    """
    Test whether run-1 accuracy predicts stimulation effects on other DVs.
    
    Tests moderation for:
    - WSLS parameters (if wsls_h2 provided)
    - R-W parameters (if rw_df provided)
    - DDM parameters (if ddm_df provided)
    
    Parameters
    ----------
    run_df : DataFrame
        Output from compute_run_level_accuracy()
    wsls_h2 : DataFrame, optional
        WSLS results with sham/active columns
    rw_df : DataFrame, optional
        R-W parameters by subject × condition
    ddm_df : DataFrame, optional
        DDM parameters by subject × condition
    verbose : bool
        Print results
    
    Returns
    -------
    dict with moderation results for each DV
    """
    results = {}
    
    # Get run-1 accuracy
    run1 = run_df[run_df['run'] == 1][['subject_id', 'accuracy']].copy()
    run1 = run1.set_index('subject_id')
    run1 = run1.rename(columns={'accuracy': 'run1_accuracy'})
    
    # WSLS moderation
    if wsls_h2 is not None:
        for dv in ['p_stay_win', 'p_shift_lose']:
            if f'{dv}_sham' in wsls_h2.columns and f'{dv}_active' in wsls_h2.columns:
                wsls_data = wsls_h2[[f'{dv}_sham', f'{dv}_active']].copy()
                wsls_data['delta'] = wsls_data[f'{dv}_active'] - wsls_data[f'{dv}_sham']
                
                merged = wsls_data.join(run1, how='inner').dropna()
                
                if len(merged) >= 5:
                    r, p = stats.pearsonr(merged['run1_accuracy'], merged['delta'])
                    results[f'wsls_{dv}'] = {'r': r, 'p': p, 'n': len(merged)}
    
    # R-W moderation
    if rw_df is not None:
        for param in ['alpha', 'beta']:
            sham_col = f'{param}_sham' if f'{param}_sham' in rw_df.columns else None
            active_col = f'{param}_active' if f'{param}_active' in rw_df.columns else None
            
            # Try alternative column naming
            if sham_col is None:
                sham_data = rw_df[(rw_df['condition'] == 'sham')][['subject_id', param]].set_index('subject_id') if 'condition' in rw_df.columns else None
                active_data = rw_df[(rw_df['condition'] == 'active')][['subject_id', param]].set_index('subject_id') if 'condition' in rw_df.columns else None
                
                if sham_data is not None and active_data is not None:
                    rw_paired = sham_data.join(active_data, lsuffix='_sham', rsuffix='_active')
                    rw_paired['delta'] = rw_paired[f'{param}_active'] - rw_paired[f'{param}_sham']
                    
                    merged = rw_paired.join(run1, how='inner').dropna()
                    
                    if len(merged) >= 5:
                        r, p = stats.pearsonr(merged['run1_accuracy'], merged['delta'])
                        results[f'rw_{param}'] = {'r': r, 'p': p, 'n': len(merged)}
    
    # DDM moderation
    if ddm_df is not None:
        for param in ['v', 'a', 't']:
            if f'{param}_sham' in ddm_df.columns and f'{param}_active' in ddm_df.columns:
                ddm_data = ddm_df[[f'{param}_sham', f'{param}_active']].copy()
                ddm_data['delta'] = ddm_data[f'{param}_active'] - ddm_data[f'{param}_sham']
                
                merged = ddm_data.join(run1, how='inner').dropna()
                
                if len(merged) >= 5:
                    r, p = stats.pearsonr(merged['run1_accuracy'], merged['delta'])
                    results[f'ddm_{param}'] = {'r': r, 'p': p, 'n': len(merged)}
    
    if verbose and results:
        print("\n" + "="*60)
        print("BASELINE MODERATION OF OTHER DVs")
        print("="*60)
        print("\nRun-1 accuracy × Δ(active - sham) correlations:")
        
        for dv, r in results.items():
            sig = '*' if r['p'] < 0.05 else ''
            print(f"  {dv}: r = {r['r']:.3f}, p = {r['p']:.3f}{sig}, n = {r['n']}")
    
    return results


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_accuracy_analysis(
    data: pd.DataFrame,
    conditions: List[str] = ['sham', 'active'],
    show_plots: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run complete accuracy analysis pipeline.
    
    Parameters
    ----------
    data : DataFrame
        Trial-level data with subject_id, run, condition, correct, reward columns
    conditions : list
        Conditions to analyze
    show_plots : bool
        Display visualizations
    verbose : bool
        Print results
    
    Returns
    -------
    dict with all analysis results and figures
    """
    results = {
        'run_level': None,
        'condition_level': None,
        'counterbalance': None,
        'stimulation_effects': None,
        'learning_curves': None,
        'contingency_curves': None,
        'age_effects': None,
        'baseline_moderation': None,
        'figures': {}
    }
    
    if verbose:
        print("="*70)
        print("ACCURACY ANALYSIS")
        print("="*70)
    
    # -------------------------------------------------------------------------
    # 1. Run-level accuracy
    # -------------------------------------------------------------------------
    run_df = compute_run_level_accuracy(data, verbose=verbose)
    results['run_level'] = run_df
    
    # -------------------------------------------------------------------------
    # 2. Condition-level accuracy
    # -------------------------------------------------------------------------
    cond_df = compute_condition_level_accuracy(data, conditions=conditions, verbose=verbose)
    results['condition_level'] = cond_df
    
    # -------------------------------------------------------------------------
    # 3. Counterbalance effects
    # -------------------------------------------------------------------------
    cb_results = analyze_counterbalance_effects(run_df, verbose=verbose)
    results['counterbalance'] = cb_results
    
    # -------------------------------------------------------------------------
    # 4. Stimulation effects
    # -------------------------------------------------------------------------
    stim_results = test_stimulation_effects_accuracy(cond_df, verbose=verbose)
    results['stimulation_effects'] = stim_results
    
    # -------------------------------------------------------------------------
    # 5. Learning curves
    # -------------------------------------------------------------------------
    curves_subj, curves_group = compute_learning_curves(data, conditions=conditions, verbose=verbose)
    results['learning_curves'] = {'subject': curves_subj, 'group': curves_group}
    
    cont_subj, cont_group = compute_contingency_learning_curves(data, conditions=conditions, verbose=verbose)
    results['contingency_curves'] = {'subject': cont_subj, 'group': cont_group}
    
    # -------------------------------------------------------------------------
    # 6. Age effects
    # -------------------------------------------------------------------------
    age_acc = test_age_effects_accuracy(cond_df, metric='accuracy', verbose=verbose)
    age_win = test_age_effects_accuracy(cond_df, metric='win_rate', verbose=verbose)
    results['age_effects'] = {'accuracy': age_acc, 'win_rate': age_win}
    
    # -------------------------------------------------------------------------
    # 7. Baseline moderation of stimulation effects
    # -------------------------------------------------------------------------
    baseline_mod = compute_baseline_moderation(run_df, cond_df, verbose=verbose)
    results['baseline_moderation'] = baseline_mod
    
    # -------------------------------------------------------------------------
    # 8. Visualizations
    # -------------------------------------------------------------------------
    if show_plots:
        if verbose:
            print("\n" + "="*70)
            print("GENERATING VISUALIZATIONS")
            print("="*70)
        
        # Run-level accuracy
        results['figures']['run_accuracy'] = plot_run_level_accuracy(run_df, show_fig=True)
        
        # Counterbalance comparison
        results['figures']['counterbalance'] = plot_counterbalance_comparison(run_df, show_fig=True)
        
        # Condition paired plots
        results['figures']['accuracy_paired'] = plot_condition_accuracy_paired(
            cond_df, metric='accuracy', show_fig=True
        )
        results['figures']['winrate_paired'] = plot_condition_accuracy_paired(
            cond_df, metric='win_rate', show_fig=True
        )
        
        # Learning curves
        results['figures']['learning_curves'] = plot_learning_curves(
            curves_group, conditions=conditions, show_fig=True
        )
        
        # Contingency learning
        if cont_group is not None:
            results['figures']['contingency_learning'] = plot_contingency_learning(
                cont_group, conditions=conditions, show_fig=True
            )
        
        # Age effects
        results['figures']['age_accuracy_sham'] = plot_age_accuracy_scatter(
            cond_df, metric='accuracy', condition='sham', show_fig=True
        )
        results['figures']['age_accuracy_active'] = plot_age_accuracy_scatter(
            cond_df, metric='accuracy', condition='active', show_fig=True
        )
        
        # Baseline moderation
        if baseline_mod and 'accuracy' in baseline_mod:
            results['figures']['baseline_mod_accuracy'] = plot_baseline_moderation(
                baseline_mod, metric='accuracy', show_fig=True
            )
        if baseline_mod and 'win_rate' in baseline_mod:
            results['figures']['baseline_mod_winrate'] = plot_baseline_moderation(
                baseline_mod, metric='win_rate', show_fig=True
            )
    
    if verbose:
        print("\n" + "="*70)
        print("ACCURACY ANALYSIS COMPLETE")
        print("="*70)
    
    return results


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing accuracy_analysis module...")
    print("\nMain functions:")
    print("  - compute_run_level_accuracy()")
    print("  - compute_condition_level_accuracy()")
    print("  - analyze_counterbalance_effects()")
    print("  - test_stimulation_effects_accuracy()")
    print("  - compute_learning_curves()")
    print("  - compute_contingency_learning_curves()")
    print("  - test_age_effects_accuracy()")
    print("  - compute_baseline_moderation()")
    print("  - plot_baseline_moderation()")
    print("  - test_baseline_moderates_other_dvs()")
    print("  - run_accuracy_analysis()")
