"""
exclusions.py — Pre-registered exclusion criteria for tACS Bandit analyses

Implements exclusion logic per Section 6 of the pre-registration (https://osf.io/vhyz2):

Run-level behavioral exclusions:
1. >20% missed trials
2. Side bias (>95% same spatial location)
3. Stimulus bias (>95% same stimulus regardless of position)
4. Rapid responding (median RT <200ms)
5. Feedback invariance (10+ consecutive losses without shift)

Stimulation administration exclusions:
- Runs where delivered stimulation did not match counterbalance assignment

Produces filtered datasets:
- data_h1: sham condition only, behavioral exclusions applied
- data_h2: active & sham, all exclusions + minimum run threshold
- data_clean: all conditions, behavioral exclusions applied
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set

from config import (
    STIM_EXCLUSIONS,
    EXCLUSION_THRESHOLDS,
    MIN_CLEAN_RUNS_PER_CONDITION,
)


# =============================================================================
# Core Exclusion Functions
# =============================================================================

def compute_max_consecutive_loss_no_shift(group: pd.DataFrame) -> int:
    """
    Find the maximum number of consecutive losses without a shift.
    
    Pre-registration criterion: exclude if participant fails to shift
    following 10 or more consecutive 'lose' trials.
    
    Logic: For each streak of consecutive losses, check if the participant
    shifted at any point during the streak. A "shift" on trial t means the
    participant chose a different stimulus than on trial t-1.
    
    Parameters
    ----------
    group : DataFrame
        Trial-level data for a single run (must have 'reward', 'shift' columns)
    
    Returns
    -------
    int
        Maximum consecutive losses without any shift response
    """
    sort_cols = ['run', 'trial_num'] if 'trial_num' in group.columns else ['run']
    df = group.sort_values(sort_cols).copy()
    
    # Need reward and shift columns
    if 'reward' not in df.columns or 'shift' not in df.columns:
        return 0
    
    # Work with valid trials only
    df = df.dropna(subset=['choice', 'reward'])
    if len(df) == 0:
        return 0
    
    # Create loss indicator (handles bool, int, and string representations)
    df['is_loss'] = df['reward'].apply(
        lambda x: not x if isinstance(x, bool) else str(x).lower() in ('0', 'false', '0.0')
    )
    
    max_consec = 0
    current_consec = 0
    shifted_in_run = False
    
    for idx, row in df.iterrows():
        if row['is_loss']:
            current_consec += 1
            if pd.notna(row.get('shift')) and row['shift'] == 1:
                shifted_in_run = True
        else:
            # Win breaks the loss streak
            if current_consec > 0 and not shifted_in_run:
                max_consec = max(max_consec, current_consec)
            current_consec = 0
            shifted_in_run = False
    
    # Check final streak if it ended in losses
    if current_consec > 0 and not shifted_in_run:
        max_consec = max(max_consec, current_consec)
    
    return max_consec


def compute_exclusion_criteria(
    df: pd.DataFrame,
    group_cols: List[str] = ['subject_id', 'run'],
    thresholds: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Compute pre-registered exclusion criteria for each grouping.
    
    Default grouping is subject × run (run-level exclusions). Can also be
    called with group_cols=['subject_id'] for subject-level summaries or
    group_cols=['subject_id', 'condition'] for condition-level checks.
    
    Parameters
    ----------
    df : DataFrame
        Trial-level data with columns: subject_id, run, choice, reward,
        rt, slot1_position, slot2_position.
    group_cols : list
        Columns to group by.
    thresholds : dict, optional
        Override default thresholds; keys: missed_pct, side_bias_pct,
        stim_bias_pct, rapid_rt_ms, feedback_invariance
    
    Returns
    -------
    DataFrame
        One row per group with exclusion flags and metrics:
        - n_trials, n_missed, pct_missed, flag_missed
        - max_side_pct, flag_side_bias
        - max_stim_pct, flag_stim_bias
        - median_rt_ms, flag_rapid_rt
        - max_consec_loss_no_shift, flag_feedback_invariance
        - exclude_behavioral (True if any flag is True)
    """
    if thresholds is None:
        thresholds = EXCLUSION_THRESHOLDS
    
    results = []
    
    # Identify RT column (handle naming variations across task versions)
    rt_col = None
    for col in ['rt', 'RT', 'response_time', 'reaction_time']:
        if col in df.columns:
            rt_col = col
            break
    
    for name, group in df.groupby(group_cols):
        row = dict(zip(group_cols, name if isinstance(name, tuple) else [name]))
        n_trials = len(group)
        
        # --- Criterion 1: >20% missed trials ---
        n_missed = group['choice'].isna().sum()
        pct_missed = 100 * n_missed / n_trials if n_trials > 0 else 0
        row['n_trials'] = n_trials
        row['n_missed'] = n_missed
        row['pct_missed'] = pct_missed
        row['flag_missed'] = pct_missed > thresholds['missed_pct']
        
        # --- Valid (responded) trials for remaining criteria ---
        valid_choices = group.dropna(subset=['choice'])
        n_valid = len(valid_choices)
        
        # --- Criterion 2: Side bias (>95% same spatial location) ---
        if 'slot1_position' in valid_choices.columns and 'slot2_position' in valid_choices.columns:
            chosen_side = valid_choices.apply(
                lambda r: r['slot1_position'] if r['choice'] == 1 else r['slot2_position'],
                axis=1
            )
            side_counts = chosen_side.value_counts(normalize=True)
            max_side_pct = side_counts.max() * 100 if len(side_counts) > 0 else 0
        else:
            # Fallback: cannot compute side bias without position columns
            max_side_pct = np.nan
        
        row['max_side_pct'] = max_side_pct
        row['flag_side_bias'] = max_side_pct > thresholds['side_bias_pct'] if not np.isnan(max_side_pct) else False
        
        # --- Criterion 3: Stimulus bias (>95% same stimulus) ---
        stim_counts = valid_choices['choice'].value_counts(normalize=True)
        max_stim_pct = stim_counts.max() * 100 if len(stim_counts) > 0 else 0
        row['max_stim_pct'] = max_stim_pct
        row['flag_stim_bias'] = max_stim_pct > thresholds['stim_bias_pct']
        
        # --- Criterion 4: Rapid responding (median RT < 200ms) ---
        if rt_col:
            valid_rt = valid_choices[rt_col].dropna()
            median_rt = valid_rt.median() if len(valid_rt) > 0 else np.nan
            # Safety check: if median < 100, assume seconds and convert to ms
            if not np.isnan(median_rt) and median_rt < 100:
                median_rt_ms = median_rt * 1000
            else:
                median_rt_ms = median_rt
            row['median_rt_ms'] = median_rt_ms
            row['flag_rapid_rt'] = median_rt_ms < thresholds['rapid_rt_ms'] if not np.isnan(median_rt_ms) else False
        else:
            row['median_rt_ms'] = np.nan
            row['flag_rapid_rt'] = False
        
        # --- Criterion 5: Feedback invariance ---
        max_consec_loss_no_shift = compute_max_consecutive_loss_no_shift(group)
        row['max_consec_loss_no_shift'] = max_consec_loss_no_shift
        row['flag_feedback_invariance'] = max_consec_loss_no_shift >= thresholds['feedback_invariance']
        
        # --- Overall behavioral exclusion flag ---
        row['exclude_behavioral'] = any([
            row['flag_missed'],
            row['flag_side_bias'],
            row['flag_stim_bias'],
            row['flag_rapid_rt'],
            row['flag_feedback_invariance']
        ])
        
        results.append(row)
    
    return pd.DataFrame(results)


# =============================================================================
# Stimulation Exclusion Functions
# =============================================================================

def apply_stim_exclusions(
    run_exclusions: pd.DataFrame,
    stim_exclusions: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Add stimulation administration exclusion flags to run-level table.
    
    Parameters
    ----------
    run_exclusions : DataFrame
        Run-level exclusion table from compute_exclusion_criteria()
    stim_exclusions : dict, optional
        Stim exclusion registry; defaults to STIM_EXCLUSIONS from config
    
    Returns
    -------
    DataFrame
        Input table with added columns:
        - exclude_stim: True if run is in stim exclusion registry
        - stim_exclusion_reason: reason text if excluded
        - exclude_h2: True if excluded for H2 (behavioral OR stim)
        - exclude_h1: True if excluded for H1 (behavioral only)
    """
    if stim_exclusions is None:
        stim_exclusions = STIM_EXCLUSIONS
    
    run_exclusions = run_exclusions.copy()
    
    # Apply stim exclusion flags
    run_exclusions['exclude_stim'] = run_exclusions.apply(
        lambda r: (str(r['subject_id']), int(r['run'])) in stim_exclusions, axis=1
    )
    run_exclusions['stim_exclusion_reason'] = run_exclusions.apply(
        lambda r: stim_exclusions.get(
            (str(r['subject_id']), int(r['run'])), {}
        ).get('reason', ''),
        axis=1
    )
    
    # Combined exclusion flags
    run_exclusions['exclude_h2'] = (
        run_exclusions['exclude_behavioral'] | run_exclusions['exclude_stim']
    )
    run_exclusions['exclude_h1'] = run_exclusions['exclude_behavioral']
    
    return run_exclusions


# =============================================================================
# Trial-Level Exclusion Tagging
# =============================================================================

def tag_trial_exclusions(
    data: pd.DataFrame,
    run_exclusions: pd.DataFrame,
    stim_exclusions: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Add exclusion flags to trial-level data based on run-level exclusions.
    
    Parameters
    ----------
    data : DataFrame
        Trial-level data
    run_exclusions : DataFrame
        Run-level exclusion table with exclude_behavioral column
    stim_exclusions : dict, optional
        Stim exclusion registry
    
    Returns
    -------
    DataFrame
        Input data with added columns: exclude_behavioral, exclude_stim
    """
    if stim_exclusions is None:
        stim_exclusions = STIM_EXCLUSIONS
    
    data = data.copy()
    
    # Create exclusion sets for fast lookup
    behavioral_exclusion_set = set(
        zip(
            run_exclusions.loc[run_exclusions['exclude_behavioral'], 'subject_id'].astype(str),
            run_exclusions.loc[run_exclusions['exclude_behavioral'], 'run'].astype(int)
        )
    )
    stim_exclusion_set = set(
        (str(sid), int(run)) for (sid, run) in stim_exclusions.keys()
    )
    
    # Tag each trial
    run_keys = list(zip(data['subject_id'].astype(str), data['run'].astype(int)))
    data['exclude_behavioral'] = [k in behavioral_exclusion_set for k in run_keys]
    data['exclude_stim'] = [k in stim_exclusion_set for k in run_keys]
    
    return data


# =============================================================================
# Filtered Dataset Creation
# =============================================================================

def create_filtered_datasets(
    data: pd.DataFrame,
    min_clean_runs: int = MIN_CLEAN_RUNS_PER_CONDITION,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Create filtered datasets for H1, H2, and general analyses.
    
    Parameters
    ----------
    data : DataFrame
        Trial-level data with exclude_behavioral and exclude_stim columns
    min_clean_runs : int
        Minimum clean runs required per condition for H2 eligibility
    verbose : bool
        If True, print summary
    
    Returns
    -------
    tuple of (data_h1, data_h2, data_clean, h2_eligible_subjects)
        data_h1: sham condition, behavioral exclusions applied
        data_h2: active & sham, all exclusions + min-run threshold
        data_clean: all conditions, behavioral exclusions applied
        h2_eligible_subjects: list of subject IDs eligible for H2
    """
    # data_clean: all conditions, behavioral exclusions only
    data_clean = data[~data['exclude_behavioral']].copy()
    
    # data_h1: sham condition only, behavioral exclusions
    data_h1 = data[
        (~data['exclude_behavioral']) &
        (data['condition'] == 'sham')
    ].copy()
    
    # data_h2: active & sham, both exclusion types
    data_h2_prelim = data[
        (~data['exclude_behavioral']) &
        (~data['exclude_stim']) &
        (data['condition'].isin(['active', 'sham']))
    ].copy()
    
    # Enforce minimum run threshold for H2
    h2_run_counts = data_h2_prelim.groupby(['subject_id', 'condition'])['run'].nunique().reset_index()
    h2_run_counts.columns = ['subject_id', 'condition', 'n_clean_runs']
    h2_pivot = h2_run_counts.pivot(index='subject_id', columns='condition', values='n_clean_runs').fillna(0)
    
    # Subjects meeting threshold in BOTH conditions
    h2_eligible = h2_pivot[
        (h2_pivot.get('active', 0) >= min_clean_runs) &
        (h2_pivot.get('sham', 0) >= min_clean_runs)
    ].index.tolist()
    
    # Filter to eligible subjects
    data_h2 = data_h2_prelim[data_h2_prelim['subject_id'].isin(h2_eligible)].copy()
    
    if verbose:
        n_h1 = data_h1['subject_id'].nunique()
        n_h2 = len(h2_eligible)
        print(f'H1 sample: {n_h1} subjects ({len(data_h1)} trials)')
        print(f'H2 sample: {n_h2} subjects ({len(data_h2)} trials)')
    
    return data_h1, data_h2, data_clean, h2_eligible


# =============================================================================
# Main Pipeline Function
# =============================================================================

def apply_all_exclusions(
    data: pd.DataFrame,
    min_clean_runs: int = MIN_CLEAN_RUNS_PER_CONDITION,
    verbose: bool = True
) -> Dict:
    """
    Full exclusion pipeline: compute criteria, apply flags, create filtered datasets.
    
    Parameters
    ----------
    data : DataFrame
        Trial-level data (output from data_loading.load_all_subjects)
    min_clean_runs : int
        Minimum clean runs per condition for H2
    verbose : bool
        If True, print detailed summary
    
    Returns
    -------
    dict with keys:
        'data': original data with exclusion flags added
        'data_h1': filtered for H1 analyses
        'data_h2': filtered for H2 analyses  
        'data_clean': behavioral exclusions applied, all conditions
        'run_exclusions': run-level exclusion table
        'h2_eligible': list of H2-eligible subject IDs
    """
    # Step 1: Compute run-level behavioral exclusions
    if verbose:
        print('='*70)
        print('Computing Pre-Registered Exclusion Criteria')
        print('='*70)
        print()
    
    run_exclusions = compute_exclusion_criteria(data)
    
    # Add condition labels
    run_condition_map = data.groupby(['subject_id', 'run'])['condition'].first().reset_index()
    run_exclusions = run_exclusions.merge(run_condition_map, on=['subject_id', 'run'], how='left')
    
    # Step 2: Apply stimulation exclusions
    run_exclusions = apply_stim_exclusions(run_exclusions)
    
    # Step 3: Tag trial-level data
    data = tag_trial_exclusions(data, run_exclusions)
    
    # Step 4: Create filtered datasets
    data_h1, data_h2, data_clean, h2_eligible = create_filtered_datasets(
        data, min_clean_runs, verbose=False
    )
    
    # Print summary
    if verbose:
        _print_exclusion_summary(data, run_exclusions, data_h1, data_h2, h2_eligible, min_clean_runs)
    
    return {
        'data': data,
        'data_h1': data_h1,
        'data_h2': data_h2,
        'data_clean': data_clean,
        'run_exclusions': run_exclusions,
        'h2_eligible': h2_eligible,
    }


def _print_exclusion_summary(
    data: pd.DataFrame,
    run_exclusions: pd.DataFrame,
    data_h1: pd.DataFrame,
    data_h2: pd.DataFrame,
    h2_eligible: List[str],
    min_clean_runs: int
):
    """Print detailed exclusion summary."""
    
    # Behavioral criteria summary
    print('Behavioral Exclusion Summary (Run-Level):')
    print('-'*50)
    n_runs = len(run_exclusions)
    for criterion, label in [
        ('flag_missed', '>20% missed trials'),
        ('flag_side_bias', 'Side bias (>95%)'),
        ('flag_stim_bias', 'Stimulus bias (>95%)'),
        ('flag_rapid_rt', 'Rapid responding (<200ms)'),
        ('flag_feedback_invariance', 'Feedback invariance (10+ losses)')
    ]:
        n_flagged = run_exclusions[criterion].sum()
        pct = 100 * n_flagged / n_runs if n_runs > 0 else 0
        status = '⚠' if n_flagged > 0 else '✓'
        print(f'  {status} {label}: {n_flagged}/{n_runs} runs ({pct:.0f}%)')
    
    print()
    
    # Stimulation exclusions
    n_stim = run_exclusions['exclude_stim'].sum()
    print(f'Stimulation Administration Exclusions: {n_stim} runs')
    if n_stim > 0:
        stim_flagged = run_exclusions[run_exclusions['exclude_stim']]
        for _, row in stim_flagged.iterrows():
            print(f"  sub-{row['subject_id']} Run {row['run']}: {row['stim_exclusion_reason']}")
    print()
    
    # Combined summary
    print('='*70)
    print('Combined Exclusion Summary')
    print('='*70)
    print()
    
    n_subjects = data['subject_id'].nunique()
    n_behavioral = run_exclusions['exclude_behavioral'].sum()
    n_h2_excluded = run_exclusions['exclude_h2'].sum()
    
    print(f'Total subjects: {n_subjects}')
    print(f'Total runs: {n_runs}')
    print(f'Runs excluded (behavioral): {n_behavioral}')
    print(f'Runs excluded (stim admin): {n_stim}')
    print(f'Unique runs excluded for H2: {n_h2_excluded}')
    print()
    
    # H1 sample
    n_h1_subjects = data_h1['subject_id'].nunique()
    n_h1_runs = data_h1.groupby('subject_id')['run'].nunique().sum()
    print(f'H1 sample (sham, behavioral exclusions):')
    print(f'  {n_h1_subjects} subjects, {n_h1_runs} clean sham runs, {len(data_h1)} trials')
    print()
    
    # H2 sample
    print(f'H2 sample (active vs. sham, all exclusions + min-run threshold):')
    print(f'  {len(h2_eligible)} subjects eligible for paired comparisons, {len(data_h2)} trials')
    
    # List excluded from H2
    all_subjects = set(data['subject_id'].unique())
    h2_excluded = all_subjects - set(h2_eligible)
    if h2_excluded:
        print(f'\n  Subjects excluded from H2:')
        for sid in sorted(h2_excluded):
            print(f'    sub-{sid}')


# =============================================================================
# Utility Functions
# =============================================================================

def get_exclusion_reasons(subject_id: str, run: int, run_exclusions: pd.DataFrame) -> List[str]:
    """Get list of exclusion reasons for a specific run."""
    row = run_exclusions[
        (run_exclusions['subject_id'] == str(subject_id)) &
        (run_exclusions['run'] == int(run))
    ]
    
    if len(row) == 0:
        return []
    
    row = row.iloc[0]
    reasons = []
    
    if row.get('flag_missed', False):
        reasons.append(f"missed trials ({row['pct_missed']:.0f}%)")
    if row.get('flag_side_bias', False):
        reasons.append(f"side bias ({row['max_side_pct']:.0f}%)")
    if row.get('flag_stim_bias', False):
        reasons.append(f"stimulus bias ({row['max_stim_pct']:.0f}%)")
    if row.get('flag_rapid_rt', False):
        reasons.append(f"rapid RT ({row['median_rt_ms']:.0f}ms)")
    if row.get('flag_feedback_invariance', False):
        reasons.append(f"feedback invariance ({row['max_consec_loss_no_shift']} losses)")
    if row.get('exclude_stim', False):
        reasons.append(f"stim admin error")
    
    return reasons


def summarize_exclusions_by_subject(run_exclusions: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize exclusions at the subject level.
    
    Returns DataFrame with: subject_id, n_runs, n_excluded, n_clean, exclusion_rate
    """
    summary = run_exclusions.groupby('subject_id').agg({
        'run': 'count',
        'exclude_behavioral': 'sum',
        'exclude_stim': 'sum',
        'exclude_h2': 'sum',
    }).reset_index()
    
    summary.columns = ['subject_id', 'n_runs', 'n_behavioral', 'n_stim', 'n_excluded_h2']
    summary['n_clean_h2'] = summary['n_runs'] - summary['n_excluded_h2']
    summary['exclusion_rate'] = 100 * summary['n_excluded_h2'] / summary['n_runs']
    
    return summary.sort_values('subject_id')


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing exclusions module...")
    print(f"Exclusion thresholds: {EXCLUSION_THRESHOLDS}")
    print(f"Stim exclusions registered: {len(STIM_EXCLUSIONS)}")
    print(f"Min clean runs for H2: {MIN_CLEAN_RUNS_PER_CONDITION}")
