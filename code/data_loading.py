"""
data_loading.py — Data loading and preprocessing for tACS Bandit analyses

Functions for:
- Loading trial-level CSV files from disk
- Handling duplicate run files (false starts)
- Assigning condition labels based on counterbalance
- Adding WSLS (win-stay/lose-shift) trial-level variables
- Loading all subjects and producing the master DataFrame
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from config import (
    SUBJECT_INFO,
    DATA_DIR,
    EXCLUDE_PREFIXES,
    DISSERTATION_SUBJECTS,
    get_condition_map,
)


# =============================================================================
# File Loading Utilities
# =============================================================================

def get_latest_run_files(subject_dir: Path, verbose: bool = True) -> Dict[int, str]:
    """
    For a given subject directory, find all CSV files and return only the
    latest file for each run number (handles false starts/restarts).
    
    Parameters
    ----------
    subject_dir : Path
        Path to subject's data directory (e.g., DATA_DIR / 'sub-10998')
    verbose : bool
        If True, print info when multiple files found for same run
    
    Returns
    -------
    dict
        {run_number: filepath} mapping
    """
    csv_files = sorted(glob.glob(str(subject_dir / '*.csv')))
    
    run_files = {}  # run_num -> list of (mtime, filepath)
    for f in csv_files:
        fname = os.path.basename(f)
        # Parse run number from filename: sub-XXXXX_ses-1_run-XX_task-bandit_DATE.csv
        parts = fname.split('_')
        run_str = [p for p in parts if p.startswith('run-') or p.startswith('run0')]
        if not run_str:
            continue
        
        # Handle both 'run-01' and 'run01' formats
        run_num_str = run_str[0].replace('run-', '').replace('run', '')
        try:
            run_num = int(run_num_str)
        except ValueError:
            continue
        
        # Use file modification time as tiebreaker (later = more recent)
        mtime = os.path.getmtime(f)
        
        if run_num not in run_files:
            run_files[run_num] = []
        run_files[run_num].append((mtime, f))
    
    # Keep only the latest file per run
    latest = {}
    for run_num, files in run_files.items():
        files.sort(key=lambda x: x[0])
        latest[run_num] = files[-1][1]  # last = most recent
        if len(files) > 1 and verbose:
            print(f'  Run {run_num}: {len(files)} files found, using latest')
    
    return latest


def load_subject_data(
    subject_id: str, 
    subject_dir: Path,
    subject_info: Optional[Dict] = None,
    verbose: bool = True
) -> Optional[pd.DataFrame]:
    """
    Load and concatenate all runs for a single subject.
    Adds condition labels based on counterbalance order.
    
    Parameters
    ----------
    subject_id : str
        Subject ID (e.g., '10998')
    subject_dir : Path
        Path to subject's data directory
    subject_info : dict, optional
        Subject info dict; if None, uses SUBJECT_INFO from config
    verbose : bool
        If True, print loading progress
    
    Returns
    -------
    DataFrame or None
        Trial-level data with condition labels, or None if no data found
    """
    if subject_info is None:
        subject_info = SUBJECT_INFO
    
    if verbose:
        print(f'Loading sub-{subject_id}...')
    
    run_files = get_latest_run_files(subject_dir, verbose=verbose)
    
    if not run_files:
        if verbose:
            print(f'  WARNING: No CSV files found')
        return None
    
    dfs = []
    for run_num in sorted(run_files.keys()):
        df = pd.read_csv(run_files[run_num])
        dfs.append(df)
        if verbose:
            n_trials = len(df)
            n_missed = df['choice'].isna().sum()
            print(f'  Run {run_num}: {n_trials} trials, {n_missed} missed')
    
    data = pd.concat(dfs, ignore_index=True)
    
    # Add condition labels based on counterbalance order
    cb = subject_info.get(subject_id, {}).get('counterbalance', 'UNKNOWN')
    data['counterbalance'] = cb
    
    # Map runs to condition phases
    condition_map = get_condition_map(cb)
    if not condition_map and verbose:
        print(f'  WARNING: Unknown counterbalance "{cb}" for sub-{subject_id}')
    
    data['condition'] = data['run'].map(condition_map)
    data['subject_id'] = str(subject_id)
    
    return data


# =============================================================================
# Trial-Level Preprocessing
# =============================================================================

def preprocess_trials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trial-level WSLS variables within each subject and run.
    
    Adds columns:
        prev_choice  : choice on previous trial (within run)
        prev_reward  : reward on previous trial (within run)
        stay         : 1 if same choice as previous trial, else 0
        shift        : 1 if different choice from previous trial, else 0
        valid_wsls   : True if trial can be used for WSLS analysis
    
    Parameters
    ----------
    df : DataFrame
        Trial-level data with columns: subject_id, run, choice, reward
    
    Returns
    -------
    DataFrame
        Input data with WSLS columns added
    """
    df = df.copy()
    
    # Previous trial's choice and reward (within each run)
    df['prev_choice'] = df.groupby(['subject_id', 'run'])['choice'].shift(1)
    df['prev_reward'] = df.groupby(['subject_id', 'run'])['reward'].shift(1)
    
    # Stay = same choice as previous trial
    df['stay'] = (df['choice'] == df['prev_choice']).astype(float)
    
    # Shift = different choice from previous trial
    df['shift'] = (df['choice'] != df['prev_choice']).astype(float)
    
    # Mark valid trials (not first trial in run, no missed responses)
    df['valid_wsls'] = (
        df['prev_choice'].notna() & 
        df['choice'].notna() & 
        df['prev_reward'].notna()
    )
    
    return df


# =============================================================================
# Main Loading Function
# =============================================================================

def load_all_subjects(
    data_dir: Optional[Path] = None,
    subject_info: Optional[Dict] = None,
    exclude_prefixes: Optional[List[str]] = None,
    preprocess: bool = True,
    verbose: bool = True,
    sample: str = 'dissertation',
) -> pd.DataFrame:
    """
    Load trial-level data for all subjects in the subject registry.
    
    Parameters
    ----------
    data_dir : Path, optional
        Directory containing subject folders; defaults to DATA_DIR from config
    subject_info : dict, optional
        Subject registry; defaults to SUBJECT_INFO from config
    exclude_prefixes : list, optional
        Subject ID prefixes to skip; defaults to EXCLUDE_PREFIXES from config
    preprocess : bool
        If True, add WSLS columns via preprocess_trials()
    verbose : bool
        If True, print loading progress
    sample : str
        Which subject pool to use:
        - 'dissertation' : frozen N=39 sample (default, backward-compatible)
        - 'all'          : all subjects in SUBJECT_INFO
        - 'new'          : only subjects added after the dissertation freeze
    
    Returns
    -------
    DataFrame
        Concatenated trial-level data for all subjects
    
    Raises
    ------
    ValueError
        If no data could be loaded or invalid sample specified
    """
    if sample not in ('dissertation', 'all', 'new'):
        raise ValueError(f"sample must be 'dissertation', 'all', or 'new'; got '{sample}'")
    
    if data_dir is None:
        data_dir = DATA_DIR
    if subject_info is None:
        subject_info = SUBJECT_INFO
    if exclude_prefixes is None:
        exclude_prefixes = EXCLUDE_PREFIXES
    
    # Determine which subjects to include based on sample
    if sample == 'dissertation':
        allowed_subjects = set(DISSERTATION_SUBJECTS)
    elif sample == 'new':
        allowed_subjects = set(subject_info.keys()) - set(DISSERTATION_SUBJECTS)
    else:  # 'all'
        allowed_subjects = None  # no additional filter
    
    if verbose and sample != 'all':
        pool_size = len(allowed_subjects) if allowed_subjects else len(subject_info)
        print(f'Sample: {sample} ({pool_size} subjects in pool)')
    
    all_data = []
    
    for sub_dir in sorted(data_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        
        sub_id = sub_dir.name.replace('sub-', '')
        
        # Skip pilot/test subjects
        if any(sub_id.startswith(prefix) or sub_id == prefix for prefix in exclude_prefixes):
            continue
        
        # Skip subjects not in our tracker
        if sub_id not in subject_info:
            continue
        
        # Apply sample filter
        if allowed_subjects is not None and sub_id not in allowed_subjects:
            continue
        
        df = load_subject_data(sub_id, sub_dir, subject_info, verbose=verbose)
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        raise ValueError(f"No data loaded from {data_dir} (sample='{sample}')")
    
    data = pd.concat(all_data, ignore_index=True)
    
    if verbose:
        print(f'\nLoaded {len(data)} total trials from {data["subject_id"].nunique()} subjects')
    
    # Add WSLS preprocessing
    if preprocess:
        data = preprocess_trials(data)
        if verbose:
            valid = data[data['valid_wsls']]
            print(f'Valid WSLS trials: {len(valid)} / {len(data)} ({100*len(valid)/len(data):.1f}%)')
    
    return data


# =============================================================================
# Convenience Functions
# =============================================================================

def get_subject_data(subject_id: str, data: pd.DataFrame) -> pd.DataFrame:
    """Extract data for a single subject."""
    return data[data['subject_id'] == str(subject_id)].copy()


def get_condition_data(condition: str, data: pd.DataFrame) -> pd.DataFrame:
    """Extract data for a single condition (e.g., 'sham', 'active')."""
    return data[data['condition'] == condition].copy()


def get_run_data(subject_id: str, run: int, data: pd.DataFrame) -> pd.DataFrame:
    """Extract data for a specific subject and run."""
    return data[
        (data['subject_id'] == str(subject_id)) & 
        (data['run'] == run)
    ].copy()


def summarize_loading(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary table of loaded data by subject.
    
    Returns DataFrame with columns: subject_id, n_runs, n_trials, n_missed, 
                                    counterbalance, conditions
    """
    summary = []
    for sub_id, group in data.groupby('subject_id'):
        summary.append({
            'subject_id': sub_id,
            'n_runs': group['run'].nunique(),
            'n_trials': len(group),
            'n_missed': group['choice'].isna().sum(),
            'pct_missed': 100 * group['choice'].isna().mean(),
            'counterbalance': group['counterbalance'].iloc[0],
            'conditions': sorted(group['condition'].dropna().unique().tolist()),
        })
    
    return pd.DataFrame(summary).sort_values('subject_id')


# =============================================================================
# Quick Stats
# =============================================================================

def print_wsls_summary(data: pd.DataFrame):
    """Print overall WSLS rates from preprocessed data."""
    if 'valid_wsls' not in data.columns:
        print("Data not preprocessed. Run preprocess_trials() first.")
        return
    
    valid = data[data['valid_wsls']]
    
    print(f'Valid WSLS trials: {len(valid)} / {len(data)} ({100*len(valid)/len(data):.1f}%)')
    
    win_trials = valid[valid['prev_reward'] == True]
    lose_trials = valid[valid['prev_reward'] == False]
    
    if len(win_trials) > 0:
        print(f'Overall p(stay|win): {win_trials["stay"].mean():.3f}')
    if len(lose_trials) > 0:
        print(f'Overall p(shift|lose): {lose_trials["shift"].mean():.3f}')


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    # Test loading if run directly
    print("Testing data_loading module...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Subjects in registry: {len(SUBJECT_INFO)}")
    
    if DATA_DIR.exists():
        try:
            data = load_all_subjects(verbose=True)
            print("\n" + "="*50)
            print("Loading Summary:")
            print(summarize_loading(data).to_string(index=False))
        except Exception as e:
            print(f"Could not load data: {e}")
    else:
        print(f"Data directory not found: {DATA_DIR}")
