"""
theta_visualization.py — Theta reactivity visualization for tACS Bandit study

Creates publication-quality figures comparing high vs. low theta reactivity subjects.
Designed to explain what theta reactivity means as an individual difference.

Main visualization (Concept 1):
- Top row: Continuous theta power timecourse for one run
- Middle row: Trial-locked theta power (feedback-locked)
- Bottom row: Distribution of theta power values
"""

import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import warnings

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    SUBJECT_INFO,
    DATA_DIR,
    EEG_DIR,
    PLOTLY_TEMPLATE,
    FONT_FAMILY,
    COLOR_SHAM as COLOR_BLUE,
    COLOR_ACTIVE as COLOR_RED,
    COLOR_GOLD,
)

from eeg_theta import (
    FS,
    EEG_CH_LABELS,
    THETA_BAND,
    ARTIFACT_THRESH_UV,
    SKIP_FIRST_SEC,
    find_eeg_run,
    get_latest_behavioral_file,
    load_eeg_run,
    preprocess_eeg,
    bandpass_filter,
    align_timestamps,
    extract_epochs,
    compute_band_power_timecourse,
)

warnings.filterwarnings('ignore')


# =============================================================================
# Constants
# =============================================================================

# Colors for high vs low theta
COLOR_HIGH_THETA = '#E64A19'  # Orange-red (same as COLOR_ACTIVE)
COLOR_LOW_THETA = '#1565C0'   # Blue (same as COLOR_SHAM)

# Visualization parameters
TIMECOURSE_DURATION_SEC = 60  # Show first 60 seconds of run
EPOCH_PRE = 0.5               # Seconds before feedback
EPOCH_POST = 1.0              # Seconds after feedback


# =============================================================================
# Data Extraction for Visualization
# =============================================================================

def extract_theta_timecourse(
    subject_id: str,
    run_num: int = 1,
    channel_idx: int = 0,
    eeg_dir: Optional[Path] = None,
    artifact_thresh_uv: float = ARTIFACT_THRESH_UV,
    skip_first_sec: float = SKIP_FIRST_SEC
) -> Optional[Dict]:
    """
    Extract continuous theta power timecourse for visualization.
    
    Parameters
    ----------
    subject_id : str
        Subject ID
    run_num : int
        Run number (default 1 = first baseline run)
    channel_idx : int
        EEG channel index (0=F4, 1=P4, 2=P3)
    eeg_dir : Path, optional
        EEG data directory
    artifact_thresh_uv : float
        Artifact rejection threshold in µV
    skip_first_sec : float
        Skip first N seconds of recording
    
    Returns
    -------
    dict with timecourse data or None if failed
    """
    if eeg_dir is None:
        eeg_dir = EEG_DIR
    
    eeg_path = find_eeg_run(eeg_dir, subject_id, run_num)
    if eeg_path is None:
        print(f"No EEG file found for sub-{subject_id} run-{run_num}")
        return None
    
    try:
        eeg_data = load_eeg_run(eeg_path)
    except Exception as e:
        print(f"Error loading EEG for sub-{subject_id}: {e}")
        return None
    
    # Preprocess
    sub_info = SUBJECT_INFO.get(str(subject_id), {})
    has_earclip = sub_info.get('earclip', False)
    eeg_data = preprocess_eeg(eeg_data, has_earclip=has_earclip)
    
    # Extract raw and theta-filtered signals
    raw_signal = eeg_data['eeg'][:, channel_idx].copy()
    eeg_times = eeg_data['time_sec']
    
    # Theta filtering
    theta_filtered = bandpass_filter(raw_signal, THETA_BAND, FS)
    
    # Instantaneous theta power via Hilbert
    analytic_signal = signal.hilbert(theta_filtered)
    theta_power = np.abs(analytic_signal) ** 2
    
    # Artifact masking
    artifact_mask = np.abs(raw_signal) > artifact_thresh_uv
    startup_mask = eeg_times < skip_first_sec
    combined_mask = artifact_mask | startup_mask
    
    # Smooth theta power
    window_samples = int(0.1 * FS)  # 100ms smoothing
    theta_power_smooth = np.convolve(
        theta_power, 
        np.ones(window_samples) / window_samples, 
        mode='same'
    )
    theta_power_smooth[combined_mask] = np.nan
    
    # Normalize to percent change from mean
    clean_vals = theta_power_smooth[~np.isnan(theta_power_smooth)]
    if len(clean_vals) == 0:
        return None
    
    clean_mean = np.nanmean(theta_power_smooth)
    theta_pct = (theta_power_smooth / clean_mean - 1) * 100
    
    return {
        'subject_id': subject_id,
        'run': run_num,
        'channel': EEG_CH_LABELS[channel_idx],
        'time_sec': eeg_times,
        'raw_signal': raw_signal,
        'theta_filtered': theta_filtered,
        'theta_power_pct': theta_pct,
        'artifact_mask': combined_mask,
        'theta_p95': np.nanpercentile(theta_pct, 95),
        'theta_median': np.nanmedian(theta_pct),
        'has_earclip': has_earclip,
    }


def extract_theta_epochs(
    subject_id: str,
    run_num: int = 1,
    channel_idx: int = 0,
    eeg_dir: Optional[Path] = None,
    behav_dir: Optional[Path] = None,
    epoch_pre: float = EPOCH_PRE,
    epoch_post: float = EPOCH_POST,
) -> Optional[Dict]:
    """
    Extract feedback-locked theta epochs for visualization.
    
    Parameters
    ----------
    subject_id : str
        Subject ID
    run_num : int
        Run number
    channel_idx : int
        EEG channel index
    eeg_dir : Path, optional
        EEG data directory
    behav_dir : Path, optional
        Behavioral data directory
    epoch_pre : float
        Seconds before feedback
    epoch_post : float
        Seconds after feedback
    
    Returns
    -------
    dict with epoch data or None if failed
    """
    if eeg_dir is None:
        eeg_dir = EEG_DIR
    if behav_dir is None:
        behav_dir = DATA_DIR  # DATA_DIR already points to .../data/bandit
    
    # Load EEG
    eeg_path = find_eeg_run(eeg_dir, subject_id, run_num)
    if eeg_path is None:
        print(f"  No EEG file found for sub-{subject_id} run-{run_num}")
        return None
    
    try:
        eeg_data = load_eeg_run(eeg_path)
    except Exception as e:
        print(f"  Error loading EEG: {e}")
        return None
    
    # Preprocess
    sub_info = SUBJECT_INFO.get(str(subject_id), {})
    has_earclip = sub_info.get('earclip', False)
    eeg_data = preprocess_eeg(eeg_data, has_earclip=has_earclip)
    
    # Load behavioral data
    sub_behav_dir = behav_dir / f'sub-{subject_id}'
    behav_path = get_latest_behavioral_file(sub_behav_dir, run_num)
    if behav_path is None:
        print(f"  No behavioral file found for sub-{subject_id} run-{run_num}")
        return None
    
    try:
        behav_df = pd.read_csv(behav_path)
    except Exception as e:
        print(f"  Error loading behavioral data: {e}")
        return None
    
    # Compute feedback times from behavioral data
    # Remove missed trials
    behav_df = behav_df.dropna(subset=['rt']).copy()
    
    if len(behav_df) == 0:
        print(f"  No valid trials for sub-{subject_id} run-{run_num}")
        return None
    
    # Convert RT and ITI from ms to seconds (if needed)
    if 'rt' in behav_df.columns:
        # Check if already in seconds (values < 10) or milliseconds (values > 100)
        if behav_df['rt'].mean() > 100:
            behav_df['rt_sec'] = behav_df['rt'] / 1000.0
        else:
            behav_df['rt_sec'] = behav_df['rt']
    
    if 'iti' in behav_df.columns:
        if behav_df['iti'].mean() > 100:
            behav_df['iti_sec'] = behav_df['iti'] / 1000.0
        else:
            behav_df['iti_sec'] = behav_df['iti']
    else:
        behav_df['iti_sec'] = 1.0  # Default ITI
    
    # Work backwards from timestamp to get feedback onset
    # timestamp = feedback_onset + FEEDBACK_DURATION + ITI
    FEEDBACK_DURATION = 1.0  # seconds
    
    if 'timestamp' not in behav_df.columns:
        print(f"  No timestamp column for sub-{subject_id} run-{run_num}")
        return None
    
    behav_df['feedback_time'] = behav_df['timestamp'] - FEEDBACK_DURATION - behav_df['iti_sec']
    
    # Compute relative timing (zero to first feedback)
    first_feedback = behav_df['feedback_time'].iloc[0]
    behav_df['feedback_time_relative'] = behav_df['feedback_time'] - first_feedback
    
    # Align to EEG time
    try:
        behav_df = align_timestamps(eeg_data, behav_df)
    except Exception as e:
        print(f"  Alignment failed for sub-{subject_id}: {e}")
        return None
    
    # Get feedback times in EEG coordinates
    feedback_times = behav_df['feedback_time_eeg'].values
    
    # Extract epochs
    epoch_result = extract_epochs(
        eeg_data, 
        feedback_times,
        pre_sec=epoch_pre,
        post_sec=epoch_post,
        reject_artifacts=True
    )
    
    if epoch_result['n_valid'] == 0:
        print(f"  All epochs rejected for sub-{subject_id} run-{run_num}")
        return None
    
    # Compute theta power for epochs
    epochs = epoch_result['epochs'][:, :, channel_idx:channel_idx+1]
    times = epoch_result['times']
    
    theta_power = compute_band_power_timecourse(
        epochs, times, THETA_BAND, FS
    )[:, :, 0]  # Shape: (n_epochs, n_samples)
    
    # Baseline normalize
    baseline_mask = times < 0
    baseline_power = theta_power[:, baseline_mask].mean(axis=1, keepdims=True)
    baseline_power = np.maximum(baseline_power, 1e-10)
    theta_pct = (theta_power / baseline_power - 1) * 100
    
    print(f"  Extracted {epoch_result['n_valid']} epochs for sub-{subject_id}")
    
    return {
        'subject_id': subject_id,
        'run': run_num,
        'channel': EEG_CH_LABELS[channel_idx],
        'times': times,
        'theta_power_epochs': theta_pct,
        'n_epochs': epoch_result['n_valid'],
        'n_rejected': epoch_result['n_rejected'],
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def extract_theta_epochs_by_outcome(
    subject_id: str,
    run_num: int = 1,
    channel_idx: int = 0,
    eeg_dir: Optional[Path] = None,
    behav_dir: Optional[Path] = None,
    epoch_pre: float = EPOCH_PRE,
    epoch_post: float = EPOCH_POST,
) -> Optional[Dict]:
    """
    Extract feedback-locked theta epochs split by outcome (win vs loss).
    
    Win = rewarded trial (received points)
    Loss = unrewarded trial (no points)
    
    Note: This is based on the 'reward' column, not 'correct'. In an 80/20 
    probabilistic task, reward and accuracy are correlated but not identical.
    
    Returns
    -------
    dict with 'win_epochs', 'loss_epochs', 'times', diagnostic counts, or None if failed
    """
    if eeg_dir is None:
        eeg_dir = EEG_DIR
    if behav_dir is None:
        behav_dir = DATA_DIR
    
    # Load EEG
    eeg_path = find_eeg_run(eeg_dir, subject_id, run_num)
    if eeg_path is None:
        return None
    
    try:
        eeg_data = load_eeg_run(eeg_path)
    except Exception:
        return None
    
    # Preprocess
    sub_info = SUBJECT_INFO.get(str(subject_id), {})
    has_earclip = sub_info.get('earclip', False)
    eeg_data = preprocess_eeg(eeg_data, has_earclip=has_earclip)
    
    # Load behavioral data
    sub_behav_dir = behav_dir / f'sub-{subject_id}'
    behav_path = get_latest_behavioral_file(sub_behav_dir, run_num)
    if behav_path is None:
        return None
    
    try:
        behav_df = pd.read_csv(behav_path)
    except Exception:
        return None
    
    # Remove missed trials
    behav_df = behav_df.dropna(subset=['rt']).copy()
    
    if len(behav_df) == 0:
        return None
    
    # Convert RT and ITI
    if behav_df['rt'].mean() > 100:
        behav_df['rt_sec'] = behav_df['rt'] / 1000.0
    else:
        behav_df['rt_sec'] = behav_df['rt']
    
    if 'iti' in behav_df.columns:
        if behav_df['iti'].mean() > 100:
            behav_df['iti_sec'] = behav_df['iti'] / 1000.0
        else:
            behav_df['iti_sec'] = behav_df['iti']
    else:
        behav_df['iti_sec'] = 1.0
    
    # Compute feedback times
    FEEDBACK_DURATION = 1.0
    if 'timestamp' not in behav_df.columns:
        return None
    
    behav_df['feedback_time'] = behav_df['timestamp'] - FEEDBACK_DURATION - behav_df['iti_sec']
    first_feedback = behav_df['feedback_time'].iloc[0]
    behav_df['feedback_time_relative'] = behav_df['feedback_time'] - first_feedback
    
    # Align to EEG
    try:
        behav_df = align_timestamps(eeg_data, behav_df)
    except Exception:
        return None
    
    # Parse REWARD column (win = rewarded, loss = not rewarded)
    if 'reward' not in behav_df.columns:
        print(f"  WARNING: No 'reward' column for sub-{subject_id} run-{run_num}")
        return None
    
    reward_col = behav_df['reward']
    # Check dtype and convert appropriately
    if reward_col.dtype == bool:
        behav_df['is_win'] = reward_col
    elif reward_col.dtype in ['int64', 'float64', 'int', 'float']:
        behav_df['is_win'] = reward_col.astype(bool)
    elif reward_col.dtype == object:
        # String values like 'True'/'False'
        behav_df['is_win'] = reward_col.astype(str).str.lower().isin(['true', '1', 'yes'])
    else:
        behav_df['is_win'] = reward_col.astype(bool)
    
    # Split by outcome
    win_trials = behav_df[behav_df['is_win']]
    loss_trials = behav_df[~behav_df['is_win']]
    
    n_win_behavioral = len(win_trials)
    n_loss_behavioral = len(loss_trials)
    
    win_times = win_trials['feedback_time_eeg'].values
    loss_times = loss_trials['feedback_time_eeg'].values
    
    # Extract epochs for each outcome
    results = {
        'times': None, 
        'win_epochs': None, 
        'loss_epochs': None,
        'n_win_behavioral': n_win_behavioral,
        'n_loss_behavioral': n_loss_behavioral,
        'n_win_valid': 0,
        'n_loss_valid': 0,
        'n_win_rejected': 0,
        'n_loss_rejected': 0,
    }
    
    for outcome, times_arr, epochs_key, valid_key, rejected_key in [
        ('win', win_times, 'win_epochs', 'n_win_valid', 'n_win_rejected'), 
        ('loss', loss_times, 'loss_epochs', 'n_loss_valid', 'n_loss_rejected')
    ]:
        if len(times_arr) == 0:
            continue
            
        epoch_result = extract_epochs(
            eeg_data, times_arr,
            pre_sec=epoch_pre, post_sec=epoch_post,
            reject_artifacts=True
        )
        
        results[valid_key] = epoch_result['n_valid']
        results[rejected_key] = epoch_result['n_rejected']
        
        if epoch_result['n_valid'] == 0:
            continue
        
        epochs = epoch_result['epochs'][:, :, channel_idx:channel_idx+1]
        times = epoch_result['times']
        
        # Compute theta power
        theta_power = compute_band_power_timecourse(epochs, times, THETA_BAND, FS)[:, :, 0]
        
        # Baseline normalize
        baseline_mask = times < 0
        baseline_power = theta_power[:, baseline_mask].mean(axis=1, keepdims=True)
        baseline_power = np.maximum(baseline_power, 1e-10)
        theta_pct = (theta_power / baseline_power - 1) * 100
        
        results[epochs_key] = theta_pct
        results['times'] = times
    
    if results['win_epochs'] is None and results['loss_epochs'] is None:
        return None
    
    results['subject_id'] = subject_id
    results['run'] = run_num
    
    return results


def compute_grand_average_theta(
    subject_ids: List[str],
    run_nums: List[int] = [1, 5],
    channel_idx: int = 0,
    eeg_dir: Optional[Path] = None,
    behav_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Optional[Dict]:
    """
    Compute grand average feedback-locked theta across all subjects.
    
    Averages within subject first (across runs), then across subjects.
    Splits by outcome (win = rewarded vs loss = not rewarded).
    
    Returns
    -------
    dict with 'times', 'win_mean', 'win_sem', 'loss_mean', 'loss_sem', etc.
    """
    all_win_epochs = []
    all_loss_epochs = []
    times = None
    
    subjects_included = []
    total_win_trials = 0
    total_loss_trials = 0
    
    for subj in subject_ids:
        subj_win = []
        subj_loss = []
        subj_n_win = 0
        subj_n_loss = 0
        runs_with_data = []
        
        for run in run_nums:
            result = extract_theta_epochs_by_outcome(
                subj, run, channel_idx, eeg_dir, behav_dir
            )
            
            if result is None:
                continue
            
            if times is None:
                times = result['times']
            
            run_had_data = False
            
            if result['win_epochs'] is not None:
                subj_win.append(result['win_epochs'])
                subj_n_win += result['n_win_valid']
                run_had_data = True
                
            if result['loss_epochs'] is not None:
                subj_loss.append(result['loss_epochs'])
                subj_n_loss += result['n_loss_valid']
                run_had_data = True
            
            if run_had_data:
                runs_with_data.append(run)
        
        # Average within subject across runs
        if subj_win:
            subj_win_concat = np.vstack(subj_win)
            subj_win_mean = np.nanmean(subj_win_concat, axis=0)
            all_win_epochs.append(subj_win_mean)
            total_win_trials += subj_n_win
        
        if subj_loss:
            subj_loss_concat = np.vstack(subj_loss)
            subj_loss_mean = np.nanmean(subj_loss_concat, axis=0)
            all_loss_epochs.append(subj_loss_mean)
            total_loss_trials += subj_n_loss
        
        if subj_win or subj_loss:
            subjects_included.append(subj)
            if verbose:
                runs_str = ','.join(map(str, runs_with_data))
                print(f"  Sub-{subj} (runs {runs_str}): "
                      f"{subj_n_win} win trials, {subj_n_loss} loss trials")
    
    if not all_win_epochs and not all_loss_epochs:
        print("No valid epochs extracted")
        return None
    
    results = {'times': times, 'subjects_included': subjects_included}
    
    # Grand average for wins
    if all_win_epochs:
        win_matrix = np.vstack(all_win_epochs)  # subjects × timepoints
        results['win_mean'] = np.nanmean(win_matrix, axis=0)
        results['win_sem'] = np.nanstd(win_matrix, axis=0) / np.sqrt(len(all_win_epochs))
        results['win_n_subjects'] = len(all_win_epochs)
        results['win_n_trials'] = total_win_trials
    
    # Grand average for losses
    if all_loss_epochs:
        loss_matrix = np.vstack(all_loss_epochs)
        results['loss_mean'] = np.nanmean(loss_matrix, axis=0)
        results['loss_sem'] = np.nanstd(loss_matrix, axis=0) / np.sqrt(len(all_loss_epochs))
        results['loss_n_subjects'] = len(all_loss_epochs)
        results['loss_n_trials'] = total_loss_trials
    
    if verbose:
        print(f"\nGrand average computed:")
        print(f"  Subjects included: {len(subjects_included)}")
        if 'win_n_subjects' in results:
            print(f"  Win: N = {results['win_n_subjects']} subjects, {results['win_n_trials']} total trials")
        if 'loss_n_subjects' in results:
            print(f"  Loss: N = {results['loss_n_subjects']} subjects, {results['loss_n_trials']} total trials")
    
    return results


def plot_grand_average_theta(
    grand_avg: Dict,
    show_fig: bool = True,
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    Plot grand average feedback-locked theta for win vs loss trials.
    
    Shows mean ± SEM for each outcome type.
    """
    times = grand_avg['times']
    
    fig = go.Figure()
    
    # Loss trials (typically show larger theta — prediction error)
    if 'loss_mean' in grand_avg:
        loss_mean = grand_avg['loss_mean']
        loss_sem = grand_avg['loss_sem']
        
        # SEM shading
        fig.add_trace(go.Scatter(
            x=np.concatenate([times, times[::-1]]),
            y=np.concatenate([loss_mean + loss_sem, (loss_mean - loss_sem)[::-1]]),
            fill='toself',
            fillcolor='rgba(198, 40, 40, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name='Loss SEM'
        ))
        
        # Mean line
        n_subj = grand_avg['loss_n_subjects']
        n_trials = grand_avg['loss_n_trials']
        fig.add_trace(go.Scatter(
            x=times,
            y=loss_mean,
            mode='lines',
            line=dict(color='#C62828', width=3),
            name=f"Loss (N={n_subj}, {n_trials} trials)"
        ))
    
    # Win trials
    if 'win_mean' in grand_avg:
        win_mean = grand_avg['win_mean']
        win_sem = grand_avg['win_sem']
        
        # SEM shading
        fig.add_trace(go.Scatter(
            x=np.concatenate([times, times[::-1]]),
            y=np.concatenate([win_mean + win_sem, (win_mean - win_sem)[::-1]]),
            fill='toself',
            fillcolor='rgba(46, 125, 50, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name='Win SEM'
        ))
        
        # Mean line
        n_subj = grand_avg['win_n_subjects']
        n_trials = grand_avg['win_n_trials']
        fig.add_trace(go.Scatter(
            x=times,
            y=win_mean,
            mode='lines',
            line=dict(color='#2E7D32', width=3),
            name=f"Win (N={n_subj}, {n_trials} trials)"
        ))
    
    # Feedback onset marker
    fig.add_vline(x=0, line_dash='solid', line_color='black', line_width=2)
    fig.add_annotation(
        x=0.02, y=1.0,
        xref='x', yref='paper',
        text="Feedback",
        showarrow=False,
        font=dict(size=12),
        textangle=-90,
        yanchor='top'
    )
    
    # Zero line
    fig.add_hline(y=0, line_dash='dot', line_color='gray', line_width=1)
    
    n_subj = len(grand_avg['subjects_included'])
    
    fig.update_layout(
        title=dict(
            text=f"Grand Average Feedback-Locked Theta Power<br>"
                 f"<sup>N = {n_subj} subjects | Channel: F4 | Theta: 4-8 Hz | Shading: ±SEM</sup>",
            font=dict(size=16),
        ),
        xaxis_title="Time from Feedback (s)",
        yaxis_title="Theta Power (% change from baseline)",
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        legend=dict(
            yanchor='top', y=0.99,
            xanchor='right', x=0.99,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        height=500,
        width=700,
    )
    
    # Clean axes
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor='black', linewidth=1)
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor='black', linewidth=1)
    
    if save_path:
        fig.write_image(save_path, scale=2)
        print(f"Saved: {save_path}")
    
    if show_fig:
        fig.show()
    
    return fig


def plot_theta_timecourse_simple(
    high_subject_id: str,
    low_subject_id: str,
    run_num: int = 1,
    channel_idx: int = 0,
    timecourse_duration: float = 60,
    eeg_dir: Optional[Path] = None,
    show_fig: bool = True,
) -> Optional[go.Figure]:
    """
    Simplified comparison figure — just the continuous timecourse (Row 1 only).
    
    Cleaner visualization for showing high vs low theta reactivity individual differences.
    """
    high_tc = extract_theta_timecourse(high_subject_id, run_num, channel_idx, eeg_dir)
    low_tc = extract_theta_timecourse(low_subject_id, run_num, channel_idx, eeg_dir)
    
    if high_tc is None or low_tc is None:
        print("Failed to extract timecourse data")
        return None
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"High Theta Reactivity (Sub-{high_subject_id})",
            f"Low Theta Reactivity (Sub-{low_subject_id})",
        ),
        horizontal_spacing=0.08,
    )
    
    # Match y-axis range
    y_max = max(
        np.nanpercentile(high_tc['theta_power_pct'], 99.5),
        np.nanpercentile(low_tc['theta_power_pct'], 99.5)
    )
    y_min = min(
        np.nanpercentile(high_tc['theta_power_pct'], 0.5),
        np.nanpercentile(low_tc['theta_power_pct'], 0.5)
    )
    
    for col, (tc_data, color) in enumerate([
        (high_tc, COLOR_HIGH_THETA),
        (low_tc, COLOR_LOW_THETA)
    ], start=1):
        
        time_mask = tc_data['time_sec'] <= timecourse_duration
        t = tc_data['time_sec'][time_mask]
        theta = tc_data['theta_power_pct'][time_mask]
        
        fig.add_trace(
            go.Scatter(
                x=t, y=theta,
                mode='lines',
                line=dict(color=color, width=1),
                showlegend=False,
            ),
            row=1, col=col
        )
        
        # p95 line
        p95 = tc_data['theta_p95']
        fig.add_hline(
            y=p95, line_dash='dash', line_color='gray',
            annotation_text=f'p95={p95:.0f}%',
            annotation_position='right',
            row=1, col=col
        )
        
        # Zero line
        fig.add_hline(y=0, line_dash='dot', line_color='lightgray', row=1, col=col)
        
        fig.update_yaxes(range=[y_min, y_max], row=1, col=col)
    
    fig.update_layout(
        title=dict(
            text="Theta Reactivity: High vs. Low Individual Differences<br>"
                 f"<sup>Channel: {EEG_CH_LABELS[channel_idx]} | Run: {run_num} | Theta: 4-8 Hz</sup>",
            font=dict(size=16),
        ),
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=400,
        width=1000,
    )
    
    fig.update_yaxes(title_text="Theta Power (% change)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    
    if show_fig:
        fig.show()
    
    return fig


def plot_theta_exemplar_comparison(
    high_subject_id: str,
    low_subject_id: str,
    run_num: int = 1,
    channel_idx: int = 0,
    timecourse_duration: float = TIMECOURSE_DURATION_SEC,
    eeg_dir: Optional[Path] = None,
    behav_dir: Optional[Path] = None,
    show_fig: bool = True,
    save_path: Optional[str] = None,
) -> Optional[go.Figure]:
    """
    Create comparison figure for high vs. low theta reactivity subjects.
    
    Three-row layout:
    - Top: Continuous theta power timecourse
    - Middle: Trial-locked theta power (feedback epochs)
    - Bottom: Distribution of theta power values
    
    Parameters
    ----------
    high_subject_id : str
        Subject ID with high theta reactivity
    low_subject_id : str
        Subject ID with low theta reactivity  
    run_num : int
        Run number to visualize
    channel_idx : int
        EEG channel (0=F4)
    timecourse_duration : float
        Seconds of timecourse to display
    eeg_dir : Path, optional
        EEG data directory
    behav_dir : Path, optional
        Behavioral data directory
    show_fig : bool
        Display figure
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    go.Figure or None if data extraction failed
    """
    print(f"Extracting data for high theta subject: {high_subject_id}")
    high_tc = extract_theta_timecourse(high_subject_id, run_num, channel_idx, eeg_dir)
    high_ep = extract_theta_epochs(high_subject_id, run_num, channel_idx, eeg_dir, behav_dir)
    
    print(f"Extracting data for low theta subject: {low_subject_id}")
    low_tc = extract_theta_timecourse(low_subject_id, run_num, channel_idx, eeg_dir)
    low_ep = extract_theta_epochs(low_subject_id, run_num, channel_idx, eeg_dir, behav_dir)
    
    if high_tc is None or low_tc is None:
        print("Failed to extract timecourse data")
        return None
    
    # Create figure with 3 rows × 2 columns
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            f"High Theta Reactivity (Sub-{high_subject_id})",
            f"Low Theta Reactivity (Sub-{low_subject_id})",
            "Feedback-Locked Theta",
            "Feedback-Locked Theta",
            "Theta Power Distribution",
            "Theta Power Distribution",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        row_heights=[0.4, 0.35, 0.25],
    )
    
    # -------------------------------------------------------------------------
    # Row 1: Continuous timecourse
    # -------------------------------------------------------------------------
    for col, (tc_data, color, label) in enumerate([
        (high_tc, COLOR_HIGH_THETA, 'High'),
        (low_tc, COLOR_LOW_THETA, 'Low')
    ], start=1):
        
        # Trim to display duration
        time_mask = tc_data['time_sec'] <= timecourse_duration
        t = tc_data['time_sec'][time_mask]
        theta = tc_data['theta_power_pct'][time_mask]
        
        fig.add_trace(
            go.Scatter(
                x=t,
                y=theta,
                mode='lines',
                line=dict(color=color, width=1),
                name=f'{label} theta',
                showlegend=False,
            ),
            row=1, col=col
        )
        
        # Add p95 threshold line
        p95 = tc_data['theta_p95']
        fig.add_hline(
            y=p95, 
            line_dash='dash', 
            line_color='gray',
            annotation_text=f'p95={p95:.0f}%',
            annotation_position='right',
            row=1, col=col
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash='dot', line_color='lightgray', row=1, col=col)
    
    # -------------------------------------------------------------------------
    # Row 2: Trial-locked epochs
    # -------------------------------------------------------------------------
    for col, (ep_data, color, label) in enumerate([
        (high_ep, COLOR_HIGH_THETA, 'High'),
        (low_ep, COLOR_LOW_THETA, 'Low')
    ], start=1):
        
        if ep_data is None:
            fig.add_annotation(
                text="Epoch data unavailable",
                xref=f"x{col+2}", yref=f"y{col+2}",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=12, color='gray'),
            )
            continue
        
        times = ep_data['times']
        epochs = ep_data['theta_power_epochs']
        
        # Plot individual epochs (faint)
        for i in range(min(epochs.shape[0], 30)):  # Cap at 30 for clarity
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=epochs[i, :],
                    mode='lines',
                    line=dict(color=color, width=0.5),
                    opacity=0.2,
                    showlegend=False,
                ),
                row=2, col=col
            )
        
        # Plot mean epoch (bold)
        mean_epoch = np.nanmean(epochs, axis=0)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=mean_epoch,
                mode='lines',
                line=dict(color=color, width=3),
                name=f'{label} mean',
                showlegend=False,
            ),
            row=2, col=col
        )
        
        # Add feedback marker
        fig.add_vline(x=0, line_dash='solid', line_color='black', line_width=1, row=2, col=col)
        fig.add_annotation(
            text="Feedback",
            x=0.02, y=0.95,
            xref=f"x{col+2} domain", yref=f"y{col+2} domain",
            showarrow=False,
            font=dict(size=10),
        )
    
    # -------------------------------------------------------------------------
    # Row 3: Distributions
    # -------------------------------------------------------------------------
    for col, (tc_data, color, label) in enumerate([
        (high_tc, COLOR_HIGH_THETA, 'High'),
        (low_tc, COLOR_LOW_THETA, 'Low')
    ], start=1):
        
        theta_vals = tc_data['theta_power_pct']
        clean_vals = theta_vals[~np.isnan(theta_vals)]
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=clean_vals,
                nbinsx=50,
                marker=dict(color=color, line=dict(color='white', width=0.5)),
                opacity=0.7,
                name=f'{label} dist',
                showlegend=False,
            ),
            row=3, col=col
        )
        
        # Add p95 line
        p95 = tc_data['theta_p95']
        fig.add_vline(
            x=p95,
            line_dash='dash',
            line_color='black',
            line_width=2,
            row=3, col=col
        )
        fig.add_annotation(
            text=f"p95",
            x=p95 + 5,
            y=0.9,
            xref=f"x{col+4}",
            yref=f"y{col+4} domain",
            showarrow=False,
            font=dict(size=10),
        )
    
    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------
    fig.update_layout(
        title=dict(
            text="Theta Reactivity: High vs. Low Individual Differences<br>"
                 f"<sup>Channel: {EEG_CH_LABELS[channel_idx]} | Run: {run_num} | "
                 f"Theta band: {THETA_BAND[0]}-{THETA_BAND[1]} Hz</sup>",
            font=dict(size=16),
        ),
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=900,
        width=1000,
        showlegend=False,
    )
    
    # Axis labels
    fig.update_yaxes(title_text="Theta Power (% change)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    
    fig.update_yaxes(title_text="Theta Power (% change)", row=2, col=1)
    fig.update_xaxes(title_text="Time from Feedback (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time from Feedback (s)", row=2, col=2)
    
    fig.update_yaxes(title_text="Count", row=3, col=1)
    fig.update_xaxes(title_text="Theta Power (% change)", row=3, col=1)
    fig.update_xaxes(title_text="Theta Power (% change)", row=3, col=2)
    
    # Match y-axis ranges within each row for fair comparison
    # Row 1: timecourse
    y1_max = max(
        np.nanpercentile(high_tc['theta_power_pct'], 99.5),
        np.nanpercentile(low_tc['theta_power_pct'], 99.5)
    )
    y1_min = min(
        np.nanpercentile(high_tc['theta_power_pct'], 0.5),
        np.nanpercentile(low_tc['theta_power_pct'], 0.5)
    )
    fig.update_yaxes(range=[y1_min, y1_max], row=1, col=1)
    fig.update_yaxes(range=[y1_min, y1_max], row=1, col=2)
    
    # Row 3: distributions - match x range
    x3_max = max(
        np.nanpercentile(high_tc['theta_power_pct'], 99.9),
        np.nanpercentile(low_tc['theta_power_pct'], 99.9)
    )
    x3_min = min(
        np.nanpercentile(high_tc['theta_power_pct'], 0.1),
        np.nanpercentile(low_tc['theta_power_pct'], 0.1)
    )
    fig.update_xaxes(range=[x3_min, x3_max * 1.1], row=3, col=1)
    fig.update_xaxes(range=[x3_min, x3_max * 1.1], row=3, col=2)
    
    if save_path:
        fig.write_image(save_path, scale=2)
        print(f"Figure saved to: {save_path}")
    
    if show_fig:
        fig.show()
    
    return fig


def identify_theta_exemplars(
    theta_df: pd.DataFrame,
    metric: str = 'theta_p95',
    exclude_subjects: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict:
    """
    Identify high and low theta reactivity exemplar subjects.
    
    Parameters
    ----------
    theta_df : DataFrame
        Subject-level theta data with theta_p95 column
    metric : str
        Metric to use for ranking (default: theta_p95)
    exclude_subjects : list, optional
        Subject IDs to exclude from consideration
    verbose : bool
        Print results
    
    Returns
    -------
    dict with 'high' and 'low' subject IDs and their values
    """
    df = theta_df.copy()
    
    if exclude_subjects:
        df = df[~df['subject_id'].astype(str).isin([str(s) for s in exclude_subjects])]
    
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not in DataFrame")
    
    # Sort by metric
    df_sorted = df.sort_values(metric, ascending=False)
    
    high_row = df_sorted.iloc[0]
    low_row = df_sorted.iloc[-1]
    
    result = {
        'high': {
            'subject_id': str(high_row['subject_id']),
            metric: high_row[metric],
        },
        'low': {
            'subject_id': str(low_row['subject_id']),
            metric: low_row[metric],
        },
        'all_ranked': df_sorted[['subject_id', metric]].reset_index(drop=True),
    }
    
    if verbose:
        print("="*60)
        print("THETA REACTIVITY EXEMPLARS")
        print("="*60)
        print(f"\nMetric: {metric}")
        print(f"\nHigh theta: Sub-{result['high']['subject_id']} "
              f"({metric} = {result['high'][metric]:.1f})")
        print(f"Low theta:  Sub-{result['low']['subject_id']} "
              f"({metric} = {result['low'][metric]:.1f})")
        print(f"\nFull ranking:")
        print(result['all_ranked'].to_string(index=False))
    
    return result


# =============================================================================
# Quick Diagnostic Function
# =============================================================================

def preview_theta_timecourse(
    subject_id: str,
    run_num: int = 1,
    duration_sec: float = 30,
    channel_idx: int = 0,
    eeg_dir: Optional[Path] = None,
    show_fig: bool = True,
) -> Optional[go.Figure]:
    """
    Quick preview of a single subject's theta timecourse.
    
    Useful for checking data quality before including in comparison figure.
    """
    tc = extract_theta_timecourse(subject_id, run_num, channel_idx, eeg_dir)
    
    if tc is None:
        print(f"Failed to extract data for sub-{subject_id}")
        return None
    
    time_mask = tc['time_sec'] <= duration_sec
    t = tc['time_sec'][time_mask]
    theta = tc['theta_power_pct'][time_mask]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=t,
        y=theta,
        mode='lines',
        line=dict(width=1),
        name='Theta power'
    ))
    
    fig.add_hline(y=tc['theta_p95'], line_dash='dash', 
                  annotation_text=f"p95={tc['theta_p95']:.0f}%")
    fig.add_hline(y=0, line_dash='dot', line_color='gray')
    
    fig.update_layout(
        title=f"Sub-{subject_id} Run-{run_num} | p95={tc['theta_p95']:.1f}% | "
              f"Earclip: {tc['has_earclip']}",
        xaxis_title="Time (s)",
        yaxis_title="Theta Power (% change)",
        template=PLOTLY_TEMPLATE,
        height=400,
        width=800,
    )
    
    if show_fig:
        fig.show()
    
    return fig


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Theta Visualization Module")
    print("="*60)
    print("\nMain functions:")
    print("  - identify_theta_exemplars(theta_df)")
    print("  - preview_theta_timecourse(subject_id)")
    print("  - plot_theta_exemplar_comparison(high_id, low_id)")
    print("\nUsage:")
    print("  1. Run theta analysis to get theta_df with subject-level metrics")
    print("  2. Call identify_theta_exemplars(theta_df) to find high/low subjects")
    print("  3. Call preview_theta_timecourse() on each to verify data quality")
    print("  4. Call plot_theta_exemplar_comparison(high_id, low_id) for final figure")
