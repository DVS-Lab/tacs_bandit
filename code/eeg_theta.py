"""
eeg_theta.py — EEG theta analysis for tACS Bandit study

Analyzes task-related theta (4-8 Hz) oscillations from baseline runs.
Extracts theta reactivity as an individual difference measure.

Key methodological notes:
- Behavioral task and EEG run on separate computers without hardware sync
- Heuristic timestamp alignment used (sensitivity analysis shows individual 
  differences robust to ±1s shifts)
- Only 3 EEG channels available: F4, P4, P3 (stimulation channels blanked)
- Subjects without earclip use software average re-reference

Primary metric: theta_p95 = 95th percentile of theta power (% change)
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import glob
import os
import warnings

from config import (
    SUBJECT_INFO,
    DATA_DIR,
    EEG_DIR,
    PLOTLY_TEMPLATE,
    FONT_FAMILY,
    COLOR_SHAM as COLOR_BLUE,
    COLOR_GOLD,
    COLOR_GREEN,
    COLOR_RED,
)
from nic_files import find_run as nic_find_run

warnings.filterwarnings('ignore', message='loadtxt: input contained no data')


# =============================================================================
# EEG Constants
# =============================================================================

FS = 500                        # Sampling rate (Hz)
EEG_CH_IDX = [4, 5, 6]          # EEG-only channels in .easy file (F4, P4, P3)
EEG_CH_LABELS = ['F4', 'P4', 'P3']

EPOCH_PRE = 1.0
EPOCH_POST = 1.5
BASELINE_WIN = (-0.5, -0.1)

THETA_BAND = (4, 8)

HIGHPASS_FREQ = 0.5
ARTIFACT_THRESH_UV = 150

ARTIFACT_PCT_THRESH = 10
THETA_P99_THRESH = 1500
SKIP_FIRST_SEC = 10


# =============================================================================
# File Discovery
# =============================================================================

def find_eeg_run(eeg_dir: Path, subject_id: str, run_num: int) -> Optional[str]:
    """
    Find EEG .easy file for a given subject and run.

    Delegates to nic_files, which knows every filename convention in the raw
    directory. This previously globbed only for `*sub-{id}_Run*`, so subjects
    recorded as `Bandit-{id}_Run N` or `{id} TACS_Run N` looked like they had
    no EEG at all.
    """
    path = nic_find_run(str(subject_id), run_num, Path(eeg_dir))
    return str(path) if path is not None else None


def get_latest_behavioral_file(subject_dir: Path, run_num: int) -> Optional[str]:
    """Find latest behavioral CSV for a run."""
    pattern = str(subject_dir / f'*run-{run_num:02d}*.csv')
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    return max(files, key=os.path.getmtime)


# =============================================================================
# Data Loading
# =============================================================================

def load_eeg_run(easy_path: str) -> Dict:
    """Load EEG data from NIC2 .easy file."""
    raw = np.loadtxt(easy_path)
    
    eeg = raw[:, EEG_CH_IDX] / 1000.0
    timestamps_unix_ms = raw[:, 12]
    time_sec = (timestamps_unix_ms - timestamps_unix_ms[0]) / 1000.0
    
    return {
        'eeg': eeg,
        'timestamps_unix_ms': timestamps_unix_ms,
        'time_sec': time_sec,
        'fs': FS,
        'duration_sec': time_sec[-1]
    }


def load_behavioral_run(csv_path: str) -> pd.DataFrame:
    """Load behavioral data and compute feedback times."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['rt']).copy()
    
    df['rt_sec'] = df['rt'] / 1000.0
    df['iti_sec'] = df['iti'] / 1000.0
    
    FEEDBACK_DURATION = 1.0
    df['feedback_time'] = df['timestamp'] - FEEDBACK_DURATION - df['iti_sec']
    
    first_feedback = df['feedback_time'].iloc[0]
    df['feedback_time_relative'] = df['feedback_time'] - first_feedback
    df.attrs['first_feedback_in_run'] = first_feedback
    
    return df


# =============================================================================
# Preprocessing
# =============================================================================

def highpass_filter(data: np.ndarray, cutoff: float = HIGHPASS_FREQ, 
                    fs: int = FS, order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth high-pass filter."""
    nyq = fs / 2
    b, a = signal.butter(order, cutoff / nyq, btype='high')
    
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return np.column_stack([signal.filtfilt(b, a, data[:, i]) 
                                for i in range(data.shape[1])])


def bandpass_filter(data: np.ndarray, band: Tuple[float, float] = THETA_BAND,
                    fs: int = FS, order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter."""
    nyq = fs / 2
    low, high = band
    b, a = signal.butter(order, [low / nyq, high / nyq], btype='band')
    
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return np.column_stack([signal.filtfilt(b, a, data[:, i]) 
                                for i in range(data.shape[1])])


def avg_rereference(eeg: np.ndarray) -> np.ndarray:
    """Average re-reference (for subjects without earclip)."""
    avg = np.mean(eeg, axis=1, keepdims=True)
    return eeg - avg


def detect_artifacts(eeg: np.ndarray, 
                     threshold_uv: float = ARTIFACT_THRESH_UV) -> np.ndarray:
    """Detect samples exceeding artifact threshold."""
    return np.any(np.abs(eeg) > threshold_uv, axis=1)


def preprocess_eeg(eeg_data: Dict, has_earclip: bool = False) -> Dict:
    """Full preprocessing pipeline."""
    eeg = eeg_data['eeg'].copy()
    
    eeg = highpass_filter(eeg, cutoff=HIGHPASS_FREQ)
    
    if not has_earclip:
        eeg = avg_rereference(eeg)
    
    artifact_mask = detect_artifacts(eeg)
    
    result = eeg_data.copy()
    result['eeg'] = eeg
    result['artifact_mask'] = artifact_mask
    result['artifact_pct'] = 100 * np.mean(artifact_mask)
    
    return result


# =============================================================================
# Timestamp Alignment
# =============================================================================

def align_timestamps(eeg_data: Dict, behav_data: pd.DataFrame) -> pd.DataFrame:
    """Align behavioral feedback times to EEG recording using heuristic."""
    behav_data = behav_data.copy()
    
    eeg_duration = eeg_data['duration_sec']
    behav_span = behav_data['feedback_time_relative'].max()
    
    margin_start = EPOCH_PRE + 0.5
    margin_end = EPOCH_POST + 0.5
    
    if behav_span > eeg_duration:
        raise ValueError(f'Behavioral ({behav_span:.1f}s) exceeds EEG ({eeg_duration:.1f}s)')
    
    estimated_first_feedback = 20.0
    
    last_feedback_time = estimated_first_feedback + behav_span
    if last_feedback_time > eeg_duration - margin_end:
        estimated_first_feedback = eeg_duration - margin_end - behav_span
    
    if estimated_first_feedback < margin_start:
        estimated_first_feedback = margin_start
    
    behav_data['feedback_time_eeg'] = behav_data['feedback_time_relative'] + estimated_first_feedback
    behav_data.attrs['offset_sec'] = estimated_first_feedback
    behav_data.attrs['behav_span'] = behav_span
    
    return behav_data


# =============================================================================
# Epoching
# =============================================================================

def extract_epochs(eeg_data: Dict, event_times: np.ndarray,
                   pre_sec: float = EPOCH_PRE, post_sec: float = EPOCH_POST,
                   reject_artifacts: bool = True) -> Dict:
    """Extract epochs time-locked to events."""
    eeg = eeg_data['eeg']
    fs = eeg_data['fs']
    artifact_mask = eeg_data.get('artifact_mask', np.zeros(len(eeg), dtype=bool))
    
    n_pre = int(pre_sec * fs)
    n_post = int(post_sec * fs)
    n_samples = n_pre + n_post
    n_channels = eeg.shape[1]
    
    times = np.linspace(-pre_sec, post_sec, n_samples, endpoint=False)
    
    epochs = []
    valid_idx = []
    reject_idx = []
    
    for i, t in enumerate(event_times):
        center_sample = int(t * fs)
        start_sample = center_sample - n_pre
        end_sample = center_sample + n_post
        
        if start_sample < 0 or end_sample > len(eeg):
            reject_idx.append(i)
            continue
        
        segment = eeg[start_sample:end_sample, :]
        
        if reject_artifacts and np.any(artifact_mask[start_sample:end_sample]):
            reject_idx.append(i)
            continue
        
        epochs.append(segment)
        valid_idx.append(i)
    
    if len(epochs) == 0:
        return {
            'epochs': np.zeros((0, n_samples, n_channels)),
            'times': times,
            'valid_idx': np.array([], dtype=int),
            'reject_idx': np.array(reject_idx, dtype=int),
            'n_valid': 0,
            'n_rejected': len(reject_idx)
        }
    
    return {
        'epochs': np.stack(epochs, axis=0),
        'times': times,
        'valid_idx': np.array(valid_idx, dtype=int),
        'reject_idx': np.array(reject_idx, dtype=int),
        'n_valid': len(epochs),
        'n_rejected': len(reject_idx)
    }


# =============================================================================
# Theta Power Analysis
# =============================================================================

def compute_band_power_timecourse(epochs: np.ndarray, times: np.ndarray,
                                   band: Tuple[float, float] = THETA_BAND,
                                   fs: int = FS) -> np.ndarray:
    """Compute instantaneous band power using Hilbert transform."""
    n_epochs, n_samples, n_channels = epochs.shape
    power = np.zeros_like(epochs)
    
    for ch in range(n_channels):
        for ep in range(n_epochs):
            filtered = bandpass_filter(epochs[ep, :, ch], band, fs)
            analytic = signal.hilbert(filtered)
            envelope = np.abs(analytic)
            power[ep, :, ch] = envelope ** 2
    
    return power


def compute_ersp(epochs: np.ndarray, times: np.ndarray,
                 band: Tuple[float, float] = THETA_BAND,
                 baseline_win: Tuple[float, float] = BASELINE_WIN,
                 fs: int = FS) -> Dict:
    """Compute Event-Related Spectral Perturbation (ERSP)."""
    power = compute_band_power_timecourse(epochs, times, band, fs)
    
    baseline_mask = (times >= baseline_win[0]) & (times < baseline_win[1])
    baseline_power = power[:, baseline_mask, :].mean(axis=1, keepdims=True)
    baseline_power = np.maximum(baseline_power, 1e-10)
    
    ersp_epochs = (power - baseline_power) / baseline_power * 100
    ersp = ersp_epochs.mean(axis=0)
    
    return {
        'ersp': ersp,
        'power': power,
        'times': times
    }


# =============================================================================
# Theta Reactivity Metric
# =============================================================================

def compute_theta_reactivity_run(
    subject_id: str,
    run_num: int,
    channel_idx: int = 0,
    eeg_dir: Optional[Path] = None,
    behav_dir: Optional[Path] = None,
    artifact_thresh_uv: float = ARTIFACT_THRESH_UV,
    skip_first_sec: float = SKIP_FIRST_SEC
) -> Optional[Dict]:
    """Compute theta reactivity for a single run."""
    if eeg_dir is None:
        eeg_dir = EEG_DIR
    if behav_dir is None:
        behav_dir = DATA_DIR  # DATA_DIR already points to .../data/bandit
    
    eeg_path = find_eeg_run(eeg_dir, subject_id, run_num)
    if eeg_path is None:
        return None
    
    try:
        eeg_data = load_eeg_run(eeg_path)
    except Exception:
        return None
    
    eeg_times = eeg_data['time_sec']
    
    # Get age
    sub_behav_dir = behav_dir / f'sub-{subject_id}'
    behav_path = get_latest_behavioral_file(sub_behav_dir, run_num)
    age = None
    if behav_path:
        try:
            behav_df = pd.read_csv(behav_path)
            if 'age' in behav_df.columns:
                age = behav_df['age'].iloc[0]
        except Exception:
            pass
    
    # Preprocess
    sub_info = SUBJECT_INFO.get(str(subject_id), {})
    has_earclip = sub_info.get('earclip', False)
    eeg_data = preprocess_eeg(eeg_data, has_earclip=has_earclip)
    
    # Theta power
    raw_signal = eeg_data['eeg'][:, channel_idx].copy()
    theta_filtered = bandpass_filter(raw_signal, THETA_BAND, FS)
    analytic_signal = signal.hilbert(theta_filtered)
    theta_power = np.abs(analytic_signal) ** 2
    
    # Masking
    artifact_mask = np.abs(raw_signal) > artifact_thresh_uv
    startup_mask = eeg_times < skip_first_sec
    combined_mask = artifact_mask | startup_mask
    
    # Smooth and normalize
    window_samples = int(0.1 * FS)
    theta_power_smooth = np.convolve(theta_power, 
                                      np.ones(window_samples) / window_samples, 
                                      mode='same')
    theta_power_smooth[combined_mask] = np.nan
    
    clean_mean = np.nanmean(theta_power_smooth)
    if clean_mean <= 0:
        return None
    
    theta_pct = (theta_power_smooth / clean_mean - 1) * 100
    clean_vals = theta_pct[~np.isnan(theta_pct)]
    
    if len(clean_vals) == 0:
        return None
    
    return {
        'subject_id': str(subject_id),
        'run': run_num,
        'age': age,
        'channel': EEG_CH_LABELS[channel_idx],
        'theta_median': np.median(clean_vals),
        'theta_p75': np.percentile(clean_vals, 75),
        'theta_p95': np.percentile(clean_vals, 95),
        'theta_p99': np.percentile(clean_vals, 99),
        'theta_max': np.max(clean_vals),
        'artifact_pct': 100 * combined_mask.sum() / len(combined_mask),
        'earclip': has_earclip,
        'duration_sec': eeg_data['duration_sec'],
    }


def compute_theta_reactivity_all(
    subject_info: Optional[Dict] = None,
    baseline_runs: List[int] = [1, 5],
    channel_idx: int = 0,
    verbose: bool = True
) -> pd.DataFrame:
    """Compute theta reactivity for all subjects and baseline runs."""
    if subject_info is None:
        subject_info = SUBJECT_INFO
    
    results = []
    
    for subject_id in subject_info.keys():
        for run_num in baseline_runs:
            result = compute_theta_reactivity_run(str(subject_id), run_num, channel_idx)
            if result is not None:
                results.append(result)
                if verbose:
                    print(f"  Sub-{subject_id} Run {run_num}: θ_p95 = {result['theta_p95']:.1f}%")
    
    return pd.DataFrame(results)


def apply_theta_qc(
    theta_df: pd.DataFrame,
    artifact_pct_thresh: float = ARTIFACT_PCT_THRESH,
    theta_p99_thresh: float = THETA_P99_THRESH
) -> pd.DataFrame:
    """Apply QC exclusion criteria to theta data."""
    clean = theta_df[
        (theta_df['artifact_pct'] < artifact_pct_thresh) &
        (theta_df['theta_p95'] > 0) &
        (theta_df['theta_p99'] < theta_p99_thresh)
    ].copy()
    
    return clean


def compute_subject_theta_average(clean_theta_df: pd.DataFrame) -> pd.DataFrame:
    """Compute subject-level average theta reactivity from clean runs."""
    subj_avg = clean_theta_df.groupby('subject_id').agg({
        'theta_p95': 'mean',
        'theta_p75': 'mean',
        'theta_median': 'mean',
        'age': 'first',
        'earclip': 'first',
        'run': 'count'
    }).rename(columns={'run': 'n_clean_runs'})
    
    return subj_avg.reset_index()


# =============================================================================
# Test-Retest Reliability
# =============================================================================

def compute_theta_reliability(
    theta_df: pd.DataFrame,
    metric: str = 'theta_p95'
) -> Dict:
    """Compute test-retest reliability between Run 1 and Run 5."""
    run1 = theta_df[theta_df['run'] == 1].set_index('subject_id')
    run5 = theta_df[theta_df['run'] == 5].set_index('subject_id')
    
    common_subs = run1.index.intersection(run5.index)
    
    if len(common_subs) < 3:
        return {'r': np.nan, 'p': np.nan, 'ICC': np.nan, 'n': len(common_subs)}
    
    r1_vals = run1.loc[common_subs, metric].values
    r5_vals = run5.loc[common_subs, metric].values
    
    r, p = pearsonr(r1_vals, r5_vals)
    
    # ICC
    subject_means = (r1_vals + r5_vals) / 2
    ms_subjects = np.var(subject_means, ddof=1) * 2
    diffs = r1_vals - r5_vals
    ms_error = np.var(diffs, ddof=1) / 2
    icc = (ms_subjects - ms_error) / (ms_subjects + ms_error) if (ms_subjects + ms_error) > 0 else np.nan
    
    return {
        'r': r,
        'p': p,
        'ICC': icc,
        'n': len(common_subs),
        'run1_values': r1_vals,
        'run5_values': r5_vals,
        'subjects': list(common_subs)
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_theta_reliability(reliability_result: Dict, show_fig: bool = True):
    """Plot Run 1 vs Run 5 scatter for reliability."""
    import plotly.graph_objects as go
    
    if reliability_result['n'] < 3:
        print("Insufficient data for reliability plot")
        return None
    
    r1 = reliability_result['run1_values']
    r5 = reliability_result['run5_values']
    subs = reliability_result['subjects']
    r = reliability_result['r']
    p = reliability_result['p']
    icc = reliability_result['ICC']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=r1, y=r5,
        mode='markers+text',
        marker=dict(size=12, color=COLOR_BLUE, opacity=0.7,
                    line=dict(width=1, color='black')),
        text=subs,
        textposition='top right',
        textfont=dict(size=9),
        name='Subjects'
    ))
    
    # Identity line
    all_vals = np.concatenate([r1, r5])
    line_range = [all_vals.min() * 0.9, all_vals.max() * 1.1]
    fig.add_trace(go.Scatter(
        x=line_range, y=line_range,
        mode='lines',
        line=dict(dash='dash', color='gray', width=1),
        name='Identity',
        showlegend=False
    ))
    
    # Regression line
    z = np.polyfit(r1, r5, 1)
    fig.add_trace(go.Scatter(
        x=np.sort(r1),
        y=np.poly1d(z)(np.sort(r1)),
        mode='lines',
        line=dict(color=COLOR_RED, width=2),
        name='Regression'
    ))
    
    fig.update_layout(
        title=f'Theta Reactivity Test-Retest<br><sup>r = {r:.2f}, p = {p:.3f}, ICC = {icc:.2f}, n = {len(subs)}</sup>',
        xaxis_title='Run 1 θ_p95 (% change)',
        yaxis_title='Run 5 θ_p95 (% change)',
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        height=500,
        width=550
    )
    
    fig.update_xaxes(showgrid=False, linecolor='black', linewidth=1)
    fig.update_yaxes(showgrid=False, linecolor='black', linewidth=1)
    
    if show_fig:
        fig.show()
    
    return fig


def plot_theta_vs_age(theta_subj_avg: pd.DataFrame, show_fig: bool = True):
    """Plot theta reactivity vs age scatter."""
    import plotly.graph_objects as go
    
    df = theta_subj_avg.dropna(subset=['age', 'theta_p95'])
    
    if len(df) < 3:
        print("Insufficient data for age plot")
        return None
    
    r, p = pearsonr(df['age'], df['theta_p95'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['age'],
        y=df['theta_p95'],
        mode='markers+text',
        marker=dict(size=12, color=COLOR_BLUE, opacity=0.7,
                    line=dict(width=1, color='black')),
        text=df['subject_id'],
        textposition='top right',
        textfont=dict(size=9),
        name='Subjects'
    ))
    
    # Regression line
    z = np.polyfit(df['age'], df['theta_p95'], 1)
    x_line = np.linspace(df['age'].min() - 5, df['age'].max() + 5, 100)
    fig.add_trace(go.Scatter(
        x=x_line,
        y=np.poly1d(z)(x_line),
        mode='lines',
        line=dict(color=COLOR_RED, width=2, dash='dash'),
        name=f'Regression (r={r:.2f})'
    ))
    
    fig.update_layout(
        title=f'Theta Reactivity vs Age<br><sup>N={len(df)}, r={r:.2f}, p={p:.3f}</sup>',
        xaxis_title='Age (years)',
        yaxis_title='Theta Reactivity (p95, % change)',
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY),
        legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
        height=500,
        width=700
    )
    
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor='black', linewidth=1)
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor='black', linewidth=1)
    
    if show_fig:
        fig.show()
    
    return fig


# =============================================================================
# Main Pipeline
# =============================================================================

def run_theta_analysis(
    verbose: bool = True,
    show_plots: bool = True
) -> Dict:
    """
    Run complete theta reactivity analysis pipeline.
    
    Returns
    -------
    dict with keys:
        'theta_all': all run-level data
        'theta_clean': QC-filtered data
        'theta_subject': subject-level averages
        'reliability': test-retest reliability
        'fig_reliability': reliability plot
        'fig_age': age association plot
    """
    results = {}
    
    if verbose:
        print('='*70)
        print('Theta Reactivity Analysis (Baseline Runs)')
        print('='*70)
        print()
    
    # Compute theta for all runs
    if verbose:
        print('Computing theta reactivity...')
    theta_all = compute_theta_reactivity_all(verbose=verbose)
    results['theta_all'] = theta_all
    
    if len(theta_all) == 0:
        if verbose:
            print('No theta data found.')
        return results
    
    if verbose:
        print(f'\nTotal runs: {len(theta_all)}')
    
    # Apply QC
    theta_clean = apply_theta_qc(theta_all)
    results['theta_clean'] = theta_clean
    
    if verbose:
        print(f'Clean runs (QC passed): {len(theta_clean)}')
    
    # Subject averages
    theta_subject = compute_subject_theta_average(theta_clean)
    results['theta_subject'] = theta_subject
    
    if verbose:
        print(f'Subjects with clean data: {len(theta_subject)}')
        print(f'\nTheta p95 range: {theta_subject["theta_p95"].min():.1f} - {theta_subject["theta_p95"].max():.1f}%')
    
    # Reliability
    if verbose:
        print('\n' + '-'*50)
        print('Test-Retest Reliability (Run 1 vs Run 5)')
        print('-'*50)
    
    reliability = compute_theta_reliability(theta_clean)
    results['reliability'] = reliability
    
    if verbose:
        print(f'  N = {reliability["n"]}')
        print(f'  Pearson r = {reliability["r"]:.3f}, p = {reliability["p"]:.3f}')
        print(f'  ICC = {reliability["ICC"]:.3f}')
        
        if reliability['ICC'] >= 0.75:
            print('  → Excellent reliability')
        elif reliability['ICC'] >= 0.60:
            print('  → Good reliability')
        elif reliability['ICC'] >= 0.40:
            print('  → Fair reliability')
        else:
            print('  → Poor reliability')
    
    # Plots
    if show_plots:
        if reliability['n'] >= 3:
            results['fig_reliability'] = plot_theta_reliability(reliability)
        
        if len(theta_subject) >= 3:
            results['fig_age'] = plot_theta_vs_age(theta_subject)
    
    return results


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing eeg_theta module...")
    print(f"EEG channels: {EEG_CH_LABELS}")
    print(f"Theta band: {THETA_BAND} Hz")
    print("Functions: compute_theta_reactivity_all, apply_theta_qc,")
    print("           compute_theta_reliability, run_theta_analysis")
