"""
stim_verification.py — Counterbalance validation from EEG ground truth

A wrong 'counterbalance' value in SUBJECT_INFO silently swaps the active/sham
condition labels for a subject. Nothing errors; the condition contrasts just
get corrupted. This module cross-checks the config value against what the
stimulator actually delivered, recovered from the 6 Hz tACS artifact in the
NIC recordings.

The detection logic is ported from the archived EEG notebook
(archive/tacs_bandit_eeg_v4.ipynb), which produced
derivatives/eeg/group_stim_detection_v4.csv for 9 subjects. The thresholds
and window definitions are unchanged; what is new is file discovery that
handles every NIC naming convention in the raw directory, so the check can
run on the full sample.

Classification is deliberately blind to the config counterbalance: each run is
labelled ACTIVE / SHAM / NONE from its own signal, and the counterbalance is
inferred from the pattern across runs. Only then is it compared to config and
REDCap. (`earclip` is read from config, but that only selects the detection
thresholds and is independent of condition assignment.)

Usage
-----
    python stim_verification.py                    # all registered subjects
    python stim_verification.py --sample new
    python stim_verification.py --subjects 10641 10810

Writes run-level and subject-level CSVs to derivatives/eeg/.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from config import (
    REPO_ROOT,
    SUBJECT_INFO,
    DISSERTATION_SUBJECTS,
    REDCAP_TACS_PATH,
    get_condition_map,
)
from nic_files import discover_runs

OUTPUT_DIR = REPO_ROOT / 'derivatives' / 'eeg'

# =============================================================================
# EEG constants (from archive/tacs_bandit_eeg_v4.ipynb)
# =============================================================================

FS = 500
CH_LABELS = ['F3', 'Fp1', 'FCz', 'FT7', 'F4', 'P4', 'P3', 'Ch8']
DETECTION_CH_IDX = 4  # F4 — a recording channel, not a stimulating one
TACS_FREQ = 6.0

# Absolute dB thresholds. Observed levels:
#   baseline/post ~65-85 dB; sham ~90-100 dB for the first 45 s then back down;
#   active ~95-100 dB sustained (earclip) or ~140-150 dB (no earclip).
#
# active_sustained_diff_db is relaxed from the v4 value of 5 to 10 for earclip
# subjects. On the full sample, middle_db is sharply bimodal — 293 runs between
# 60 and 85 dB, 98 runs at 98 dB and above, and almost nothing in between — so
# middle_db alone is what separates active from sham. The sustained check only
# guards against a run that spikes and decays. At a tolerance of 5 it was
# instead rejecting 25 of 98 genuinely-active runs (their |middle - late| tops
# out at 7.6 dB) and dumping them into NONE, which made whole subjects
# unresolvable. Relaxing it cannot promote a sham run to ACTIVE: sham fails on
# middle_high, whose 90 dB threshold sits inside the empty part of the
# distribution. Verified to leave all 71 runs of the v4 reference set unchanged.
STIM_THRESHOLDS = {
    'earclip': {
        'active_middle_min_db': 90,
        'active_sustained_diff_db': 10,
        'sham_early_min_db': 85,
        'sham_early_late_diff_db': 10,
    },
    'no_earclip': {
        'active_middle_min_db': 120,
        'active_sustained_diff_db': 10,
        'sham_early_min_db': 100,
        'sham_early_late_diff_db': 15,
    },
}


# =============================================================================
# File Discovery
# =============================================================================
# Lives in nic_files so the theta pipeline uses the same conventions; see the
# note there about how many ways these recordings got named.


def load_detection_channel(easy_path: Path, ch_idx: int = DETECTION_CH_IDX) -> np.ndarray:
    """
    Read a single channel from a NIC .easy file.

    .easy files are headerless tab-separated integers: 8 EEG channels, 3
    accelerometer, trigger, timestamp. Only one channel is needed for stim
    detection, and these files run to ~180k samples, so read just that column.
    """
    series = pd.read_csv(
        easy_path, sep='\t', header=None, usecols=[ch_idx], dtype=np.float64
    )
    return series.iloc[:, 0].to_numpy()


def read_info_stim_amplitude(info_path: Optional[Path]) -> Optional[float]:
    """
    Read the configured tACS amplitude (max Atacs, uA) from a NIC .info file.

    This is what the loaded protocol *specified*, which is weaker evidence than
    the recorded artifact — a sham protocol also specifies a nonzero amplitude,
    it just ramps straight back down. Reported alongside the detection as
    context, never used to classify.
    """
    if info_path is None or not info_path.exists():
        return None

    amplitudes = []
    for line in info_path.read_text(errors='ignore').splitlines():
        if 'Atacs (uA)' in line:
            try:
                amplitudes.append(float(line.split(':')[1].strip()))
            except (IndexError, ValueError):
                continue

    return max(amplitudes) if amplitudes else None


# =============================================================================
# Stimulation Detection
# =============================================================================

def compute_window_powers(eeg_data: np.ndarray, fs: int = FS) -> Dict:
    """
    6 Hz power (dB) in the early, middle, and late windows of a run.

    Early covers the first 45 s, which is where a sham protocol's ramp
    up-and-down lives. Middle and late capture whether power is *sustained*,
    which is what separates active from sham.
    """
    f, t_spec, sxx = signal.spectrogram(
        signal.detrend(eeg_data), fs=fs, nperseg=1000, noverlap=750, scaling='density'
    )

    six_hz_mask = (f >= 5.5) & (f <= 6.5)
    power_db = 10 * np.log10(np.mean(sxx[six_hz_mask, :], axis=0) + 1e-30)

    if len(t_spec) == 0:
        return {'error': 'empty spectrogram'}

    duration = t_spec[-1]
    early_mask = t_spec <= 45
    middle_mask = (t_spec >= 60) & (t_spec <= min(300, duration - 45))
    late_mask = t_spec >= (duration - 45)

    return {
        'early_db': np.mean(power_db[early_mask]) if np.any(early_mask) else np.nan,
        'middle_db': np.mean(power_db[middle_mask]) if np.any(middle_mask) else np.nan,
        'late_db': np.mean(power_db[late_mask]) if np.any(late_mask) else np.nan,
        'duration_s': duration,
    }


def classify_stimulation(
    eeg_data: np.ndarray, has_earclip: bool, fs: int = FS
) -> Tuple[str, Dict]:
    """
    Classify one run as ACTIVE, SHAM, or NONE from its 6 Hz profile.

    ACTIVE : middle power above threshold and sustained into the late window.
    SHAM   : early power elevated but dropped off by the late window (ramp only).
    NONE   : no meaningful 6 Hz elevation — a baseline or post run.
    """
    thresholds = STIM_THRESHOLDS['earclip' if has_earclip else 'no_earclip']
    powers = compute_window_powers(eeg_data, fs)

    if 'error' in powers:
        return 'NONE', powers

    early, middle, late = powers['early_db'], powers['middle_db'], powers['late_db']

    middle_high = middle >= thresholds['active_middle_min_db']
    sustained = abs(middle - late) <= thresholds['active_sustained_diff_db']

    if middle_high and sustained:
        classification = 'ACTIVE'
    elif (
        early >= thresholds['sham_early_min_db']
        and (early - late) >= thresholds['sham_early_late_diff_db']
    ):
        classification = 'SHAM'
    else:
        classification = 'NONE'

    powers['middle_high'] = middle_high
    powers['sustained'] = sustained
    return classification, powers


# =============================================================================
# Per-Subject Verification
# =============================================================================

STIM_RUNS_EARLY = (2, 3)
STIM_RUNS_LATE = (6, 7)


def verify_subject(
    subject_id: str, eeg_dir: Optional[Path] = None, verbose: bool = True
) -> pd.DataFrame:
    """
    Classify every discoverable run for one subject.

    Returns a run-level DataFrame. Empty if no recordings were found.
    """
    info = SUBJECT_INFO.get(subject_id, {})
    has_earclip = info.get('earclip', True)
    config_cb = info.get('counterbalance', '?')
    expected_map = get_condition_map(config_cb)

    runs = discover_runs(subject_id, eeg_dir)
    if not runs:
        if verbose:
            print(f'  sub-{subject_id}: no EEG recordings found')
        return pd.DataFrame()

    rows = []
    for run_num in sorted(runs):
        files = runs[run_num]
        try:
            eeg = load_detection_channel(files['easy'])
            detected, powers = classify_stimulation(eeg, has_earclip)
            error = ''
        except Exception as exc:  # a corrupt or truncated recording
            detected, powers, error = 'ERROR', {}, str(exc)

        rows.append({
            'subject_id': subject_id,
            'run': run_num,
            'expected': expected_map.get(run_num, 'unknown'),
            'detected': detected,
            'early_db': powers.get('early_db', np.nan),
            'middle_db': powers.get('middle_db', np.nan),
            'late_db': powers.get('late_db', np.nan),
            'duration_s': powers.get('duration_s', np.nan),
            'configured_uA': read_info_stim_amplitude(files['info']),
            'has_earclip': has_earclip,
            'file': files['easy'].name,
            'error': error,
        })

        if verbose:
            print(f'  run {run_num}: expected {expected_map.get(run_num, "?"):<9} '
                  f'detected {detected}')

    return pd.DataFrame(rows)


def infer_counterbalance(run_df: pd.DataFrame) -> Tuple[str, str]:
    """
    Infer counterbalance from the detected pattern alone.

    A = active on runs 2-3, sham on 6-7. B = the reverse.

    Returns (inferred, basis) where inferred is 'A', 'B', or '' when the
    evidence does not settle it. `basis` records what was actually seen, so an
    unresolved subject can be triaged rather than silently dropped.
    """
    early = set(run_df[run_df['run'].isin(STIM_RUNS_EARLY)]['detected'])
    late = set(run_df[run_df['run'].isin(STIM_RUNS_LATE)]['detected'])
    basis = f'early={sorted(early) or "-"} late={sorted(late) or "-"}'

    if 'ACTIVE' in early and 'ACTIVE' not in late:
        return 'A', basis
    if 'ACTIVE' in late and 'ACTIVE' not in early:
        return 'B', basis
    return '', basis


# =============================================================================
# REDCap Cross-Reference
# =============================================================================

def load_redcap_counterbalance(path: Optional[Path] = None) -> Dict[str, str]:
    """
    Read the session-log counterbalance recorded in the tACS REDCap export.

    An independent record of what was *intended*, useful for triaging subjects
    whose EEG cannot settle the question.
    """
    if path is None:
        path = REDCAP_TACS_PATH

    if not Path(path).exists():
        return {}

    redcap = pd.read_csv(path, low_memory=False)
    if 'Counterbalance' not in redcap.columns:
        return {}

    redcap['sid'] = redcap[redcap.columns[0]].astype(str).str.extract(r'(\d{5})')[0]
    pairs = redcap[['sid', 'Counterbalance']].dropna().drop_duplicates()

    # A subject with conflicting entries is not usable as a reference.
    counts = pairs.groupby('sid')['Counterbalance'].nunique()
    usable = counts[counts == 1].index

    return (
        pairs[pairs['sid'].isin(usable)]
        .groupby('sid')['Counterbalance']
        .first()
        .astype(str)
        .str.strip()
        .to_dict()
    )


# =============================================================================
# Full Sample Verification
# =============================================================================

def verify_all(
    subjects: Optional[List[str]] = None,
    sample: str = 'all',
    eeg_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Verify counterbalance for a set of subjects.

    Returns
    -------
    (run_df, subject_df)
        run_df     : one row per run, with the detection and its dB windows
        subject_df : one row per subject, comparing config / REDCap / EEG
    """
    if subjects is None:
        if sample == 'dissertation':
            subjects = list(DISSERTATION_SUBJECTS)
        elif sample == 'new':
            subjects = [s for s in SUBJECT_INFO if s not in DISSERTATION_SUBJECTS]
        elif sample == 'all':
            subjects = list(SUBJECT_INFO)
        else:
            raise ValueError(
                f"sample must be 'dissertation', 'all', or 'new'; got '{sample}'"
            )

    redcap_cb = load_redcap_counterbalance()
    if verbose:
        print(f'REDCap counterbalance available for {len(redcap_cb)} subjects\n')

    run_frames, subject_rows = [], []

    for i, sid in enumerate(sorted(subjects), 1):
        if verbose:
            print(f'[{i}/{len(subjects)}] sub-{sid}')

        run_df = verify_subject(sid, eeg_dir, verbose=verbose)
        config_cb = SUBJECT_INFO.get(sid, {}).get('counterbalance', '?')
        rc_cb = redcap_cb.get(sid, '')

        if len(run_df) == 0:
            subject_rows.append({
                'subject_id': sid,
                'config_cb': config_cb,
                'redcap_cb': rc_cb,
                'eeg_cb': '',
                'verdict': 'NO_EEG',
                'basis': '',
                'n_runs': 0,
                'in_dissertation': sid in DISSERTATION_SUBJECTS,
            })
            continue

        run_frames.append(run_df)
        eeg_cb, basis = infer_counterbalance(run_df)

        if not eeg_cb:
            verdict = 'UNRESOLVED'
        elif config_cb == eeg_cb:
            verdict = 'CONFIRMED'
        else:
            verdict = 'MISMATCH'

        subject_rows.append({
            'subject_id': sid,
            'config_cb': config_cb,
            'redcap_cb': rc_cb,
            'eeg_cb': eeg_cb,
            'verdict': verdict,
            'basis': basis,
            'n_runs': len(run_df),
            'in_dissertation': sid in DISSERTATION_SUBJECTS,
        })

    run_all = pd.concat(run_frames, ignore_index=True) if run_frames else pd.DataFrame()
    subject_all = pd.DataFrame(subject_rows)

    return run_all, subject_all


def print_summary(subject_df: pd.DataFrame) -> None:
    """Print the triage view: what is confirmed, what needs a human."""
    print('\n' + '=' * 70)
    print('Counterbalance Verification Summary')
    print('=' * 70)

    counts = subject_df['verdict'].value_counts()
    for verdict in ['CONFIRMED', 'MISMATCH', 'UNRESOLVED', 'NO_EEG']:
        if verdict in counts:
            print(f'  {verdict:<12} {counts[verdict]:>3}')

    mismatches = subject_df[subject_df['verdict'] == 'MISMATCH']
    if len(mismatches):
        print(f'\n⚠ {len(mismatches)} subject(s) where config disagrees with the EEG.')
        print('  config_cb is WRONG for these — the EEG is what was delivered.\n')
        print(mismatches[['subject_id', 'config_cb', 'redcap_cb', 'eeg_cb',
                          'in_dissertation', 'basis']].to_string(index=False))

    unresolved = subject_df[subject_df['verdict'] == 'UNRESOLVED']
    if len(unresolved):
        print(f'\n{len(unresolved)} subject(s) the EEG could not settle '
              f'(incomplete runs, or stimulation not delivered as planned):\n')
        print(unresolved[['subject_id', 'config_cb', 'redcap_cb', 'n_runs', 'basis']]
              .to_string(index=False))

    no_eeg = subject_df[subject_df['verdict'] == 'NO_EEG']
    if len(no_eeg):
        print(f'\n{len(no_eeg)} subject(s) with no EEG recordings — '
              f'config unverifiable from this source:')
        print('  ' + ', '.join(no_eeg['subject_id'].tolist()))

    # Where config and REDCap disagree, the EEG is the tiebreaker.
    both = subject_df[(subject_df['redcap_cb'] != '') & (subject_df['config_cb'] != '?')]
    disagree = both[both['config_cb'] != both['redcap_cb']]
    if len(disagree):
        print(f'\nconfig vs REDCap disagreements: {len(disagree)} '
              f'({(disagree["eeg_cb"] != "").sum()} settled by EEG)')
        print(disagree[['subject_id', 'config_cb', 'redcap_cb', 'eeg_cb', 'verdict',
                        'in_dissertation']].to_string(index=False))


# =============================================================================
# CLI
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--sample', default='all',
                        choices=['all', 'dissertation', 'new'])
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='explicit subject IDs (overrides --sample)')
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR))
    parser.add_argument('--no-write', action='store_true',
                        help='print the summary without writing CSVs')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    run_df, subject_df = verify_all(
        subjects=args.subjects, sample=args.sample, verbose=not args.quiet
    )

    print_summary(subject_df)

    if not args.no_write:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_path = out_dir / 'stim_verification_runs.csv'
        subj_path = out_dir / 'stim_verification_subjects.csv'
        run_df.to_csv(run_path, index=False)
        subject_df.to_csv(subj_path, index=False)
        print(f'\nWrote {run_path}')
        print(f'Wrote {subj_path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
