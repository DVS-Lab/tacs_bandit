"""
individual_theta.py — Individualized theta frequency (iTF) per subject

Every participant was stimulated at a fixed 6.0 Hz — all 237 stimulation runs,
no exceptions. So iTF is not a delivered parameter here; it is a *moderator*.
The question the paper can ask is whether stimulation worked better for people
whose endogenous theta sits closer to the 6 Hz that was actually delivered.

Two estimators, deliberately:

**Klimesch anchor (primary).** Individual alpha frequency from the posterior
channels, minus a fixed offset. Alpha is the most reliably detectable rhythm in
scalp EEG, and P3/P4 are available for *every* subject, so this estimator
covers the whole sample. Klimesch's transition frequency is properly defined
from how theta and alpha move in opposite directions between rest and task;
with no resting block in this design, the fixed offset is the available
approximation and is labelled as such.

**Direct theta peak (validation).** A spectral peak inside the theta band at
frontal-midline FCz. This is the theoretically preferred measure, but FCz is a
stimulating electrode for the dissertation-era protocol and only records during
non-stimulation runs for the later subjects. It therefore covers a subset — but
on that subset it provides an empirical check on the offset assumed above,
rather than taking 5 Hz on faith.

Estimation uses run 1 only by default. Run 1 is the only run that precedes any
stimulation for every subject: run 5 follows the first stimulation block, which
is active for counterbalance A and sham for counterbalance B, so including it
would introduce an asymmetry between the counterbalance groups.

Usage
-----
    python individual_theta.py
    python individual_theta.py --runs 1 5 --output-dir ../derivatives/eeg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from config import REPO_ROOT, SUBJECT_INFO, DISSERTATION_SUBJECTS
from nic_files import discover_runs

OUTPUT_DIR = REPO_ROOT / 'derivatives' / 'eeg'

FS = 500
# Channel order in the .easy file. The first four stimulate during stimulation
# runs and record EEG otherwise.
CH_LABELS = ['F3', 'Fp1', 'FCz', 'FT7', 'F4', 'P4', 'P3', 'EXT']
POSTERIOR_CH = ['P3', 'P4']      # alpha is largest posteriorly
MIDLINE_CH = ['FCz']             # frontal-midline theta

ALPHA_BAND = (7.5, 13.0)
THETA_BAND = (4.0, 8.0)
STIM_FREQ_HZ = 6.0

# Klimesch's anchor. The theta/alpha transition sits several Hz below IAF; 5 Hz
# is the common shorthand when no rest-vs-task contrast is available to locate
# the crossing directly.
KLIMESCH_OFFSET_HZ = 5.0
ITF_BOUNDS = (4.0, 8.0)

SKIP_FIRST_SEC = 10
ARTIFACT_THRESH_UV = 150


def load_easy(path: Path) -> np.ndarray:
    """Read the 8 EEG channels from a NIC .easy file (tab-separated, no header)."""
    raw = pd.read_csv(path, sep='\t', header=None, usecols=range(8),
                      dtype=np.float64).to_numpy()
    return raw / 1000.0     # nV -> uV


def channel_index(label: str) -> int:
    return CH_LABELS.index(label)


def usable_channels(easy_path: Path, info_path: Optional[Path]) -> List[str]:
    """
    Which channels carry EEG in this run.

    Runs where four channels are stimulating leave only F4, P4, P3 usable. The
    .info file records how many EEG channels the protocol configured, which is
    what distinguishes the two cases.
    """
    n_eeg = None
    if info_path is not None and info_path.exists():
        for line in info_path.read_text(errors='ignore').splitlines():
            if 'Number of EEG channels' in line:
                try:
                    n_eeg = int(line.split(':')[1].strip())
                except (IndexError, ValueError):
                    pass
                break

    if n_eeg is not None and n_eeg >= 7:
        return ['F3', 'Fp1', 'FCz', 'FT7', 'F4', 'P4', 'P3']
    return ['F4', 'P4', 'P3']


def compute_psd(x: np.ndarray, fs: int = FS) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD after dropping the startup transient and gross artifacts."""
    x = x[int(SKIP_FIRST_SEC * fs):]
    if len(x) < fs * 20:
        return np.array([]), np.array([])

    x = signal.detrend(x)
    x = x[np.abs(x) < ARTIFACT_THRESH_UV]
    if len(x) < fs * 20:
        return np.array([]), np.array([])

    freqs, psd = signal.welch(x, fs=fs, nperseg=int(4 * fs),
                              noverlap=int(2 * fs), scaling='density')
    return freqs, psd


def band_peak(freqs: np.ndarray, psd: np.ndarray,
              band: Tuple[float, float]) -> Optional[float]:
    """
    Peak frequency within a band, after removing the 1/f background.

    The aperiodic component is subtracted by fitting a line to log-log power,
    because otherwise the peak of a band nearly always lands at its lower edge
    — the 1/f slope dominates any genuine oscillation. Returns None when the
    maximum sits on a band edge, since that indicates no resolved peak.
    """
    if len(freqs) == 0:
        return None

    mask = (freqs >= 1) & (freqs <= 45) & (freqs > 0)
    if mask.sum() < 10:
        return None

    log_f = np.log10(freqs[mask])
    log_p = np.log10(np.maximum(psd[mask], 1e-30))
    slope, intercept = np.polyfit(log_f, log_p, 1)
    flattened = log_p - (intercept + slope * log_f)

    f_band = freqs[mask]
    in_band = (f_band >= band[0]) & (f_band <= band[1])
    if in_band.sum() < 3:
        return None

    band_freqs = f_band[in_band]
    band_power = flattened[in_band]
    idx = int(np.argmax(band_power))

    # A maximum at either edge is a slope artifact, not a peak.
    if idx == 0 or idx == len(band_power) - 1:
        return None
    return float(band_freqs[idx])


def estimate_subject(subject_id: str, runs: List[int],
                     verbose: bool = False) -> Optional[Dict]:
    """Estimate IAF and theta peak for one subject, averaged over the runs given."""
    available = discover_runs(subject_id)
    wanted = [r for r in runs if r in available]
    if not wanted:
        return None

    iafs, theta_peaks, n_channels_seen = [], [], set()

    for run_num in wanted:
        files = available[run_num]
        try:
            eeg = load_easy(files['easy'])
        except Exception:
            continue

        chans = usable_channels(files['easy'], files['info'])
        n_channels_seen.add(len(chans))

        # --- IAF from posterior channels ---
        run_iaf = []
        for ch in POSTERIOR_CH:
            if ch not in chans:
                continue
            freqs, psd = compute_psd(eeg[:, channel_index(ch)])
            peak = band_peak(freqs, psd, ALPHA_BAND)
            if peak is not None:
                run_iaf.append(peak)
        if run_iaf:
            iafs.append(float(np.mean(run_iaf)))

        # --- Direct theta peak at frontal midline, where recorded ---
        for ch in MIDLINE_CH:
            if ch not in chans:
                continue
            freqs, psd = compute_psd(eeg[:, channel_index(ch)])
            peak = band_peak(freqs, psd, THETA_BAND)
            if peak is not None:
                theta_peaks.append(peak)

    iaf = float(np.mean(iafs)) if iafs else np.nan
    theta_peak = float(np.mean(theta_peaks)) if theta_peaks else np.nan

    itf_klimesch = (float(np.clip(iaf - KLIMESCH_OFFSET_HZ, *ITF_BOUNDS))
                    if np.isfinite(iaf) else np.nan)

    return {
        'subject_id': subject_id,
        'iaf': iaf,
        'n_iaf_runs': len(iafs),
        'itf_klimesch': itf_klimesch,
        'theta_peak_fcz': theta_peak,
        'n_theta_runs': len(theta_peaks),
        'max_channels': max(n_channels_seen) if n_channels_seen else 0,
        'runs_used': ','.join(str(r) for r in wanted),
        'in_dissertation': subject_id in DISSERTATION_SUBJECTS,
        # Distance from the frequency everyone actually received. This is the
        # moderator the paper tests.
        'itf_distance': (abs(itf_klimesch - STIM_FREQ_HZ)
                         if np.isfinite(itf_klimesch) else np.nan),
    }


def estimate_all(runs: List[int] = [1], subjects: Optional[List[str]] = None,
                 verbose: bool = True) -> pd.DataFrame:
    if subjects is None:
        subjects = list(SUBJECT_INFO)

    rows = []
    for i, sid in enumerate(sorted(subjects), 1):
        result = estimate_subject(sid, runs)
        if result is None:
            if verbose:
                print(f'[{i}/{len(subjects)}] sub-{sid}: no EEG')
            continue
        rows.append(result)
        if verbose:
            print(f'[{i}/{len(subjects)}] sub-{sid}: '
                  f'IAF {result["iaf"]:.2f} Hz -> iTF {result["itf_klimesch"]:.2f} Hz'
                  if np.isfinite(result['iaf'])
                  else f'[{i}/{len(subjects)}] sub-{sid}: no alpha peak resolved')

    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--runs', type=int, nargs='+', default=[1],
                        help='baseline runs to estimate from (default: 1 only)')
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR))
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    df = estimate_all(runs=args.runs, verbose=not args.quiet)

    print('\n' + '=' * 66)
    print('Individualized theta frequency')
    print('=' * 66)
    print(f'  subjects with EEG           : {len(df)}')
    print(f'  IAF resolved                : {df["iaf"].notna().sum()}')
    print(f'  iTF (Klimesch)              : {df["itf_klimesch"].notna().sum()}')
    print(f'  direct theta peak at FCz    : {df["theta_peak_fcz"].notna().sum()}')

    if df['iaf'].notna().any():
        print(f'\n  IAF  : M = {df["iaf"].mean():.2f} Hz, '
              f'SD = {df["iaf"].std():.2f}, '
              f'range {df["iaf"].min():.2f}-{df["iaf"].max():.2f}')
        print(f'  iTF  : M = {df["itf_klimesch"].mean():.2f} Hz, '
              f'SD = {df["itf_klimesch"].std():.2f}')
        print(f'  |iTF - {STIM_FREQ_HZ} Hz| : M = {df["itf_distance"].mean():.2f} Hz, '
              f'max = {df["itf_distance"].max():.2f}')

    # Empirical check on the assumed offset, where both estimators exist.
    both = df[df['iaf'].notna() & df['theta_peak_fcz'].notna()]
    if len(both) >= 5:
        offset = both['iaf'] - both['theta_peak_fcz']
        r = both['iaf'].corr(both['theta_peak_fcz'])
        print(f'\n  Both estimators available for {len(both)} subjects:')
        print(f'    observed IAF - theta peak = {offset.mean():.2f} Hz '
              f'(SD {offset.std():.2f}); assumed offset is {KLIMESCH_OFFSET_HZ}')
        print(f'    IAF vs direct theta peak: r = {r:+.3f}')
    else:
        print(f'\n  Too few subjects ({len(both)}) have both estimators to check '
              f'the offset empirically.')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'individual_theta_frequency.csv'
    df.to_csv(path, index=False)
    print(f'\nWrote {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
