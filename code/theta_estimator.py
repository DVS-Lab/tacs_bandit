"""
Estimate participant-specific feedback-locked theta from bandit localizer EEG.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d

try:  # pragma: no cover - optional dependency
    import mne
except ImportError:  # pragma: no cover - optional dependency
    mne = None

try:  # pragma: no cover - optional dependency
    import pyxdf
except ImportError:  # pragma: no cover - optional dependency
    pyxdf = None


EPSILON = 1e-12
DEFAULT_ALIGNMENT_TOLERANCE_SEC = 0.05


@dataclass
class EEGData:
    samples: np.ndarray
    timestamps: np.ndarray
    channel_names: list[str]
    sampling_rate_hz: float
    metadata: Dict[str, Any]


@dataclass
class ThetaEstimateArtifacts:
    json_path: str
    csv_path: str
    plot_paths: Dict[str, str]
    html_report_path: Optional[str]


def _script_root() -> Path:
    return Path(__file__).resolve().parent


def _repo_root() -> Path:
    return _script_root().parent


def load_config(config_path: Optional[Path | str] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = _script_root() / "config.json"
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_analysis_version() -> str:
    repo_root = _repo_root()
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return output or "unknown"
    except Exception:
        return "snapshot"


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, np.ndarray):
        if raw.size == 0:
            return {}
        raw = raw.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw_metadata": raw}
    if isinstance(raw, dict):
        return raw
    return {"raw_metadata": raw}


def load_eeg_recording(eeg_path: Path | str) -> EEGData:
    eeg_path = Path(eeg_path)
    suffix = eeg_path.suffix.lower()

    if suffix == ".npz":
        with np.load(eeg_path, allow_pickle=True) as data:
            samples = np.asarray(data["samples"], dtype=float)
            timestamps = np.asarray(data["timestamps"], dtype=float)
            channel_names = [str(item) for item in data["channel_names"].tolist()]
            srate_array = np.asarray(data["sampling_rate_hz"], dtype=float).reshape(-1)
            sampling_rate_hz = float(srate_array[0])
            metadata = _parse_metadata(data.get("metadata_json"))
        return EEGData(samples, timestamps, channel_names, sampling_rate_hz, metadata)

    if suffix == ".csv":
        df = pd.read_csv(eeg_path)
        if "lsl_timestamp" not in df.columns:
            raise ValueError("EEG CSV must contain an 'lsl_timestamp' column.")
        channel_names = [column for column in df.columns if column != "lsl_timestamp"]
        timestamps = df["lsl_timestamp"].to_numpy(dtype=float)
        samples = df[channel_names].to_numpy(dtype=float)
        sampling_rate_hz = estimate_sampling_rate_from_timestamps(timestamps)
        return EEGData(samples, timestamps, channel_names, sampling_rate_hz, {})

    if suffix in {".edf", ".bdf", ".set"}:
        if mne is None:
            raise ImportError(
                f"Loading {suffix} files requires MNE-Python. Install mne to use offline import."
            )
        if suffix == ".edf":
            raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose="ERROR")
        elif suffix == ".bdf":
            raw = mne.io.read_raw_bdf(eeg_path, preload=True, verbose="ERROR")
        else:
            raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose="ERROR")
        samples = raw.get_data().T * 1e6
        srate = float(raw.info["sfreq"])
        timestamps = np.arange(samples.shape[0], dtype=float) / srate
        return EEGData(samples, timestamps, raw.ch_names, srate, {"source_format": suffix})

    if suffix == ".xdf":
        if pyxdf is None:
            raise ImportError(
                "Loading .xdf files requires pyxdf. Install pyxdf to use offline import."
            )
        streams, header = pyxdf.load_xdf(str(eeg_path))
        eeg_stream = next(
            (stream for stream in streams if stream["info"]["type"][0].lower() == "eeg"),
            None,
        )
        if eeg_stream is None:
            raise ValueError("No EEG stream was found in the XDF file.")
        samples = np.asarray(eeg_stream["time_series"], dtype=float)
        timestamps = np.asarray(eeg_stream["time_stamps"], dtype=float)
        channel_names = []
        channels_desc = (
            eeg_stream["info"]
            .get("desc", [{}])[0]
            .get("channels", [{}])[0]
            .get("channel", [])
        )
        for idx, channel in enumerate(channels_desc):
            label = channel.get("label", [f"ch{idx + 1}"])[0]
            channel_names.append(label)
        if not channel_names:
            channel_names = [f"ch{i + 1}" for i in range(samples.shape[1])]
        srate = float(eeg_stream["info"]["nominal_srate"][0])
        return EEGData(samples, timestamps, channel_names, srate, {"xdf_header": header})

    raise ValueError(f"Unsupported EEG file format: {eeg_path.suffix}")


def load_behavioral_events(localizer_csv: Path | str) -> pd.DataFrame:
    df = pd.read_csv(localizer_csv)
    if "feedback_marker" not in df.columns:
        raise ValueError("Behavioral CSV must contain a 'feedback_marker' column.")
    return df


def estimate_sampling_rate_from_timestamps(timestamps: np.ndarray) -> float:
    if len(timestamps) < 2:
        return float("nan")
    diffs = np.diff(timestamps)
    median_diff = float(np.median(diffs[diffs > 0]))
    if not math.isfinite(median_diff) or median_diff <= 0:
        return float("nan")
    return 1.0 / median_diff


def summarize_timestamp_quality(
    timestamps: np.ndarray,
    sampling_rate_hz: float,
) -> Dict[str, Any]:
    if len(timestamps) < 2:
        return {
            "median_step_sec": None,
            "duplicate_timestamps": 0,
            "gaps_detected": 0,
            "irregular_intervals": 0,
        }
    diffs = np.diff(timestamps)
    positive_diffs = diffs[diffs > 0]
    median_step = float(np.median(positive_diffs)) if len(positive_diffs) else None
    expected_step = 1.0 / sampling_rate_hz if sampling_rate_hz and math.isfinite(sampling_rate_hz) else None
    duplicate_timestamps = int(np.sum(diffs == 0))
    gaps_detected = 0
    irregular_intervals = 0
    if expected_step:
        gaps_detected = int(np.sum(diffs > expected_step * 1.5))
        irregular_intervals = int(np.sum(np.abs(diffs - expected_step) > expected_step * 0.2))
    return {
        "median_step_sec": median_step,
        "expected_step_sec": expected_step,
        "duplicate_timestamps": duplicate_timestamps,
        "gaps_detected": gaps_detected,
        "irregular_intervals": irregular_intervals,
    }


def _longest_flatline_seconds(channel: np.ndarray, sampling_rate_hz: float) -> float:
    if len(channel) < 2 or sampling_rate_hz <= 0:
        return 0.0
    diffs = np.abs(np.diff(channel))
    flat_mask = diffs < 1e-6
    if not np.any(flat_mask):
        return 0.0
    longest = 0
    current = 0
    for is_flat in flat_mask:
        if is_flat:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest / sampling_rate_hz


def detect_bad_channels(
    eeg: np.ndarray,
    channel_names: Sequence[str],
    sampling_rate_hz: float,
    config: Dict[str, Any],
) -> tuple[list[str], Dict[str, Dict[str, float]]]:
    flatline_threshold_sec = float(config.get("flatline_threshold_sec", 2.0))
    max_bad_channels = int(config.get("max_bad_channels", 2))

    metrics: Dict[str, Dict[str, float]] = {}
    stds = np.std(eeg, axis=0)
    median_std = float(np.median(stds)) if len(stds) else 0.0
    bad_channels: list[str] = []

    for index, name in enumerate(channel_names):
        channel = eeg[:, index]
        longest_flat = _longest_flatline_seconds(channel, sampling_rate_hz)
        std = float(stds[index])
        variance_ratio = std / max(median_std, EPSILON)

        freqs, psd = signal.welch(channel, fs=sampling_rate_hz, nperseg=min(1024, len(channel)))
        band_mask = (freqs >= 4.0) & (freqs <= 40.0)
        line_mask = (freqs >= 59.0) & (freqs <= 61.0)
        line_noise_ratio = (
            float(np.mean(psd[line_mask]) / max(np.mean(psd[band_mask]), EPSILON))
            if np.any(line_mask) and np.any(band_mask)
            else 0.0
        )

        saturation_fraction = float(
            np.mean(np.isclose(np.abs(channel), np.max(np.abs(channel))))
        )

        metrics[name] = {
            "std_uv": std,
            "variance_ratio": variance_ratio,
            "longest_flatline_sec": longest_flat,
            "line_noise_ratio": line_noise_ratio,
            "saturation_fraction": saturation_fraction,
        }

        if (
            longest_flat >= flatline_threshold_sec
            or std < 0.1
            or variance_ratio > 8.0
            or line_noise_ratio > 3.0
            or saturation_fraction > 0.02
        ):
            bad_channels.append(name)

    if len(bad_channels) > max_bad_channels:
        bad_channels = sorted(
            bad_channels,
            key=lambda name: (
                metrics[name]["variance_ratio"],
                metrics[name]["line_noise_ratio"],
                metrics[name]["longest_flatline_sec"],
            ),
            reverse=True,
        )
    return bad_channels, metrics


def _zero_phase_filter(
    eeg: np.ndarray,
    sampling_rate_hz: float,
    *,
    highpass_hz: float,
    lowpass_hz: float,
) -> np.ndarray:
    nyquist = sampling_rate_hz / 2.0
    highpass_hz = max(0.0, float(highpass_hz))
    lowpass_hz = min(float(lowpass_hz), nyquist - 0.5)
    if highpass_hz <= 0 and lowpass_hz <= 0:
        return eeg
    if highpass_hz > 0 and lowpass_hz > 0 and highpass_hz < lowpass_hz:
        b, a = signal.butter(4, [highpass_hz / nyquist, lowpass_hz / nyquist], btype="band")
    elif highpass_hz > 0:
        b, a = signal.butter(4, highpass_hz / nyquist, btype="highpass")
    else:
        b, a = signal.butter(4, lowpass_hz / nyquist, btype="lowpass")
    return signal.filtfilt(b, a, eeg, axis=0)


def _apply_notch(eeg: np.ndarray, sampling_rate_hz: float, notch_hz: float) -> np.ndarray:
    if notch_hz <= 0 or notch_hz >= sampling_rate_hz / 2.0:
        return eeg
    b, a = signal.iirnotch(notch_hz / (sampling_rate_hz / 2.0), Q=30.0)
    return signal.filtfilt(b, a, eeg, axis=0)


def resample_eeg(
    eeg: np.ndarray,
    timestamps: np.ndarray,
    sampling_rate_hz: float,
    target_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not target_rate_hz or not math.isfinite(target_rate_hz) or target_rate_hz <= 0:
        return eeg, timestamps, sampling_rate_hz
    if abs(target_rate_hz - sampling_rate_hz) < 1e-6:
        return eeg, timestamps, sampling_rate_hz
    target_samples = max(1, int(round(len(eeg) * target_rate_hz / sampling_rate_hz)))
    resampled = signal.resample(eeg, target_samples, axis=0)
    new_timestamps = np.linspace(timestamps[0], timestamps[-1], target_samples)
    return resampled, new_timestamps, float(target_rate_hz)


def rereference_eeg(
    eeg: np.ndarray,
    channel_names: Sequence[str],
    bad_channels: Sequence[str],
    reference_mode: str,
) -> np.ndarray:
    if reference_mode != "average_available":
        return eeg
    good_indices = [idx for idx, name in enumerate(channel_names) if name not in bad_channels]
    if not good_indices:
        return eeg
    reference = np.mean(eeg[:, good_indices], axis=1, keepdims=True)
    return eeg - reference


def get_feedback_events(
    behavior_df: pd.DataFrame,
    *,
    alignment_tolerance_sec: float = DEFAULT_ALIGNMENT_TOLERANCE_SEC,
) -> tuple[np.ndarray, list[str], Dict[str, Any]]:
    feedback_df = behavior_df[behavior_df["feedback_marker"].isin([31, 32, 33])].copy()
    if feedback_df.empty:
        raise ValueError("No feedback markers (31/32/33) were found in the behavioral CSV.")

    used_behavioral_fallback = False
    if "feedback_onset_lsl_time" in feedback_df.columns and feedback_df["feedback_onset_lsl_time"].notna().any():
        event_times = feedback_df["feedback_onset_lsl_time"].to_numpy(dtype=float)
    elif {
        "run_start_lsl_time",
        "feedback_onset_task_time",
    }.issubset(feedback_df.columns):
        event_times = (
            feedback_df["run_start_lsl_time"].astype(float)
            + feedback_df["feedback_onset_task_time"].astype(float)
        ).to_numpy(dtype=float)
        used_behavioral_fallback = True
    else:
        raise ValueError(
            "Behavioral CSV is missing both feedback_onset_lsl_time and "
            "run_start_lsl_time + feedback_onset_task_time fallback columns."
        )

    labels = feedback_df["feedback_marker"].map({31: "win", 32: "loss", 33: "miss"}).tolist()

    mismatch_warning = None
    if "lsl_marker_send_time" in feedback_df.columns and feedback_df["lsl_marker_send_time"].notna().any():
        marker_times = feedback_df["lsl_marker_send_time"].to_numpy(dtype=float)
        mismatch = np.abs(marker_times - event_times)
        if len(mismatch) and float(np.nanmax(mismatch)) > alignment_tolerance_sec:
            mismatch_warning = (
                "Feedback marker send times differed from feedback_onset_lsl_time by more than "
                f"{alignment_tolerance_sec:.3f} s."
            )

    return event_times, labels, {
        "used_behavioral_timestamp_fallback": used_behavioral_fallback,
        "alignment_warning": mismatch_warning,
        "n_feedback_epochs_total": int(len(event_times)),
    }


def extract_epochs(
    eeg_data: EEGData,
    event_times: np.ndarray,
    *,
    tmin_sec: float,
    tmax_sec: float,
    epoch_reject_uv: float,
    step_reject_uv: float,
) -> tuple[np.ndarray, np.ndarray, list[int], Dict[int, str]]:
    n_pre = int(round(abs(tmin_sec) * eeg_data.sampling_rate_hz))
    n_post = int(round(tmax_sec * eeg_data.sampling_rate_hz))
    n_samples = n_pre + n_post
    times = np.arange(n_samples, dtype=float) / eeg_data.sampling_rate_hz + tmin_sec

    retained: list[np.ndarray] = []
    retained_indices: list[int] = []
    rejection_reasons: Dict[int, str] = {}

    for event_index, event_time in enumerate(event_times):
        center = int(np.searchsorted(eeg_data.timestamps, event_time))
        start = center - n_pre
        stop = center + n_post
        if start < 0 or stop > len(eeg_data.samples):
            rejection_reasons[event_index] = "epoch_out_of_bounds"
            continue

        epoch = eeg_data.samples[start:stop].T
        if epoch.shape[1] != n_samples:
            rejection_reasons[event_index] = "epoch_shape_mismatch"
            continue
        if np.max(np.abs(epoch)) > epoch_reject_uv:
            rejection_reasons[event_index] = "amplitude_reject"
            continue
        if np.max(np.abs(np.diff(epoch, axis=1))) > step_reject_uv:
            rejection_reasons[event_index] = "step_reject"
            continue

        retained.append(epoch)
        retained_indices.append(event_index)

    if retained:
        epochs = np.stack(retained, axis=0)
    else:
        epochs = np.empty((0, eeg_data.samples.shape[1], n_samples), dtype=float)
    return epochs, times, retained_indices, rejection_reasons


def select_roi_channels(
    channel_names: Sequence[str],
    bad_channels: Sequence[str],
    config: Dict[str, Any],
) -> list[str]:
    requested = list(config.get("frontocentral_roi", ["Fz", "FCz", "Cz", "F3", "F4"]))
    available = [name for name in requested if name in channel_names and name not in bad_channels]
    if available:
        return available
    if config.get("fallback_roi_strategy") == "use_available_frontocentral_channels":
        frontocentral_hints = {"f3", "f4", "fz", "fcz", "cz", "fc1", "fc2", "c3", "c4"}
        return [
            name
            for name in channel_names
            if name.lower() in frontocentral_hints and name not in bad_channels
        ]
    return []


def _morlet_wavelet(freq_hz: float, sampling_rate_hz: float, n_cycles: float = 5.0) -> np.ndarray:
    sigma_t = n_cycles / (2.0 * math.pi * freq_hz)
    half_width = int(math.ceil(3.5 * sigma_t * sampling_rate_hz))
    times = np.arange(-half_width, half_width + 1, dtype=float) / sampling_rate_hz
    wavelet = np.exp(2j * math.pi * freq_hz * times) * np.exp(-(times**2) / (2 * sigma_t**2))
    wavelet /= math.sqrt(np.sum(np.abs(wavelet) ** 2))
    return wavelet


def compute_epoch_tfr(
    epochs: np.ndarray,
    sampling_rate_hz: float,
    frequencies_hz: np.ndarray,
    *,
    baseline_mask: np.ndarray,
) -> np.ndarray:
    n_epochs, n_channels, n_times = epochs.shape
    power = np.empty((n_epochs, n_channels, len(frequencies_hz), n_times), dtype=float)
    for freq_index, freq_hz in enumerate(frequencies_hz):
        wavelet = _morlet_wavelet(freq_hz, sampling_rate_hz)
        for epoch_index in range(n_epochs):
            for channel_index in range(n_channels):
                analytic = signal.fftconvolve(
                    epochs[epoch_index, channel_index],
                    wavelet,
                    mode="same",
                )
                power[epoch_index, channel_index, freq_index] = np.abs(analytic) ** 2

    baseline = np.mean(power[..., baseline_mask], axis=-1, keepdims=True)
    baseline = np.maximum(baseline, EPSILON)
    return 10.0 * np.log10(np.maximum(power, EPSILON) / baseline)


def smooth_spectrum(
    spectrum: np.ndarray,
    *,
    smooth_hz: float,
    frequency_step_hz: float,
) -> np.ndarray:
    if smooth_hz <= 0 or frequency_step_hz <= 0:
        return spectrum
    sigma_bins = smooth_hz / max(frequency_step_hz, EPSILON) / 2.355
    if sigma_bins <= 0:
        return spectrum
    return gaussian_filter1d(spectrum, sigma=sigma_bins, mode="nearest")


def estimate_peak_from_spectrum(
    frequencies_hz: np.ndarray,
    spectrum: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    theta_band = config.get("primary_theta_band_hz", [4.0, 8.0])
    lower, upper = float(theta_band[0]), float(theta_band[1])
    band_mask = (frequencies_hz >= lower) & (frequencies_hz <= upper)
    band_freqs = frequencies_hz[band_mask]
    band_spec = spectrum[band_mask]
    if len(band_freqs) == 0:
        return {
            "peak_hz_raw": None,
            "peak_hz_rounded": None,
            "peak_prominence_z": 0.0,
            "edge_peak": True,
            "passes_peak_criteria": False,
        }

    peak_indices, properties = signal.find_peaks(band_spec, prominence=0)
    prominence = 0.0
    if len(peak_indices):
        heights = band_spec[peak_indices]
        best_peak_position = int(np.argmax(heights))
        peak_index = int(peak_indices[best_peak_position])
        prominence = float(properties["prominences"][best_peak_position])
    else:
        peak_index = int(np.argmax(band_spec))

    raw_hz = float(band_freqs[peak_index])
    rounding = float(config.get("round_to_nearest_hz", 0.5))
    rounded_hz = math.floor(raw_hz / rounding + 0.5) * rounding if rounding else raw_hz
    edge_peak = peak_index in {0, len(band_freqs) - 1}
    prominence_z = prominence / max(float(np.std(band_spec)), EPSILON)
    passes = prominence_z >= float(config.get("min_peak_prominence_z", 0.5))
    if config.get("reject_edge_peaks", True) and edge_peak:
        passes = False
    return {
        "peak_hz_raw": raw_hz,
        "peak_hz_rounded": rounded_hz,
        "peak_prominence_z": float(prominence_z),
        "edge_peak": bool(edge_peak),
        "passes_peak_criteria": bool(passes),
    }


def bootstrap_peak_distribution(
    epoch_spectra: np.ndarray,
    frequencies_hz: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    if len(epoch_spectra) == 0:
        return np.empty((0,), dtype=float)
    rng = np.random.default_rng(42)
    peaks = []
    smooth_hz = float(config.get("smooth_spectrum_hz", 0.5))
    step_hz = float(config.get("frequency_step_hz", 0.25))
    for _ in range(int(config.get("bootstrap_iterations", 500))):
        sample_indices = rng.integers(0, len(epoch_spectra), size=len(epoch_spectra))
        sampled_spectrum = np.mean(epoch_spectra[sample_indices], axis=0)
        smoothed = smooth_spectrum(
            sampled_spectrum,
            smooth_hz=smooth_hz,
            frequency_step_hz=step_hz,
        )
        peak = estimate_peak_from_spectrum(frequencies_hz, smoothed, config)["peak_hz_raw"]
        if peak is not None:
            peaks.append(float(peak))
    return np.asarray(peaks, dtype=float)


def split_half_peak_estimates(
    epoch_spectra: np.ndarray,
    frequencies_hz: np.ndarray,
    config: Dict[str, Any],
) -> tuple[Optional[float], Optional[float]]:
    if len(epoch_spectra) < 2:
        return None, None
    midpoint = len(epoch_spectra) // 2
    first_half = epoch_spectra[:midpoint]
    second_half = epoch_spectra[midpoint:]
    if len(first_half) == 0 or len(second_half) == 0:
        return None, None
    smooth_hz = float(config.get("smooth_spectrum_hz", 0.5))
    step_hz = float(config.get("frequency_step_hz", 0.25))
    odd_peak = estimate_peak_from_spectrum(
        frequencies_hz,
        smooth_spectrum(np.mean(first_half, axis=0), smooth_hz=smooth_hz, frequency_step_hz=step_hz),
        config,
    )["peak_hz_raw"]
    even_peak = estimate_peak_from_spectrum(
        frequencies_hz,
        smooth_spectrum(np.mean(second_half, axis=0), smooth_hz=smooth_hz, frequency_step_hz=step_hz),
        config,
    )["peak_hz_raw"]
    return odd_peak, even_peak


def _fallback_frequency(config: Dict[str, Any]) -> float:
    return float(
        config.get("theta_estimation", {}).get(
            "default_fixed_theta_hz",
            config.get("stimulation_frequency_selection", {}).get("default_fixed_theta_hz", 6.0),
        )
    )


def _valid_theta_range(config: Dict[str, Any]) -> tuple[float, float]:
    theta_config = config.get("theta_estimation", {})
    selection_config = config.get("stimulation_frequency_selection", {})
    values = selection_config.get("valid_theta_range_hz", theta_config.get("primary_theta_band_hz", [4.0, 8.0]))
    return float(values[0]), float(values[1])


def _decision_from_failures(failures: list[str]) -> str:
    if not failures:
        return "Reliable feedback-locked theta estimate. Rounded to nearest 0.5 Hz."
    return "Unreliable estimate: " + "; ".join(failures)


def estimate_feedback_theta(
    eeg_data: EEGData,
    behavior_df: pd.DataFrame,
    config: Dict[str, Any],
    *,
    subject_id: str,
    session_id: str,
    localizer_run: str = "run-localizer",
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    theta_config = config.get("theta_estimation", {})
    preprocessing_config = config.get("eeg_preprocessing", {})

    eeg = np.asarray(eeg_data.samples, dtype=float)
    timestamps = np.asarray(eeg_data.timestamps, dtype=float)
    channel_names = list(eeg_data.channel_names)
    sampling_rate_hz = float(eeg_data.sampling_rate_hz)

    timestamp_qc_before = summarize_timestamp_quality(timestamps, sampling_rate_hz)
    bad_channels, channel_metrics = detect_bad_channels(
        eeg,
        channel_names,
        sampling_rate_hz,
        preprocessing_config,
    )

    eeg = _zero_phase_filter(
        eeg,
        sampling_rate_hz,
        highpass_hz=float(preprocessing_config.get("highpass_hz", 0.5)),
        lowpass_hz=float(preprocessing_config.get("lowpass_hz", 40.0)),
    )
    eeg = _apply_notch(
        eeg,
        sampling_rate_hz,
        float(preprocessing_config.get("notch_hz", 60.0)),
    )

    target_rate = float(preprocessing_config.get("resample_hz", sampling_rate_hz))
    eeg, timestamps, sampling_rate_hz = resample_eeg(
        eeg,
        timestamps,
        sampling_rate_hz,
        target_rate,
    )
    eeg = rereference_eeg(
        eeg,
        channel_names,
        bad_channels,
        preprocessing_config.get("reference", "average_available"),
    )

    eeg_data = EEGData(eeg, timestamps, channel_names, sampling_rate_hz, eeg_data.metadata)
    timestamp_qc_after = summarize_timestamp_quality(timestamps, sampling_rate_hz)

    roi_channels = select_roi_channels(channel_names, bad_channels, theta_config)
    roi_indices = [channel_names.index(name) for name in roi_channels]

    event_times, event_labels, alignment_qc = get_feedback_events(
        behavior_df,
        alignment_tolerance_sec=float(theta_config.get("event_alignment_tolerance_sec", DEFAULT_ALIGNMENT_TOLERANCE_SEC)),
    )

    epochs, epoch_times, retained_indices, rejection_reasons = extract_epochs(
        eeg_data,
        event_times,
        tmin_sec=float(theta_config.get("epoch_tmin_sec", -1.0)),
        tmax_sec=float(theta_config.get("epoch_tmax_sec", 1.5)),
        epoch_reject_uv=float(preprocessing_config.get("epoch_reject_uv", 150.0)),
        step_reject_uv=float(preprocessing_config.get("step_reject_uv", 75.0)),
    )

    retained_labels = [event_labels[index] for index in retained_indices]
    total_epochs = int(len(event_times))
    retained_epochs = int(len(retained_indices))
    usable_epoch_fraction = retained_epochs / total_epochs if total_epochs else 0.0

    frequencies_hz = np.arange(
        float(theta_config.get("exploratory_theta_band_hz", [3.0, 10.0])[0]),
        float(theta_config.get("exploratory_theta_band_hz", [3.0, 10.0])[1]) + EPSILON,
        float(theta_config.get("frequency_step_hz", 0.25)),
    )
    baseline_window = theta_config.get("baseline_window_sec", [-0.5, -0.1])
    theta_window = theta_config.get("theta_window_sec", [0.2, 0.8])
    baseline_mask = (epoch_times >= float(baseline_window[0])) & (epoch_times <= float(baseline_window[1]))
    theta_mask = (epoch_times >= float(theta_window[0])) & (epoch_times <= float(theta_window[1]))

    epoch_spectra = np.empty((0, len(frequencies_hz)), dtype=float)
    roi_tfr = np.empty((len(frequencies_hz), len(epoch_times)), dtype=float)
    spectrum = np.zeros(len(frequencies_hz), dtype=float)
    win_spectrum = np.zeros(len(frequencies_hz), dtype=float)
    loss_spectrum = np.zeros(len(frequencies_hz), dtype=float)
    peak_info = {
        "peak_hz_raw": None,
        "peak_hz_rounded": None,
        "peak_prominence_z": 0.0,
        "edge_peak": False,
        "passes_peak_criteria": False,
    }
    bootstrap_peaks = np.empty((0,), dtype=float)
    split_half_1 = None
    split_half_2 = None

    failures: list[str] = []

    if len(roi_indices) < 2:
        failures.append("fewer than 2 usable frontocentral ROI channels")

    if retained_epochs:
        roi_epochs = epochs[:, roi_indices]
        tfr = compute_epoch_tfr(
            roi_epochs,
            sampling_rate_hz,
            frequencies_hz,
            baseline_mask=baseline_mask,
        )
        roi_tfr = np.mean(tfr, axis=(0, 1))
        epoch_spectra = np.mean(tfr[..., theta_mask], axis=(1, 3))

        smooth_hz = float(theta_config.get("smooth_spectrum_hz", 0.5))
        step_hz = float(theta_config.get("frequency_step_hz", 0.25))
        spectrum = smooth_spectrum(np.mean(epoch_spectra, axis=0), smooth_hz=smooth_hz, frequency_step_hz=step_hz)

        if np.any(np.array(retained_labels) == "win"):
            win_spectrum = smooth_spectrum(
                np.mean(epoch_spectra[np.array(retained_labels) == "win"], axis=0),
                smooth_hz=smooth_hz,
                frequency_step_hz=step_hz,
            )
        if np.any(np.array(retained_labels) == "loss"):
            loss_spectrum = smooth_spectrum(
                np.mean(epoch_spectra[np.array(retained_labels) == "loss"], axis=0),
                smooth_hz=smooth_hz,
                frequency_step_hz=step_hz,
            )

        peak_info = estimate_peak_from_spectrum(frequencies_hz, spectrum, theta_config)
        split_half_1, split_half_2 = split_half_peak_estimates(epoch_spectra, frequencies_hz, theta_config)
        bootstrap_peaks = bootstrap_peak_distribution(epoch_spectra, frequencies_hz, theta_config)

    hard_min_epochs = int(config.get("localizer_fast_theta", {}).get("hard_min_usable_epochs", 60))
    recommended_min_epochs = int(config.get("localizer_fast_theta", {}).get("recommended_min_usable_epochs", 80))
    min_fraction = float(theta_config.get("min_usable_epoch_fraction", 0.60))
    recommended_fraction = float(theta_config.get("recommended_usable_epoch_fraction", 0.70))
    split_half_max_diff = float(theta_config.get("split_half_max_peak_diff_hz", 1.0))
    max_bootstrap_ci_width = float(theta_config.get("max_bootstrap_ci_width_hz", 2.0))

    if retained_epochs < hard_min_epochs:
        failures.append(f"only {retained_epochs} usable epochs retained")
    if usable_epoch_fraction < min_fraction:
        failures.append(f"usable epoch fraction {usable_epoch_fraction:.3f} < {min_fraction:.2f}")
    if retained_epochs < recommended_min_epochs:
        failures.append(
            f"retained epochs below recommended minimum ({retained_epochs} < {recommended_min_epochs})"
        )
    if usable_epoch_fraction < recommended_fraction:
        failures.append(
            f"usable epoch fraction below recommended threshold ({usable_epoch_fraction:.3f} < {recommended_fraction:.2f})"
        )

    if not peak_info["passes_peak_criteria"]:
        failures.append(
            "peak prominence criterion failed"
            if not peak_info["edge_peak"]
            else "peak occurred at the edge of the search band"
        )

    split_half_diff = None
    if split_half_1 is None or split_half_2 is None:
        failures.append("split-half peak estimate could not be computed")
    else:
        split_half_diff = abs(split_half_1 - split_half_2)
        if split_half_diff > split_half_max_diff:
            failures.append(
                f"split-half peaks differed by {split_half_diff:.2f} Hz"
            )

    if len(bootstrap_peaks) == 0:
        bootstrap_ci_low = None
        bootstrap_ci_high = None
        bootstrap_ci_width = None
        failures.append("bootstrap peak distribution could not be computed")
    else:
        bootstrap_ci_low = float(np.quantile(bootstrap_peaks, 0.025))
        bootstrap_ci_high = float(np.quantile(bootstrap_peaks, 0.975))
        bootstrap_ci_width = bootstrap_ci_high - bootstrap_ci_low
        if bootstrap_ci_width > max_bootstrap_ci_width:
            failures.append(
                f"bootstrap CI width was {bootstrap_ci_width:.2f} Hz"
            )

    reliable = not failures
    fallback_hz = _fallback_frequency(config)
    valid_theta_range = _valid_theta_range(config)

    if reliable and peak_info["peak_hz_raw"] is not None:
        rounded_hz = min(
            valid_theta_range[1],
            max(valid_theta_range[0], float(peak_info["peak_hz_rounded"])),
        )
        theta_source = "reliable_itheta"
        frequency_to_use_hz = rounded_hz
        decision_reason = _decision_from_failures([])
        use_for_stimulation = True
    else:
        rounded_hz = None
        theta_source = "fallback_fixed_6hz"
        frequency_to_use_hz = fallback_hz
        decision_reason = _decision_from_failures(failures)
        use_for_stimulation = False

    result = {
        "subject_id": subject_id,
        "session_id": session_id,
        "localizer_run": localizer_run,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "analysis_version": get_analysis_version(),
        "theta_label": theta_config.get(
            "theta_label",
            "participant_specific_feedback_theta",
        ),
        "reliable": reliable,
        "itheta_hz_raw": peak_info["peak_hz_raw"],
        "itheta_hz_rounded": rounded_hz,
        "frequency_to_use_hz": frequency_to_use_hz,
        "fallback_theta_hz": fallback_hz,
        "theta_source": theta_source,
        "reliability": {
            "n_feedback_epochs_total": total_epochs,
            "n_epochs_retained": retained_epochs,
            "usable_epoch_fraction": usable_epoch_fraction,
            "split_half_peak_1_hz": split_half_1,
            "split_half_peak_2_hz": split_half_2,
            "split_half_diff_hz": split_half_diff,
            "bootstrap_ci_low_hz": bootstrap_ci_low,
            "bootstrap_ci_high_hz": bootstrap_ci_high,
            "bootstrap_ci_width_hz": bootstrap_ci_width,
            "peak_prominence_z": peak_info["peak_prominence_z"],
            "edge_peak": peak_info["edge_peak"],
        },
        "preprocessing": {
            "highpass_hz": float(preprocessing_config.get("highpass_hz", 0.5)),
            "lowpass_hz": float(preprocessing_config.get("lowpass_hz", 40.0)),
            "notch_hz": float(preprocessing_config.get("notch_hz", 60.0)),
            "resample_hz": sampling_rate_hz,
            "reference": preprocessing_config.get("reference", "average_available"),
            "bad_channels": bad_channels,
            "roi_channels_requested": theta_config.get("frontocentral_roi", ["Fz", "FCz", "Cz", "F3", "F4"]),
            "roi_channels_used": roi_channels,
        },
        "windows": {
            "epoch_tmin_sec": float(theta_config.get("epoch_tmin_sec", -1.0)),
            "epoch_tmax_sec": float(theta_config.get("epoch_tmax_sec", 1.5)),
            "baseline_window_sec": baseline_window,
            "theta_window_sec": theta_window,
            "primary_theta_band_hz": theta_config.get("primary_theta_band_hz", [4.0, 8.0]),
        },
        "decision": {
            "use_for_stimulation": use_for_stimulation,
            "reason": decision_reason,
        },
        "operator_summary": (
            f"Participant-specific feedback theta estimate: {frequency_to_use_hz:.1f} Hz. "
            f"{'Reliability criteria passed.' if reliable else decision_reason} "
            f"Use {frequency_to_use_hz:.1f} Hz for theta-tACS."
        ),
        "qc": {
            "timestamp_quality_before_preprocessing": timestamp_qc_before,
            "timestamp_quality_after_preprocessing": timestamp_qc_after,
            "channel_metrics": channel_metrics,
            "alignment": alignment_qc,
            "rejection_reasons": rejection_reasons,
            "retained_event_indices": retained_indices,
        },
    }

    debug = {
        "frequencies_hz": frequencies_hz,
        "epoch_times_sec": epoch_times,
        "roi_tfr_db": roi_tfr,
        "theta_spectrum_db": spectrum,
        "win_spectrum_db": win_spectrum,
        "loss_spectrum_db": loss_spectrum,
        "bootstrap_peaks_hz": bootstrap_peaks,
    }
    return result, debug


def _write_summary_csv(result: Dict[str, Any], csv_path: Path) -> None:
    row = {
        "subject_id": result["subject_id"],
        "session_id": result["session_id"],
        "localizer_run": result["localizer_run"],
        "reliable": result["reliable"],
        "itheta_hz_raw": result["itheta_hz_raw"],
        "itheta_hz_rounded": result["itheta_hz_rounded"],
        "frequency_to_use_hz": result["frequency_to_use_hz"],
        "theta_source": result["theta_source"],
        "n_feedback_epochs_total": result["reliability"]["n_feedback_epochs_total"],
        "n_epochs_retained": result["reliability"]["n_epochs_retained"],
        "usable_epoch_fraction": result["reliability"]["usable_epoch_fraction"],
        "split_half_diff_hz": result["reliability"]["split_half_diff_hz"],
        "bootstrap_ci_width_hz": result["reliability"]["bootstrap_ci_width_hz"],
        "peak_prominence_z": result["reliability"]["peak_prominence_z"],
        "edge_peak": result["reliability"]["edge_peak"],
        "theta_reliability_reason": result["decision"]["reason"],
    }
    pd.DataFrame([row]).to_csv(csv_path, index=False)


def _plot_channel_quality(
    result: Dict[str, Any],
    output_path: Path,
) -> None:
    metrics = result["qc"]["channel_metrics"]
    names = list(metrics.keys())
    stds = [metrics[name]["std_uv"] for name in names]
    line_noise = [metrics[name]["line_noise_ratio"] for name in names]
    bad_channels = set(result["preprocessing"]["bad_channels"])
    colors = ["tab:red" if name in bad_channels else "tab:blue" for name in names]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    axes[0].bar(names, stds, color=colors)
    axes[0].set_title("Raw channel quality summary")
    axes[0].set_ylabel("SD (uV)")
    axes[1].bar(names, line_noise, color=colors)
    axes[1].set_ylabel("Line-noise ratio")
    axes[1].set_xlabel("Channel")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_epoch_retention(result: Dict[str, Any], output_path: Path) -> None:
    retained = result["reliability"]["n_epochs_retained"]
    rejected = result["reliability"]["n_feedback_epochs_total"] - retained
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Retained", "Rejected"], [retained, rejected], color=["tab:green", "tab:orange"])
    ax.set_title("Retained vs rejected feedback epochs")
    ax.set_ylabel("Epoch count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_tfr_heatmap(debug: Dict[str, Any], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(
        debug["roi_tfr_db"],
        aspect="auto",
        origin="lower",
        extent=[
            float(debug["epoch_times_sec"][0]),
            float(debug["epoch_times_sec"][-1]),
            float(debug["frequencies_hz"][0]),
            float(debug["frequencies_hz"][-1]),
        ],
        cmap="viridis",
    )
    ax.set_title("Feedback-locked theta time-frequency power (frontocentral ROI)")
    ax.set_xlabel("Time from feedback onset (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(image, ax=ax, label="Power (dB)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_spectrum(result: Dict[str, Any], debug: Dict[str, Any], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(debug["frequencies_hz"], debug["theta_spectrum_db"], label="All feedback", linewidth=2.0)
    if np.any(debug["win_spectrum_db"]):
        ax.plot(debug["frequencies_hz"], debug["win_spectrum_db"], label="Win-only", alpha=0.8)
    if np.any(debug["loss_spectrum_db"]):
        ax.plot(debug["frequencies_hz"], debug["loss_spectrum_db"], label="Loss-only", alpha=0.8)
    peak = result["itheta_hz_raw"]
    if peak is not None:
        ax.axvline(peak, color="tab:red", linestyle="--", label=f"Peak {peak:.2f} Hz")
    ax.set_title("Theta spectrum averaged over +0.2 to +0.8 s")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_split_half(result: Dict[str, Any], debug: Dict[str, Any], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(debug["frequencies_hz"], debug["theta_spectrum_db"], label="Full sample", linewidth=2.0)
    peak1 = result["reliability"]["split_half_peak_1_hz"]
    peak2 = result["reliability"]["split_half_peak_2_hz"]
    if peak1 is not None:
        ax.axvline(peak1, color="tab:green", linestyle="--", label=f"Odd-half peak {peak1:.2f} Hz")
    if peak2 is not None:
        ax.axvline(peak2, color="tab:orange", linestyle="--", label=f"Even-half peak {peak2:.2f} Hz")
    ax.set_title("Split-half feedback theta peaks")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_bootstrap(debug: Dict[str, Any], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    peaks = debug["bootstrap_peaks_hz"]
    if len(peaks):
        ax.hist(peaks, bins=min(20, max(5, len(peaks) // 10)), color="tab:purple", alpha=0.8)
    else:
        ax.text(0.5, 0.5, "No bootstrap peaks available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Bootstrap peak histogram")
    ax.set_xlabel("Peak frequency (Hz)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_html_report(
    result: Dict[str, Any],
    artifacts: ThetaEstimateArtifacts,
    output_path: Path,
) -> None:
    rows = []
    for label, path in artifacts.plot_paths.items():
        rows.append(f"<h2>{label.replace('_', ' ').title()}</h2><img src='{Path(path).name}' width='900'>")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Feedback-Locked Theta QC Report</title>
</head>
<body>
  <h1>Participant-specific feedback theta QC</h1>
  <p><strong>Subject:</strong> {result['subject_id']}<br>
  <strong>Session:</strong> {result['session_id']}<br>
  <strong>Reliable:</strong> {result['reliable']}<br>
  <strong>Frequency to use:</strong> {result['frequency_to_use_hz']} Hz<br>
  <strong>Reason:</strong> {result['decision']['reason']}</p>
  {''.join(rows)}
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def write_theta_estimate_outputs(
    result: Dict[str, Any],
    debug: Dict[str, Any],
    output_dir: Path | str,
    *,
    write_html_report: bool = True,
) -> ThetaEstimateArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = f"sub-{result['subject_id']}_ses-{result['session_id']}_theta_estimate"

    json_path = output_dir / f"{basename}.json"
    csv_path = output_dir / f"{basename}.csv"
    plot_paths = {
        "raw_channel_quality_summary": str(output_dir / f"{basename}_channel_qc.png"),
        "retained_vs_rejected_epochs": str(output_dir / f"{basename}_epoch_retention.png"),
        "time_frequency_heatmap": str(output_dir / f"{basename}_roi_tfr.png"),
        "theta_spectrum": str(output_dir / f"{basename}_theta_spectrum.png"),
        "split_half_spectra": str(output_dir / f"{basename}_split_half.png"),
        "bootstrap_peak_histogram": str(output_dir / f"{basename}_bootstrap_histogram.png"),
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    _write_summary_csv(result, csv_path)
    _plot_channel_quality(result, Path(plot_paths["raw_channel_quality_summary"]))
    _plot_epoch_retention(result, Path(plot_paths["retained_vs_rejected_epochs"]))
    _plot_tfr_heatmap(debug, Path(plot_paths["time_frequency_heatmap"]))
    _plot_spectrum(result, debug, Path(plot_paths["theta_spectrum"]))
    _plot_split_half(result, debug, Path(plot_paths["split_half_spectra"]))
    _plot_bootstrap(debug, Path(plot_paths["bootstrap_peak_histogram"]))

    html_report_path = None
    if write_html_report:
        html_path = output_dir / f"{basename}_report.html"
        artifacts = ThetaEstimateArtifacts(
            json_path=str(json_path),
            csv_path=str(csv_path),
            plot_paths=plot_paths,
            html_report_path=str(html_path),
        )
        _write_html_report(result, artifacts, html_path)
        html_report_path = str(html_path)

    return ThetaEstimateArtifacts(
        json_path=str(json_path),
        csv_path=str(csv_path),
        plot_paths=plot_paths,
        html_report_path=html_report_path,
    )


def estimate_feedback_theta_from_files(
    *,
    subject_id: str,
    session_id: str,
    localizer_csv: Path | str,
    eeg_path: Path | str,
    config_path: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    localizer_run: str = "run-localizer",
) -> tuple[Dict[str, Any], ThetaEstimateArtifacts]:
    config = load_config(config_path)
    eeg_data = load_eeg_recording(eeg_path)
    behavior_df = load_behavioral_events(localizer_csv)
    result, debug = estimate_feedback_theta(
        eeg_data,
        behavior_df,
        config,
        subject_id=subject_id,
        session_id=session_id,
        localizer_run=localizer_run,
    )

    if output_dir is None:
        output_dir = _repo_root() / "data" / f"sub-{subject_id}" / "qc"
    artifacts = write_theta_estimate_outputs(
        result,
        debug,
        output_dir,
        write_html_report=bool(config.get("qc_outputs", {}).get("write_html_report", True)),
    )
    return result, artifacts
