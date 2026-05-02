import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
import sys

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from select_theta_frequency import select_stimulation_frequency
from theta_estimator import EEGData, estimate_feedback_theta, load_config

try:
    from bandit_main import IndividualizedThetaBanditTask

    PYGAME_TASKS_AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name == "pygame":
        IndividualizedThetaBanditTask = None
        PYGAME_TASKS_AVAILABLE = False
    else:
        raise


def make_test_config(tmp_dir: Path):
    config = load_config(CODE_DIR / "config.json")
    config = copy.deepcopy(config)
    config["paths"]["data_dir"] = str(tmp_dir / "data")
    config["stimulation"]["test_mode"] = True
    config["localizer_fast_theta"]["target_trials"] = 8
    config["localizer_fast_theta"]["duration_minutes"] = 0.02
    config["localizer_fast_theta"]["hard_min_usable_epochs"] = 6
    config["localizer_fast_theta"]["recommended_min_usable_epochs"] = 8
    config["theta_estimation"]["bootstrap_iterations"] = 60
    config["theta_estimation"]["max_bootstrap_ci_width_hz"] = 2.0
    config["theta_estimation"]["min_peak_prominence_z"] = 0.3
    config["theta_estimation"]["frequency_step_hz"] = 0.25
    config["theta_estimation"]["smooth_spectrum_hz"] = 0.5
    config["theta_estimation"]["min_usable_epoch_fraction"] = 0.6
    config["theta_estimation"]["recommended_usable_epoch_fraction"] = 0.7
    config["eeg_recording"]["record_lsl_eeg_during_localizer"] = False
    return config


def make_synthetic_feedback_localizer(
    *,
    n_epochs=90,
    base_freq_hz=6.5,
    second_half_freq_hz=None,
    amplitude_uv=30.0,
    noise_uv=8.0,
    edge_freq_hz=None,
):
    rng = np.random.default_rng(7)
    sampling_rate_hz = 250.0
    channel_names = ["Fz", "FCz", "Cz", "F3", "F4", "Pz", "O1", "O2"]
    epoch_tmin = -1.0
    epoch_tmax = 1.5
    spacing_sec = 3.0
    spacing_samples = int(spacing_sec * sampling_rate_hz)
    n_samples = spacing_samples * (n_epochs + 3)
    timestamps = np.arange(n_samples, dtype=float) / sampling_rate_hz
    samples = rng.normal(0.0, noise_uv, size=(n_samples, len(channel_names)))

    event_times = []
    roi_indices = [0, 1, 2, 3, 4]
    epoch_time = np.arange(int((epoch_tmax - epoch_tmin) * sampling_rate_hz), dtype=float) / sampling_rate_hz + epoch_tmin
    burst_mask = (epoch_time >= 0.2) & (epoch_time <= 0.8)
    window = np.sin(np.linspace(0, np.pi, np.sum(burst_mask)))

    for epoch_index in range(n_epochs):
        center = spacing_samples * (epoch_index + 1)
        event_times.append(timestamps[center])
        freq_hz = edge_freq_hz or base_freq_hz
        if second_half_freq_hz is not None and epoch_index >= n_epochs // 2:
            freq_hz = second_half_freq_hz
        burst = np.sin(2 * np.pi * freq_hz * epoch_time[burst_mask]) * window * amplitude_uv
        start = center + int(epoch_tmin * sampling_rate_hz)
        stop = start + len(epoch_time)
        if 0 <= start and stop <= n_samples:
            for channel_index in roi_indices:
                samples[start:stop, channel_index][burst_mask] += burst

    behavior = pd.DataFrame(
        {
            "feedback_marker": np.where(np.arange(n_epochs) % 2 == 0, 31, 32),
            "feedback_onset_lsl_time": event_times,
            "lsl_marker_send_time": event_times,
            "run_start_lsl_time": event_times[0] - 2.0,
            "feedback_onset_task_time": np.asarray(event_times) - (event_times[0] - 2.0),
        }
    )
    eeg_data = EEGData(samples=samples, timestamps=timestamps, channel_names=channel_names, sampling_rate_hz=sampling_rate_hz, metadata={})
    return eeg_data, behavior


class ThetaEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.config = make_test_config(self.tmp_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_reliable_6_5_hz_theta(self):
        eeg_data, behavior = make_synthetic_feedback_localizer(n_epochs=96, base_freq_hz=6.5)
        result, _ = estimate_feedback_theta(
            eeg_data,
            behavior,
            self.config,
            subject_id="001",
            session_id="001",
        )
        self.assertTrue(result["reliable"])
        self.assertEqual(result["itheta_hz_rounded"], 6.5)
        self.assertEqual(result["frequency_to_use_hz"], 6.5)

    def test_noisy_no_peak_falls_back_to_fixed_6hz(self):
        eeg_data, behavior = make_synthetic_feedback_localizer(n_epochs=96, amplitude_uv=0.0, noise_uv=20.0)
        result, _ = estimate_feedback_theta(
            eeg_data,
            behavior,
            self.config,
            subject_id="001",
            session_id="001",
        )
        self.assertFalse(result["reliable"])
        self.assertEqual(result["frequency_to_use_hz"], 6.0)
        self.assertEqual(result["theta_source"], "fallback_fixed_6hz")

    def test_edge_peak_is_unreliable(self):
        eeg_data, behavior = make_synthetic_feedback_localizer(n_epochs=96, edge_freq_hz=4.0)
        result, _ = estimate_feedback_theta(
            eeg_data,
            behavior,
            self.config,
            subject_id="001",
            session_id="001",
        )
        self.assertFalse(result["reliable"])
        self.assertEqual(result["theta_source"], "fallback_fixed_6hz")

    def test_split_half_disagreement_falls_back(self):
        eeg_data, behavior = make_synthetic_feedback_localizer(
            n_epochs=96,
            base_freq_hz=5.0,
            second_half_freq_hz=7.5,
        )
        result, _ = estimate_feedback_theta(
            eeg_data,
            behavior,
            self.config,
            subject_id="001",
            session_id="001",
        )
        self.assertFalse(result["reliable"])
        self.assertEqual(result["frequency_to_use_hz"], 6.0)

    def test_too_few_epochs_is_unreliable(self):
        eeg_data, behavior = make_synthetic_feedback_localizer(n_epochs=5, base_freq_hz=6.5)
        result, _ = estimate_feedback_theta(
            eeg_data,
            behavior,
            self.config,
            subject_id="001",
            session_id="001",
        )
        self.assertFalse(result["reliable"])
        self.assertEqual(result["theta_source"], "fallback_fixed_6hz")


class TaskWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.config = make_test_config(self.tmp_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    @unittest.skipUnless(PYGAME_TASKS_AVAILABLE, "pygame is not installed in this environment")
    def test_localizer_mode_runs_without_hardware(self):
        args = SimpleNamespace(
            config=None,
            mode="LOCALIZER_FAST_THETA",
            subject="001",
            session="001",
            run=1,
            frequency=None,
            age="",
            gender="",
            test_mode=True,
            auto_respond=True,
            duration_minutes=0.01,
        )
        task = IndividualizedThetaBanditTask(config=self.config, cli_args=args)
        task.run()

        data_dir = Path(self.config["paths"]["data_dir"]) / "sub-001"
        csv_files = list(data_dir.glob("*run-localizer*task-bandit*.csv"))
        marker_logs = list((data_dir / "logs").glob("*run-localizer*_markers.jsonl"))
        self.assertTrue(csv_files, "Expected a localizer behavioral CSV to be written.")
        self.assertTrue(marker_logs, "Expected a marker log to be written.")

    def test_itheta_selection_reads_json_and_fallbacks(self):
        data_dir = Path(self.config["paths"]["data_dir"]) / "sub-001" / "qc"
        data_dir.mkdir(parents=True, exist_ok=True)
        theta_json = data_dir / "sub-001_ses-001_theta_estimate.json"
        theta_json.write_text(
            json.dumps(
                {
                    "reliable": True,
                    "frequency_to_use_hz": 6.5,
                    "theta_source": "reliable_itheta",
                    "decision": {"reason": "Reliable feedback-locked theta estimate."},
                }
            ),
            encoding="utf-8",
        )

        decision = select_stimulation_frequency("001", "001", self.config, search_root=Path(self.config["paths"]["data_dir"]))
        self.assertEqual(decision.frequency_to_use_hz, 6.5)
        self.assertEqual(decision.theta_source, "reliable_itheta")

        theta_json.write_text(
            json.dumps(
                {
                    "reliable": False,
                    "frequency_to_use_hz": 6.0,
                    "theta_source": "fallback_fixed_6hz",
                    "decision": {"reason": "Unreliable estimate."},
                }
            ),
            encoding="utf-8",
        )
        decision = select_stimulation_frequency("001", "001", self.config, search_root=Path(self.config["paths"]["data_dir"]))
        self.assertEqual(decision.frequency_to_use_hz, 6.0)
        self.assertEqual(decision.theta_source, "fallback_fixed_6hz")

        theta_json.unlink()
        decision = select_stimulation_frequency("001", "001", self.config, search_root=Path(self.config["paths"]["data_dir"]))
        self.assertEqual(decision.frequency_to_use_hz, 6.0)
        self.assertEqual(decision.theta_source, "fallback_fixed_6hz")


if __name__ == "__main__":
    unittest.main()
