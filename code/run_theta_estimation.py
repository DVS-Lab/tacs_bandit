#!/usr/bin/env python3
"""
Command-line entry point for participant-specific feedback theta estimation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from theta_estimator import estimate_feedback_theta_from_files


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _find_latest_file(base_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(base_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _auto_find_localizer_inputs(subject_id: str, session_id: str) -> tuple[Optional[Path], Optional[Path]]:
    subject_dir = _repo_root() / "data" / f"sub-{subject_id}"
    localizer_csv = _find_latest_file(
        subject_dir,
        f"**/*ses-{session_id}*run-localizer*task-bandit*.csv",
    ) or _find_latest_file(subject_dir, f"**/*ses-{session_id}*localizer*.csv")
    eeg_file = _find_latest_file(
        subject_dir,
        f"**/*ses-{session_id}*localizer*eeg.npz",
    ) or _find_latest_file(subject_dir, f"**/*ses-{session_id}*localizer*eeg.csv")
    return localizer_csv, eeg_file


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate participant-specific feedback theta from a localizer EEG run.",
    )
    parser.add_argument("--subject", required=True, help="Subject ID without the sub- prefix.")
    parser.add_argument("--session", required=True, help="Session ID without the ses- prefix.")
    parser.add_argument("--localizer-csv", help="Path to the localizer behavioral CSV.")
    parser.add_argument("--eeg", help="Path to the EEG recording file (.npz, .csv, .edf, .bdf, .xdf, .set).")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.json"))
    parser.add_argument("--out", help="Output directory for JSON/CSV/QC plots.")
    parser.add_argument(
        "--auto-find",
        action="store_true",
        help="Auto-find the most recent localizer CSV and EEG file for this subject/session.",
    )
    parser.add_argument(
        "--localizer-run",
        default="run-localizer",
        help="Run label to store in the output JSON.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    localizer_csv = Path(args.localizer_csv) if args.localizer_csv else None
    eeg_path = Path(args.eeg) if args.eeg else None

    if args.auto_find:
        auto_csv, auto_eeg = _auto_find_localizer_inputs(args.subject, args.session)
        localizer_csv = localizer_csv or auto_csv
        eeg_path = eeg_path or auto_eeg

    if localizer_csv is None or not localizer_csv.exists():
        raise FileNotFoundError("Could not locate the localizer behavioral CSV.")
    if eeg_path is None or not eeg_path.exists():
        raise FileNotFoundError("Could not locate the localizer EEG file.")

    output_dir = Path(args.out) if args.out else None
    result, artifacts = estimate_feedback_theta_from_files(
        subject_id=args.subject,
        session_id=args.session,
        localizer_csv=localizer_csv,
        eeg_path=eeg_path,
        config_path=args.config,
        output_dir=output_dir,
        localizer_run=args.localizer_run,
    )

    print(result["operator_summary"])
    print(f"Theta estimate JSON: {artifacts.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
