"""
Helpers for selecting the stimulation frequency for individualized theta runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_THETA_ESTIMATE_GLOB = "*theta_estimate*.json"


@dataclass
class ThetaSelectionDecision:
    subject_id: str
    session_id: str
    frequency_to_use_hz: float
    theta_source: str
    reliable: bool
    reason: str
    theta_estimate_file: Optional[str]
    intended_protocol_label: str
    manual_override_used: bool = False
    manual_override_reason: Optional[str] = None

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "frequency_to_use_hz": self.frequency_to_use_hz,
            "theta_source": self.theta_source,
            "theta_reliable": self.reliable,
            "theta_reliability_reason": self.reason,
            "theta_estimate_file": self.theta_estimate_file or "",
            "protocol_label_to_show": self.intended_protocol_label,
            "manual_override_used": self.manual_override_used,
            "manual_override_reason": self.manual_override_reason or "",
        }


def _as_float_range(config: Dict[str, Any], key: str, default: Iterable[float]) -> tuple[float, float]:
    values = config.get(key, default)
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return float(default[0]), float(default[1])
    return float(values[0]), float(values[1])


def _clamp(value: float, valid_range: tuple[float, float]) -> float:
    return max(valid_range[0], min(valid_range[1], value))


def _build_protocol_label(frequency_hz: float, theta_source: str) -> str:
    if theta_source == "reliable_itheta":
        return f"ITHETA_TACS_{frequency_hz:.1f}Hz"
    if theta_source == "fallback_fixed_6hz":
        return "FIXED_THETA_TACS_6.0Hz_FALLBACK"
    if theta_source == "fixed_6hz":
        return "FIXED_THETA_TACS_6.0Hz"
    return f"THETA_TACS_{frequency_hz:.1f}Hz"


def find_theta_estimate_files(
    subject_id: str,
    session_id: Optional[str] = None,
    search_root: Optional[Path] = None,
) -> list[Path]:
    if search_root is None:
        search_root = Path(__file__).resolve().parent.parent / "data"

    patterns = []
    subject_dir = search_root / f"sub-{subject_id}"
    if session_id:
        patterns.append(subject_dir.glob(f"**/*ses-{session_id}*theta_estimate*.json"))
    patterns.append(subject_dir.glob(f"**/{DEFAULT_THETA_ESTIMATE_GLOB}"))

    matches: list[Path] = []
    for iterator in patterns:
        for path in iterator:
            if path.is_file() and path not in matches:
                matches.append(path)

    matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return matches


def load_theta_estimate(theta_estimate_file: Path | str) -> Dict[str, Any]:
    path = Path(theta_estimate_file)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_stimulation_frequency(
    subject_id: str,
    session_id: str,
    config: Dict[str, Any],
    *,
    theta_estimate_file: Optional[Path | str] = None,
    search_root: Optional[Path] = None,
    manual_override_hz: Optional[float] = None,
    manual_override_reason: Optional[str] = None,
) -> ThetaSelectionDecision:
    selection_config = config.get("stimulation_frequency_selection", {})
    theta_config = config.get("theta_estimation", {})

    valid_range = _as_float_range(
        selection_config,
        "valid_theta_range_hz",
        theta_config.get("primary_theta_band_hz", (4.0, 8.0)),
    )
    fallback_hz = float(
        selection_config.get(
            "default_fixed_theta_hz",
            theta_config.get("default_fixed_theta_hz", 6.0),
        )
    )

    if manual_override_hz is not None:
        if not selection_config.get("manual_override_allowed", True):
            raise ValueError("Manual override is disabled in config.")
        if selection_config.get("manual_override_requires_reason", True) and not manual_override_reason:
            raise ValueError("Manual override requires a reason.")
        if not (valid_range[0] <= manual_override_hz <= valid_range[1]):
            raise ValueError(
                f"Manual override {manual_override_hz:.2f} Hz is outside the valid theta range "
                f"{valid_range[0]:.1f}-{valid_range[1]:.1f} Hz."
            )
        manual_override_hz = _clamp(float(manual_override_hz), valid_range)
        return ThetaSelectionDecision(
            subject_id=subject_id,
            session_id=session_id,
            frequency_to_use_hz=manual_override_hz,
            theta_source="manual_override",
            reliable=False,
            reason=f"Manual override applied: {manual_override_reason}",
            theta_estimate_file=str(theta_estimate_file) if theta_estimate_file else None,
            intended_protocol_label=f"MANUAL_THETA_TACS_{manual_override_hz:.1f}Hz",
            manual_override_used=True,
            manual_override_reason=manual_override_reason,
        )

    if theta_estimate_file is None:
        matches = find_theta_estimate_files(subject_id, session_id, search_root)
        theta_estimate_file = matches[0] if matches else None

    if theta_estimate_file is None:
        if selection_config.get("stop_if_no_theta_file", False):
            raise FileNotFoundError(
                f"No theta estimate file found for sub-{subject_id}, ses-{session_id}."
            )
        return ThetaSelectionDecision(
            subject_id=subject_id,
            session_id=session_id,
            frequency_to_use_hz=fallback_hz,
            theta_source="fallback_fixed_6hz",
            reliable=False,
            reason="No theta estimate file found. Using fixed 6.0 Hz fallback.",
            theta_estimate_file=None,
            intended_protocol_label=_build_protocol_label(fallback_hz, "fallback_fixed_6hz"),
        )

    theta_estimate = load_theta_estimate(theta_estimate_file)
    reliable = bool(theta_estimate.get("reliable", False))
    decision_reason = (
        theta_estimate.get("decision", {}).get("reason")
        or theta_estimate.get("theta_reliability_reason")
        or "Theta selection loaded from estimate file."
    )

    file_frequency = theta_estimate.get("frequency_to_use_hz")
    rounded_frequency = theta_estimate.get("itheta_hz_rounded")
    raw_frequency = theta_estimate.get("itheta_hz_raw")

    candidate_frequency = file_frequency or rounded_frequency or raw_frequency
    if candidate_frequency is not None:
        candidate_frequency = float(candidate_frequency)

    if reliable and selection_config.get("use_itheta_if_reliable", True) and candidate_frequency is not None:
        if valid_range[0] <= candidate_frequency <= valid_range[1]:
            source = theta_estimate.get("theta_source", "reliable_itheta")
            if source not in {"reliable_itheta", "reliable_iTheta", "reliable_itheta".lower()}:
                source = "reliable_itheta"
            return ThetaSelectionDecision(
                subject_id=subject_id,
                session_id=session_id,
                frequency_to_use_hz=_clamp(candidate_frequency, valid_range),
                theta_source="reliable_itheta",
                reliable=True,
                reason=decision_reason,
                theta_estimate_file=str(theta_estimate_file),
                intended_protocol_label=_build_protocol_label(candidate_frequency, "reliable_itheta"),
            )
        reliable = False
        decision_reason = (
            f"Theta estimate file suggested {candidate_frequency:.2f} Hz, which falls outside "
            f"the valid theta range."
        )

    fallback_policy = selection_config.get("fallback_if_unreliable", "fixed_6hz")
    if fallback_policy == "fixed_6hz":
        return ThetaSelectionDecision(
            subject_id=subject_id,
            session_id=session_id,
            frequency_to_use_hz=fallback_hz,
            theta_source="fallback_fixed_6hz",
            reliable=False,
            reason=decision_reason,
            theta_estimate_file=str(theta_estimate_file),
            intended_protocol_label=_build_protocol_label(fallback_hz, "fallback_fixed_6hz"),
        )

    raise ValueError(
        "Theta estimate was unreliable and the config does not allow fixed-frequency fallback."
    )
