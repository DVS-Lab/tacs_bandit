"""
Data loading and preprocessing for hierarchical RL model fitting.

Reads bandit task CSVs (from bandit_main.py output), remaps choice encoding
from 1/2 → 0/1, filters missed trials, and pads/stacks sequences for
vectorized JAX computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import jax.numpy as jnp


REQUIRED_COLUMNS = {"choice", "reward", "trial_num", "run"}
OPTIONAL_COLUMNS = {"stim_condition", "current_good", "correct", "rt", "subject_id"}

# Condition label → integer mapping
DEFAULT_CONDITION_MAP = {
    "baseline": 0,
    "active": 1,
    "sham": 2,
}


@dataclass
class SubjectData:
    """Preprocessed trial data for a single subject."""

    subject_id: str
    choices: np.ndarray  # (n_trials,) int, 0 or 1
    rewards: np.ndarray  # (n_trials,) float, 0.0 or 1.0
    n_trials: int
    run_ids: np.ndarray  # (n_trials,) int
    conditions: np.ndarray  # (n_trials,) int — mapped from stim_condition
    correct: np.ndarray  # (n_trials,) bool
    rts: np.ndarray  # (n_trials,) float, in ms
    raw_df: pd.DataFrame  # original filtered DataFrame

    @property
    def accuracy(self) -> float:
        return float(np.mean(self.correct))

    @property
    def reward_rate(self) -> float:
        return float(np.mean(self.rewards))


@dataclass
class BanditDataset:
    """Collection of subject data, padded for JAX vectorization."""

    subjects: list[SubjectData]
    choices: jnp.ndarray  # (total_trials,) int
    rewards: jnp.ndarray  # (total_trials,) float
    subject_ids: jnp.ndarray  # (total_trials,) int — subject index
    condition_ids: jnp.ndarray  # (total_trials,) int
    n_subjects: int
    n_conditions: int
    subject_id_map: dict[str, int]  # original ID → integer index
    condition_map: dict[str, int]

    def summary(self) -> str:
        lines = [
            f"BanditDataset: {self.n_subjects} subjects, "
            f"{len(self.choices)} total trials, "
            f"{self.n_conditions} conditions",
            "",
        ]
        for subj in self.subjects:
            lines.append(
                f"  {subj.subject_id}: {subj.n_trials} trials, "
                f"acc={subj.accuracy:.1%}, reward={subj.reward_rate:.1%}"
            )
        return "\n".join(lines)


def load_subject_csv(
    filepath: Path,
    *,
    condition_map: Optional[dict[str, int]] = None,
    subject_id_override: Optional[str] = None,
) -> SubjectData:
    """Load and preprocess a single subject's bandit CSV.

    Parameters
    ----------
    filepath : Path to the CSV file.
    condition_map : mapping from stim_condition strings to integers.
    subject_id_override : if provided, use this instead of inferring from filename.

    Returns
    -------
    SubjectData with choices remapped to 0/1 and missed trials removed.
    """
    if condition_map is None:
        condition_map = DEFAULT_CONDITION_MAP

    df = pd.read_csv(filepath)

    # Validate required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {filepath}: {missing}")

    # Infer subject ID
    if subject_id_override:
        subject_id = subject_id_override
    elif "subject_id" in df.columns:
        subject_id = str(df["subject_id"].iloc[0])
    else:
        # Parse from filename: sub-{id}_ses-{ses}_...
        name = filepath.stem
        parts = name.split("_")
        subject_id = next(
            (p.replace("sub-", "") for p in parts if p.startswith("sub-")),
            filepath.stem,
        )

    # Filter out missed trials (choice is None/NaN)
    df = df.dropna(subset=["choice"]).copy()
    df["choice"] = df["choice"].astype(int)

    # Remove any invalid choices (not 1 or 2)
    df = df[df["choice"].isin([1, 2])].copy()

    if len(df) == 0:
        raise ValueError(f"No valid trials in {filepath}")

    # Remap choice: 1,2 → 0,1
    choices = df["choice"].values - 1

    # Reward: ensure boolean → float
    rewards = df["reward"].astype(float).values

    # Run IDs
    run_ids = df["run"].astype(int).values

    # Condition mapping
    if "stim_condition" in df.columns:
        conditions = df["stim_condition"].map(condition_map).fillna(-1).astype(int).values
    else:
        conditions = np.zeros(len(df), dtype=int)

    # Correct
    if "correct" in df.columns:
        correct = df["correct"].astype(bool).values
    else:
        correct = np.zeros(len(df), dtype=bool)

    # RT
    if "rt" in df.columns:
        rts = df["rt"].astype(float).values
    else:
        rts = np.full(len(df), np.nan)

    return SubjectData(
        subject_id=subject_id,
        choices=choices,
        rewards=rewards,
        n_trials=len(choices),
        run_ids=run_ids,
        conditions=conditions,
        correct=correct,
        rts=rts,
        raw_df=df,
    )


def load_dataset(
    csv_paths: Sequence[Path],
    *,
    condition_map: Optional[dict[str, int]] = None,
    runs: Optional[Sequence[int]] = None,
    conditions: Optional[Sequence[str]] = None,
) -> BanditDataset:
    """Load multiple subject CSVs into a BanditDataset.

    Parameters
    ----------
    csv_paths : paths to individual subject CSV files.
    condition_map : stim_condition string → int mapping.
    runs : if provided, only include trials from these run numbers.
    conditions : if provided, only include trials with these stim_condition values.

    Returns
    -------
    BanditDataset ready for model fitting.
    """
    if condition_map is None:
        condition_map = DEFAULT_CONDITION_MAP

    subjects: list[SubjectData] = []
    for path in csv_paths:
        subj = load_subject_csv(Path(path), condition_map=condition_map)

        # Filter runs if requested
        if runs is not None:
            mask = np.isin(subj.run_ids, runs)
            subj = _filter_subject(subj, mask)

        # Filter conditions if requested
        if conditions is not None:
            cond_ints = [condition_map[c] for c in conditions if c in condition_map]
            mask = np.isin(subj.conditions, cond_ints)
            subj = _filter_subject(subj, mask)

        if subj.n_trials > 0:
            subjects.append(subj)

    if not subjects:
        raise ValueError("No valid subjects/trials after filtering.")

    # Assign integer subject indices
    subject_id_map = {s.subject_id: i for i, s in enumerate(subjects)}

    # Concatenate all trials
    all_choices = np.concatenate([s.choices for s in subjects])
    all_rewards = np.concatenate([s.rewards for s in subjects])
    all_subject_ids = np.concatenate(
        [np.full(s.n_trials, subject_id_map[s.subject_id], dtype=int) for s in subjects]
    )
    all_conditions = np.concatenate([s.conditions for s in subjects])

    n_conditions = len(set(all_conditions[all_conditions >= 0]))

    return BanditDataset(
        subjects=subjects,
        choices=jnp.array(all_choices, dtype=jnp.int32),
        rewards=jnp.array(all_rewards, dtype=jnp.float32),
        subject_ids=jnp.array(all_subject_ids, dtype=jnp.int32),
        condition_ids=jnp.array(all_conditions, dtype=jnp.int32),
        n_subjects=len(subjects),
        n_conditions=max(n_conditions, 1),
        subject_id_map=subject_id_map,
        condition_map=condition_map,
    )


def load_dataset_from_directory(
    data_dir: Path,
    *,
    glob_pattern: str = "**/sub-*_ses-*_task-bandit_*.csv",
    **kwargs,
) -> BanditDataset:
    """Convenience: find all bandit CSVs under a directory and load them."""
    paths = sorted(Path(data_dir).glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No bandit CSVs found in {data_dir} with pattern {glob_pattern}")
    return load_dataset(paths, **kwargs)


def _filter_subject(subj: SubjectData, mask: np.ndarray) -> SubjectData:
    """Return a new SubjectData with only the trials where mask is True."""
    return SubjectData(
        subject_id=subj.subject_id,
        choices=subj.choices[mask],
        rewards=subj.rewards[mask],
        n_trials=int(mask.sum()),
        run_ids=subj.run_ids[mask],
        conditions=subj.conditions[mask],
        correct=subj.correct[mask],
        rts=subj.rts[mask],
        raw_df=subj.raw_df.iloc[mask].copy(),
    )
