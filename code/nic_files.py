"""
nic_files.py — Locating NIC (Neuroelectrics) recordings on disk

The raw NIC directory accumulated several filename conventions over the course
of the study: `sub-10369_Run 1.easy`, `Bandit-10804_Run 2.easy`,
`10559 TACS_Run 1.easy`, and a few one-offs. Code that globs for only the first
convention silently finds nothing for the subjects recorded under the others,
which reads as "this subject has no EEG" rather than as an error.

This module is the single place that knows those conventions. Callers ask for a
subject's runs and get back whatever exists, regardless of how it was named.
"""

import glob
import re
from pathlib import Path
from typing import Dict, Optional

from config import EEG_DIR

# Ordered most- to least-common. A single subject's session may mix them.
FILENAME_PATTERNS = [
    'sub-{sid}_Run *.easy',
    'Bandit-{sid}_Run *.easy',
    '{sid} TACS_Run *.easy',
    'sub-{sid}_*_run-*_Run *.easy',
    '*_{sid}_Run *.easy',
]

_RUN_RE = re.compile(r'Run\s*(\d+)')


def parse_run_number(path: Path) -> Optional[int]:
    """Pull the run number out of a NIC filename, or None if absent."""
    match = _RUN_RE.search(path.stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def discover_runs(subject_id: str, eeg_dir: Optional[Path] = None) -> Dict[int, Dict]:
    """
    Find NIC recordings for one subject across all known naming conventions.

    Where a run has several files (restarts), the largest is kept — a restarted
    run leaves a short truncated file behind and the full recording is the one
    that matters.

    Returns
    -------
    dict
        {run_number: {'easy': Path, 'info': Path or None, 'pattern': str}}
    """
    if eeg_dir is None:
        eeg_dir = EEG_DIR

    runs: Dict[int, Dict] = {}

    for pattern in FILENAME_PATTERNS:
        for match in glob.glob(str(eeg_dir / ('*' + pattern.format(sid=subject_id)))):
            easy_path = Path(match)
            run_num = parse_run_number(easy_path)
            if run_num is None:
                continue

            size = easy_path.stat().st_size
            if run_num in runs and runs[run_num]['size'] >= size:
                continue

            info_path = easy_path.with_suffix('.info')
            runs[run_num] = {
                'easy': easy_path,
                'info': info_path if info_path.exists() else None,
                'pattern': pattern,
                'size': size,
            }

    for run in runs.values():
        del run['size']

    return runs


def find_run(
    subject_id: str, run_num: int, eeg_dir: Optional[Path] = None
) -> Optional[Path]:
    """Path to a single run's .easy file, or None if that run was not recorded."""
    run = discover_runs(subject_id, eeg_dir).get(run_num)
    return run['easy'] if run else None
