"""
run_theta_metrics.py — Extract subject-level theta reactivity for the master CSV

Thin driver around eeg_theta.run_theta_analysis(). That pipeline computes theta
power across the baseline runs, applies the QC thresholds, and averages to one
row per subject; this script just runs it headless and saves the subject-level
table where build_master_data.py can pick it up.

Coverage here depends on nic_files discovery. Before that was shared, the theta
pipeline globbed only for `sub-{id}_Run *`, so subjects recorded under the
other NIC naming conventions silently produced no theta at all.

Usage
-----
    python run_theta_metrics.py
"""

import sys
from pathlib import Path

from config import REPO_ROOT
from eeg_theta import run_theta_analysis

OUTPUT_DIR = REPO_ROOT / 'derivatives' / 'eeg'
OUTPUT_PATH = OUTPUT_DIR / 'theta_subject_metrics.csv'
RUNS_PATH = OUTPUT_DIR / 'theta_run_metrics.csv'


def main() -> int:
    results = run_theta_analysis(verbose=True, show_plots=False)

    theta_subject = results.get('theta_subject')
    if theta_subject is None or len(theta_subject) == 0:
        print('No theta data produced; nothing written.')
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    theta_subject.to_csv(OUTPUT_PATH, index=False)
    print(f'\nWrote {OUTPUT_PATH}  ({len(theta_subject)} subjects)')

    theta_all = results.get('theta_all')
    if theta_all is not None and len(theta_all) > 0:
        theta_all.to_csv(RUNS_PATH, index=False)
        print(f'Wrote {RUNS_PATH}  ({len(theta_all)} runs)')

    return 0


if __name__ == '__main__':
    sys.exit(main())
