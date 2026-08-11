"""
build_master_data.py — Regenerate master_subject_data.csv end to end

cognitive_merge.run_cognitive_merge() only populates the behavioral columns
when the behavioral DataFrames are handed to it. Called bare, it silently
produces a CSV with the questionnaire and cognitive columns but none of the
WSLS, reinforcement-learning, or accuracy measures — the columns are absent
rather than NaN, so downstream code fails on KeyError rather than on missing
data. This script computes those frames and passes them in.

Order matters: condition labels come from the counterbalance in SUBJECT_INFO,
so every measure split by condition (WSLS, R-W, accuracy) inherits whatever
that registry says. Run stim_verification.py first if the counterbalance
assignments have not been checked against the EEG.

Usage
-----
    python build_master_data.py                  # all registered subjects
    python build_master_data.py --sample dissertation
    python build_master_data.py --no-write       # report coverage, write nothing
"""

import argparse
import sys
from typing import Optional, List

import pandas as pd

from data_loading import load_all_subjects
from exclusions import apply_all_exclusions
from wsls import compute_wsls_h1_h2
from rescorla_wagner import fit_rw_by_condition
from accuracy_analysis import compute_condition_level_accuracy
from cognitive_merge import run_cognitive_merge


def build(
    sample: str = 'all',
    export_csv: bool = True,
    include_exploratory: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Compute the behavioral frames and merge them into the master CSV.

    Returns the dict from run_cognitive_merge(), with the intermediate
    behavioral frames added under their own keys for inspection.
    """
    print('=' * 70)
    print(f'Loading trial-level data (sample={sample!r})')
    print('=' * 70)
    data = load_all_subjects(sample=sample, verbose=False)
    print(f'  {data["subject_id"].nunique()} subjects, {len(data)} trials')

    print('\n' + '=' * 70)
    print('Applying pre-registered exclusions')
    print('=' * 70)
    exclusions = apply_all_exclusions(data, verbose=verbose)
    data_h1 = exclusions['data_h1']
    data_h2 = exclusions['data_h2']
    data_clean = exclusions['data_clean']
    h2_eligible = [str(s) for s in exclusions['h2_eligible']]

    print('\n' + '=' * 70)
    print('Computing behavioral measures')
    print('=' * 70)

    wsls_h1, wsls_h2 = compute_wsls_h1_h2(data_h1, data_h2)
    print(f'  WSLS: {len(wsls_h1)} subjects (H1), {len(wsls_h2)} subject-conditions (H2)')

    # R-W is fit on the behaviorally-clean data across both conditions;
    # build_subject_df restricts the active-condition fits to h2_eligible.
    rw_mle = fit_rw_by_condition(data_clean, method='mle', verbose=False)
    print(f'  R-W (MLE): {len(rw_mle)} subject-condition fits')

    accuracy = compute_condition_level_accuracy(data_clean, verbose=False)
    print(f'  Accuracy: {len(accuracy)} subject-conditions')

    print('\n' + '=' * 70)
    print('Merging into master subject data')
    print('=' * 70)
    results = run_cognitive_merge(
        wsls_h1=wsls_h1,
        wsls_h2=wsls_h2,
        rw_mle=rw_mle,
        accuracy=accuracy,
        h2_subjects=h2_eligible,
        include_exploratory=include_exploratory,
        export_csv=export_csv,
        verbose=verbose,
    )

    results.update({
        'data': data,
        'exclusions': exclusions,
        'wsls_h1': wsls_h1,
        'wsls_h2': wsls_h2,
        'rw_mle': rw_mle,
        'accuracy': accuracy,
    })
    return results


def report_coverage(subj_df: pd.DataFrame) -> None:
    """Print non-null counts for the behavioral columns, which are the ones
    that silently go missing when the frames are not passed through."""
    groups = {
        'WSLS': ['sham_p_stay_win', 'sham_p_shift_lose',
                 'active_p_stay_win', 'active_p_shift_lose',
                 'delta_p_stay_win', 'delta_p_shift_lose'],
        'R-W': ['sham_alpha', 'sham_beta', 'active_alpha', 'active_beta',
                'delta_alpha', 'delta_beta'],
        'Accuracy': ['sham_accuracy', 'sham_win_rate', 'active_accuracy',
                     'active_win_rate', 'delta_accuracy', 'delta_win_rate'],
    }
    n = len(subj_df)
    print('\n' + '=' * 70)
    print(f'Behavioral column coverage ({n} subjects)')
    print('=' * 70)
    for label, cols in groups.items():
        print(f'\n{label}:')
        for col in cols:
            if col in subj_df.columns:
                print(f'  {col:22s}: {subj_df[col].notna().sum():3d}/{n}')
            else:
                print(f'  {col:22s}: MISSING')


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--sample', default='all',
                        choices=['all', 'dissertation', 'new'])
    parser.add_argument('--no-write', action='store_true',
                        help='compute and report without writing the CSV')
    parser.add_argument('--no-exploratory', action='store_true',
                        help='skip the RF1 exploratory measures')
    args = parser.parse_args(argv)

    results = build(
        sample=args.sample,
        export_csv=not args.no_write,
        include_exploratory=not args.no_exploratory,
    )
    report_coverage(results['subj_df'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
