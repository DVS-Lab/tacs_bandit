"""
audit_sample.py — Where every subject and every variable is lost, and why

Different analyses in this project run on different Ns, which is legitimate but
easy to lose track of. This produces one accounting: for each subject, what raw
data exists; for each analysis variable, how many subjects have it and which are
missing; and for each missing cell, which of four reasons applies.

The four reasons matter because only one of them is fixable:

  by_design  the measure was never intended for this subject (e.g. tests
             administered only to participants aged 40+)
  excluded   the subject has data but fails a pre-registered criterion, or is
             not eligible for the contrast the variable describes
  absent     the measure was intended but no value exists in any source
  technical  a value exists upstream but does not reach the analysis — a
             preprocessing failure, a wrong lookup, a naming mismatch

`technical` is the category worth hunting. Several have already been found and
fixed in this project (a wrong REDCap event emptying the loneliness columns; a
missing high-pass filter that made 15 subjects look like they had no alpha
rhythm; an EEG file glob that saw only one of four naming conventions). Each
looked exactly like `absent` from downstream.

Usage
-----
    python audit_sample.py
    python audit_sample.py --sample dissertation
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from config import (
    DATA_DIR, DISSERTATION_SUBJECTS, REPO_ROOT, SUBJECT_INFO,
)
from nic_files import discover_runs

OUTPUT_DIR = REPO_ROOT / 'derivatives'

# Measures administered only to a subset by protocol. Missing values here are
# by design, not loss.
BY_DESIGN = {
    'digit_span_total': 'administered to participants aged 40+ only',
    'bvmt_total': 'administered to participants aged 40+ only',
    'trails_a_time': 'administered to participants aged 40+ only',
    'trails_b_time': 'administered to participants aged 40+ only',
    'bbs_avg': 'instrument added partway through data collection',
}

# Variables describing the active condition are defined only for subjects
# eligible for the active-vs-sham contrast.
H2_ONLY_PREFIXES = ('active_', 'delta_')


def raw_availability() -> pd.DataFrame:
    """One row per registered subject: what raw data exists on disk."""
    rows = []
    for sid in SUBJECT_INFO:
        beh_files = glob.glob(str(DATA_DIR / f'sub-{sid}' / '*.csv'))
        beh_runs = {int(Path(f).name.split('run-')[1][:2])
                    for f in beh_files if 'run-' in Path(f).name}
        eeg = discover_runs(sid)
        rows.append({
            'subject_id': sid,
            'in_dissertation': sid in DISSERTATION_SUBJECTS,
            'counterbalance': SUBJECT_INFO[sid].get('counterbalance', '?'),
            'behavioral_runs': len(beh_runs),
            'eeg_runs': len(eeg),
            'has_behavioral': len(beh_runs) > 0,
            'has_eeg': len(eeg) > 0,
        })
    return pd.DataFrame(rows)


def sample_attrition(sample: str = 'all') -> List[Dict]:
    """Subject counts at each stage from registry to analysis sample."""
    from data_loading import load_all_subjects
    from exclusions import apply_all_exclusions

    if sample == 'dissertation':
        registered = set(DISSERTATION_SUBJECTS)
    elif sample == 'new':
        registered = set(SUBJECT_INFO) - set(DISSERTATION_SUBJECTS)
    else:
        registered = set(SUBJECT_INFO)

    avail = raw_availability().set_index('subject_id')
    has_beh = {s for s in registered if avail.loc[s, 'has_behavioral']}

    raw = load_all_subjects(sample=sample, verbose=False)
    results = apply_all_exclusions(raw, verbose=False)
    loaded = set(raw['subject_id'].unique())
    clean = set(results['data_clean']['subject_id'].unique())
    h2 = {str(s) for s in results['h2_eligible']}

    stages = [
        ('registered in SUBJECT_INFO', registered),
        ('behavioral files on disk', has_beh),
        ('loaded by the pipeline', loaded),
        ('survive behavioral exclusions', clean),
        ('eligible for active-vs-sham', h2),
    ]

    out, previous = [], None
    for name, subjects in stages:
        lost = sorted(previous - subjects) if previous is not None else []
        out.append({'stage': name, 'n': len(subjects), 'lost': lost})
        previous = subjects
    return out


def classify_missing(column: str, subject_id: str, master: pd.DataFrame,
                     clean_subjects: Set[str], h2_subjects: Set[str],
                     avail: pd.DataFrame) -> str:
    """Assign one of the four reasons to a missing cell."""
    if column in BY_DESIGN:
        return 'by_design'

    if not avail.loc[subject_id, 'has_behavioral']:
        return 'absent'                      # no behavioral session in the repo

    if column.startswith(H2_ONLY_PREFIXES) and subject_id not in h2_subjects:
        return 'excluded'
    if column.startswith('sham_') and subject_id not in clean_subjects:
        return 'excluded'
    if column.startswith('theta_') and not avail.loc[subject_id, 'has_eeg']:
        return 'absent'

    return 'absent'


def variable_audit(sample: str = 'all') -> pd.DataFrame:
    """Coverage and missingness reasons for every analysis variable."""
    from data_loading import load_all_subjects
    from exclusions import apply_all_exclusions

    master = pd.read_csv(REPO_ROOT / 'data' / 'master_subject_data.csv',
                         dtype={'subject_id': str})
    avail = raw_availability().set_index('subject_id')

    if sample == 'dissertation':
        pool = set(DISSERTATION_SUBJECTS)
    elif sample == 'new':
        pool = set(SUBJECT_INFO) - set(DISSERTATION_SUBJECTS)
    else:
        pool = set(SUBJECT_INFO)

    master = master[master['subject_id'].isin(pool)]

    results = apply_all_exclusions(
        load_all_subjects(sample=sample, verbose=False), verbose=False)
    clean_subjects = set(results['data_clean']['subject_id'].unique())
    h2_subjects = {str(s) for s in results['h2_eligible']}

    groups = {
        'behavioral (sham)': ['sham_p_stay_win', 'sham_p_shift_lose',
                              'sham_alpha', 'sham_beta', 'sham_accuracy',
                              'sham_win_rate'],
        'behavioral (active)': ['active_p_stay_win', 'active_p_shift_lose',
                                'active_alpha', 'active_beta',
                                'active_accuracy', 'active_win_rate'],
        'EEG': ['theta_p95', 'theta_p75', 'theta_median'],
        'demographics': ['age', 'gender', 'race', 'ethnicity', 'education_years'],
        'cognition': ['hvlt_total', 'flanker_score', 'running_dots_score',
                      'salthouse_letter', 'salthouse_pattern',
                      'digit_span_total', 'bvmt_total', 'trails_a_time'],
        'questionnaires': ['spsrq_sr', 'spsrq_sp', 'bpsqi_global', 'ffmq_total',
                           'crt_total', 'bbs_avg', 'audit_total', 'ctq_total',
                           'promis_anxiety', 'loneliness_total'],
    }

    rows = []
    for group, columns in groups.items():
        for col in columns:
            if col not in master.columns:
                rows.append({'group': group, 'variable': col, 'n_present': 0,
                             'n_missing': len(master), 'status': 'COLUMN ABSENT',
                             'by_design': 0, 'excluded': 0, 'absent': 0})
                continue

            present = master[col].notna()
            missing_ids = master.loc[~present, 'subject_id'].tolist()
            reasons = [classify_missing(col, s, master, clean_subjects,
                                        h2_subjects, avail)
                       for s in missing_ids if s in avail.index]
            counts = pd.Series(reasons).value_counts().to_dict()

            rows.append({
                'group': group, 'variable': col,
                'n_present': int(present.sum()),
                'n_missing': int((~present).sum()),
                'status': 'ok',
                'by_design': counts.get('by_design', 0),
                'excluded': counts.get('excluded', 0),
                'absent': counts.get('absent', 0),
            })

    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--sample', default='all',
                        choices=['all', 'dissertation', 'new'])
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR))
    args = parser.parse_args(argv)

    print('=' * 74)
    print(f'SAMPLE AUDIT  (sample={args.sample!r})')
    print('=' * 74)

    avail = raw_availability()
    pool = avail if args.sample == 'all' else (
        avail[avail['in_dissertation']] if args.sample == 'dissertation'
        else avail[~avail['in_dissertation']])

    print(f'\nRaw data on disk, {len(pool)} registered subjects:')
    print(f'  behavioral session : {pool["has_behavioral"].sum()}')
    print(f'  EEG session        : {pool["has_eeg"].sum()}')
    print(f'  both               : {(pool["has_behavioral"] & pool["has_eeg"]).sum()}')
    print(f'  neither            : {(~pool["has_behavioral"] & ~pool["has_eeg"]).sum()}')

    gap = pool[pool['has_eeg'] & ~pool['has_behavioral']]
    if len(gap):
        print(f'\n  EEG but no behavioral ({len(gap)}) — the session happened, so '
              f'these files\n  may exist on the acquisition machine:')
        print('   ', ', '.join(gap['subject_id']))

    print('\n' + '-' * 74)
    print('SAMPLE ATTRITION')
    print('-' * 74)
    for stage in sample_attrition(args.sample):
        line = f'  {stage["stage"]:34s} {stage["n"]:3d}'
        if stage['lost']:
            line += f'   -{len(stage["lost"])}: {", ".join(stage["lost"])}'
        print(line)

    print('\n' + '-' * 74)
    print('VARIABLE COVERAGE')
    print('-' * 74)
    audit = variable_audit(args.sample)
    n_pool = len(pool)
    current = None
    for _, row in audit.iterrows():
        if row['group'] != current:
            current = row['group']
            print(f'\n  {current}')
        bits = []
        for key in ['by_design', 'excluded', 'absent']:
            if row[key]:
                bits.append(f'{key}={row[key]}')
        detail = ('  [' + ', '.join(bits) + ']') if bits else ''
        flag = '' if row['status'] == 'ok' else f'  <{row["status"]}>'
        print(f'    {row["variable"]:22s} {row["n_present"]:3d}/{n_pool}'
              f'{detail}{flag}')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    avail.to_csv(out_dir / f'audit_availability_{args.sample}.csv', index=False)
    audit.to_csv(out_dir / f'audit_variables_{args.sample}.csv', index=False)
    print(f'\nWrote {out_dir / f"audit_availability_{args.sample}.csv"}')
    print(f'Wrote {out_dir / f"audit_variables_{args.sample}.csv"}')

    print('\n' + '-' * 74)
    print('Anything counted as `absent` for a subject who has the relevant raw')
    print('data is worth a second look — that is what a technical failure looks')
    print('like from here.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
