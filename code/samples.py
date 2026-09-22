"""
samples.py — The analysis samples, defined once

Each analysis uses everyone who has the data *that analysis needs*, which is
the largest N available for each question. A single "has everything" sample
was considered and rejected: every data stream has its own gaps, and their
intersection collapses to 35 -- below the defended N of 39 -- while dropping
subjects from the preregistered tests for reasons unrelated to them (a missing
FLAIR scan says nothing about stimulation response).

So there is a small set of named samples, each with one rule:

  H1       passed the preregistered H1 exclusions, and has age
  H1_RL    H1 with a hierarchical RL fit -- alpha/beta analyses (57 of 61)
  H2       H2-eligible: passed the exclusions in both sessions
  Dose     a FLAIR head model (T1-only excluded) and age
  Dose_H2  H2 and Dose -- the field as a moderator of the stimulation response

Moderator analyses (theta, iTF, fMRI, surveys) run on H1 or H2 intersected
with that measure, and report their own N.

**Why this module exists.** Before it, each script decided its own sample --
dropping T1-only here, requiring education there -- and the same result could
appear with two different Ns in one paper (E-field x age: 56 in the notebook,
59 in the figure). Now every analysis asks for a sample by name, and three
checks guard it:

1. **Expected N.** Each sample declares its N. If the data change and the N
   moves, `check()` fails until the number here is updated -- so a change of
   sample is always a deliberate edit, never a side effect.
2. **Required variables.** Each sample lists the variables its analyses need.
   Any subject in the sample missing one is reported, which is how a stale
   upstream output is caught (the hierarchical RL fit covering 55 of the 57
   H2 subjects, for instance).
3. **Scripts assert their sample.** `assert_sample(ids, name)` raises if a
   script's working set differs from the named sample, naming the subjects.

Usage
-----
    python samples.py            # table + checks; exit 1 on any problem
"""

from __future__ import annotations

import sys
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

from config import REPO_ROOT, EFIELD_CSV_PATH, FREESURFER_MORPH_PATH, COG_COMPOSITE

MASTER_PATH = REPO_ROOT / 'data' / 'master_subject_data.csv'

# Each rule is a function of the joined master + E-field frame.
SAMPLES: Dict[str, dict] = {
    'H1': dict(
        rule='passed the preregistered H1 exclusions (sham-session behaviour); has age',
        define=lambda d: d['sham_p_stay_win'].notna() & d['age'].notna(),
        expected_n=61,
        used_for='H1.1, H1.1.1, H1.2; baseline behaviour; exploratory baseline sweep',
        requires=['age', COG_COMPOSITE, 'sham_p_stay_win', 'sham_p_shift_lose',
                  'sham_alpha', 'sham_beta', 'sham_accuracy', 'sham_win_rate'],
    ),
    'H1_RL': dict(
        rule='H1 with a hierarchical RL fit (the fit needs both sessions)',
        define=lambda d: SAMPLES['H1']['define'](d) & d['sham_alpha_hb'].notna(),
        expected_n=57,
        used_for='H1 analyses of alpha and beta (H1.1.1, H1.1.2) under RL_ESTIMATES = hb',
        requires=['age', COG_COMPOSITE, 'sham_alpha_hb', 'sham_beta_hb',
                  'sham_p_shift_lose'],
    ),
    'H2': dict(
        rule='H2-eligible: passed the preregistered exclusions in both sessions',
        define=lambda d: d['delta_accuracy'].notna(),
        expected_n=57,
        used_for='H2.1 paired tests and equivalence; H2.2 age moderation',
        requires=['age', COG_COMPOSITE, 'delta_p_stay_win', 'delta_p_shift_lose',
                  'delta_alpha', 'delta_beta', 'delta_accuracy', 'delta_win_rate',
                  'delta_alpha_hb', 'delta_beta_hb'],
    ),
    'Dose': dict(
        rule='FLAIR head model (T1-only excluded); has age',
        define=lambda d: d['t1_only'].eq(False) & d['age'].notna(),
        expected_n=59,
        used_for='E-field x age; geometry, skull and atrophy analyses; Figs 1-3',
        requires=['age', 'mean_magnE', 'dist_pial_dlpfc_p1', 'layer_skull',
                  'skull_t1_span', 'csf_charm', 'icv_charm'],
    ),
    'Dose_H2': dict(
        rule='H2 and Dose',
        define=lambda d: SAMPLES['H2']['define'](d) & SAMPLES['Dose']['define'](d),
        expected_n=50,
        used_for='E-field as a moderator of the stimulation response',
        requires=['mean_magnE', 'delta_accuracy', 'delta_alpha'],
    ),
}


def load() -> pd.DataFrame:
    """
    Master CSV joined to the E-field and FreeSurfer tables, one row per subject.

    The one place these three are joined. Scripts that need anatomy should
    start from here (or `frame(name)`) rather than merging the files
    themselves, which is how their samples used to drift apart.
    """
    m = pd.read_csv(MASTER_PATH, dtype={'subject_id': str}, low_memory=False)
    d = m
    for path in (EFIELD_CSV_PATH, FREESURFER_MORPH_PATH):
        t = pd.read_csv(path, dtype={'subject_id': str})
        keep = [c for c in t.columns if c not in d.columns or c == 'subject_id']
        d = d.merge(t[keep], on='subject_id', how='left')
    d['age'] = pd.to_numeric(d['age'], errors='coerce')
    # Subjects with no head model at all are not in any Dose sample.
    d['t1_only'] = d['t1_only'].map({True: True, False: False, 'True': True,
                                     'False': False}).astype('object')
    return d


def ids(name: str, d: Optional[pd.DataFrame] = None) -> Set[str]:
    if name not in SAMPLES:
        raise KeyError(f'unknown sample {name!r}; defined: {list(SAMPLES)}')
    d = load() if d is None else d
    return set(d.loc[SAMPLES[name]['define'](d).fillna(False).astype(bool), 'subject_id'])


def frame(name: str, d: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The joined frame restricted to one sample."""
    d = load() if d is None else d
    return d[d['subject_id'].isin(ids(name, d))].reset_index(drop=True)


def assert_sample(subject_ids: Iterable, name: str) -> None:
    """Raise if a script's working set is not exactly the named sample."""
    got, want = set(map(str, subject_ids)), ids(name)
    if got != want:
        raise AssertionError(
            f'working set is not the {name} sample ({len(got)} vs {len(want)}).\n'
            f'  extra  : {sorted(got - want)}\n  missing: {sorted(want - got)}\n'
            f'Use samples.frame({name!r}) or fix the filter.')


def check(verbose: bool = True) -> List[str]:
    """Expected N and required-variable coverage for every sample."""
    d = load()
    problems = []
    rows = []
    for name, s in SAMPLES.items():
        members = d[d['subject_id'].isin(ids(name, d))]
        n = len(members)
        if n != s['expected_n']:
            problems.append(f'{name}: N = {n}, expected {s["expected_n"]}. If the data '
                            f'changed deliberately, update expected_n in samples.py.')
        for v in s['requires']:
            if v not in members.columns:
                problems.append(f'{name}: required variable `{v}` is not in the data')
                continue
            gap = members.loc[members[v].isna(), 'subject_id']
            if len(gap):
                problems.append(f'{name}: `{v}` missing for {len(gap)} of {n} '
                                f'({" ".join(sorted(gap))})')
        rows.append(dict(sample=name, N=n, expected=s['expected_n'], rule=s['rule'],
                         used_for=s['used_for']))
    if verbose:
        print(pd.DataFrame(rows)[['sample', 'N', 'expected', 'rule']].to_string(index=False))
        print()
        for p in problems:
            print(f'  PROBLEM  {p}')
        if not problems:
            print('  all samples at expected N, all required variables complete')
    return problems


def table_markdown() -> str:
    d = load()
    lines = ['| sample | N | rule | used for |', '|---|---|---|---|']
    for name, s in SAMPLES.items():
        lines.append(f'| **{name}** | {len(ids(name, d))} | {s["rule"]} | {s["used_for"]} |')
    return '\n'.join(lines)


if __name__ == '__main__':
    sys.exit(1 if check() else 0)
