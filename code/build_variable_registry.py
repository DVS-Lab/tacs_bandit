"""
build_variable_registry.py — Where every analysis variable comes from

Joins the hand-written provenance in `variable_spec.csv` to everything that can
be computed from the data itself, and writes:

  data/variable_registry.csv   one row per variable, machine-readable
  VARIABLES.md                 the same, grouped by domain, for people

**It fails when the spec and the data disagree.** A column that appears in a
canonical table without a spec row, or a spec row whose column has vanished,
is an error, and the script exits non-zero. That is the point of it: every
earlier provenance problem in this project was silent -- a stale CSV read by
a bare relative path, five BPAQ columns that were all NaN, a composite built
from different tests for different people. A registry that only documents
would drift the same way. This one refuses to build until it is true.

What it computes, per variable:

- coverage (n, missing IDs when few enough to chase)
- range and type
- **missingness pattern**, detected rather than asserted. For each partially
  missing variable, presence is scored against age and against subject-ID
  order (a proxy for enrollment date) as an AUC. Near 0 or 1 means presence
  is almost perfectly sorted by that variable -- an age gate or a protocol
  change -- and the declared explanation in the spec is checked against it.
- **consumers**: which analysis scripts reference the column by name. A
  variable nothing reads is either dead weight or an analysis never run.

Known limit of the consumer scan: names built at runtime (f'{cond}_accuracy')
are not found by a literal search, so a variable reached only that way will
look unused. The scan resolves COG_COMPOSITE, the one indirection that matters.

Usage
-----
    python build_variable_registry.py
    python build_variable_registry.py --check     # exit status only, no files
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from config import REPO_ROOT, EFIELD_CSV_PATH, FREESURFER_MORPH_PATH, COG_COMPOSITE

CODE = REPO_ROOT / 'code'
SPEC = CODE / 'variable_spec.csv'
OUT_CSV = REPO_ROOT / 'data' / 'variable_registry.csv'
OUT_MD = REPO_ROOT / 'VARIABLES.md'

TABLES = {
    'master': REPO_ROOT / 'data' / 'master_subject_data.csv',
    'efield': EFIELD_CSV_PATH,
    'freesurfer': FREESURFER_MORPH_PATH,
}

# Scripts that *produce* the tables. Their references are definitions, not use.
PRODUCERS = {'cognitive_merge.py', 'build_master_data.py', 'freesurfer_morph.py',
             'build_variable_registry.py'}

# AUC beyond which presence counts as sorted by a variable.
GATED = 0.90
LEANS = 0.75

DOMAIN_ORDER = [
    'identifier', 'exclusion', 'demographics', 'cognition', 'cognitive_composite',
    'behaviour_wsls', 'behaviour_accuracy', 'behaviour_rl_mle', 'behaviour_rl_hb',
    'eeg', 'efield', 'geometry', 'skull', 'morphometry',
    'survey_reward', 'survey_decision', 'survey_sleep', 'survey_mindfulness',
    'survey_affect_social', 'survey_substance', 'survey_health', 'survey_other',
    'survey_exploratory', 'item_count', 'qc',
]
QUIET_ROLES = {'identifier', 'qc'}


def load_tables() -> Dict[str, pd.DataFrame]:
    out = {}
    for name, path in TABLES.items():
        out[name] = pd.read_csv(path, dtype={'subject_id': str}, low_memory=False)
    return out


# A pattern must also be distinguishable from chance. With one subject
# missing, the AUC is just that subject's rank: if they happen to be the
# youngest it reads as a perfect age gate. The p-value is what stops that.
ALPHA = 0.01


def auc(present: pd.Series, x: pd.Series):
    """(AUC, p). AUC = P(x of a present subject > x of a missing one)."""
    m = pd.concat([present, x], axis=1).dropna()
    a = m[m.iloc[:, 0]].iloc[:, 1]
    b = m[~m.iloc[:, 0]].iloc[:, 1]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    res = stats.mannwhitneyu(a, b, alternative='two-sided')
    return res.statistic / (len(a) * len(b)), res.pvalue


def _strength(a: float) -> float:
    return 0.0 if np.isnan(a) else abs(a - 0.5) * 2   # 0 unrelated, 1 perfectly sorted


def gate_level(a: float, p: float) -> str:
    """'gated', 'leans' or '' for one dimension."""
    if np.isnan(a) or np.isnan(p) or p >= ALPHA:
        return ''
    s = _strength(a)
    if s >= (GATED - 0.5) * 2:
        return 'gated'
    if s >= (LEANS - 0.5) * 2:
        return 'leans'
    return ''


def classify(age_res, enr_res) -> str:
    parts = []
    for name, (a, p) in (('age', age_res), ('enrollment', enr_res)):
        lvl = gate_level(a, p)
        if not lvl:
            continue
        side = (('older' if a > .5 else 'younger') if name == 'age'
                else ('later' if a > .5 else 'earlier'))
        parts.append(f'{name}-gated ({side} present)' if lvl == 'gated'
                     else f'leans {name} ({side} present)')
    return '; '.join(parts) if parts else 'scattered'


def consumer_index() -> Dict[str, str]:
    """File name -> source text, for every analysis script."""
    files = {}
    for p in sorted(CODE.rglob('*.py')):
        rel = p.relative_to(CODE).as_posix()
        if (rel.startswith('archive/') or '__pycache__' in rel or
                '.ipynb_checkpoints' in rel or p.name in PRODUCERS):
            continue
        try:
            files[rel] = p.read_text()
        except UnicodeDecodeError:
            continue
    return files


def consumers_of(var: str, files: Dict[str, str]) -> List[str]:
    pat = re.compile(rf"""['"]{re.escape(var)}['"]""")
    hits = [f for f, s in files.items() if pat.search(s)]
    if var == COG_COMPOSITE:
        hits += [f for f, s in files.items() if 'COG_COMPOSITE' in s and f not in hits]
    return sorted(hits)


def build(write: bool = True) -> int:
    spec = pd.read_csv(SPEC, dtype=str).fillna('')
    tables = load_tables()
    age = tables['master'].set_index('subject_id')['age'].pipe(pd.to_numeric, errors='coerce')
    files = consumer_index()

    errors, warnings = [], []

    # --- the contract: spec and data describe the same columns ------------
    for t, df in tables.items():
        live = set(df.columns)
        documented = set(spec.loc[spec.table == t, 'variable'])
        for v in sorted(live - documented):
            errors.append(f'{t}: column `{v}` is in the data but not in variable_spec.csv')
        for v in sorted(documented - live):
            errors.append(f'{t}: `{v}` is in variable_spec.csv but no longer in the data')

    rows = []
    for _, s in spec.iterrows():
        df = tables.get(s.table)
        if df is None or s.variable not in df.columns:
            continue
        col = df[s.variable]
        n_total = len(df)
        n = int(col.notna().sum())
        num = pd.to_numeric(col, errors='coerce')
        is_num = num.notna().sum() == n and n > 0
        r = dict(s)
        r.update(n=n, n_total=n_total, n_missing=n_total - n,
                 dtype='numeric' if is_num else 'text',
                 n_unique=int(col.nunique()))
        if is_num and n:
            r.update(min=float(num.min()), median=float(num.median()), max=float(num.max()))
        if n == 0:
            errors.append(f'{s.table}: `{s.variable}` is entirely empty')

        # Missingness pattern
        ids = df['subject_id']
        present = col.notna().set_axis(ids)
        if 0 < n < n_total and s.variable != 'subject_id':
            res = {'age': auc(present, age.reindex(ids)),
                   'enrollment': auc(present, pd.to_numeric(ids, errors='coerce')
                                     .rank().set_axis(ids))}
            detected = classify(res['age'], res['enrollment'])
            r.update(auc_age=round(res['age'][0], 3), p_age=res['age'][1],
                     auc_enrollment=round(res['enrollment'][0], 3),
                     p_enrollment=res['enrollment'][1], detected_missing=detected)
            if n_total - n <= 10:
                r['missing_ids'] = ' '.join(sorted(ids[~col.notna().values]))
            # A declared gate is checked on its own dimension: a variable can be
            # both (loneliness was given only to older, early enrollees).
            declared = [d.strip() for d in s.expected_missing.split(';') if d.strip()]
            for dec in declared:
                if dec in ('age-gated', 'enrollment-gated'):
                    dim = dec.split('-')[0]
                    if not gate_level(*res[dim]):
                        warnings.append(f'`{s.variable}`: declared {dec}, but presence is '
                                        f'not sorted by {dim} (AUC {res[dim][0]:.2f}, '
                                        f'p = {res[dim][1]:.3f})')
            undeclared = [d for d in ('age', 'enrollment')
                          if gate_level(*res[d]) and f'{d}-gated' not in declared]
            if (undeclared and not any(d in ('by-design', 'task-level', 'individual') for d in declared)
                    and s.role not in QUIET_ROLES):
                warnings.append(f'`{s.variable}` ({n}/{n_total}): presence is {detected}, '
                                f'which the spec does not explain')
        else:
            r['detected_missing'] = 'complete' if n == n_total else ''

        cons = consumers_of(s.variable, files) if s.variable != 'subject_id' else []
        r['consumers'] = ' '.join(cons)
        r['n_consumers'] = len(cons)
        rows.append(r)

    reg = pd.DataFrame(rows)
    unused = reg[(reg.n_consumers == 0) & ~reg.role.isin(QUIET_ROLES | {'item_count'})
                 & (reg.domain != 'item_count')]

    # --- report -----------------------------------------------------------
    print(f'{len(reg)} variables across {len(tables)} tables')
    for t in tables:
        print(f'  {t:11} {int((reg.table == t).sum()):3d}')
    print(f'\nerrors: {len(errors)}   warnings: {len(warnings)}   '
          f'unreferenced by any analysis script: {len(unused)}')
    for e in errors:
        print(f'  ERROR  {e}')
    for w in warnings:
        print(f'  warn   {w}')

    if write and not errors:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        reg.to_csv(OUT_CSV, index=False)
        OUT_MD.write_text(render_md(reg, errors, warnings, unused))
        print(f'\nwrote {OUT_CSV.relative_to(REPO_ROOT)}\nwrote {OUT_MD.relative_to(REPO_ROOT)}')
    elif errors:
        print('\nNot written: fix variable_spec.csv (or the data) first.')
    return 1 if errors else 0


# ---------------------------------------------------------------------------

def _fmt(x):
    if x == '' or x is None or (isinstance(x, float) and np.isnan(x)):
        return ''
    if isinstance(x, float):
        return f'{x:.3g}' if abs(x) < 1e4 else f'{x:.3e}'
    return str(x)


def render_md(reg: pd.DataFrame, errors, warnings, unused) -> str:
    L = []
    L.append('# Variable registry\n')
    L.append('Generated by `code/build_variable_registry.py` from `code/variable_spec.csv` '
             'and the three canonical tables. **Do not edit by hand** -- edit the spec and '
             're-run. The build fails if the spec and the data disagree, so if this file '
             'exists it matches the data it was built from.\n')
    L.append('| table | file | variables |\n|---|---|---|')
    for t, p in TABLES.items():
        L.append(f'| {t} | `{p.relative_to(REPO_ROOT)}` | {int((reg.table == t).sum())} |')
    L.append('')
    L.append(f'Cognitive composite in use: `COG_COMPOSITE = {COG_COMPOSITE!r}` (config.py).\n')

    L.append('## Analysis samples\n')
    L.append('Defined in `code/samples.py`; `python code/samples.py` checks them. '
             'Every analysis uses one by name.\n')
    import samples
    L.append(samples.table_markdown() + '\n')

    L.append('## How to read the missingness columns\n')
    L.append('*Declared* is the reason given in the spec. *Detected* is computed: presence is '
             'scored against age and against subject-ID order (a proxy for enrollment date) '
             f'as an AUC. Beyond {GATED:.2f} (or below {1 - GATED:.2f}) presence is almost '
             'perfectly sorted by that variable, which means an age gate or a protocol change, '
             'not chance. A declared reason the data contradict, or a strong pattern with no '
             'declared reason, is listed under findings.\n')

    L.append('## Findings\n')
    if not warnings:
        L.append('No unexplained missingness patterns.\n')
    else:
        L.append('**Missingness the spec does not explain, or explains wrongly**\n')
        L += [f'- {w}' for w in warnings]
        L.append('')
    if len(unused):
        L.append(f'**{len(unused)} variables no analysis script references by name.** '
                 'Either unused, or reached only through a runtime-built name (see the '
                 'builder docstring). Worth a decision each: analyse, or stop carrying.\n')
        for d, g in unused.groupby('domain', sort=False):
            L.append(f'- *{d}*: ' + ', '.join(f'`{v}`' for v in g.variable))
        L.append('')

    L.append('## Regenerating the tables\n')
    L.append('**master** — `python code/build_master_data.py`. Behaviour from `data/bandit`, '
             'EEG theta from `derivatives/eeg/theta_subject_metrics.csv` '
             '(`python code/run_theta_metrics.py`), hierarchical RL from '
             '`derivatives/rl_models/rw_within_all_subjects.csv` '
             '(`code/rl_models/run_fit.py`, see its README), everything else from the '
             'REDCap and TabCAT exports named in `config.py`.\n')
    L.append('**efield** — built in the SimNIBS repo (`DVS-Lab/tacs_bandit_simnibs`) in two '
             'steps. The order matters and was previously unrecorded:\n')
    L.append('1. `extract_efield_roi.py` writes the first 12 columns (F3 position, ROI |E| '
             'statistics, `t1_only`) to `simNIBS/efield_roi_summary.csv`, which is copied to '
             '`data/efield_roi_summary.csv`.')
    L.append('2. Seven extractors then merge their columns into **the repo copy** with '
             '`--merge-into data/efield_roi_summary.csv`: `extract_efield_parcel`, '
             '`extract_scalp_cortex_distance`, `extract_charm_thickness`, '
             '`extract_skull_layers`, `extract_skull_t1`, `extract_intracranial`, '
             '`extract_slice_landmark`.\n')
    L.append('The SimNIBS-side file therefore holds only the step-1 columns; the repo copy '
             'is canonical. Its 12 shared columns match the upstream exactly (checked '
             '2026-09-21).\n')
    L.append('**freesurfer** — `python code/freesurfer_morph.py`, over every delivery under '
             '`FREESURFER_ROOT`. Aborts on mixed FreeSurfer versions.\n')

    order = {d: i for i, d in enumerate(DOMAIN_ORDER)}
    reg = reg.assign(_o=reg.domain.map(order).fillna(99))
    for (t, d), g in reg.sort_values(['_o']).groupby(['table', 'domain'], sort=False):
        L.append(f'## {d.replace("_", " ")}  ·  `{t}`\n')
        L.append('| variable | n | description | range | produced by | role | missing: declared / detected | used by |')
        L.append('|---|---|---|---|---|---|---|---|')
        for _, r in g.iterrows():
            rng = (f'{_fmt(r.get("min"))} – {_fmt(r.get("max"))}'
                   if r.get('dtype') == 'numeric' else f'{int(r.n_unique)} values')
            status = '' if r.status in ('current', r.role) else f' **({r.status})**'
            miss = ''
            if r.n_missing:
                dec = r.expected_missing or '—'
                det = r.get('detected_missing', '') or ''
                miss = f'{dec} / {det}'
                if r.get('missing_ids'):
                    miss += f'<br><sub>{r.missing_ids}</sub>'
            desc = r.description
            if r.missing_note:
                desc += f'<br><sub>{r.missing_note}</sub>'
            if r.notes:
                desc += f'<br><sub>{r.notes}</sub>'
            hyp = f'<br><sub>{r.hypotheses}</sub>' if r.hypotheses else ''
            used = str(r.n_consumers) if r.n_consumers else '**0**'
            L.append(f'| `{r.variable}` | {r.n}/{r.n_total} | {desc} | {rng} | '
                     f'<sub>{r.producer}</sub> | {r.role}{status}{hyp} | {miss} | {used} |')
        L.append('')
    return '\n'.join(L) + '\n'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--check', action='store_true', help='validate only; write nothing')
    a = ap.parse_args(argv)
    return build(write=not a.check)


if __name__ == '__main__':
    sys.exit(main())
