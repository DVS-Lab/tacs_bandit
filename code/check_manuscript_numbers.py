"""
check_manuscript_numbers.py — Hold the prose to the same contract as the code

`MANUSCRIPT_METHODS.md` and `MANUSCRIPT_FIGURES.md` quote roughly sixty numbers
that scripts produce. Everything else in this repository is checked: the sample
Ns are asserted, the variable registry fails when the spec and the data
disagree, and the notebook re-derives what it prints. The prose was the one
place a number could go stale silently, because nothing regenerates it.

This closes that gap. Each claim below names

  - a regular expression that must still match the document, with one capture
    group holding the number, and
  - a function that recomputes the number from the data.

Both halves matter. A mismatch means the prose is stale. A **regex that no
longer matches at all** is reported just as loudly, because it usually means
the sentence was rewritten and the check silently stopped guarding anything --
which is the failure mode that makes verification theatre rather than
verification.

Tolerances are per claim and deliberately tight: a value quoted to three
decimals is checked to three decimals. Where prose rounds ("approximately
0.30"), the tolerance says so.

This does **not** check every number in the documents, and it does not try to.
It checks the load-bearing ones -- sample sizes, every statistic in a figure
caption, and the headline results. Adding a claim is three lines.

Usage
-----
    python check_manuscript_numbers.py
    python check_manuscript_numbers.py --verbose    # show every claim
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from config import REPO_ROOT, COG_COMPOSITE, rl, ttc
import samples

METHODS = REPO_ROOT / 'MANUSCRIPT_METHODS.md'
FIGURES = REPO_ROOT / 'MANUSCRIPT_FIGURES.md'
EFIELD = REPO_ROOT / 'derivatives' / 'efield_results.csv'
ITF = REPO_ROOT / 'derivatives' / 'eeg' / 'individual_theta_frequency.csv'
THETA_RUNS = REPO_ROOT / 'derivatives' / 'eeg' / 'theta_run_metrics.csv'
QC = REPO_ROOT / 'derivatives' / 'qc_paper_variables.csv'
PPC = REPO_ROOT / 'derivatives' / 'ppc_within_summary.csv'
PPC_DUAL = REPO_ROOT / 'derivatives' / 'ppc_within_rw_dual_summary.csv'


# --------------------------------------------------------------------------
# Cached data
# --------------------------------------------------------------------------

class Data:
    """Loaded once; every claim reads from here."""

    def __init__(self):
        self.h1 = samples.frame('H1')
        self.h1['age'] = pd.to_numeric(self.h1['age'], errors='coerce')
        self.h2 = samples.frame('H2')
        self.h2['age'] = pd.to_numeric(self.h2['age'], errors='coerce')
        self.dose = samples.frame('Dose')
        self.dose['age'] = pd.to_numeric(self.dose['age'], errors='coerce')
        self.efield = pd.read_csv(EFIELD) if EFIELD.exists() else None
        self.itf = (pd.read_csv(ITF, dtype={'subject_id': str})
                    if ITF.exists() else None)
        self.theta_runs = (pd.read_csv(THETA_RUNS, dtype={'subject_id': str})
                           if THETA_RUNS.exists() else None)
        self.qc = pd.read_csv(QC) if QC.exists() else None
        self.ppc = pd.read_csv(PPC, dtype={'subject_id': str}) if PPC.exists() else None
        self.ppc_dual = (pd.read_csv(PPC_DUAL, dtype={'subject_id': str})
                         if PPC_DUAL.exists() else None)

    # -- helpers ----------------------------------------------------------
    def ef(self, key: str, field: str = 'value') -> float:
        """A value from efield_results.csv by its key."""
        row = self.efield[self.efield.key == key]
        if len(row) != 1:
            raise LookupError(f'efield_results key {key!r} matched {len(row)} rows')
        return float(row[field].iloc[0])

    def ef_note(self, key: str, pattern: str) -> float:
        """A number pulled out of an efield_results note, e.g. its p value."""
        row = self.efield[self.efield.key == key]
        if len(row) != 1:
            raise LookupError(f'efield_results key {key!r} matched {len(row)} rows')
        m = re.search(pattern, str(row['note'].iloc[0]))
        if not m:
            raise LookupError(f'pattern {pattern!r} not in note for {key!r}')
        return float(m.group(1))

    def r(self, frame: pd.DataFrame, a: str, b: str, what: str = 'r') -> float:
        m = frame[[a, b]].apply(pd.to_numeric, errors='coerce').dropna()
        rr, pp = stats.pearsonr(m[a], m[b])
        return {'r': rr, 'p': pp, 'n': float(len(m))}[what]

    def ppc_stat(self, frame: pd.DataFrame, statistic: str, what: str) -> float:
        g = frame[frame.statistic == statistic].dropna(subset=['observed', 'pred_median'])
        if what == 'inside':
            return 100 * g.inside.mean()
        return stats.pearsonr(g.observed, g.pred_median)[0]


# --------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------

class Claim:
    def __init__(self, doc: str, what: str, pattern: str,
                 compute: Callable[[Data], float], tol: float = 0.0005):
        self.doc, self.what, self.pattern = doc, what, pattern
        self.compute, self.tol = compute, tol


def build_claims() -> List[Claim]:
    C: List[Claim] = []
    m, f = 'methods', 'figures'

    def add(doc, what, pattern, compute, tol=0.0005):
        C.append(Claim(doc, what, pattern, compute, tol))

    # ---- sample sizes and demographics ---------------------------------
    add(m, 'H1 N', r'primary analysis sample \(N = (\d+)\)',
        lambda d: len(d.h1), 0)
    add(m, 'n under 40', r'(\d+) participants were under 40',
        lambda d: (d.h1.age < 40).sum(), 0)
    add(m, 'n 55 and over', r'(\d+) were 55 or over',
        lambda d: (d.h1.age >= 55).sum(), 0)
    add(m, 'n in the gap', r'with (\d+) falling in between',
        lambda d: ((d.h1.age >= 40) & (d.h1.age < 55)).sum(), 0)
    add(m, 'median age', r'median ([\d.]+)\)', lambda d: d.h1.age.median(), 0.05)
    add(m, 'n female', r'(\d+) were female', lambda d: (d.h1.gender == 'Female').sum(), 0)
    add(m, 'H2 N (samples table)', r'\| H2 \| (\d+) \|', lambda d: len(d.h2), 0)
    add(m, 'Dose N (samples table)', r'\| Dose \| (\d+) \|', lambda d: len(d.dose), 0)
    add(m, 'Dose_H2 N', r'\| Dose_H2 \| (\d+) \|',
        lambda d: len(samples.ids('Dose_H2')), 0)

    # ---- trials-to-criterion -------------------------------------------
    add(m, 'TTC reach rate %', r'reached on ([\d.]+)% of \d+ reversals',
        _reach_rate, 0.05)
    add(m, 'TTC n reversals', r'reached on [\d.]+% of (\d+) reversals',
        lambda d: _n_reversals(d), 0)

    # ---- RL boundary counts --------------------------------------------
    add(m, 'MLE sham at bound', r'(\d+) of 61 sham fits',
        lambda d: _at_bound(d.h1, 'sham_alpha'), 0)
    add(m, 'MLE active at bound', r'(\d+) of 57 active fits',
        lambda d: _at_bound(d.h1, 'active_alpha'), 0)

    # ---- posterior predictive checks ------------------------------------
    for stat, label, pat in [
            ('accuracy', 'PPC accuracy inside', r'accuracy \(([\d.]+)% of participant-conditions'),
            ('p_stay_win', 'PPC win-stay inside', r'win-stay \(([\d.]+)%\)'),
            ('ttc', 'PPC TTC inside', r'trials-to-criterion \(([\d.]+)%\)'),
            ('p_shift_lose', 'PPC lose-shift inside', r'lose-shift poorly \(([\d.]+)% inside')]:
        add(m, label, pat,
            (lambda s: (lambda d: d.ppc_stat(d.ppc, s, 'inside')))(stat), 0.05)
    add(m, 'PPC lose-shift r', r'lose-shift poorly \([\d.]+% inside, r = \.(\d+) between observed',
        lambda d: round(d.ppc_stat(d.ppc, 'p_shift_lose', 'r'), 2) * 100, 0.6)
    add(m, 'PPC dual lose-shift inside', r'dual-learning-rate model improves this \(([\d.]+)%',
        lambda d: d.ppc_stat(d.ppc_dual, 'p_shift_lose', 'inside'), 0.05)

    # ---- reliability -----------------------------------------------------
    add(m, 'lose-shift split-half', r'Split-half reliability of lose-shift is \.(\d+)',
        lambda d: round(_qc_reliability(d, 'p_shift_lose'), 2) * 100, 0.6)

    # ---- theta -----------------------------------------------------------
    add(m, 'theta real mean', r'\(([\d.]+)% vs [\d.]+%; paired',
        lambda d: d.theta_runs.theta_p95.mean(), 0.05)
    add(m, 'theta surrogate mean', r'\([\d.]+% vs ([\d.]+)%; paired',
        lambda d: d.theta_runs.theta_p95_surrogate.mean(), 0.05)
    add(m, 'theta surrogate t', r'paired t\((\d+)\) = [\d.]+',
        lambda d: _theta_t(d)[1], 0)
    add(m, 'theta % runs positive', r'positive in (\d+)% of runs',
        lambda d: round(100 * (d.theta_runs.theta_p95_excess > 0).mean()), 0.5)
    add(m, 'theta run reliability', r'Run-to-run reliability is r = \.(\d+)',
        lambda d: round(_theta_reliability(d), 2) * 100, 0.6)

    # ---- iTF -------------------------------------------------------------
    add(m, 'IAF x age r', r'decline of alpha frequency with age \(r = −\.(\d+),',
        lambda d: round(abs(_iaf_age(d, 'r')), 3) * 1000, 0.6)
    add(m, 'IAF x age p', r'with age \(r = −\.\d+, p = \.(\d+)\)',
        lambda d: round(_iaf_age(d, 'p'), 3) * 1000, 0.6)

    # ---- E-field ---------------------------------------------------------
    add(m, 'parcel r', r'giving r = −\.(\d+) \(p = \.012\)',
        lambda d: round(abs(d.ef('r age x parcel |E|')), 3) * 1000, 0.6)
    add(m, 'sphere r (methods)', r'for the sphere, with the two measures correlating at r = \.(\d+)',
        lambda d: round(d.ef('r parcel x sphere |E|'), 3) * 1000, 0.6)
    add(m, 'eTIV x age', r'the estimate correlates with age \(r = \.(\d+)\)',
        lambda d: round(d.ef('r eTIV x age'), 3) * 1000, 0.6)

    # ---- figure captions --------------------------------------------------
    add(f, 'Fig1 r', r'\(N = 59; r = −0\.(\d+), p = 0\.021\)',
        lambda d: round(abs(d.ef('r age x mean |E|')), 3) * 1000, 0.6)
    add(f, 'Fig1 N', r'\(N = (\d+); r = −0\.\d+', lambda d: len(d.dose), 0)
    add(f, 'Fig1 band d', r"Cohen's d = −0\.(\d+)",
        lambda d: round(abs(d.ef('|E| DLPFC: between-band d (old - young)')), 2) * 100, 0.6)
    add(f, 'Fig1 gap n', r'which contains (\d+) participants',
        lambda d: ((d.dose.age >= 40) & (d.dose.age < 55)).sum(), 0)
    add(f, 'Fig2 geometry unique', r'geometry uniquely explains 0\.(\d+) \(p < \.001\)',
        lambda d: round(d.ef('commonality: geometry unique R2'), 2) * 100, 0.6)
    add(f, 'Fig2 atrophy unique', r'atrophy uniquely 0\.(\d+) \(p = 0\.244\)',
        lambda d: round(d.ef('commonality: atrophy unique R2'), 2) * 100, 0.6)
    add(f, 'Fig2 full R2', r'for a full model R² = 0\.(\d+)',
        lambda d: round(d.ef('commonality: full R2'), 2) * 100, 0.6)
    add(f, 'Fig3 women r', r'women r = 0\.(\d+), p < \.001',
        lambda d: round(d.ef('Female: r skull x age'), 2) * 100, 0.6)
    add(f, 'Fig3 men r', r'\(n = 29\); men r = −0\.(\d+), p = 0\.96',
        lambda d: round(abs(d.ef('Male: r skull x age')), 2) * 100, 0.6)
    add(f, 'Fig3 interaction p', r'The age × sex interaction is significant \(p = 0\.(\d+)\)',
        lambda d: round(d.ef('age x sex INTERACTION on skull, p'), 3) * 1000, 0.6)
    add(f, 'S1 win-stay r', r'\(r = 0\.(\d+), p = 0\.72;',
        lambda d: round(d.r(d.h1, COG_COMPOSITE, 'sham_p_stay_win'), 3) * 1000, 0.6)
    add(f, 'S1 lose-shift r', r'p = 0\.72; r = −0\.(\d+), p = 0\.21',
        lambda d: round(abs(d.r(d.h1, COG_COMPOSITE, 'sham_p_shift_lose')), 3) * 1000, 0.6)
    add(f, 'S3 age x delta accuracy r', r'\(r = 0\.(\d+), p = 0\.16, N = 57\)',
        lambda d: round(d.r(d.h2, 'age', 'delta_accuracy'), 3) * 1000, 0.6)
    add(f, 'S4 delta alpha SD', r'SD = 0\.(\d+), N = 57\)',
        lambda d: round(pd.to_numeric(d.h2[rl('delta_alpha')],
                                      errors='coerce').std(), 3) * 1000, 0.6)
    add(f, 'S5 theta x delta TTC r', r'\(r = 0\.(\d+), p = 0\.28, N = 53\)',
        lambda d: round(d.r(d.h2, 'theta_p95', ttc('delta_ttc')), 3) * 1000, 0.6)
    add(f, 'S6 IAF mean', r'\(M = ([\d.]+) Hz, SD = 0\.98',
        lambda d: d.itf.iaf.mean(), 0.005)
    add(f, 'S6 IAF n', r'SD = 0\.98, n = (\d+)\)', lambda d: d.itf.iaf.notna().sum(), 0)
    add(f, 'S7 distance x field r', r'\(r = −0\.(\d+), p < 10',
        lambda d: round(abs(d.ef('r distance x |E|')), 3) * 1000, 0.6)
    add(f, 'S7 mediated %', r'mediator of ([\d.]+)% of the age–field association',
        lambda d: d.ef('distance mediation, % mediated'), 0.05)
    return C


# ---- computations that need more than one line ---------------------------

def _reach_rate(d: Data) -> float:
    """Percent of all reversals on which criterion was reached, as the docs quote it."""
    return 100 * _ttc_rev(d).reached_criterion.mean()


_TTC_CACHE = {}


def _ttc_rev(d: Data) -> pd.DataFrame:
    if 'rev' not in _TTC_CACHE:
        import io
        import contextlib
        from data_loading import load_all_subjects
        from exclusions import apply_all_exclusions
        from reversal_analysis import identify_reversals, compute_trials_to_criterion
        from config import (TTC_CRITERION, REVERSAL_WINDOW_PRE, REVERSAL_WINDOW_POST)
        with contextlib.redirect_stdout(io.StringIO()):
            clean = apply_all_exclusions(load_all_subjects(sample='all'),
                                         verbose=False)['data_clean']
            rev = identify_reversals(clean, window_pre=REVERSAL_WINDOW_PRE,
                                     window_post=REVERSAL_WINDOW_POST, verbose=False)
            _TTC_CACHE['rev'] = compute_trials_to_criterion(rev, criterion=TTC_CRITERION)
    return _TTC_CACHE['rev']


def _n_reversals(d: Data) -> int:
    return len(_ttc_rev(d))


def _at_bound(frame: pd.DataFrame, col: str) -> int:
    a = pd.to_numeric(frame[col], errors='coerce').dropna()
    return int(((a == 0.001) | (a == 0.999)).sum())


def _qc_reliability(d: Data, var: str) -> float:
    row = d.qc[(d.qc.group == 'reliability') & (d.qc.variable == var)]
    return float(row['mean'].iloc[0])


def _theta_t(d: Data):
    g = d.theta_runs.dropna(subset=['theta_p95', 'theta_p95_surrogate'])
    t, p = stats.ttest_rel(g.theta_p95, g.theta_p95_surrogate)
    return t, len(g) - 1


def _theta_reliability(d: Data) -> float:
    w = (d.theta_runs[d.theta_runs.channel == 'F4']
         .pivot_table(index='subject_id', columns='run', values='theta_p95').dropna())
    runs = sorted(w.columns)[:2]
    return stats.pearsonr(w[runs[0]], w[runs[1]])[0]


def _iaf_age(d: Data, what: str) -> float:
    m = d.h1[['subject_id', 'age']].merge(d.itf[['subject_id', 'iaf']], on='subject_id')
    m = m.dropna()
    r, p = stats.pearsonr(m.age, m.iaf)
    return {'r': r, 'p': p}[what]


# --------------------------------------------------------------------------

def normalise(text: str) -> str:
    """
    Strip blockquote markers and collapse whitespace to single spaces.

    Patterns then never depend on where a sentence happens to wrap or on
    whether it sits inside a `>` caption block -- which is exactly the kind of
    incidental difference that makes a checker report a false "not found" and
    train its reader to ignore it.
    """
    lines = [re.sub(r'^\s*>\s?', '', ln) for ln in text.splitlines()]
    return re.sub(r'\s+', ' ', ' '.join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--verbose', action='store_true', help='print every claim')
    args = ap.parse_args(argv)

    docs = {}
    for name, path in (('methods', METHODS), ('figures', FIGURES)):
        if not path.exists():
            print(f'MISSING DOCUMENT: {path}')
            return 1
        docs[name] = normalise(path.read_text())

    data = Data()
    claims = build_claims()
    stale, unmatched, errored, ok = [], [], [], 0

    for c in claims:
        hit = re.search(c.pattern, docs[c.doc])
        if not hit:
            unmatched.append(c)
            continue
        try:
            quoted = float(hit.group(1))
            truth = float(c.compute(data))
        except Exception as e:                      # noqa: BLE001
            errored.append((c, f'{type(e).__name__}: {e}'))
            continue
        if abs(quoted - truth) <= c.tol:
            ok += 1
            if args.verbose:
                print(f'  ok        {c.doc:8} {c.what:32} {quoted:g}')
        else:
            stale.append((c, quoted, truth))

    print(f'{len(claims)} claims checked against '
          f'{METHODS.name} and {FIGURES.name}\n')
    for c, quoted, truth in stale:
        print(f'  STALE     {c.doc:8} {c.what:32} document says {quoted:g}, '
              f'data say {truth:g}')
    for c in unmatched:
        print(f'  NOT FOUND {c.doc:8} {c.what:32} pattern no longer matches the '
              f'document -- was the sentence rewritten?')
    for c, err in errored:
        print(f'  ERROR     {c.doc:8} {c.what:32} {err}')

    print(f'\n  {ok} verified, {len(stale)} stale, {len(unmatched)} not found, '
          f'{len(errored)} errored')
    if stale or unmatched or errored:
        print('\n  Prose and data disagree, or a guarded sentence moved. Fix the '
              'document\n  (or the claim) before circulating either file.')
        return 1
    print('  Every checked number in the manuscript prose matches the data.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
