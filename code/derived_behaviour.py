"""
derived_behaviour.py — Response times, choice dynamics and the blinding check

Everything here comes from trial columns that were recorded from the first
session and never used. `rt` is present on all 34,319 clean trials and no
measure in the pipeline read it until 2026-09-22; `stim_guess` records what the
participant thought after every run and was likewise unread.

**Response time.** The task has a response deadline just under 2 s, so RT is
naturally bounded; trials outside 200-1990 ms are dropped as anticipations and
deadline misses rather than winsorised. Mean RT is the one measure in this
module with a clear age effect (r = +.40 with age, and unusually for this
sample it is a genuine within-band gradient, r = +.41 inside the older band).
It converges with the processing-speed composite (r = -.47) and is independent
of accuracy, which is the useful part: older participants are *slower without
being less accurate*, so the age effect is on responding, not on learning.

**Post-error and post-loss slowing** are the standard adjustment measures:
mean RT after an error minus mean RT after a correct response, and the same
split by reward. Both are computed within run, since the trial before the
first trial of a run belongs to a different block.

**Choice dynamics.** Switch rate (any change of choice between consecutive
trials in a run), lapse rate (no response), perseverative errors (incorrect
choices in the five trials after a reversal, i.e. sticking with the option that
used to pay) and asymptotic accuracy (the five trials before a reversal, when
the contingency has been stable longest). Perseveration and asymptote separate
flexibility from stable performance, which trials-to-criterion conflates.

**Blinding.** `score_blinding` compares how often participants said they
believed they were being stimulated under active versus sham. It belongs here
because it is the check that decides how the H2 null should be read: a null
under broken blinding means something different from a null under intact
blinding. In this sample blinding held (69% vs 64%, dz = +0.12, p = .38).

None of the stimulation contrasts in this module is significant, which is the
point of computing them -- the preregistered DVs are not the only way the
effect could have shown up, and now that is a demonstrated claim rather than an
assumed one.

Usage
-----
    python derived_behaviour.py          # summary over the H1/H2 samples
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import pandas as pd

# Response-deadline bounds. Below RT_MIN is an anticipation (the stimuli have
# barely been seen); at RT_MAX the trial hit the deadline and the "RT" is the
# deadline, not a decision time.
RT_MIN_MS = 200.0
RT_MAX_MS = 1990.0

# Trials after a reversal counted as perseverative, and before one as asymptotic.
PERSEV_WINDOW = 5
ASYMPTOTE_WINDOW = 5


def _as_bool(s: pd.Series) -> pd.Series:
    """Trial flags arrive as bool, 'True'/'False' strings or 0/1 across files."""
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.upper().eq('TRUE') | s.astype(str).eq('1')


def _subject_condition_features(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values(['run', 'trial_num'])
    rt = pd.to_numeric(g['rt'], errors='coerce')
    rt = rt.where(rt.between(RT_MIN_MS, RT_MAX_MS))

    correct = _as_bool(g['correct'])
    reward = _as_bool(g['reward'])
    # Only compare against the previous trial when it exists and is in the
    # same run; across a run boundary the "previous trial" is another block.
    same_run = g['run'].eq(g['run'].shift(1))
    has_prev = g['correct'].shift(1).notna() & same_run
    prev_correct = correct.shift(1).fillna(False).astype(bool)
    prev_reward = reward.shift(1).fillna(False).astype(bool)

    switched = (g['choice'].ne(g['choice'].shift(1)) & same_run
                & g['choice'].notna() & g['choice'].shift(1).notna())

    out = {
        'rt_mean': rt.mean(),
        'rt_sd': rt.std(),
        'rt_cv': rt.std() / rt.mean() if rt.mean() else np.nan,
        'rt_post_error': (rt.where(has_prev & ~prev_correct).mean()
                          - rt.where(has_prev & prev_correct).mean()),
        'rt_post_loss': (rt.where(has_prev & ~prev_reward).mean()
                         - rt.where(has_prev & prev_reward).mean()),
        'switch_rate': switched.mean(),
        'lapse_rate': g['choice'].isna().mean(),
        'n_rt_trials': int(rt.notna().sum()),
    }
    if 'in_rev_window' in g.columns and 'trial_from_rev' in g.columns:
        post = g[g['in_rev_window'] & g['trial_from_rev'].between(0, PERSEV_WINDOW - 1)]
        pre = g[g['in_rev_window'] & g['trial_from_rev'].between(-ASYMPTOTE_WINDOW, -1)]
        out['persev_errors'] = (~_as_bool(post['correct'])).mean() if len(post) else np.nan
        out['asymptotic_acc'] = _as_bool(pre['correct']).mean() if len(pre) else np.nan
    return pd.Series(out)


FEATURES = ('rt_mean', 'rt_sd', 'rt_cv', 'rt_post_error', 'rt_post_loss',
            'switch_rate', 'lapse_rate', 'persev_errors', 'asymptotic_acc')


def compute_derived_behaviour(data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    One row per subject x condition, with the measures listed in FEATURES.

    `data` is the cleaned trial frame. Pass it through
    `reversal_analysis.identify_reversals` first if the perseveration and
    asymptote columns are wanted; without the reversal columns they are simply
    absent, and the rest are still computed.
    """
    out = (data.groupby(['subject_id', 'condition'], observed=True)
               .apply(_subject_condition_features, include_groups=False)
               .reset_index())
    out['subject_id'] = out['subject_id'].astype(str)
    if verbose:
        n = out[out.condition.isin(['sham', 'active'])]
        print(f'  Derived behaviour: {len(out)} subject-conditions, '
              f'median RT {n.rt_mean.median():.0f} ms')
    return out


def score_blinding(data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Per subject x condition, the proportion of runs the participant said they
    believed they were being stimulated.

    `stim_guess` is recorded once per trial but is a single end-of-run answer
    repeated down the run, so it is collapsed per run before averaging --
    otherwise a long run would outvote a short one.
    """
    if 'stim_guess' not in data.columns:
        return pd.DataFrame(columns=['subject_id', 'condition', 'guess_stim_rate'])
    g = data.dropna(subset=['stim_guess']).copy()
    g['yes'] = g['stim_guess'].astype(str).str.strip().str.lower().eq('yes')
    per_run = g.groupby(['subject_id', 'condition', 'run'], observed=True)['yes'].max()
    out = (per_run.groupby(['subject_id', 'condition'], observed=True).mean()
                  .reset_index().rename(columns={'yes': 'guess_stim_rate'}))
    out['subject_id'] = out['subject_id'].astype(str)
    if verbose and len(out):
        w = out[out.condition.isin(['sham', 'active'])].pivot(
            index='subject_id', columns='condition', values='guess_stim_rate').dropna()
        if {'sham', 'active'} <= set(w.columns):
            print(f'  Blinding: guessed "stimulated" on {100 * w["sham"].mean():.0f}% of '
                  f'sham runs vs {100 * w["active"].mean():.0f}% of active runs '
                  f'(N = {len(w)})')
    return out


def main(argv=None) -> int:
    from scipy import stats
    from data_loading import load_all_subjects
    from exclusions import apply_all_exclusions
    from reversal_analysis import identify_reversals
    from config import REVERSAL_WINDOW_PRE, REVERSAL_WINDOW_POST
    import samples

    clean = apply_all_exclusions(load_all_subjects(sample='all'))['data_clean']
    clean = identify_reversals(clean, window_pre=REVERSAL_WINDOW_PRE,
                               window_post=REVERSAL_WINDOW_POST, verbose=False)
    feats = compute_derived_behaviour(clean)
    blind = score_blinding(clean)

    w = feats.pivot(index='subject_id', columns='condition')
    w.columns = [f'{c}_{a}' for a, c in w.columns]
    w = w.reset_index()
    ages = samples.frame('H1')[['subject_id', 'age']]
    ages['age'] = pd.to_numeric(ages['age'], errors='coerce')
    w = w.merge(ages, on='subject_id')
    h2 = samples.ids('H2')

    print(f"\n{'measure':17}{'age r (sham)':>14}{'p':>9}{'N':>5} |"
          f"{'active-sham dz':>16}{'p':>9}{'N':>5}")
    print('-' * 78)
    for k in FEATURES:
        s = w[['age', f'sham_{k}']].dropna()
        r, p = stats.pearsonr(s['age'], s[f'sham_{k}'])
        q = w[w.subject_id.isin(h2)][[f'sham_{k}', f'active_{k}']].dropna()
        diff = q[f'active_{k}'] - q[f'sham_{k}']
        _, pp = stats.ttest_rel(q[f'active_{k}'], q[f'sham_{k}'])
        print(f'{k:17}{r:+14.3f}{p:9.4f}{len(s):5d} |'
              f'{diff.mean() / diff.std(ddof=1):+16.3f}{pp:9.4f}{len(q):5d}')

    bw = blind[blind.condition.isin(['sham', 'active'])].pivot(
        index='subject_id', columns='condition', values='guess_stim_rate').dropna()
    t, p = stats.ttest_rel(bw['active'], bw['sham'])
    d = bw['active'] - bw['sham']
    print(f'\nBlinding: sham {bw["sham"].mean():.3f} vs active {bw["active"].mean():.3f}, '
          f't({len(bw)-1}) = {t:+.2f}, p = {p:.3f}, dz = {d.mean()/d.std(ddof=1):+.3f}, '
          f'N = {len(bw)}')
    print('  A null H2 under intact blinding is a different claim from a null '
          'under broken blinding.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
