"""
Age-Moderated Stimulation Effects: Decomposing the Accuracy Finding

If older adults show better accuracy and win rate under active tACS,
*something* in their trial-level responses must be driving it. This module
systematically tests six candidate mechanisms:

    1. RT change scores          — rule out speed-accuracy tradeoff
    2. Subgroup paired tests     — do older adults alone show significant effects?
    3. Correct-stay decomposition — is the improvement contingency-appropriate?
    4. Reversal performance       — is the effect strongest when cognitive control matters most?
    5. Within-run learning slope  — does tACS accelerate learning rather than shift asymptote?
    6. DDM drift rate             — does evidence accumulation quality improve?

All analyses use the same convention: Δ = active − sham (positive = tACS benefit).
Age moderation tested via Pearson correlation and OLS regression controlling for
global cognition.

Usage in main_analyses.ipynb:
    from age_moderated_stimulation import run_age_moderated_stimulation
    age_stim_results = run_age_moderated_stimulation(
        data_clean=data_clean,
        subj_df=subj_df,
        h2_eligible=h2_eligible,
        wsls_h2=wsls_results.get('wsls_h2'),
        rw_mle=rw_results.get('rw_mle'),
        show_plots=True,
        verbose=True,
    )
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import matplotlib.colors as mcolors
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ── Visualization helpers ────────────────────────────────────────────────────

AGE_MIN, AGE_MAX = 20, 80
_age_cmap = None

def _get_age_cmap():
    global _age_cmap
    if _age_cmap is None:
        _age_cmap = mcolors.LinearSegmentedColormap.from_list(
            'blue_gold', ['#1565C0', '#FFB300'])
    return _age_cmap

def _age_to_rgb(age):
    cmap = _get_age_cmap()
    norm = np.clip((age - AGE_MIN) / (AGE_MAX - AGE_MIN), 0, 1)
    r, g, b, _ = cmap(norm)
    return f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'


def _scatter_with_regression(fig, x, y, age, labels, xlabel, ylabel,
                              title, row=1, col=1, show_legend=False):
    """Add age-colored scatter + regression line to a subplot."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y, age, labels = x[mask], y[mask], age[mask], labels[mask]
    n = len(x)
    if n < 5:
        return None

    slope, intercept, r, p, se = stats.linregress(x, y)

    # Points
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(size=10, opacity=0.8,
                    color=[_age_to_rgb(a) for a in age],
                    line=dict(width=0.5, color='white')),
        text=[f'{lab}<br>Age: {a:.0f}' for lab, a in zip(labels, age)],
        hoverinfo='text+x+y', showlegend=False,
    ), row=row, col=col)

    # Regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = intercept + slope * x_line
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode='lines',
        line=dict(color='#404040', width=2),
        showlegend=False, hoverinfo='skip',
    ), row=row, col=col)

    # Zero line
    fig.add_hline(y=0, line=dict(color='#999999', width=1, dash='dash'),
                  row=row, col=col)

    # Annotation
    sig = '*' if p < .05 else ''
    fig.add_annotation(
        text=f'r = {r:.3f}, p = {p:.3f}{sig}<br>N = {n}',
        x=0.95, y=0.95, xref=f'x{col if col > 1 else ""} domain',
        yref=f'y{row if row > 1 else ""} domain',
        showarrow=False, font=dict(size=11, color='#404040'),
        xanchor='right', yanchor='top',
        bgcolor='rgba(255,255,255,0.8)',
    )

    fig.update_xaxes(title_text=xlabel, row=row, col=col)
    fig.update_yaxes(title_text=ylabel, row=row, col=col)

    return {'r': r, 'p': p, 'n': n, 'slope': slope, 'intercept': intercept}


# ── 1. RT Change Scores ─────────────────────────────────────────────────────

def compute_rt_change(data_clean, h2_eligible, verbose=True):
    """Compute mean RT per subject × condition and test age moderation."""
    df = data_clean[
        data_clean['condition'].isin(['sham', 'active']) &
        data_clean['subject_id'].isin(h2_eligible) &
        data_clean['choice'].notna() &
        data_clean['rt'].notna()
    ].copy()

    # Condition-level mean RT per subject
    rt_cond = (df.groupby(['subject_id', 'condition'])['rt']
               .mean().reset_index().rename(columns={'rt': 'mean_rt'}))

    pivot = rt_cond.pivot(index='subject_id', columns='condition',
                          values='mean_rt').dropna()
    pivot['delta_rt'] = pivot['active'] - pivot['sham']

    # Paired test
    t_stat, p_val = stats.ttest_rel(pivot['active'], pivot['sham'])
    diff = pivot['delta_rt']
    dz = diff.mean() / diff.std() if diff.std() > 0 else 0

    # Detect units: if mean RT > 100, assume milliseconds
    rt_unit = 'ms' if pivot['sham'].mean() > 100 else 's'

    results = {
        'rt_data': pivot.reset_index(),
        'paired_t': t_stat, 'paired_p': p_val, 'dz': dz,
        'sham_mean': pivot['sham'].mean(), 'active_mean': pivot['active'].mean(),
        'n': len(pivot), 'rt_unit': rt_unit,
    }

    if verbose:
        print(f'\n--- 1. RT Change Scores ---')
        print(f'  Sham RT:   M = {pivot["sham"].mean():.1f}{rt_unit}, SD = {pivot["sham"].std():.1f}')
        print(f'  Active RT: M = {pivot["active"].mean():.1f}{rt_unit}, SD = {pivot["active"].std():.1f}')
        print(f'  Δ RT: {diff.mean():+.1f}{rt_unit}, t({len(pivot)-1}) = {t_stat:.3f}, p = {p_val:.3f}, dz = {dz:.3f}')
        if p_val > .05:
            print(f'  → No speed-accuracy tradeoff: RT does not differ between conditions')

    return results


# ── 2. Subgroup Paired Tests ─────────────────────────────────────────────────

def compute_subgroup_tests(subj_df, split_var='age', verbose=True):
    """
    Split sample at median age and run H2.1 paired tests within each subgroup.
    Tests all available DVs: WSLS, RW, accuracy, win rate, DDM drift rate.
    """
    df = subj_df.dropna(subset=[split_var]).copy()
    median_val = df[split_var].median()
    df['age_group'] = np.where(df[split_var] >= median_val, 'older', 'younger')

    # Define all DVs to test
    dv_pairs = [
        ('sham_p_stay_win', 'active_p_stay_win', 'Δ p(stay|win)'),
        ('sham_p_shift_lose', 'active_p_shift_lose', 'Δ p(shift|lose)'),
        ('sham_alpha', 'active_alpha', 'Δ α'),
        ('sham_beta', 'active_beta', 'Δ β'),
        ('sham_accuracy', 'active_accuracy', 'Δ Accuracy'),
        ('sham_win_rate', 'active_win_rate', 'Δ Win Rate'),
    ]

    # Add DDM drift rate if available (handle both naming conventions)
    if 'sham_ddm_v' in df.columns and 'active_ddm_v' in df.columns:
        dv_pairs.append(('sham_ddm_v', 'active_ddm_v', 'Δ Drift Rate'))
    elif 'ddm_v_sham' in df.columns and 'ddm_v_active' in df.columns:
        dv_pairs.append(('ddm_v_sham', 'ddm_v_active', 'Δ Drift Rate'))

    results = {}

    if verbose:
        print(f'\n--- 2. Subgroup Paired Tests (median split at age {median_val:.1f}) ---')

    for group_name in ['younger', 'older']:
        group = df[df['age_group'] == group_name]
        n_group = len(group)
        age_range = f'{group[split_var].min():.0f}–{group[split_var].max():.0f}'

        if verbose:
            print(f'\n  {group_name.upper()} (n = {n_group}, ages {age_range}):')

        group_results = {}
        for sham_col, active_col, label in dv_pairs:
            if sham_col not in group.columns or active_col not in group.columns:
                continue
            paired = group[[sham_col, active_col]].dropna()
            if len(paired) < 3:
                continue

            t_stat, p_val = stats.ttest_rel(paired[active_col], paired[sham_col])
            diff = paired[active_col] - paired[sham_col]
            dz = diff.mean() / diff.std() if diff.std() > 0 else 0

            group_results[label] = {
                't': t_stat, 'p': p_val, 'dz': dz,
                'delta_mean': diff.mean(), 'n': len(paired),
            }

            if verbose:
                sig = '  *' if p_val < .05 else ''
                print(f'    {label:20s}: Δ = {diff.mean():+.4f}, '
                      f't({len(paired)-1}) = {t_stat:+.3f}, p = {p_val:.3f}, '
                      f'dz = {dz:+.3f}{sig}')

        results[group_name] = group_results

    results['median_age'] = median_val
    results['n_younger'] = len(df[df['age_group'] == 'younger'])
    results['n_older'] = len(df[df['age_group'] == 'older'])

    return results


# ── 3. Correct-Stay Decomposition ───────────────────────────────────────────

def compute_correctstay_decomposition(data_clean, h2_eligible, verbose=True):
    """
    Split p(stay|win) into:
      - p(stay|win, good option) — staying with the high-reward option after a win
      - p(stay|win, bad option)  — staying with the low-reward option after a win

    The accuracy benefit should show up as increased correct-staying, not
    increased incorrect-staying.
    """
    df = data_clean[
        data_clean['condition'].isin(['sham', 'active']) &
        data_clean['subject_id'].isin(h2_eligible) &
        data_clean['choice'].notna()
    ].copy()

    # Determine if previous choice was the good option
    # prev_choice matches current_good on the PREVIOUS trial
    df['prev_current_good'] = df.groupby(['subject_id', 'run'])['current_good'].shift(1)
    df['prev_chose_good'] = (df['prev_choice'] == df['prev_current_good'])

    # Filter to valid post-win trials
    post_win = df[df['valid_wsls'] & (df['prev_reward'] == True)].copy()

    results_rows = []
    for (sid, cond), grp in post_win.groupby(['subject_id', 'condition']):
        good_wins = grp[grp['prev_chose_good'] == True]
        bad_wins = grp[grp['prev_chose_good'] == False]

        results_rows.append({
            'subject_id': sid,
            'condition': cond,
            'stay_after_good_win': good_wins['stay'].mean() if len(good_wins) >= 2 else np.nan,
            'stay_after_bad_win': bad_wins['stay'].mean() if len(bad_wins) >= 2 else np.nan,
            'n_good_wins': len(good_wins),
            'n_bad_wins': len(bad_wins),
        })

    cs_df = pd.DataFrame(results_rows)

    # Compute change scores
    pivot_good = cs_df.pivot(index='subject_id', columns='condition',
                              values='stay_after_good_win')
    pivot_bad = cs_df.pivot(index='subject_id', columns='condition',
                             values='stay_after_bad_win')

    change_scores = pd.DataFrame({'subject_id': pivot_good.index})
    if 'sham' in pivot_good.columns and 'active' in pivot_good.columns:
        change_scores['delta_stay_good_win'] = (
            pivot_good['active'] - pivot_good['sham']).values
    if 'sham' in pivot_bad.columns and 'active' in pivot_bad.columns:
        change_scores['delta_stay_bad_win'] = (
            pivot_bad['active'] - pivot_bad['sham']).values

    # Similarly for lose-shift: split by whether previous choice was good
    post_lose = df[df['valid_wsls'] & (df['prev_reward'] == False)].copy()
    lose_rows = []
    for (sid, cond), grp in post_lose.groupby(['subject_id', 'condition']):
        good_losses = grp[grp['prev_chose_good'] == True]
        bad_losses = grp[grp['prev_chose_good'] == False]
        lose_rows.append({
            'subject_id': sid,
            'condition': cond,
            'shift_after_good_loss': good_losses['shift'].mean() if len(good_losses) >= 2 else np.nan,
            'shift_after_bad_loss': bad_losses['shift'].mean() if len(bad_losses) >= 2 else np.nan,
        })
    ls_df = pd.DataFrame(lose_rows)

    pivot_gl = ls_df.pivot(index='subject_id', columns='condition',
                            values='shift_after_good_loss')
    pivot_bl = ls_df.pivot(index='subject_id', columns='condition',
                            values='shift_after_bad_loss')

    if 'sham' in pivot_gl.columns and 'active' in pivot_gl.columns:
        change_scores['delta_shift_good_loss'] = (
            pivot_gl['active'] - pivot_gl['sham']).values
    if 'sham' in pivot_bl.columns and 'active' in pivot_bl.columns:
        change_scores['delta_shift_bad_loss'] = (
            pivot_bl['active'] - pivot_bl['sham']).values

    results = {
        'condition_level': cs_df,
        'change_scores': change_scores,
    }

    if verbose:
        print(f'\n--- 3. Correct-Stay Decomposition ---')
        for metric, label in [
            ('delta_stay_good_win', 'Δ stay after win on GOOD option'),
            ('delta_stay_bad_win', 'Δ stay after win on BAD option'),
            ('delta_shift_good_loss', 'Δ shift after loss on GOOD option'),
            ('delta_shift_bad_loss', 'Δ shift after loss on BAD option'),
        ]:
            if metric in change_scores.columns:
                vals = change_scores[metric].dropna()
                if len(vals) >= 3:
                    t, p = stats.ttest_1samp(vals, 0)
                    sig = '  *' if p < .05 else ''
                    print(f'  {label:50s}: Δ = {vals.mean():+.4f}, '
                          f't({len(vals)-1}) = {t:+.3f}, p = {p:.3f}{sig}')

    return results


# ── 4. Reversal Performance (Age-Moderated) ─────────────────────────────────

def compute_reversal_age_moderation(data_clean, subj_df, h2_eligible,
                                     post_rev_window=(1, 10), verbose=True):
    """
    Compute post-reversal accuracy per subject × condition, then test
    whether age predicts Δ post-reversal accuracy.
    """
    df = data_clean[
        data_clean['condition'].isin(['sham', 'active']) &
        data_clean['subject_id'].isin(h2_eligible) &
        data_clean['choice'].notna()
    ].copy()

    # Check if reversal columns exist; if not, compute them
    if 'trial_from_rev' not in df.columns or 'in_rev_window' not in df.columns:
        if verbose:
            print('\n--- 4. Reversal Age Moderation ---')
            print('  Reversal columns not in data_clean — computing inline...')

        if 'current_good' not in df.columns:
            if verbose:
                print('  current_good column not found. Cannot compute reversals.')
            return None

        # Inline reversal detection
        df['is_reversal'] = False
        df['trial_from_rev'] = np.nan
        df['in_rev_window'] = False
        window_pre, window_post = 5, post_rev_window[1]

        for (sub_id, run_num), group in df.groupby(['subject_id', 'run']):
            idx = group.index
            good = group['current_good'].values
            rev_mask = np.zeros(len(good), dtype=bool)
            if len(good) > 1:
                rev_mask[1:] = good[1:] != good[:-1]
            rev_positions = np.where(rev_mask)[0]

            for rev_pos in rev_positions:
                df.loc[idx[rev_pos], 'is_reversal'] = True
                for offset in range(-window_pre, window_post + 1):
                    t = rev_pos + offset
                    if 0 <= t < len(idx):
                        trial_idx = idx[t]
                        current = df.loc[trial_idx, 'trial_from_rev']
                        if pd.isna(current) or abs(offset) < abs(current):
                            df.loc[trial_idx, 'trial_from_rev'] = offset
                            df.loc[trial_idx, 'in_rev_window'] = True

        n_rev = df['is_reversal'].sum()
        if verbose:
            print(f'  Found {n_rev} reversals')

    post_rev = df[
        df['in_rev_window'] &
        df['trial_from_rev'].between(*post_rev_window)
    ]

    post_rev_acc = (post_rev.groupby(['subject_id', 'condition'])
                    .agg(post_rev_accuracy=('correct', 'mean'),
                         n_trials=('correct', 'count'))
                    .reset_index())

    pivot = post_rev_acc.pivot(index='subject_id', columns='condition',
                                values='post_rev_accuracy')
    if 'sham' not in pivot.columns or 'active' not in pivot.columns:
        return None

    paired = pivot[['sham', 'active']].dropna()
    paired['delta_post_rev'] = paired['active'] - paired['sham']

    # Merge with age (force float to avoid object dtype issues with scipy)
    age_lookup = subj_df.set_index('subject_id')['age'].to_dict()
    paired['age'] = paired.index.map(lambda s: age_lookup.get(str(s), np.nan))
    for c in ['sham', 'active', 'delta_post_rev', 'age']:
        paired[c] = pd.to_numeric(paired[c], errors='coerce')
    paired = paired.dropna(subset=['age', 'delta_post_rev'])

    results = {'post_rev_data': paired.reset_index()}

    if len(paired) >= 5:
        r, p = stats.pearsonr(paired['age'].values, paired['delta_post_rev'].values)
        results['age_r'] = r
        results['age_p'] = p
        results['n'] = len(paired)

    # Paired test on post-rev accuracy
    t, p_pair = stats.ttest_rel(paired['active'].values, paired['sham'].values)
    results['paired_t'] = t
    results['paired_p'] = p_pair
    results['dz'] = (paired['delta_post_rev'].mean() /
                      paired['delta_post_rev'].std()
                      if paired['delta_post_rev'].std() > 0 else 0)

    if verbose:
        print(f'\n--- 4. Reversal Performance (Age-Moderated) ---')
        print(f'  Post-reversal accuracy (trials {post_rev_window[0]}–{post_rev_window[1]}):')
        print(f'    Sham:   M = {paired["sham"].mean():.3f}')
        print(f'    Active: M = {paired["active"].mean():.3f}')
        print(f'    Paired: t({len(paired)-1}) = {t:.3f}, p = {p_pair:.3f}, dz = {results["dz"]:.3f}')
        if 'age_r' in results:
            sig = '  *' if results['age_p'] < .05 else ''
            print(f'    Age × Δ post-rev: r = {results["age_r"]:.3f}, '
                  f'p = {results["age_p"]:.3f}, n = {results["n"]}{sig}')

    return results


# ── 5. Within-Run Learning Slope ─────────────────────────────────────────────

def compute_learning_slope_moderation(data_clean, subj_df, h2_eligible,
                                       n_bins=5, verbose=True):
    """
    Compute per-subject, per-condition within-run learning slopes by
    regressing accuracy on trial bin within each run, then averaging across
    runs. Tests whether age moderates the Δ learning slope.
    """
    df = data_clean[
        data_clean['condition'].isin(['sham', 'active']) &
        data_clean['subject_id'].isin(h2_eligible) &
        data_clean['choice'].notna()
    ].copy()

    # Create within-run trial bins
    df['trial_in_run'] = df.groupby(['subject_id', 'run']).cumcount()
    max_trial = df.groupby(['subject_id', 'run'])['trial_in_run'].transform('max')
    df['trial_bin'] = pd.cut(df['trial_in_run'], bins=n_bins, labels=False)

    # Ensure correct is numeric
    df['correct_num'] = df['correct'].astype(float)

    # Compute per-run learning slope
    slope_rows = []
    for (sid, run_num, cond), grp in df.groupby(['subject_id', 'run', 'condition']):
        bin_acc = grp.groupby('trial_bin')['correct_num'].mean().dropna()
        # Need at least 3 distinct bins with data
        if len(bin_acc) >= 3:
            x = bin_acc.index.values.astype(float)
            y = bin_acc.values.astype(float)
            # linregress needs variance in x
            if np.std(x) == 0:
                continue
            try:
                s, _, _, _, _ = stats.linregress(x, y)
            except Exception:
                continue
            slope_rows.append({
                'subject_id': sid, 'run': run_num, 'condition': cond,
                'learning_slope': s,
            })

    slopes = pd.DataFrame(slope_rows)
    if len(slopes) == 0:
        if verbose:
            print('\n--- 5. Within-Run Learning Slope ---')
            print('  Insufficient data to compute learning slopes.')
        return None

    # Average across runs within each condition
    subj_slopes = (slopes.groupby(['subject_id', 'condition'])['learning_slope']
                   .mean().reset_index())

    pivot = subj_slopes.pivot(index='subject_id', columns='condition',
                               values='learning_slope')
    if 'sham' not in pivot.columns or 'active' not in pivot.columns:
        return None

    paired = pivot[['sham', 'active']].dropna()
    paired['delta_slope'] = paired['active'] - paired['sham']

    # Merge with age (force float to avoid object dtype issues with scipy)
    age_lookup = subj_df.set_index('subject_id')['age'].to_dict()
    paired['age'] = paired.index.map(lambda s: age_lookup.get(str(s), np.nan))
    for c in ['sham', 'active', 'delta_slope', 'age']:
        paired[c] = pd.to_numeric(paired[c], errors='coerce')
    paired = paired.dropna(subset=['age', 'delta_slope'])

    results = {'slope_data': paired.reset_index()}

    # Paired test
    t, p_pair = stats.ttest_rel(paired['active'].values, paired['sham'].values)
    results['paired_t'] = t
    results['paired_p'] = p_pair
    results['dz'] = (paired['delta_slope'].mean() / paired['delta_slope'].std()
                      if paired['delta_slope'].std() > 0 else 0)

    # Age moderation
    if len(paired) >= 5:
        r, p = stats.pearsonr(paired['age'].values, paired['delta_slope'].values)
        results['age_r'] = r
        results['age_p'] = p
        results['n'] = len(paired)

    if verbose:
        print(f'\n--- 5. Within-Run Learning Slope (Age-Moderated) ---')
        print(f'  Mean slope (accuracy gain per bin):')
        print(f'    Sham:   {paired["sham"].mean():+.4f}')
        print(f'    Active: {paired["active"].mean():+.4f}')
        print(f'    Paired: t({len(paired)-1}) = {t:.3f}, p = {p_pair:.3f}, dz = {results["dz"]:.3f}')
        if 'age_r' in results:
            sig = '  *' if results['age_p'] < .05 else ''
            print(f'    Age × Δ slope: r = {results["age_r"]:.3f}, '
                  f'p = {results["age_p"]:.3f}, n = {results["n"]}{sig}')

    return results


# ── 6. DDM Drift Rate (Age-Moderated) ───────────────────────────────────────

def compute_ddm_age_moderation(subj_df, verbose=True):
    """
    Test whether age moderates the tACS effect on DDM drift rate.
    Requires sham_ddm_v and active_ddm_v columns in subj_df.
    """
    # Check for DDM columns (handle various naming conventions)
    sham_v_col = None
    active_v_col = None

    # Convention 1: condition_param (e.g., sham_ddm_v)
    for prefix in ['ddm_v', 'ddm_drift', 'drift_rate']:
        if f'sham_{prefix}' in subj_df.columns:
            sham_v_col = f'sham_{prefix}'
            active_v_col = f'active_{prefix}'
            break

    # Convention 2: param_condition (e.g., ddm_v_sham) — used by ddm.py
    if sham_v_col is None:
        for param in ['ddm_v', 'ddm_drift', 'drift_rate']:
            if f'{param}_sham' in subj_df.columns:
                sham_v_col = f'{param}_sham'
                active_v_col = f'{param}_active'
                break

    # Convention 3: bare names
    if sham_v_col is None:
        if 'sham_v' in subj_df.columns:
            sham_v_col, active_v_col = 'sham_v', 'active_v'

    if sham_v_col is None or sham_v_col not in subj_df.columns:
        if verbose:
            print(f'\n--- 6. DDM Drift Rate (Age-Moderated) ---')
            print(f'  DDM drift rate columns not found in subj_df.')
            ddm_cols = [c for c in subj_df.columns if 'ddm' in c.lower() or 'drift' in c.lower()]
            if ddm_cols:
                print(f'  Available DDM-related columns: {ddm_cols}')
            print(f'  Run DDM fitting with REFIT_DDM = True first.')
        return None

    df = subj_df[['subject_id', 'age', sham_v_col, active_v_col]].dropna().copy()
    df['delta_v'] = df[active_v_col] - df[sham_v_col]

    results = {'drift_data': df}

    # Paired test
    t, p = stats.ttest_rel(df[active_v_col], df[sham_v_col])
    diff = df['delta_v']
    dz = diff.mean() / diff.std() if diff.std() > 0 else 0
    results['paired_t'] = t
    results['paired_p'] = p
    results['dz'] = dz

    # Age moderation
    if len(df) >= 5:
        r, p_age = stats.pearsonr(df['age'], df['delta_v'])
        results['age_r'] = r
        results['age_p'] = p_age
        results['n'] = len(df)

    if verbose:
        print(f'\n--- 6. DDM Drift Rate (Age-Moderated) ---')
        print(f'  Sham v:   M = {df[sham_v_col].mean():.3f}')
        print(f'  Active v: M = {df[active_v_col].mean():.3f}')
        print(f'  Paired: t({len(df)-1}) = {t:.3f}, p = {p:.3f}, dz = {dz:.3f}')
        if 'age_r' in results:
            sig = '  *' if results['age_p'] < .05 else ''
            print(f'  Age × Δv: r = {results["age_r"]:.3f}, '
                  f'p = {results["age_p"]:.3f}, n = {results["n"]}{sig}')

    return results


# ── Age Moderation Regression (shared utility) ──────────────────────────────

def _run_age_moderation_regression(subj_df, delta_col, label, verbose=True):
    """OLS: delta_col ~ age + global_composite."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return None

    cols = ['age', 'global_composite', delta_col]
    df = subj_df[['subject_id'] + cols].dropna()
    if len(df) < 6:
        return None

    X = sm.add_constant(df[['age', 'global_composite']])
    y = df[delta_col]
    model = sm.OLS(y, X).fit()

    result = {
        'r2': model.rsquared,
        'f': model.fvalue,
        'f_p': model.f_pvalue,
        'age_b': model.params['age'],
        'age_p': model.pvalues['age'],
        'cog_b': model.params['global_composite'],
        'cog_p': model.pvalues['global_composite'],
        'n': len(df),
    }

    if verbose:
        sig_a = '*' if result['age_p'] < .05 else ''
        sig_c = '*' if result['cog_p'] < .05 else ''
        print(f'    OLS {label} ~ age + cog (n={result["n"]}): '
              f'R²={result["r2"]:.3f}, '
              f'age b={result["age_b"]:+.4f} p={result["age_p"]:.3f}{sig_a}, '
              f'cog b={result["cog_b"]:+.4f} p={result["cog_p"]:.3f}{sig_c}')

    return result


# ── Summary Plot ─────────────────────────────────────────────────────────────

def _make_summary_plot(subj_df, all_results):
    """Create a multi-panel figure summarizing all age × Δ relationships."""
    if not HAS_PLOTLY:
        return None

    # Collect all plottable age × delta pairs
    panels = []

    # Accuracy and win rate (always available from subj_df)
    for col, label in [('delta_accuracy', 'Δ Accuracy'),
                        ('delta_win_rate', 'Δ Win Rate'),
                        ('delta_rt', 'Δ RT (s)')]:
        if col in subj_df.columns:
            panels.append((col, label))

    # WSLS
    for col, label in [('delta_p_stay_win', 'Δ p(stay|win)'),
                        ('delta_p_shift_lose', 'Δ p(shift|lose)')]:
        if col in subj_df.columns:
            panels.append((col, label))

    # RW
    for col, label in [('delta_alpha', 'Δ α'), ('delta_beta', 'Δ β')]:
        if col in subj_df.columns:
            panels.append((col, label))

    if len(panels) == 0:
        return None

    n_cols = min(3, len(panels))
    n_rows = int(np.ceil(len(panels) / n_cols))

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[p[1] + ' ~ Age' for p in panels],
        horizontal_spacing=0.10, vertical_spacing=0.12,
    )

    for i, (col, label) in enumerate(panels):
        row = i // n_cols + 1
        c = i % n_cols + 1
        df_plot = subj_df[['subject_id', 'age', col]].dropna()
        if len(df_plot) < 5:
            continue
        _scatter_with_regression(
            fig,
            x=df_plot['age'].values,
            y=df_plot[col].values,
            age=df_plot['age'].values,
            labels=df_plot['subject_id'].values,
            xlabel='Age', ylabel=label,
            title='', row=row, col=c,
        )

    fig.update_layout(
        height=350 * n_rows, width=min(950, 320 * n_cols),
        template='plotly_white',
        font=dict(family='Arial', size=12),
        margin=dict(l=60, r=40, t=50, b=50),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='black')

    return fig


# ── Master Runner ────────────────────────────────────────────────────────────

def run_age_moderated_stimulation(
    data_clean,
    subj_df,
    h2_eligible,
    wsls_h2=None,
    rw_mle=None,
    show_plots=True,
    verbose=True,
):
    """
    Run all six analyses decomposing the age-moderated stimulation effect.

    Parameters
    ----------
    data_clean : DataFrame
        Trial-level data with behavioral exclusions applied.
    subj_df : DataFrame
        Subject-level master DataFrame (from cognitive_merge).
    h2_eligible : list
        Subject IDs eligible for H2 paired comparisons.
    wsls_h2 : DataFrame, optional
        WSLS results from wsls module.
    rw_mle : DataFrame, optional
        RW parameters from rescorla_wagner module.
    show_plots : bool
        Display summary Plotly figure.
    verbose : bool
        Print results to console.

    Returns
    -------
    dict with keys:
        'rt_change', 'subgroup_tests', 'correctstay', 'reversal_moderation',
        'learning_slope', 'ddm_moderation', 'summary_fig',
        'age_regressions' (OLS results for each Δ DV)
    """
    print('=' * 70)
    print('AGE-MODERATED STIMULATION EFFECTS')
    print('Decomposing the accuracy finding: what drives better performance')
    print('in older adults under active theta-tACS?')
    print('=' * 70)

    results = {}

    # Ensure h2_eligible are strings
    h2_eligible_str = [str(s) for s in h2_eligible]

    # ── 1. RT ────────────────────────────────────────────────────────────
    results['rt_change'] = compute_rt_change(data_clean, h2_eligible_str, verbose)

    # Merge delta_rt into subj_df for downstream use
    if results['rt_change'] is not None:
        rt_df = results['rt_change']['rt_data'][['subject_id', 'delta_rt']].copy()
        rt_df['subject_id'] = rt_df['subject_id'].astype(str)
        if 'delta_rt' not in subj_df.columns:
            subj_df = subj_df.merge(rt_df, on='subject_id', how='left')
        else:
            subj_df['delta_rt'] = subj_df['subject_id'].map(
                rt_df.set_index('subject_id')['delta_rt'])

    # ── 2. Subgroup ──────────────────────────────────────────────────────
    results['subgroup_tests'] = compute_subgroup_tests(subj_df, verbose=verbose)

    # ── 3. Correct-stay ──────────────────────────────────────────────────
    results['correctstay'] = compute_correctstay_decomposition(
        data_clean, h2_eligible_str, verbose)

    # Merge correct-stay change scores into subj_df
    if results['correctstay'] is not None:
        cs = results['correctstay']['change_scores'].copy()
        cs['subject_id'] = cs['subject_id'].astype(str)
        for col in cs.columns:
            if col != 'subject_id' and col not in subj_df.columns:
                subj_df = subj_df.merge(
                    cs[['subject_id', col]], on='subject_id', how='left')

    # ── 4. Reversal ──────────────────────────────────────────────────────
    results['reversal_moderation'] = compute_reversal_age_moderation(
        data_clean, subj_df, h2_eligible_str, verbose=verbose)

    # ── 5. Learning slope ────────────────────────────────────────────────
    results['learning_slope'] = compute_learning_slope_moderation(
        data_clean, subj_df, h2_eligible_str, verbose=verbose)

    # ── 6. DDM ───────────────────────────────────────────────────────────
    results['ddm_moderation'] = compute_ddm_age_moderation(subj_df, verbose)

    # Merge delta_v into subj_df if DDM was computed
    if results['ddm_moderation'] is not None and 'drift_data' in results['ddm_moderation']:
        dv_df = results['ddm_moderation']['drift_data']
        if 'delta_v' in dv_df.columns and 'delta_ddm_v' not in subj_df.columns:
            merge_v = dv_df[['subject_id', 'delta_v']].copy()
            merge_v['subject_id'] = merge_v['subject_id'].astype(str)
            merge_v = merge_v.rename(columns={'delta_v': 'delta_ddm_v'})
            subj_df = subj_df.merge(merge_v, on='subject_id', how='left')

    # ── Age moderation regressions (OLS) ─────────────────────────────────
    if verbose:
        print(f'\n--- Age Moderation Regressions (OLS: Δ ~ age + global_cog) ---')

    age_regs = {}
    delta_dvs = [
        ('delta_accuracy', 'Δ Accuracy'),
        ('delta_win_rate', 'Δ Win Rate'),
        ('delta_p_stay_win', 'Δ p(stay|win)'),
        ('delta_p_shift_lose', 'Δ p(shift|lose)'),
        ('delta_alpha', 'Δ α'),
        ('delta_beta', 'Δ β'),
        ('delta_ddm_v', 'Δ Drift Rate'),
        ('delta_stay_good_win', 'Δ stay|win (good)'),
        ('delta_stay_bad_win', 'Δ stay|win (bad)'),
        ('delta_shift_good_loss', 'Δ shift|lose (good)'),
        ('delta_shift_bad_loss', 'Δ shift|lose (bad)'),
    ]

    # Add RT if available
    if 'delta_rt' in subj_df.columns:
        delta_dvs.insert(2, ('delta_rt', 'Δ RT'))

    for col, label in delta_dvs:
        if col in subj_df.columns and subj_df[col].notna().sum() >= 6:
            reg = _run_age_moderation_regression(subj_df, col, label, verbose)
            if reg is not None:
                age_regs[col] = reg

    results['age_regressions'] = age_regs

    # ── Summary plot ─────────────────────────────────────────────────────
    if show_plots and HAS_PLOTLY:
        fig = _make_summary_plot(subj_df, results)
        if fig is not None:
            fig.show()
            results['summary_fig'] = fig

    # ── Narrative summary ────────────────────────────────────────────────
    if verbose:
        print(f'\n{"=" * 70}')
        print('NARRATIVE SUMMARY')
        print('=' * 70)

        # RT
        rt = results.get('rt_change')
        if rt and rt['paired_p'] > .05:
            print(f'  ✓ No speed-accuracy tradeoff (Δ RT = {rt["dz"]:+.3f}, p = {rt["paired_p"]:.3f})')
        elif rt:
            print(f'  ⚠ RT differs between conditions (dz = {rt["dz"]:+.3f}, p = {rt["paired_p"]:.3f})')

        # Subgroup
        sg = results.get('subgroup_tests', {})
        older = sg.get('older', {})
        younger = sg.get('younger', {})
        sig_older = [k for k, v in older.items() if v['p'] < .05]
        sig_younger = [k for k, v in younger.items() if v['p'] < .05]
        if sig_older:
            print(f'  ✓ Older subgroup significant: {", ".join(sig_older)}')
        else:
            print(f'  – No significant effects in older subgroup alone')
        if sig_younger:
            print(f'  ⚠ Younger subgroup also significant: {", ".join(sig_younger)}')

        # Age regressions
        sig_regs = {k: v for k, v in age_regs.items() if v['age_p'] < .05}
        if sig_regs:
            print(f'  ✓ Age significantly predicts: {", ".join(sig_regs.keys())}')

    # Store modified subj_df back
    results['subj_df'] = subj_df

    return results
