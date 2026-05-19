"""
Diagnostic: Does lower lose-shifting in young high-cognition adults
reflect strategic behavior (higher accuracy) or perseveration (lower accuracy)?

Run after subj_df is assembled in the notebook.
"""

import pandas as pd
import numpy as np
from scipy import stats

def diagnose_young_loseshift_accuracy(subj_df, age_cutoff=33.1):
    """
    Test whether lower lose-shifting in young adults is adaptive or maladaptive
    by checking its relationship with accuracy.
    
    If lower lose-shifting → higher accuracy: strategic (optimal)
    If lower lose-shifting → lower accuracy: perseverative (rigid)
    """
    df = subj_df.copy()
    for col in ['age', 'sham_p_shift_lose', 'sham_accuracy', 'sham_win_rate',
                'global_composite', 'sham_p_stay_win']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    young = df[df['age'] <= age_cutoff].dropna(subset=['sham_p_shift_lose', 'sham_accuracy'])
    old = df[df['age'] > age_cutoff].dropna(subset=['sham_p_shift_lose', 'sham_accuracy'])

    print('=' * 70)
    print(f'LOSE-SHIFT × ACCURACY DIAGNOSTIC (J-N cutoff = {age_cutoff})')
    print('=' * 70)
    print(f'\n  Young (≤{age_cutoff}): n = {len(young)}, ages {young["age"].min():.0f}–{young["age"].max():.0f}')
    print(f'  Older (>{age_cutoff}):  n = {len(old)}, ages {old["age"].min():.0f}–{old["age"].max():.0f}')

    # ── Test 1: Lose-shift × Accuracy within each group ──
    print(f'\n{"─" * 50}')
    print('Test 1: p(shift|lose) × Accuracy')
    print(f'{"─" * 50}')

    for label, group in [('Young', young), ('Older', old)]:
        if len(group) < 4:
            print(f'\n  {label}: insufficient data (n={len(group)})')
            continue
        r, p = stats.pearsonr(group['sham_p_shift_lose'], group['sham_accuracy'])
        print(f'\n  {label} (n={len(group)}):')
        print(f'    r(shift, accuracy) = {r:+.3f}, p = {p:.4f}')
        if r > 0:
            print(f'    → More shifting = higher accuracy (shifting is adaptive)')
        else:
            print(f'    → Less shifting = higher accuracy (low shifting is strategic)')

        # Also check win rate
        if 'sham_win_rate' in group.columns:
            valid = group.dropna(subset=['sham_win_rate'])
            r_wr, p_wr = stats.pearsonr(valid['sham_p_shift_lose'], valid['sham_win_rate'])
            print(f'    r(shift, win_rate) = {r_wr:+.3f}, p = {p_wr:.4f}')

    # ── Test 2: Cognition × Accuracy within each group ──
    print(f'\n{"─" * 50}')
    print('Test 2: Global Cognition × Accuracy')
    print(f'{"─" * 50}')

    for label, group in [('Young', young), ('Older', old)]:
        valid = group.dropna(subset=['global_composite'])
        if len(valid) < 4:
            print(f'\n  {label}: insufficient data')
            continue
        r, p = stats.pearsonr(valid['global_composite'], valid['sham_accuracy'])
        print(f'\n  {label} (n={len(valid)}):')
        print(f'    r(cognition, accuracy) = {r:+.3f}, p = {p:.4f}')

        # Cognition × win rate
        if 'sham_win_rate' in valid.columns:
            r_wr, p_wr = stats.pearsonr(valid['global_composite'], valid['sham_win_rate'])
            print(f'    r(cognition, win_rate) = {r_wr:+.3f}, p = {p_wr:.4f}')

    # ── Test 3: Cognition × Lose-shift within each group (replicating J-N) ──
    print(f'\n{"─" * 50}')
    print('Test 3: Global Cognition × p(shift|lose) [replicating J-N split]')
    print(f'{"─" * 50}')

    for label, group in [('Young', young), ('Older', old)]:
        valid = group.dropna(subset=['global_composite'])
        if len(valid) < 4:
            print(f'\n  {label}: insufficient data')
            continue
        r, p = stats.pearsonr(valid['global_composite'], valid['sham_p_shift_lose'])
        print(f'\n  {label} (n={len(valid)}):')
        print(f'    r(cognition, shift) = {r:+.3f}, p = {p:.4f}')

    # ── Test 4: Descriptive comparison ──
    print(f'\n{"─" * 50}')
    print('Descriptive Comparison: Young vs Older')
    print(f'{"─" * 50}')

    for var, label in [('sham_p_shift_lose', 'p(shift|lose)'),
                       ('sham_accuracy', 'Accuracy'),
                       ('sham_win_rate', 'Win Rate'),
                       ('sham_p_stay_win', 'p(stay|win)'),
                       ('global_composite', 'Global Cognition')]:
        if var not in df.columns:
            continue
        y_vals = young[var].dropna()
        o_vals = old[var].dropna()
        if len(y_vals) > 1 and len(o_vals) > 1:
            t, p = stats.ttest_ind(y_vals, o_vals)
            print(f'\n  {label}:')
            print(f'    Young: M = {y_vals.mean():.3f}, SD = {y_vals.std():.3f} (n={len(y_vals)})')
            print(f'    Older: M = {o_vals.mean():.3f}, SD = {o_vals.std():.3f} (n={len(o_vals)})')
            print(f'    t = {t:.2f}, p = {p:.4f}')

    # ── Test 5: Partial correlation ──
    # Within young: does cognition predict accuracy AFTER controlling for lose-shift?
    print(f'\n{"─" * 50}')
    print('Test 5: Does cognition → accuracy hold after controlling for lose-shift?')
    print(f'{"─" * 50}')

    for label, group in [('Young', young), ('Older', old)]:
        valid = group.dropna(subset=['global_composite', 'sham_accuracy', 'sham_p_shift_lose'])
        if len(valid) < 5:
            print(f'\n  {label}: insufficient data')
            continue

        # Partial correlation: cognition × accuracy | lose-shift
        # Residualize both on lose-shift
        from scipy.stats import linregress
        _, _, _, _, _ = linregress(valid['sham_p_shift_lose'], valid['global_composite'])
        resid_cog = valid['global_composite'] - (linregress(valid['sham_p_shift_lose'], valid['global_composite']).intercept +
                     linregress(valid['sham_p_shift_lose'], valid['global_composite']).slope * valid['sham_p_shift_lose'])
        resid_acc = valid['sham_accuracy'] - (linregress(valid['sham_p_shift_lose'], valid['sham_accuracy']).intercept +
                     linregress(valid['sham_p_shift_lose'], valid['sham_accuracy']).slope * valid['sham_p_shift_lose'])

        r_partial, p_partial = stats.pearsonr(resid_cog, resid_acc)
        print(f'\n  {label} (n={len(valid)}):')
        print(f'    Partial r(cognition, accuracy | lose-shift) = {r_partial:+.3f}, p = {p_partial:.4f}')

    # ── Summary ──
    print(f'\n{"=" * 70}')
    print('INTERPRETATION GUIDE')
    print(f'{"=" * 70}')
    print("""
    If in young adults:
      r(shift, accuracy) < 0  → less shifting = better performance = STRATEGIC
      r(shift, accuracy) > 0  → less shifting = worse performance = PERSEVERATIVE
    
    If r(cognition, accuracy) > 0 in young → high cognition = better performance
    Combined with r(cognition, shift) < 0 → high cognition = less shift = better performance
    → Full chain supports STRATEGIC interpretation
    """)


if __name__ == '__main__':
    print("Run this after subj_df is assembled:")
    print("  diagnose_young_loseshift_accuracy(subj_df)")
