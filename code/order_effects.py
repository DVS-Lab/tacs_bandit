"""
order_effects.py — Counterbalance and session effect analyses for tACS Bandit study

Examines potential confounds in the crossover design:

1. Practice/fatigue effects — Session 1 vs. Session 2 (regardless of condition)
2. Carryover effects — Lingering effects of active tACS on subsequent sham
3. Order × Condition interactions — Differential tACS effects by counterbalance

If order significantly moderates stimulation effects, this informs interpretation
of H2.1 results.

Counterbalance structure:
- Order A: Active first (runs 2-3), then Sham (runs 6-7)
- Order B: Sham first (runs 2-3), then Active (runs 6-7)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, List, Tuple

from config import (
    SUBJECT_INFO,
    PLOTLY_TEMPLATE,
    FONT_FAMILY,
)


# =============================================================================
# Counterbalance Assignment
# =============================================================================

def add_counterbalance(
    subj_df: pd.DataFrame,
    subject_info: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Add counterbalance column to subject DataFrame.
    
    Parameters
    ----------
    subj_df : DataFrame
        Subject-level data
    subject_info : dict, optional
        SUBJECT_INFO dict from config
    
    Returns
    -------
    DataFrame with 'counterbalance' column added
    """
    if subject_info is None:
        subject_info = SUBJECT_INFO
    
    df = subj_df.copy()
    
    if 'counterbalance' not in df.columns:
        cb_map = {str(sid): info.get('counterbalance', 'UNKNOWN')
                  for sid, info in subject_info.items()}
        df['counterbalance'] = df['subject_id'].astype(str).map(cb_map)
    
    return df


def get_counterbalance_summary(
    subj_df: pd.DataFrame,
    verbose: bool = True
) -> Dict:
    """
    Summarize sample by counterbalance order.
    
    Parameters
    ----------
    subj_df : DataFrame
        Subject-level data with 'counterbalance' column
    verbose : bool
        If True, print summary
    
    Returns
    -------
    dict with counts by order
    """
    df = add_counterbalance(subj_df)
    
    counts = df['counterbalance'].value_counts().to_dict()
    
    if verbose:
        print('Sample by Counterbalance Order:')
        print('-'*40)
        for order in ['A', 'B']:
            n = counts.get(order, 0)
            session1 = 'Active' if order == 'A' else 'Sham'
            print(f'  Order {order} ({session1} first): n = {n}')
    
    return counts


# =============================================================================
# Order × Condition Interaction
# =============================================================================

def test_order_interaction(
    sham_col: str,
    active_col: str,
    data: pd.DataFrame,
    label: str,
    verbose: bool = True
) -> Optional[Dict]:
    """
    Test Order × Condition interaction using difference score approach.
    
    If the tACS effect (active - sham) differs by counterbalance order,
    the interaction is significant.
    
    Parameters
    ----------
    sham_col : str
        Sham condition column
    active_col : str
        Active condition column
    data : DataFrame
        Subject-level data with 'counterbalance' column
    label : str
        Label for output
    verbose : bool
        If True, print results
    
    Returns
    -------
    dict with t, p, Cohen's d, means by order
    """
    df = add_counterbalance(data)
    df = df[[sham_col, active_col, 'counterbalance']].dropna()
    df['diff'] = df[active_col] - df[sham_col]
    
    order_a = df[df['counterbalance'] == 'A']['diff']
    order_b = df[df['counterbalance'] == 'B']['diff']
    
    if len(order_a) < 2 or len(order_b) < 2:
        if verbose:
            print(f'  {label}: Insufficient data (A: n={len(order_a)}, B: n={len(order_b)})')
        return None
    
    # Independent samples t-test on difference scores
    t_stat, p_val = stats.ttest_ind(order_a, order_b)
    
    # Cohen's d
    pooled_std = np.sqrt(
        ((len(order_a)-1) * order_a.std()**2 + (len(order_b)-1) * order_b.std()**2) /
        (len(order_a) + len(order_b) - 2)
    )
    cohens_d = (order_a.mean() - order_b.mean()) / pooled_std if pooled_std > 0 else 0
    
    result = {
        't': t_stat,
        'p': p_val,
        'd': cohens_d,
        'mean_A': order_a.mean(),
        'sd_A': order_a.std(),
        'n_A': len(order_a),
        'mean_B': order_b.mean(),
        'sd_B': order_b.std(),
        'n_B': len(order_b),
    }
    
    if verbose:
        print(f'  {label}:')
        print(f'    Order A (Active first): Δ = {result["mean_A"]:+.4f} (SD = {result["sd_A"]:.4f}, n = {result["n_A"]})')
        print(f'    Order B (Sham first):   Δ = {result["mean_B"]:+.4f} (SD = {result["sd_B"]:.4f}, n = {result["n_B"]})')
        print(f'    Interaction: t({result["n_A"]+result["n_B"]-2}) = {t_stat:.3f}, p = {p_val:.4f}, d = {cohens_d:.3f}')
        
        if p_val < 0.05:
            print(f'    ⚠ Significant order effect — interpret H2.1 with caution')
        else:
            print(f'    ✓ No significant order × condition interaction')
        print()
    
    return result


def test_all_order_interactions(
    subj_df: pd.DataFrame,
    verbose: bool = True
) -> Dict:
    """
    Test Order × Condition interaction for all DVs.
    
    Parameters
    ----------
    subj_df : DataFrame
        Subject-level data
    verbose : bool
        If True, print results
    
    Returns
    -------
    dict with results for each DV
    """
    if verbose:
        print('='*70)
        print('Order × Condition Interaction Test')
        print('='*70)
        print()
        print('Testing whether tACS effect magnitude differs by counterbalance order.')
        print('Significant interaction suggests order-dependent stimulation efficacy.')
        print()
    
    results = {}
    
    dvs = [
        ('sham_p_stay_win', 'active_p_stay_win', 'p(stay|win)'),
        ('sham_p_shift_lose', 'active_p_shift_lose', 'p(shift|lose)'),
        ('sham_alpha', 'active_alpha', 'α (learning rate)'),
        ('sham_beta', 'active_beta', 'β (inv. temperature)'),
    ]
    
    for sham_col, active_col, label in dvs:
        if sham_col in subj_df.columns and active_col in subj_df.columns:
            results[label] = test_order_interaction(sham_col, active_col, subj_df, label, verbose=verbose)
    
    return results


# =============================================================================
# Session Effects (Practice/Fatigue)
# =============================================================================

def create_session_columns(
    subj_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create session-based columns (Session 1 vs Session 2).
    
    Session assignment depends on counterbalance:
    - Order A: Session 1 = active, Session 2 = sham
    - Order B: Session 1 = sham, Session 2 = active
    
    Parameters
    ----------
    subj_df : DataFrame
        Subject-level data with counterbalance and condition columns
    
    Returns
    -------
    DataFrame with session1_* and session2_* columns added
    """
    df = add_counterbalance(subj_df).copy()
    
    dvs = ['p_stay_win', 'p_shift_lose', 'alpha', 'beta']
    
    for dv in dvs:
        sham_col = f'sham_{dv}'
        active_col = f'active_{dv}'
        
        if sham_col in df.columns and active_col in df.columns:
            # Session 1: active for A, sham for B
            df[f'session1_{dv}'] = df.apply(
                lambda r: r[active_col] if r['counterbalance'] == 'A' else r[sham_col],
                axis=1
            )
            # Session 2: sham for A, active for B
            df[f'session2_{dv}'] = df.apply(
                lambda r: r[sham_col] if r['counterbalance'] == 'A' else r[active_col],
                axis=1
            )
    
    return df


def test_session_effect(
    session1_col: str,
    session2_col: str,
    data: pd.DataFrame,
    label: str,
    verbose: bool = True
) -> Optional[Dict]:
    """
    Test Session 1 vs. Session 2 difference (practice/fatigue).
    
    Parameters
    ----------
    session1_col : str
        Session 1 column
    session2_col : str
        Session 2 column
    data : DataFrame
        Subject-level data
    label : str
        Label for output
    verbose : bool
        If True, print results
    
    Returns
    -------
    dict with paired t-test results
    """
    df = data[[session1_col, session2_col]].dropna()
    n = len(df)
    
    if n < 3:
        if verbose:
            print(f'  {label}: Insufficient data (n={n})')
        return None
    
    s1 = df[session1_col]
    s2 = df[session2_col]
    diff = s2 - s1
    
    t_stat, p_val = stats.ttest_rel(s2, s1)
    dz = diff.mean() / diff.std() if diff.std() > 0 else 0
    
    result = {
        'n': n,
        't': t_stat,
        'p': p_val,
        'dz': dz,
        'session1_mean': s1.mean(),
        'session1_sd': s1.std(),
        'session2_mean': s2.mean(),
        'session2_sd': s2.std(),
        'diff_mean': diff.mean(),
        'diff_sd': diff.std(),
    }
    
    if verbose:
        print(f'  {label}:')
        print(f'    Session 1: M = {result["session1_mean"]:.3f}, SD = {result["session1_sd"]:.3f}')
        print(f'    Session 2: M = {result["session2_mean"]:.3f}, SD = {result["session2_sd"]:.3f}')
        print(f'    Diff:      M = {result["diff_mean"]:+.4f}, SD = {result["diff_sd"]:.4f}')
        print(f"    t({n-1}) = {t_stat:.3f}, p = {p_val:.4f}, dz = {dz:.3f}")
        
        if p_val < 0.05:
            direction = 'improvement' if diff.mean() > 0 else 'decline'
            print(f'    ⚠ Significant session effect ({direction})')
        else:
            print(f'    ✓ No significant practice/fatigue effect')
        print()
    
    return result


def test_all_session_effects(
    subj_df: pd.DataFrame,
    verbose: bool = True
) -> Dict:
    """
    Test Session 1 vs. Session 2 for all DVs.
    
    Parameters
    ----------
    subj_df : DataFrame
        Subject-level data
    verbose : bool
        If True, print results
    
    Returns
    -------
    dict with results for each DV
    """
    if verbose:
        print('='*70)
        print('Session Effects (Practice/Fatigue Check)')
        print('='*70)
        print()
        print('Comparing Session 1 vs Session 2, collapsing across condition.')
        print('Significant effects suggest practice or fatigue independent of tACS.')
        print()
    
    # Create session columns
    session_df = create_session_columns(subj_df)
    
    results = {}
    
    dvs = [
        ('session1_p_stay_win', 'session2_p_stay_win', 'p(stay|win)'),
        ('session1_p_shift_lose', 'session2_p_shift_lose', 'p(shift|lose)'),
        ('session1_alpha', 'session2_alpha', 'α (learning rate)'),
        ('session1_beta', 'session2_beta', 'β (inv. temperature)'),
    ]
    
    for s1_col, s2_col, label in dvs:
        if s1_col in session_df.columns and s2_col in session_df.columns:
            results[label] = test_session_effect(s1_col, s2_col, session_df, label, verbose=verbose)
    
    return results


# =============================================================================
# Descriptive Statistics by Order
# =============================================================================

def descriptives_by_order(
    subj_df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute descriptive statistics by counterbalance order and condition.
    
    Parameters
    ----------
    subj_df : DataFrame
        Subject-level data
    verbose : bool
        If True, print results
    
    Returns
    -------
    DataFrame with descriptives
    """
    df = add_counterbalance(subj_df)
    
    results = []
    
    dvs = [
        ('sham_p_stay_win', 'Sham', 'p(stay|win)'),
        ('active_p_stay_win', 'Active', 'p(stay|win)'),
        ('sham_p_shift_lose', 'Sham', 'p(shift|lose)'),
        ('active_p_shift_lose', 'Active', 'p(shift|lose)'),
        ('sham_alpha', 'Sham', 'α'),
        ('active_alpha', 'Active', 'α'),
        ('sham_beta', 'Sham', 'β'),
        ('active_beta', 'Active', 'β'),
    ]
    
    for col, condition, param in dvs:
        if col in df.columns:
            for order in ['A', 'B']:
                vals = df[df['counterbalance'] == order][col].dropna()
                if len(vals) > 0:
                    results.append({
                        'Parameter': param,
                        'Condition': condition,
                        'Order': order,
                        'N': len(vals),
                        'Mean': vals.mean(),
                        'SD': vals.std(),
                    })
    
    desc_df = pd.DataFrame(results)
    
    if verbose and len(desc_df) > 0:
        print('WSLS/RW Parameters by Order and Condition:')
        print('-'*70)
        
        for param in desc_df['Parameter'].unique():
            print(f'\n{param}:')
            param_data = desc_df[desc_df['Parameter'] == param]
            for _, row in param_data.iterrows():
                print(f"  Order {row['Order']} / {row['Condition']:6s}: "
                      f"M = {row['Mean']:.3f}, SD = {row['SD']:.3f} (n={row['N']})")
    
    return desc_df


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_order_effects_analysis(
    subj_df: pd.DataFrame,
    verbose: bool = True
) -> Dict:
    """
    Run complete order effects analysis.
    
    Parameters
    ----------
    subj_df : DataFrame
        Subject-level data
    verbose : bool
        If True, print summaries
    
    Returns
    -------
    dict with keys:
        'counterbalance_counts': sample by order
        'descriptives': descriptive stats by order × condition
        'order_interactions': Order × Condition interaction tests
        'session_effects': Session 1 vs 2 tests
        'any_significant_order': bool indicating if any order effects found
    """
    results = {}
    
    # Counterbalance summary
    results['counterbalance_counts'] = get_counterbalance_summary(subj_df, verbose=verbose)
    if verbose:
        print()
    
    # Descriptives by order
    results['descriptives'] = descriptives_by_order(subj_df, verbose=verbose)
    if verbose:
        print()
    
    # Order × Condition interactions
    results['order_interactions'] = test_all_order_interactions(subj_df, verbose=verbose)
    
    # Session effects
    results['session_effects'] = test_all_session_effects(subj_df, verbose=verbose)
    
    # Summary: any significant effects?
    any_order = False
    any_session = False
    
    for label, res in results['order_interactions'].items():
        if res is not None and res['p'] < 0.05:
            any_order = True
            break
    
    for label, res in results['session_effects'].items():
        if res is not None and res['p'] < 0.05:
            any_session = True
            break
    
    results['any_significant_order'] = any_order
    results['any_significant_session'] = any_session
    
    if verbose:
        print('='*70)
        print('Summary')
        print('='*70)
        if any_order:
            print('  ⚠ Significant Order × Condition interaction(s) detected')
            print('    → Interpret H2.1 main effects with caution')
        else:
            print('  ✓ No significant Order × Condition interactions')
        
        if any_session:
            print('  ⚠ Significant session (practice/fatigue) effect(s) detected')
        else:
            print('  ✓ No significant session effects')
    
    return results


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing order_effects module...")
    print("Functions: test_order_interaction, test_session_effect")
    print("           test_all_order_interactions, test_all_session_effects")
    print("           descriptives_by_order, run_order_effects_analysis")
