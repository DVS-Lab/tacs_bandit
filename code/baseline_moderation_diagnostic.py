"""
baseline_moderation_diagnostic.py — Explore baseline-dependent tACS effects

Tests whether baseline performance moderates stimulation efficacy,
and controls for regression to the mean.

Usage:
    from baseline_moderation_diagnostic import run_baseline_moderation_diagnostic
    results = run_baseline_moderation_diagnostic(subj_df, show_plots=True)
"""

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors

from config import (
    AGE_MIN,
    AGE_MAX,
    AGE_COLORSCALE,
    PLOTLY_TEMPLATE,
    FONT_FAMILY,
)


# =============================================================================
# Age color utilities
# =============================================================================

def _get_age_colormap():
    return mcolors.LinearSegmentedColormap.from_list('blue_gold', ['#1565C0', '#FFB300'])

def _age_to_rgb(age: float) -> str:
    cmap = _get_age_colormap()
    norm = np.clip((age - AGE_MIN) / (AGE_MAX - AGE_MIN), 0, 1)
    rgba = cmap(norm)
    return f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'


# =============================================================================
# Test 1: Convergence across accuracy and win rate
# =============================================================================

def test_convergence(subj_df, show_plots=True):
    """Check whether baseline-dependence holds for both accuracy and win rate."""
    df = subj_df.copy()
    for col in ['run1_accuracy', 'run1_win_rate', 'delta_accuracy', 'delta_win_rate',
                'sham_accuracy', 'sham_win_rate', 'age']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print('=' * 70)
    print('TEST 1: CONVERGENCE ACROSS PERFORMANCE MEASURES')
    print('=' * 70)

    results = {}

    # Test with Run-1 as baseline
    pairs = [
        ('run1_accuracy', 'delta_accuracy', 'Run-1 Accuracy', 'Δ Accuracy'),
        ('run1_accuracy', 'delta_win_rate', 'Run-1 Accuracy', 'Δ Win Rate'),
        ('run1_win_rate', 'delta_accuracy', 'Run-1 Win Rate', 'Δ Accuracy'),
        ('run1_win_rate', 'delta_win_rate', 'Run-1 Win Rate', 'Δ Win Rate'),
    ]

    for base_var, change_var, base_label, change_label in pairs:
        if base_var not in df.columns or change_var not in df.columns:
            continue
        valid = df.dropna(subset=[base_var, change_var])
        if len(valid) < 5:
            continue
        r, p = stats.pearsonr(valid[base_var], valid[change_var])
        sig = '*' if p < 0.05 else '†' if p < 0.10 else ''
        results[(base_var, change_var)] = {'r': r, 'p': p, 'n': len(valid)}
        print(f'  {base_label} × {change_label}: r = {r:+.3f}, p = {p:.4f}{sig}, n = {len(valid)}')

    # Also test with sham-condition performance as baseline
    print('\n  --- Using sham-condition accuracy as baseline ---')
    sham_pairs = [
        ('sham_accuracy', 'delta_accuracy', 'Sham Accuracy', 'Δ Accuracy'),
        ('sham_accuracy', 'delta_win_rate', 'Sham Accuracy', 'Δ Win Rate'),
    ]
    for base_var, change_var, base_label, change_label in sham_pairs:
        if base_var not in df.columns or change_var not in df.columns:
            continue
        valid = df.dropna(subset=[base_var, change_var])
        if len(valid) < 5:
            continue
        r, p = stats.pearsonr(valid[base_var], valid[change_var])
        sig = '*' if p < 0.05 else '†' if p < 0.10 else ''
        results[(base_var, change_var)] = {'r': r, 'p': p, 'n': len(valid)}
        print(f'  {base_label} × {change_label}: r = {r:+.3f}, p = {p:.4f}{sig}, n = {len(valid)}')

    # Global cognition as baseline
    print('\n  --- Using global cognition as baseline ---')
    cog_pairs = [
        ('global_composite', 'delta_accuracy', 'Global Cognition', 'Δ Accuracy'),
        ('global_composite', 'delta_win_rate', 'Global Cognition', 'Δ Win Rate'),
    ]
    for base_var, change_var, base_label, change_label in cog_pairs:
        if base_var not in df.columns or change_var not in df.columns:
            continue
        valid = df.dropna(subset=[base_var, change_var])
        if len(valid) < 5:
            continue
        r, p = stats.pearsonr(valid[base_var], valid[change_var])
        sig = '*' if p < 0.05 else '†' if p < 0.10 else ''
        results[(base_var, change_var)] = {'r': r, 'p': p, 'n': len(valid)}
        print(f'  {base_label} × {change_label}: r = {r:+.3f}, p = {p:.4f}{sig}, n = {len(valid)}')

    if show_plots:
        # 2x2 panel: Run-1 Acc × ΔAcc, Run-1 Acc × ΔWR, Sham Acc × ΔAcc, Cognition × ΔAcc
        plot_pairs = [
            ('run1_accuracy', 'delta_accuracy', 'Run-1 Accuracy', 'Δ Accuracy'),
            ('run1_accuracy', 'delta_win_rate', 'Run-1 Accuracy', 'Δ Win Rate'),
            ('sham_accuracy', 'delta_accuracy', 'Sham Accuracy', 'Δ Accuracy'),
            ('global_composite', 'delta_accuracy', 'Global Cognition', 'Δ Accuracy'),
        ]
        plot_pairs = [(b, c, bl, cl) for b, c, bl, cl in plot_pairs
                      if b in df.columns and c in df.columns]

        if plot_pairs:
            n_plots = len(plot_pairs)
            fig = make_subplots(rows=1, cols=n_plots,
                                subplot_titles=[f'{bl} × {cl}' for _, _, bl, cl in plot_pairs],
                                horizontal_spacing=0.08)

            for idx, (base_var, change_var, base_label, change_label) in enumerate(plot_pairs):
                col = idx + 1
                valid = df.dropna(subset=[base_var, change_var, 'age'])
                x, y, ages = valid[base_var].values, valid[change_var].values, valid['age'].values
                r, p = stats.pearsonr(x, y)

                for xi, yi, ai in zip(x, y, ages):
                    fig.add_trace(go.Scatter(
                        x=[xi], y=[yi], mode='markers',
                        marker=dict(size=8, color=_age_to_rgb(ai),
                                    line=dict(width=0.5, color='white')),
                        showlegend=False,
                        hovertemplate=f'Age: {ai:.0f}<br>{base_label}: {xi:.3f}<br>{change_label}: {yi:.3f}<extra></extra>',
                    ), row=1, col=col)

                # Regression line
                slope, intercept, _, _, _ = stats.linregress(x, y)
                x_line = np.linspace(x.min(), x.max(), 50)
                line_color = '#C62828' if p < 0.05 else '#E64A19' if p < 0.10 else '#757575'
                line_dash = 'solid' if p < 0.05 else 'dash'
                fig.add_trace(go.Scatter(
                    x=x_line, y=intercept + slope * x_line, mode='lines',
                    line=dict(color=line_color, width=2, dash=line_dash),
                    showlegend=False, hoverinfo='skip',
                ), row=1, col=col)

                fig.add_hline(y=0, line_dash='dot', line_color='gray', line_width=1, row=1, col=col)

                sig_str = '*' if p < 0.05 else '†' if p < 0.10 else ''
                xref = 'x domain' if col == 1 else f'x{col} domain'
                yref = 'y domain' if col == 1 else f'y{col} domain'
                fig.add_annotation(
                    x=0.95, y=0.95, xref=xref, yref=yref,
                    text=f'r={r:+.3f}, p={p:.3f}{sig_str}', showarrow=False,
                    font=dict(size=10), xanchor='right', yanchor='top')

                fig.update_xaxes(title_text=base_label, row=1, col=col)
                fig.update_yaxes(title_text=change_label if col == 1 else '', row=1, col=col)

            fig.update_layout(
                height=400, width=300 * n_plots,
                template=PLOTLY_TEMPLATE, font=dict(family=FONT_FAMILY, size=11),
                title=dict(text='Baseline-Dependent tACS Effects: Convergence Across Measures',
                           font=dict(size=13)),
                margin=dict(l=60, r=40, t=80, b=60),
            )
            fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor='black')
            fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='black')
            fig.show(config=dict(toImageButtonOptions=dict(filename='baseline_convergence')))

    return results


# =============================================================================
# Test 2: Median split — tACS effect within low vs high performers
# =============================================================================

def test_median_split(subj_df, show_plots=True):
    """Split by baseline performance and test tACS effect within each group."""
    df = subj_df.copy()
    for col in ['run1_accuracy', 'sham_accuracy', 'active_accuracy',
                'sham_win_rate', 'active_win_rate', 'delta_accuracy',
                'delta_win_rate', 'age']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f'\n{"=" * 70}')
    print('TEST 2: MEDIAN SPLIT — tACS EFFECT IN LOW vs HIGH PERFORMERS')
    print('=' * 70)

    results = {}

    for base_var, base_label in [('run1_accuracy', 'Run-1 Accuracy'),
                                  ('sham_accuracy', 'Sham Accuracy')]:
        if base_var not in df.columns:
            continue

        valid = df.dropna(subset=[base_var, 'sham_accuracy', 'active_accuracy']).copy()
        median_val = valid[base_var].median()
        valid['perf_group'] = np.where(valid[base_var] <= median_val, 'Low', 'High')

        print(f'\n  --- Split on {base_label} (median = {median_val:.3f}) ---')

        for measure, sham_col, active_col, delta_col, m_label in [
            ('accuracy', 'sham_accuracy', 'active_accuracy', 'delta_accuracy', 'Accuracy'),
            ('win_rate', 'sham_win_rate', 'active_win_rate', 'delta_win_rate', 'Win Rate'),
        ]:
            if sham_col not in valid.columns or active_col not in valid.columns:
                continue

            for group_label in ['Low', 'High']:
                g = valid[valid['perf_group'] == group_label].dropna(subset=[sham_col, active_col])
                if len(g) < 3:
                    continue

                sham_vals = g[sham_col].values
                active_vals = g[active_col].values
                diff = active_vals - sham_vals

                t, p = stats.ttest_rel(active_vals, sham_vals)
                d = diff.mean() / diff.std() if diff.std() > 0 else 0
                sig = '*' if p < 0.05 else '†' if p < 0.10 else ''

                key = f'{base_var}_{group_label}_{measure}'
                results[key] = {
                    'group': group_label, 'n': len(g),
                    'sham_mean': sham_vals.mean(), 'active_mean': active_vals.mean(),
                    'diff_mean': diff.mean(), 'diff_sd': diff.std(),
                    't': t, 'p': p, 'dz': d,
                }

                print(f'\n    {group_label} performers (n={len(g)}): {m_label}')
                print(f'      Sham:   M = {sham_vals.mean():.3f}, SD = {sham_vals.std():.3f}')
                print(f'      Active: M = {active_vals.mean():.3f}, SD = {active_vals.std():.3f}')
                print(f'      Diff:   M = {diff.mean():+.3f}, SD = {diff.std():.3f}')
                print(f'      t({len(g)-1}) = {t:.3f}, p = {p:.4f}{sig}, dz = {d:.3f}')

    if show_plots and results:
        # Bar plot: low vs high, sham vs active, for accuracy
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=['Accuracy by Group', 'Win Rate by Group'],
                            horizontal_spacing=0.15)

        for col_idx, (measure, sham_col, active_col) in enumerate([
            ('accuracy', 'sham_accuracy', 'active_accuracy'),
            ('win_rate', 'sham_win_rate', 'active_win_rate'),
        ], 1):
            base_var = 'run1_accuracy'
            if base_var not in df.columns:
                continue

            valid = df.dropna(subset=[base_var, sham_col, active_col]).copy()
            median_val = valid[base_var].median()
            valid['perf_group'] = np.where(valid[base_var] <= median_val, 'Low', 'High')

            for group_label, group_color_sham, group_color_active in [
                ('Low', '#1565C0', '#E64A19'),
                ('High', '#64B5F6', '#FF8A65'),
            ]:
                g = valid[valid['perf_group'] == group_label]
                sham_m = g[sham_col].mean()
                sham_se = g[sham_col].std() / np.sqrt(len(g))
                active_m = g[active_col].mean()
                active_se = g[active_col].std() / np.sqrt(len(g))

                x_offset = -0.15 if group_label == 'Low' else 0.15

                fig.add_trace(go.Bar(
                    x=[f'{group_label}<br>Sham'],
                    y=[sham_m], error_y=dict(type='data', array=[sham_se]),
                    marker_color=group_color_sham, name=f'{group_label} Sham',
                    showlegend=(col_idx == 1), width=0.35,
                ), row=1, col=col_idx)

                fig.add_trace(go.Bar(
                    x=[f'{group_label}<br>Active'],
                    y=[active_m], error_y=dict(type='data', array=[active_se]),
                    marker_color=group_color_active, name=f'{group_label} Active',
                    showlegend=(col_idx == 1), width=0.35,
                ), row=1, col=col_idx)

            # Add significance annotations
            for group_label in ['Low', 'High']:
                key = f'{base_var}_{group_label}_{measure}'
                if key in results:
                    res = results[key]
                    if res['p'] < 0.10:
                        sig_text = f'p={res["p"]:.3f}{"*" if res["p"] < 0.05 else "†"}'
                        # Simple annotation
                        fig.add_annotation(
                            x=f'{group_label}<br>Active',
                            y=res['active_mean'] + res['diff_sd'] * 0.3,
                            text=sig_text, showarrow=False,
                            font=dict(size=9, color='#C62828'),
                            row=1, col=col_idx)

        fig.update_layout(
            height=450, width=800, barmode='group',
            template=PLOTLY_TEMPLATE, font=dict(family=FONT_FAMILY, size=11),
            title=dict(text='tACS Effect by Baseline Performance Group (Median Split on Run-1 Accuracy)',
                       font=dict(size=13)),
            margin=dict(l=60, r=40, t=80, b=60),
        )
        fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='black')
        fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='black')
        fig.show(config=dict(toImageButtonOptions=dict(filename='baseline_median_split')))

    return results


# =============================================================================
# Test 3: Regression to the mean control
# =============================================================================

def test_regression_to_mean(subj_df, show_plots=True):
    """
    Critical control: does the baseline × change relationship exist
    within the sham condition alone? If so, it's regression to the mean,
    not a stimulation effect.
    """
    df = subj_df.copy()

    print(f'\n{"=" * 70}')
    print('TEST 3: REGRESSION TO THE MEAN CONTROL')
    print('=' * 70)
    print('\n  If baseline × change exists within SHAM too, the active-condition')
    print('  pattern may reflect regression to the mean, not genuine stimulation.')

    results = {}

    # We need run-level data to compute within-condition change
    # Use run1 vs later runs within each condition
    # But if we only have subject-level data, we can compare:
    #   - Run-1 accuracy (before any condition assignment) vs sham accuracy
    #   - Run-1 accuracy vs active accuracy
    # And test whether the baseline-change correlation is STRONGER for active than sham

    for col in ['run1_accuracy', 'sham_accuracy', 'active_accuracy',
                'sham_win_rate', 'active_win_rate', 'age']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'run1_accuracy' not in df.columns:
        print('\n  ⚠ run1_accuracy not found. Cannot run regression-to-mean control.')
        return results

    valid = df.dropna(subset=['run1_accuracy', 'sham_accuracy', 'active_accuracy']).copy()

    # Compute "change" from Run-1 to each condition
    valid['change_to_sham'] = valid['sham_accuracy'] - valid['run1_accuracy']
    valid['change_to_active'] = valid['active_accuracy'] - valid['run1_accuracy']

    # Also compute the stimulation-specific change
    valid['delta_accuracy'] = valid['active_accuracy'] - valid['sham_accuracy']

    print(f'\n  N = {len(valid)}')

    # Key comparisons
    comparisons = [
        ('run1_accuracy', 'change_to_sham', 'Run-1 Acc', 'Change to Sham'),
        ('run1_accuracy', 'change_to_active', 'Run-1 Acc', 'Change to Active'),
        ('run1_accuracy', 'delta_accuracy', 'Run-1 Acc', 'Δ Accuracy (Active − Sham)'),
    ]

    for base_var, change_var, base_label, change_label in comparisons:
        x, y = valid[base_var].values, valid[change_var].values
        r, p = stats.pearsonr(x, y)
        sig = '*' if p < 0.05 else '†' if p < 0.10 else ''
        results[(base_var, change_var)] = {'r': r, 'p': p, 'n': len(valid)}
        print(f'\n  {base_label} × {change_label}:')
        print(f'    r = {r:+.3f}, p = {p:.4f}{sig}, n = {len(valid)}')

    # Compare the two correlations: is active stronger than sham?
    r_sham = results.get(('run1_accuracy', 'change_to_sham'), {}).get('r', 0)
    r_active = results.get(('run1_accuracy', 'change_to_active'), {}).get('r', 0)
    r_delta = results.get(('run1_accuracy', 'delta_accuracy'), {}).get('r', 0)

    print(f'\n  --- Comparison ---')
    print(f'  Baseline × Change_to_Sham:   r = {r_sham:+.3f}')
    print(f'  Baseline × Change_to_Active: r = {r_active:+.3f}')
    print(f'  Baseline × Δ(Active−Sham):   r = {r_delta:+.3f}')

    # Steiger's test for comparing dependent correlations
    r_sham_active = valid['change_to_sham'].corr(valid['change_to_active'])
    n = len(valid)
    # Use Steiger's Z for dependent correlations
    r12 = abs(r_sham)
    r13 = abs(r_active)
    r23 = abs(r_sham_active)
    det = 1 - r12**2 - r13**2 - r23**2 + 2*r12*r13*r23
    if det > 0:
        z_diff = (np.arctanh(r_active) - np.arctanh(r_sham)) * np.sqrt(
            (n - 3) / (2 * (1 - r23) * (1 + (r12 + r13)/2)**2 / (1 - (r12 + r13)**2/4) + 1e-10)
        )
        # Simplified version — just report the difference
        print(f'\n  Difference in |r|: {abs(r_active) - abs(r_sham):+.3f}')
        if abs(r_active) > abs(r_sham):
            print(f'  → Active shows STRONGER baseline dependence than Sham')
            print(f'    This could reflect genuine baseline-dependent stimulation')
        else:
            print(f'  → Sham shows EQUAL or STRONGER baseline dependence')
            print(f'    This suggests regression to the mean, NOT stimulation-specific')

    print(f'\n  --- Interpretation ---')
    p_delta = results.get(('run1_accuracy', 'delta_accuracy'), {}).get('p', 1)
    p_sham = results.get(('run1_accuracy', 'change_to_sham'), {}).get('p', 1)
    if p_sham < 0.05:
        print('  ⚠ Baseline × Change is significant within SHAM.')
        print('    The Δ(Active−Sham) correlation may partly reflect regression to the mean.')
    elif p_delta < 0.10:
        print('  ✓ Baseline × Change is NOT significant within SHAM alone.')
        print('    The Δ(Active−Sham) pattern appears stimulation-specific.')
    else:
        print('  Neither correlation is significant. Inconclusive.')

    if show_plots:
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=[
                                f'Run-1 × Change to Sham<br><sub>r={r_sham:+.3f}</sub>',
                                f'Run-1 × Change to Active<br><sub>r={r_active:+.3f}</sub>',
                                f'Run-1 × Δ(Active−Sham)<br><sub>r={r_delta:+.3f}</sub>',
                            ],
                            horizontal_spacing=0.08)

        for col_idx, (change_var, change_label, color) in enumerate([
            ('change_to_sham', 'Change to Sham', '#1565C0'),
            ('change_to_active', 'Change to Active', '#E64A19'),
            ('delta_accuracy', 'Δ Accuracy', '#2E7D32'),
        ], 1):
            x = valid['run1_accuracy'].values
            y = valid[change_var].values
            ages = valid['age'].values if 'age' in valid.columns else np.zeros(len(valid))

            for xi, yi, ai in zip(x, y, ages):
                fig.add_trace(go.Scatter(
                    x=[xi], y=[yi], mode='markers',
                    marker=dict(size=8, color=_age_to_rgb(ai),
                                line=dict(width=0.5, color='white')),
                    showlegend=False,
                    hovertemplate=f'Age: {ai:.0f}<br>Run-1: {xi:.3f}<br>{change_label}: {yi:.3f}<extra></extra>',
                ), row=1, col=col_idx)

            slope, intercept, _, _, _ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 50)
            r_val, p_val = stats.pearsonr(x, y)
            line_color = '#C62828' if p_val < 0.05 else color if p_val < 0.10 else '#757575'
            line_dash = 'solid' if p_val < 0.05 else 'dash'

            fig.add_trace(go.Scatter(
                x=x_line, y=intercept + slope * x_line, mode='lines',
                line=dict(color=line_color, width=2, dash=line_dash),
                showlegend=False, hoverinfo='skip',
            ), row=1, col=col_idx)

            fig.add_hline(y=0, line_dash='dot', line_color='gray', line_width=1, row=1, col=col_idx)

            # Add chance line on x-axis
            fig.add_vline(x=0.5, line_dash='dot', line_color='gray', line_width=1, row=1, col=col_idx)

            fig.update_xaxes(title_text='Run-1 Accuracy', row=1, col=col_idx)
            fig.update_yaxes(title_text=change_label if col_idx == 1 else '', row=1, col=col_idx)

        # Age colorbar
        fig.add_trace(go.Scatter(
            x=[None]*50, y=[None]*50, mode='markers',
            marker=dict(size=0.1, color=np.linspace(AGE_MIN, AGE_MAX, 50),
                        colorscale=AGE_COLORSCALE, cmin=AGE_MIN, cmax=AGE_MAX,
                        colorbar=dict(x=1.04, y=0.5, len=0.5, thickness=10,
                                      title='Age', titleside='right')),
            showlegend=False, hoverinfo='skip',
        ))

        fig.update_layout(
            height=400, width=1000,
            template=PLOTLY_TEMPLATE, font=dict(family=FONT_FAMILY, size=11),
            title=dict(text='Regression to the Mean Control: Baseline × Change by Condition',
                       font=dict(size=13)),
            margin=dict(l=60, r=80, t=80, b=60),
        )
        fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='black')
        fig.show(config=dict(toImageButtonOptions=dict(filename='regression_to_mean_control')))

    return results


# =============================================================================
# Main entry point
# =============================================================================

def run_baseline_moderation_diagnostic(subj_df, show_plots=True):
    """Run all baseline moderation tests."""

    print('\n' + '#' * 70)
    print('  BASELINE-DEPENDENT tACS EFFECTS: COMPREHENSIVE DIAGNOSTIC')
    print('#' * 70)

    results = {}

    # Test 1: Convergence across measures
    results['convergence'] = test_convergence(subj_df, show_plots=show_plots)

    # Test 2: Median split
    results['median_split'] = test_median_split(subj_df, show_plots=show_plots)

    # Test 3: Regression to the mean
    results['rtm_control'] = test_regression_to_mean(subj_df, show_plots=show_plots)

    # Summary
    print(f'\n{"=" * 70}')
    print('SUMMARY')
    print('=' * 70)

    # Check convergence
    conv = results.get('convergence', {})
    n_sig = sum(1 for v in conv.values() if v['p'] < 0.05)
    n_marginal = sum(1 for v in conv.values() if 0.05 <= v['p'] < 0.10)
    print(f'\n  Convergence: {n_sig} significant, {n_marginal} marginal across {len(conv)} baseline × change pairs')

    # Check RTM
    rtm = results.get('rtm_control', {})
    rtm_sham = rtm.get(('run1_accuracy', 'change_to_sham'), {})
    rtm_delta = rtm.get(('run1_accuracy', 'delta_accuracy'), {})
    if rtm_sham.get('p', 1) < 0.05:
        print('\n  ⚠ CAUTION: Regression to the mean detected in sham condition.')
        print('    Baseline-dependent stimulation effects should be interpreted cautiously.')
    elif rtm_delta.get('p', 1) < 0.10:
        print('\n  ✓ Baseline-dependent effect appears stimulation-specific.')
        print('    (Not present within sham condition alone.)')

    return results
