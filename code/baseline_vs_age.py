"""
Baseline vs. Age as Predictors of Stimulation Responsiveness

The age moderation on Δ accuracy (bivariate r = .332, p = .048; OLS
controlling for cognition p = .096) raises a question: is the moderator
really *age*, or is it *baseline performance*?  Age and baseline
performance are correlated — older adults tend to perform worse — so the
bivariate age effect could partly reflect baseline performance level.

This module disentangles the two by:

    1. Testing baseline accuracy as a predictor of Δ accuracy (simple r)
    2. Horse race: Δ accuracy ~ age + baseline_accuracy + global_cog
       → which predictor survives when both are in the model?
    3. Δ accuracy ~ baseline_accuracy + global_cog (age-free model)
    4. Repeating 1–3 for all available Δ DVs
    5. Testing the younger-adult over-reactivity mechanism continuously:
       does baseline performance predict Δα?

Also reports the preregistered secondary analysis: baseline WSLS rigidity
as a moderator of tACS effects.

Usage in main_analyses.ipynb:
    from baseline_vs_age import run_baseline_vs_age_analysis
    bva_results = run_baseline_vs_age_analysis(
        data_clean=data_clean,
        subj_df=subj_df,
        h2_eligible=h2_eligible,
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
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import matplotlib.colors as mcolors
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ── Visualization helpers ────────────────────────────────────────────────────

AGE_MIN, AGE_MAX = 20, 80

def _age_to_rgb(age):
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'blue_gold', ['#1565C0', '#FFB300'])
    norm = np.clip((age - AGE_MIN) / (AGE_MAX - AGE_MIN), 0, 1)
    r, g, b, _ = cmap(norm)
    return f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'


def _ols_summary(y, X_df, label, verbose=True):
    """Run OLS and return results dict. X_df should NOT include constant."""
    if not HAS_SM:
        return None
    X = sm.add_constant(X_df)
    model = sm.OLS(y, X).fit()
    result = {
        'r2': model.rsquared,
        'adj_r2': model.rsquared_adj,
        'f': model.fvalue,
        'f_p': model.f_pvalue,
        'n': int(model.nobs),
        'params': {},
        'model': model,
    }
    for var in X_df.columns:
        result['params'][var] = {
            'b': model.params[var],
            'se': model.bse[var],
            'p': model.pvalues[var],
            'beta': model.params[var] * (X_df[var].std() / y.std()) if y.std() > 0 else np.nan,
        }
    return result


def _print_ols(result, label, verbose=True):
    """Pretty-print an OLS result."""
    if result is None or not verbose:
        return
    parts = [f'R²={result["r2"]:.3f}']
    for var, p in result['params'].items():
        sig = '*' if p['p'] < .05 else ''
        parts.append(f'{var} β={p["beta"]:+.3f} p={p["p"]:.3f}{sig}')
    print(f'    {label} (n={result["n"]}): {", ".join(parts)}')


# ── 1. Compute Baseline Performance ─────────────────────────────────────────

def compute_baseline_accuracy(data_clean, h2_eligible):
    """
    Compute Run 1 accuracy per subject as a pre-stimulation baseline measure.
    Run 1 is always unstimulated regardless of counterbalance order.
    """
    df = data_clean[
        (data_clean['run'] == 1) &
        data_clean['subject_id'].isin([str(s) for s in h2_eligible]) &
        data_clean['choice'].notna()
    ].copy()

    df['correct_num'] = df['correct'].astype(float)

    baseline = (df.groupby('subject_id')
                .agg(baseline_accuracy=('correct_num', 'mean'),
                     baseline_n_trials=('correct_num', 'count'))
                .reset_index())

    return baseline


# ── 2. Horse Race Models ─────────────────────────────────────────────────────

def run_horse_race(subj_df, delta_col, label, verbose=True):
    """
    Compare three models predicting a stimulation change score:
      Model 1: Δ ~ age + global_cog
      Model 2: Δ ~ baseline_accuracy + global_cog
      Model 3: Δ ~ age + baseline_accuracy + global_cog  (horse race)

    Returns dict with all three model results.
    """
    if not HAS_SM:
        if verbose:
            print(f'  statsmodels not available — skipping {label}')
        return None

    required = [delta_col, 'age', 'baseline_accuracy', 'global_composite']
    df = subj_df[['subject_id'] + required].dropna()

    if len(df) < 8:
        if verbose:
            print(f'  {label}: insufficient data (n={len(df)})')
        return None

    y = df[delta_col].astype(float)

    results = {}

    # Model 1: age + cognition (preregistered H2.2)
    m1 = _ols_summary(y, df[['age', 'global_composite']].astype(float),
                       f'M1: {label} ~ age + cog')
    results['age_only'] = m1

    # Model 2: baseline + cognition
    m2 = _ols_summary(y, df[['baseline_accuracy', 'global_composite']].astype(float),
                       f'M2: {label} ~ baseline + cog')
    results['baseline_only'] = m2

    # Model 3: full horse race
    m3 = _ols_summary(y, df[['age', 'baseline_accuracy', 'global_composite']].astype(float),
                       f'M3: {label} ~ age + baseline + cog')
    results['horse_race'] = m3

    if verbose:
        print(f'\n  {label}:')
        _print_ols(m1, 'M1 (age + cog)')
        _print_ols(m2, 'M2 (baseline + cog)')
        _print_ols(m3, 'M3 (age + baseline + cog)')

        # Interpretation
        if m3 is not None:
            age_p = m3['params']['age']['p']
            base_p = m3['params']['baseline_accuracy']['p']
            if age_p < .05 and base_p >= .05:
                print(f'    → Age survives; baseline does not')
            elif base_p < .05 and age_p >= .05:
                print(f'    → Baseline survives; age does not')
            elif age_p < .05 and base_p < .05:
                print(f'    → Both survive — independent contributions')
            else:
                print(f'    → Neither survives in the full model')

    return results


# ── 3. Baseline Rigidity as Moderator (Preregistered Secondary) ──────────────

def run_baseline_rigidity_moderation(subj_df, verbose=True):
    """
    Preregistered secondary analysis: does baseline WSLS rigidity moderate
    the tACS effect? Rigidity = high p(stay|win) + low p(shift|lose).

    Tests whether sham WSLS parameters predict Δ accuracy and Δ WSLS.
    """
    results = {}

    predictors = [
        ('sham_p_stay_win', 'Baseline p(stay|win)'),
        ('sham_p_shift_lose', 'Baseline p(shift|lose)'),
    ]

    targets = [
        ('delta_accuracy', 'Δ Accuracy'),
        ('delta_p_stay_win', 'Δ p(stay|win)'),
        ('delta_p_shift_lose', 'Δ p(shift|lose)'),
        ('delta_alpha', 'Δ α'),
    ]

    if verbose:
        print(f'\n--- Baseline WSLS Rigidity as Moderator (Preregistered Secondary) ---')

    for pred_col, pred_label in predictors:
        if pred_col not in subj_df.columns:
            continue
        for target_col, target_label in targets:
            if target_col not in subj_df.columns:
                continue

            df = subj_df[[pred_col, target_col]].dropna()
            if len(df) < 5:
                continue

            x = pd.to_numeric(df[pred_col], errors='coerce').values
            y = pd.to_numeric(df[target_col], errors='coerce').values
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]

            if len(x) < 5:
                continue

            r, p = stats.pearsonr(x, y)
            key = f'{pred_col}_x_{target_col}'
            results[key] = {'r': r, 'p': p, 'n': len(x)}

            if verbose:
                sig = '  *' if p < .05 else ''
                print(f'  {pred_label} → {target_label}: '
                      f'r = {r:+.3f}, p = {p:.3f}, n = {len(x)}{sig}')

    return results


# ── 4. Over-Reactivity Mechanism (Continuous) ────────────────────────────────

def run_overreactivity_analysis(subj_df, verbose=True):
    """
    Test the over-reactivity mechanism continuously rather than via
    median split. If tACS disrupts well-calibrated systems:
      - Higher baseline accuracy → larger Δα (more over-reactive)
      - Higher baseline accuracy → more negative Δ accuracy (more harm)
      - Δα should correlate with Δ accuracy (mechanism → consequence)
    """
    results = {}

    tests = [
        ('baseline_accuracy', 'delta_alpha',
         'Baseline acc → Δα', 'Higher baseline → more over-reactivity?'),
        ('baseline_accuracy', 'delta_accuracy',
         'Baseline acc → Δ accuracy', 'Higher baseline → more harm?'),
        ('delta_alpha', 'delta_accuracy',
         'Δα → Δ accuracy', 'Over-reactivity → performance decline?'),
        ('sham_alpha', 'delta_alpha',
         'Sham α → Δα', 'Baseline learning rate → tACS-induced change?'),
    ]

    if verbose:
        print(f'\n--- Over-Reactivity Mechanism (Continuous) ---')

    for x_col, y_col, label, question in tests:
        if x_col not in subj_df.columns or y_col not in subj_df.columns:
            continue

        df = subj_df[[x_col, y_col]].dropna()
        x = pd.to_numeric(df[x_col], errors='coerce').values
        y = pd.to_numeric(df[y_col], errors='coerce').values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        if len(x) < 5:
            continue

        r, p = stats.pearsonr(x, y)
        results[f'{x_col}_x_{y_col}'] = {'r': r, 'p': p, 'n': len(x)}

        if verbose:
            sig = '  *' if p < .05 else ''
            print(f'  {label}: r = {r:+.3f}, p = {p:.3f}, n = {len(x)}{sig}')
            if p < .05:
                print(f'    → {question} YES')

    # Mediation-style check: does controlling for Δα attenuate baseline → Δ accuracy?
    if HAS_SM:
        med_cols = ['baseline_accuracy', 'delta_alpha', 'delta_accuracy']
        df = subj_df[med_cols].dropna()
        for c in med_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna()

        if len(df) >= 8:
            y = df['delta_accuracy'].astype(float)

            # Without mediator
            X1 = df[['baseline_accuracy']].astype(float)
            m1 = _ols_summary(y, X1, 'Δ acc ~ baseline')

            # With mediator
            X2 = df[['baseline_accuracy', 'delta_alpha']].astype(float)
            m2 = _ols_summary(y, X2, 'Δ acc ~ baseline + Δα')

            results['mediation_without'] = m1
            results['mediation_with'] = m2

            if verbose and m1 is not None and m2 is not None:
                b_before = m1['params']['baseline_accuracy']['p']
                b_after = m2['params']['baseline_accuracy']['p']
                print(f'\n  Mediation check (does Δα account for baseline → Δ accuracy?):')
                _print_ols(m1, 'Δ acc ~ baseline')
                _print_ols(m2, 'Δ acc ~ baseline + Δα')
                if b_before < .10 and b_after > b_before:
                    print(f'    → Baseline effect attenuated after controlling for Δα')

    return results


# ── 5. Summary Plot ──────────────────────────────────────────────────────────

def _make_horse_race_plot(subj_df):
    """Four-panel figure: age and baseline as predictors of Δ accuracy."""
    if not HAS_PLOTLY:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Δ Accuracy ~ Age',
            'Δ Accuracy ~ Baseline Accuracy',
            'Δα ~ Baseline Accuracy',
            'Δ Accuracy ~ Δα',
        ],
        horizontal_spacing=0.12, vertical_spacing=0.14,
    )

    panels = [
        ('age', 'delta_accuracy', 'Age (years)', 'Δ Accuracy', 1, 1),
        ('baseline_accuracy', 'delta_accuracy', 'Baseline Accuracy (Run 1)', 'Δ Accuracy', 1, 2),
        ('baseline_accuracy', 'delta_alpha', 'Baseline Accuracy (Run 1)', 'Δ α', 2, 1),
        ('delta_alpha', 'delta_accuracy', 'Δ α', 'Δ Accuracy', 2, 2),
    ]

    for x_col, y_col, xlabel, ylabel, row, col in panels:
        if x_col not in subj_df.columns or y_col not in subj_df.columns:
            continue

        df = subj_df[['subject_id', 'age', x_col, y_col]].dropna()
        # Handle potential duplicate columns by selecting first occurrence
        if isinstance(df[x_col], pd.DataFrame):
            x = pd.to_numeric(df[x_col].iloc[:, 0], errors='coerce')
        else:
            x = pd.to_numeric(df[x_col], errors='coerce')
        if isinstance(df[y_col], pd.DataFrame):
            y = pd.to_numeric(df[y_col].iloc[:, 0], errors='coerce')
        else:
            y = pd.to_numeric(df[y_col], errors='coerce')
        if isinstance(df['age'], pd.DataFrame):
            age = pd.to_numeric(df['age'].iloc[:, 0], errors='coerce')
        else:
            age = pd.to_numeric(df['age'], errors='coerce')
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(age)
        x, y, age = x[mask].values, y[mask].values, age[mask].values
        labels = df.loc[mask.values, 'subject_id'].values if hasattr(mask, 'values') else df['subject_id'].values[:len(x)]

        if len(x) < 5:
            continue

        slope, intercept, r, p, se = stats.linregress(x, y)

        # Points colored by age
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='markers',
            marker=dict(size=10, opacity=0.8,
                        color=[_age_to_rgb(a) for a in age],
                        line=dict(width=0.5, color='white')),
            text=[f'sub-{s}<br>Age: {a:.0f}' for s, a in zip(labels, age)],
            hoverinfo='text+x+y', showlegend=False,
        ), row=row, col=col)

        # Regression line
        x_line = np.linspace(x.min(), x.max(), 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=intercept + slope * x_line,
            mode='lines', line=dict(color='#404040', width=2),
            showlegend=False, hoverinfo='skip',
        ), row=row, col=col)

        # Zero line
        fig.add_hline(y=0, line=dict(color='#999999', width=1, dash='dash'),
                      row=row, col=col)

        # Stats annotation
        sig = '*' if p < .05 else ''
        xref = f'x{col if row == 1 else col + 2} domain'
        yref = f'y{col if row == 1 else col + 2} domain'
        # Simpler: use subplot index
        subplot_idx = (row - 1) * 2 + col
        fig.add_annotation(
            text=f'r = {r:.3f}, p = {p:.3f}{sig}<br>N = {len(x)}',
            x=0.95, y=0.95,
            xref=f'x{subplot_idx} domain' if subplot_idx > 1 else 'x domain',
            yref=f'y{subplot_idx} domain' if subplot_idx > 1 else 'y domain',
            showarrow=False, font=dict(size=11),
            xanchor='right', yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
        )

        fig.update_xaxes(title_text=xlabel, row=row, col=col)
        fig.update_yaxes(title_text=ylabel, row=row, col=col)

    fig.update_layout(
        height=700, width=850,
        template='plotly_white',
        font=dict(family='Arial', size=12),
        margin=dict(l=60, r=40, t=50, b=50),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='black')

    return fig


# ── Master Runner ────────────────────────────────────────────────────────────

def run_baseline_vs_age_analysis(
    data_clean,
    subj_df,
    h2_eligible,
    show_plots=True,
    verbose=True,
):
    """
    Disentangle age vs. baseline performance as predictors of tACS response.

    Parameters
    ----------
    data_clean : DataFrame
        Trial-level data with exclusions applied.
    subj_df : DataFrame
        Subject-level master DataFrame.
    h2_eligible : list
        Subject IDs eligible for H2 analyses.
    show_plots : bool
        Display summary figure.
    verbose : bool
        Print results.

    Returns
    -------
    dict with keys:
        'baseline_data', 'horse_race', 'rigidity_moderation',
        'overreactivity', 'summary_fig', 'subj_df'
    """
    print('=' * 70)
    print('BASELINE vs. AGE AS PREDICTORS OF STIMULATION RESPONSE')
    print('Is the moderator age per se, or baseline performance level?')
    print('=' * 70)

    results = {}
    h2_str = [str(s) for s in h2_eligible]

    # ── Compute baseline accuracy and merge ──────────────────────────────
    baseline = compute_baseline_accuracy(data_clean, h2_str)
    baseline['subject_id'] = baseline['subject_id'].astype(str)

    if 'baseline_accuracy' not in subj_df.columns:
        baseline_merge = baseline[['subject_id', 'baseline_accuracy']].drop_duplicates(subset='subject_id')
        subj_df = subj_df.merge(baseline_merge, on='subject_id', how='left')
    else:
        # Update existing values
        lookup = baseline.set_index('subject_id')['baseline_accuracy'].to_dict()
        subj_df['baseline_accuracy'] = subj_df['subject_id'].map(
            lambda s: lookup.get(str(s), np.nan)).fillna(subj_df['baseline_accuracy'])

    results['baseline_data'] = baseline

    # ── Report age-baseline correlation ──────────────────────────────────
    df_check = subj_df[['age', 'baseline_accuracy']].dropna()
    df_check['age'] = pd.to_numeric(df_check['age'], errors='coerce')
    df_check['baseline_accuracy'] = pd.to_numeric(df_check['baseline_accuracy'], errors='coerce')
    df_check = df_check.dropna()

    if len(df_check) >= 5 and verbose:
        r, p = stats.pearsonr(df_check['age'].values, df_check['baseline_accuracy'].values)
        print(f'\nAge × Baseline accuracy: r = {r:.3f}, p = {p:.3f}, n = {len(df_check)}')
        if abs(r) > .3:
            print(f'  → Moderate correlation — collinearity warrants the horse race')

    # ── Bivariate correlations first ─────────────────────────────────────
    if verbose:
        print(f'\n--- Bivariate Correlations with Δ Accuracy ---')

    biv_predictors = [
        ('age', 'Age'),
        ('baseline_accuracy', 'Baseline accuracy'),
        ('global_composite', 'Global cognition'),
        ('sham_alpha', 'Sham α'),
        ('sham_beta', 'Sham β'),
    ]

    for pred_col, pred_label in biv_predictors:
        if pred_col not in subj_df.columns or 'delta_accuracy' not in subj_df.columns:
            continue
        df = subj_df[[pred_col, 'delta_accuracy']].dropna()
        x = pd.to_numeric(df[pred_col], errors='coerce').values
        y = pd.to_numeric(df['delta_accuracy'], errors='coerce').values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 5:
            r, p = stats.pearsonr(x[mask], y[mask])
            sig = '  *' if p < .05 else ''
            if verbose:
                print(f'  {pred_label} × Δ accuracy: r = {r:+.3f}, p = {p:.3f}, n = {mask.sum()}{sig}')

    # ── Horse race models ────────────────────────────────────────────────
    if verbose:
        print(f'\n--- Horse Race: Age vs. Baseline Performance ---')

    horse_race_results = {}
    delta_dvs = [
        ('delta_accuracy', 'Δ Accuracy'),
        ('delta_alpha', 'Δ α'),
        ('delta_p_stay_win', 'Δ p(stay|win)'),
        ('delta_p_shift_lose', 'Δ p(shift|lose)'),
    ]

    for col, label in delta_dvs:
        if col in subj_df.columns:
            hr = run_horse_race(subj_df, col, label, verbose)
            if hr is not None:
                horse_race_results[col] = hr

    results['horse_race'] = horse_race_results

    # ── Baseline rigidity moderation (preregistered secondary) ───────────
    results['rigidity_moderation'] = run_baseline_rigidity_moderation(
        subj_df, verbose)

    # ── Over-reactivity mechanism ────────────────────────────────────────
    results['overreactivity'] = run_overreactivity_analysis(subj_df, verbose)

    # ── Summary plot ─────────────────────────────────────────────────────
    if show_plots and HAS_PLOTLY:
        fig = _make_horse_race_plot(subj_df)
        if fig is not None:
            fig.show()
            results['summary_fig'] = fig

    # ── Narrative ────────────────────────────────────────────────────────
    if verbose:
        print(f'\n{"=" * 70}')
        print('INTERPRETATION')
        print('=' * 70)

        hr_acc = horse_race_results.get('delta_accuracy', {}).get('horse_race')
        if hr_acc is not None:
            age_p = hr_acc['params']['age']['p']
            base_p = hr_acc['params']['baseline_accuracy']['p']
            if base_p < .05 and age_p >= .05:
                print('  The moderator is baseline performance, not age per se.')
                print('  tACS benefits people performing suboptimally, regardless of age.')
                print('  This aligns with Grover et al. (2022) / Reinhart & Nguyen (2019).')
            elif age_p < .05 and base_p >= .05:
                print('  The moderator is age per se, not just baseline performance.')
                print('  Something biological about aging (cortical state, theta dynamics)')
                print('  determines tACS responsiveness beyond performance level.')
            elif age_p < .05 and base_p < .05:
                print('  Both age and baseline performance independently predict tACS response.')
            else:
                print('  Neither age nor baseline performance survives in the full model.')
                print('  The moderation may be too weak to decompose at current N.')
                print('  This is expected — the preregistration anticipated underpowered')
                print('  moderation analyses at N < 50.')

    results['subj_df'] = subj_df
    return results
