"""
cognitive_merge.py — External data integration for tACS Bandit study

Loads and scores survey/cognitive measures from multiple sources:
1. RF1 FullScoring — HVLT, Salthouse, SCD-Q, etc.
2. RF1 Raw REDCap — Digit Span, BVMT, Trail Making
3. TabCAT — Flanker, RunningDots, Set Shifting
4. tACS Bandit REDCap — SPSRQ, B-PSQI, BBS, CRT, FFMQ

Domain composites per pre-registration (Section 4):
- Attention: Digit Span + RunningDots + Flanker
- Episodic Memory: HVLT Total + BVMT Total
- Processing Speed: Salthouse Letter + Pattern + Trails A
- Global: average of domain composites
- Executive Function (secondary): RunningDots + Flanker + Trails B−A

All raw scores z-transformed within sample before averaging.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from config import (
    SUBJECT_INFO,
    REDCAP_RF1_PATH,
    REDCAP_RF1_RAW_PATH,
    REDCAP_TACS_PATH,
    TABCAT_PATH,
)


# =============================================================================
# RF1 FullScoring Loader
# =============================================================================

def load_rf1_fullscoring(
    filepath: str = REDCAP_RF1_PATH,
    study_subjects: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load pre-scored cognitive measures from RF1 FullScoring export.
    
    Parameters
    ----------
    filepath : str
        Path to RF1 FullScoring CSV
    study_subjects : list, optional
        Subject IDs to filter to
    
    Returns
    -------
    DataFrame with columns: subject_id, hvlt_total, hvlt_delay, hvlt_recognition,
                           salthouse_letter, salthouse_pattern, scd_q, har_score, kbit_iq
    """
    if study_subjects is None:
        study_subjects = list(SUBJECT_INFO.keys())
    
    rf1 = pd.read_csv(filepath, low_memory=False)
    rf1['subject_id'] = rf1['Subject ID'].astype(str)
    
    col_map = {
        'HVLT Total Immediate Recall': 'hvlt_total',
        'HVLT Delay Score': 'hvlt_delay',
        'Recognition Discrimination Index': 'hvlt_recognition',
        'Salthouse Letter Comparison': 'salthouse_letter',
        'Salthouse Pattern Comparison': 'salthouse_pattern',
        'scd_score_total': 'scd_q',
        'HAR score': 'har_score',
        'KBIT IQ (age adjusted)': 'kbit_iq',
        'Age': 'rf1_age',
    }
    
    # Select and rename columns
    cols_present = ['subject_id'] + [c for c in col_map.keys() if c in rf1.columns]
    rf1_cog = rf1[cols_present].copy()
    rf1_cog = rf1_cog.rename(columns=col_map)
    
    # Filter to study participants
    rf1_cog = rf1_cog[rf1_cog['subject_id'].isin(study_subjects)].copy()
    
    return rf1_cog


# =============================================================================
# RF1 Raw REDCap Loader
# =============================================================================

def load_rf1_raw(
    filepath: str = REDCAP_RF1_RAW_PATH,
    study_subjects: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load cognitive measures from RF1 Raw export.
    
    Includes: Digit Span, BVMT, Trail Making, Education
    
    Parameters
    ----------
    filepath : str
        Path to RF1 Raw CSV
    study_subjects : list, optional
        Subject IDs to filter to
    
    Returns
    -------
    DataFrame with digit_span_*, bvmt_*, trails_*, education_years
    """
    if study_subjects is None:
        study_subjects = list(SUBJECT_INFO.keys())
    
    rf1_raw = pd.read_csv(filepath, low_memory=False)
    rf1_raw['subject_id'] = rf1_raw['Subject ID'].astype(str)
    
    col_map = {
        'Digit Span Forward Score': 'digit_span_fwd',
        'Digit Span Backward Score': 'digit_span_bwd',
        'Digit Span Total Score': 'digit_span_total',
        'BVMT Total': 'bvmt_total',
        'BVMT Delay Recall': 'bvmt_delay',
        'Trails A Score': 'trails_a_time',
        'Trails B Score': 'trails_b_time',
        'Years of education': 'education_years',
    }
    
    cols_present = ['subject_id'] + [c for c in col_map.keys() if c in rf1_raw.columns]
    rf1_neuro = rf1_raw[cols_present].copy()
    rf1_neuro = rf1_neuro.rename(columns=col_map)
    
    # Some subjects have multiple rows; take first non-null per subject
    rf1_neuro = rf1_neuro.groupby('subject_id').first().reset_index()
    rf1_neuro = rf1_neuro[rf1_neuro['subject_id'].isin(study_subjects)].copy()
    
    return rf1_neuro


# =============================================================================
# TabCAT Loader
# =============================================================================

def load_tabcat(
    filepath: str = TABCAT_PATH,
    study_subjects: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load TabCAT cognitive measures (Flanker, RunningDots, SetShifting).
    
    Parameters
    ----------
    filepath : str
        Path to TabCAT CSV
    study_subjects : list, optional
        Subject IDs to filter to
    
    Returns
    -------
    DataFrame with flanker_*, running_dots_*, set_shifting_*
    """
    if study_subjects is None:
        study_subjects = list(SUBJECT_INFO.keys())
    
    tabcat = pd.read_csv(filepath)
    tabcat['subject_id'] = tabcat['Examinee_Identifier'].astype(str)
    
    col_map = {
        'Flanker_TotalScore': 'flanker_score',
        'Flanker_Correct_MedianRT': 'flanker_rt',
        'Flanker_IncongrCorrect_MedianRT': 'flanker_incongruent_rt',
        'RunningDots_TrialScore': 'running_dots_score',
        'RunningDots_PercentCorrect': 'running_dots_pct',
        'SetShifting_TotalScore': 'set_shifting_score',
        'SetShifting_ShiftCorrect_Total': 'set_shifting_shift_correct',
    }
    
    cols_present = ['subject_id'] + [c for c in col_map.keys() if c in tabcat.columns]
    tc = tabcat[cols_present].copy()
    tc = tc.rename(columns=col_map)
    
    # Handle duplicates (keep latest per subject)
    tc = tc.drop_duplicates(subset='subject_id', keep='last')
    tc = tc[tc['subject_id'].isin(study_subjects)].copy()
    
    return tc


# =============================================================================
# SPSRQ Scoring
# =============================================================================

def score_spsrq(redcap_tacs: pd.DataFrame) -> pd.DataFrame:
    """
    Score SPSRQ (Revised & Clarified, 20-item, 5-point Likert).
    
    Odd items (1,3,5,...,19) → SP (Sensitivity to Punishment)
    Even items (2,4,6,...,20) → SR (Sensitivity to Reward)
    
    Parameters
    ----------
    redcap_tacs : DataFrame
        tACS Bandit REDCap export
    
    Returns
    -------
    DataFrame with spsrq_sp, spsrq_sr columns added
    """
    df = redcap_tacs.copy()
    
    # Item content (first 40 chars used for matching)
    spsrq_items = [
        'I am afraid of new or unexpected situations.',
        'I like being the center of attention at a party or a social gathering.',
        'I am easily discouraged in difficult situations.',
        'When I am in a group, I try to make my opinions the most intelligent or the funniest.',
        'I am a shy person.',
        'I take the opportunity to pick up people I find attractive.',
        'I avoid demonstrating my skills for fear of being embarrassed.',
        'The possibility of social advancement moves me to action, even if this involves not playing fair.',
        'I worry about things that I said or did.',
        'I prefer activities that lead to an immediate gain.',
        'I think that I could do more things if it was not for my insecurity or fear.',
        'I like to compete and do everything I can to win.',
        'Compared to people I know, I am afraid of many things.',
        'I do things for quick gains.',
        'I find myself worrying about things so much that my ability to perform other mental tasks is impaired.',
        'I like to make a competition out of all of my activities.',
        'I refrain from doing something I like in order to not be rejected by or disapproved of by others.',
        'I would like to be a socially powerful person.',
        'I refrain from doing something because of my fear of being embarrassed.',
        'I like displaying my physical abilities even though this may involve danger.',
    ]
    
    likert_map = {
        'Very Untrue': 1,
        'Somewhat Untrue': 2,
        'Neither Untrue nor True': 3,
        'Somewhat True': 4,
        'Very True': 5,
    }
    
    # Find matching columns
    spsrq_cols = []
    for item in spsrq_items:
        matches = [c for c in df.columns if c.startswith(item[:40])]
        spsrq_cols.append(matches[0] if matches else None)
    
    # Recode to numeric
    for col in spsrq_cols:
        if col is not None and col in df.columns:
            df[col] = df[col].map(likert_map)
    
    # SP = odd items (0, 2, 4, ..., 18), SR = even items (1, 3, 5, ..., 19)
    sp_cols = [spsrq_cols[i] for i in range(0, 20, 2) if spsrq_cols[i] is not None]
    sr_cols = [spsrq_cols[i] for i in range(1, 20, 2) if spsrq_cols[i] is not None]
    
    df['spsrq_sp'] = df[sp_cols].sum(axis=1, min_count=1) if sp_cols else np.nan
    df['spsrq_sr'] = df[sr_cols].sum(axis=1, min_count=1) if sr_cols else np.nan
    df['spsrq_sp_n_items'] = df[sp_cols].notna().sum(axis=1) if sp_cols else 0
    df['spsrq_sr_n_items'] = df[sr_cols].notna().sum(axis=1) if sr_cols else 0
    
    return df


# =============================================================================
# tACS REDCap Loader (with SPSRQ scoring)
# =============================================================================

def load_tacs_redcap(
    filepath: str = REDCAP_TACS_PATH,
    study_subjects: Optional[List[str]] = None,
    score_surveys: bool = True
) -> pd.DataFrame:
    """
    Load tACS Bandit REDCap export with optional survey scoring.
    
    Parameters
    ----------
    filepath : str
        Path to tACS REDCap CSV
    study_subjects : list, optional
        Subject IDs to filter to
    score_surveys : bool
        If True, score SPSRQ
    
    Returns
    -------
    DataFrame with demographics and scored surveys
    """
    if study_subjects is None:
        study_subjects = list(SUBJECT_INFO.keys())
    
    redcap = pd.read_csv(filepath, low_memory=False)
    redcap['subject_id'] = redcap['Record ID'].astype(str)
    
    if score_surveys:
        redcap = score_spsrq(redcap)
    
    redcap = redcap[redcap['subject_id'].isin(study_subjects)].copy()
    
    return redcap


def extract_demographics(redcap_tacs: pd.DataFrame) -> pd.DataFrame:
    """
    Extract demographic variables from tACS REDCap export.
    
    Returns
    -------
    DataFrame with subject_id, age, gender, race, ethnicity, spsrq_*
    """
    cols = ['subject_id']
    
    # Map column names
    col_renames = {}
    if 'Age in years:' in redcap_tacs.columns:
        cols.append('Age in years:')
        col_renames['Age in years:'] = 'age'
    if 'Gender' in redcap_tacs.columns:
        cols.append('Gender')
        col_renames['Gender'] = 'gender'
    if 'Race' in redcap_tacs.columns:
        cols.append('Race')
        col_renames['Race'] = 'race'
    if 'Ethnicity' in redcap_tacs.columns:
        cols.append('Ethnicity')
        col_renames['Ethnicity'] = 'ethnicity'
    
    # Add SPSRQ if scored
    for col in ['spsrq_sp', 'spsrq_sr', 'spsrq_sp_n_items', 'spsrq_sr_n_items']:
        if col in redcap_tacs.columns:
            cols.append(col)
    
    demo = redcap_tacs[cols].copy()
    demo = demo.rename(columns=col_renames)
    
    return demo


# =============================================================================
# Domain Composites
# =============================================================================

def compute_cognitive_composites(
    rf1_cog: pd.DataFrame,
    rf1_neuro: pd.DataFrame,
    tabcat: pd.DataFrame,
    study_subjects: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute cognitive domain composites per pre-registration.
    
    Method: raw → z-transform → average per domain
    Time-based measures (Trails) are sign-flipped (lower = better).
    
    Parameters
    ----------
    rf1_cog : DataFrame
        From load_rf1_fullscoring()
    rf1_neuro : DataFrame
        From load_rf1_raw()
    tabcat : DataFrame
        From load_tabcat()
    study_subjects : list, optional
        Subject IDs
    
    Returns
    -------
    DataFrame with domain composites
    """
    if study_subjects is None:
        study_subjects = list(SUBJECT_INFO.keys())
    
    # Start with subject list
    cog = pd.DataFrame({'subject_id': study_subjects})
    
    # Merge sources
    cog = cog.merge(
        rf1_cog[['subject_id', 'hvlt_total', 'salthouse_letter', 'salthouse_pattern']],
        on='subject_id', how='left'
    )
    
    cog = cog.merge(
        rf1_neuro[['subject_id', 'digit_span_total', 'bvmt_total',
                   'trails_a_time', 'trails_b_time']],
        on='subject_id', how='left'
    )
    
    cog = cog.merge(
        tabcat[['subject_id', 'flanker_score', 'running_dots_score']],
        on='subject_id', how='left'
    )
    
    # Z-transform measures
    measures_to_z = [
        'digit_span_total', 'running_dots_score', 'flanker_score',
        'hvlt_total', 'bvmt_total',
        'salthouse_letter', 'salthouse_pattern', 'trails_a_time', 'trails_b_time',
    ]
    
    z_cols = {}
    for col in measures_to_z:
        z_name = f'{col}_z'
        if col in cog.columns:
            vals = cog[col].dropna()
            if len(vals) > 1:
                cog[z_name] = (cog[col] - vals.mean()) / vals.std()
            else:
                cog[z_name] = np.nan
            z_cols[col] = z_name
    
    # Flip sign for time-based measures
    for time_col in ['trails_a_time', 'trails_b_time']:
        z_name = z_cols.get(time_col)
        if z_name and z_name in cog.columns:
            cog[z_name] = -cog[z_name]
    
    # --- Domain Composites ---
    
    # Attention: Digit Span + RunningDots + Flanker
    att_cols = [z_cols[m] for m in ['digit_span_total', 'running_dots_score', 'flanker_score']
                if m in z_cols and z_cols[m] in cog.columns]
    cog['attention_composite'] = cog[att_cols].mean(axis=1) if att_cols else np.nan
    cog['attention_n_measures'] = cog[att_cols].notna().sum(axis=1) if att_cols else 0
    
    # Episodic Memory: HVLT + BVMT
    mem_cols = [z_cols[m] for m in ['hvlt_total', 'bvmt_total']
                if m in z_cols and z_cols[m] in cog.columns]
    cog['memory_composite'] = cog[mem_cols].mean(axis=1) if mem_cols else np.nan
    cog['memory_n_measures'] = cog[mem_cols].notna().sum(axis=1) if mem_cols else 0
    
    # Processing Speed: Salthouse Letter + Pattern + Trails A
    speed_cols = [z_cols[m] for m in ['salthouse_letter', 'salthouse_pattern', 'trails_a_time']
                  if m in z_cols and z_cols[m] in cog.columns]
    cog['speed_composite'] = cog[speed_cols].mean(axis=1) if speed_cols else np.nan
    cog['speed_n_measures'] = cog[speed_cols].notna().sum(axis=1) if speed_cols else 0
    
    # Global: average of domain composites
    domain_cols = ['attention_composite', 'memory_composite', 'speed_composite']
    cog['global_composite'] = cog[domain_cols].mean(axis=1)
    cog['global_n_domains'] = cog[domain_cols].notna().sum(axis=1)
    
    # Executive Function: RunningDots + Flanker + Trails B-A
    if 'trails_b_time' in cog.columns and 'trails_a_time' in cog.columns:
        cog['trails_ba_raw'] = cog['trails_b_time'] - cog['trails_a_time']
        vals = cog['trails_ba_raw'].dropna()
        if len(vals) > 1:
            cog['trails_ba_z'] = (cog['trails_ba_raw'] - vals.mean()) / vals.std()
            cog['trails_ba_z'] = -cog['trails_ba_z']  # flip: lower B-A = better
        else:
            cog['trails_ba_z'] = np.nan
    
    ef_cols = [z_cols.get(m) for m in ['running_dots_score', 'flanker_score']]
    ef_cols = [c for c in ef_cols if c is not None and c in cog.columns]
    if 'trails_ba_z' in cog.columns:
        ef_cols.append('trails_ba_z')
    cog['ef_composite'] = cog[ef_cols].mean(axis=1) if ef_cols else np.nan
    cog['ef_n_measures'] = cog[ef_cols].notna().sum(axis=1) if ef_cols else 0
    
    return cog


# =============================================================================
# Master Subject DataFrame Assembly
# =============================================================================

def build_subject_df(
    study_subjects: Optional[List[str]] = None,
    wsls_h1: Optional[pd.DataFrame] = None,
    wsls_h2: Optional[pd.DataFrame] = None,
    rw_mle: Optional[pd.DataFrame] = None,
    ddm_params: Optional[pd.DataFrame] = None,
    h2_subjects: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Assemble master subject-level DataFrame.
    
    Merges: demographics, cognitive composites, SPSRQ, behavioral parameters
    
    Parameters
    ----------
    study_subjects : list, optional
        Subject IDs
    wsls_h1, wsls_h2 : DataFrame, optional
        WSLS parameters from wsls module
    rw_mle : DataFrame, optional
        R-W parameters from rescorla_wagner module
    ddm_params : DataFrame, optional
        DDM parameters from ddm module
    h2_subjects : list, optional
        H2-eligible subjects for change scores
    verbose : bool
        If True, print summary
    
    Returns
    -------
    DataFrame with all subject-level variables
    """
    if study_subjects is None:
        study_subjects = list(SUBJECT_INFO.keys())
    
    subj_df = pd.DataFrame({'subject_id': study_subjects})
    
    # --- Load external data ---
    try:
        rf1_cog = load_rf1_fullscoring(study_subjects=study_subjects)
        rf1_neuro = load_rf1_raw(study_subjects=study_subjects)
        tabcat = load_tabcat(study_subjects=study_subjects)
        tacs_redcap = load_tacs_redcap(study_subjects=study_subjects)
    except FileNotFoundError as e:
        if verbose:
            print(f'WARNING: Could not load external data: {e}')
        return subj_df
    
    # --- Demographics ---
    demo = extract_demographics(tacs_redcap)
    subj_df = subj_df.merge(demo, on='subject_id', how='left')
    
    # Education from RF1 Raw
    if 'education_years' in rf1_neuro.columns:
        subj_df = subj_df.merge(
            rf1_neuro[['subject_id', 'education_years']],
            on='subject_id', how='left'
        )
    
    # --- Cognitive composites ---
    cog = compute_cognitive_composites(rf1_cog, rf1_neuro, tabcat, study_subjects)
    cog_cols = ['subject_id',
                'attention_composite', 'attention_n_measures',
                'memory_composite', 'memory_n_measures',
                'speed_composite', 'speed_n_measures',
                'global_composite', 'global_n_domains',
                'ef_composite', 'ef_n_measures']
    subj_df = subj_df.merge(cog[cog_cols], on='subject_id', how='left')
    
    # --- Additional RF1 measures ---
    rf1_extra_cols = ['subject_id']
    for col in ['scd_q', 'har_score', 'kbit_iq']:
        if col in rf1_cog.columns:
            rf1_extra_cols.append(col)
    if len(rf1_extra_cols) > 1:
        subj_df = subj_df.merge(rf1_cog[rf1_extra_cols], on='subject_id', how='left')
    
    # --- WSLS parameters ---
    if wsls_h1 is not None and len(wsls_h1) > 0:
        wsls_sham = wsls_h1.set_index('subject_id')[['p_stay_win', 'p_shift_lose']]
        wsls_sham = wsls_sham.rename(columns={
            'p_stay_win': 'sham_p_stay_win',
            'p_shift_lose': 'sham_p_shift_lose'
        })
        subj_df = subj_df.merge(wsls_sham, on='subject_id', how='left')
    
    if wsls_h2 is not None and len(wsls_h2) > 0:
        wsls_active = wsls_h2[wsls_h2['condition'] == 'active'].set_index('subject_id')
        if len(wsls_active) > 0:
            wsls_active = wsls_active[['p_stay_win', 'p_shift_lose']]
            wsls_active = wsls_active.rename(columns={
                'p_stay_win': 'active_p_stay_win',
                'p_shift_lose': 'active_p_shift_lose'
            })
            subj_df = subj_df.merge(wsls_active, on='subject_id', how='left')
            
            # Change scores
            if 'sham_p_stay_win' in subj_df.columns:
                subj_df['delta_p_stay_win'] = subj_df['active_p_stay_win'] - subj_df['sham_p_stay_win']
                subj_df['delta_p_shift_lose'] = subj_df['active_p_shift_lose'] - subj_df['sham_p_shift_lose']
    
    # --- R-W parameters ---
    if rw_mle is not None and len(rw_mle) > 0:
        # Sham
        rw_sham = rw_mle[rw_mle['condition'] == 'sham'].set_index('subject_id')
        if len(rw_sham) > 0:
            rw_sham = rw_sham[['alpha', 'beta']]
            rw_sham = rw_sham.rename(columns={'alpha': 'sham_alpha', 'beta': 'sham_beta'})
            subj_df = subj_df.merge(rw_sham, on='subject_id', how='left')
        
        # Active (H2-eligible only)
        rw_active = rw_mle[rw_mle['condition'] == 'active']
        if h2_subjects is not None:
            rw_active = rw_active[rw_active['subject_id'].isin(h2_subjects)]
        if len(rw_active) > 0:
            rw_active = rw_active.set_index('subject_id')[['alpha', 'beta']]
            rw_active = rw_active.rename(columns={'alpha': 'active_alpha', 'beta': 'active_beta'})
            subj_df = subj_df.merge(rw_active, on='subject_id', how='left')
            
            if 'sham_alpha' in subj_df.columns:
                subj_df['delta_alpha'] = subj_df['active_alpha'] - subj_df['sham_alpha']
                subj_df['delta_beta'] = subj_df['active_beta'] - subj_df['sham_beta']
    
    # --- DDM parameters ---
    if ddm_params is not None and len(ddm_params) > 0:
        ddm_cols = ['subject_id'] + [c for c in ddm_params.columns if c.startswith('ddm_')]
        subj_df = subj_df.merge(ddm_params[ddm_cols], on='subject_id', how='left')
    
    if verbose:
        print(f'Subject DataFrame assembled: {len(subj_df)} subjects, {len(subj_df.columns)} variables')
        
        # Summarize coverage
        n_with_cog = subj_df['global_composite'].notna().sum()
        n_with_wsls = subj_df.get('sham_p_stay_win', pd.Series()).notna().sum()
        n_with_rw = subj_df.get('sham_alpha', pd.Series()).notna().sum()
        
        print(f'  Cognitive composites: {n_with_cog}')
        print(f'  WSLS (sham): {n_with_wsls}')
        print(f'  R-W (sham): {n_with_rw}')
    
    return subj_df


# =============================================================================
# Main Pipeline
# =============================================================================

def run_cognitive_merge(
    wsls_h1: Optional[pd.DataFrame] = None,
    wsls_h2: Optional[pd.DataFrame] = None,
    rw_mle: Optional[pd.DataFrame] = None,
    ddm_params: Optional[pd.DataFrame] = None,
    h2_subjects: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict:
    """
    Run complete cognitive merge pipeline.
    
    Parameters
    ----------
    wsls_h1, wsls_h2 : DataFrame, optional
        WSLS parameters
    rw_mle : DataFrame, optional
        R-W parameters
    ddm_params : DataFrame, optional
        DDM parameters
    h2_subjects : list, optional
        H2-eligible subjects
    verbose : bool
        If True, print summaries
    
    Returns
    -------
    dict with keys:
        'rf1_cog': RF1 FullScoring data
        'rf1_neuro': RF1 Raw data
        'tabcat': TabCAT data
        'tacs_redcap': tACS REDCap data
        'cog_composites': cognitive composites
        'subj_df': master subject DataFrame
    """
    study_subjects = list(SUBJECT_INFO.keys())
    results = {}
    
    if verbose:
        print('='*70)
        print('Loading External Data Sources')
        print('='*70)
    
    # Load sources
    try:
        rf1_cog = load_rf1_fullscoring(study_subjects=study_subjects)
        results['rf1_cog'] = rf1_cog
        if verbose:
            print(f'  RF1 FullScoring: {len(rf1_cog)} subjects')
    except FileNotFoundError:
        if verbose:
            print('  RF1 FullScoring: NOT FOUND')
        rf1_cog = pd.DataFrame({'subject_id': study_subjects})
    
    try:
        rf1_neuro = load_rf1_raw(study_subjects=study_subjects)
        results['rf1_neuro'] = rf1_neuro
        if verbose:
            print(f'  RF1 Raw: {len(rf1_neuro)} subjects')
    except FileNotFoundError:
        if verbose:
            print('  RF1 Raw: NOT FOUND')
        rf1_neuro = pd.DataFrame({'subject_id': study_subjects})
    
    try:
        tabcat = load_tabcat(study_subjects=study_subjects)
        results['tabcat'] = tabcat
        if verbose:
            print(f'  TabCAT: {len(tabcat)} subjects')
    except FileNotFoundError:
        if verbose:
            print('  TabCAT: NOT FOUND')
        tabcat = pd.DataFrame({'subject_id': study_subjects})
    
    try:
        tacs_redcap = load_tacs_redcap(study_subjects=study_subjects)
        results['tacs_redcap'] = tacs_redcap
        if verbose:
            print(f'  tACS REDCap: {len(tacs_redcap)} subjects')
    except FileNotFoundError:
        if verbose:
            print('  tACS REDCap: NOT FOUND')
        tacs_redcap = pd.DataFrame({'subject_id': study_subjects})
    
    # Compute composites
    if verbose:
        print('\n' + '-'*50)
        print('Computing Cognitive Composites')
        print('-'*50)
    
    cog = compute_cognitive_composites(rf1_cog, rf1_neuro, tabcat, study_subjects)
    results['cog_composites'] = cog
    
    if verbose:
        for comp in ['attention_composite', 'memory_composite', 'speed_composite',
                     'global_composite', 'ef_composite']:
            n = cog[comp].notna().sum()
            print(f'  {comp}: {n}/{len(cog)} with data')
    
    # Build master DataFrame
    if verbose:
        print('\n' + '-'*50)
        print('Building Subject DataFrame')
        print('-'*50)
    
    subj_df = build_subject_df(
        study_subjects=study_subjects,
        wsls_h1=wsls_h1,
        wsls_h2=wsls_h2,
        rw_mle=rw_mle,
        ddm_params=ddm_params,
        h2_subjects=h2_subjects,
        verbose=verbose
    )
    results['subj_df'] = subj_df
    
    return results


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing cognitive_merge module...")
    print("Functions: load_rf1_fullscoring, load_rf1_raw, load_tabcat, load_tacs_redcap")
    print("           compute_cognitive_composites, build_subject_df, run_cognitive_merge")
