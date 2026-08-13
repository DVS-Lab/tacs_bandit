"""
cognitive_merge.py — External data integration for tACS Bandit study

Loads and scores survey/cognitive measures from multiple sources:
1. RF1 FullScoring — HVLT, Salthouse, SCD-Q, SUSD, SCAARED, etc.
2. RF1 Raw REDCap — Digit Span, BVMT, Trail Making, AUDIT, DUDIT, CTQ-SF,
                    PROMIS, UCLA Loneliness
3. TabCAT — Flanker, RunningDots, Set Shifting
4. tACS Bandit REDCap — SPSRQ, B-PSQI, BBS, CRT, FFMQ-15

Domain composites per pre-registration (Section 4):
- Attention: Digit Span + RunningDots + Flanker
- Episodic Memory: HVLT Total + BVMT Total
- Processing Speed: Salthouse Letter + Pattern + Trails A
- Global: average of domain composites
- Executive Function (secondary): RunningDots + Flanker + Trails B−A

Measure tiers:
- PRIMARY: Pre-registered cognitive composites, SPSRQ
- THEORETICALLY RELEVANT: Substance use (AUDIT, DUDIT), mood/depression
  (SUSD, PROMIS Depression), anxiety (SCAARED, PROMIS Anxiety), loneliness
  (UCLA-3), social support (MSPSS), trauma (CTQ-SF), impulsivity (BIS-11),
  sleep (B-PSQI, PROMIS Sleep), gambling (SOGS, NORC/NODS), emotion
  regulation (EROS), social media (BSMAS, NBS), fatigue (PROMIS),
  physical function (PROMIS), pain (PROMIS), cognitive reflection (CRT),
  mindfulness (FFMQ), bias blind spot (BBS)
- EXPLORATORY: TEI, AQ, PANAS, IOS, BPAQ, PNR, Mach-IV, BSAS, Planfulness,
  Present Bias, GS, GTI, OAFEM, PAM, financial wellbeing, USI, FEVS,
  food addiction (YFAS), ECog

All raw scores z-transformed within sample before averaging for composites.
Missing values coded as NaN throughout.
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
    TABCAT_LEGACY_PATHS,
    DATA_DIR,
)


# =============================================================================
# Constants
# =============================================================================

MASTER_CSV_FILENAME = 'master_subject_data.csv'


# =============================================================================
# RF1 FullScoring Loader
# =============================================================================

def load_rf1_fullscoring(
    filepath: str = REDCAP_RF1_PATH,
    study_subjects: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load pre-scored cognitive measures from RF1 FullScoring export.

    Returns DataFrame with subject_id plus all primary cognitive vars.
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

    cols_present = ['subject_id'] + [c for c in col_map.keys() if c in rf1.columns]
    rf1_cog = rf1[cols_present].copy()
    rf1_cog = rf1_cog.rename(columns=col_map)
    rf1_cog = rf1_cog[rf1_cog['subject_id'].isin(study_subjects)].copy()

    return rf1_cog


def load_rf1_extended(
    filepath: str = REDCAP_RF1_PATH,
    study_subjects: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load theoretically relevant pre-scored measures from RF1 FullScoring.

    Includes mood, anxiety, substance use/gambling, social media, emotion
    regulation, and social support.
    """
    if study_subjects is None:
        study_subjects = list(SUBJECT_INFO.keys())

    rf1 = pd.read_csv(filepath, low_memory=False)
    rf1['subject_id'] = rf1['Subject ID'].astype(str)

    # Theoretically relevant measures
    col_map = {
        # Mood / Internalizing
        'SUSD Depression Scale (sum)': 'susd_depression',
        'SUSD Mania Scale (sum)': 'susd_mania',
        # Anxiety (SCAARED subscales)
        'SCAARED - Total Score (Sum)': 'scaared_total',
        'SCAARED - Somatic/Panic/Agoraphibia Score (Sum)': 'scaared_somatic',
        'SCAARED - Generalized Anxiety Disorder Score (Sum)': 'scaared_gad',
        'SCAARED - Separation Anxiety Disorder Score (Sum)': 'scaared_separation',
        'SCAARED - Social Anxiety Disorder Score (Sum)': 'scaared_social',
        # Gambling
        'SOGS total score': 'sogs_total',
        'NORC(NODS) lifetime total score': 'norc_lifetime',
        'NORC(NODS) past year total score': 'norc_past_year',
        # Social media / internet
        'BSMAS Total (sum) ': 'bsmas_total',
        'Sum Score NBS (Range from 1-5)': 'nbs_total',
        # Emotion Regulation
        'EROS External Improving Score': 'eros_ext_improving',
        'EROS External Worsening Score': 'eros_ext_worsening',
        'EROS Internal Improving Score': 'eros_int_improving',
        'EROS Internal Worsening Score': 'eros_int_worsening',
        # Social support
        'MSPSS Retest Sum': 'mspss_total',
        'MSPSS Friends Sub-Scale': 'mspss_friends',
        'MSPSS Family Sub-Scale': 'mspss_family',
        'MSPSS Significant Other Sub-Scale': 'mspss_significant_other',
        # Aggression (BPAQ)
        'BPAQ Total Score (Sum) ': 'bpaq_total',
        'BPAQ Physical Subscale (Sum) ': 'bpaq_physical',
        'BPAQ Anger Subscale (Sum) ': 'bpaq_anger',
        'BPAQ Verbal Subscale (Sum)': 'bpaq_verbal',
        'BPAQ Hostility Subscale (Sum)': 'bpaq_hostility',
    }

    cols_present = ['subject_id'] + [c for c in col_map.keys() if c in rf1.columns]
    rf1_ext = rf1[cols_present].copy()
    rf1_ext = rf1_ext.rename(columns=col_map)
    rf1_ext = rf1_ext[rf1_ext['subject_id'].isin(study_subjects)].copy()

    return rf1_ext


def load_rf1_exploratory(
    filepath: str = REDCAP_RF1_PATH,
    study_subjects: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load exploratory pre-scored measures from RF1 FullScoring.

    TEI, AQ, PANAS, IOS, PNR, Mach-IV, Planfulness, Present Bias, GS, GTI.
    """
    if study_subjects is None:
        study_subjects = list(SUBJECT_INFO.keys())

    rf1 = pd.read_csv(filepath, low_memory=False)
    rf1['subject_id'] = rf1['Subject ID'].astype(str)

    col_map = {
        # Emotional intelligence (TEI)
        'TEI Subscale wellbeing': 'tei_wellbeing',
        'TEI Subscale selfcontrol': 'tei_selfcontrol',
        'TEI Subscale emotionality': 'tei_emotionality',
        'TEI Subscale sociability': 'tei_sociability',
        'TEI Total or global trait emotional intelligence': 'tei_total',
        # Autism Quotient (AQ)
        'AQ subscale Social Skill': 'aq_social_skill',
        'AQ subscale Attention Switching': 'aq_attention_switching',
        'AQ subscale Attention to detail': 'aq_attention_detail',
        'AQ subscale Communication': 'aq_communication',
        'AQ subscale Imagination': 'aq_imagination',
        'AQ Total': 'aq_total',
        # PANAS (pre-scan)
        'PANAS Prescan Positive Total': 'panas_pre_positive',
        'PANAS Prescan Negative Total': 'panas_pre_negative',
        'PANAS_Postscan Positive Total': 'panas_post_positive',
        'PANAS Postscan Negative Total': 'panas_post_negative',
        # IOS (social closeness)
        'IOS investment stranger': 'ios_investment_stranger',
        'IOS shared reward stranger': 'ios_shared_reward_stranger',
        'IOS friend': 'ios_friend',
        'IOS computer': 'ios_computer',
        'IOS stranger': 'ios_stranger',
        # Reciprocity
        'PNR p Total': 'pnr_positive',
        'PNR n Total': 'pnr_negative',
        # Machiavellianism
        'Mach-iv total': 'mach_iv_total',
        # Planfulness
        'Planfullness Mental Flexibility Score': 'planfulness_mf',
        'Planfullness Temporal Orientation Score': 'planfulness_to',
        'Planfullness Cognitive Strategies Score': 'planfulness_cs',
        'Planfullness Total Score': 'planfulness_total',
        # Present bias
        'Today vs 5 Weeks (Low score = High Present Bias)': 'present_bias_5wk',
        'Today vs 9 Weeks (Low score = High Present Bias)': 'present_bias_9wk',
        '5 Weeks vs 10 Weeks (Low score = High Present Bias)': 'present_bias_10wk',
        '5 Weeks vs 14 Weeks (Low score = High Present Bias)': 'present_bias_14wk',
        # Gullibility
        'GS Short Form Total': 'gullibility_total',
        'GS Insensitivity Scale': 'gullibility_insensitivity',
        # General trust
        'GTI score': 'gti_score',
        # CRT (from RF1)
        'CRT total score': 'rf1_crt_total',
    }

    cols_present = ['subject_id'] + [c for c in col_map.keys() if c in rf1.columns]
    rf1_exp = rf1[cols_present].copy()
    rf1_exp = rf1_exp.rename(columns=col_map)
    rf1_exp = rf1_exp[rf1_exp['subject_id'].isin(study_subjects)].copy()

    return rf1_exp


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

    # Ensure numeric columns are properly typed (REDCap exports can store numbers as strings)
    numeric_cols = [c for c in rf1_neuro.columns if c != 'subject_id']
    for col in numeric_cols:
        rf1_neuro[col] = pd.to_numeric(rf1_neuro[col], errors='coerce')

    # Some subjects have multiple rows; take first non-null per subject
    rf1_neuro = rf1_neuro.groupby('subject_id').first().reset_index()
    rf1_neuro = rf1_neuro[rf1_neuro['subject_id'].isin(study_subjects)].copy()

    return rf1_neuro


def load_rf1_substance_mood(
    filepath: str = REDCAP_RF1_RAW_PATH,
    study_subjects: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Load substance use, mood, and health measures from RF1 Raw export.

    Extracts from specific REDCap events:
    - Subject Information event: AUDIT, DUDIT, CTQ-SF
    - Mock Scan Questionnaires event: PROMIS subscales, UCLA Loneliness

    Returns dict with keys: 'audit_dudit', 'promis', 'loneliness'
    """
    if study_subjects is None:
        study_subjects = list(SUBJECT_INFO.keys())

    rf1_raw = pd.read_csv(filepath, low_memory=False)
    rf1_raw['subject_id'] = rf1_raw['Subject ID'].astype(str)

    results = {}

    # ---- AUDIT, DUDIT, CTQ-SF (Subject Information event) ----
    si_event = rf1_raw[rf1_raw['Event Name'] == 'Subject Information'].copy()
    audit_dudit_cols = {
        'AUDIT (Sum)': 'audit_total',
        'DUDIT (Sum)': 'dudit_total',
        'CTQ-SF Total Score (Sum) - CUTOFF': 'ctq_total',
        # Note: CTQ subscale columns (Emotional Abuse, Physical Abuse, etc.)
        # have only 2 non-null values in the entire RF1 Raw CSV and are not extracted.
    }

    cols_present = ['subject_id'] + [c for c in audit_dudit_cols.keys() if c in si_event.columns]
    audit_dudit = si_event[cols_present].copy()
    audit_dudit = audit_dudit.rename(columns=audit_dudit_cols)
    audit_dudit = audit_dudit.groupby('subject_id').first().reset_index()
    audit_dudit = audit_dudit[audit_dudit['subject_id'].isin(study_subjects)].copy()
    results['audit_dudit'] = audit_dudit

    if verbose:
        print(f'  AUDIT/DUDIT/CTQ-SF: {len(audit_dudit)} subjects')
        for col in audit_dudit_cols.values():
            if col in audit_dudit.columns:
                n = audit_dudit[col].notna().sum()
                print(f'    {col}: {n}/{len(audit_dudit)}')

    # ---- PROMIS (Mock Scan Questionnaires event) ----
    mock_event = rf1_raw[rf1_raw['Event Name'] == 'Mock Scan Questionnaires'].copy()
    mock_study = mock_event[mock_event['subject_id'].isin(study_subjects)].copy()

    promis_scores = _score_promis(mock_study, study_subjects)
    results['promis'] = promis_scores

    if verbose:
        promis_cols = [c for c in promis_scores.columns if c.startswith('promis_')]
        print(f'  PROMIS: {len(promis_scores)} subjects')
        for col in promis_cols:
            n = promis_scores[col].notna().sum()
            print(f'    {col}: {n}/{len(promis_scores)}')

    # ---- UCLA Loneliness (3-item) ----
    # These items are administered at the follow-up appointment, not the mock
    # scan. Reading them from the mock-scan frame returns no rows at all, so
    # every loneliness column came out empty for every subject while looking
    # like a legitimately missing measure. Search the whole export instead of
    # assuming an event, so a future change of schedule does not silently
    # empty the column again.
    loneliness_items = {
        'How often do you feel that you lack companionship?': 'loneliness_companionship',
        'How often do you feel left out?': 'loneliness_left_out',
        'How often do you feel Isolated from others?': 'loneliness_isolated',
    }

    loneliness_map = {
        'Hardly ever': 1, 'Some of the time': 2, 'Often': 3,
    }

    lone_source = rf1_raw[rf1_raw['subject_id'].isin(study_subjects)].copy()
    lone_cols_present = ['subject_id'] + [c for c in loneliness_items.keys()
                                          if c in lone_source.columns]
    loneliness = lone_source[lone_cols_present].copy()
    # Keep the row that actually carries the responses, whichever event it is.
    item_cols = [c for c in lone_cols_present if c != 'subject_id']
    if item_cols:
        loneliness = loneliness[loneliness[item_cols].notna().any(axis=1)]
    loneliness = loneliness.rename(columns=loneliness_items)
    loneliness = loneliness.groupby('subject_id').first().reset_index()

    # Apply response map
    for col in loneliness_items.values():
        if col in loneliness.columns:
            loneliness[col] = loneliness[col].map(loneliness_map)

    lone_score_cols = [c for c in loneliness_items.values() if c in loneliness.columns]
    if lone_score_cols:
        loneliness['loneliness_total'] = loneliness[lone_score_cols].sum(axis=1, min_count=1)

    loneliness = loneliness[loneliness['subject_id'].isin(study_subjects)].copy()
    results['loneliness'] = loneliness

    if verbose:
        n_lone = loneliness['loneliness_total'].notna().sum() if 'loneliness_total' in loneliness.columns else 0
        print(f'  UCLA Loneliness: {n_lone}/{len(loneliness)}')

    return results


def _score_promis(
    mock_data: pd.DataFrame,
    study_subjects: List[str]
) -> pd.DataFrame:
    """Score PROMIS short-form subscales from Mock Scan event data."""

    # Response maps
    frequency_map = {
        'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Always': 5,
    }
    intensity_map = {
        'Not at all': 1, 'A little bit': 2, 'Somewhat': 3,
        'Quite a bit': 4, 'Very much': 5,
    }
    sleep_map = {
        'Very Poor': 1, 'Poor': 2, 'Fair': 3, 'Good': 4, 'Very Good': 5,
    }
    physical_map = {
        'Unable to do': 1, 'With much difficulty': 2,
        'With some difficulty': 3, 'With little difficulty': 4,
        'Without any difficulty': 5,
    }

    def score_pain(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        try:
            return float(s[0])
        except (ValueError, IndexError):
            return np.nan

    # Subscale definitions
    promis_defs = {
        'promis_anxiety': {
            'items': [
                'I felt fearful',
                'I found it hard to focus on anything other than my anxiety',
                'My worries overwhelmed me',
                'I felt uneasy',
            ],
            'map': frequency_map,
        },
        'promis_depression': {
            'items': [
                'I felt worthless',
                'I felt helpless',
                'I felt depressed',
                'I felt hopeless',
            ],
            'map': frequency_map,
        },
        'promis_fatigue': {
            'items': [
                'I feel fatigued',
                'How run down did you feel on average?',
                'How fatigued were you on average?',
            ],
            'map': 'fatigue_special',
        },
        'promis_sleep': {
            'items': ['In the past 7 days my sleep quality was'],
            'map': sleep_map,
        },
        'promis_social': {
            'items': [
                'I have trouble doing all of my regular leisure activities with others',
                'I have trouble doing all of the family activities that I want to do',
                'I have trouble doing all of my usual work (include work at home)',
                'I have trouble doing all of the activities with friends that I want to do',
            ],
            'map': frequency_map,
        },
        'promis_pain': {
            'items': ['How would you rate your pain on average?'],
            'map': 'pain_special',
        },
        'promis_physical': {
            'items': [
                'Are you able to do chores such as vacuuming or yard work?',
                'Are you able to go up and down stairs at a normal pace?',
                'Are you able to go for a walk of at least 15 minutes?',
                'Are you able to run errands and shop?',
            ],
            'map': physical_map,
        },
    }

    promis_scores = mock_data[['subject_id']].drop_duplicates().copy()

    for subscale, spec in promis_defs.items():
        items = spec['items']
        rmap = spec['map']
        valid_items = [c for c in items if c in mock_data.columns]

        if not valid_items:
            promis_scores[subscale] = np.nan
            continue

        sub_data = mock_data.groupby('subject_id')[valid_items].first().reset_index()

        for col in valid_items:
            if rmap == 'pain_special':
                sub_data[col] = sub_data[col].apply(score_pain)
            elif rmap == 'fatigue_special':
                if col == 'I feel fatigued':
                    sub_data[col] = sub_data[col].map(frequency_map)
                else:
                    sub_data[col] = sub_data[col].map(intensity_map)
            else:
                sub_data[col] = sub_data[col].map(rmap)

        sub_data[subscale] = sub_data[valid_items].sum(axis=1, min_count=1)
        promis_scores = promis_scores.merge(
            sub_data[['subject_id', subscale]], on='subject_id', how='left'
        )

    promis_scores = promis_scores[promis_scores['subject_id'].isin(study_subjects)].copy()
    return promis_scores


# =============================================================================
# TabCAT Loader
# =============================================================================

def load_tabcat(
    filepath: str = TABCAT_PATH,
    study_subjects: Optional[List[str]] = None,
    legacy_paths: Optional[List] = None,
) -> pd.DataFrame:
    """
    Load TabCAT cognitive measures (Flanker, RunningDots, SetShifting).

    Successive TabCAT exports are not nested: the August 2026 export covers
    five more subjects than February but drops 36 columns. Taking either alone
    loses something, so the newest is used as the base and older exports fill
    in subjects and columns it lacks. Values from the newest export always win
    where both have one.
    """
    if study_subjects is None:
        study_subjects = list(SUBJECT_INFO.keys())
    if legacy_paths is None:
        legacy_paths = TABCAT_LEGACY_PATHS

    tabcat = pd.read_csv(filepath)
    tabcat['subject_id'] = tabcat['Examinee_Identifier'].astype(str)

    for legacy in legacy_paths:
        if not Path(legacy).exists():
            continue
        old = pd.read_csv(legacy, low_memory=False)
        old['subject_id'] = old['Examinee_Identifier'].astype(str)

        # Columns the current export no longer carries.
        extra_cols = [c for c in old.columns if c not in tabcat.columns]
        if extra_cols:
            tabcat = tabcat.merge(old[['subject_id'] + extra_cols],
                                  on='subject_id', how='left')

        # Subjects absent from the current export.
        missing = old[~old['subject_id'].isin(set(tabcat['subject_id']))]
        if len(missing):
            tabcat = pd.concat([tabcat, missing], ignore_index=True)

    col_map = {
        'Flanker_TotalScore': 'flanker_score',
        'Flanker_Correct_MedianRT': 'flanker_rt',
        'Flanker_IncongrCorrect_MedianRT': 'flanker_incongruent_rt',
        'RunningDots_TrialScore': 'running_dots_score',
        'RunningDots_PercentCorrect': 'running_dots_pct',
        'SetShifting_TotalScore': 'set_shifting_score',
        'SetShifting_ShiftCorrect_Total': 'set_shifting_shift_correct',
        'Examinee_Education': 'tabcat_education_years',
    }

    cols_present = ['subject_id'] + [c for c in col_map.keys() if c in tabcat.columns]
    tc = tabcat[cols_present].copy()
    tc = tc.rename(columns=col_map)

    tc = tc.drop_duplicates(subset='subject_id', keep='last')
    tc = tc[tc['subject_id'].isin(study_subjects)].copy()

    return tc


# =============================================================================
# SPSRQ Scoring
# =============================================================================

def score_spsrq(redcap_tacs: pd.DataFrame) -> pd.DataFrame:
    """
    Score SPSRQ (Revised & Clarified, 20-item, 5-point Likert).

    Odd items (1,3,...,19) → SP (Sensitivity to Punishment)
    Even items (2,4,...,20) → SR (Sensitivity to Reward)
    """
    df = redcap_tacs.copy()

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
        'Very Untrue': 1, 'Somewhat Untrue': 2, 'Neither Untrue nor True': 3,
        'Somewhat True': 4, 'Very True': 5,
    }

    spsrq_cols = []
    for item in spsrq_items:
        matches = [c for c in df.columns if c.startswith(item[:40])]
        spsrq_cols.append(matches[0] if matches else None)

    for col in spsrq_cols:
        if col is not None and col in df.columns:
            df[col] = df[col].map(likert_map)

    sp_cols = [spsrq_cols[i] for i in range(0, 20, 2) if spsrq_cols[i] is not None]
    sr_cols = [spsrq_cols[i] for i in range(1, 20, 2) if spsrq_cols[i] is not None]

    df['spsrq_sp'] = df[sp_cols].sum(axis=1, min_count=1) if sp_cols else np.nan
    df['spsrq_sr'] = df[sr_cols].sum(axis=1, min_count=1) if sr_cols else np.nan
    df['spsrq_sp_n_items'] = df[sp_cols].notna().sum(axis=1) if sp_cols else 0
    df['spsrq_sr_n_items'] = df[sr_cols].notna().sum(axis=1) if sr_cols else 0

    return df


# =============================================================================
# tACS REDCap Loader + Survey Scoring
# =============================================================================

def load_tacs_redcap(
    filepath: str = REDCAP_TACS_PATH,
    study_subjects: Optional[List[str]] = None,
    score_surveys: bool = True
) -> pd.DataFrame:
    """Load tACS Bandit REDCap export with survey scoring."""
    if study_subjects is None:
        study_subjects = list(SUBJECT_INFO.keys())

    redcap = pd.read_csv(filepath, low_memory=False)
    redcap['subject_id'] = redcap['Record ID'].astype(str)

    if score_surveys:
        redcap = score_spsrq(redcap)
        redcap = score_bpsqi(redcap)
        redcap = score_bbs(redcap)
        redcap = score_crt(redcap)
        redcap = score_ffmq(redcap)

    redcap = redcap[redcap['subject_id'].isin(study_subjects)].copy()

    return redcap


def score_bpsqi(redcap_tacs: pd.DataFrame) -> pd.DataFrame:
    """Score Brief Pittsburgh Sleep Quality Index (B-PSQI) — 4 components."""
    df = redcap_tacs.copy()

    bpsqi_cols = {
        'latency': 'How long has it usually taken you to fall asleep each night?',
        'duration': 'How many hours of actual sleep did you get at night?',
        'disturbance': 'Have you had trouble sleeping because you wake up in the middle of the night or early morning?',
        'quality': 'How would you rate your sleep quality overall?',
    }

    bpsqi_col_map = {}
    for key, label in bpsqi_cols.items():
        matches = [c for c in df.columns if c.startswith(label[:50])]
        if matches:
            bpsqi_col_map[key] = matches[0]

    scoring_maps = {
        'latency': {'Less than 15': 0, '15 to 30 minutes': 1, '30 to 60 minutes': 2, 'More than 60 minutes': 3},
        'duration': {'More than 7 hours': 0, 'Between 6 and 7 hours': 1, 'Less than 7 hours': 1,
                     'Between 5 and 6 hours': 2, 'Less than 5 hours': 3},
        'disturbance': {'Not during the past month': 0, 'Less than once a week': 1,
                        'Once or twice a week': 2, 'Three or more times a week': 3},
        'quality': {'Very good': 0, 'Fairly good': 1, 'Fairly bad': 2, 'Very bad': 3},
    }

    for key, smap in scoring_maps.items():
        if key in bpsqi_col_map:
            df[f'bpsqi_{key}'] = df[bpsqi_col_map[key]].map(smap)

    component_cols = [f'bpsqi_{k}' for k in scoring_maps if f'bpsqi_{k}' in df.columns]
    df['bpsqi_global'] = df[component_cols].sum(axis=1, min_count=1)
    df['bpsqi_n_items'] = df[component_cols].notna().sum(axis=1)

    return df


def score_bbs(redcap_tacs: pd.DataFrame) -> pd.DataFrame:
    """Score Bias Blind Spot (BBS) — 14 biases × 2 ratings (self vs. others).

    Each item is rated on a 1-7 scale. The REDCap labels export shows
    endpoint text ('Not much at all' = 1, 'Very much' = 7) but intermediate
    values (2-6) may appear as numeric strings. We handle both cases.

    BBS score = mean(self - other) across all biases.
    Positive = sees bias more in others than self (bias blind spot).
    """
    df = redcap_tacs.copy()

    bbs_response_map = {
        'Not much at all': 1,
        'Very much': 7,
        # Intermediate values may appear as numeric strings in labels export
        '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
    }

    bbs_labels = [
        'tendency to judge a harmful action',
        'tendency to do or believe a thing only because many other',
        'react to counterevidence by actually strengthening',
        'disconfirmation tendency',
        'diffusion of responsibility',
        'irrational decisions to justify actions',
        'overly dispositional inferences',
        'halo effect',
        'automatic tendency to be less generous',
        'ostrich effect',
        'underestimate the impact or the strength',
        'self-interest effect',
        'self-serving tendency',
        'gender biases lead people to associate',
    ]

    bbs_self_cols = []
    bbs_other_cols = []
    for label in bbs_labels:
        self_m = [c for c in df.columns if label.lower() in c.lower() and not c.endswith('.1')]
        other_m = [c for c in df.columns if label.lower() in c.lower() and c.endswith('.1')]
        bbs_self_cols.append(self_m[0] if self_m else None)
        bbs_other_cols.append(other_m[0] if other_m else None)

    for col in bbs_self_cols + bbs_other_cols:
        if col is not None and col in df.columns:
            mapped = df[col].map(bbs_response_map)
            # Fallback: try numeric coercion for values the map didn't catch
            unmapped = mapped.isna() & df[col].notna()
            if unmapped.any():
                mapped.loc[unmapped] = pd.to_numeric(df.loc[unmapped, col], errors='coerce')
            df[col + '_num'] = mapped

    bbs_scores = []
    for i in range(len(bbs_labels)):
        s, o = bbs_self_cols[i], bbs_other_cols[i]
        if s and o:
            sn, on = s + '_num', o + '_num'
            if sn in df.columns and on in df.columns:
                sc = f'bbs_bias_{i+1}'
                df[sc] = df[sn] - df[on]
                bbs_scores.append(sc)

    if bbs_scores:
        df['bbs_avg'] = df[bbs_scores].mean(axis=1, skipna=True)
        df['bbs_n_items'] = df[bbs_scores].notna().sum(axis=1)

    return df


def score_crt(redcap_tacs: pd.DataFrame) -> pd.DataFrame:
    """Score Cognitive Reflection Test (6-item)."""
    df = redcap_tacs.copy()

    crt_items = [
        ('A bat and a ball cost $1.10 in total', ['5', '0.05', '.05', '$0.05', '5 cents', 'five cents', '5c']),
        ('If it takes 5 machines 5 minutes', ['5', '5 minutes', 'five', 'five minutes']),
        ('In a lake, there is a patch of lily pads', ['47', '47 days', 'forty-seven', 'forty seven']),
        ('If three elves can wrap three toys', ['3', '3 elves', 'three', 'three elves']),
        ('Jerry received both the 15th highest', ['29', '29 students', 'twenty-nine', 'twenty nine']),
        ('In an athletics team, tall members', ['15', '15 medals', 'fifteen']),
    ]

    crt_item_scores = []
    for i, (label, correct) in enumerate(crt_items):
        matches = [c for c in df.columns if c.startswith(label[:40])]
        if matches:
            col = matches[0]
            sc = f'crt_item_{i+1}'
            responses = df[col].astype(str).str.lower().str.strip()
            pattern = '|'.join([f'^{a.lower()}$' for a in correct])
            df[sc] = responses.str.match(pattern, na=False).astype(int)
            crt_item_scores.append(sc)

    if crt_item_scores:
        df['crt_total'] = df[crt_item_scores].sum(axis=1)
        df['crt_n_items'] = df[crt_item_scores].notna().sum(axis=1)

    return df


def score_ffmq(redcap_tacs: pd.DataFrame) -> pd.DataFrame:
    """Score Five Facet Mindfulness Questionnaire (FFMQ-15)."""
    df = redcap_tacs.copy()

    ffmq_labels = [
        'When I take a shower or a bath, I stay alert',
        "I'm good at finding words to describe my feelings",
        "I don't pay attention to what I'm doing because I'm daydreaming",
        "I believe some of my thoughts are abnormal or bad",
        "When I have distressing thoughts or images, I step back",
        'I notice how foods and drinks affect my thoughts',
        'I have trouble thinking of the right words to express',
        'I do jobs or tasks automatically without being aware',
        'I think some of my emotions are bad or inappropriate',
        'When I have distressing thoughts or images I am able just to notice',
        'I pay attention to sensations, such as the wind',
        "Even when I'm feeling terribly upset I can find a way",
        'I find myself doing things without paying attention',
        "I tell myself I shouldn't be feeling the way",
        'When I have distressing thoughts or images I just notice them and let',
    ]

    ffmq_response_map = {
        'Never, or very rarely true': 1, 'Rarely true': 2,
        'Sometimes true': 3, 'Often true': 4, 'Very often, or always true': 5,
    }

    ffmq_cols = []
    for label in ffmq_labels:
        matches = [c for c in df.columns if c.startswith(label[:40])]
        ffmq_cols.append(matches[0] if matches else None)

    for col in ffmq_cols:
        if col is not None and col in df.columns:
            df[col + '_num'] = df[col].map(ffmq_response_map)

    ffmq_num = [col + '_num' if col else None for col in ffmq_cols]

    # Reverse-score: items 3,4,7,8,9,13,14 (0-indexed: 2,3,6,7,8,12,13)
    for i in [2, 3, 6, 7, 8, 12, 13]:
        if ffmq_num[i] and ffmq_num[i] in df.columns:
            df[ffmq_num[i]] = 6 - df[ffmq_num[i]]

    subscales = {
        'ffmq_observe': [0, 5, 10],
        'ffmq_describe': [1, 6, 11],
        'ffmq_actaware': [2, 7, 12],
        'ffmq_nonjudge': [3, 8, 13],
        'ffmq_nonreact': [4, 9, 14],
    }

    for subscale, indices in subscales.items():
        cols = [ffmq_num[i] for i in indices if ffmq_num[i] and ffmq_num[i] in df.columns]
        if cols:
            df[subscale] = df[cols].mean(axis=1, skipna=True)

    all_ffmq = [c for c in ffmq_num if c and c in df.columns]
    if all_ffmq:
        df['ffmq_total'] = df[all_ffmq].mean(axis=1, skipna=True)
        df['ffmq_n_items'] = df[all_ffmq].notna().sum(axis=1)

    # Total minus Observing
    no_obs_indices = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]
    no_obs_cols = [ffmq_num[i] for i in no_obs_indices if ffmq_num[i] and ffmq_num[i] in df.columns]
    if no_obs_cols:
        df['ffmq_total_no_obs'] = df[no_obs_cols].mean(axis=1, skipna=True)

    return df


def extract_demographics(redcap_tacs: pd.DataFrame) -> pd.DataFrame:
    """Extract demographic variables from tACS REDCap export."""
    cols = ['subject_id']
    col_renames = {}

    rename_map = {
        'Age in years:': 'age',
        'Gender': 'gender',
        'Race': 'race',
        'Ethnicity': 'ethnicity',
    }

    for orig, new in rename_map.items():
        if orig in redcap_tacs.columns:
            cols.append(orig)
            col_renames[orig] = new

    # SPSRQ scores
    for col in ['spsrq_sp', 'spsrq_sr', 'spsrq_sp_n_items', 'spsrq_sr_n_items']:
        if col in redcap_tacs.columns:
            cols.append(col)

    # Additional survey scores
    survey_cols = [
        'bpsqi_global', 'bpsqi_latency', 'bpsqi_duration', 'bpsqi_disturbance',
        'bpsqi_quality', 'bpsqi_n_items',
        'bbs_avg', 'bbs_n_items',
        'crt_total', 'crt_n_items',
        'ffmq_total', 'ffmq_total_no_obs', 'ffmq_observe', 'ffmq_describe',
        'ffmq_actaware', 'ffmq_nonjudge', 'ffmq_nonreact', 'ffmq_n_items',
    ]
    for col in survey_cols:
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

    Method: raw → z-transform → average per domain.
    Time-based measures (Trails) are sign-flipped (lower = better).
    """
    if study_subjects is None:
        study_subjects = list(SUBJECT_INFO.keys())

    cog = pd.DataFrame({'subject_id': study_subjects})

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

    # Z-transform
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

    # Flip time-based
    for time_col in ['trails_a_time', 'trails_b_time']:
        z_name = z_cols.get(time_col)
        if z_name and z_name in cog.columns:
            cog[z_name] = -cog[z_name]

    # Attention
    att_cols = [z_cols[m] for m in ['digit_span_total', 'running_dots_score', 'flanker_score']
                if m in z_cols and z_cols[m] in cog.columns]
    cog['attention_composite'] = cog[att_cols].mean(axis=1) if att_cols else np.nan
    cog['attention_n_measures'] = cog[att_cols].notna().sum(axis=1) if att_cols else 0

    # Episodic Memory
    mem_cols = [z_cols[m] for m in ['hvlt_total', 'bvmt_total']
                if m in z_cols and z_cols[m] in cog.columns]
    cog['memory_composite'] = cog[mem_cols].mean(axis=1) if mem_cols else np.nan
    cog['memory_n_measures'] = cog[mem_cols].notna().sum(axis=1) if mem_cols else 0

    # Processing Speed
    speed_cols = [z_cols[m] for m in ['salthouse_letter', 'salthouse_pattern', 'trails_a_time']
                  if m in z_cols and z_cols[m] in cog.columns]
    cog['speed_composite'] = cog[speed_cols].mean(axis=1) if speed_cols else np.nan
    cog['speed_n_measures'] = cog[speed_cols].notna().sum(axis=1) if speed_cols else 0

    # Global
    domain_cols = ['attention_composite', 'memory_composite', 'speed_composite']
    cog['global_composite'] = cog[domain_cols].mean(axis=1)
    cog['global_n_domains'] = cog[domain_cols].notna().sum(axis=1)

    # Executive Function
    if 'trails_b_time' in cog.columns and 'trails_a_time' in cog.columns:
        cog['trails_ba_raw'] = cog['trails_b_time'] - cog['trails_a_time']
        vals = cog['trails_ba_raw'].dropna()
        if len(vals) > 1:
            cog['trails_ba_z'] = (cog['trails_ba_raw'] - vals.mean()) / vals.std()
            cog['trails_ba_z'] = -cog['trails_ba_z']
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
    theta_subject: Optional[pd.DataFrame] = None,
    accuracy: Optional[pd.DataFrame] = None,
    h2_subjects: Optional[List[str]] = None,
    include_exploratory: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Assemble master subject-level DataFrame.

    Merges demographics, cognitive composites, SPSRQ, tACS surveys,
    RF1 mood/substance/health measures, behavioral parameters, and theta.

    Parameters
    ----------
    include_exploratory : bool
        If True, also load RF1 exploratory measures (TEI, AQ, PANAS, etc.)
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

    # Education from RF1 Raw (primary source)
    if 'education_years' in rf1_neuro.columns:
        subj_df = subj_df.merge(
            rf1_neuro[['subject_id', 'education_years']],
            on='subject_id', how='left'
        )

    # Backfill education from TabCAT where RF1 Raw is missing
    if 'tabcat_education_years' in tabcat.columns:
        tc_edu = tabcat[['subject_id', 'tabcat_education_years']].copy()
        tc_edu['tabcat_education_years'] = pd.to_numeric(tc_edu['tabcat_education_years'], errors='coerce')
        subj_df = subj_df.merge(tc_edu, on='subject_id', how='left')
        missing_mask = subj_df['education_years'].isna() & subj_df['tabcat_education_years'].notna()
        n_recovered = missing_mask.sum()
        if n_recovered > 0:
            subj_df.loc[missing_mask, 'education_years'] = subj_df.loc[missing_mask, 'tabcat_education_years']
            print(f'  Education: recovered {n_recovered} values from TabCAT '
                  f'({subj_df["education_years"].notna().sum()}/{ len(subj_df)} total)')
        subj_df = subj_df.drop(columns=['tabcat_education_years'])

    # --- Cognitive composites ---
    cog = compute_cognitive_composites(rf1_cog, rf1_neuro, tabcat, study_subjects)
    cog_cols = ['subject_id',
                'attention_composite', 'attention_n_measures',
                'memory_composite', 'memory_n_measures',
                'speed_composite', 'speed_n_measures',
                'global_composite', 'global_n_domains',
                'ef_composite', 'ef_n_measures']
    subj_df = subj_df.merge(cog[cog_cols], on='subject_id', how='left')

    # --- Raw cognitive scores ---
    cog_raw_cols = ['subject_id', 'hvlt_total', 'salthouse_letter', 'salthouse_pattern',
                    'digit_span_total', 'bvmt_total', 'trails_a_time', 'trails_b_time',
                    'flanker_score', 'running_dots_score']
    cog_raw_cols = [c for c in cog_raw_cols if c in cog.columns]
    subj_df = subj_df.merge(cog[cog_raw_cols], on='subject_id', how='left')

    # --- Additional RF1 cognitive (SCD-Q, HAR, KBIT) ---
    rf1_extra_cols = ['subject_id']
    for col in ['scd_q', 'har_score', 'kbit_iq']:
        if col in rf1_cog.columns:
            rf1_extra_cols.append(col)
    if len(rf1_extra_cols) > 1:
        subj_df = subj_df.merge(rf1_cog[rf1_extra_cols], on='subject_id', how='left')

    # --- RF1 Extended: mood, anxiety, substance use, social ---
    try:
        rf1_ext = load_rf1_extended(study_subjects=study_subjects)
        subj_df = subj_df.merge(rf1_ext, on='subject_id', how='left')
        if verbose:
            ext_vars = [c for c in rf1_ext.columns if c != 'subject_id']
            n_with = rf1_ext[ext_vars[0]].notna().sum() if ext_vars else 0
            print(f'  RF1 extended measures: {len(ext_vars)} vars, ~{n_with} with data')
    except FileNotFoundError:
        if verbose:
            print('  RF1 FullScoring: NOT FOUND (extended measures skipped)')

    # --- RF1 Substance/Mood: AUDIT, DUDIT, CTQ, PROMIS, Loneliness ---
    try:
        sm_results = load_rf1_substance_mood(study_subjects=study_subjects, verbose=verbose)

        if 'audit_dudit' in sm_results:
            subj_df = subj_df.merge(sm_results['audit_dudit'], on='subject_id', how='left')

        if 'promis' in sm_results:
            subj_df = subj_df.merge(sm_results['promis'], on='subject_id', how='left')

        if 'loneliness' in sm_results:
            subj_df = subj_df.merge(sm_results['loneliness'], on='subject_id', how='left')

    except FileNotFoundError:
        if verbose:
            print('  RF1 Raw: NOT FOUND (substance/mood measures skipped)')

    # --- RF1 Exploratory (optional) ---
    if include_exploratory:
        try:
            rf1_exp = load_rf1_exploratory(study_subjects=study_subjects)
            subj_df = subj_df.merge(rf1_exp, on='subject_id', how='left')
            if verbose:
                exp_vars = [c for c in rf1_exp.columns if c != 'subject_id']
                print(f'  RF1 exploratory measures: {len(exp_vars)} vars')
        except FileNotFoundError:
            if verbose:
                print('  RF1 FullScoring: NOT FOUND (exploratory measures skipped)')

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

            if 'sham_p_stay_win' in subj_df.columns:
                subj_df['delta_p_stay_win'] = subj_df['active_p_stay_win'] - subj_df['sham_p_stay_win']
                subj_df['delta_p_shift_lose'] = subj_df['active_p_shift_lose'] - subj_df['sham_p_shift_lose']

    # --- R-W parameters ---
    if rw_mle is not None and len(rw_mle) > 0:
        rw_sham = rw_mle[rw_mle['condition'] == 'sham'].set_index('subject_id')
        if len(rw_sham) > 0:
            rw_sham = rw_sham[['alpha', 'beta']]
            rw_sham = rw_sham.rename(columns={'alpha': 'sham_alpha', 'beta': 'sham_beta'})
            subj_df = subj_df.merge(rw_sham, on='subject_id', how='left')

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

    # --- Accuracy and win rate ---
    # Takes the subject x condition frame from
    # accuracy_analysis.compute_condition_level_accuracy() and pivots it into
    # the sham_/active_/delta_ columns the notebook expects.
    if accuracy is not None and len(accuracy) > 0:
        for cond in ['sham', 'active']:
            cond_rows = accuracy[accuracy['condition'] == cond]

            # Restrict the active condition to H2-eligible subjects, matching
            # how the R-W and WSLS parameters are handled. Without this the
            # accuracy columns cover a wider sample than every other
            # active-condition measure, and they include subjects excluded
            # from H2 for good reason — 11773's "active" runs both delivered
            # sham, so its active_accuracy and delta_accuracy describe a
            # comparison that did not happen.
            if cond == 'active' and h2_subjects is not None:
                cond_rows = cond_rows[cond_rows['subject_id'].isin(h2_subjects)]

            if len(cond_rows) == 0:
                continue
            cond_rows = cond_rows.set_index('subject_id')[['accuracy', 'win_rate']]
            cond_rows = cond_rows.rename(columns={
                'accuracy': f'{cond}_accuracy',
                'win_rate': f'{cond}_win_rate',
            })
            subj_df = subj_df.merge(cond_rows, on='subject_id', how='left')

        if 'sham_accuracy' in subj_df.columns and 'active_accuracy' in subj_df.columns:
            subj_df['delta_accuracy'] = subj_df['active_accuracy'] - subj_df['sham_accuracy']
            subj_df['delta_win_rate'] = subj_df['active_win_rate'] - subj_df['sham_win_rate']

        if verbose:
            n_acc = subj_df['sham_accuracy'].notna().sum() if 'sham_accuracy' in subj_df.columns else 0
            print(f'  Accuracy (sham): {n_acc}')

    # --- DDM parameters ---
    if ddm_params is not None and len(ddm_params) > 0:
        ddm_cols = ['subject_id'] + [c for c in ddm_params.columns if c.startswith('ddm_')]
        subj_df = subj_df.merge(ddm_params[ddm_cols], on='subject_id', how='left')

    # --- Theta reactivity ---
    if theta_subject is not None and len(theta_subject) > 0:
        theta_cols = ['subject_id', 'theta_p95', 'theta_p75', 'theta_median']
        available_cols = [c for c in theta_cols if c in theta_subject.columns]
        if 'subject_id' in available_cols:
            subj_df = subj_df.merge(theta_subject[available_cols], on='subject_id', how='left')
            if verbose:
                n_theta = subj_df['theta_p95'].notna().sum() if 'theta_p95' in subj_df.columns else 0
                print(f'  Theta reactivity: {n_theta}')

    if verbose:
        print(f'\nSubject DataFrame assembled: {len(subj_df)} subjects, {len(subj_df.columns)} variables')
        n_with_cog = subj_df['global_composite'].notna().sum()
        n_with_wsls = subj_df.get('sham_p_stay_win', pd.Series()).notna().sum()
        n_with_rw = subj_df.get('sham_alpha', pd.Series()).notna().sum()
        print(f'  Cognitive composites: {n_with_cog}')
        print(f'  WSLS (sham): {n_with_wsls}')
        print(f'  R-W (sham): {n_with_rw}')

    return subj_df


# =============================================================================
# Master CSV Export
# =============================================================================

def export_master_csv(
    subj_df: pd.DataFrame,
    output_dir: Optional[str] = None,
    filename: str = MASTER_CSV_FILENAME,
    verbose: bool = True
) -> str:
    """
    Export master subject DataFrame to CSV.

    All missing values coded as NaN (empty cells in CSV).

    Parameters
    ----------
    subj_df : DataFrame
        Master subject DataFrame from build_subject_df()
    output_dir : str, optional
        Directory for output. Defaults to DATA_DIR parent.
    filename : str
        Output filename

    Returns
    -------
    str : full path to saved file
    """
    if output_dir is None:
        output_dir = DATA_DIR.parent

    output_path = Path(output_dir) / filename
    subj_df.to_csv(output_path, index=False, na_rep='NaN')

    if verbose:
        print(f'\nMaster CSV exported: {output_path}')
        print(f'  {len(subj_df)} subjects × {len(subj_df.columns)} variables')

        # Quick coverage summary
        n_complete = subj_df.notna().all(axis=1).sum()
        n_vars_full = (subj_df.notna().sum() == len(subj_df)).sum()
        print(f'  Subjects with complete data: {n_complete}/{len(subj_df)}')
        print(f'  Variables with no missing: {n_vars_full}/{len(subj_df.columns)}')

    return str(output_path)


# =============================================================================
# Missing Data Audit
# =============================================================================

def run_missing_data_audit(
    subj_df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Comprehensive missing data audit across all measures.

    Returns DataFrame with variable, domain, n_available, n_missing, pct_missing.
    """
    AUDIT_SPEC = {
        'Demographics': [
            ('Age', 'age'),
            ('Gender', 'gender'),
            ('Race', 'race'),
            ('Ethnicity', 'ethnicity'),
            ('Education', 'education_years'),
        ],
        'Cognitive Measures': [
            ('HVLT Total', 'hvlt_total'),
            ('Salthouse Letter', 'salthouse_letter'),
            ('Salthouse Pattern', 'salthouse_pattern'),
            ('Digit Span Total', 'digit_span_total'),
            ('BVMT Total', 'bvmt_total'),
            ('Trails A', 'trails_a_time'),
            ('Trails B', 'trails_b_time'),
            ('Flanker', 'flanker_score'),
            ('Running Dots', 'running_dots_score'),
            ('KBIT IQ', 'kbit_iq'),
            ('HAR Score', 'har_score'),
            ('SCD-Q', 'scd_q'),
        ],
        'Cognitive Composites': [
            ('Attention', 'attention_composite'),
            ('Memory', 'memory_composite'),
            ('Speed', 'speed_composite'),
            ('EF', 'ef_composite'),
            ('Global', 'global_composite'),
        ],
        'tACS Bandit Surveys': [
            ('SPSRQ-SP', 'spsrq_sp'),
            ('SPSRQ-SR', 'spsrq_sr'),
            ('B-PSQI', 'bpsqi_global'),
            ('BBS', 'bbs_avg'),
            ('CRT', 'crt_total'),
            ('FFMQ', 'ffmq_total'),
        ],
        'Mood / Internalizing': [
            ('SUSD Depression', 'susd_depression'),
            ('SUSD Mania', 'susd_mania'),
            ('PROMIS Depression', 'promis_depression'),
            ('PROMIS Anxiety', 'promis_anxiety'),
        ],
        'Anxiety (SCAARED)': [
            ('SCAARED Total', 'scaared_total'),
            ('SCAARED Somatic/Panic', 'scaared_somatic'),
            ('SCAARED GAD', 'scaared_gad'),
            ('SCAARED Separation', 'scaared_separation'),
            ('SCAARED Social', 'scaared_social'),
        ],
        'Substance Use / Gambling': [
            ('AUDIT', 'audit_total'),
            ('DUDIT', 'dudit_total'),
            ('SOGS', 'sogs_total'),
            ('NORC Lifetime', 'norc_lifetime'),
            ('NORC Past Year', 'norc_past_year'),
        ],
        'Social / Loneliness': [
            ('UCLA Loneliness', 'loneliness_total'),
            ('MSPSS Total', 'mspss_total'),
            ('MSPSS Friends', 'mspss_friends'),
            ('MSPSS Family', 'mspss_family'),
            ('BSMAS', 'bsmas_total'),
            ('NBS', 'nbs_total'),
        ],
        'Trauma': [
            ('CTQ-SF Total', 'ctq_total'),
        ],
        'Health (PROMIS)': [
            ('PROMIS Fatigue', 'promis_fatigue'),
            ('PROMIS Sleep', 'promis_sleep'),
            ('PROMIS Social', 'promis_social'),
            ('PROMIS Pain', 'promis_pain'),
            ('PROMIS Physical', 'promis_physical'),
        ],
        'Emotion Regulation': [
            ('EROS Ext Improving', 'eros_ext_improving'),
            ('EROS Ext Worsening', 'eros_ext_worsening'),
            ('EROS Int Improving', 'eros_int_improving'),
            ('EROS Int Worsening', 'eros_int_worsening'),
        ],
        'Aggression (BPAQ)': [
            ('BPAQ Total', 'bpaq_total'),
            ('BPAQ Physical', 'bpaq_physical'),
            ('BPAQ Anger', 'bpaq_anger'),
            ('BPAQ Verbal', 'bpaq_verbal'),
            ('BPAQ Hostility', 'bpaq_hostility'),
        ],
        'Behavioral Parameters (Sham)': [
            ('p(stay|win)', 'sham_p_stay_win'),
            ('p(shift|lose)', 'sham_p_shift_lose'),
            ('RW α', 'sham_alpha'),
            ('RW β', 'sham_beta'),
        ],
        'Behavioral Parameters (Active)': [
            ('p(stay|win)', 'active_p_stay_win'),
            ('p(shift|lose)', 'active_p_shift_lose'),
            ('RW α', 'active_alpha'),
            ('RW β', 'active_beta'),
        ],
        'EEG': [
            ('Theta p95', 'theta_p95'),
        ],
    }

    N = len(subj_df)
    rows = []

    for domain, measures in AUDIT_SPEC.items():
        for label, varname in measures:
            if varname in subj_df.columns:
                n_avail = subj_df[varname].notna().sum()
                n_miss = N - n_avail
                pct_miss = 100 * n_miss / N if N > 0 else 0
            else:
                n_avail = 0
                n_miss = N
                pct_miss = 100.0

            rows.append({
                'domain': domain,
                'measure': label,
                'variable': varname,
                'n_available': n_avail,
                'n_missing': n_miss,
                'pct_missing': round(pct_miss, 1),
            })

    audit_df = pd.DataFrame(rows)

    if verbose:
        print('\n' + '='*70)
        print('MISSING DATA AUDIT')
        print('='*70)
        print(f'Total subjects: {N}\n')

        for domain in AUDIT_SPEC:
            domain_rows = audit_df[audit_df['domain'] == domain]
            any_missing = domain_rows['n_missing'].sum() > 0
            flag = '⚠' if any_missing else '✓'

            print(f'{flag} {domain}:')
            for _, row in domain_rows.iterrows():
                status = '✓' if row['n_missing'] == 0 else f"  MISSING {row['n_missing']}"
                print(f"    {row['measure']:<25s}: {row['n_available']:>3d}/{N}  {status}")

            # Per-subject summary for this domain
            domain_vars = [r['variable'] for _, r in domain_rows.iterrows()
                           if r['variable'] in subj_df.columns]
            if domain_vars:
                subs_missing = subj_df[subj_df[domain_vars].isna().any(axis=1)]['subject_id'].tolist()
                if subs_missing:
                    print(f'    → Subjects with any missing: {subs_missing}')
            print()

    return audit_df


# =============================================================================
# Main Pipeline
# =============================================================================

def run_cognitive_merge(
    wsls_h1: Optional[pd.DataFrame] = None,
    wsls_h2: Optional[pd.DataFrame] = None,
    rw_mle: Optional[pd.DataFrame] = None,
    ddm_params: Optional[pd.DataFrame] = None,
    theta_subject: Optional[pd.DataFrame] = None,
    accuracy: Optional[pd.DataFrame] = None,
    h2_subjects: Optional[List[str]] = None,
    include_exploratory: bool = False,
    export_csv: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run complete cognitive merge pipeline.

    Parameters
    ----------
    include_exploratory : bool
        If True, also load exploratory measures (TEI, AQ, PANAS, etc.)
    export_csv : bool
        If True, write master_subject_data.csv to data directory

    Returns
    -------
    dict with keys:
        'rf1_cog', 'rf1_neuro', 'tabcat', 'tacs_redcap',
        'cog_composites', 'subj_df', 'missing_audit',
        'master_csv_path' (if export_csv)
    """
    study_subjects = list(SUBJECT_INFO.keys())
    results = {}

    if verbose:
        print('='*70)
        print('Loading External Data Sources')
        print('='*70)

    # Load primary sources
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
        theta_subject=theta_subject,
        accuracy=accuracy,
        h2_subjects=h2_subjects,
        include_exploratory=include_exploratory,
        verbose=verbose
    )
    results['subj_df'] = subj_df

    # Missing data audit
    if verbose:
        audit_df = run_missing_data_audit(subj_df, verbose=verbose)
        results['missing_audit'] = audit_df

    # Export master CSV
    if export_csv:
        try:
            csv_path = export_master_csv(subj_df, verbose=verbose)
            results['master_csv_path'] = csv_path
        except Exception as e:
            if verbose:
                print(f'WARNING: Could not export master CSV: {e}')

    return results


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing cognitive_merge module...")
    print(f"Study subjects: {len(SUBJECT_INFO)}")
    print(f"Data sources:")
    print(f"  RF1 FullScoring: {REDCAP_RF1_PATH}")
    print(f"  RF1 Raw: {REDCAP_RF1_RAW_PATH}")
    print(f"  TabCAT: {TABCAT_PATH}")
    print(f"  tACS REDCap: {REDCAP_TACS_PATH}")
    print()
    print("Functions:")
    print("  load_rf1_fullscoring, load_rf1_extended, load_rf1_exploratory")
    print("  load_rf1_raw, load_rf1_substance_mood")
    print("  load_tabcat, load_tacs_redcap")
    print("  score_spsrq, score_bpsqi, score_bbs, score_crt, score_ffmq")
    print("  compute_cognitive_composites, build_subject_df")
    print("  export_master_csv, run_missing_data_audit")
    print("  run_cognitive_merge")
