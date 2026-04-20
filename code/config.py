"""
config.py — Configuration and constants for tACS Bandit analyses

Central configuration file containing:
- Path definitions
- Subject registry (SUBJECT_INFO)
- Stimulation exclusion registry (STIM_EXCLUSIONS)
- Task parameters
- Plotting constants and color palette
- Exclusion criteria thresholds

All modules import from here to ensure consistency.
"""

from pathlib import Path

# =============================================================================
# Path Configuration
# =============================================================================
# Assumes this file lives in tacs_bandit/code/
# Adjust REPO_ROOT if running from a different location

REPO_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = REPO_ROOT / 'data' / 'bandit'
EEG_DIR = REPO_ROOT / 'data' / 'nic' / 'raw'

# REDCap exports
REDCAP_RF1_PATH = REPO_ROOT / 'data' / 'RF1SocialRewardProce-FullScoringFinishedS_DATA_LABELS_2026-02-28_1613.csv'
REDCAP_RF1_RAW_PATH = REPO_ROOT / 'data' / 'RF1SocialRewardProce_DATA_LABELS_2026-02-28_1436.csv'
REDCAP_TACS_PATH = REPO_ROOT / 'data' / 'TACSBandit-TACSBandittestReport_DATA_LABELS_2026-04-10_1000.csv'

# TabCAT cognitive battery
TABCAT_PATH = REPO_ROOT / 'data' / 'TabCATStudyData_rf1SocialRewardProcessingAcrossTheLifespan_2026-2-28.csv'


# =============================================================================
# Subject Registry
# =============================================================================
# Master tracking dictionary for all study participants.
# Keys are subject IDs (as strings).
# 
# Fields:
#   counterbalance : 'A' (active first) or 'B' (sham first)
#   earclip        : True if CMS/DRL earclip reference was used
#   notes          : Free-text for subject-specific issues
#
# Subjects not in this dict are excluded from analyses (pilots, tests, etc.)

SUBJECT_INFO = {
    '10998': {'counterbalance': 'B', 'earclip': False, 'notes': 'Missing Run 1; One intended stim run was sham (experimenter error)'},
    '11773': {'counterbalance': 'B', 'earclip': False, 'notes': 'Intended stim runs were actually sham (experimenter error)'},
    '10886': {'counterbalance': 'B', 'earclip': False, 'notes': ''},
    '10656': {'counterbalance': 'A', 'earclip': False, 'notes': ''},
    '10951': {'counterbalance': 'A', 'earclip': False, 'notes': 'Electrode artifact (loose connection suspected)'},
    '10418': {'counterbalance': 'B', 'earclip': False, 'notes': ''},
    '10636': {'counterbalance': 'B', 'earclip': True, 'notes': ''},
    '11318': {'counterbalance': 'B', 'earclip': True, 'notes': ''},
    '10369': {'counterbalance': 'B', 'earclip': True, 'notes': ''},
    '10606': {'counterbalance': 'A', 'earclip': True, 'notes': ''},
    '11628': {'counterbalance': 'A', 'earclip': True, 'notes': ''},
    '11286': {'counterbalance': 'A', 'earclip': True, 'notes': ''},
    '10866': {'counterbalance': 'B', 'earclip': True, 'notes': ''},
    '11329': {'counterbalance': 'B', 'earclip': True, 'notes': ''},
    '10608': {'counterbalance': 'A', 'earclip': True, 'notes': ''},
    '10559': {'counterbalance': 'A', 'earclip': True, 'notes': ''},
    '10589': {'counterbalance': 'A', 'earclip': True, 'notes': ''},
    '11563': {'counterbalance': 'A', 'earclip': True, 'notes': ''},
    '10649': {'counterbalance': 'A', 'earclip': True, 'notes': ''},
    '11030': {'counterbalance': 'A', 'earclip': True, 'notes': ''},
    '10809': {'counterbalance': 'B', 'earclip': True, 'notes': ''},
    '10961': {'counterbalance': 'B', 'earclip': True, 'notes': ''},
    '10541': {'counterbalance': 'B', 'earclip': True, 'notes': ''},
    '10641': {'counterbalance': 'B', 'earclip': True, 'notes': ''},
    '10862': {'counterbalance': 'B', 'earclip': True, 'notes': ''},
    '10661': {'counterbalance': 'B', 'earclip': True, 'notes': ''},
    '10638': {'counterbalance': 'B', 'earclip': True, 'notes': ''},
    '11631': {'counterbalance': 'A', 'earclip': True, 'notes': ''},
    '11016': {'counterbalance': 'B', 'earclip': True, 'notes': ''},
    '10716': {'counterbalance': 'B', 'earclip': True, 'notes': ''},
    '11526': {'counterbalance': 'B', 'earclip': True, 'notes': ''},
    '10810': {'counterbalance': 'A', 'earclip': True, 'notes': ''},
    '10804': {'counterbalance': 'B', 'earclip': True, 'notes': ''},
    '10898': {'counterbalance': 'A', 'earclip': True, 'notes': ''},
    '10590': {'counterbalance': 'B', 'earclip': True, 'notes': ''},
    '10950': {'counterbalance': 'A', 'earclip': True, 'notes': ''},
    '11036': {'counterbalance': 'B', 'earclip': True, 'notes': ''},
    '11031': {'counterbalance': 'A', 'earclip': True, 'notes': ''}
}

# Subjects without hardware earclip reference (first 4 participants)
# These require software average re-referencing across EEG-only channels
NO_EARCLIP_SUBJECTS = [sid for sid, info in SUBJECT_INFO.items() if not info['earclip']]


# =============================================================================
# Stimulation Administration Exclusion Registry
# =============================================================================
# Populated from EEG notebook verification (tacs_bandit_eeg.ipynb).
# Each entry documents a run where the delivered stimulation did not match
# the counterbalance-assigned condition.
#
# Format: (subject_id, run) → dict with:
#   'reason'           : free-text explanation
#   'assigned'         : what the counterbalance said this run should be
#   'actual_delivered' : what was actually delivered (confirmed via EEG)
#
# These runs are excluded from H2 (active vs. sham) comparisons.
# They are NOT excluded from H1 (sham-only baseline) unless also flagged
# behaviorally, because the behavioral data itself is still valid.

STIM_EXCLUSIONS = {
    ('10998', 6): {
        'reason': 'Experimenter error: sham protocol loaded instead of active',
        'assigned': 'active',
        'actual_delivered': 'sham',
    },
    ('11773', 6): {
        'reason': 'Experimenter error: sham protocol loaded instead of active',
        'assigned': 'active',
        'actual_delivered': 'sham',
    },
    ('11773', 7): {
        'reason': 'Experimenter error: sham protocol loaded instead of active',
        'assigned': 'active',
        'actual_delivered': 'sham',
    },
}


# =============================================================================
# Task Parameters
# =============================================================================

WIN_FRACTION = 0.75  # Reward probability for the "good" option (per pre-registration)

# Counterbalance condition mapping
# Maps run number → condition based on counterbalance order
CONDITION_MAP_A = {
    1: 'baseline', 2: 'active', 3: 'active', 4: 'post',
    5: 'baseline', 6: 'sham', 7: 'sham', 8: 'post'
}
CONDITION_MAP_B = {
    1: 'baseline', 2: 'sham', 3: 'sham', 4: 'post',
    5: 'baseline', 6: 'active', 7: 'active', 8: 'post'
}

# Pilot/test subject prefixes to exclude from loading
EXCLUDE_PREFIXES = ['avi-pilot', 'p1001', '00002', '1122', '1432']


# =============================================================================
# Pre-Registered Exclusion Criteria Thresholds
# =============================================================================
# Per Section 6 of the pre-registration (https://osf.io/vhyz2)

EXCLUSION_THRESHOLDS = {
    'missed_pct': 20.0,           # >20% missed trials → exclude run
    'side_bias_pct': 95.0,        # >95% same spatial location → exclude run
    'stim_bias_pct': 95.0,        # >95% same stimulus → exclude run
    'rapid_rt_ms': 200.0,         # median RT <200ms → exclude run
    'feedback_invariance': 10,    # 10+ consecutive losses without shift → exclude run
}

# Minimum clean runs per condition for H2 inclusion
MIN_CLEAN_RUNS_PER_CONDITION = 1

# Minimum trials for model fitting
MIN_TRIALS_FOR_FITTING = 10


# =============================================================================
# Plotting Constants
# =============================================================================

# Primary condition colors (colorblind-friendly palette)
COLOR_SHAM = '#1565C0'      # Blue
COLOR_ACTIVE = '#E64A19'    # Red/Orange (also '#E65100' used in some places)
COLOR_BASELINE = '#757575'  # Gray
COLOR_POST = '#9E9E9E'      # Light gray

# Condition color dictionary for easy lookup
CONDITION_COLORS = {
    'sham': COLOR_SHAM,
    'active': COLOR_ACTIVE,
    'baseline': COLOR_BASELINE,
    'post': COLOR_POST,
}

# Secondary accent colors
COLOR_GOLD = '#FFB300'      # Gold (age colormap endpoint)
COLOR_GREEN = '#2E7D32'     # Green (correct/match indicators)
COLOR_RED = '#C62828'       # Red (errors/mismatches)
COLOR_PURPLE = '#7B1FA2'    # Purple (tertiary accent)

# Counterbalance order colors
CB_COLORS = {
    'A': COLOR_GREEN,
    'B': COLOR_RED,
}

# Age colormap endpoints (for matplotlib LinearSegmentedColormap)
AGE_CMAP_COLORS = [COLOR_SHAM, COLOR_GOLD]  # Blue → Gold
AGE_MIN = 20
AGE_MAX = 80

# Plotly colorscale for age (ready to use in marker dict)
AGE_COLORSCALE = [[0, COLOR_SHAM], [1, COLOR_GOLD]]

# Correlation heatmap colorscale (diverging)
CORR_COLORSCALE = [[0, COLOR_SHAM], [0.5, '#FFFFFF'], [1, '#E65100']]

# EEG frequency band colors
BAND_COLORS = {
    'theta': COLOR_SHAM,    # θ: 4-8 Hz (our target)
    'alpha': '#E65100',     # α: 8-13 Hz
    'beta': COLOR_GREEN,    # β: 13-30 Hz
}

# Standard Plotly template
PLOTLY_TEMPLATE = 'plotly_white'
FONT_FAMILY = 'Arial'


# =============================================================================
# Utility Functions
# =============================================================================

def get_condition_map(counterbalance: str) -> dict:
    """Return the run → condition mapping for a given counterbalance order."""
    if counterbalance == 'A':
        return CONDITION_MAP_A.copy()
    elif counterbalance == 'B':
        return CONDITION_MAP_B.copy()
    else:
        return {}


def get_subject_ids() -> list:
    """Return list of all valid subject IDs."""
    return list(SUBJECT_INFO.keys())


def is_stim_excluded(subject_id: str, run: int) -> bool:
    """Check if a specific run is in the stimulation exclusion registry."""
    return (str(subject_id), int(run)) in STIM_EXCLUSIONS


# =============================================================================
# Validation (runs on import)
# =============================================================================

def _validate_config():
    """Basic validation of configuration values."""
    # Check that all STIM_EXCLUSIONS reference valid subjects
    for (sid, run), info in STIM_EXCLUSIONS.items():
        if sid not in SUBJECT_INFO:
            print(f"WARNING: STIM_EXCLUSIONS contains unknown subject {sid}")
    
    # Check counterbalance values
    valid_cb = {'A', 'B', '?'}
    for sid, info in SUBJECT_INFO.items():
        if info['counterbalance'] not in valid_cb:
            print(f"WARNING: Subject {sid} has invalid counterbalance '{info['counterbalance']}'")


# Run validation on import (can be disabled by commenting out)
_validate_config()
