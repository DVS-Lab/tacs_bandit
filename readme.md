# LSL-Triggered Two-Armed Bandit Task

A Python implementation of a reinforcement learning task with reversal learning, designed for cognitive neuroscience research with real-time EEG/tACS stimulation integration via Lab Streaming Layer (LSL).

## Overview

This task implements a two-armed bandit paradigm where participants choose between two flower images to maximize rewards. The reward probabilities reverse periodically, requiring participants to adapt their choices. The implementation features automatic synchronization with Neuroelectrics Starstim devices via LSL markers for precise timing in brain stimulation experiments.

## Key Features

### **Core Task**
- Two-choice bandit task with configurable reward probabilities (default 75/25%)
- Reversal learning with pseudorandom contingency durations (25±4 trials)
- Flower images as stimuli with unique combinations per run
- Configurable timing optimized for different experimental protocols (behavioral, fMRI, EEG)
- Multiple response modalities (keyboard: 1/2 or A/L keys)

### **LSL-Triggered Stimulation Integration**
- **Real-time synchronization** with Neuroelectrics Starstim via Lab Streaming Layer
- **Automatic task triggering** when stimulation protocols start (marker 203)
- **Double-blind compatible** - researcher loads protocols without knowing condition
- **Perfect timing alignment** - task and stimulation run simultaneously
- Support for both tACS (active) and sham stimulation protocols
- Comprehensive event marker logging for EEG analysis

### **Experimental Protocols**
- **8-run counterbalanced design** with baseline and stimulation blocks
- **DLPFC stimulation protocols** (active vs sham, counterbalanced across subjects)
- **Multiple timing modes**: THETA_NIC, DC_NIC, BEHAVIORAL
- **Test mode** for development and training without hardware

### **Data & Analysis**
- Trial-by-trial data logging with stimulation synchronization info
- Contingency reversal tracking and performance metrics
- CSV output compatible with standard analysis pipelines
- Event marker logs for EEG/stimulation analysis

## Installation

### Prerequisites

- **Python 3.8+**
- **macOS/Linux** (tested on macOS)
- **Neuroelectrics NIC-2 software** (for stimulation integration)
- **Lab Streaming Layer (LSL)** for real-time data streaming

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/tacs_bandit.git
cd tacs_bandit
```

2. **Install Python dependencies:**
```bash
pip install pygame pandas numpy pylsl
```

3. **Install LSL library (macOS):**
```bash
# Using Homebrew
brew install labstreaminglayer/tap/lsl

# Set environment variable
export DYLD_LIBRARY_PATH=/opt/homebrew/lib
```

4. **Setup stimuli:**
   - Place flower images in `stimuli/images/` as `001-flowers.png` through `050-flowers.png`
   - Add feedback images: `001-win.png` through `009-win.png`, `001-loss.png` through `009-loss.png`
   - Include `question-mark.png` for missed trials

## Usage

### Basic Usage (No Stimulation)

```bash
cd code/
python bandit_main.py
```

**Test parameters:**
- Subject ID: Any number
- Run: 1, 4, 5, or 8 (baseline runs)

### LSL-Triggered Stimulation Mode

1. **Configure stimulation in `config.json`:**
```json
{
  "stimulation": {
    "enabled": true,
    "test_mode": false
  }
}
```

2. **Start NIC-2 software and enable LSL:**
   - Protocol Settings → LSL Server → Enable
   - Load appropriate protocol (DLPFC_Active or DLPFC_Sham)

3. **Run the task:**
```bash
python bandit_main.py
```

4. **Follow the workflow:**
   - Task shows which protocol to load (based on counterbalancing)
   - Task waits: "Waiting for stimulation to start..."
   - Start protocol in NIC-2 → Task begins automatically!

### Counterbalancing

**Even subject IDs (2, 4, 6, 8...):**
- Runs 2-3: DLPFC_Active
- Runs 6-7: DLPFC_Sham

**Odd subject IDs (1, 3, 5, 7...):**
- Runs 2-3: DLPFC_Sham  
- Runs 6-7: DLPFC_Active

### Configuration

Edit `code/config.json` to customize parameters:

```json
{
  "experiment": {
    "run_duration_minutes": 6,
    "mode": "THETA_NIC"
  },
  "task": {
    "min_trials_same_contingency": 25,
    "contingency_jitter": 4,
    "win_fraction": 0.75
  },
  "timing": {
    "fixation_duration": 0.5,
    "max_response_time": 2.0,
    "wait_duration_min": 2.0,
    "wait_duration_max": 2.0,
    "outcome_duration": 1.0,
    "iti_duration": 0.25
  },
  "stimulation": {
    "enabled": false,
    "test_mode": true,
    "protocols": {
      "active": "DLPFC_Active",
      "sham": "DLPFC_Sham"
    }
  }
}
```

## Experimental Workflow

### Standard 8-Run Protocol

1. **Run 1**: Baseline (6 min)
2. **Runs 2-3**: First stimulation condition (6 min each)
3. **Run 4**: Post-stimulation baseline (6 min)
4. **Run 5**: Baseline (6 min)
5. **Runs 6-7**: Second stimulation condition (6 min each)
6. **Run 8**: Final baseline (6 min)

### Per-Run Procedure

1. **Setup**: Load correct protocol in NIC-2 (as instructed by task)
2. **Start task**: `python bandit_main.py`
3. **Enter subject info** and run number
4. **Wait for trigger**: Task displays "Waiting for stimulation..."
5. **Start stimulation**: Click start in NIC-2 → Task begins immediately
6. **Monitor**: Task runs for exactly 6 minutes
7. **Data saved**: Automatic CSV export with performance summary

## Data Output

### File Structure
```
data/
└── sub-{ID}/
    ├── sub-{ID}_ses-{SESSION}_run-{RUN}_task-bandit_{TIMESTAMP}.csv
    └── ...
```

### Data Columns
- `trial_num`: Trial number within run
- `run`, `run_type`, `stim_condition`: Experimental condition info
- `choice`, `rt`: Participant response and reaction time
- `correct`, `reward`: Trial outcome
- `current_good`: Which flower had higher reward probability
- `trial_in_contingency`: Trials since last reversal
- `flower1`, `flower2`: Specific flower images used
- `slot1_position`, `slot2_position`: Left/right randomization
- `timestamp`: Time relative to run start

### LSL Event Markers
Event markers are logged for EEG analysis:
- `100`: Run start
- `10`: Trial start  
- `20`: Choice made
- `31/32/33`: Feedback (win/loss/miss)
- `200`: Run end

## Technical Details

### LSL Integration
- **Marker stream**: `LSLOutletStreamName-Markers`
- **Trigger marker**: 203 (stimulation start)
- **Auto-detection**: Task automatically finds NIC-2 marker streams
- **Fallback**: Graceful degradation if LSL unavailable

### Timing Precision
- **Stimulation sync**: Task starts exactly when NIC-2 sends marker 203
- **Duration matching**: Both task and stimulation run for 6 minutes
- **Event logging**: Sub-millisecond precision for all task events

## Troubleshooting

### LSL Connection Issues

**Check LSL streams:**
```bash
python lsl_debug_tool.py
```

**Common fixes:**
- Enable LSL Server in NIC-2 Protocol Settings
- Ensure Starstim device is connected and powered
- Check that marker sending is enabled in NIC-2

### Window Focus Issues
After starting stimulation, click the task window to ensure participant responses work.

### Image Loading Issues
- Verify image files are named correctly (`001-flowers.png`, etc.)
- Check file paths in `config.json`
- Ensure images are in `stimuli/images/` directory

## Project Structure

```
tacs_bandit/
├── code/
│   ├── bandit_main.py           # Main LSL-triggered task
│   ├── local_starstim_module.py # Stimulation interface
│   ├── lsl_debug_tool.py        # LSL debugging utility
│   └── config.json              # Configuration file
├── data/                        # Data output directory
├── stimuli/
│   └── images/                  # Flower and feedback images
├── logs/                        # LSL event logs
└── README.md                    # This file
```

## Testing

### Test Mode (No Hardware)
```json
{
  "stimulation": {
    "enabled": true,
    "test_mode": true
  }
}
```
Press ENTER when prompted to simulate stimulation trigger.

### Hardware Testing
1. **Basic connectivity**: `python lsl_debug_tool.py`
2. **Stimulation trigger**: Test with real protocol start/stop
3. **Full integration**: Complete run with oscilloscope verification

## Hardware Requirements

### Neuroelectrics Setup
- **Starstim device** (8-channel or tES)
- **NIC-2 software** with LSL enabled
- **Electrode montage** appropriate for DLPFC stimulation
- **6-minute protocols** configured in NIC-2

### Computer Requirements
- **macOS/Linux** (Windows possible with modifications)
- **Python 3.8+** with LSL support
- **Sufficient processing power** for real-time LSL streaming
- **Reliable network connection** between task and NIC-2 computers

## Contributing

This implementation follows research-grade standards for timing precision and experimental control. Contributions should maintain:
- **Millisecond-precise timing**
- **Double-blind compatibility**  
- **Robust error handling**
- **Comprehensive data logging**

## Acknowledgments

- Original MATLAB/Neurostim implementation
- Neuroelectrics for LSL integration documentation
- Lab Streaming Layer development team
- Research protocols adapted from cognitive neuroscience literature

## Support

For technical issues:
- Check LSL connectivity with debug tools
- Verify NIC-2 configuration settings
- Ensure proper timing protocol selection
- Contact: [your.email@institution.edu]

---

**Note**: This implementation prioritizes timing precision and experimental control suitable for publication-quality neuroscience research. The LSL integration ensures sub-millisecond synchronization between brain stimulation and behavioral measurements.