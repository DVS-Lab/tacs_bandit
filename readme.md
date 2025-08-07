# Two-Armed Bandit Task

A Python implementation of a reinforcement learning task with reversal learning, designed for cognitive neuroscience research with EEG/tACS stimulation support.

## Overview

This task implements a two-armed bandit paradigm where participants choose between two options ("slot machines") to maximize rewards. The reward probabilities reverse periodically, requiring participants to adapt their choices. The implementation supports integration with Neuroelectrics Starstim devices for brain stimulation.

## Features

- **Core Task**
  - Two-choice bandit task with configurable reward probabilities
  - Reversal learning with pseudorandom contingency durations
  - Configurable timing parameters for all task phases
  - Support for different response devices (keyboard, button box)

- **Stimulation Support**
  - Integration with Neuroelectrics Starstim devices
  - tACS (transcranial alternating current stimulation) protocols
  - Sham stimulation controls
  - Multiple electrode montages (lPFC, rTPJ)
  - Test mode for development without hardware

- **Data & Analysis**
  - Comprehensive trial-by-trial data logging
  - Built-in reinforcement learning models (Rescorla-Wagner, WSLS, Choice Kernel)
  - Automated analysis scripts with visualizations
  - CSV output format for compatibility

## Installation

### Prerequisites

- Python 3.8 or higher
- PsychoPy (for stimulus presentation)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/DVS-Lab/bandit-task.git
cd bandit-task
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the task with default settings:
```bash
python bandit_main.py
```

### Configuration

Edit `config.json` to customize task parameters:

```json
{
  "task": {
    "n_blocks": 4,
    "trials_per_block": 50,
    "win_fraction": 0.75,
    "min_trials_same_contingency": 25
  },
  "timing": {
    "max_response_time": 1.5,
    "outcome_duration": 0.5,
    "iti_mean": 0.25
  }
}
```

### Stimulation Setup

To use with Starstim device:

1. Set stimulation parameters in `config.json`:
```json
{
  "stimulation": {
    "enabled": true,
    "type": "tACS",
    "frequency": 6,
    "protocol": "theta"
  }
}
```

2. Run with stimulation:
```python
from starstim import StarstimController
controller = StarstimController(config, test_mode=False)
controller.connect()
```

### Test Mode

For development and testing without hardware:
```python
task = TwoArmedBanditTask(config)
task.subject_info['mode'] = 'test'
task.run()
```

## Data Output

Data is saved in CSV format with the following structure:
- `trial_num`: Trial number
- `block_num`: Block number
- `choice`: Participant's choice (1 or 2)
- `rt`: Reaction time in milliseconds
- `correct`: Whether the better option was chosen
- `reward`: Whether reward was received
- `current_good`: Which option had higher reward probability
- `trial_in_contingency`: Trials since last reversal

### File Naming Convention
```
sub-{ID}_ses-{SESSION}_run-{RUN}_task-bandit_{TIMESTAMP}.csv
```

## Analysis

### Quick Analysis

Analyze a single data file:
```bash
python analysis.py data/sub-001_ses-1_run-1_task-bandit_2024-01-01_10-00-00.csv
```

### Detailed Analysis

```python
from analysis import BanditAnalyzer

# Load data
analyzer = BanditAnalyzer('path/to/data.csv')

# Get summary statistics
stats = analyzer.calculate_summary_stats()

# Plot learning curves
analyzer.plot_learning_curve()

# Analyze reversal behavior
analyzer.plot_reversal_analysis()

# Export full analysis
analyzer.export_summary('output/directory')
```

## Reinforcement Learning Models

The package includes several RL models for fitting and simulation:

```python
from rl_model import RescorlaWagner, WinStayLoseShift

# Fit Rescorla-Wagner model
model = RescorlaWagner(learning_rate=0.3, inverse_temperature=3.0)

# Simulate choices
for trial in trials:
    choice_prob = model.get_choice_probability()
    choice = model.simulate_choice()
    model.update(choice, reward)
```

## Project Structure

```
bandit-task/
├── bandit_main.py          # Main task script
├── config.json             # Configuration file
├── rl_model.py            # Reinforcement learning models
├── starstim.py            # Starstim device interface
├── analysis.py            # Analysis utilities
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # Data output directory
├── stimuli/              # Stimulus images
│   └── images/
│       ├── flowers/      # Slot machine images
│       └── feedback/     # Win/loss images
└── logs/                 # Log files
```

## Troubleshooting

### Common Issues

1. **PsychoPy window doesn't open**: Ensure you have proper graphics drivers installed
2. **Starstim connection fails**: Check that NIC software is running and device is connected
3. **Data not saving**: Ensure write permissions for the data directory

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Converting from MATLAB

This implementation is converted from a MATLAB/Neurostim version. Key differences:

- **Timing**: PsychoPy handles timing differently than Neurostim
- **File I/O**: Uses CSV instead of MATLAB .mat files
- **Stimulation**: Modular design allows easy swapping of stimulation protocols
- **Analysis**: Python-native analysis tools with pandas/matplotlib

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For questions or issues:
- Open an issue on GitHub
- Contact: [james.wyngaarden@temple.edu]

## Acknowledgments

- Original MATLAB implementation by [Original Authors]
- PsychoPy development team
- Neuroelectrics for device support documentation
