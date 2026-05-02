# tACS Bandit with Fast Feedback-Theta Localizer

This repository runs a two-armed reversal-learning bandit task for StarStim 8 / NIC-2 studies and now supports a fast no-stimulation localizer for participant-specific feedback-locked theta.

## Overview

The main scientific target is **task-evoked, feedback-locked theta** during reward learning, not a generic resting “intrinsic theta” estimate. The intended workflow is:

1. Run `LOCALIZER_FAST_THETA` for 5-10 minutes with no stimulation.
2. Record StarStim EEG over LSL during the localizer or import an offline EEG export later.
3. Preprocess the EEG automatically.
4. Estimate a **participant-specific feedback theta** peak.
5. Save that estimate in a subject-level JSON file.
6. Run `ITHETA_TACS` using the selected frequency.
7. Fall back to fixed 6.0 Hz when the individualized estimate is unreliable.

## Experiment Modes

- `LOCALIZER_FAST_THETA`: Fast no-stimulation bandit localizer with event-rich feedback logging.
- `ITHETA_TACS`: Theta-tACS using the most recent reliable participant-specific feedback theta estimate.
- `FIXED_THETA_TACS`: Fixed 6.0 Hz theta-tACS for controls and fallback.
- `SHAM`: Sham run with the same task timing and logging structure.
- `THETA_NIC`, `DC_NIC`, `BEHAVIORAL`: Legacy aliases preserved for backward compatibility.

## Fast Localizer

`LOCALIZER_FAST_THETA` is a faster version of the same two-choice reversal-learning bandit task:

- Two choices, probabilistic reward, reversals, and win/loss/miss feedback.
- Default duration: 6 minutes.
- Configurable duration: 5, 6, 8, or 10 minutes.
- Default target: about 120 trials and at least 100 feedback events before artifact rejection.
- Localizer timing defaults are shorter than the legacy stimulation blocks so the operator can estimate theta before the next run.

The localizer sends the standard feedback markers (`31`, `32`, `33`), stores detailed feedback timing in the behavioral CSV, and can subscribe to a live StarStim EEG LSL stream during the run.

## Standard Workflow

### 1. Run the localizer

```bash
cd code
python bandit_main.py --mode LOCALIZER_FAST_THETA --subject 001 --session 001
```

### 2. Estimate participant-specific feedback theta

```bash
python run_theta_estimation.py --subject 001 --session 001 --auto-find
```

This writes:

- A subject/session theta estimate JSON
- A one-row CSV summary
- QC plots for channel quality, epoch retention, ROI time-frequency power, theta spectrum, split-half peaks, and bootstrap peak stability
- An HTML report when enabled in `config.json`

### 3. Run individualized theta stimulation

```bash
python bandit_main.py --mode ITHETA_TACS --subject 001 --session 001 --run 1
```

### 4. Run fixed theta control or fallback

```bash
python bandit_main.py --mode FIXED_THETA_TACS --subject 001 --session 001 --run 1 --frequency 6.0
```

## Frequency Decision Rules

- Reliable feedback-locked theta estimate: use the rounded participant-specific frequency.
- Unreliable feedback-locked theta estimate: use fixed 6.0 Hz when fallback is enabled.
- No theta file found: follow `stimulation_frequency_selection.stop_if_no_theta_file`.

The task never silently substitutes frequencies. Operator-facing summaries, logs, and trial CSVs store:

- Intended stimulation frequency
- Operator-confirmed protocol
- Operator-confirmed frequency
- Theta source (`reliable_itheta`, `fallback_fixed_6hz`, `fixed_6hz`, `sham`, or `none`)
- Theta estimate file path
- Reliability decision and reason

## EEG Recording and Estimation

### Live LSL pathway

During `LOCALIZER_FAST_THETA`, the code can:

- Resolve an incoming EEG LSL stream
- Record samples and timestamps continuously
- Save raw EEG as `.npz`
- Optionally save a `.csv` export and metadata JSON

### Offline import pathway

`run_theta_estimation.py` supports `.npz`, `.csv`, `.edf`, `.bdf`, `.xdf`, and `.set` inputs. `edf`, `bdf`, and `set` loading use MNE when available; `xdf` uses `pyxdf` when available.

### Preprocessing defaults

- High-pass: 0.5 Hz
- Low-pass: 40 Hz
- Notch: 60 Hz
- Resample: 250 Hz
- Reference: average of available good channels
- ROI: `Fz`, `FCz`, `Cz`, `F3`, `F4` with frontocentral fallback logic
- Epoch window: `-1.0` to `+1.5` s around feedback
- Baseline window: `-0.5` to `-0.1` s
- Theta window: `+0.2` to `+0.8` s

### Reliability gates

The estimator rejects or downgrades estimates when any of the following fail:

- Too few usable feedback epochs
- Too few usable ROI channels
- Weak peak prominence
- Peak at the edge of the theta search band
- Split-half disagreement
- Excessively wide bootstrap confidence interval
- Poor usable epoch fraction

When the estimate is unreliable, the JSON explicitly records the fallback to fixed 6.0 Hz.

## StarStim / NIC-2 Practical Notes

- Python does **not** pretend to program NIC-2 directly unless your lab has separately validated that path.
- For stimulation runs, the task determines the intended frequency and shows the protocol label and frequency to the operator.
- The operator still confirms what was actually loaded in NIC-2.
- The task can wait for NIC-2 marker `203` before starting the bandit run.

## EEG Limitations

- StarStim 8 EEG can be useful for a quick feedback-theta localizer, but the estimate is QC-gated.
- The main theta estimate comes from the **no-stimulation localizer**.
- Simultaneous EEG during tACS should be treated as exploratory.
- Baseline correction is used for feedback-locked spectral estimates, but it does not by itself remove the aperiodic component.

## Data Outputs

Behavioral localizer and stimulation CSVs now include legacy fields plus additional columns such as:

- `subject_id`
- `session_id`
- `run`
- `mode`
- `phase`
- `stim_condition`
- `protocol_label_to_show`
- `operator_confirmed_protocol`
- `operator_confirmed_frequency_hz`
- `intended_stimulation_frequency_hz`
- `actual_or_confirmed_stimulation_frequency_hz`
- `theta_source`
- `theta_estimate_file`
- `theta_reliable`
- `theta_reliability_reason`
- `feedback_marker`
- `feedback_onset_task_time`
- `feedback_onset_lsl_time`
- `lsl_marker_send_time`
- `run_start_task_time`
- `run_start_lsl_time`
- `run_end_task_time`
- `run_end_lsl_time`

## Marker Dictionary

Existing marker codes are preserved:

- `10`: Trial start
- `20`: Choice
- `31`: Feedback win
- `32`: Feedback loss
- `33`: Feedback miss
- `100`: Run start
- `200`: Run end
- `203`: NIC-2 stimulation start marker used to begin stimulation runs

## LSL Debugging

Use the debug utility to inspect marker and EEG streams:

```bash
python lsl_debug_tool.py
```

It reports available streams, tests a marker inlet, and samples from the first EEG stream it finds.

## Tests

Synthetic verification lives in `tests/test_theta_workflow.py` and covers:

- Reliable 6.5 Hz task-evoked theta
- Noisy/no-peak fallback
- Edge-peak rejection
- Split-half disagreement fallback
- Too-few-epochs fallback
- Localizer smoke test when `pygame` is available
- ITHETA JSON selection and fallback behavior

Run the tests with:

```bash
python -m unittest tests/test_theta_workflow.py
```

## Repo Layout

```text
tacs_bandit/
├── code/
│   ├── bandit_main.py
│   ├── eeg_lsl_recorder.py
│   ├── local_starstim_module.py
│   ├── lsl_debug_tool.py
│   ├── run_theta_estimation.py
│   ├── select_theta_frequency.py
│   ├── theta_estimator.py
│   └── config.json
├── data/
├── tests/
└── readme.md
```
