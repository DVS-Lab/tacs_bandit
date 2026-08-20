"""
data_prep.py — Build RL model input from the verified analysis pipeline

The original data_loader reads raw CSVs straight off disk. That path has three
problems for the tACS contrast:

  1. It takes the condition label from the `stim_condition` column in the
     behavioral CSV. That column is written by a subject-ID-parity fallback in
     bandit_main.py, not by the actual counterbalance — it disagrees with the
     EEG-verified assignment for roughly half the sample. Fitting on it would
     invert active and sham for those subjects and bias the effect toward zero.
  2. It applies no exclusions, so runs failing the pre-registered criteria and
     runs in STIM_EXCLUSIONS go into the fit.
  3. It concatenates every run into one sequence per subject, so Q-values carry
     across run boundaries. Each run is a fresh bandit with new contingencies,
     so values must reset at every run start.

This module goes through data_loading + exclusions instead, so conditions come
from the verified counterbalance in SUBJECT_INFO and every exclusion applies.

It also changes the unit of analysis from the subject to the **run**. That
resets Q per run (fixing 3) and lets condition vary within subject, which the
design requires — every subject contributes both sham and active runs, and a
model with one condition per subject cannot express that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd

# Stimulation blocks: runs 2-3 are the first, runs 6-7 the second. Which is
# active and which is sham depends on the counterbalance.
FIRST_BLOCK_RUNS = (2, 3)
SECOND_BLOCK_RUNS = (6, 7)

CONDITION_CODES = {'sham': 0, 'active': 1}


@dataclass
class RunDataset:
    """Trial sequences with one row per run, padded for JAX.

    Attributes
    ----------
    choices, rewards, masks : (n_sequences, max_trials)
    seq_subject   : (n_sequences,) subject index into `subject_ids`
    seq_condition : (n_sequences,) 0 = sham, 1 = active
    seq_block     : (n_sequences,) 0 = first stim block, 1 = second
    """

    choices: jnp.ndarray
    rewards: jnp.ndarray
    masks: jnp.ndarray
    seq_subject: jnp.ndarray
    seq_condition: jnp.ndarray
    seq_block: jnp.ndarray
    subject_ids: List[str]
    run_index: pd.DataFrame = field(repr=False)

    @property
    def n_sequences(self) -> int:
        return int(self.choices.shape[0])

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)

    @property
    def max_trials(self) -> int:
        return int(self.choices.shape[1])

    def summary(self) -> str:
        n_trials = int(jnp.sum(self.masks))
        per_cond = {
            name: int(jnp.sum(self.seq_condition == code))
            for name, code in CONDITION_CODES.items()
        }
        both = self.run_index.groupby('subject_id')['condition'].nunique()
        return (
            f'RunDataset: {self.n_sequences} runs from {self.n_subjects} subjects, '
            f'{n_trials} trials (padded to {self.max_trials})\n'
            f'  runs by condition: {per_cond}\n'
            f'  subjects with both conditions: {int((both == 2).sum())}/{self.n_subjects}'
        )


def _centered(values: np.ndarray) -> np.ndarray:
    """Map a 0/1 indicator to -0.5/+0.5.

    Centering keeps the intercept interpretable as the grand mean rather than
    as the reference level, so mu_alpha is the average learning rate and
    delta_alpha is the full sham-to-active difference.
    """
    return values.astype(np.float32) - 0.5


def build_run_dataset(
    data: Optional[pd.DataFrame] = None,
    sample: str = 'all',
    conditions: Sequence[str] = ('sham', 'active'),
    min_trials: int = 10,
    verbose: bool = True,
) -> RunDataset:
    """
    Assemble run-level sequences for the hierarchical models.

    Parameters
    ----------
    data : pre-filtered trial-level DataFrame. When None, loads via
        load_all_subjects(sample) and applies apply_all_exclusions(), taking
        the H2 dataset (both conditions, all exclusions applied).
    sample : passed to load_all_subjects when `data` is None.
    conditions : which conditions to include, in code order.
    min_trials : runs with fewer valid trials than this are dropped.
    """
    # Imported here so the module can be used with a pre-built frame without
    # pulling in the whole analysis pipeline.
    from data_loading import load_all_subjects
    from exclusions import apply_all_exclusions

    if data is None:
        raw = load_all_subjects(sample=sample, verbose=False)
        results = apply_all_exclusions(raw, verbose=False)
        data = results['data_h2']
        if verbose:
            print(f'  exclusions applied: {data["subject_id"].nunique()} subjects, '
                  f'{len(data)} trials in the H2 dataset')

    df = data[data['condition'].isin(conditions)].copy()
    df = df.dropna(subset=['choice', 'reward'])

    sequences = []
    for (sub_id, run), group in df.groupby(['subject_id', 'run'], sort=True):
        if len(group) < min_trials:
            continue

        condition = str(group['condition'].iloc[0])
        if condition not in CONDITION_CODES:
            continue

        run = int(run)
        if run in FIRST_BLOCK_RUNS:
            block = 0
        elif run in SECOND_BLOCK_RUNS:
            block = 1
        else:
            # A stimulation condition on a run outside both blocks means the
            # condition map and the run numbering disagree; skip rather than
            # guess which is right.
            continue

        # Choices arrive as 1/2 and the models expect 0/1.
        choices = group['choice'].to_numpy()
        rewards = group['reward'].to_numpy()

        sequences.append({
            'subject_id': str(sub_id),
            'run': run,
            'condition': condition,
            'condition_code': CONDITION_CODES[condition],
            'block': block,
            'n_trials': len(group),
            'choices': (choices - 1).astype(np.int32),
            'rewards': rewards.astype(np.float32),
        })

    if not sequences:
        raise ValueError('No runs survived filtering; nothing to fit.')

    subject_ids = sorted({s['subject_id'] for s in sequences})
    subject_index = {sid: i for i, sid in enumerate(subject_ids)}

    n_seq = len(sequences)
    max_trials = max(s['n_trials'] for s in sequences)

    choices = np.zeros((n_seq, max_trials), dtype=np.int32)
    rewards = np.zeros((n_seq, max_trials), dtype=np.float32)
    masks = np.zeros((n_seq, max_trials), dtype=np.float32)
    seq_subject = np.zeros(n_seq, dtype=np.int32)
    seq_condition = np.zeros(n_seq, dtype=np.int32)
    seq_block = np.zeros(n_seq, dtype=np.int32)

    for i, seq in enumerate(sequences):
        n = seq['n_trials']
        choices[i, :n] = seq['choices']
        rewards[i, :n] = seq['rewards']
        masks[i, :n] = 1.0
        seq_subject[i] = subject_index[seq['subject_id']]
        seq_condition[i] = seq['condition_code']
        seq_block[i] = seq['block']

    run_index = pd.DataFrame([
        {k: s[k] for k in ('subject_id', 'run', 'condition', 'block', 'n_trials')}
        for s in sequences
    ])

    dataset = RunDataset(
        choices=jnp.array(choices),
        rewards=jnp.array(rewards),
        masks=jnp.array(masks),
        seq_subject=jnp.array(seq_subject),
        seq_condition=jnp.array(seq_condition),
        seq_block=jnp.array(seq_block),
        subject_ids=subject_ids,
        run_index=run_index,
    )

    if verbose:
        print(dataset.summary())

    return dataset


def condition_contrast(dataset: RunDataset) -> np.ndarray:
    """Centered active-vs-sham contrast, one value per sequence."""
    return _centered(np.asarray(dataset.seq_condition))


def block_contrast(dataset: RunDataset) -> np.ndarray:
    """
    Centered first-vs-second block contrast, one value per sequence.

    Within a subject, condition is perfectly confounded with block order — a
    counterbalance-A subject gets active first and sham second, always. The
    counterbalance breaks that confound across the group, but only if the model
    is given the block term; without it, any practice or fatigue drift between
    the two halves of the session is free to load onto the condition effect.
    """
    return _centered(np.asarray(dataset.seq_block))
