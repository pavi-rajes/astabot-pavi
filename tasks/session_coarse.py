"""
Session generator for coarse-dt (dt=100ms) variant.
Identical logic to session.py but uses dynamic_routing_coarse constants.
"""

import numpy as np
from tasks.dynamic_routing_coarse import (
    VIS1, AUD1, sample_block_trials
)


def generate_session_coarse(rnn_fn, first_rewarded=None, n_blocks=6,
                             n_instruction=5, rng=None, stage=3,
                             n_blocks_override=None):
    if rng is None:
        rng = np.random.default_rng()
    if n_blocks_override is not None:
        n_blocks = n_blocks_override
    if first_rewarded is None:
        first_rewarded = rng.choice([VIS1, AUD1])

    if stage == 1:
        rewarded_sequence = [VIS1] * n_blocks
    elif stage == 2:
        rewarded_sequence = [AUD1] * n_blocks
    else:
        rewarded_sequence = [
            first_rewarded if b % 2 == 0 else (AUD1 if first_rewarded == VIS1 else VIS1)
            for b in range(n_blocks)
        ]

    all_trial_specs = []
    for b in range(n_blocks):
        if stage <= 1:
            trials = _stage1_trials(rewarded_sequence[b], rng)
        elif stage == 2:
            trials = _stage2_trials(rewarded_sequence[b], rng)
        else:
            trials = sample_block_trials(rewarded_sequence[b],
                                         n_instruction=n_instruction, rng=rng)
        all_trial_specs.append((b, rewarded_sequence[b], trials))

    session = {
        'inputs': [], 'targets': [], 'masks': [],
        'block_ids': [], 'rewarded': [], 'stimulus': [],
        'instruction': [], 'licks': [], 'rewards': [], 'z_seqs': [],
    }
    h = None

    for b_idx, rewarded, trials in all_trial_specs:
        for spec in trials:
            u, target, mask, lick, reward, z_seq, h = rnn_fn(
                h, spec['stimulus_idx'], rewarded, spec['instruction'], rng
            )
            session['inputs'].append(u)
            session['targets'].append(target)
            session['masks'].append(mask)
            session['block_ids'].append(b_idx)
            session['rewarded'].append(rewarded)
            session['stimulus'].append(spec['stimulus_idx'])
            session['instruction'].append(spec['instruction'])
            session['licks'].append(lick)
            session['rewards'].append(reward)
            session['z_seqs'].append(z_seq)

    session['h_final'] = h
    return session


def _stage1_trials(rewarded_target, rng, n_trials=80):
    return [{'stimulus_idx': rng.choice([VIS1, 1]), 'instruction': False}
            for _ in range(n_trials)]


def _stage2_trials(rewarded_target, rng, n_trials=80):
    return [{'stimulus_idx': rng.choice([AUD1, 3]), 'instruction': False}
            for _ in range(n_trials)]
