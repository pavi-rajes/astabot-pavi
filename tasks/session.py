"""
Session and block sequence generation for the dynamic routing task.

A session consists of 6 blocks, each with 5 instruction trials followed by
~85 regular trials. The rewarded target alternates each block (vis1 → aud1 → ...).
"""

import numpy as np
from tasks.dynamic_routing import (
    VIS1, VIS2, AUD1, AUD2, sample_block_trials, build_trial_inputs,
    STEPS_PER_TRIAL, N_INPUTS
)


def generate_session(
    rnn_fn,
    first_rewarded=None,
    n_blocks=6,
    n_instruction=5,
    rng=None,
    stage=3,
    n_blocks_override=None,
):
    """
    Generate one full session by running the RNN forward trial-by-trial.

    Parameters
    ----------
    rnn_fn : callable
        Function that takes (h, u_trial) where h is the hidden state array
        (N,) and u_trial is (N_INPUTS, STEPS_PER_TRIAL), and returns
        (h_new, z_sequence) where z_sequence has shape (STEPS_PER_TRIAL,).
    first_rewarded : int or None
        First block's rewarded target (VIS1 or AUD1). If None, chosen randomly.
    n_blocks : int
        Number of blocks per session.
    n_instruction : int
        Number of instruction trials at the start of each block.
    rng : np.random.Generator or None
    stage : int
        Current training stage (affects what stimuli are used).

    Returns
    -------
    session : dict with keys:
        'inputs'     : list of (N_INPUTS, STEPS_PER_TRIAL) arrays, one per trial
        'targets'    : list of (STEPS_PER_TRIAL,) arrays
        'masks'      : list of (STEPS_PER_TRIAL,) bool arrays
        'block_ids'  : list of int (block index for each trial)
        'rewarded'   : list of int (rewarded target for each trial)
        'stimulus'   : list of int or None (stimulus identity per trial)
        'instruction': list of bool (whether instruction trial)
        'licks'      : list of bool (whether network licked during resp window)
        'rewards'    : list of bool (whether reward was delivered)
        'z_seqs'     : list of (STEPS_PER_TRIAL,) arrays (network output)
        'h_final'    : np.ndarray, final hidden state
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_blocks_override is not None:
        n_blocks = n_blocks_override

    if first_rewarded is None:
        first_rewarded = rng.choice([VIS1, AUD1])

    # Stages 1-2: fixed rewarded target (no alternation — avoids contradictory supervision)
    # Stage 1: always VIS1; Stage 2: always AUD1
    # Stage 3+: normal alternation (context inference is the task)
    if stage == 1:
        rewarded_sequence = [VIS1] * n_blocks
    elif stage == 2:
        rewarded_sequence = [AUD1] * n_blocks
    else:
        rewarded_sequence = []
        for b in range(n_blocks):
            if b % 2 == 0:
                rewarded_sequence.append(first_rewarded)
            else:
                rewarded_sequence.append(AUD1 if first_rewarded == VIS1 else VIS1)

    # Generate trial lists per block (stage-dependent)
    all_trial_specs = []
    for b in range(n_blocks):
        if stage <= 1:
            trials = _stage1_trials(rewarded_sequence[b], rng)
        elif stage == 2:
            trials = _stage2_trials(rewarded_sequence[b], rng)
        else:
            # Stage 3/4: all stimuli
            trials = sample_block_trials(
                rewarded_sequence[b],
                n_instruction=n_instruction,
                rng=rng,
            )
        all_trial_specs.append((b, rewarded_sequence[b], trials))

    # Run the session
    session = {
        'inputs': [], 'targets': [], 'masks': [],
        'block_ids': [], 'rewarded': [], 'stimulus': [],
        'instruction': [], 'licks': [], 'rewards': [], 'z_seqs': [],
    }

    h = None  # hidden state; None means rnn_fn initializes to zero

    for b_idx, rewarded, trials in all_trial_specs:
        for spec in trials:
            stim = spec['stimulus_idx']
            instr = spec['instruction']

            # First pass with z=None to get z_sequence (or zero)
            # For closed-loop feedback we need to integrate forward in time
            # Delegate to rnn_fn which handles the closed-loop internally
            u, target, mask, lick, reward, z_seq, h = rnn_fn(
                h, stim, rewarded, instr, rng
            )

            session['inputs'].append(u)
            session['targets'].append(target)
            session['masks'].append(mask)
            session['block_ids'].append(b_idx)
            session['rewarded'].append(rewarded)
            session['stimulus'].append(stim)
            session['instruction'].append(instr)
            session['licks'].append(lick)
            session['rewards'].append(reward)
            session['z_seqs'].append(z_seq)

    session['h_final'] = h
    return session


def _stage1_trials(rewarded_target, rng, n_trials=80):
    """Stage 1: visual discrimination only (vis1 vs vis2)."""
    trials = []
    for _ in range(n_trials):
        s = rng.choice([VIS1, VIS2])
        trials.append({'stimulus_idx': s, 'instruction': False})
    return trials


def _stage2_trials(rewarded_target, rng, n_trials=80):
    """Stage 2: auditory discrimination only (aud1 vs aud2)."""
    trials = []
    for _ in range(n_trials):
        s = rng.choice([AUD1, AUD2])
        trials.append({'stimulus_idx': s, 'instruction': False})
    return trials
