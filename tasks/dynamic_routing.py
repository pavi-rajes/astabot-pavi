"""
Dynamic routing task: trial generation, stimulus sampling, and target computation.

Trial structure (all times in seconds, steps at dt=20ms):
  Quiescent period: t=-1.5 to 0 s   (steps 0-74,  75 steps)
  Stimulus on:      t=0 to 0.5 s    (steps 75-99, 25 steps)
  Response window:  t=0.1 to 1.0 s  (steps 80-124, 45 steps)
  Total:            125 steps per trial
"""

import numpy as np

# ---- Constants ---------------------------------------------------------------
DT = 0.020          # 20 ms per step
TAU = 0.100         # 100 ms time constant
ALPHA = DT / TAU    # 0.2

STEPS_PER_TRIAL = 125
QUIESCENT_END   = 75    # step index where stimulus can begin
STIM_ON         = 75    # step at which stimulus turns on
STIM_OFF        = 100   # step at which stimulus turns off
RESP_START      = 80    # first step of response window (t=0.1s)
RESP_END        = 125   # one past last step of response window (t=1.0s)
REWARD_DURATION = 2     # reward/lick channels pulse for 2 steps (40 ms)

N_INPUTS  = 7           # u[0..3]=stimuli, u[4]=reward, u[5]=lick, u[6]=phase
N_OUTPUTS = 1

# Stimulus indices (u channel)
VIS1 = 0
VIS2 = 1
AUD1 = 2
AUD2 = 3

STIMULUS_NAMES = {VIS1: 'vis1', VIS2: 'vis2', AUD1: 'aud1', AUD2: 'aud2', None: 'catch'}


def build_trial_inputs(stimulus_idx, rewarded_target_idx, z_sequence=None,
                       instruction=False, dt_steps=STEPS_PER_TRIAL):
    """
    Build the input tensor u for a single trial.

    Parameters
    ----------
    stimulus_idx : int or None
        Which stimulus is presented (VIS1, VIS2, AUD1, AUD2) or None for catch.
    rewarded_target_idx : int
        Which stimulus is currently rewarded (VIS1 or AUD1).
    z_sequence : array-like of shape (STEPS_PER_TRIAL,) or None
        Network output at each step (used to compute closed-loop reward/lick).
        If None, a zero-output sequence is assumed (no licks).
    instruction : bool
        If True, apply late_autoreward logic: non-contingent reward at end of
        response window if no lick occurred during window.

    Returns
    -------
    u : np.ndarray, shape (N_INPUTS, STEPS_PER_TRIAL)
        Input tensor. Channels: [vis1, vis2, aud1, aud2, reward, own_lick, phase]
    target : np.ndarray, shape (STEPS_PER_TRIAL,)
        Target output (1.0 = lick, 0.0 = no lick). Defined only during response window.
    response_mask : np.ndarray, shape (STEPS_PER_TRIAL,), bool
        True for timesteps in the response window.
    lick_occurred : bool
        Whether a lick (z > 0.5) occurred during the response window.
    reward_delivered : bool
        Whether a reward was delivered.
    """
    u = np.zeros((N_INPUTS, dt_steps), dtype=np.float32)

    # Stimulus channels (on for steps 75–99)
    if stimulus_idx is not None:
        u[stimulus_idx, STIM_ON:STIM_OFF] = 1.0

    # Trial phase channel (on during response window)
    u[6, RESP_START:RESP_END] = 1.0

    # Response mask
    response_mask = np.zeros(dt_steps, dtype=bool)
    response_mask[RESP_START:RESP_END] = True

    # Target output
    target = np.zeros(dt_steps, dtype=np.float32)
    is_rewarded_stimulus = (stimulus_idx is not None) and (stimulus_idx == rewarded_target_idx)
    if is_rewarded_stimulus:
        target[RESP_START:RESP_END] = 1.0

    # Closed-loop reward/lick feedback
    if z_sequence is None:
        z_sequence = np.zeros(dt_steps, dtype=np.float32)

    lick_occurred = False
    reward_delivered = False
    lick_step = None

    # Find first lick in response window
    for t in range(RESP_START, RESP_END):
        if float(z_sequence[t]) > 0.5:
            lick_occurred = True
            lick_step = t
            break

    if lick_occurred and lick_step is not None:
        # Own-lick channel fires for 2 steps
        lick_end = min(lick_step + REWARD_DURATION, dt_steps)
        u[5, lick_step:lick_end] = 1.0

        if is_rewarded_stimulus:
            # Contingent reward: fires simultaneously with lick
            u[4, lick_step:lick_end] = 1.0
            reward_delivered = True

    if instruction and not reward_delivered:
        # late_autoreward: non-contingent reward at end of response window
        auto_start = RESP_END - REWARD_DURATION
        auto_end = RESP_END
        u[4, auto_start:auto_end] = 1.0
        reward_delivered = True

    return u, target, response_mask, lick_occurred, reward_delivered


def sample_block_trials(rewarded_target_idx, n_instruction=5,
                        sub_block_size=20, catch_prob=0.1, rng=None):
    """
    Sample the trial sequence for one block.

    Returns a list of dicts with keys:
        'stimulus_idx': int or None (catch)
        'instruction': bool
    """
    if rng is None:
        rng = np.random.default_rng()

    trials = []

    # Instruction trials
    for _ in range(n_instruction):
        trials.append({'stimulus_idx': rewarded_target_idx, 'instruction': True})

    # Regular trials: even sampling in sub-blocks of 20, catch at 10%
    stimuli = [VIS1, VIS2, AUD1, AUD2]
    # Target ~90 regular trials; pad to a multiple of sub_block_size
    # The sub-block has 4 stimuli × 4 repeats = 16 non-catch + 4 catch ≈ 20 trials
    # We aim for ~85 regular trials (after 5 instruction) → ~5 sub-blocks
    n_subblocks = 5
    for _ in range(n_subblocks):
        sub = []
        # 16 non-catch: each stimulus 4 times
        for s in stimuli:
            sub.extend([s] * 4)
        # ~4 catch trials to reach ~20
        n_catch = max(1, int(round(sub_block_size * catch_prob)))
        sub.extend([None] * n_catch)
        # Pad to sub_block_size
        while len(sub) < sub_block_size:
            sub.append(rng.choice(stimuli))
        sub = sub[:sub_block_size]
        rng.shuffle(sub)
        for s in sub:
            trials.append({'stimulus_idx': s, 'instruction': False})

    return trials


def compute_dprime(hit_rate, fa_rate, clip=0.01):
    """
    Compute d' = Z(hit_rate) - Z(fa_rate).
    Rates are clipped to [clip, 1-clip] to avoid ±inf.
    """
    from scipy.stats import norm
    hr = np.clip(hit_rate, clip, 1 - clip)
    fa = np.clip(fa_rate, clip, 1 - clip)
    return float(norm.ppf(hr) - norm.ppf(fa))
