"""
Dynamic routing task constants for coarse time resolution.

dt=100ms, tau=500ms → alpha=0.2 (same leak rate as standard).

Key advantage over dt=20ms:
  eff_lambda = 0.8 + 0.2*0.9 = 0.98 per STEP = 0.98 per 100ms
  After 1 trial (25 steps): 0.98^25 = 60% context retained
  vs dt=20ms: 0.98^125 = 0.8% (essentially zero)

Trial structure (same real-time durations, 5x fewer steps):
  Quiescent:      t=0 to 1.5s    → steps 0-14   (15 steps)
  Stimulus on:    t=1.5 to 2.0s  → steps 15-19  (5 steps)
  Response window:t=1.6 to 2.5s  → steps 16-24  (9 steps)
  Total:          2.5s            → 25 steps
"""

import numpy as np

DT  = 0.100
TAU = 0.500
ALPHA = DT / TAU   # 0.2 — same as standard

STEPS_PER_TRIAL  = 25
STIM_ON          = 15   # t = 1.5s
STIM_OFF         = 20   # t = 2.0s
RESP_START       = 16   # t = 1.6s
RESP_END         = 25   # t = 2.5s
REWARD_DURATION  = 1    # 1 step = 100ms (was 2 steps × 20ms = 40ms)

N_INPUTS  = 7
N_OUTPUTS = 1

VIS1 = 0
VIS2 = 1
AUD1 = 2
AUD2 = 3

STIMULUS_NAMES = {VIS1: 'vis1', VIS2: 'vis2', AUD1: 'aud1', AUD2: 'aud2', None: 'catch'}


def compute_dprime(hit_rate, fa_rate, clip=0.01):
    from scipy.stats import norm
    hr = np.clip(hit_rate, clip, 1 - clip)
    fa = np.clip(fa_rate, clip, 1 - clip)
    return float(norm.ppf(hr) - norm.ppf(fa))


def sample_block_trials(rewarded_target_idx, n_instruction=5,
                        sub_block_size=20, catch_prob=0.1, rng=None):
    """Same trial structure as standard — returns trial specs (no step constants here)."""
    if rng is None:
        rng = np.random.default_rng()
    trials = []
    for _ in range(n_instruction):
        trials.append({'stimulus_idx': rewarded_target_idx, 'instruction': True})
    stimuli = [VIS1, VIS2, AUD1, AUD2]
    for _ in range(5):
        sub = []
        for s in stimuli:
            sub.extend([s] * 4)
        n_catch = max(1, int(round(sub_block_size * catch_prob)))
        sub.extend([None] * n_catch)
        while len(sub) < sub_block_size:
            sub.append(rng.choice(stimuli))
        sub = sub[:sub_block_size]
        rng.shuffle(sub)
        for s in sub:
            trials.append({'stimulus_idx': s, 'instruction': False})
    return trials
