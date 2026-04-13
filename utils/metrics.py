"""
Metrics utilities: d', r², Pearson correlation coefficients.
"""

import numpy as np
from scipy.stats import norm


def dprime(hit_rate, fa_rate, clip=0.01):
    """d' = Z(hit_rate) - Z(fa_rate), clipped to avoid ±inf."""
    hr = np.clip(hit_rate, clip, 1 - clip)
    fa = np.clip(fa_rate,  clip, 1 - clip)
    return float(norm.ppf(hr) - norm.ppf(fa))


def r_squared(y_true, y_pred):
    """
    Coefficient of determination R².
    Works on multi-dimensional arrays (flattened).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def pearson_r(a, b):
    """Pearson correlation between two flat arrays."""
    a = np.asarray(a).ravel().astype(float)
    b = np.asarray(b).ravel().astype(float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def hit_and_fa_rates(session, rewarded_target, distractor):
    """
    Compute hit rate and FA rate for one session.

    Parameters
    ----------
    session : dict (output of generate_session)
    rewarded_target : int
    distractor : int

    Returns
    -------
    hit_rate : float
    fa_rate : float
    n_hits_trials : int
    n_fa_trials : int
    """
    stims    = np.array(session['stimulus'], dtype=object)
    licks    = np.array(session['licks'])
    instr    = np.array(session['instruction'])
    rewarded = np.array(session['rewarded'])

    regular = ~instr & (rewarded == rewarded_target)
    stims_r = stims[regular]
    licks_r = licks[regular]

    hit_mask = stims_r == rewarded_target
    fa_mask  = stims_r == distractor

    hr = float(np.mean(licks_r[hit_mask])) if hit_mask.sum() > 0 else 0.5
    fa = float(np.mean(licks_r[fa_mask]))  if fa_mask.sum()  > 0 else 0.5
    return hr, fa, int(hit_mask.sum()), int(fa_mask.sum())


def context_inference_speed(session, rewarded_target, distractor, threshold_dp=1.0,
                             window=10):
    """
    After a block transition, how many trials until d' exceeds threshold_dp?

    Returns trial index (relative to block start) or None if never exceeded.
    """
    block_ids = np.array(session['block_ids'])
    stims     = np.array(session['stimulus'], dtype=object)
    licks     = np.array(session['licks'])
    rewarded  = np.array(session['rewarded'])
    instr     = np.array(session['instruction'])

    transition_speeds = []

    unique_blocks = np.unique(block_ids)
    for b in unique_blocks[1:]:  # skip first block (no prior context)
        block_mask = (block_ids == b) & (rewarded == rewarded_target) & ~instr
        if block_mask.sum() < window:
            continue

        idx = np.where(block_mask)[0]
        for start in range(len(idx) - window + 1):
            window_idx = idx[start:start + window]
            s_w = stims[window_idx]
            l_w = licks[window_idx]

            hm = s_w == rewarded_target
            fm = s_w == distractor
            if hm.sum() == 0 or fm.sum() == 0:
                continue

            hr = float(np.mean(l_w[hm]))
            fa = float(np.mean(l_w[fm]))
            dp = dprime(hr, fa)
            if dp >= threshold_dp:
                transition_speeds.append(start + window)
                break

    return transition_speeds
