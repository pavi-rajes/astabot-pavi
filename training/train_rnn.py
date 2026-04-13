"""
Phase 1: Train an ensemble of DynamicRoutingRNNs through the 4-stage curriculum.

Usage:
    conda run -n latent_circuit python training/train_rnn.py

Key hyperparameters:
  N=200, tau=100ms, dt=20ms, alpha=0.2, sigma_rec=0.15
  Optimizer: Adam, lr=1e-3 (1e-4 in Stage 4), grad_clip=1.0
  Regularization: L2 rate (1e-3), L2 weight (1e-4)
  BPTT window: 15-20 trials
  Ensemble size: configurable (default 3 for quick test)

Outputs:
  checkpoints/<seed>/rnn_stage<N>.pt  — model checkpoints
  checkpoints/<seed>/log.csv          — training log
  plots/<seed>/                       — training curves
"""

import os
import sys
import time
import csv
import gc
import argparse
import numpy as np
import torch
import torch.nn as nn

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rnn import DynamicRoutingRNN, make_rnn_fn
from tasks.dynamic_routing import (
    VIS1, VIS2, AUD1, AUD2, STEPS_PER_TRIAL, RESP_START, RESP_END,
    sample_block_trials, N_INPUTS
)
from tasks.curriculum import CurriculumManager
from tasks.session import generate_session
import torch.nn.functional as F


# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_ensemble', type=int, default=3,
                        help='Number of RNNs to train')
    parser.add_argument('--n_units', type=int, default=200)
    parser.add_argument('--max_steps', type=int, default=5000,
                        help='Max training steps per stage (safety limit)')
    parser.add_argument('--eval_every', type=int, default=100,
                        help='Evaluate every N steps')
    parser.add_argument('--n_eval_sessions', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Parallel sessions per gradient step')
    parser.add_argument('--bptt_trials', type=int, default=15,
                        help='Trials per BPTT window')
    parser.add_argument('--max_bptt_windows', type=int, default=8,
                        help='Max BPTT windows per session (randomly sampled). 0=all')
    parser.add_argument('--train_n_blocks', type=int, default=2,
                        help='Blocks per training session for stage 3+ (fewer=faster)')
    parser.add_argument('--outdir', type=str, default='checkpoints')
    parser.add_argument('--seed_start', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    return args


# -------------------------------------------------------------------------
# Loss computation
# -------------------------------------------------------------------------
def compute_loss(rnn, session_data, stage, fa_loss_weight, miss_loss_weight=1.0,
                 lambda_rate=1e-3, lambda_weight=1e-4, device='cpu',
                 bptt_window=15, max_bptt_windows=0, rng=None):
    """
    Compute total loss for one training session.

    session_data : list of (u, target, mask, lick, reward, z_seq) tuples
    bptt_window : int, trials per BPTT window
    max_bptt_windows : int, if >0 randomly sample this many windows (speeds up training)
    """
    if rng is None:
        rng = np.random.default_rng()

    gc.collect()  # prevent heap fragmentation on Windows from accumulated small tensors

    bce_fn = nn.BCELoss(reduction='none')

    total_bce = torch.tensor(0.0, device=device)
    total_rate = torch.tensor(0.0, device=device)
    total_tokens = 0

    n_trials = len(session_data)
    # Build all window start indices
    all_window_starts = list(range(0, n_trials, bptt_window))

    # First pass (no grad): carry hidden state through whole session
    y_states = []  # detached hidden state at the start of each window
    y = torch.zeros(rnn.N, device=device)
    with torch.no_grad():
        for w_start in all_window_starts:
            y_states.append(y.clone())
            w_end = min(w_start + bptt_window, n_trials)
            for ti in range(w_start, w_end):
                u_np = session_data[ti][0]
                u_trial = torch.from_numpy(u_np).float().to(device)  # (T, n_inputs)
                # Precompute static input contribution
                wu = u_trial @ rnn.W_in.T + rnn.b_rec  # (T, N)
                y_t = y.unsqueeze(0)
                for t in range(u_trial.shape[0]):
                    pre = y_t @ rnn.W_rec.T + wu[t]
                    y_t = (1 - rnn.alpha) * y_t + rnn.alpha * torch.relu(pre)
                y = y_t.squeeze(0)

    # Select windows for BPTT (optionally subsample)
    selected_windows = list(range(len(all_window_starts)))
    if max_bptt_windows > 0 and len(selected_windows) > max_bptt_windows:
        selected_windows = sorted(rng.choice(selected_windows, size=max_bptt_windows,
                                             replace=False).tolist())

    # Second pass (with grad): BPTT through selected windows
    for w_idx in selected_windows:
        w_start = all_window_starts[w_idx]
        w_end = min(w_start + bptt_window, n_trials)
        y_w = y_states[w_idx].detach()

        # Collect window data
        window_u      = []
        window_target = []
        window_mask   = []
        window_is_fa  = []
        window_is_miss = []
        for ti in range(w_start, w_end):
            u_np, tgt_np, mask_np, lick, reward, _ = session_data[ti]
            window_u.append(torch.from_numpy(u_np).float().to(device))
            window_target.append(torch.from_numpy(tgt_np).float().to(device))
            window_mask.append(torch.from_numpy(mask_np.astype(np.float32)).to(device))
            window_is_fa.append(lick and (not reward))
            window_is_miss.append((not lick) and (tgt_np.max() > 0.5))

        # Concatenate all trials in window into one long sequence
        u_window = torch.cat(window_u, dim=0)           # (w_size*T, n_inputs)
        wu_window = u_window @ rnn.W_in.T + rnn.b_rec   # precomputed: (w_size*T, N)
        T_per = window_u[0].shape[0]

        y_t = y_w.unsqueeze(0)  # (1, N)
        y_seq_all = []

        for t in range(u_window.shape[0]):
            pre = y_t @ rnn.W_rec.T + wu_window[t]
            # No noise during BPTT — noise is captured via closed-loop session generation
            y_t = (1 - rnn.alpha) * y_t + rnn.alpha * torch.relu(pre)
            y_seq_all.append(y_t)

        y_seq_cat = torch.cat(y_seq_all, dim=0)  # (w_size*T, N)
        del y_seq_all  # release list refs; y_seq_cat holds the graph
        z_seq_cat = torch.sigmoid(y_seq_cat @ rnn.w_out.T + rnn.b_out).squeeze(-1)  # (w_size*T,)

        # Compute loss per trial in window
        for wi in range(len(window_u)):
            t_start_w = wi * T_per
            t_end_w   = t_start_w + T_per
            z_pred_trial = z_seq_cat[t_start_w:t_end_w]  # (T,)
            y_seq_trial  = y_seq_cat[t_start_w:t_end_w]  # (T, N)

            resp_mask = window_mask[wi].bool()
            z_resp = z_pred_trial[resp_mask]
            t_resp = window_target[wi][resp_mask]

            if z_resp.numel() > 0:
                z_resp_clamped = z_resp.clamp(1e-6, 1 - 1e-6)
                bce_vals = bce_fn(z_resp_clamped, t_resp)
                if window_is_fa[wi]:
                    weight = fa_loss_weight
                elif window_is_miss[wi]:
                    weight = miss_loss_weight
                else:
                    weight = 1.0
                total_bce = total_bce + weight * bce_vals.sum()
                total_tokens += z_resp.numel()

            total_rate = total_rate + y_seq_trial.pow(2).mean()

    if total_tokens == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    n_selected_trials = sum(
        min(all_window_starts[w] + bptt_window, n_trials) - all_window_starts[w]
        for w in selected_windows
    )
    loss_bce  = total_bce / total_tokens
    loss_rate = lambda_rate * total_rate / max(n_selected_trials, 1)
    loss_weight = lambda_weight * (
        rnn.W_rec.pow(2).sum() + rnn.W_in.pow(2).sum()
    )

    return loss_bce + loss_rate + loss_weight


# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------
def evaluate(rnn, stage, n_sessions=5, device='cpu', rng=None):
    """
    Run evaluation sessions and compute metrics.
    Stages 1-2: noiseless (deterministic evaluation).
    Stage 3+: with noise — the context inference task requires noise to bootstrap
    licking without which the noiseless hidden state can't maintain context across trials.
    Returns list of session dicts.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Use noise for Stage 3+ to match training conditions (context inference
    # requires the positive feedback loop of lick→reward→context that noise enables)
    add_noise_eval = (stage >= 3)
    rnn_fn = make_rnn_fn(rnn, add_noise=add_noise_eval, device=device)
    sessions = []
    for _ in range(n_sessions):
        sess = generate_session(rnn_fn, stage=stage, rng=rng)
        sessions.append(sess)
    return sessions


# -------------------------------------------------------------------------
# Stage 0 training
# -------------------------------------------------------------------------
def train_stage0(rnn, optimizer, n_steps=100, device='cpu', rng=None):
    """Stage 0: output bias shaping — only train w_out and b_out."""
    if rng is None:
        rng = np.random.default_rng()

    # Freeze recurrent/input weights; only shape the output
    for name, p in rnn.named_parameters():
        p.requires_grad_(name in ('w_out', 'b_out'))

    bce_fn = nn.BCELoss()
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()
        rng_step = np.random.default_rng(rng.integers(1 << 31))

        # Simple: one trial, vis1 presented, reward non-contingent
        u = torch.zeros(STEPS_PER_TRIAL, N_INPUTS, device=device)
        u[75:100, VIS1] = 1.0  # stimulus
        u[80:125, 6] = 1.0     # trial phase
        u[123:125, 4] = 1.0    # non-contingent reward at end of window

        target = torch.ones(STEPS_PER_TRIAL, device=device)

        y0 = torch.zeros(rnn.N, device=device)
        y_seq, z_seq, _ = rnn.forward(y0, u.unsqueeze(0), add_noise=True)
        z_seq = z_seq.squeeze()  # (T,)

        # Loss in response window
        z_resp = z_seq[RESP_START:RESP_END]
        t_resp = target[RESP_START:RESP_END]
        loss = bce_fn(z_resp, t_resp)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss))

    # Unfreeze all parameters for subsequent stages
    for p in rnn.parameters():
        p.requires_grad_(True)

    return np.mean(losses[-50:]) if losses else 0.0


# -------------------------------------------------------------------------
# Context attractor initialization (pre-Stage-3)
# -------------------------------------------------------------------------
def get_context_direction(rnn):
    """
    Return the dominant real eigenvector of W_rec as the context direction.
    This is the direction with the slowest decay (highest effective eigenvalue
    after SR scaling) and therefore the best candidate for context persistence.
    Returns a normalized tensor on the same device as rnn.
    """
    with torch.no_grad():
        W = rnn.W_rec.detach().float()
        eigs, vecs = torch.linalg.eig(W)
        # Prefer real eigenvalues (imaginary part near zero)
        real_eigs = eigs.real
        imag_abs  = eigs.imag.abs()
        # Score: large real part, small imaginary part
        score = real_eigs - 1e3 * imag_abs
        idx = int(score.argmax())
        v = vecs[:, idx].real.float()
        v = v / (v.norm() + 1e-8)
    return v


def make_hinted_rnn_fn(base_rnn_fn, v_context, hint_strength_ref):
    """
    Wrap base_rnn_fn to inject a context hint into the hidden state after
    each instruction trial during Stage 3 training.

    After instruction trial in VIS1 block: h_new += hint_strength * v_context
    After instruction trial in AUD1 block:  h_new -= hint_strength * v_context

    Parameters
    ----------
    base_rnn_fn  : callable  (h, stim, rewarded, instr, rng) -> (u, ..., h_new)
    v_context    : torch.Tensor (N,), normalized context direction
    hint_strength_ref : list of length 1, e.g. [1.0]; updated externally to fade hint
    """
    def hinted_fn(h, stim, rewarded, instr, rng):
        u, target, mask, lick, reward, z_seq, h_new = base_rnn_fn(
            h, stim, rewarded, instr, rng
        )
        strength = hint_strength_ref[0]
        if instr and strength > 0.0:
            sign = +1.0 if rewarded == VIS1 else -1.0
            h_new = h_new + sign * strength * v_context
        return u, target, mask, lick, reward, z_seq, h_new
    return hinted_fn


def initialize_context_attractor(rnn, alpha=0.2, target_eff_eig=0.9999):
    """
    Scale W_rec uniformly so that its spectral radius equals lambda_target =
    (target_eff_eig - (1-alpha)) / alpha.

    This maximizes the persistence of the dominant recurrent eigenmode(s).
    The dominant eigenmode will receive differential drive from VIS1 vs AUD1
    instruction trials (since W_in[:,VIS1] ≠ W_in[:,AUD1]), so it encodes
    context. With eff_eig close to 1.0, this context persists across trials.

    Why scaling instead of rank-1 update:
      The rank-1 update approach (previous versions) created SR > 1.0 because
      the update direction v was not an eigenvector of W_rec, so the rank-1
      perturbation pushed other eigenvalues above 1.0. Post-init clipping then
      scaled everything down, negating the benefit. Uniform scaling avoids this:
      SR scales exactly by the intended factor, no eigenvalues exceed lambda_target.

    Parameters
    ----------
    target_eff_eig : float, default 0.9990
        Target effective eigenvalue (= eff max eigenvalue after scaling).
        0.9999 → SR = 0.9995, eff_eig = 0.9999, half-life = 6930 steps = 55 trials
               → after 90 trials context retains 0.9999^11250 = 32% of original
               (vs 0.980^11250 = 10^-97 without the attractor)
    """
    with torch.no_grad():
        lambda_target = (target_eff_eig - (1 - alpha)) / alpha

        eigs = torch.linalg.eigvals(rnn.W_rec)
        sr_current = float(eigs.abs().max())

        if sr_current < 1e-6:
            print("  Context attractor init skipped: W_rec ≈ 0.")
            return

        scale = lambda_target / sr_current
        rnn.W_rec.data.mul_(scale)

        eff_eig = (1 - alpha) + alpha * lambda_target
        hl_steps = 0.693 / abs(1 - eff_eig) if abs(1 - eff_eig) > 1e-9 else float('inf')
        print(f"  Context attractor (SR scaling): {sr_current:.4f} -> {lambda_target:.4f} "
              f"(x{scale:.3f}), eff_eig={eff_eig:.4f} "
              f"(half-life ~{hl_steps:.0f} steps = {hl_steps/125:.1f} trials)")


# -------------------------------------------------------------------------
# General training step
# -------------------------------------------------------------------------
def train_step(rnn, optimizer, stage, fa_loss_weight, miss_loss_weight=3.0,
               batch_size=4, device='cpu', rng=None, max_bptt_windows=8,
               bptt_trials=15, train_n_blocks=2, max_sr=2.0,
               rnn_fn_override=None):
    """One gradient step using batch_size sessions."""
    if rng is None:
        rng = np.random.default_rng()

    rnn_fn = rnn_fn_override if rnn_fn_override is not None \
             else make_rnn_fn(rnn, add_noise=True, device=device)
    batch_loss = torch.tensor(0.0, device=device)
    optimizer.zero_grad()

    # Use fewer blocks per training session for speed (applies to all stages)
    n_blocks = train_n_blocks

    for _ in range(batch_size):
        sess = generate_session(rnn_fn, stage=stage, rng=rng,
                                n_blocks_override=n_blocks)
        n_trials = len(sess['inputs'])
        session_data = []
        for ti in range(n_trials):
            session_data.append((
                sess['inputs'][ti], sess['targets'][ti], sess['masks'][ti],
                sess['licks'][ti],  sess['rewards'][ti], sess['z_seqs'][ti],
            ))

        # Miss weight only for Stage 2+ (Stage 1 doesn't need it; seeds 0/42 worked fine without it)
        effective_miss_weight = miss_loss_weight if stage >= 2 else 1.0
        loss = compute_loss(rnn, session_data, stage, fa_loss_weight,
                            miss_loss_weight=effective_miss_weight,
                            device=device, bptt_window=bptt_trials,
                            max_bptt_windows=max_bptt_windows, rng=rng)
        batch_loss = batch_loss + loss / batch_size

    gc.collect()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
    optimizer.step()

    # Stability projection for W_rec.
    # Stage 3+: use SPECTRAL RADIUS clipping (eigenvalue-based) so the
    # context attractor eigenvalue (0.999) is preserved while keeping all
    # eigenvalues ≤ max_sr=1.0. This prevents the instability (NaN) caused
    # by large operator norms when the rank-1 attractor update is in a direction
    # that increases the operator norm above the stability threshold.
    # Stages 1-2: use faster operator norm (singular value) clipping.
    with torch.no_grad():
        if stage >= 3:
            eigs = torch.linalg.eigvals(rnn.W_rec)
            spectral_radius = float(eigs.abs().max())
            if spectral_radius > max_sr:
                rnn.W_rec.data.mul_(max_sr / spectral_radius)
        else:
            op_norm = torch.linalg.norm(rnn.W_rec, ord=2)
            if op_norm > max_sr:
                rnn.W_rec.data.mul_(max_sr / op_norm)

    return float(batch_loss)


# -------------------------------------------------------------------------
# Main training loop
# -------------------------------------------------------------------------
def train_rnn(seed, cfg):
    device = torch.device(cfg.device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    outdir = os.path.join(cfg.outdir, f'seed_{seed:02d}')
    os.makedirs(outdir, exist_ok=True)

    log_path = os.path.join(outdir, 'log.csv')
    log_fields = ['step', 'stage', 'loss', 'dprime_intra', 'dprime_inter',
                  'n_pass_blocks', 'advanced', 'regressed']

    rnn = DynamicRoutingRNN(n_units=cfg.n_units).to(device)
    curriculum = CurriculumManager()

    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

    total_steps = 0
    MAX_STAGE_STEPS = cfg.max_steps

    print(f"\n=== Seed {seed} ===")

    # Stage 0
    print("Stage 0: auto-rewards")
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)
    loss0 = train_stage0(rnn, optimizer, n_steps=100, device=str(device), rng=rng)
    total_steps += 100
    curriculum.stage = 1
    print(f"  Stage 0 done. Final loss: {loss0:.4f}")
    torch.save(rnn.state_dict(), os.path.join(outdir, 'rnn_stage0.pt'))

    # Stages 1–4
    while curriculum.stage <= 4 and total_steps < MAX_STAGE_STEPS * 5:
        stage = curriculum.stage
        print(f"\nStage {stage}: training...")
        optimizer = torch.optim.Adam(rnn.parameters(), lr=curriculum.learning_rate)

        stage_steps = 0

        # Stage 3: initialize a context-persistence attractor in W_rec before training.
        # Without this, the VIS1-vs-AUD1 context signal decays ~10^-12 per trial,
        # making it impossible for BPTT to learn context-dependent suppression.
        # The rank-1 update creates a near-1 effective eigenvalue in the context
        # direction so that instruction-trial context signals persist across trials.
        # Teacher forcing state for Stage 3 (reset each time Stage 3 begins)
        hint_strength_ref = [0.0]
        v_context = None

        if stage == 3:
            print("  Initializing context attractor in W_rec...")
            initialize_context_attractor(rnn, alpha=rnn.alpha)
            # Verify SR after attractor init — SR scaling ensures SR = lambda_target < 1.0.
            with torch.no_grad():
                eigs = torch.linalg.eigvals(rnn.W_rec)
                sr_post = float(eigs.abs().max())
                if sr_post > 1.0:
                    rnn.W_rec.data.mul_(1.0 / sr_post)
                    print(f"  WARNING: Post-init SR clip triggered: {sr_post:.4f} -> 1.0000")
                else:
                    print(f"  Post-init SR verified: {sr_post:.4f} < 1.0 (attractor preserved)")

            # Teacher forcing: inject ±v_context into h after instruction trials.
            # hint_strength fades 1.0 → 0 over HINT_FADE_STEPS training steps.
            # This bootstraps context-dependent licking so reward feedback becomes
            # informative and BPTT can learn context inference.
            HINT_FADE_STEPS = 1000
            v_context = get_context_direction(rnn).to(device)
            hint_strength_ref = [1.0]
            print(f"  Teacher forcing enabled: hint fades over {HINT_FADE_STEPS} steps.")

        # Stage 3+: use full-block BPTT (90 trials/block) so gradients span the
        # instruction trial → end of block, and the initial state of each BPTT window
        # carries context encoded by the previous block (via context attractor).
        # Two windows of 90 trials each (one per block) gives the same total compute
        # as one 180-trial window, but is 3-5x faster due to shorter computation graphs.
        # Stage 1-2: use shorter windows (faster, context inference not needed).
        if stage >= 3:
            stage_bptt_trials = 90    # ~full block (5 instr + 85 regular)
            stage_max_bptt = cfg.train_n_blocks  # one window per training block
        else:
            stage_bptt_trials = cfg.bptt_trials
            stage_max_bptt = cfg.max_bptt_windows

        # Stage 3: use spectral RADIUS ≤ 1.0 (all eigenvalues bounded by 1).
        # The context attractor eigenvalue = 0.999 is preserved (< 1.0).
        # Operator norm clipping would destroy the attractor (adds large singular
        # value to W_rec); eigenvalue clipping is used in train_step for Stage 3.
        stage_max_sr = 1.0 if stage >= 3 else 2.0

        while stage_steps < MAX_STAGE_STEPS:
            # Stage 3 teacher forcing: build hinted rnn_fn with fading hint
            rnn_fn_for_step = None
            if stage == 3 and v_context is not None:
                hint_strength_ref[0] = max(0.0, 1.0 - stage_steps / HINT_FADE_STEPS)
                base_fn = make_rnn_fn(rnn, add_noise=True, device=str(device))
                rnn_fn_for_step = make_hinted_rnn_fn(base_fn, v_context, hint_strength_ref)

            loss = train_step(
                rnn, optimizer, stage, curriculum.fa_loss_weight,
                miss_loss_weight=3.0,
                batch_size=cfg.batch_size, device=str(device), rng=rng,
                max_bptt_windows=stage_max_bptt,
                bptt_trials=stage_bptt_trials,
                train_n_blocks=cfg.train_n_blocks,
                max_sr=stage_max_sr,
                rnn_fn_override=rnn_fn_for_step,
            )
            total_steps += 1
            stage_steps += 1

            if stage_steps % cfg.eval_every == 0:
                eval_sessions = evaluate(
                    rnn, stage, n_sessions=cfg.n_eval_sessions,
                    device=str(device), rng=rng
                )
                metrics, advanced, regressed = curriculum.evaluate(
                    eval_sessions, total_steps
                )

                dp_intra = metrics.get('dprime_intra', metrics.get('dprime', 0.0))
                dp_inter = metrics.get('dprime_inter', 0.0)
                n_pass   = metrics.get('n_pass_blocks', 0)

                hint_str = (f" | hint {hint_strength_ref[0]:.2f}"
                            if stage == 3 and v_context is not None else "")
                print(f"  step {total_steps:5d} | stage {stage} | "
                      f"loss {loss:.4f} | d'_intra {dp_intra:.2f} | "
                      f"d'_inter {dp_inter:.2f} | pass_blocks {n_pass}"
                      f"{hint_str} | "
                      f"{'ADVANCED' if advanced else 'REGRESSED' if regressed else ''}")

                with open(log_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=log_fields)
                    writer.writerow({
                        'step': total_steps, 'stage': stage, 'loss': f'{loss:.6f}',
                        'dprime_intra': f'{dp_intra:.4f}',
                        'dprime_inter': f'{dp_inter:.4f}',
                        'n_pass_blocks': n_pass,
                        'advanced': int(advanced), 'regressed': int(regressed),
                    })

                if advanced:
                    torch.save(rnn.state_dict(),
                               os.path.join(outdir, f'rnn_stage{stage}.pt'))
                    print(f"  -> Advanced to Stage {curriculum.stage}")
                    if stage == 4:
                        # Stage 4 done = convergence
                        print("  -> Stage 4 converged! Training complete.")
                        torch.save(rnn.state_dict(),
                                   os.path.join(outdir, 'rnn_final.pt'))
                        return rnn
                    break

                if regressed:
                    print(f"  -> Regressed to Stage {curriculum.stage}")
                    break

        else:
            print(f"  WARNING: Stage {stage} hit max_steps limit ({MAX_STAGE_STEPS}). "
                  f"Forcing advancement.")
            curriculum.stage = min(stage + 1, 4)
            torch.save(rnn.state_dict(), os.path.join(outdir, f'rnn_stage{stage}.pt'))

    torch.save(rnn.state_dict(), os.path.join(outdir, 'rnn_final.pt'))
    print(f"\nTraining complete. Final stage: {curriculum.stage}")
    return rnn


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == '__main__':
    cfg = get_config()

    os.makedirs(cfg.outdir, exist_ok=True)
    start = time.time()

    trained_rnns = []
    for i in range(cfg.n_ensemble):
        seed = cfg.seed_start + i
        rnn = train_rnn(seed, cfg)
        trained_rnns.append(rnn)
        elapsed = time.time() - start
        print(f"[Ensemble {i+1}/{cfg.n_ensemble}] Total elapsed: {elapsed/60:.1f} min")

    print(f"\nAll {cfg.n_ensemble} RNNs trained. Checkpoints in: {cfg.outdir}/")
