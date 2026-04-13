"""
Context Teacher Forcing — bridges explicit context (works) and pure integrator (doesn't yet).

The key insight from ablations:
  - Explicit ctx (±1.0 on channel 7): Stage 3 in 750 steps, monotonic learning ✅
  - Pure integrator (c learned from rewards): oscillates near 0 after 10000 steps ❌
  - Root cause of failure: bootstrap — c decays between rewards before network learns to use it

Solution: blend explicit and learned context with a fading schedule.
  ctx_input(t) = (1 - alpha) * c_integrator  +  alpha * c_explicit
  alpha decays linearly from alpha_start → 0 over ctx_fade steps.

Early training (alpha≈1): network sees near-perfect context signal, learns routing quickly.
As alpha fades: gradient forces W_in[:,7] to replicate the explicit signal.
Late training (alpha=0): pure integrator — but W_in[:,7] already trained.

This is exactly analogous to teacher forcing for the output (v25), but applied to the
context channel instead of the lick output.

Key difference from v25 failure: v25 faded the hint on the OUTPUT signal. When the hint
faded, the RNN had no mechanism to maintain context (0.8%/trial). Here, the integrator
DOES maintain context (c persists across trials) — it just needs to learn to be used.
"""

import os
import sys
import csv
import gc
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rnn_integrator_coarse import IntegratorRNNCoarse
from tasks.session_coarse import generate_session_coarse
from tasks.dynamic_routing_coarse import VIS1, VIS2, AUD1, AUD2
from tasks.curriculum import CurriculumManager
from training.train_rnn import initialize_context_attractor
from training.train_coarse import train_stage0_coarse
from training.train_integrator_coarse_ctx import compute_loss_ctx

VIS_STIMS = {VIS1, VIS2}
AUD_STIMS  = {AUD1, AUD2}


def make_ctx_teacher_rnn_fn(rnn, gamma=0.90, alpha=0.0, add_noise=True, device='cpu'):
    """
    Session-compatible rnn_fn that blends explicit and integrator context.

    ctx_input = (1 - alpha) * c_integrator + alpha * c_explicit
    c_explicit = +1.0 if rewarded == VIS1, -1.0 if rewarded == AUD1
    alpha=1.0 → pure explicit (same as train_explicit_ctx)
    alpha=0.0 → pure integrator (same as train_integrator_coarse_ctx)
    """
    c = [0.0]

    def rnn_fn(h, stim, rewarded, instr, rng):
        if h is None:
            h = torch.zeros(rnn.N, device=device)
            c[0] = 0.0

        c_explicit = 1.0 if rewarded == VIS1 else -1.0
        ctx_input = (1.0 - alpha) * c[0] + alpha * c_explicit

        u, target, mask, lick, reward, z_seq, h_new = \
            rnn.forward_trial_closedloop(
                h, stim, rewarded, instr,
                add_noise=add_noise, device=device,
                integrator_value=ctx_input,
            )

        # Always update the integrator from actual reward signal
        is_target = (stim is not None) and (stim == rewarded)
        if reward:
            delta = 1.0
        elif is_target and not lick:
            delta = -0.5
        else:
            delta = 0.0
        c[0] = float(np.tanh(gamma * c[0] + (1 - gamma) * delta))

        return u, target, mask, lick, reward, z_seq, h_new

    return rnn_fn


def get_config():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', default='checkpoints_ctx_teacher')
    p.add_argument('--stage2_ckpt', default='checkpoints_v24/seed_42/rnn_stage2.pt')
    p.add_argument('--n_units', type=int, default=100)
    p.add_argument('--gamma', type=float, default=0.90)
    p.add_argument('--alpha_start', type=float, default=1.0,
                   help='Initial explicit context weight (1.0 = pure explicit)')
    p.add_argument('--ctx_fade', type=int, default=2000,
                   help='Steps over which alpha decays from alpha_start to 0')
    p.add_argument('--eval_alpha', type=float, default=0.0,
                   help='Alpha used during evaluation (0.0 = pure integrator). '
                        'Set to 0 so Stage 3 is only passed when integrator alone works.')
    p.add_argument('--lambda_ctx', type=float, default=0.5)
    p.add_argument('--cross_fa_mult', type=float, default=3.0)
    p.add_argument('--max_steps', type=int, default=4000)
    p.add_argument('--eval_every', type=int, default=50)
    p.add_argument('--n_eval_sessions', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def evaluate(rnn, stage, gamma, alpha=0.0, n_sessions=5, device='cpu', rng=None):
    if rng is None:
        rng = np.random.default_rng()
    add_noise = (stage >= 3)
    rnn_fn = make_ctx_teacher_rnn_fn(rnn, gamma=gamma, alpha=alpha,
                                     add_noise=add_noise, device=device)
    return [generate_session_coarse(rnn_fn, stage=stage, rng=rng) for _ in range(n_sessions)]


def train_step(rnn, optimizer, stage, fa_loss_weight, gamma, alpha,
               lambda_ctx=0.5, cross_fa_mult=3.0,
               batch_size=4, device='cpu', rng=None,
               bptt_trials=15, max_bptt_windows=8, train_n_blocks=2, max_sr=1.2):
    if rng is None:
        rng = np.random.default_rng()

    rnn_fn = make_ctx_teacher_rnn_fn(rnn, gamma=gamma, alpha=alpha,
                                     add_noise=True, device=device)
    batch_loss = torch.tensor(0.0)
    optimizer.zero_grad()

    for _ in range(batch_size):
        sess = generate_session_coarse(rnn_fn, stage=stage, rng=rng,
                                       n_blocks_override=train_n_blocks)
        n = len(sess['inputs'])
        session_data = [
            (sess['inputs'][ti], sess['targets'][ti], sess['masks'][ti],
             sess['licks'][ti], sess['rewards'][ti], sess['z_seqs'][ti])
            for ti in range(n)
        ]
        trial_meta = [
            (sess['rewarded'][ti], sess['stimulus'][ti], sess['instruction'][ti])
            for ti in range(n)
        ]
        miss_w = 3.0 if stage >= 2 else 1.0
        loss = compute_loss_ctx(
            rnn, session_data, trial_meta, stage, fa_loss_weight,
            miss_loss_weight=miss_w, device=device,
            lambda_ctx=lambda_ctx, cross_fa_mult=cross_fa_mult,
            bptt_window=bptt_trials, max_bptt_windows=max_bptt_windows, rng=rng,
        )
        batch_loss = batch_loss + loss / batch_size

    gc.collect()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
    optimizer.step()

    with torch.no_grad():
        try:
            op = float(torch.linalg.matrix_norm(rnn.W_rec, ord=2))
        except Exception:
            op = float(rnn.W_rec.norm('fro')) / (rnn.W_rec.shape[0] ** 0.5)
        if op > max_sr:
            rnn.W_rec.data.mul_(max_sr / op)

    return float(batch_loss)


def main():
    cfg = get_config()
    device = cfg.device
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    outdir = os.path.join(cfg.outdir, f'seed_{cfg.seed:02d}')
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, 'log.csv')
    fields = ['step', 'stage', 'loss', 'dprime_intra', 'dprime_inter',
              'n_pass_blocks', 'advanced', 'regressed', 'alpha', 'lambda_ctx']

    rnn = IntegratorRNNCoarse(n_units=cfg.n_units).to(device)

    ckpt = cfg.stage2_ckpt
    if os.path.exists(ckpt):
        rnn.load_from_vanilla(torch.load(ckpt, map_location=device))
        print(f"Loaded Stage 2 weights from {ckpt}")
        curriculum = CurriculumManager()
        curriculum.stage = 3
        total_steps = 300
    else:
        print(f"WARNING: {ckpt} not found — training from scratch")
        curriculum = CurriculumManager()
        optimizer0 = torch.optim.Adam(rnn.parameters(), lr=1e-3)
        loss0 = train_stage0_coarse(rnn, optimizer0, n_steps=100, device=device, rng=rng)
        curriculum.stage = 1
        total_steps = 100
        print(f"Stage 0 done. Loss: {loss0:.4f}")

    with open(log_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print(f"\n=== Context Teacher Forcing | alpha {cfg.alpha_start}→0 over {cfg.ctx_fade} steps | "
          f"gamma={cfg.gamma} | lambda_ctx={cfg.lambda_ctx} | batch={cfg.batch_size} ===")
    print(f"    Explicit ctx fades linearly: step 300 alpha={cfg.alpha_start:.2f} → "
          f"step {300 + cfg.ctx_fade} alpha=0.00 (pure integrator)")

    while curriculum.stage <= 4 and total_steps < cfg.max_steps * 5:
        stage = curriculum.stage
        lr = 1e-4 if stage == 4 else 5e-4
        optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
        stage_steps = 0

        if stage == 3:
            print(f"\nStage 3: initializing context attractor...")
            initialize_context_attractor(rnn, alpha=rnn.alpha)
            stage_bptt   = 180
            stage_bptt_w = 1
            stage_max_sr = 1.2
            lmb_ctx = cfg.lambda_ctx
        else:
            stage_bptt   = 15
            stage_bptt_w = 8
            stage_max_sr = 2.0
            lmb_ctx = 0.0

        print(f"Stage {stage}: lr={lr}, bptt={stage_bptt}, batch={cfg.batch_size}, lambda_ctx={lmb_ctx}")

        while stage_steps < cfg.max_steps:
            # Alpha schedule: fade from alpha_start to 0 over ctx_fade steps (Stage 3 only)
            if stage == 3:
                frac = min(stage_steps / max(cfg.ctx_fade, 1), 1.0)
                alpha_t = cfg.alpha_start * (1.0 - frac)
            else:
                alpha_t = 0.0  # pure integrator for Stage 4

            loss = train_step(
                rnn, optimizer, stage, curriculum.fa_loss_weight, cfg.gamma, alpha_t,
                lambda_ctx=lmb_ctx, cross_fa_mult=cfg.cross_fa_mult,
                batch_size=cfg.batch_size, device=device, rng=rng,
                bptt_trials=stage_bptt, max_bptt_windows=stage_bptt_w,
                train_n_blocks=2, max_sr=stage_max_sr,
            )
            total_steps += 1
            stage_steps += 1

            if stage_steps % cfg.eval_every == 0:
                # Always evaluate with eval_alpha (default 0 = pure integrator)
                # so Stage 3 advancement requires the integrator to work alone
                sessions = evaluate(rnn, stage, cfg.gamma, alpha=cfg.eval_alpha,
                                    n_sessions=cfg.n_eval_sessions,
                                    device=device, rng=rng)
                metrics, advanced, regressed = curriculum.evaluate(sessions, total_steps)

                dp_intra = metrics.get('dprime_intra', metrics.get('dprime', 0.0))
                dp_inter = metrics.get('dprime_inter', 0.0)
                n_pass   = metrics.get('n_pass_blocks', 0)

                tag = 'ADVANCED' if advanced else ('REGRESSED' if regressed else '')
                print(f"  step {total_steps:5d} | stage {stage} | loss {loss:.4f} | "
                      f"d'_intra {dp_intra:.2f} | d'_inter {dp_inter:.2f} | "
                      f"pass {n_pass} | train_alpha={alpha_t:.3f} eval_alpha={cfg.eval_alpha:.1f} {tag}")

                with open(log_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fields).writerow({
                        'step': total_steps, 'stage': stage,
                        'loss': f'{loss:.6f}',
                        'dprime_intra': f'{dp_intra:.4f}',
                        'dprime_inter': f'{dp_inter:.4f}',
                        'n_pass_blocks': n_pass,
                        'advanced': int(advanced),
                        'regressed': int(regressed),
                        'alpha': f'{alpha_t:.4f}',
                        'lambda_ctx': lmb_ctx,
                    })

                if advanced:
                    torch.save(rnn.state_dict(),
                               os.path.join(outdir, f'rnn_stage{stage}.pt'))
                    print(f"  -> Stage {curriculum.stage}")
                    if stage == 4:
                        torch.save(rnn.state_dict(),
                                   os.path.join(outdir, 'rnn_final.pt'))
                        print("  -> DONE!")
                        return
                    break
                if regressed:
                    break
        else:
            print(f"  WARNING: Stage {stage} hit max_steps. Advancing.")
            curriculum.stage = min(stage + 1, 4)
            torch.save(rnn.state_dict(), os.path.join(outdir, f'rnn_stage{stage}.pt'))

    torch.save(rnn.state_dict(), os.path.join(outdir, 'rnn_final.pt'))
    print(f"Training complete. Final stage: {curriculum.stage}")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"Total time: {(time.time()-t0)/60:.1f} min")
