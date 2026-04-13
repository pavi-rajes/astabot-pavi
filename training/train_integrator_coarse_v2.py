"""
IntegratorRNNCoarse v2 — fixes the gradient variance problem.

Changes vs train_integrator_coarse_ctx.py:
1. batch_size=4 (default): 4× variance reduction
2. gamma curriculum: start at gamma_start (default 0.50) for first gamma_warmup steps,
   then anneal linearly to gamma (default 0.90). Stronger early context signal.
3. lambda_ctx=0.5 (default, up from 0.2): stronger context supervision loss
4. Stage 3 LR=5e-4 (down from 1e-3): more stable optimization
5. max_sr=1.2 for Stage 3 (up from 1.0): allows slight SR > 1 in context direction

Motivation: The v1 run (integrator_coarse_ctx) oscillated with no upward trend over 2000 steps
(mean d'_inter=0.055, std=0.097). Root cause: gradient variance too high (batch=1) + weak ctx
supervision. The network briefly learned context at step 1500 but couldn't hold it.

Usage:
    conda run -n latent_circuit python training/train_integrator_coarse_v2.py \\
        --outdir checkpoints_integrator_coarse_v2 --seed 42
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

from models.rnn_integrator_coarse import IntegratorRNNCoarse, make_integrator_coarse_rnn_fn
from tasks.session_coarse import generate_session_coarse
from tasks.dynamic_routing_coarse import VIS1, VIS2, AUD1, AUD2
from tasks.curriculum import CurriculumManager
from training.train_rnn import initialize_context_attractor
from training.train_coarse import train_stage0_coarse
from training.train_integrator_coarse_ctx import compute_loss_ctx

VIS_STIMS = {VIS1, VIS2}
AUD_STIMS = {AUD1, AUD2}


def get_config():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', default='checkpoints_integrator_coarse_v2')
    p.add_argument('--stage2_ckpt', default='checkpoints_v24/seed_42/rnn_stage2.pt')
    p.add_argument('--n_units', type=int, default=100)
    p.add_argument('--gamma', type=float, default=0.90,
                   help='Final gamma (end of curriculum)')
    p.add_argument('--gamma_start', type=float, default=0.50,
                   help='Starting gamma (strong early signal; c≈0.97 after 5 trials)')
    p.add_argument('--gamma_warmup', type=int, default=500,
                   help='Steps to linearly anneal from gamma_start to gamma')
    p.add_argument('--lambda_ctx', type=float, default=0.5)
    p.add_argument('--cross_fa_mult', type=float, default=3.0)
    p.add_argument('--max_steps', type=int, default=3000)
    p.add_argument('--eval_every', type=int, default=50)
    p.add_argument('--n_eval_sessions', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def evaluate(rnn, stage, gamma, n_sessions=5, device='cpu', rng=None):
    if rng is None:
        rng = np.random.default_rng()
    add_noise = (stage >= 3)
    rnn_fn = make_integrator_coarse_rnn_fn(rnn, gamma=gamma, add_noise=add_noise, device=device)
    return [generate_session_coarse(rnn_fn, stage=stage, rng=rng) for _ in range(n_sessions)]


def train_step(rnn, optimizer, stage, fa_loss_weight, gamma,
               lambda_ctx=0.5, cross_fa_mult=3.0,
               batch_size=4, device='cpu', rng=None,
               bptt_trials=15, max_bptt_windows=8, train_n_blocks=2, max_sr=1.2):
    if rng is None:
        rng = np.random.default_rng()

    rnn_fn = make_integrator_coarse_rnn_fn(rnn, gamma=gamma, add_noise=True, device=device)
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
        # Use operator norm (largest singular value) as SR proxy — stable via SVD.
        # Conservative: operator norm >= spectral radius, so this clips at least as much.
        try:
            op = float(torch.linalg.matrix_norm(rnn.W_rec, ord=2))
        except Exception:
            # Fallback: Frobenius norm / sqrt(N) is a looser upper bound
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
              'n_pass_blocks', 'advanced', 'regressed', 'gamma', 'lambda_ctx']

    rnn = IntegratorRNNCoarse(n_units=cfg.n_units).to(device)

    ckpt = cfg.stage2_ckpt
    if os.path.exists(ckpt):
        rnn.load_from_vanilla(torch.load(ckpt, map_location=device))
        print(f"Loaded Stage 2 weights from {ckpt}")
        print(f"W_in[:,7] mean abs: {rnn.W_in_raw[:,7].abs().mean().item():.4f}  (nonzero = good)")
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

    print(f"\n=== IntegratorRNNCoarse v2 | gamma {cfg.gamma_start}→{cfg.gamma} over {cfg.gamma_warmup} steps | "
          f"lambda_ctx={cfg.lambda_ctx} | batch={cfg.batch_size} | seed={cfg.seed} ===")
    print(f"    c after 5 instr trials at gamma_start={cfg.gamma_start}: {1 - cfg.gamma_start**5:.3f}")
    print(f"    c after 5 instr trials at gamma={cfg.gamma}: {1 - cfg.gamma**5:.3f}")

    while curriculum.stage <= 4 and total_steps < cfg.max_steps * 5:
        stage = curriculum.stage
        lr = 1e-4 if stage == 4 else 5e-4  # lower LR for stability
        optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
        stage_steps = 0

        if stage == 3:
            print(f"\nStage 3: initializing context attractor...")
            initialize_context_attractor(rnn, alpha=rnn.alpha)
            stage_bptt   = 180
            stage_bptt_w = 1
            stage_max_sr = 1.2  # allow slight SR > 1 for attractor formation
            lmb_ctx = cfg.lambda_ctx
        else:
            stage_bptt   = 15
            stage_bptt_w = 8
            stage_max_sr = 2.0
            lmb_ctx = 0.0

        print(f"Stage {stage}: lr={lr}, bptt={stage_bptt} trials, lambda_ctx={lmb_ctx}, "
              f"batch={cfg.batch_size}, max_sr={stage_max_sr}")

        while stage_steps < cfg.max_steps:
            # Gamma curriculum: linearly anneal from gamma_start to gamma
            if stage == 3:
                frac = min(stage_steps / max(cfg.gamma_warmup, 1), 1.0)
                gamma_t = cfg.gamma_start + frac * (cfg.gamma - cfg.gamma_start)
            else:
                gamma_t = cfg.gamma

            loss = train_step(
                rnn, optimizer, stage, curriculum.fa_loss_weight, gamma_t,
                lambda_ctx=lmb_ctx, cross_fa_mult=cfg.cross_fa_mult,
                batch_size=cfg.batch_size, device=device, rng=rng,
                bptt_trials=stage_bptt, max_bptt_windows=stage_bptt_w,
                train_n_blocks=2, max_sr=stage_max_sr,
            )
            total_steps += 1
            stage_steps += 1

            if stage_steps % cfg.eval_every == 0:
                # Evaluate at current (possibly mid-curriculum) gamma
                sessions = evaluate(rnn, stage, gamma_t,
                                    n_sessions=cfg.n_eval_sessions,
                                    device=device, rng=rng)
                metrics, advanced, regressed = curriculum.evaluate(sessions, total_steps)

                dp_intra = metrics.get('dprime_intra', metrics.get('dprime', 0.0))
                dp_inter = metrics.get('dprime_inter', 0.0)
                n_pass   = metrics.get('n_pass_blocks', 0)

                tag = 'ADVANCED' if advanced else ('REGRESSED' if regressed else '')
                print(f"  step {total_steps:5d} | stage {stage} | loss {loss:.4f} | "
                      f"d'_intra {dp_intra:.2f} | d'_inter {dp_inter:.2f} | "
                      f"pass {n_pass} | γ={gamma_t:.3f} {tag}")

                with open(log_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fields).writerow({
                        'step': total_steps, 'stage': stage,
                        'loss': f'{loss:.6f}',
                        'dprime_intra': f'{dp_intra:.4f}',
                        'dprime_inter': f'{dp_inter:.4f}',
                        'n_pass_blocks': n_pass,
                        'advanced': int(advanced),
                        'regressed': int(regressed),
                        'gamma': f'{gamma_t:.4f}',
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
