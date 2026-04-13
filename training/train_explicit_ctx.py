"""
Explicit context input ablation — ceiling benchmark.

Instead of learning context from reward feedback, injects ground-truth context
(c = +1.0 for VIS1-blocks, c = -1.0 for AUD1-blocks) directly as channel 7.

Purpose: verify that the behavioral loss is sufficient for Stage 3 advancement
when context is perfectly available. If this fails, there is a deeper problem
with the task/architecture that is independent of context inference.

If it succeeds: confirms the bottleneck is specifically the context *learning*
mechanism (integrator), not the downstream routing.

Uses IntegratorRNNCoarse (same architecture) but with a deterministic ctx_fn.
"""

import os
import sys
import csv
import gc
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rnn_integrator_coarse import IntegratorRNNCoarse
from tasks.session_coarse import generate_session_coarse
from tasks.dynamic_routing_coarse import VIS1, AUD1, VIS2, AUD2
from tasks.curriculum import CurriculumManager
from training.train_coarse import train_stage0_coarse
from training.train_integrator_coarse_ctx import compute_loss_ctx


def make_explicit_ctx_rnn_fn(rnn, add_noise=True, device='cpu', ctx_scale=1.0):
    """
    rnn_fn that uses ground-truth context as channel 7 input.
    c = +ctx_scale for VIS1-rewarded blocks, -ctx_scale for AUD1-rewarded.
    """
    def rnn_fn(h, stim, rewarded, instr, rng):
        if h is None:
            h = torch.zeros(rnn.N, device=device)

        # Ground-truth context signal
        c = ctx_scale if rewarded == VIS1 else -ctx_scale

        u, target, mask, lick, reward, z_seq, h_new = \
            rnn.forward_trial_closedloop(
                h, stim, rewarded, instr,
                add_noise=add_noise, device=device,
                integrator_value=c,
            )
        return u, target, mask, lick, reward, z_seq, h_new

    return rnn_fn


def get_config():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', default='checkpoints_explicit_ctx')
    p.add_argument('--stage2_ckpt', default='checkpoints_v24/seed_42/rnn_stage2.pt')
    p.add_argument('--n_units', type=int, default=100)
    p.add_argument('--ctx_scale', type=float, default=1.0,
                   help='Magnitude of injected context signal (1.0 = perfect)')
    p.add_argument('--lambda_ctx', type=float, default=0.5)
    p.add_argument('--cross_fa_mult', type=float, default=3.0)
    p.add_argument('--max_steps', type=int, default=3000)
    p.add_argument('--eval_every', type=int, default=50)
    p.add_argument('--n_eval_sessions', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def evaluate(rnn, stage, ctx_scale=1.0, n_sessions=5, device='cpu', rng=None):
    if rng is None:
        rng = np.random.default_rng()
    add_noise = (stage >= 3)
    rnn_fn = make_explicit_ctx_rnn_fn(rnn, add_noise=add_noise, device=device,
                                      ctx_scale=ctx_scale)
    return [generate_session_coarse(rnn_fn, stage=stage, rng=rng) for _ in range(n_sessions)]


def train_step(rnn, optimizer, stage, fa_loss_weight, ctx_scale=1.0,
               lambda_ctx=0.5, cross_fa_mult=3.0,
               batch_size=4, device='cpu', rng=None,
               bptt_trials=15, max_bptt_windows=8, train_n_blocks=2):
    if rng is None:
        rng = np.random.default_rng()

    rnn_fn = make_explicit_ctx_rnn_fn(rnn, add_noise=True, device=device, ctx_scale=ctx_scale)
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
              'n_pass_blocks', 'advanced', 'regressed', 'ctx_scale']

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

    print(f"\n=== EXPLICIT CONTEXT ABLATION | ctx_scale={cfg.ctx_scale} | "
          f"batch={cfg.batch_size} | lambda_ctx={cfg.lambda_ctx} | seed={cfg.seed} ===")
    print(f"    Channel 7 = ±{cfg.ctx_scale} (ground truth context — no learning required)")
    print(f"    This is the performance ceiling. If this fails, task/arch is broken.")

    while curriculum.stage <= 4 and total_steps < cfg.max_steps * 5:
        stage = curriculum.stage
        lr = 1e-4 if stage == 4 else 5e-4
        optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
        stage_steps = 0

        if stage == 3:
            stage_bptt   = 180
            stage_bptt_w = 1
            lmb_ctx = cfg.lambda_ctx
        else:
            stage_bptt   = 15
            stage_bptt_w = 8
            lmb_ctx = 0.0

        print(f"Stage {stage}: lr={lr}, bptt={stage_bptt} trials, batch={cfg.batch_size}")

        while stage_steps < cfg.max_steps:
            loss = train_step(
                rnn, optimizer, stage, curriculum.fa_loss_weight, cfg.ctx_scale,
                lambda_ctx=lmb_ctx, cross_fa_mult=cfg.cross_fa_mult,
                batch_size=cfg.batch_size, device=device, rng=rng,
                bptt_trials=stage_bptt, max_bptt_windows=stage_bptt_w,
                train_n_blocks=2,
            )
            total_steps += 1
            stage_steps += 1

            if stage_steps % cfg.eval_every == 0:
                sessions = evaluate(rnn, stage, cfg.ctx_scale,
                                    n_sessions=cfg.n_eval_sessions,
                                    device=device, rng=rng)
                metrics, advanced, regressed = curriculum.evaluate(sessions, total_steps)

                dp_intra = metrics.get('dprime_intra', metrics.get('dprime', 0.0))
                dp_inter = metrics.get('dprime_inter', 0.0)
                n_pass   = metrics.get('n_pass_blocks', 0)

                tag = 'ADVANCED' if advanced else ('REGRESSED' if regressed else '')
                print(f"  step {total_steps:5d} | stage {stage} | loss {loss:.4f} | "
                      f"d'_intra {dp_intra:.2f} | d'_inter {dp_inter:.2f} | "
                      f"pass {n_pass} {tag}")

                with open(log_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fields).writerow({
                        'step': total_steps, 'stage': stage,
                        'loss': f'{loss:.6f}',
                        'dprime_intra': f'{dp_intra:.4f}',
                        'dprime_inter': f'{dp_inter:.4f}',
                        'n_pass_blocks': n_pass,
                        'advanced': int(advanced),
                        'regressed': int(regressed),
                        'ctx_scale': cfg.ctx_scale,
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
