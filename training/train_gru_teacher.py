"""
Train GatedDynamicRoutingRNN with teacher forcing through Stage 3.

Combines the two approaches that each individually failed:
- GRU gating: capacity for indefinite context persistence (88% signal retention
  vs 0.8% vanilla), but gate never learns without a training signal
- Teacher forcing: provides the training signal (hint_strength 1→0 over N steps),
  but vanilla RNN can't form a stable attractor before the hint fades

Together: teacher forcing bootstraps context-dependent behavior → gate gets
gradient signal → gate learns to hold context → hint can fade without collapse.

Usage:
    conda run -n latent_circuit python training/train_gru_teacher.py \
        --outdir checkpoints_gru_teacher --hint_fade_steps 2000 --seed 42
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

from models.rnn_gru import GatedDynamicRoutingRNN, make_gated_rnn_fn
from tasks.curriculum import CurriculumManager
from tasks.session import generate_session
from training.train_rnn import (
    compute_loss, train_stage0, initialize_context_attractor,
    get_context_direction, make_hinted_rnn_fn,
)


def get_config():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', default='checkpoints_gru_teacher')
    p.add_argument('--stage2_ckpt', default='checkpoints_v24/seed_42/rnn_stage2.pt')
    p.add_argument('--n_units', type=int, default=100)
    p.add_argument('--hint_fade_steps', type=int, default=2000,
                   help='Steps over which hint_strength fades 1→0. '
                        'Longer fade gives gate more time to learn. Try 1000-3000.')
    p.add_argument('--max_steps', type=int, default=5000)
    p.add_argument('--eval_every', type=int, default=50)
    p.add_argument('--n_eval_sessions', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def evaluate(rnn, stage, n_sessions=5, device='cpu', rng=None):
    if rng is None:
        rng = np.random.default_rng()
    add_noise = (stage >= 3)
    rnn_fn = make_gated_rnn_fn(rnn, add_noise=add_noise, device=device)
    sessions = []
    for _ in range(n_sessions):
        sess = generate_session(rnn_fn, stage=stage, rng=rng)
        sessions.append(sess)
    return sessions


def train_step(rnn, optimizer, stage, fa_loss_weight,
               batch_size=1, device='cpu', rng=None,
               bptt_trials=15, max_bptt_windows=8, train_n_blocks=2,
               max_sr=2.0, rnn_fn_override=None):
    if rng is None:
        rng = np.random.default_rng()

    rnn_fn = rnn_fn_override or make_gated_rnn_fn(rnn, add_noise=True, device=device)
    batch_loss = torch.tensor(0.0)
    optimizer.zero_grad()

    for _ in range(batch_size):
        sess = generate_session(rnn_fn, stage=stage, rng=rng,
                                n_blocks_override=train_n_blocks)
        session_data = [
            (sess['inputs'][ti], sess['targets'][ti], sess['masks'][ti],
             sess['licks'][ti], sess['rewards'][ti], sess['z_seqs'][ti])
            for ti in range(len(sess['inputs']))
        ]
        miss_w = 3.0 if stage >= 2 else 1.0
        loss = compute_loss(rnn, session_data, stage, fa_loss_weight,
                            miss_loss_weight=miss_w, device=device,
                            bptt_window=bptt_trials,
                            max_bptt_windows=max_bptt_windows, rng=rng)
        batch_loss = batch_loss + loss / batch_size

    gc.collect()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
    optimizer.step()

    with torch.no_grad():
        if stage >= 3:
            eigs = torch.linalg.eigvals(rnn.W_rec)
            sr = float(eigs.abs().max())
            if sr > max_sr:
                rnn.W_rec.data.mul_(max_sr / sr)
        else:
            op = torch.linalg.norm(rnn.W_rec, ord=2)
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
    fields = ['step', 'stage', 'loss', 'dprime_intra', 'dprime_inter', 'n_pass_blocks',
              'advanced', 'regressed', 'hint_strength']

    rnn = GatedDynamicRoutingRNN(n_units=cfg.n_units).to(device)

    # Load Stage 2 weights if available
    ckpt = cfg.stage2_ckpt
    if os.path.exists(ckpt):
        vanilla_sd = torch.load(ckpt, map_location=device)
        rnn.load_from_vanilla(vanilla_sd)
        print(f"Loaded Stage 2 weights from {ckpt}")
        curriculum = CurriculumManager()
        curriculum.stage = 3
        total_steps = 300
        print(f"Starting directly at Stage 3 (hint_fade_steps={cfg.hint_fade_steps})")
    else:
        print(f"WARNING: {ckpt} not found — training from scratch")
        curriculum = CurriculumManager()
        optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)
        loss0 = train_stage0(rnn, optimizer, n_steps=100, device=device, rng=rng)
        curriculum.stage = 1
        total_steps = 100
        print(f"Stage 0 done. Loss: {loss0:.4f}")

    with open(log_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print(f"\n=== GRU + Teacher Forcing | hint_fade={cfg.hint_fade_steps} | seed={cfg.seed} ===")

    hint_strength_ref = [0.0]
    v_context = None

    while curriculum.stage <= 4 and total_steps < cfg.max_steps * 5:
        stage = curriculum.stage
        lr = 1e-4 if stage == 4 else 1e-3
        optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
        stage_steps = 0

        if stage == 3:
            print(f"\nStage 3: initializing context attractor + teacher forcing...")
            initialize_context_attractor(rnn, alpha=rnn.alpha)
            v_context = get_context_direction(rnn).to(device)
            hint_strength_ref = [1.0]
            stage_bptt   = 90
            stage_bptt_w = 2
            stage_max_sr = 1.0
            print(f"  Teacher forcing enabled: fades over {cfg.hint_fade_steps} steps")
        else:
            hint_strength_ref = [0.0]
            v_context = None
            stage_bptt   = 15
            stage_bptt_w = 8
            stage_max_sr = 2.0

        print(f"Stage {stage}: lr={lr}, bptt={stage_bptt}")

        while stage_steps < cfg.max_steps:
            rnn_fn_for_step = None
            if stage == 3 and v_context is not None:
                hint_strength_ref[0] = max(0.0, 1.0 - stage_steps / cfg.hint_fade_steps)
                base_fn = make_gated_rnn_fn(rnn, add_noise=True, device=device)
                rnn_fn_for_step = make_hinted_rnn_fn(base_fn, v_context, hint_strength_ref)

            loss = train_step(
                rnn, optimizer, stage, curriculum.fa_loss_weight,
                batch_size=cfg.batch_size, device=device, rng=rng,
                bptt_trials=stage_bptt, max_bptt_windows=stage_bptt_w,
                train_n_blocks=2, max_sr=stage_max_sr,
                rnn_fn_override=rnn_fn_for_step,
            )
            total_steps += 1
            stage_steps += 1

            if stage_steps % cfg.eval_every == 0:
                sessions = evaluate(rnn, stage, n_sessions=cfg.n_eval_sessions,
                                    device=device, rng=rng)
                metrics, advanced, regressed = curriculum.evaluate(sessions, total_steps)

                dp_intra = metrics.get('dprime_intra', metrics.get('dprime', 0.0))
                dp_inter = metrics.get('dprime_inter', 0.0)
                n_pass   = metrics.get('n_pass_blocks', 0)
                hint     = hint_strength_ref[0]

                tag = 'ADVANCED' if advanced else ('REGRESSED' if regressed else '')
                print(f"  step {total_steps:5d} | stage {stage} | loss {loss:.4f} | "
                      f"d'_intra {dp_intra:.2f} | d'_inter {dp_inter:.2f} | "
                      f"hint {hint:.2f} | pass {n_pass} {tag}")

                with open(log_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fields).writerow({
                        'step': total_steps, 'stage': stage,
                        'loss': f'{loss:.6f}',
                        'dprime_intra': f'{dp_intra:.4f}',
                        'dprime_inter': f'{dp_inter:.4f}',
                        'n_pass_blocks': n_pass,
                        'advanced': int(advanced),
                        'regressed': int(regressed),
                        'hint_strength': f'{hint:.4f}',
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
