"""
Train DynamicRoutingRNNCoarse (dt=100ms, tau=500ms) through Stage 3.

Key advantage: 0.98^25 = 60% context per trial (vs 0.98^125 = 0.8% at dt=20ms).
Training is 5x faster (25 steps/trial vs 125).

Usage:
    conda run -n latent_circuit python training/train_coarse.py \
        --outdir checkpoints_coarse --seed 42
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

from models.rnn_coarse import DynamicRoutingRNNCoarse, make_rnn_fn_coarse
from tasks.dynamic_routing_coarse import VIS1, VIS2, AUD1, AUD2, N_INPUTS
from tasks.session_coarse import generate_session_coarse
from tasks.curriculum import CurriculumManager
from training.train_rnn import initialize_context_attractor, get_context_direction, make_hinted_rnn_fn


def get_config():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', default='checkpoints_coarse')
    p.add_argument('--stage2_ckpt', default='checkpoints_v24/seed_42/rnn_stage2.pt')
    p.add_argument('--n_units', type=int, default=100)
    p.add_argument('--max_steps', type=int, default=2000)
    p.add_argument('--eval_every', type=int, default=50)
    p.add_argument('--n_eval_sessions', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def compute_loss_coarse(rnn, session_data, stage, fa_loss_weight,
                        miss_loss_weight=1.0, lambda_rate=1e-3, lambda_weight=1e-4,
                        device='cpu', bptt_window=15, max_bptt_windows=0, rng=None):
    """compute_loss adapted for coarse-dt sessions (same logic, different step counts)."""
    if rng is None:
        rng = np.random.default_rng()
    gc.collect()

    bce_fn = nn.BCELoss(reduction='none')
    total_bce = torch.tensor(0.0, device=device)
    total_rate = torch.tensor(0.0, device=device)
    total_tokens = 0

    n_trials = len(session_data)
    all_window_starts = list(range(0, n_trials, bptt_window))

    # First pass: carry hidden state (no grad)
    y_states = []
    y = torch.zeros(rnn.N, device=device)
    with torch.no_grad():
        for w_start in all_window_starts:
            y_states.append(y.clone())
            w_end = min(w_start + bptt_window, n_trials)
            for ti in range(w_start, w_end):
                u_np = session_data[ti][0]
                u_trial = torch.from_numpy(u_np).float().to(device)
                wu = u_trial @ rnn.W_in.T + rnn.b_rec
                y_t = y.unsqueeze(0)
                for t in range(u_trial.shape[0]):
                    pre = y_t @ rnn.W_rec.T + wu[t]
                    y_t = (1 - rnn.alpha) * y_t + rnn.alpha * torch.relu(pre)
                y = y_t.squeeze(0)

    selected_windows = list(range(len(all_window_starts)))
    if max_bptt_windows > 0 and len(selected_windows) > max_bptt_windows:
        selected_windows = sorted(
            rng.choice(selected_windows, size=max_bptt_windows, replace=False).tolist()
        )

    for w_idx in selected_windows:
        w_start = all_window_starts[w_idx]
        w_end = min(w_start + bptt_window, n_trials)
        y_w = y_states[w_idx].detach()

        window_u, window_target, window_mask = [], [], []
        window_is_fa, window_is_miss = [], []
        for ti in range(w_start, w_end):
            u_np, tgt_np, mask_np, lick, reward, _ = session_data[ti]
            window_u.append(torch.from_numpy(u_np).float().to(device))
            window_target.append(torch.from_numpy(tgt_np).float().to(device))
            window_mask.append(torch.from_numpy(mask_np.astype(np.float32)).to(device))
            window_is_fa.append(lick and (not reward))
            window_is_miss.append((not lick) and (tgt_np.max() > 0.5))

        u_window = torch.cat(window_u, dim=0)
        wu_window = u_window @ rnn.W_in.T + rnn.b_rec
        T_per = window_u[0].shape[0]

        y_t = y_w.unsqueeze(0)
        y_seq_all = []
        for t in range(u_window.shape[0]):
            pre = y_t @ rnn.W_rec.T + wu_window[t]
            y_t = (1 - rnn.alpha) * y_t + rnn.alpha * torch.relu(pre)
            y_seq_all.append(y_t)

        y_seq_cat = torch.cat(y_seq_all, dim=0)
        del y_seq_all
        z_seq_cat = torch.sigmoid(y_seq_cat @ rnn.w_out.T + rnn.b_out).squeeze(-1)

        for wi in range(len(window_u)):
            t_start_w = wi * T_per
            t_end_w   = t_start_w + T_per
            z_pred = z_seq_cat[t_start_w:t_end_w]
            y_seq_trial = y_seq_cat[t_start_w:t_end_w]

            resp_mask = window_mask[wi].bool()
            z_resp = z_pred[resp_mask]
            t_resp = window_target[wi][resp_mask]

            if z_resp.numel() > 0:
                z_resp_c = z_resp.clamp(1e-6, 1 - 1e-6)
                bce_vals = bce_fn(z_resp_c, t_resp)
                weight = (fa_loss_weight if window_is_fa[wi]
                          else miss_loss_weight if window_is_miss[wi] else 1.0)
                total_bce = total_bce + weight * bce_vals.sum()
                total_tokens += z_resp.numel()

            total_rate = total_rate + y_seq_trial.pow(2).mean()

    if total_tokens == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    n_sel = sum(
        min(all_window_starts[w] + bptt_window, n_trials) - all_window_starts[w]
        for w in selected_windows
    )
    return (total_bce / total_tokens
            + lambda_rate * total_rate / max(n_sel, 1)
            + lambda_weight * (rnn.W_rec.pow(2).sum() + rnn.W_in.pow(2).sum()))


def train_stage0_coarse(rnn, optimizer, n_steps=100, device='cpu', rng=None):
    """Stage 0: output bias shaping using coarse-dt trial structure."""
    from tasks.dynamic_routing_coarse import STEPS_PER_TRIAL, RESP_START, RESP_END, VIS1
    if rng is None:
        rng = np.random.default_rng()
    for name, p in rnn.named_parameters():
        p.requires_grad_(name in ('w_out', 'b_out'))

    bce_fn = nn.BCELoss()
    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        u = torch.zeros(STEPS_PER_TRIAL, N_INPUTS, device=device)
        u[15:20, VIS1] = 1.0        # stimulus
        u[16:25, 6] = 1.0           # trial phase
        u[24:25, 4] = 1.0           # auto-reward at end
        target = torch.ones(STEPS_PER_TRIAL, device=device)
        y0 = torch.zeros(rnn.N, device=device)
        y_seq, z_seq, _ = rnn.forward(y0, u.unsqueeze(0), add_noise=True)
        z_resp = z_seq.squeeze()[RESP_START:RESP_END]
        t_resp = target[RESP_START:RESP_END]
        loss = bce_fn(z_resp, t_resp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss))

    for p in rnn.parameters():
        p.requires_grad_(True)
    return np.mean(losses[-50:]) if losses else 0.0


def evaluate_coarse(rnn, stage, n_sessions=5, device='cpu', rng=None):
    if rng is None:
        rng = np.random.default_rng()
    add_noise = (stage >= 3)
    rnn_fn = make_rnn_fn_coarse(rnn, add_noise=add_noise, device=device)
    return [generate_session_coarse(rnn_fn, stage=stage, rng=rng)
            for _ in range(n_sessions)]


def train_step_coarse(rnn, optimizer, stage, fa_loss_weight,
                      batch_size=1, device='cpu', rng=None,
                      bptt_trials=15, max_bptt_windows=8, train_n_blocks=2,
                      max_sr=2.0, rnn_fn_override=None):
    if rng is None:
        rng = np.random.default_rng()
    rnn_fn = rnn_fn_override or make_rnn_fn_coarse(rnn, add_noise=True, device=device)
    batch_loss = torch.tensor(0.0)
    optimizer.zero_grad()

    for _ in range(batch_size):
        sess = generate_session_coarse(rnn_fn, stage=stage, rng=rng,
                                       n_blocks_override=train_n_blocks)
        session_data = [
            (sess['inputs'][ti], sess['targets'][ti], sess['masks'][ti],
             sess['licks'][ti], sess['rewards'][ti], sess['z_seqs'][ti])
            for ti in range(len(sess['inputs']))
        ]
        miss_w = 3.0 if stage >= 2 else 1.0
        loss = compute_loss_coarse(rnn, session_data, stage, fa_loss_weight,
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
    fields = ['step', 'stage', 'loss', 'dprime_intra', 'dprime_inter',
              'n_pass_blocks', 'advanced', 'regressed']

    rnn = DynamicRoutingRNNCoarse(n_units=cfg.n_units).to(device)

    # Try loading Stage 2 weights; if unavailable, train from scratch
    ckpt = cfg.stage2_ckpt
    if os.path.exists(ckpt):
        rnn.load_from_vanilla(torch.load(ckpt, map_location=device))
        print(f"Loaded Stage 2 weights from {ckpt}")
        curriculum = CurriculumManager()
        curriculum.stage = 3
        total_steps = 300
        print("Starting at Stage 3")
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

    print(f"\n=== CoarseRNN (dt=100ms, tau=500ms) | seed={cfg.seed} ===")
    print(f"    Context retention per trial: 0.98^25 = {0.98**25:.3f}")

    while curriculum.stage <= 4 and total_steps < cfg.max_steps * 5:
        stage = curriculum.stage
        lr = 1e-4 if stage == 4 else 1e-3
        optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
        stage_steps = 0

        hint_strength_ref = [0.0]
        v_context = None

        if stage == 3:
            print(f"\nStage 3: initializing context attractor...")
            initialize_context_attractor(rnn, alpha=rnn.alpha)
            # Also enable teacher forcing with long fade — helps bootstrap
            # the gate learning even with better context retention
            HINT_FADE = 500   # fewer steps needed since context already persists
            v_context = get_context_direction(rnn).to(device)
            hint_strength_ref = [1.0]
            print(f"  Teacher forcing: hint fades over {HINT_FADE} steps")
            stage_bptt   = 90   # full block (90 × 25 = 2250 steps, vs 11250 before — 5× faster)
            stage_bptt_w = 2    # one window per block
            stage_max_sr = 1.0
        else:
            stage_bptt   = 15
            stage_bptt_w = 8
            stage_max_sr = 2.0

        print(f"Stage {stage}: lr={lr}, bptt={stage_bptt} trials × 25 steps")

        while stage_steps < cfg.max_steps:
            rnn_fn_for_step = None
            if stage == 3 and v_context is not None:
                hint_strength_ref[0] = max(0.0, 1.0 - stage_steps / HINT_FADE)
                base_fn = make_rnn_fn_coarse(rnn, add_noise=True, device=device)
                rnn_fn_for_step = make_hinted_rnn_fn(base_fn, v_context, hint_strength_ref)

            loss = train_step_coarse(
                rnn, optimizer, stage, curriculum.fa_loss_weight,
                batch_size=cfg.batch_size, device=device, rng=rng,
                bptt_trials=stage_bptt, max_bptt_windows=stage_bptt_w,
                train_n_blocks=2, max_sr=stage_max_sr,
                rnn_fn_override=rnn_fn_for_step,
            )
            total_steps += 1
            stage_steps += 1

            if stage_steps % cfg.eval_every == 0:
                sessions = evaluate_coarse(rnn, stage, n_sessions=cfg.n_eval_sessions,
                                           device=device, rng=rng)
                metrics, advanced, regressed = curriculum.evaluate(sessions, total_steps)

                dp_intra = metrics.get('dprime_intra', metrics.get('dprime', 0.0))
                dp_inter = metrics.get('dprime_inter', 0.0)
                n_pass   = metrics.get('n_pass_blocks', 0)
                hint     = hint_strength_ref[0] if stage == 3 else 0.0

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
