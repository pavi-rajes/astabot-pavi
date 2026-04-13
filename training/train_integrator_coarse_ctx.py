"""
Train IntegratorRNNCoarse with three complementary context-learning improvements:

1. Auxiliary context readout loss (lambda_ctx=0.2):
   - w_ctx head on hidden state predicts VIS1-block(1) vs AUD1-block(0)
   - Forces hidden state to represent context even when behavioral gradient is
     satisfied by indiscriminate licking
   - Applied to all non-instruction trials

2. Cross-modal FA penalty boost (cross_fa_mult=3.0):
   - FA weight multiplied by 3x when licked stimulus is cross-modal
     (AUD stim in VIS-block, or VIS stim in AUD-block)
   - Breaks the indiscriminate licking equilibrium explicitly

3. 2-block single BPTT window (stage_bptt=180, stage_bptt_w=1):
   - Gradient spans the block boundary
   - Needed for ctx loss to teach context *switching*, not just context *holding*

Usage:
    conda run -n latent_circuit python training/train_integrator_coarse_ctx.py \
        --outdir checkpoints_integrator_coarse_ctx --gamma 0.90 --seed 42
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

VIS_STIMS = {VIS1, VIS2}
AUD_STIMS = {AUD1, AUD2}


def get_config():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', default='checkpoints_integrator_coarse_ctx')
    p.add_argument('--stage2_ckpt', default='checkpoints_v24/seed_42/rnn_stage2.pt')
    p.add_argument('--n_units', type=int, default=100)
    p.add_argument('--gamma', type=float, default=0.90)
    p.add_argument('--lambda_ctx', type=float, default=0.2)
    p.add_argument('--cross_fa_mult', type=float, default=3.0)
    p.add_argument('--max_steps', type=int, default=2000)
    p.add_argument('--eval_every', type=int, default=50)
    p.add_argument('--n_eval_sessions', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def compute_loss_ctx(rnn, session_data, trial_meta, stage, fa_loss_weight,
                     miss_loss_weight=1.0, lambda_rate=1e-3, lambda_weight=1e-4,
                     lambda_ctx=0.2, cross_fa_mult=3.0,
                     device='cpu', bptt_window=180, max_bptt_windows=1, rng=None):
    """
    Loss function with auxiliary context head + cross-modal FA boost.

    trial_meta: list of (rewarded, stimulus, is_instruction) per trial
    """
    if rng is None:
        rng = np.random.default_rng()
    gc.collect()

    bce_fn = nn.BCELoss(reduction='none')
    total_bce = torch.tensor(0.0, device=device)
    total_ctx = torch.tensor(0.0, device=device)
    total_rate = torch.tensor(0.0, device=device)
    total_tokens = 0
    total_ctx_trials = 0

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
        window_is_cross_fa, window_ctx_label, window_is_instr = [], [], []

        for ti in range(w_start, w_end):
            u_np, tgt_np, mask_np, lick, reward, _ = session_data[ti]
            rewarded_t, stim_t, is_instr_t = trial_meta[ti]

            window_u.append(torch.from_numpy(u_np).float().to(device))
            window_target.append(torch.from_numpy(tgt_np).float().to(device))
            window_mask.append(torch.from_numpy(mask_np.astype(np.float32)).to(device))
            window_is_fa.append(lick and (not reward))
            window_is_miss.append((not lick) and (tgt_np.max() > 0.5))
            window_is_instr.append(is_instr_t)
            window_ctx_label.append(1.0 if rewarded_t == VIS1 else 0.0)

            is_cross = (stim_t is not None and (
                (stim_t in AUD_STIMS and rewarded_t == VIS1) or
                (stim_t in VIS_STIMS and rewarded_t == AUD1)
            ))
            window_is_cross_fa.append(is_cross)

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

                # Cross-modal FA gets boosted penalty
                if window_is_fa[wi] and window_is_cross_fa[wi]:
                    weight = fa_loss_weight * cross_fa_mult
                elif window_is_fa[wi]:
                    weight = fa_loss_weight
                elif window_is_miss[wi]:
                    weight = miss_loss_weight
                else:
                    weight = 1.0

                total_bce = total_bce + weight * bce_vals.sum()
                total_tokens += z_resp.numel()

            total_rate = total_rate + y_seq_trial.pow(2).mean()

            # Auxiliary context loss on non-instruction trials
            if not window_is_instr[wi] and resp_mask.any():
                h_resp = y_seq_trial[resp_mask]  # (T_resp, N)
                h_mean = h_resp.mean(dim=0)       # (N,)
                ctx_pred = torch.sigmoid(
                    h_mean @ rnn.w_ctx.T + rnn.b_ctx
                ).squeeze()
                ctx_label = torch.tensor(
                    [window_ctx_label[wi]], dtype=torch.float32, device=device
                )
                ctx_pred_c = ctx_pred.unsqueeze(0).clamp(1e-6, 1 - 1e-6)
                total_ctx = total_ctx + bce_fn(ctx_pred_c, ctx_label).sum()
                total_ctx_trials += 1

    if total_tokens == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    n_sel = sum(
        min(all_window_starts[w] + bptt_window, n_trials) - all_window_starts[w]
        for w in selected_windows
    )
    ctx_term = lambda_ctx * total_ctx / max(total_ctx_trials, 1) if total_ctx_trials > 0 else 0.0
    return (total_bce / total_tokens
            + ctx_term
            + lambda_rate * total_rate / max(n_sel, 1)
            + lambda_weight * (rnn.W_rec.pow(2).sum() + rnn.W_in.pow(2).sum()))


def evaluate(rnn, stage, gamma, n_sessions=5, device='cpu', rng=None):
    if rng is None:
        rng = np.random.default_rng()
    add_noise = (stage >= 3)
    rnn_fn = make_integrator_coarse_rnn_fn(rnn, gamma=gamma, add_noise=add_noise, device=device)
    return [generate_session_coarse(rnn_fn, stage=stage, rng=rng) for _ in range(n_sessions)]


def train_step(rnn, optimizer, stage, fa_loss_weight, gamma,
               lambda_ctx=0.2, cross_fa_mult=3.0,
               batch_size=1, device='cpu', rng=None,
               bptt_trials=15, max_bptt_windows=8, train_n_blocks=2, max_sr=2.0):
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

    print(f"\n=== IntegratorRNNCoarse+CTX | gamma={cfg.gamma} | lambda_ctx={cfg.lambda_ctx} | seed={cfg.seed} ===")
    print(f"    Context retention per trial (RNN): {0.98**25:.3f}")
    print(f"    Integrator c after 5 instr trials: {1 - cfg.gamma**5:.3f}")
    print(f"    Cross-modal FA multiplier: {cfg.cross_fa_mult}x")
    print(f"    BPTT: 180 trials x 25 steps (single 2-block window)")

    while curriculum.stage <= 4 and total_steps < cfg.max_steps * 5:
        stage = curriculum.stage
        lr = 1e-4 if stage == 4 else 1e-3
        optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
        stage_steps = 0

        if stage == 3:
            print(f"\nStage 3: initializing context attractor...")
            initialize_context_attractor(rnn, alpha=rnn.alpha)
            stage_bptt   = 180  # full 2-block session (180 trials × 25 steps = 4500 steps)
            stage_bptt_w = 1    # single window — gradient spans both blocks
            stage_max_sr = 1.0
            lmb_ctx = cfg.lambda_ctx
        else:
            stage_bptt   = 15
            stage_bptt_w = 8
            stage_max_sr = 2.0
            lmb_ctx = 0.0  # context loss only for Stage 3

        print(f"Stage {stage}: lr={lr}, bptt={stage_bptt} trials x 25 steps, lambda_ctx={lmb_ctx}")

        while stage_steps < cfg.max_steps:
            loss = train_step(
                rnn, optimizer, stage, curriculum.fa_loss_weight, cfg.gamma,
                lambda_ctx=lmb_ctx, cross_fa_mult=cfg.cross_fa_mult,
                batch_size=cfg.batch_size, device=device, rng=rng,
                bptt_trials=stage_bptt, max_bptt_windows=stage_bptt_w,
                train_n_blocks=2, max_sr=stage_max_sr,
            )
            total_steps += 1
            stage_steps += 1

            if stage_steps % cfg.eval_every == 0:
                sessions = evaluate(rnn, stage, cfg.gamma,
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
                        'gamma': cfg.gamma,
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
