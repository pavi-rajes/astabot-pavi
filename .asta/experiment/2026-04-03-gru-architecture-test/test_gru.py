"""
Step 3: GRU-style gating architecture test.

Tests whether the GatedDynamicRoutingRNN can:
1. Train through Stages 0-2 (sanity check)
2. Learn inter-block context (d'_inter > 0) in Stage 3 without teacher forcing
3. Maintain a signal for 125+ timesteps with learned gate values
"""

import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from models.rnn_gru import GatedDynamicRoutingRNN, make_gated_rnn_fn
from training.train_rnn import (
    train_stage0, compute_loss, evaluate as evaluate_vanilla,
    initialize_context_attractor, get_context_direction, make_hinted_rnn_fn
)
from tasks.curriculum import CurriculumManager
from tasks.session import generate_session

OUTDIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_V24 = os.path.join(OUTDIR, '..', '..', '..', 'checkpoints_v24', 'seed_42', 'rnn_stage2.pt')
DEVICE = 'cpu'
SEED = 42


def make_rnn_fn_gated(rnn, add_noise=True):
    return make_gated_rnn_fn(rnn, add_noise=add_noise, device=DEVICE)


def evaluate_gated(rnn, stage, n_sessions=3, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    add_noise_eval = (stage >= 3)
    rnn_fn = make_rnn_fn_gated(rnn, add_noise=add_noise_eval)
    sessions = []
    for _ in range(n_sessions):
        sess = generate_session(rnn_fn, stage=stage, rng=rng)
        sessions.append(sess)
    return sessions


def train_step_gated(rnn, optimizer, stage, fa_loss_weight, batch_size=1,
                     rng=None, bptt_trials=15, max_bptt_windows=8, train_n_blocks=2,
                     max_sr=2.0, rnn_fn_override=None):
    """One gradient step for GatedDynamicRoutingRNN (mirrors train_step from train_rnn.py)."""
    import gc
    if rng is None:
        rng = np.random.default_rng()

    rnn_fn = rnn_fn_override if rnn_fn_override is not None else make_rnn_fn_gated(rnn, add_noise=True)
    batch_loss = torch.tensor(0.0)
    optimizer.zero_grad()

    for _ in range(batch_size):
        sess = generate_session(rnn_fn, stage=stage, rng=rng, n_blocks_override=train_n_blocks)
        n_trials = len(sess['inputs'])
        session_data = [
            (sess['inputs'][ti], sess['targets'][ti], sess['masks'][ti],
             sess['licks'][ti], sess['rewards'][ti], sess['z_seqs'][ti])
            for ti in range(n_trials)
        ]
        loss = compute_loss(rnn, session_data, stage, fa_loss_weight,
                            miss_loss_weight=(3.0 if stage >= 2 else 1.0),
                            device=DEVICE, bptt_window=bptt_trials,
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


def gate_persistence_test(rnn):
    """
    Test: inject a signal at t=0, measure how much remains at t=125 (1 trial).
    Compare vanilla-equivalent gate (0.2) vs. learned gate for context units.
    Returns float: fraction of signal retained after 125 steps.
    """
    rng = np.random.default_rng(0)
    N = rnn.N
    # Use dominant eigenvector of W_rec as the "context signal"
    with torch.no_grad():
        eigs, vecs = torch.linalg.eig(rnn.W_rec)
        idx = int((eigs.real - 1e3 * eigs.imag.abs()).argmax())
        v = vecs[:, idx].real.float()
        v = v / (v.norm() + 1e-8)

    # Inject signal, run 125 steps with no input (u=0)
    y = v.clone()
    y0_norm = float(y.norm())

    u_zero = torch.zeros(rnn.n_inputs)
    with torch.no_grad():
        for _ in range(125):
            g_pre = y @ rnn.W_gate.T + rnn.b_gate
            pre   = y @ rnn.W_rec.T + rnn.b_rec
            gate  = torch.sigmoid(g_pre)
            y     = gate * y + (1 - gate) * torch.relu(pre)

    retained = float(y.norm()) / (y0_norm + 1e-8)
    avg_gate = float(torch.sigmoid(rnn.b_gate).mean())
    return retained, avg_gate


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("=" * 60)
    print("Step 3: GRU-style gating architecture test")
    print("=" * 60)

    # ── Part A: Sanity check — train Stages 0-2 from scratch ──────────────────
    print("\nPart A: Stages 0-2 sanity check (from scratch)")
    rnn = GatedDynamicRoutingRNN(n_units=100).to(DEVICE)
    curriculum = CurriculumManager()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)

    # Stage 0
    loss0 = train_stage0(rnn, optimizer, n_steps=100, device=DEVICE, rng=rng)
    curriculum.stage = 1
    print(f"  Stage 0 done. Loss: {loss0:.4f}")

    stages_ok = False
    stage1_step = None
    stage2_step = None
    total_steps = 0

    while curriculum.stage <= 2 and total_steps < 1000:
        stage = curriculum.stage
        loss = train_step_gated(rnn, optimizer, stage, curriculum.fa_loss_weight,
                                batch_size=1, rng=rng,
                                bptt_trials=15, max_bptt_windows=8, train_n_blocks=2)
        total_steps += 1
        if total_steps % 100 == 0:
            sessions = evaluate_gated(rnn, stage, n_sessions=3, rng=rng)
            metrics, advanced, regressed = curriculum.evaluate(sessions, total_steps)
            dp = metrics.get('dprime_intra', metrics.get('dprime', 0.0))
            print(f"  step {total_steps:4d} | stage {stage} | loss {loss:.4f} | d'_intra {dp:.2f}"
                  f"{' ADVANCED' if advanced else ''}")
            if advanced:
                if stage == 1:
                    stage1_step = total_steps
                elif stage == 2:
                    stage2_step = total_steps
                    stages_ok = True
                    break

    print(f"\n  Stages 0-2 complete: {stages_ok}")
    print(f"  Stage 1 advanced at step: {stage1_step}")
    print(f"  Stage 2 advanced at step: {stage2_step}")

    # ── Part B: Load Stage 2 weights and test Stage 3 ─────────────────────────
    print("\nPart B: Stage 3 test (loading Stage 2 checkpoint from v24)")
    rnn2 = GatedDynamicRoutingRNN(n_units=100).to(DEVICE)
    ckpt_path = os.path.normpath(CHECKPOINT_V24)
    if os.path.exists(ckpt_path):
        vanilla_sd = torch.load(ckpt_path, map_location=DEVICE)
        rnn2.load_from_vanilla(vanilla_sd)
        print(f"  Loaded Stage 2 weights from {ckpt_path}")
    else:
        # Fall back to freshly trained weights from Part A
        print(f"  WARNING: v24 checkpoint not found at {ckpt_path}, using Part A weights")
        rnn2.load_state_dict(rnn.state_dict())

    curriculum2 = CurriculumManager()
    curriculum2.stage = 3

    print("  Initializing context attractor...")
    initialize_context_attractor(rnn2, alpha=rnn2.alpha)
    optimizer2 = torch.optim.Adam(rnn2.parameters(), lr=1e-3)

    d_inter_100 = 0.0
    d_inter_200 = 0.0
    d_inter_300 = 0.0
    rng2 = np.random.default_rng(SEED)

    for step in range(1, 301):
        loss = train_step_gated(rnn2, optimizer2, 3, curriculum2.fa_loss_weight,
                                batch_size=1, rng=rng2,
                                bptt_trials=90, max_bptt_windows=2, train_n_blocks=2,
                                max_sr=1.0)
        if step % 100 == 0:
            sessions = evaluate_gated(rnn2, 3, n_sessions=3, rng=rng2)
            metrics, advanced, _ = curriculum2.evaluate(sessions, step)
            dp_intra = metrics.get('dprime_intra', metrics.get('dprime', 0.0))
            dp_inter = metrics.get('dprime_inter', 0.0)
            print(f"  step {step:3d} | loss {loss:.4f} | d'_intra {dp_intra:.2f} | d'_inter {dp_inter:.2f}"
                  f"{' ADVANCED' if advanced else ''}")
            if step == 100:
                d_inter_100 = dp_inter
            elif step == 200:
                d_inter_200 = dp_inter
            elif step == 300:
                d_inter_300 = dp_inter

    # ── Part C: Gate persistence test ─────────────────────────────────────────
    print("\nPart C: Gate persistence test (signal retention after 125 steps)")
    retention, avg_gate = gate_persistence_test(rnn2)
    print(f"  Signal retained after 125 steps: {retention:.4f} (vanilla RNN: ~0.0079)")
    print(f"  Mean gate value (b_gate only): {avg_gate:.4f} (init ~0.2)")

    # ── Results ───────────────────────────────────────────────────────────────
    result = {
        "gru_stages_0_2_ok": stages_ok,
        "stage1_step": stage1_step,
        "stage2_step": stage2_step,
        "d_prime_inter_at_100": round(d_inter_100, 4),
        "d_prime_inter_at_200": round(d_inter_200, 4),
        "d_prime_inter_at_300": round(d_inter_300, 4),
        "signal_retention_after_125_steps": round(retention, 4),
        "avg_gate_value_after_training": round(avg_gate, 4),
        "key_findings": (
            f"GRU Stages 0-2 {'OK' if stages_ok else 'FAILED'}. "
            f"Stage 3 d'_inter: {d_inter_100:.2f} / {d_inter_200:.2f} / {d_inter_300:.2f} "
            f"at steps 100/200/300. "
            f"Signal retention {retention:.4f} vs vanilla ~0.008. "
            f"Avg gate {avg_gate:.3f}."
        ),
        "recommendation": (
            "Use GRU if d'_inter > 0 without teacher forcing, otherwise combine GRU + teacher forcing"
            if d_inter_300 > 0 else
            "GRU alone insufficient; combine with teacher forcing or increase gate initialization"
        )
    }

    result_path = os.path.join(OUTDIR, 'result.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {result_path}")
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
