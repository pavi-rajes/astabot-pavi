"""
Hint Strength Fade Schedule Sweep — Step 2 Experiment
======================================================
Tests 3 HINT_FADE_STEPS values [200, 500, 1000] for Stage 3 teacher forcing.
Each condition trains for 300 Stage-3 steps from the same Stage 2 checkpoint
(checkpoints_v24/seed_42/rnn_stage2.pt) and records d'_inter at steps 100, 200, 300.

Usage:
    conda run -n latent_circuit python .asta/experiment/2026-04-03-hint-strength-sweep/sweep.py
"""

import os
import sys
import copy
import json
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — allow imports from project root
# ---------------------------------------------------------------------------
# The script lives at .asta/experiment/2026-04-03-.../sweep.py
# Project root is 3 levels up: .asta/ -> experiment/ -> 2026-...-slug/ -> sweep.py
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Imports from project codebase
# ---------------------------------------------------------------------------
from models.rnn import DynamicRoutingRNN, make_rnn_fn
from training.train_rnn import (
    initialize_context_attractor,
    get_context_direction,
    make_hinted_rnn_fn,
    train_step,
    evaluate,
)
from tasks.curriculum import CurriculumManager

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'checkpoints_v24', 'seed_42', 'rnn_stage2.pt')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(OUTPUT_DIR, 'result.json')

DEVICE = 'cpu'
SEED = 42
N_UNITS = 100          # v24 checkpoint uses N=100 (per spec; fall back if mismatch)
STAGE = 3
BATCH_SIZE = 4
BPTT_TRIALS = 90       # full-block BPTT for Stage 3
MAX_BPTT_WINDOWS = 2   # one per training block
TRAIN_N_BLOCKS = 2
MAX_SR = 1.0           # spectral radius clipping for Stage 3
FA_LOSS_WEIGHT = 3.0   # curriculum.fa_loss_weight at stage 3
MISS_LOSS_WEIGHT = 3.0
LR = 1e-3

HINT_FADE_STEPS_LIST = [200, 500, 1000]
TOTAL_STEPS = 300
EVAL_AT_STEPS = [100, 200, 300]
N_EVAL_SESSIONS = 8    # enough for stable d' estimate


# ---------------------------------------------------------------------------
# Helper: load checkpoint into an RNN, auto-detecting N
# ---------------------------------------------------------------------------
def load_stage2_checkpoint(path, device='cpu'):
    """Load Stage 2 checkpoint, auto-detecting n_units from saved state dict."""
    state = torch.load(path, map_location=device)
    # Infer N from W_rec shape
    n_units = state['W_rec'].shape[0]
    print(f"  Detected n_units={n_units} from checkpoint.")
    rnn = DynamicRoutingRNN(n_units=n_units).to(device)
    rnn.load_state_dict(state)
    return rnn


# ---------------------------------------------------------------------------
# Helper: extract d'_inter from evaluate() output
# ---------------------------------------------------------------------------
def compute_dprime_inter(rnn, device='cpu', rng=None):
    """Run evaluation and return mean d'_inter across all blocks."""
    sessions = evaluate(rnn, stage=STAGE, n_sessions=N_EVAL_SESSIONS,
                        device=device, rng=rng)
    curriculum = CurriculumManager()
    curriculum.stage = STAGE
    metrics, _, _ = curriculum.evaluate(sessions, training_step=0)
    return metrics.get('dprime_inter', 0.0)


# ---------------------------------------------------------------------------
# Run one condition
# ---------------------------------------------------------------------------
def run_condition(hint_fade_steps, base_state_dict, device='cpu'):
    """
    Train Stage 3 for TOTAL_STEPS from Stage 2 weights with given fade schedule.
    Returns dict with d'_inter at EVAL_AT_STEPS.
    """
    print(f"\n{'='*60}")
    print(f"  HINT_FADE_STEPS = {hint_fade_steps}")
    print(f"{'='*60}")

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    # Load fresh copy of Stage 2 weights
    n_units = base_state_dict['W_rec'].shape[0]
    rnn = DynamicRoutingRNN(n_units=n_units).to(device)
    rnn.load_state_dict(copy.deepcopy(base_state_dict))

    # Initialize context attractor (must be done before Stage 3 training)
    print("  Initializing context attractor...")
    initialize_context_attractor(rnn, alpha=rnn.alpha)

    # Verify SR after init
    with torch.no_grad():
        eigs = torch.linalg.eigvals(rnn.W_rec)
        sr_post = float(eigs.abs().max())
        if sr_post > 1.0:
            rnn.W_rec.data.mul_(1.0 / sr_post)
            print(f"  Post-init SR clip: {sr_post:.4f} -> 1.0000")
        else:
            print(f"  Post-init SR: {sr_post:.4f}")

    # Compute context direction
    v_context = get_context_direction(rnn).to(device)
    hint_strength_ref = [1.0]

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)

    results_at = {}

    for step in range(1, TOTAL_STEPS + 1):
        # Fade hint strength
        hint_strength_ref[0] = max(0.0, 1.0 - (step - 1) / hint_fade_steps)

        # Build hinted rnn_fn
        base_fn = make_rnn_fn(rnn, add_noise=True, device=device)
        hinted_fn = make_hinted_rnn_fn(base_fn, v_context, hint_strength_ref)

        loss = train_step(
            rnn, optimizer, stage=STAGE,
            fa_loss_weight=FA_LOSS_WEIGHT,
            miss_loss_weight=MISS_LOSS_WEIGHT,
            batch_size=BATCH_SIZE,
            device=device,
            rng=rng,
            max_bptt_windows=MAX_BPTT_WINDOWS,
            bptt_trials=BPTT_TRIALS,
            train_n_blocks=TRAIN_N_BLOCKS,
            max_sr=MAX_SR,
            rnn_fn_override=hinted_fn,
        )

        if step in EVAL_AT_STEPS:
            dp_inter = compute_dprime_inter(rnn, device=device, rng=rng)
            results_at[step] = dp_inter
            print(f"  step {step:4d} | hint={hint_strength_ref[0]:.3f} | "
                  f"loss={loss:.4f} | d'_inter={dp_inter:.3f}")
        else:
            if step % 50 == 0:
                print(f"  step {step:4d} | hint={hint_strength_ref[0]:.3f} | loss={loss:.4f}")

    return {
        'hint_fade_steps': hint_fade_steps,
        'd_prime_inter_at_100': float(results_at.get(100, 0.0)),
        'd_prime_inter_at_200': float(results_at.get(200, 0.0)),
        'd_prime_inter_at_300': float(results_at.get(300, 0.0)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading Stage 2 checkpoint: {CHECKPOINT_PATH}")
    base_state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    sweep_results = []
    for hint_fade_steps in HINT_FADE_STEPS_LIST:
        result = run_condition(hint_fade_steps, base_state_dict, device=DEVICE)
        sweep_results.append(result)
        print(f"\n  => HINT_FADE_STEPS={hint_fade_steps}: "
              f"d'_inter @ [100,200,300] = "
              f"[{result['d_prime_inter_at_100']:.3f}, "
              f"{result['d_prime_inter_at_200']:.3f}, "
              f"{result['d_prime_inter_at_300']:.3f}]")

    # Determine best fade schedule (highest d'_inter at step 300)
    best = max(sweep_results, key=lambda r: r['d_prime_inter_at_300'])
    best_fade_steps = best['hint_fade_steps']

    # Build key findings string
    lines = []
    for r in sweep_results:
        lines.append(
            f"HINT_FADE_STEPS={r['hint_fade_steps']}: "
            f"d'_inter @ 100={r['d_prime_inter_at_100']:.3f}, "
            f"@ 200={r['d_prime_inter_at_200']:.3f}, "
            f"@ 300={r['d_prime_inter_at_300']:.3f}"
        )
    key_findings = "; ".join(lines)

    # Determine if faster fading helps (compare 200 vs 1000 at step 300)
    r200 = next(r for r in sweep_results if r['hint_fade_steps'] == 200)
    r500 = next(r for r in sweep_results if r['hint_fade_steps'] == 500)
    r1000 = next(r for r in sweep_results if r['hint_fade_steps'] == 1000)

    if r200['d_prime_inter_at_300'] > r1000['d_prime_inter_at_300'] + 0.1:
        recommendation = (
            f"Faster fade (HINT_FADE_STEPS=200) outperforms slower fade at step 300 "
            f"(d'_inter={r200['d_prime_inter_at_300']:.3f} vs {r1000['d_prime_inter_at_300']:.3f}). "
            f"Recommend HINT_FADE_STEPS={best_fade_steps} for Stage 3."
        )
    elif r1000['d_prime_inter_at_300'] > r200['d_prime_inter_at_300'] + 0.1:
        recommendation = (
            f"Slower fade (HINT_FADE_STEPS=1000) outperforms faster fade at step 300 "
            f"(d'_inter={r1000['d_prime_inter_at_300']:.3f} vs {r200['d_prime_inter_at_300']:.3f}). "
            f"The longer scaffold period benefits learning. "
            f"Recommend HINT_FADE_STEPS={best_fade_steps} for Stage 3."
        )
    else:
        recommendation = (
            f"All fade schedules perform similarly at step 300 "
            f"(d'_inter range: "
            f"{min(r['d_prime_inter_at_300'] for r in sweep_results):.3f}–"
            f"{max(r['d_prime_inter_at_300'] for r in sweep_results):.3f}). "
            f"Best so far: HINT_FADE_STEPS={best_fade_steps} "
            f"(d'_inter={best['d_prime_inter_at_300']:.3f}). "
            f"300 steps may be insufficient to differentiate schedules — "
            f"consider running longer (500–1000 steps) for definitive conclusions."
        )

    output = {
        'sweep_results': sweep_results,
        'best_fade_steps': best_fade_steps,
        'key_findings': key_findings,
        'recommendation': recommendation,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {RESULT_PATH}")
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
