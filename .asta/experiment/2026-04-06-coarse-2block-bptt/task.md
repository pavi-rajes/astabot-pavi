# Task: Coarse-dt RNN with Full 2-Block BPTT Window

## Goal

Implement and run Stage 3 training using a **single BPTT window spanning both blocks**
(180 trials × 25 steps = 4,500 steps per window) with the coarse-dt RNN (dt=100ms, tau=500ms).

## Why this is the key experiment

Every approach so far has failed because the BPTT gradient cannot span block boundaries:
- 1-block BPTT: gradient only within a single block. Context decays to ~0 by mid-block.
- 2-block BPTT with carry state: the carry state between blocks has 0.98^2250 ≈ 0 context.

With a **single 2-block BPTT window** (no split), the gradient flows from Block 2 instruction
trials ALL THE WAY BACK to Block 1 activity. The network sees:
  "Block 1 had VIS1 rewarded → Block 2 has AUD1 rewarded → must switch"

The coarse-dt model makes this feasible: 4,500 steps/window vs 22,500 for standard dt.

## What to implement

Modify `training/train_coarse.py` to create a new training script
`training/train_coarse_2block.py` with these changes for Stage 3:

```python
# Stage 3 BPTT settings:
stage_bptt   = 180   # full 2-block session (180 trials × 25 steps = 4500 steps)
stage_bptt_w = 1     # single window (no split — gradient spans both blocks)
train_n_blocks = 2   # 2 blocks per session (already default)
```

Everything else identical to `training/train_coarse.py`.

## Implementation details

- Copy `training/train_coarse.py` to `training/train_coarse_2block.py`
- Change Stage 3 settings: `stage_bptt = 180`, `stage_bptt_w = 1`
- Output to `checkpoints_coarse_2block/`
- Load from `checkpoints_v24/seed_42/rnn_stage2.pt`
- Keep teacher forcing with hint_fade=500 (same as train_coarse.py)
- Keep all other hyperparameters identical

## Run it

After implementing, run:
```bash
conda run -n latent_circuit python training/train_coarse_2block.py \
  --outdir checkpoints_coarse_2block \
  --stage2_ckpt checkpoints_v24/seed_42/rnn_stage2.pt \
  --n_units 100 --max_steps 2000 \
  --eval_every 50 --n_eval_sessions 5 --batch_size 1 --seed 42
```

Monitor and report results from `checkpoints_coarse_2block/seed_42/log.csv`.

## Success criterion

d'_inter > 0.5 sustained over 3+ consecutive evals = breakthrough.
d'_inter > 1.5 for ≥4/6 blocks, 2 consecutive evals = Stage 3 advancement.

## Background

- All previous approaches stuck at d'_inter ≈ 0
- Coarse dt=100ms run: peaked at 0.19, then collapsed (single-block BPTT)
- Models live in `models/rnn_coarse.py`, session in `tasks/session_coarse.py`
- Use conda environment `latent_circuit` for all Python execution

## Output

Write results to `result.json` in this directory:
- max_d_prime_inter: float
- step_of_max: int  
- trend: "improving" | "oscillating" | "collapsed" | "advancing"
- stage3_advanced: bool
- steps_per_minute: float
- key_finding: str
- recommendation: str
