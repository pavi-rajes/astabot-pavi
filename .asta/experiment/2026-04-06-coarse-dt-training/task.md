# Task: Coarse-dt RNN (dt=100ms, tau=500ms) — Stage 3 Training

Train a DynamicRoutingRNNCoarse on the dynamic routing task and evaluate whether
the coarser time resolution (dt=100ms instead of 20ms) enables inter-block context
discrimination (d'_inter > 0) in Stage 3.

## Key hypothesis

At dt=100ms, tau=500ms (alpha=0.2 — same leak rate):
- eff_lambda = 0.98 per step = 0.98 per 100ms
- After 1 trial (25 steps): 0.98^25 = **60% context retained** (vs 0.8% at dt=20ms)
- Training is 5x faster (25 steps/trial vs 125 steps/trial)

With 60% context retention per trial, BPTT should be able to learn from context
signals established by instruction trials, without any architectural changes.

## What to monitor

**Log file**: `checkpoints_coarse/seed_42/log.csv`

Wait for at least 10 Stage 3 rows (stage=3), then analyze:
1. Does d'_inter ever exceed 0.5? At what step?
2. What is the trajectory — is it improving, flat, or oscillating near 0?
3. How does it compare to the failed approaches (all stuck at d'_inter ≈ 0):
   - Teacher forcing (v25): peaked 0.22, collapsed
   - GRU alone: 0.00 throughout
   - Integrator γ=0.99: ±0.06 oscillating
   - Integrator γ=0.90: ±0.06 oscillating (step ~700, still running)
4. How fast are the training steps? (expected ~2 min/step vs ~10 min/step before)

## Architecture

- DynamicRoutingRNNCoarse: vanilla RNN N=100, dt=100ms, tau=500ms, alpha=0.2
- Loaded from Stage 2 checkpoint (checkpoints_v24/seed_42/rnn_stage2.pt)
- Stage 3 with context attractor init + light teacher forcing (hint fades over 500 steps)
- BPTT: 90-trial windows × 25 steps = 2250 steps/window (5x faster than before)

## Output

Write result to `result.json` in this directory:
- max_d_prime_inter: float
- step_of_max: int
- trend: "improving" | "declining" | "flat" | "oscillating"
- steps_per_minute: float (training speed)
- key_finding: str
- recommendation: str
