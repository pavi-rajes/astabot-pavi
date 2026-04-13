# Task: Reward Integrator RNN — Stage 3 Training Run

Train an IntegratorRNN on the dynamic routing task and evaluate whether the reward
integrator input (channel 7) enables inter-block context discrimination (d'_inter > 0)
in Stage 3.

## What to run

The training script already exists and a Python process is running it. Monitor the
training log and report results:

**Log file**: `checkpoints_integrator/seed_42/log.csv`

Wait for the log to contain at least 10 Stage 3 eval rows (step column with stage=3),
then analyze and report:

1. Does d'_inter ever exceed 0.5? If so, at which step?
2. What is the trend of d'_inter over Stage 3 steps?
3. Compare d'_inter trajectory to the teacher forcing baseline (which peaked at 0.22
   and collapsed back to 0): is the integrator doing better?
4. Is d'_intra maintained (should stay > 3.0)?

If the log has fewer than 10 Stage 3 rows, wait and check again every 2 minutes.

## Architecture

IntegratorRNN: vanilla RNN (N=100) with n_inputs=8. Channel 7 = leaky reward integrator:
- c_{t+1} = tanh(gamma * c_t + (1-gamma) * delta)
- delta: +1 reward, -0.5 miss, 0 otherwise
- gamma=0.99, so tau ≈ 100 trials

After 5 VIS1 instruction trials: c → +1 (VIS1 context)
After block switch: c → -1 (AUD1 context)

## Success criterion

Stage 3 advancement requires d'_intra > 1.5 AND d'_inter > 1.5 for ≥4/6 blocks,
2 consecutive evaluations. But even d'_inter > 0.5 sustained over 5+ evals would be
a breakthrough (all previous approaches were stuck at 0).

## Output

Write a result summary to `result.json` in this directory with:
- max_d_prime_inter: float
- step_of_max: int
- trend: "improving" | "declining" | "flat" | "collapsed"
- stage3_advanced: bool
- key_finding: str
- recommendation: str
