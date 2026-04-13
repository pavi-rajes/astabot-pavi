# Research State: Latent Circuit Inference for Dynamic Routing Task

## 1. Research Question & Scope

**Primary Goal:** Implement and train a ground-truth RNN (Phase 1) on a dynamic routing task where context is inferred from reward feedback (not explicitly cued).

**Scope:** Phase 1 only — train RNN through 4-stage curriculum (Stages 0→4).

## 2. Operational Definitions

- **Context belief:** Network's internal representation of rewarded target, inferred from reward+lick.
- **d' (d-prime):** d' = Z(hit_rate) - Z(FA_rate), clipped to ±4.65.
- **Instruction trial:** Block start trial with late_autoreward (fires reward regardless of lick).
- **BPTT window:** 15 trials for Stages 1-2; 90 trials (full block) for Stage 3.

## 3. Related Work

- **Langdon & Engel (2025, Nature Neuroscience):** Latent circuit inference. Ref: engellab/latentcircuit.

## 4. Hypotheses

**H1** (90%): Trained RNN shows context-dependent suppression of irrelevant sensory representations.
**H2** (85%): Latent circuit reveals reward→context-belief pathway.
**H3** (80%): Context-belief node shows block-locked dynamics that flip at transitions.
**H4** (70%): 6-10 PCs capture >90% of task-related variance.

## 5. Experimental Design

- **Architecture:** DynamicRoutingRNN, N=100, tau=100ms, dt=20ms, sigma=0.15
- **W_rec init:** Spectral radius = 0.8
- **Stage 0:** Only trains output weights (w_out, b_out)
- **Training:** 2 blocks per session, eval_every=100
- **Python:** `/c/Users/peter/anaconda3/python.exe` (PyTorch 2.1.2+cpu)

## 6. Results Summary

### COMPLETED: Stages 0-2

Eight bugs fixed through debugging:
1. Spectral radius init (0.8 via eigvals)
2. Stage 0 saturation (freeze recurrent weights)
3. BPTT NaN from noise (removed from BPTT)
4. Regression condition (n_hits<10 → n_hits==0)
5. Operator norm over-clipping (max_sr 1.05 → 2.0)
6. Miss weight in Stage 1 (only apply for stage≥2)
7. Contradictory Stage 2 supervision (Stage 1/2 use fixed rewarded targets)
8. **Instruction trial auto-reward bug (models/rnn.py):** Auto-reward was added to u_trial AFTER the forward loop, so wu (precomputed W_in@u) never reflected it. The network never processed instruction rewards. Fixed: fire auto-reward inline at t==auto_start during the forward loop.

**Seed 42 v8 (checkpoints_v8/seed_42/) results:**

| step | stage | loss  | d'_intra | advanced |
|------|-------|-------|----------|----------|
| 300  | 1     | 0.010 | 4.653    | ✅ Stage 1 |
| 500  | 2     | 0.009 | 4.653    | ✅ Stage 2 |
| 600-1100 | 3 | 0.058→0.012 | 0.000 | 0 |

### IN PROGRESS: Stage 3 (seed 42 v8)

Stage 3 training shows very low loss (0.012) in noisy training but persistent d'=0 in noiseless evaluation. Process still running (PID 20676), 1400 Stage 3 steps remaining out of 2000 max.

**Diagnosis: Noise-dependent context inference**

The network learns to discriminate in noisy training sessions (loss=0.012) but fails in noiseless evaluation (d'=0). Root cause analysis:

1. The auto-reward on instruction trials fires for only REWARD_DURATION=2 timesteps (40ms) at the very end of the trial (steps 123-124 of 125).
2. With alpha=0.2 dynamics, the reward signal decays as 0.8^T per step. After just 2 timesteps, the context signal in y_final is: Δy ≈ 0.288 × |W_in[:,4]|.
3. After 1 subsequent trial (125 steps), this signal decays to: 0.288 × 0.8^125 × |W_in[:,4]| ≈ 10^{-13} × |W_in[:,4]|. Essentially zero.
4. Without noise to bootstrap licking, the context signal cannot persist.
5. In noisy training: noise → lick → reward → strong positive feedback → low loss.

**The fundamental challenge:** Context needs to persist for ~11,000 timesteps (90 trials × 125 steps) but the RNN's natural time constant allows persistence of ~5 steps (τ/dt = 100ms/20ms).

**Required mechanism:** The network must develop RECURRENT ATTRACTOR dynamics — self-sustaining activation patterns that persist without exponential decay. This requires W_rec to develop near-1 eigenvalues for the context subspace, which is hard to learn from gradient descent alone.

## 7. Open Questions & Confusions

1. **Will Stage 3 eventually learn with more steps?** The loss is decreasing (0.058→0.012). Gradient descent may slowly shape W_rec toward bistable attractors. May require 500-2000 more steps. Confidence: 40%.

2. **Evaluation with noise:** If Stage 3 advancement used noisy evaluation (matching training conditions), d' would likely be non-zero. This could be a valid modification — use noise during eval for Stage 3 advancement, then validate noiseless separately.

3. **Should REWARD_DURATION be longer?** Current REWARD_DURATION=2 gives only 40ms of reward signal at instruction trial end. A longer duration (e.g., 10-20 timesteps) would strengthen the context signal in y_final. Check what the actual mouse task uses.

4. **Is Stage 3 actually solvable with this architecture?** The literature (Langdon & Engel 2025) trains on the full task with all stages. Their implementation might use different training tricks or longer sequences.

## 8. Implementation Progress

### Code Status

```
models/rnn.py          ✅ Auto-reward bug fixed (inline at t==auto_start)
training/train_rnn.py  ✅ Full-block BPTT for Stage 3 (bptt_trials=90, max_windows=2)
tasks/session.py       ✅ Fixed block structure for Stages 1-2
tasks/curriculum.py    ✅ Regression: n_hits==0
checkpoints_v8/seed_42/ 🔄 Stage 3, step 1100, d'=0 (running)
```

### Next Steps

1. **Wait for Stage 3 to complete** (1400 more steps, ~3.9 hours at 10s/step)
2. **If d' remains 0:** Consider evaluating with noise for Stage 3 curriculum, OR increasing REWARD_DURATION, OR other architectural changes
3. **Check Langdon & Engel (2025) code** for how they handle Stage 3 training
4. **If Stage 3 works:** Stage 4 fine-tuning, then validation plots

### Completed
- [x] All core modules implemented
- [x] Eight bugs fixed
- [x] Stages 0-2 all working cleanly
- [x] Stage 3 entered and partially trained (600 steps)

### Still To Do
- [ ] Stage 3 advancement (d' > 1.5 for ≥4/6 blocks)
- [ ] Stage 4 fine-tuning
- [ ] Validation plots (hit/FA rates, block transitions, PCA)
- [ ] Phase 2: latent circuit model
