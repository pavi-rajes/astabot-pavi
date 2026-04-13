# Research State: Latent Circuit Inference for Dynamic Routing Task

## 1. Research Question & Scope

**Primary Goal:** Implement and train a ground-truth RNN (Phase 1) on a dynamic routing task where context is inferred from reward feedback (not explicitly cued).

**Scope:** Phase 1 only — train RNN through 4-stage curriculum (Stages 0→4).

## 2. Operational Definitions

- **Context belief:** Network's internal representation of rewarded target, inferred from reward+lick.
- **d' (d-prime):** d' = Z(hit_rate) - Z(FA_rate), clipped to ±4.65.
- **Instruction trial:** Block start trial with late_autoreward.
- **BPTT window:** 15 trials per gradient window; 8 randomly sampled per step.

## 3. Related Work

- **Langdon & Engel (2025, Nature Neuroscience):** Latent circuit inference. Ref: engellab/latentcircuit.

## 4. Hypotheses

**H1** (90%): Trained RNN shows context-dependent suppression of irrelevant sensory representations.
**H2** (85%): Latent circuit reveals reward→context-belief pathway.
**H3** (80%): Context-belief node shows block-locked dynamics that flip at transitions.
**H4** (70%): 6-10 PCs capture >90% of task-related variance.

## 5. Experimental Design

- **Architecture:** DynamicRoutingRNN, N=100, tau=100ms, dt=20ms, sigma=0.15
- **W_rec init:** Spectral radius = 0.8 (critical for stability)
- **Stage 0:** Only trains output weights (w_out, b_out) — recurrent weights frozen
- **Training:** 2 blocks per session, 8 random BPTT windows, eval_every=50-100
- **Python:** `/c/Users/peter/anaconda3/python.exe` (PyTorch 2.1.2+cpu)

## 6. Results Summary

### COMPLETED: Training infrastructure

All modules implemented, smoke-tested, three critical bugs fixed (spectral radius, Stage 0 saturation, BPTT NaN).

### IN PROGRESS: Two parallel training runs

**Seed 0 (checkpoints/seed_00/) — 6-block training sessions (slower, pre-fix):**

| step | stage | loss   | d'_intra | advanced |
|------|-------|--------|----------|----------|
| 150  | 1     | 0.493  | 0.000    | 0        |
| 200  | 1     | 0.386  | **4.653**| 0        |
| 250  | 1     | 0.333  | **4.653**| **1** ✅ |
| 300  | 2     | 0.316  | 2.248    | 0        |
| 350  | 2     | 0.077  | 0.000    | 0        |

**Key findings from Seed 0:**
- Stage 1 (visual discrim): Advanced at step 250, d'=4.65 in just 100 steps of actual learning. **Very fast learning!**
- Stage 2 (auditory discrim): Started well (d'=2.25 at step 300) but dropped to d'=0 at step 350. Loss also dropped steeply (0.316→0.077). This suggests the network may be collapsing to z≈0 (never licking) — low loss because most trials are no-lick, but d'=0 because hit_rate≈0.
- Training will continue and may recover (curriculum doesn't force regression unless n_hits < 10 for 2 evals, which requires aud1 trials to be absent).

**Seed 42 (checkpoints_fast/seed_42/) — 2-block training sessions (faster, post-fix):**

| step | stage | loss   | d'_intra | advanced |
|------|-------|--------|----------|----------|
| 200  | 1     | 0.338  | **4.653**| 0        |
| 300  | 1     | 0.425  | **4.653**| **1** ✅ |

**Key findings from Seed 42:**
- Stage 1 advanced at step 300 with d'=4.65
- Training is faster (~4s/step vs ~12s/step for seed 0)
- Just entered Stage 2 at step 300; no Stage 2 data yet

### Identified issue: Stage 2 collapse

**Symptom:** After Stage 1 advances, Stage 2 shows d'=2.25 → d'=0 (step 300→350).
**Likely cause:** Network collapses to z≈0 (stops licking), giving low BCE loss on no-lick trials but zero hit rate on aud1 trials.
**Root cause hypothesis:** The random BPTT window sampling (8/14 windows) may undersample the aud1 (lick-target) windows when the proportion of lick trials is < 50%. The gradient then pushes toward z≈0.
**Possible fixes:**
1. Use `fa_loss_weight=3` for Stage 2 as well (add extra penalty for missed licks rather than for false alarms — i.e., use a "miss penalty" weight)
2. Use deterministic window selection (include first few windows which contain instruction trials)
3. Reduce eval_every to catch the collapse sooner and debug
4. Increase the lick-target loss weight relative to no-lick trials

## 7. Open Questions & Confusions

1. **Stage 2 collapse:** Is the d'=0 at step 350 a transient dip (will it recover) or persistent? Need to see step 400 results.

2. **Loss 0.077 with d'=0:** Mathematically: if z≈0 for all trials, loss should be ~4.6 for lick-target trials. Loss 0.077 is too low for z=0 everywhere. Alternative: z is very small but nonzero, or the BPTT window sampling accidentally selects mostly no-lick windows.

3. **Stage 3 difficulty:** Stage 3 requires context inference (reward-based block switching). This is the hardest stage. Will need 500-2000 steps with proper context dynamics.

4. **Need for a "miss weight":** The spec has ×3 FA loss weight for Stage 3 but not for Stages 1-2. A hit-miss asymmetry might help prevent z≈0 collapse.

## 8. Implementation Progress

### Code Status

```
training/train_rnn.py  ✅ Fixed: n_blocks=train_n_blocks for ALL stages (not just ≥3)
                          train_n_blocks=2 now default for faster training
checkpoints/seed_00/   📊 Stage 2, step 350, d'=0 (possible collapse)
checkpoints_fast/seed_42/ 📊 Stage 1 advanced at step 300, entering Stage 2
```

### Next Steps

1. **Immediate:** Wait for both runs to produce more Stage 2 data (background runs active)
2. **Debug Stage 2 collapse:** Add a "miss weight" (opposite of FA weight) — penalize missed licks more — to prevent z=0 collapse
3. **Monitor Stage 3:** If either run reaches Stage 3, check if reward-based context inference works
4. **After training completes:** Run validation (hit/FA rates, block transitions, PCA)

### Completed
- [x] All core modules implemented
- [x] Three critical bugs fixed
- [x] Training infrastructure working
- [x] Stage 1 confirmed working (d'=4.65 in ~100 steps)
- [x] Two parallel training runs started

### Still To Do
- [ ] Fix Stage 2 collapse issue
- [ ] Stage 3 training and validation
- [ ] Validation plots (hit/FA rates, block transitions, PCA)
- [ ] Phase 2: latent circuit model
