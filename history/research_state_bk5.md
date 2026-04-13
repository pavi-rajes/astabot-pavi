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
- **Training:** 2 blocks per session, 8 random BPTT windows, eval_every=100
- **Python:** `/c/Users/peter/anaconda3/python.exe` (PyTorch 2.1.2+cpu)

## 6. Results Summary

### COMPLETED: Training infrastructure + stability fixes

All modules implemented, smoke-tested. Four critical bugs fixed:
1. Spectral radius initialization (init at 0.8 explicitly)
2. Stage 0 saturation (freeze recurrent weights during Stage 0)
3. BPTT NaN from recurrent noise (removed noise from BPTT pass)
4. **NEW:** Spectral norm clipping after each optimizer step (prevents W_rec growing unstable)
5. **NEW:** Miss weight (×3) added to prevent Stage 2 z=0 collapse

### Previous training runs:

**Seed 0 (checkpoints/seed_00/) — CRASHED at Stage 2:**

| step | stage | loss   | d'_intra | advanced |
|------|-------|--------|----------|----------|
| 250  | 1     | 0.333  | **4.653**| **1** ✅ |
| 300  | 2     | 0.316  | 2.248    | 0        |
| 350  | 2     | 0.077  | 0.000    | 0        |

- Stage 1 advanced at step 250. Stage 2 collapsed (d'→0), then crashed with BCE NaN at ~step 360.
- Root cause: W_rec spectral radius grew >1.0 after 200+ gradient updates → hidden states diverged.

**Seed 42 (checkpoints_fast/seed_42/) — Status unknown:**

| step | stage | loss   | d'_intra | advanced |
|------|-------|--------|----------|----------|
| 300  | 1     | 0.425  | **4.653**| **1** ✅ |
| 400  | 2     | 0.316  | 2.358    | 0        |

- At step 400, Stage 2 d'=2.36 (promising). Run was still active when last checked but fate unclear.

### IN PROGRESS: Seed 99 (checkpoints_v2/seed_99/) — with fixes applied

**Fixes applied:**
1. Spectral norm projection: after optimizer.step(), if ||W_rec||_2 > 1.05, rescale W_rec.data *= 1.05 / ||W_rec||_2
2. Miss weight: ×3 loss weight for missed lick trials (rewarded stimulus, no lick) — prevents z=0 collapse
3. Both fixes now in training/train_rnn.py

Run started: 2026-03-20.
Command:
```
/c/Users/peter/anaconda3/python.exe training/train_rnn.py \
  --n_ensemble 1 --n_units 100 --max_steps 2000 \
  --eval_every 100 --n_eval_sessions 3 --batch_size 1 \
  --bptt_trials 15 --max_bptt_windows 8 --train_n_blocks 2 \
  --outdir checkpoints_v2 --seed_start 99
```

Stage 0 complete (rnn_stage0.pt written). Stage 1 training in progress.

## 7. Open Questions & Confusions

1. **Will spectral norm clipping + miss weight prevent Stage 2 collapse?** The spectral norm projection bounds the largest singular value of W_rec to ≤1.05, ensuring no divergence. The miss weight ×3 should prevent the z=0 collapse. Confidence: 80%.

2. **Stage 3 difficulty:** Stage 3 requires context inference (reward-based block switching). This is the hardest stage. Will need 500-2000 steps. Need to monitor whether reward→context-belief dynamics emerge.

3. **Loss 0.077 with d'=0 in Seed 0:** Likely because random BPTT sampling accidentally selected mostly no-lick windows after the network collapsed to z≈0.

## 8. Implementation Progress

### Code Status

```
training/train_rnn.py  ✅ Spectral norm clipping after optimizer.step()
                          ✅ Miss weight ×3 for missed lick trials
                          ✅ n_blocks=train_n_blocks for ALL stages
models/rnn.py          ✅ W_rec init at spectral radius 0.8
checkpoints/seed_00/   ❌ Crashed at Stage 2, step ~360 (pre-fix)
checkpoints_fast/seed_42/ 🔄 Unknown status (pre-fix run)
checkpoints_v2/seed_99/   🔄 Stage 1 in progress (post-fix run)
```

### Next Steps

1. **Monitor seed 99:** Check checkpoints_v2/seed_99/log.csv for Stage 1 → Stage 2 progress
2. **Verify Stage 2 stability:** Confirm spectral norm + miss weight prevent d'→0 collapse
3. **Stage 3 training:** Monitor reward-based context inference, expect 500-2000 steps
4. **Validation plots:** After training completes — hit/FA rates, block transitions, PCA
5. **Phase 2:** Implement latent circuit model (models/latent_circuit.py, models/stiefel.py, training/fit_latent.py)

### Completed
- [x] All core modules implemented
- [x] Four critical bugs fixed
- [x] Training infrastructure working
- [x] Stage 1 confirmed working (d'=4.65 in ~100 steps for seeds 0, 42)
- [x] Spectral norm clipping implemented
- [x] Miss weight implemented
- [x] Seed 99 training run launched

### Still To Do
- [ ] Confirm Stage 2 stability with new fixes
- [ ] Stage 3 training and validation
- [ ] Validation plots (hit/FA rates, block transitions, PCA)
- [ ] Phase 2: latent circuit model
