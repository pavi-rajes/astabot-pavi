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

### COMPLETED: Training infrastructure + 5 bugs fixed

All modules implemented. Five bugs fixed (in chronological order):

1. **Spectral radius initialization** (init at 0.8 explicitly via eigvals)
2. **Stage 0 saturation** (freeze recurrent weights during Stage 0)
3. **BPTT NaN from recurrent noise** (removed noise from BPTT pass)
4. **Regression condition** (`n_hits < 10` → `n_hits == 0` — can't trigger on 3 eval sessions)
5. **Operator norm ≠ spectral radius** (max_sr=1.05→2.0; SVD norm is ~2× eigenvalue radius)
6. **Miss weight in Stage 1** (only apply miss_loss_weight for stage≥2, not Stage 1)
7. **Contradictory Stage 2 supervision** (Stage 1/2 now use fixed rewarded targets, not alternating)

### Previous training runs (all superseded):

- **Seeds 0, 42 (pre-fix):** Stage 1 worked (d'=4.65) but Stage 2 crashed with BCE NaN
- **Seeds 99, 100, 101 (max_sr=1.05):** Over-clamped W_rec; d' oscillated 4.65→0
- **Seed 42 v5 (max_sr=2.0, miss_weight_stage1=off):** Stage 1 advanced ✅ but Stage 2 d'=0 immediately → root cause identified: contradictory block structure

### Root cause of Stage 2 d'=0 collapse (confirmed)

In Stage 2, `_stage2_trials()` uses AUD1/AUD2 but the session's `rewarded_target` still alternates VIS1/AUD1. When rewarded=VIS1, even AUD1 trials have target=0 → network gets contradictory signals: sometimes "lick AUD1," sometimes "don't lick AUD1." Resolution: z≈0 everywhere (never lick), giving low loss (0.04-0.08) because no-lick is "correct" in VIS1-rewarded blocks.

**Fix applied (tasks/session.py):** For Stage 1: `rewarded_sequence = [VIS1]*n_blocks`. For Stage 2: `rewarded_sequence = [AUD1]*n_blocks`. Stage 3+: normal alternation.

### IN PROGRESS: Seed 42 v6 (checkpoints_v6/seed_42/)

All fixes applied:
1. Spectral norm clipping max_sr=2.0 (prevents Stage 2 crash)
2. Miss weight 3× for Stage 2+ (prevents z=0 collapse)
3. Miss weight 1.0 for Stage 1 (matches seeds 0/42 that worked)
4. Regression: n_hits==0 (prevents spurious Stage 1/2 regression)
5. Fixed block structure: Stage 1=VIS1-only blocks, Stage 2=AUD1-only blocks

Launched: 2026-03-20. Expected Stage 1 advancement ~step 300. Stage 2 first eval ~step 400.

## 7. Open Questions & Confusions

1. **Will Stage 2 work now?** The contradictory block supervision was almost certainly the cause. With consistent AUD1-rewarded blocks in Stage 2, the miss_weight should prevent z=0 collapse. Confidence: 85%.

2. **Stage 3 difficulty:** Stage 3 requires context inference (reward-based block switching). This is the hardest stage. Unlike Stage 2 (always AUD1), Stage 3 genuinely alternates contexts and the network must infer which from reward history. Will need 500-2000 steps.

3. **Evaluation metric for Stage 2:** `_stage12_metrics(rewarded=AUD1)` selects AUD1 hits and AUD2 FAs regardless of block. Now that Stage 2 eval sessions also use AUD1-only blocks, this is consistent.

## 8. Implementation Progress

### Code Status

```
training/train_rnn.py  ✅ max_sr=2.0, miss_weight only for stage≥2
tasks/session.py       ✅ Stage 1=VIS1-only, Stage 2=AUD1-only block structure
tasks/curriculum.py    ✅ Regression: n_hits==0 (not n_hits<10)
models/rnn.py          ✅ W_rec init at spectral radius 0.8
checkpoints_v6/seed_42/   🔄 Stage 1 in progress (all fixes applied)
```

### Next Steps

1. **Monitor seed 42 v6:** Check Stage 1 advancement and Stage 2 stability
2. **If Stage 2 works:** Let training continue to Stage 3 automatically
3. **Stage 3 monitoring:** Context inference requires reward-based block switching
4. **Validation plots:** After training completes
5. **Phase 2:** latent circuit model

### Completed
- [x] All core modules implemented
- [x] Seven bugs fixed
- [x] Stage 1 confirmed working (d'=4.65 → advanced in v5)
- [x] Stage 2 root cause identified and fixed (contradictory block structure)

### Still To Do
- [ ] Confirm Stage 2 stable with new fixes
- [ ] Stage 3 training and validation
- [ ] Validation plots (hit/FA rates, block transitions, PCA)
- [ ] Phase 2: latent circuit model
