# Research State: Latent Circuit Inference for Dynamic Routing Task

## 1. Research Question & Scope

**Primary Goal:** Implement and train a ground-truth RNN (Phase 1) on a dynamic routing task where context is inferred from reward feedback (not explicitly cued).

**Scope:** Phase 1 only — train RNN through 4-stage curriculum (Stages 0→4).

## 2. Operational Definitions

- **Context belief:** Network's internal representation of rewarded target, inferred from reward+lick.
- **d' (d-prime):** d' = Z(hit_rate) - Z(FA_rate), clipped to ±4.65.
- **Instruction trial:** Block start trial with late_autoreward (fires reward regardless of lick).
- **BPTT window:** 15 trials for Stages 1-2; 180 trials (full 2-block) for Stage 3.

## 3. Related Work

- **Langdon & Engel (2025, Nature Neuroscience):** Latent circuit inference. Ref: engellab/latentcircuit.

## 4. Hypotheses

**H1** (90%): Trained RNN shows context-dependent suppression of irrelevant sensory representations.
**H2** (85%): Latent circuit reveals reward→context-belief pathway.
**H3** (80%): Context-belief node shows block-locked dynamics that flip at transitions.
**H4** (70%): 6-10 PCs capture >90% of task-related variance.

## 5. Architecture

- **IntegratorRNNCoarse:** N=100, tau=500ms, dt=100ms, alpha=0.2, sigma=0.15
  - Channel 7: leaky reward integrator c = tanh(γ·c + (1-γ)·δ), δ∈{+1, -0.5, 0}
  - W_rec init: spectral radius = 0.8
- **Python:** conda env `latent_circuit` (PyTorch, CPU only)

## 6. Completed: Stages 0-2

**Seed 42 (checkpoints_v24/seed_42/):**
- Stage 1 advanced at step 200 (d'_intra=4.65)
- Stage 2 advanced at step 300 (d'_intra=4.65)
- W_rec spectral radius after Stage 2: 0.902

## 7. Stage 3 — Complete Failure History

| Approach | Max d'_inter | Root cause of failure |
|----------|-------------|----------------------|
| v9-v10: BPTT variants | 0.00 | 0.8%/trial retention; context decays in 1 trial |
| v11-v18: Rank-1 attractor | 0.00 | SR clip destroyed attractor |
| v19-v24: SR scaling | ±0.18 osc | TBPTT boundary blocks inter-block gradients |
| v25: Teacher forcing | 0.22→0 | Hint fades; attractor never crystallizes |
| GRU alone | 0.00 | Bootstrap deadlock |
| Coarse dt=100ms | 0.19 osc | Still ~0 over 45 trials; single-block BPTT |
| Coarse + 2-block BPTT | 0.19 osc | No persistent signal source |
| Integrator γ=0.90 (standalone) | 0.00 | W_in[:,7]≈0 bug; no gradient foothold |
| Integrator + Coarse (1-block BPTT) | 0.36 osc | Gradient can't learn sign-flip across block boundary |
| Integrator+Coarse+2block+CTX v1 | 0.285 osc | batch=1 gradient noise; λ_ctx=0.2 too weak (ran to 10000 steps, **DONE, FAILED**) |
| Integrator+Coarse+2block (no ctx) | 0.00 | No ctx loss; gradient sees boundary but no supervision (ran to 10000 steps, **DONE, FAILED**) |

### Four root causes

1. **Signal decay** — fix: integrator (c persists whole block)
2. **Gradient blindness** — fix: 2-block BPTT window
3. **Attractor bootstrap** — fix: ctx loss (λ_ctx)
4. **Gradient variance** *(v1 diagnosis)*: batch=1 → oscillation, no upward trend (mean d'_inter=0.055, std=0.097 over 2000 steps)

---

## 8. KEY FINDING: Explicit context ablation SUCCEEDED (2026-04-07)

**Script:** `training/train_explicit_ctx.py` | **Checkpoint:** `checkpoints_explicit_ctx/` | **Status: ✅ DONE**

Channel 7 = ±1.0 ground truth context (not learned from rewards). Stage 3 learning curve:

| step | d'_inter | n_pass |
|------|----------|--------|
| 350 | 0.48 | 0 |
| 500 | 1.00 | 5 |
| 600 | 1.65 | 5 |
| 750 | 2.32 | 10 | → **ADVANCED to Stage 4** |
| 1000 | 2.74 | 10 | → **DONE** |

**What this proves:**
- ✅ Architecture (IntegratorRNNCoarse) CAN solve the task
- ✅ Behavioral loss (BCE + cross-modal FA penalty) IS sufficient for Stage 3 advancement
- ✅ 2-block BPTT + ctx loss design IS correct
- ❌ **Sole remaining bottleneck:** learning to infer context from reward history (not provided explicitly)

---

## 9. Currently Running

| Run | PID | Checkpoint | Step | d'_inter | Status |
|-----|-----|-----------|------|----------|--------|
| **IntegratorCoarse v2** | **22244** | **checkpoints_integrator_coarse_v2/** | 350, Stage 3 | 0.16 (early) | **running** — γ=0.54, mid-warmup |

All other runs have completed (either DONE or hit max_steps limit and failed).

### v2 design (train_integrator_coarse_v2.py)
Targets the gradient variance root cause identified in v1:

| Param | v1 | v2 |
|-------|----|----|
| batch_size | 1 | **4** |
| LR Stage 3 | 1e-3 | **5e-4** |
| λ_ctx | 0.2 | **0.5** |
| max_sr Stage 3 | 1.0 | **1.2** |
| γ | fixed 0.90 | **0.50→0.90 over 500 steps** |

Note: crashed at step 550 on first launch due to `torch.linalg.eigvals` LAPACK failure. Fixed by replacing with `torch.linalg.matrix_norm(ord=2)` (SVD-based, stable). Relaunched 2026-04-07 11:38AM, PID 22244.

---

## 10. Next Steps (if v2 oscillates)

1. **Context teacher forcing:** blend explicit context (±1.0) with integrator c, fade over 500 steps — bridges the two experiments we've now run
2. **Fixed γ=0.50 (no annealing):** c≈0.97 throughout; stronger signal, less realistic
3. **λ_ctx=1.0:** make context the dominant objective in Stage 3

---

## 11. Code Structure

```
models/rnn_integrator_coarse.py  ✅ IntegratorRNNCoarse (dt=100ms + integrator)

training/train_integrator_coarse_ctx.py     ✅ v1: ctx loss (FAILED — 10000 steps, d'_inter≈0)
training/train_integrator_coarse_2block.py  ✅ 2-block BPTT, no ctx loss (FAILED)
training/train_integrator_coarse_v2.py      ✅ v2: batch=4, γ curriculum, λ_ctx=0.5 (RUNNING)
training/train_explicit_ctx.py              ✅ Ceiling ablation (DONE — d'_inter=2.74)
```

## 12. Still To Do

- [ ] Stage 3 advancement via learned context inference (d'_intra > 1.5 AND d'_inter > 1.5, ≥4/6 blocks)
- [ ] Stage 4 fine-tuning
- [ ] Validation plots (hit/FA, block transitions, PCA)
- [ ] Phase 2: latent circuit model
