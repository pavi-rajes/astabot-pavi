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

- **Standard:** DynamicRoutingRNN, N=100, tau=100ms, dt=20ms, alpha=0.2, sigma=0.15
- **Coarse:** DynamicRoutingRNNCoarse, N=100, tau=500ms, dt=100ms, alpha=0.2, sigma=0.15
- **IntegratorCoarse:** DynamicRoutingRNNCoarse + leaky integrator channel 7 (γ=0.90)
- **W_rec init:** Spectral radius = 0.8
- **Python:** conda env `latent_circuit` (PyTorch, CPU only)

## 6. Completed: Stages 0-2

**Seed 42 (checkpoints_v24/seed_42/)** — standard dt=20ms:
- Stage 1 advanced at step 200 (d'_intra=4.65)
- Stage 2 advanced at step 300 (d'_intra=4.65)
- W_rec spectral radius after Stage 2: 0.902

## 7. Stage 3 — Complete Failure History

**Core problem:** Three interacting obstacles prevent context learning.

| Approach | Max d'_inter | Root cause of failure |
|----------|-------------|----------------------|
| v9-v10: BPTT variants | 0.00 | 0.8%/trial retention (0.98^125); context decays in 1 trial |
| v11-v18: Rank-1 attractor | 0.00 | SR clip destroyed attractor |
| v19-v24: SR scaling | ±0.18 osc | TBPTT boundary blocks inter-block gradients |
| v25: Teacher forcing (fade 1000) | 0.22 → 0 | Hint fades → attractor never crystallizes |
| GRU alone | 0.00 | Bootstrap deadlock: need context gradient to learn gating |
| Coarse dt=100ms | 0.19 osc | 60%/trial → still ~0 over 45 trials; single-block BPTT |
| Coarse + 2-block BPTT | 0.19 osc | No persistent signal; hidden state near 0 at block boundary |
| Integrator γ=0.90 (standalone) | 0.00 | Bug: W_in[:,7]≈0; no gradient foothold |
| Integrator + Coarse | 0.36 osc | BPTT split by block; gradient can't learn sign-flip usage |
| **Integrator+Coarse+2block+CTX v1** | **0.285 osc** | **batch=1 gradient noise; λ_ctx=0.2 too weak** |

### Root cause analysis (three obstacles + new finding)

1. **Signal decay:** 0.8%/trial retention. Fix: integrator (c persists the whole block).
2. **Gradient blindness:** BPTT split by block. Fix: single 2-block BPTT window (180 trials × 25 steps).
3. **Attractor bootstrap:** Need context to lick, need licking to get context gradient. Fix: ctx loss.
4. **Gradient variance (v1 diagnosis):** batch=1 → oscillation, no upward trend. Fix: batch=4, LR↓, λ_ctx↑.

---

## 8. Currently Running

| Run | PID | Checkpoint | Step | d'_inter | Status |
|-----|-----|-----------|------|----------|--------|
| Integrator+Coarse+CTX v1 | 15420 | checkpoints_integrator_coarse_ctx/ | ~2050, Stage 3 | 0.285 peak, osc | running (approaching max_steps) |
| Integrator+Coarse+2block | 11748 | checkpoints_integrator_coarse_2block/ | ~3900, Stage 4 | ≈0 | running (failed, forced Stage 4) |
| **IntegratorCoarse v2** | **17970** | **checkpoints_integrator_coarse_v2/** | ~550, Stage 3 | osc | running — mid-curriculum (γ≈0.70) |
| **Explicit CTX ablation** | **17989** | **checkpoints_explicit_ctx/** | **DONE** | **2.74** | **✅ COMPLETED — ceiling benchmark** |

### KEY FINDING: Explicit context ablation succeeded

```
Stage 3 learning curve (explicit ctx, batch=4, bptt=2-block):
  step 350: d'_inter=0.48, pass=0
  step 400: d'_inter=0.64, pass=0
  step 450: d'_inter=0.94, pass=4
  step 500: d'_inter=1.00, pass=5
  step 550: d'_inter=1.35, pass=5
  step 600: d'_inter=1.65, pass=5
  step 650: d'_inter=1.57, pass=5
  step 700: d'_inter=1.92, pass=9
  step 750: d'_inter=2.32, pass=10 → ADVANCED
Stage 4: advanced at step 1000 (d'_inter=2.74). DONE.
```

**Interpretation:** The architecture is correct and the behavioral loss IS sufficient to drive Stage 3 advancement when context is available. The ONLY remaining bottleneck is the context inference mechanism — learning to map reward feedback to a context belief that drives behavior.

---

## 9. New Experiments (2026-04-07)

### Exp A: IntegratorCoarse v2 (most promising)
**Script:** `training/train_integrator_coarse_v2.py`
**Key changes vs v1 (integrator_coarse_ctx):**
- batch_size: 1 → **4** (4× variance reduction)
- LR Stage 3: 1e-3 → **5e-4** (more stable)
- λ_ctx: 0.2 → **0.5** (stronger context supervision)
- max_sr Stage 3: 1.0 → **1.2** (allows slight SR > 1 for attractor)
- **γ curriculum: 0.50→0.90 over 500 steps** (c≈0.97 early → near-perfect signal; anneals to 0.41)

**Why this should work:** At γ=0.50, c≈0.97 after 5 instruction trials. The network can learn to use channel 7 easily early in training, then the curriculum ensures generalization as γ rises to 0.90. 4× smaller gradient variance prevents the oscillation seen in v1.

### Exp B: Explicit context ablation (ceiling benchmark)
**Script:** `training/train_explicit_ctx.py`
**Design:** Channel 7 = ±1.0 ground truth context (not learned from rewards)
**Purpose:**
- If SUCCEEDS: bottleneck confirmed as context *learning*; architecture/loss are fine
- If FAILS: fundamental problem with behavioral loss or architecture (independent of context inference)

---

## 10. Next Steps (if current batch fails)

1. **Fixed γ=0.50 (no annealing):** If γ curriculum helps early but not late, try fixed low γ.
2. **λ_ctx=1.0:** Make context the dominant training objective for Stage 3.
3. **Two-area model:** Separate integrator→readout module, dedicated gradient path.

---

## 11. Code Structure

```
models/rnn.py                    ✅ DynamicRoutingRNN (dt=20ms)
models/rnn_gru.py                ✅ GatedDynamicRoutingRNN
models/rnn_integrator.py         ✅ IntegratorRNN (8 inputs, standard dt)
models/rnn_coarse.py             ✅ DynamicRoutingRNNCoarse (dt=100ms)
models/rnn_integrator_coarse.py  ✅ IntegratorRNNCoarse (dt=100ms + integrator)

tasks/dynamic_routing.py         ✅ Standard constants (dt=20ms)
tasks/dynamic_routing_coarse.py  ✅ Coarse constants (dt=100ms)
tasks/session.py                 ✅ Session generator (standard)
tasks/session_coarse.py          ✅ Session generator (coarse)
tasks/curriculum.py              ✅ Stage manager

training/train_rnn.py                       ✅ Vanilla + teacher forcing (v25)
training/train_integrator_coarse.py         ✅ Coarse + integrator (1-block BPTT)
training/train_integrator_coarse_2block.py  ✅ Coarse + integrator + 2-block BPTT
training/train_integrator_coarse_ctx.py     ✅ + ctx loss (v1, oscillating)
training/train_integrator_coarse_v2.py      ✅ v2: batch=4, γ curriculum, λ_ctx=0.5
training/train_explicit_ctx.py              ✅ Ceiling ablation (explicit context)
```

## 12. Still To Do

- [ ] Stage 3 advancement (d'_intra > 1.5 AND d'_inter > 1.5, ≥4/6 blocks, 2 evals)
- [ ] Stage 4 fine-tuning
- [ ] Validation plots (hit/FA, block transitions, PCA)
- [ ] Phase 2: latent circuit model
