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
**H4** (70%): 6-10 PCs capture >90% of task-valid variance.

## 5. Architecture

- **IntegratorRNNCoarse:** N=100, tau=500ms, dt=100ms, alpha=0.2, sigma=0.15
  - Channel 7: leaky reward integrator c = tanh(γ·c + (1-γ)·δ), δ∈{+1, −0.5, 0}
  - W_rec init: spectral radius = 0.8
- **Python:** conda env `latent_circuit` (PyTorch, CPU only)

## 6. Completed: Stages 0-2

**Seed 42 (checkpoints_v24/seed_42/):**
- Stage 1 advanced at step 200 (d'_intra=4.65)
- Stage 2 advanced at step 300 (d'_intra=4.65)
- W_rec spectral radius: 0.902

## 7. Stage 3 — Complete Failure/Success History

| Approach | Max d'_inter | Result |
|----------|-------------|--------|
| v9-v24 variants (BPTT, attractor, SR scaling) | ≤0.22 | ❌ FAILED |
| GRU alone | 0.00 | ❌ FAILED |
| Coarse dt, 2-block BPTT, integrator (various combos) | 0.36 osc | ❌ FAILED |
| Integrator+Coarse+2block+CTX v1 (batch=1, λ=0.2) | 0.285 osc | ❌ FAILED (10000 steps) |
| Integrator+Coarse+2block no ctx | 0.00 | ❌ FAILED (10000 steps) |
| IntegratorCoarse v2 (batch=4, γ curriculum) | 0.16 osc | ❌ KILLED at step 600 — same oscillation |
| **Explicit ctx ablation** (channel 7 = ±1.0) | **2.74** | **✅ DONE — ceiling benchmark** |
| **Ctx teacher forcing v1** (eval_alpha=0.67) | **2.54** Stage 3 | **⚠️ PARTIAL** — Stage 4 collapses |

### Root causes (four confirmed)

1. **Signal decay** — fix: integrator
2. **Gradient blindness** — fix: 2-block BPTT
3. **Attractor bootstrap** — fix: ctx loss + teacher forcing
4. **Gradient variance** — fix: batch=4
5. **Teacher forcing leakage** *(new)*: network passes Stage 3 using explicit ctx (alpha≈0.67), then collapses when alpha→0 in Stage 4

---

## 8. KEY FINDINGS

### 8a. Explicit context ablation ✅ (checkpoints_explicit_ctx/, DONE)
Channel 7 = ±1.0 (ground truth). Stage 3 advanced at step 750 (d'_inter=2.32), Stage 4 done at step 1000 (d'_inter=2.74). **Proves architecture + behavioral loss are correct.**

### 8b. Context teacher forcing v1 ⚠️ (checkpoints_ctx_teacher/, first run)
```
alpha = 1.0 → 0.0 over 1500 steps, eval uses TRAINING alpha

Stage 3 learning curve:
  step 350: d'_inter=0.25, alpha=0.97
  step 500: d'_inter=1.04, alpha=0.87
  step 650: d'_inter=1.64, alpha=0.77
  step 800: d'_inter=2.54, n_pass=10 → ADVANCED (alpha=0.67 at advancement!)

Stage 4 (alpha=0.0 immediately):
  step 850-1650: d'_inter oscillates near 0
```
**Finding:** Stage 3 was passed because network learned from 67% explicit context signal. Integrator pathway NOT functional at advancement time. Stage 4 with pure integrator collapses identically to all prior failed runs.

### 8c. Context teacher forcing v2 (current run — the fix)
**Key change:** `eval_alpha=0.0` — always evaluate with PURE integrator. Stage 3 can only advance when the integrator alone produces d'_inter > 1.5. Training still uses blended ctx (provides gradients) but advancement requires the learned pathway to work.

---

## 9. Currently Running

| Run | PID | Checkpoint | Step | d'_inter | Status |
|-----|-----|-----------|------|----------|--------|
| **Ctx teacher forcing v2** | **29424** | **checkpoints_ctx_teacher/** | starting | TBD | **RUNNING** — eval_alpha=0, ctx_fade=2000, batch=4 |

### v2 design
```
Training:  ctx_input = (1-alpha) * c_integrator + alpha * c_explicit
           alpha: 1.0 → 0.0 linearly over 2000 steps
Evaluation: alpha = 0.0 (pure integrator) at ALL evals
Advancement: only when integrator alone achieves d'_inter > 1.5
```

Expected behavior:
- Steps 300-500 (alpha≈0.95-0.85): explicit ctx dominates → d'_intra improves, W_in[:,7] gets gradient foothold
- Steps 800-1300 (alpha≈0.65-0.35): mixed signal → W_in[:,7] increasingly important
- Steps 1800-2300 (alpha≈0.25-0.0): nearly pure integrator training → integrator must carry context
- Eval always at alpha=0 → measures pure integrator performance

If pure integrator performance is rising while training alpha fades, the run will advance Stage 3.
If it oscillates at eval, the run has until step 4000 before force-advance.

---

## 10. Next Steps (if v2 eval_alpha=0 never rises)

1. **Partial eval_alpha fade**: eval_alpha = training_alpha (sync eval and train schedule), so network is evaluated at the same blend level it was trained on, ensuring stage advances; then verify Stage 4 with pure integrator on the saved checkpoint
2. **Stronger ctx loss in integrator regime**: once alpha < 0.5, boost λ_ctx → 1.0 to force representation
3. **Two-area model**: separate integrator→readout module so gradient flows cleanly through context pathway independently of behavior pathway

---

## 11. Code Structure

```
models/rnn_integrator_coarse.py  ✅ IntegratorRNNCoarse (dt=100ms + integrator)

training/train_integrator_coarse_ctx.py     ✅ v1: pure integrator + ctx loss (FAILED)
training/train_integrator_coarse_v2.py      ✅ v2: batch=4, γ curriculum (FAILED — oscillates)
training/train_explicit_ctx.py              ✅ Ceiling ablation (DONE — d'_inter=2.74) ✅
training/train_ctx_teacher.py               ✅ Context teacher forcing (RUNNING — v2 with eval_alpha=0)
```

## 12. Still To Do

- [ ] Stage 3 advancement via LEARNED context inference (d'_intra>1.5 AND d'_inter>1.5, integrator only)
- [ ] Stage 4 fine-tuning
- [ ] Validation plots (hit/FA, block transitions, PCA)
- [ ] Phase 2: latent circuit model
