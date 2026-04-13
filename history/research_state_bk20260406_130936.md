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
- **Training:** 2 blocks per session, eval_every=50
- **Python:** conda env `latent_circuit` (PyTorch, CPU only)

## 6. Results Summary

### COMPLETED: Stages 0-2

**Seed 42 confirmed results (checkpoints_v24/seed_42/):**

| step | stage | loss  | d'_intra | advanced |
|------|-------|-------|----------|----------|
| 200  | 1     | 0.025 | 4.653    | ✅ Stage 1 |
| 300  | 2     | 0.027 | 4.653    | ✅ Stage 2 |

W_rec spectral properties after Stage 2:
- **Spectral radius: 0.902** (→ effective eigenvalue 0.980 per timestep)
- After 1 trial (125 steps): context signal decays to 0.980^125 = **0.8%** of original

---

### FAILED: Stage 3 — Complete History

**Root cause of d'_inter=0** (confirmed across v9–v25 + GRU alone):
Context signals decay as 0.98^T per step. After 1 trial (125 steps), context decays to 0.8%.
BPTT windows see zero context → gradients cancel → network learns context-invariant strategy.

| Version | Approach | d'_inter | Notes |
|---------|----------|----------|-------|
| v9  | 90-trial BPTT | 0.00 | Context decays to 0 within 1 trial |
| v10 | 180-trial single-window BPTT | 0.00 | 5× slower, same result |
| v11-v18 | Rank-1 attractor (various) | failed | SR clip destroyed attractor |
| v19-v20 | SR scaling (eff_eig=0.9999) + TBPTT | ±0.18 osc | TBPTT blocks inter-block gradients |
| v21 | SR scaling + 200-trial + lr=1e-3 | loss explosion | Unstable |
| v22 | SR scaling + 200-trial + lr=1e-4 | ~0 | eff_eig in context dir = 0.9392 not 0.9999 |
| v23-v24 | SR scaling + 2×90-trial TBPTT | ~0 | TBPTT blocks inter-block gradients |
| v25 | Teacher forcing (hint fades 1.0→0 over 1000 steps) | peaks 0.22 → collapses to 0 | Hint doesn't build attractor |
| GRU alone | GRU-style gating (N=100) | 0.00 | 88% signal retention but gate stays at init |
| GRU + TF (2000 steps) | GRU + teacher forcing, hint fades over 2000 steps | 0.00 so far (step 800, hint=0.75) | Running |
| Integrator γ=0.99 | Leaky reward integrator input (channel 7) | ±0.06 oscillating | γ too slow: c≈0.05 after 5 instr trials |
| **Integrator γ=0.90** | **Leaky reward integrator, faster decay** | 🔄 just started | γ=0.90: c≈0.41 after 5 instr trials |

**Critical insight (γ=0.99 failure):**
After 5 instruction trials: c ≈ 1-(0.99^5) ≈ 0.05 — too weak for network to distinguish contexts.
With γ=0.90: c ≈ 1-(0.90^5) ≈ 0.41 after 5 instruction trials → strong enough signal.

---

## 7. Currently Running

| Run | Script | Checkpoint dir | Key param | Status |
|-----|--------|---------------|-----------|--------|
| Integrator γ=0.99 | train_integrator.py | checkpoints_integrator/ | gamma=0.99 | 🔄 step ~850, d'_inter≈0 |
| GRU + Teacher Forcing | train_gru_teacher.py | checkpoints_gru_teacher/ | hint_fade=2000 | 🔄 step ~800, d'_inter=0, hint=0.75 |
| **Integrator γ=0.90** | **train_integrator.py** | **checkpoints_integrator_g90/** | **gamma=0.90** | **🔄 just started** |

---

## 8. Open Questions

1. **Will γ=0.90 integrator work?** c≈0.41 after 5 instruction trials should be enough for the RNN to learn context gating. Confidence: 70%.

2. **Will GRU + teacher forcing survive hint fade?** Hint still at 0.75 at step 800. The gate needs to learn by the time hint→0 (~step 2300). Confidence: 45%.

3. **If integrator works, does it violate the "context inferred from reward" constraint?** The integrator computes a running average of reward history — it's making the temporal integration explicit rather than implicit. The reward channel is already in the input; the integrator just makes it easier to use. Borderline — needs discussion.

4. **What gamma is optimal?** γ=0.90 (τ≈10 trials) vs γ=0.80 (τ≈5 trials). Too fast risks noise sensitivity; too slow fails to build context. 0.90 is the first test.

5. **Nonlinear attractor still needed for Phase 2?** Even if integrator works, the trained RNN's internal dynamics may not develop bistable fixed points needed for latent circuit analysis. May need to ablate the integrator after training.

---

## 9. Architecture Overview

### Current models
```
models/rnn.py              ✅ DynamicRoutingRNN (N inputs=7)
models/rnn_gru.py          ✅ GatedDynamicRoutingRNN (GRU-style gates)
models/rnn_integrator.py   ✅ IntegratorRNN (N inputs=8, channel 7 = leaky reward integrator)
```

### Training scripts
```
training/train_rnn.py         ✅ Vanilla RNN, teacher forcing (v25)
training/train_integrator.py  ✅ IntegratorRNN with leaky reward integrator
training/train_gru_teacher.py ✅ GRU + teacher forcing
```

### Integrator dynamics
```python
# Per trial, after outcome observed:
delta = +1.0  if reward delivered
        -0.5  if miss (rewarded target, no lick)
         0.0  otherwise
c = tanh(gamma * c + (1 - gamma) * delta)

# Injected as constant input on channel 7 throughout trial
# Resets to 0 at session start
```

---

## 10. Implementation Progress

### Completed
- [x] All core modules implemented (v1-v9, 8 bugs fixed)
- [x] Stages 0-2 confirmed working (seed 42)
- [x] Stage 3 root cause diagnosed: vanishing context gradient
- [x] SR scaling confirmed failed
- [x] Teacher forcing (v25) confirmed failed
- [x] GRU alone confirmed failed (gate doesn't learn without signal)
- [x] GRU + teacher forcing implemented and running
- [x] Leaky reward integrator implemented and running (γ=0.99 failing, γ=0.90 started)

### Still To Do
- [ ] Stage 3 advancement (d'_intra > 1.5 AND d'_inter > 1.5 for ≥4/6 blocks, 2 consecutive evals)
- [ ] Stage 4 fine-tuning
- [ ] Validation plots (hit/FA rates, block transitions, PCA)
- [ ] Phase 2: latent circuit model

### Next fallback if current runs fail
- Sweep γ values (0.80, 0.85, 0.90, 0.95) for integrator
- GRU + integrator combined
- Explicit context input (ablation baseline — guaranteed to work but not scientifically interesting)
