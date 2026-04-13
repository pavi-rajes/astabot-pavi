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

## 5. Architecture

- **Standard:** DynamicRoutingRNN, N=100, tau=100ms, dt=20ms, alpha=0.2, sigma=0.15
- **Coarse:** DynamicRoutingRNNCoarse, N=100, tau=500ms, dt=100ms, alpha=0.2, sigma=0.15
- **W_rec init:** Spectral radius = 0.8
- **Python:** conda env `latent_circuit` (PyTorch, CPU only)

## 6. Completed: Stages 0-2

**Seed 42 (checkpoints_v24/seed_42/)** — standard dt=20ms:
- Stage 1 advanced at step 200 (d'_intra=4.65)
- Stage 2 advanced at step 300 (d'_intra=4.65)

## 7. Stage 3 — Complete Failure History

**Core problem:** All linear RNN variants fail to maintain context across trials.

| Approach | Max d'_inter | Verdict |
|----------|-------------|---------|
| v9-v10: BPTT variants | 0.00 | Context decays to 0 in 1 trial |
| v11-v18: Rank-1 attractor | 0.00 | SR clip destroyed attractor |
| v19-v24: SR scaling | ±0.18 osc | Misaligned eigenvector; TBPTT boundary |
| v25: Teacher forcing (fade 1000) | 0.22 peak → 0 | No attractor forms |
| GRU alone | 0.00 | Gate stays at init, no training signal |
| Integrator γ=0.99 (step 1250) | ±0.05 osc | c≈0.05 after 5 instruction trials (too weak) |
| Integrator γ=0.90 (step 1200) | ±0.06 osc | c≈0.41 but W_in[:,7]≈0 — no gradient foothold |
| **Coarse dt=100ms (step 4600)** | **0.19 peak, then 0** | **60% context/trial but still oscillating ~0** |
| GRU + Teacher Forcing (step 1650, hint=0.33) | 0.00 | Running — no signal yet |

### Why coarse dt also failed

Even with 60% context per trial, the context decays across the 90-trial block:
- After 5 instr trials: 8% remains (0.98^125)
- After 10 trials: 0.7% — effectively zero
- The network still can't distinguish VIS1 vs AUD1 blocks for >10 trials post-instruction
- d'_inter oscillates ±0.2, never converges

**Fundamental diagnosis:** Linear RNN dynamics CANNOT sustain context. The only solution is:
1. **Nonlinear attractor dynamics** (bistable fixed points that latch to VIS1 vs AUD1 context)
2. **Full 2-block BPTT** (gradient spans both blocks so network sees context contrast)
3. **Explicit context input** (ablation baseline)

The SR ≤ 1.0 constraint prevents attractor formation (bistable dynamics require SR > 1 locally).

---

## 8. Currently Running

| Run | PID | Checkpoint | Step | d'_inter | Status |
|-----|-----|-----------|------|----------|--------|
| Integrator γ=0.99 | 96719 | checkpoints_integrator/ | ~1250, Stage 3 | ±0.05 | running (likely failed) |
| Integrator γ=0.90 | 205 | checkpoints_integrator_g90/ | ~1200, Stage 3 | ±0.06 | running (likely failed) |
| GRU + TF (fade=2000) | 96728 | checkpoints_gru_teacher/ | ~1650, Stage 3, hint=0.33 | 0.00 | running — last hope of current batch |
| Coarse dt=100ms | 3309 | checkpoints_coarse/ | ~4600, Stage 4 | ±0.02 | running (failed, forced to Stage 4) |

---

## 9. Next Steps — New Direction Required

### Option A: Full 2-block BPTT (most promising, minimal change)
Use a single BPTT window spanning BOTH blocks (180 trials):
- Standard dt: 180 × 125 = 22,500 steps (slow but feasible)
- Coarse dt: 180 × 25 = 4,500 steps (fast — ~2 min per step)

The gradient flows from Block 2 instruction trials ALL THE WAY BACK to Block 1 activity.
Network sees: "what you do in Block 1 causally affects Block 2" → learns inter-block context.
**Recommended: coarse dt + 2-block single BPTT window.**

### Option B: Allow SR > 1 (attractor formation)
- Remove SR ≤ 1.0 constraint; allow SR up to 1.5
- Use L2 regularization to prevent explosion
- Bistable attractors require SR > 1 locally in context direction
- Risk: training instability

### Option C: Winner-take-all context module
- Add 2 dedicated context units with mutual inhibition
- Soft winner-take-all creates bistable dynamics without SR > 1 globally
- Cleanest biological interpretation

### Option D: Explicit context input (ablation baseline)
- Inject VIS1=+1/AUD1=-1 as a new input channel
- Guaranteed to work; establishes performance ceiling
- Useful for debugging whether the task itself is solvable

**Recommended immediate action:** Implement Option A (coarse dt + 2-block BPTT) as it requires only a flag change in `train_coarse.py`.

---

## 10. Code Structure

```
models/rnn.py                  ✅ DynamicRoutingRNN (dt=20ms)
models/rnn_gru.py              ✅ GatedDynamicRoutingRNN
models/rnn_integrator.py       ✅ IntegratorRNN (8 inputs)
models/rnn_coarse.py           ✅ DynamicRoutingRNNCoarse (dt=100ms)

tasks/dynamic_routing.py       ✅ Standard constants (dt=20ms)
tasks/dynamic_routing_coarse.py ✅ Coarse constants (dt=100ms)
tasks/session.py               ✅ Session generator (standard)
tasks/session_coarse.py        ✅ Session generator (coarse)
tasks/curriculum.py            ✅ Stage manager

training/train_rnn.py          ✅ Vanilla + teacher forcing (v25)
training/train_integrator.py   ✅ IntegratorRNN
training/train_gru_teacher.py  ✅ GRU + teacher forcing
training/train_coarse.py       ✅ Coarse-dt RNN (dt=100ms)
```

## 11. Still To Do

- [ ] Stage 3 advancement (d'_intra > 1.5 AND d'_inter > 1.5, ≥4/6 blocks, 2 evals)
- [ ] Stage 4 fine-tuning
- [ ] Validation plots (hit/FA, block transitions, PCA)
- [ ] Phase 2: latent circuit model
