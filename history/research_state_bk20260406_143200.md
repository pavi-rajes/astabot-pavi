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

## 6. Completed: Stages 0-2

**Seed 42 (checkpoints_v24/seed_42/):**

| step | stage | loss  | d'_intra | status |
|------|-------|-------|----------|--------|
| 200  | 1     | 0.025 | 4.65     | ✅ Advanced |
| 300  | 2     | 0.027 | 4.65     | ✅ Advanced |

W_rec spectral radius after Stage 2: **0.902**
Context decay: eff_lambda = 0.980 → 0.980^125 = **0.8% per trial** (standard dt=20ms)

---

## 7. Stage 3 — All Attempted Approaches

| Version | Approach | d'_inter | Why failed |
|---------|----------|----------|------------|
| v9 | 90-trial BPTT | 0.00 | Context decays to 0 within 1 trial |
| v10 | 180-trial single window | 0.00 | Same, 5× slower |
| v11-v18 | Rank-1 attractor | failed | SR clip destroyed attractor |
| v19-v20 | SR scaling (eff_eig=0.9999) | ±0.18 osc | TBPTT blocks inter-block gradients |
| v21-v22 | SR scaling + 200-trial window | 0.00 | eff_eig in context dir only 0.9392 |
| v23-v24 | SR scaling + 2×90-trial TBPTT | 0.00 | TBPTT blocks inter-block gradients |
| v25 | Teacher forcing (hint→0 over 1000 steps) | peaks 0.22 → collapses | No attractor forms |
| GRU alone | GRU-style gates (N=100) | 0.00 | Gate stays at init=0.2; no training signal |

---

## 8. Currently Running

| Run | PID | Checkpoint | Key params | Step (as of 2026-04-06) | d'_inter |
|-----|-----|-----------|------------|------------------------|----------|
| Integrator γ=0.99 | 96719 | checkpoints_integrator/ | gamma=0.99 | ~1250, Stage 3 | ±0.05 (oscillating ~0) |
| GRU + Teacher Forcing | 96728 | checkpoints_gru_teacher/ | hint_fade=2000 | ~1200, Stage 3, hint=0.55 | 0.00 |
| Integrator γ=0.90 | 205 | checkpoints_integrator_g90/ | gamma=0.90 | ~700, Stage 3 | ±0.06 (oscillating ~0) |
| **Coarse dt=100ms** | **3309** | **checkpoints_coarse/** | **dt=100ms, tau=500ms** | **just started, Stage 3** | **TBD** |

Also running via Asta:
- `.asta/experiment/2026-04-06-coarse-dt-training/` (monitoring coarse run)

---

## 9. Why Coarse dt=100ms is Promising

With dt=100ms, tau=500ms (alpha=0.2 — same leak rate):
```
eff_lambda = 0.8 + 0.2 × 0.9 = 0.98 per STEP = 0.98 per 100ms
After 1 trial (25 steps): 0.98^25 = 60%  ← vs 0.8% at dt=20ms
After 5 instr trials:     0.98^125 = 8%  ← vs ~0%
```
**75× better context retention per trial** with no architectural changes.
Training is also **5× faster** (25 steps/trial vs 125).

---

## 10. Why Both Integrators Are Failing

γ=0.99: After 5 instruction trials, c ≈ 0.05 (too weak)
γ=0.90: After 5 instruction trials, c ≈ 0.41 (strong enough in principle)

But both show d'_inter ≈ 0. Hypothesis: W_in[:, 7] initialized to 0 at Stage 3 start.
The network must learn to use the integrator channel FROM SCRATCH while also trying to
maintain Stage 2 performance. The gradient for W_in[:, 7] may be too small initially
because nothing routes through channel 7.

**Potential fix:** Initialize W_in[:, 7] with small nonzero weights (e.g., 0.1 × mean(|W_in[:, 0:4]|))
to give the gradient a foothold.

---

## 11. Open Questions

1. Will coarse dt=100ms solve Stage 3? (60% retention per trial should be enough). Confidence: 60%.
2. Why isn't γ=0.90 integrator working? Is W_in[:, 7] ≈ 0 the issue?
3. Will GRU + TF survive hint fade (currently hint=0.55, d'_inter=0.00)?
4. If coarse dt works: does it change what Phase 2 latent circuit analysis looks like?

---

## 12. Code Structure

```
models/rnn.py                ✅ DynamicRoutingRNN (dt=20ms)
models/rnn_gru.py            ✅ GatedDynamicRoutingRNN
models/rnn_integrator.py     ✅ IntegratorRNN (8 inputs, channel 7 = leaky integrator)
models/rnn_coarse.py         ✅ DynamicRoutingRNNCoarse (dt=100ms, tau=500ms)

tasks/dynamic_routing.py     ✅ Standard constants (dt=20ms)
tasks/dynamic_routing_coarse.py ✅ Coarse constants (dt=100ms)
tasks/session.py             ✅ Session generator (standard)
tasks/session_coarse.py      ✅ Session generator (coarse)
tasks/curriculum.py          ✅ Stage manager

training/train_rnn.py        ✅ Vanilla + teacher forcing
training/train_integrator.py ✅ Integrator RNN
training/train_gru_teacher.py ✅ GRU + teacher forcing
training/train_coarse.py     ✅ Coarse-dt RNN
```

## 13. Still To Do

- [ ] Stage 3 advancement (d'_intra > 1.5 AND d'_inter > 1.5, ≥4/6 blocks, 2 evals)
- [ ] Stage 4 fine-tuning
- [ ] Validation plots (hit/FA, block transitions, PCA)
- [ ] Phase 2: latent circuit model

### Fallback plan if all current runs fail
1. Fix integrator W_in[:, 7] initialization (nonzero)
2. Coarse dt + integrator combined
3. Coarse dt + GRU
4. Explicit context input (ablation baseline — guaranteed to work)
