# Research State: Latent Circuit Inference for Dynamic Routing Task

## 1. Research Question & Scope

**Primary Goal:** Implement and train a ground-truth RNN (Phase 1) on a dynamic routing task where context is inferred from reward feedback (not explicitly cued).

**Scope:** Phase 1 only — train RNN through 4-stage curriculum (Stages 0→4).

## 2. Operational Definitions

- **Context belief:** Network's internal representation of rewarded target, inferred from reward+lick.
- **d' (d-prime):** d' = Z(hit_rate) - Z(FA_rate), clipped to ±4.65.
- **Instruction trial:** Block start trial with late_autoreward (fires reward regardless of lick).
- **BPTT window:** 15 trials for Stages 1-2; 200 trials (full 2-block session) for Stage 3.

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

**Seed 42 confirmed results:**

| step | stage | loss  | d'_intra | advanced |
|------|-------|-------|----------|----------|
| 300  | 1     | 0.010 | 4.653    | ✅ Stage 1 |
| 500  | 2     | 0.009 | 4.653    | ✅ Stage 2 |

W_rec spectral properties after Stage 2:
- **Spectral radius: 0.902** (→ effective eigenvalue 0.980 per timestep)
- After 1 trial (125 steps): context signal decays to 0.980^125 = **0.8%** of original

### IN PROGRESS: Stage 3 — Context Attractor + Full-Session BPTT (v22)

**Root cause of d'_inter=0** (confirmed across v9-v20):
1. Context signals decay as 0.98^T per step with default SR=0.90
2. After 1 trial (125 steps), context decays to 0.8% — essentially zero
3. BPTT windows see zero context → gradients cancel → network learns context-invariant strategy

**Complete history of approaches:**

| Version | Approach | d'_inter | Notes |
|---------|----------|----------|-------|
| v9  | 90-trial BPTT | 0.00 | Context decays to 0 within 1 trial |
| v10 | 180-trial single-window BPTT | 0.00 | 5× slower, same result |
| v11 | Rank-1 attractor (inter-modal direction) | — | v12 issue discovered |
| v12 | Rank-1 in VIS1-AUD1 direction | d'_intra dropped | Wrong direction, disrupted discrimination |
| v13 | Rank-1 in reward-orthogonal direction | 0.00 | Direction doesn't encode VIS1 vs AUD1 |
| v14/v15 | Intra-modal orthogonalized rank-1, max_sr=4.0/3.0 | NaN crash | Operator norm too high |
| v16/v17 | Rank-1 + spectral radius clip to 1.0 | ~0.98 eff_eig | SR clip destroys attractor (1.0679→1.0000 scales λ from 0.999→0.934) |
| v18b | Rank-1 + pre-clip attempt | Same issue | pre-clip safe_sr=0.985 > current SR=0.902, so no pre-clip happened |
| v19 | **SR scaling** to SR=0.995 (eff_eig=0.999) + 90-trial windows | ±0.18 oscillation | Context signal persists! But d'_inter oscillates near 0 |
| v20 | SR scaling to SR=0.9995 (eff_eig=0.9999) + 90-trial windows | ±0.18 oscillation | Longer half-life didn't help with TBPTT |
| v21 | SR scaling + **200-trial single window** (full gradient flow) | -0.04 then loss explosion | Step 700: loss 0.316→2.016, instability at lr=1e-3 |
| **v22** | SR scaling + 200-trial window + **lr=1e-4** + grad_clip=0.5 | 🔄 running | Should stabilize training |

**Key insight from v19-v21:**
- v19/v20 with TBPTT (block-aligned cut): gradient within each block but NO gradient through the block boundary → the network can learn to USE context in Block 2 but Block 1 cannot learn to ENCODE better context
- v21 with full-session BPTT: gradient flows through both blocks, but instability at lr=1e-3 caused loss explosion at step 700
- v22 fix: lr=1e-4 + grad_clip=0.5 for Stage 3 to stabilize the longer BPTT

**Why v22 should work:**
- Context persistence: eff_eig=0.9999, 0.9999^11250 = 32% retained after 90 trials
- Full gradient flow: 200-trial window spans both blocks, enabling Block 1 context encoding to be trained
- Stable training: lr=1e-4 (10× smaller) + grad_clip=0.5 prevents catastrophic parameter updates
- Gradient signal: from Block 2 (90 trials downstream) back to Block 1 instruction trials; decay = 0.9999^11250 ≈ 10% (vs 0 without attractor)

**Current run:** v22 started ~17:55, PID 67972/1882 (Cygwin), expected Stage 3 eval in ~90 min

## 7. Open Questions & Confusions

1. **Will reduced lr be sufficient for Stage 3 learning?** With lr=1e-4, learning may be very slow. 5000 max_steps might not be enough. Confidence for v22 working: 55%.

2. **Is the dominant W_rec eigenvector actually encoding VIS1 vs AUD1 context?** After Stage 2 training (AUD1 vs AUD2 discrimination), the dominant eigenvector might not distinguish VIS1 vs AUD1. Need to check eigendecomposition of Stage 2 W_rec.

3. **Does BPTT with closed-loop session data give meaningful gradients?** The BPTT recomputes the forward pass with stored inputs (u_trial from closed-loop), but the closed-loop feedback (lick/reward channels) was already fixed during session generation. The gradient may be inconsistent with the actual closed-loop dynamics.

4. **Will teacher forcing be needed?** This is the nuclear option if v22 fails — add an explicit context cue input for early Stage 3 training steps, then fade it out.

## 8. Implementation Progress

### Code Status

```
models/rnn.py          ✅ Auto-reward bug fixed
training/train_rnn.py  ✅ SR scaling attractor + full-session BPTT + lr/grad_clip fix
tasks/session.py       ✅ Fixed block structure
tasks/curriculum.py    ✅ lr=1e-4 for Stage 3
checkpoints_v22/       🔄 Running
```

### Current Parameters (v22)
```
python training/train_rnn.py --n_ensemble 1 --n_units 100 --max_steps 5000
  --eval_every 100 --n_eval_sessions 3 --batch_size 1
  --bptt_trials 15 --max_bptt_windows 8 --train_n_blocks 2
  --outdir checkpoints_v22 --seed_start 42
```
Stage 3 settings (overrides above):
- bptt_trials: 200 (full 2-block session)
- max_bptt_windows: 1
- lr: 1e-4 (reduced from 1e-3)
- grad_clip: 0.5 (reduced from 1.0)
- SR clip: spectral radius ≤ 1.0 (eigenvalue-based)

### Next Steps

1. **Wait for v22 Stage 3 eval** (~90 min from now)
2. **If d'_inter > 0 and stable:** continue training, wait for advancement criterion
3. **If loss stable but d'_inter stays at 0:** Try explicit context diagnosis — log the projection of hidden state at block transitions onto the dominant eigenvector to check if context IS being encoded
4. **If v22 fails:** Try teacher forcing (add explicit context input channel for early Stage 3 training)

### Completed
- [x] All core modules implemented
- [x] Eight bugs fixed (v1-v9)
- [x] Stages 0-2 all working cleanly
- [x] Stage 3 root cause diagnosed: vanishing context gradient
- [x] Context attractor (SR scaling) preserves eff_eig=0.9999 correctly
- [x] Full-session BPTT implemented (200-trial window)

### Still To Do
- [ ] Stage 3 advancement (d'_intra > 1.5 AND d'_inter > 1.5 for ≥4/6 blocks, 2 consecutive evals)
- [ ] Stage 4 fine-tuning
- [ ] Validation plots (hit/FA rates, block transitions, PCA)
- [ ] Phase 2: latent circuit model
