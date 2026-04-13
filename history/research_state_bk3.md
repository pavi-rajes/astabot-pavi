# Research State: Latent Circuit Inference for Dynamic Routing Task

## 1. Research Question & Scope

**Primary Goal:** Implement and train a ground-truth RNN (Phase 1) on a dynamic routing task where context (which stimulus is rewarded) is never explicitly cued.

**Scope:** Phase 1 — train the ground-truth RNN through 4-stage curriculum (Stages 0→4).

## 2. Operational Definitions

- **Context belief:** Network's internal representation of which target is rewarded, inferred from reward+lick feedback.
- **d' (d-prime):** d' = Z(hit_rate) - Z(FA_rate).
- **Instruction trial:** Block start trial with non-contingent reward (late_autoreward).
- **BPTT window:** 15 trials per gradient window; only 8 randomly sampled per step.

## 3. Related Work

- **Langdon & Engel (2025, Nature Neuroscience):** Latent circuit inference framework.
- Reference: https://github.com/engellab/latentcircuit

## 4. Hypotheses

**H1** (90%): Trained RNN shows context-dependent suppression of irrelevant sensory representations.
**H2** (85%): Latent circuit reveals reward→context-belief pathway.
**H3** (80%): Context-belief node shows sustained, block-locked dynamics that flip at transitions.
**H4** (70%): 6–10 PCs capture >90% of task-related variance.

## 5. Experimental Designs

- **Architecture:** DynamicRoutingRNN, N=100 units (N=200 for final), tau=100ms, dt=20ms, sigma=0.15
- **Inputs:** 7 channels (4 stimuli + reward + own_lick + trial_phase)
- **Output:** 1 channel (sigmoid lick probability)
- **Training:** Adam, lr=1e-3, grad_clip=1.0, L2 rate (1e-3) + weight (1e-4)
- **Python env:** `/c/Users/peter/anaconda3/python.exe` (PyTorch 2.1.2+cpu)

## 6. Results Summary

### COMPLETED: Training infrastructure (Phase 1 code)

**All modules implemented and tested:**
- tasks/dynamic_routing.py, tasks/session.py, tasks/curriculum.py
- models/rnn.py, utils/metrics.py, utils/plotting.py
- training/train_rnn.py

### COMPLETED: Critical bug fixes this iteration

Three stability bugs were identified and fixed:

**Bug 1: W_rec spectral radius > 1 at initialization**
- Problem: Default `randn(N,N)/sqrt(N)` gives spectral radius ~1, which combined with positive external forcing (lick channel) causes hidden state to blow up to inf → NaN
- Fix: Initialize W_rec with spectral radius = 0.8 using `torch.linalg.eigvals` normalization

**Bug 2: Stage 0 saturates the network**
- Problem: Training ALL weights for 500 steps with target z=1 pushed the network to output z=1.0 constantly. Constant licking drove y→inf through the positive W_in weights.
- Fix: Stage 0 only trains w_out and b_out (output weights frozen for recurrent weights). Reduced to 100 steps.

**Bug 3: Noise in BPTT causes NaN**
- Problem: Adding recurrent noise (σ=0.15) during 1875-step BPTT windows caused hidden states to overflow if spectral radius was close to 1.
- Fix: No noise during BPTT forward pass. Noise is already captured via closed-loop session generation (pre-pass).

**Performance optimizations implemented:**
- Precomputed W_in@u for full trial in forward_trial_closedloop (before lick detection)
- Two-pass compute_loss: no-grad first pass to collect y_states, then BPTT on 8 random windows
- Random BPTT window subsampling (8 of ~14-42 windows per session)
- Stage 3+ uses 2 blocks per training session (not 6)

### IN PROGRESS: Training run (seed 0, N=100)

**Current status:** Training started at 01:24 (UTC), running in background (PID 1607)

| step | stage | loss | d'_intra | d'_inter |
|------|-------|------|----------|----------|
| 150  | 1     | 0.493 | 0.0     | 0.0      |

- Stage 0: 100 steps → b_out=0.048, W_rec spectral radius stable at 0.8
- Stage 1 step 150: loss=0.493 (down from ~0.86 at initialization). d'=0.0 (network hasn't learned to discriminate yet at step 50)
- Training speed: ~12s per step → very slow. 1000 stage 1 steps ≈ 3.3 hours

### Performance concern: Training is too slow

**Per-step timing breakdown:**
- Session generation (480 trials for Stage 1): ~5s
- BPTT (8 windows × 1875 steps no-grad + 8 windows × 1875 steps with-grad): ~3.5s
- Total: ~8-12s per step

**Still needed:** Further speedup. Options:
- Reduce Stage 1 session to 2 blocks (same as Stage 3) → ~2s per step
- Reduce eval_every to 100 steps to reduce evaluation overhead
- Or just wait for current run to complete (~3-4 hours)

## 7. Open Questions & Confusions

1. **Why is training so slow?** Each Stage 1 step uses 6 blocks × 80 trials = 480 trials. With `n_blocks_override=None` for stages < 3, the session uses the full 6-block default. Could be 3x faster if we use 2-block training sessions for all stages.

2. **d'=0 at step 50 for Stage 1**: The network should be able to learn visual discrimination quickly. Possible reasons: (a) loss is decreasing (0.86→0.49), so learning IS happening; (b) the threshold for d'>0 requires hit_rate > FA_rate, and we're not there yet after just 50 steps; (c) the closed-loop lick feedback may need more steps to stabilize.

3. **Will the network get stuck at intermediate stages?** The curriculum has both advancement AND regression criteria. With max_steps=1000 per stage, if Stage 3 doesn't advance, training caps out. May need to increase max_steps.

## 8. Implementation Progress

### Next Steps

**Immediate:**
1. Wait for current training run to complete (PID 1607, ~3-4 more hours for Stage 1)
2. Fix Stage 1 session length: use `train_n_blocks=2` for all stages, not just Stage 3+
3. Check evaluation results periodically

**After training completes:**
4. Run validation checks (hit/FA rates, block transitions)
5. Generate plots (training curves, behavioral performance)
6. If Stage 1-2 converge, run Stage 3 full training

**Completed:**
- [x] Directory structure
- [x] All core modules implemented
- [x] Smoke tests passed
- [x] Bug fixes: spectral radius, Stage 0 saturation, BPTT NaN
- [x] Training infrastructure (optimized)
- [ ] Training run in progress (Stage 1, step 150)
- [ ] Validation checks
- [ ] Phase 2: latent circuit

### File Status

```
tasks/dynamic_routing.py    ✅
tasks/session.py            ✅ (n_blocks_override added)
tasks/curriculum.py         ✅
models/rnn.py               ✅ (spectral radius init, optimized forward)
utils/metrics.py            ✅
utils/plotting.py           ✅
training/train_rnn.py       ✅ (bug fixes, optimizations)
checkpoints/seed_00/log.csv 📊 (1 eval so far: step 150)
```
