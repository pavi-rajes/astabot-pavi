# Research State: Latent Circuit Inference for Dynamic Routing Task

## 1. Research Question & Scope

**Primary Goal:** Implement and train a ground-truth RNN (Phase 1) on a dynamic routing task where context (which stimulus is rewarded) is never explicitly cued — the network must infer context from reward feedback across trials.

**Secondary Goals:**
- Phase 2: Fit a low-dimensional latent circuit model to the trained RNN's activity
- Phase 3: Validate the circuit mechanism through conjugation, perturbations, and decoder comparison

**Scope for current work:** Phase 1 only — train the ground-truth RNN through the full 4-stage curriculum.

## 2. Operational Definitions

- **Context belief:** The network's internal representation of which target (vis1 or aud1) is currently rewarded, inferred from reward+lick feedback across trials.
- **d' (d-prime):** Signal detection measure. d' = Z(hit_rate) - Z(false_alarm_rate), where Z is the inverse normal CDF.
- **Instruction trial:** A trial at the start of a new block presenting the rewarded target with non-contingent reward (late_autoreward variant).
- **Hit rate:** P(lick | rewarded target stimulus) during response window.
- **False alarm (FA) rate:** P(lick | non-rewarded stimulus) during response window.
- **BPTT window:** Truncated backpropagation-through-time using 15–20 trials per window.

## 3. Related Work

- **Langdon & Engel (2025, Nature Neuroscience):** Latent circuit inference framework. Reference implementation: https://github.com/engellab/latentcircuit (classes `Net` and `LatentNet`).
- **Dynamic routing task:** Experimental design described in EXPERIMENT.md and latent_circuit_dynamic_routing_spec.md.

## 4. Hypotheses

**H1** (90% confidence): The trained RNN will show context-dependent suppression of irrelevant sensory representations — responses to the non-rewarded target should be smaller in amplitude when that modality is not rewarded.

**H2** (85% confidence): The latent circuit will reveal a reward-to-context-belief pathway (w_rec[7,4] or w_rec[7,5] will be large) — the mechanism by which the network updates its block rule.

**H3** (80% confidence): The context-belief node will show sustained, block-locked dynamics that flip at block transitions with a lag proportional to the number of instruction trials + a few regular trials.

**H4** (70% confidence): PCA on trial-averaged RNN activity will show 6–10 PCs capturing >90% of task-related variance.

## 5. Experimental Designs

### Phase 1: RNN Training Pipeline
- **Architecture:** DynamicRoutingRNN, N=200 units, tau=100ms, dt=20ms, alpha=0.2, sigma=0.15
- **Inputs:** 7 channels (4 stimuli + reward + own lick + trial phase)
- **Output:** 1 channel (sigmoid lick probability)
- **Curriculum:** 4 stages (Stage 0: auto-rewards → Stage 1: visual discrimination → Stage 2: auditory discrimination → Stage 3: block switching with timeout → Stage 4: no timeout)
- **Ensemble:** Start with 3–5 RNNs; expand to 50 for final analyses

### Phase 2: Latent Circuit Fitting (future)
- n=9 latent nodes with prescribed identities
- Fit Q (embedding), w_rec, w_in (sparse), w_out (sparse)
- Loss = L_neural + λ_behav * L_behavior

## 6. Results Summary

### Status: NOT STARTED — No code implemented yet.

**Current state:** Project directory contains only specification files:
- `latent_circuit_dynamic_routing_spec.md` — Full specification
- `experiment_task_details/EXPERIMENT.md` — Experiment fact sheet
- `background_knowledge.md` — Notes from prior context
- `mission.md` — Project goals

**No Python source files exist.** Previous background_knowledge.md mentions prior runs that were supposed to have created modules, but they are absent from the directory.

## 7. Open Questions & Confusions

1. **BPTT vs. memory:** With 90 trials × 125 timesteps = 11,250 steps per block, and a window of 15–20 trials (1,875–2,500 steps), we need to be careful about detaching the hidden state at window boundaries while still carrying it forward.

2. **Closed-loop reward feedback:** The reward input u[4] depends on the network's own output z(t). During training, this requires an online thresholding step inside the forward pass — a straight-through estimator may be needed for gradients to flow through.

3. **Instruction trial timing:** The `late_autoreward` variant means non-contingent reward is given at the END of the response window if the network didn't lick. This needs careful implementation to distinguish from contingent reward.

4. **PyTorch availability:** background_knowledge.md confirms PyTorch 2.10.0 in `latent_circuit` conda environment (CPU only, no CUDA). All code should be CPU-compatible.

## 8. Implementation Progress

### Next Task: Implement core modules (Phase 1)

**Priority order:**
1. Create directory structure
2. `tasks/dynamic_routing.py` — trial generation, stimuli, targets
3. `tasks/session.py` — block/session sequence generation
4. `tasks/curriculum.py` — stage advancement logic, d' computation
5. `models/rnn.py` — DynamicRoutingRNN class
6. `utils/metrics.py` — d', r², correlation coefficients
7. `utils/plotting.py` — hit/FA rates, transition curves
8. `training/train_rnn.py` — full curriculum training loop

**Completed:** None
**In Progress:** None (starting this iteration)
