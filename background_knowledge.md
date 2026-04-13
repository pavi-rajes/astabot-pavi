# Background Knowledge for AstaBot Run

## Conda Environment
**CRITICAL**: Use the conda environment `latent_circuit` for ALL Python execution.
All Python scripts must be run with:
```
conda run -n latent_circuit python <script.py>
```
or equivalently activate the environment first. PyTorch 2.10.0 is confirmed available in this environment (CPU only, no CUDA).

## Project Goal (from mission.md)
1. Implement a RNN model trained on the dynamic routing task as specified in the task spec.
2. Implementation of the Dynamic Routing task should match as close to experiment as possible.
3. RNN architecture should infer context from reward cues (no explicit context input).

## Experiment Details (from EXPERIMENT.md)

### Trial structure
- Pre-trial interval: 1.5 s + truncated exponential (mean 1 s, max 6 s). For RNN: omit ITI.
- Quiescent interval: t = -1.5 to 0 s (last 1.5 s of pre-trial). If lick occurs, extend pre-trial.
- Stimulus onset: t = 0 s
- Response window: t = 0.1 to 0.5 s (a lick here = "response")
- Post-response window: t = 1 to 4 s (+ 3 s timeout for false alarms in Stages 1–3)
- Total ITI (onset to onset): variable, 5.5 to 12 s

### Rewards
- 3 to 5 µL water (variable due to solenoid valve timing)
- Contingent reward upon lick in response window to rewarded stimulus

### Instruction trials
- Presented at start of block and after 10 consecutive miss trials
- `late_autorewards` variant (from 2023-08-09): non-contingent reward scheduled at end of response window; if mouse responds within window, scheduled reward cancelled and contingent reward given immediately

### Block structure (Stages 3 & 4)
- 6 blocks per session, ~10 min each, median 90 trials/block
- Rewarded target alternates each block; counterbalanced across sessions
- 5 instruction trials at block start
- After instruction trials: stimuli shuffled (sub-blocks of 20)
- Catch trials: p = 0.1
- Extra instruction trial after 10 consecutive miss trials

### Stimuli
- Visual: square grating, 50° diameter, 0.04 cycles/°, 2 cycles/s
  - vis1 (target): vertical grating, rightward motion
  - vis2 (non-target): horizontal grating, downward motion
- Auditory: AM noise, bandpass 2–20 kHz, 5 ms Hanning window, 68 dB
  - aud1 (target): 12 Hz
  - aud2 (non-target): 70 Hz

## Implementation Guidelines (from General-guidelines.md)
1. Do not train RNN and latent circuit jointly — RNN first, then latent circuit on frozen data.
2. Q orthonormality must be strictly maintained (prefer Cayley retraction).
3. Latent circuit integrates its own dynamics forward — receives task inputs u, not RNN states.
4. Dale's law: enforce sign constraint at every forward pass, not just initialization.
5. Noise: same magnitude in RNN and latent circuit (sigma = 0.15).
6. Node identity in latent circuit enforced through w_in/w_out sparsity, not w_rec constraints.

## Scope for This Run
Focus on **Phase 1 only**: train the ground-truth RNN through the full curriculum (Stages 0–4).
- Train a small ensemble (e.g., 3–5 RNNs) to validate the pipeline end-to-end
- Save trained model checkpoints
- Run validation checks (hit/FA rates, block transition dynamics, context inference speed)
- Generate plots of training curves and behavioral performance

Phase 2 (latent circuit fitting) and Phase 3 (validation analyses) can follow once Phase 1 is confirmed working.

## Previous Run Context
A prior AstaBot run implemented these modules (already exist in the project directory):
- `utils/metrics.py`, `utils/plotting.py`
- `tasks/dynamic_routing.py`, `tasks/session.py`, `tasks/curriculum.py`
- `models/rnn.py`

These were implemented but NOT executed (PyTorch was unavailable). Review and reuse/fix these files rather than rewriting from scratch. The training script (`training/train_rnn.py`) was not implemented.
