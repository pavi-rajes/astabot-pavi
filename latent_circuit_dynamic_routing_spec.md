
# Latent Circuit Inference for a Dynamic Routing Task

## Project Overview

Reimplement the latent circuit inference framework from Langdon & Engel (2025, Nature Neuroscience) and adapt it to a dynamic routing task with lick/no-lick responses. The project has three phases:

1. **Phase 1:** Train a ground-truth RNN on the dynamic routing task
2. **Phase 2:** Fit a latent circuit model to the trained RNN's activity
3. **Phase 3:** Validation analyses (connectivity conjugation, perturbations, decoder comparison)

Use PyTorch throughout. Reference implementation: `https://github.com/engellab/latentcircuit` (classes `Net` and `LatentNet`).

---

## Phase 1: Ground-Truth RNN

### Task: Dynamic Routing

A block-based context-dependent sensory detection task. The network receives one of four discrete stimuli (or a catch blank) on each trial. A block rule determines which of two target stimuli is currently rewarded. The network must lick to the rewarded target and withhold licking to all other stimuli. Critically, the block rule is never explicitly cued — the network must infer it from reward feedback delivered after correct lick responses.

#### Stimuli (discrete, no coherence gradient)
- **vis1:** Vertical grating, rightward motion (visual target)
- **vis2:** Horizontal grating, downward motion (visual non-target)
- **aud1:** 12 Hz amplitude-modulated noise (auditory target)
- **aud2:** 70 Hz amplitude-modulated noise (auditory non-target)
- **catch:** Blank grey screen, no auditory stimulus

On each trial, exactly one stimulus is presented (or catch). There is no coherence parameter — stimuli are all-or-nothing.

#### Block structure (Stages 3 & 4)
- 6 blocks per session, ~90 trials per block
- In each block, one of the two targets (vis1 or aud1) is rewarded; the other target and both non-targets are unrewarded
- The rewarded target alternates each block (vis1 → aud1 → vis1 → ...)
- The first block's rewarded target is counterbalanced across sessions
- 5 instruction trials at the start of each block present the newly rewarded target with non-contingent (free) reward
- After instruction trials, the 4 stimuli are presented in shuffled order (even sampling in sub-blocks of 20 trials)
- Catch trials are interspersed with probability 0.1

#### Trial structure (temporal sequence within one trial)

The experimental trial includes a variable ITI (1.5s + exponential), a 1.5s quiescent window, stimulus onset at t=0 (stimulus stays on for 0.5s), a response window (0.1–1.0s post-stim), and a post-response period (1–4s). For the RNN, we retain the quiescent period, stimulus, and response window, but omit the ITI and post-response period. Trials are concatenated directly — the quiescent period of trial k+1 immediately follows the response window of trial k. Reward and lick feedback are delivered within the response window upon a correct lick.

```
|-- quiescent (1.5 s, 75 steps) --|-- stim on (0.5 s, 25 steps) --|-- resp window closes (t=1.0 s) --|
                                  t=0                    t=0.5       t=1.0
                                   |---- stimulus on ----|
                                        |---------- response window (0.1–1.0 s, 45 steps) ----------|

Total per trial: 2.5 s = 125 timesteps at dt = 20 ms
Per block (~90 trials): ~11,250 timesteps
Per session (6 blocks): ~67,500 timesteps
```

- **Quiescent period (t = −1.5 to 0 s, steps 0–74):** All sensory inputs are zero. The network must maintain its context belief from prior trials and keep output at baseline (no lick). No loss is computed during this period — the network is free to have internal dynamics but only the response window output is supervised.
- **Stimulus onset (t = 0 s, step 75):** One sensory input channel goes to 1.0. The stimulus stays on for 500 ms (25 steps), turning off at t = 0.5 s (step 100). For catch trials, no sensory channel fires.
- **Response window (t = 0.1–1.0 s, steps 80–124):** The behavioral loss is computed here. A lick (z > 0.5) at any point within this window counts as a response. If the stimulus is the currently rewarded target, the target output is z = 1 (lick). Otherwise z = 0 (no lick). Note: the stimulus is on for the first 400 ms of the response window (steps 80–99), then off for the remaining 500 ms (steps 100–124). The network can respond either while the stimulus is visible or shortly after it disappears.
- **Reward delivery:** In the real experiment, a lick to the rewarded target during the response window immediately triggers a contingent reward (3–5 µL of water; volume varies due to solenoid valve variability). For the RNN, reward is delivered as soon as z > 0.5 occurs during the response window AND the stimulus is the currently rewarded target. The reward input channel pulses for 2 timesteps (40 ms) starting at the timestep of the first lick. If the network does not lick within the response window, or licks to a non-rewarded stimulus, no reward is delivered.

#### Inputs (u): 7 channels, all time-varying

| Channel | Name           | Description                                                                 |
|---------|----------------|-----------------------------------------------------------------------------|
| `u[0]`  | vis1 stimulus  | 1.0 for 500 ms from stimulus onset (steps 75–99) if vis1, else 0.0         |
| `u[1]`  | vis2 stimulus  | 1.0 for 500 ms from stimulus onset (steps 75–99) if vis2, else 0.0         |
| `u[2]`  | aud1 stimulus  | 1.0 for 500 ms from stimulus onset (steps 75–99) if aud1, else 0.0         |
| `u[3]`  | aud2 stimulus  | 1.0 for 500 ms from stimulus onset (steps 75–99) if aud2, else 0.0         |
| `u[4]`  | reward         | 1.0 for 2 timesteps (40 ms) immediately upon lick to rewarded target       |
| `u[5]`  | own lick       | 1.0 for 2 timesteps (40 ms) immediately upon any lick (z > 0.5)           |
| `u[6]`  | trial phase    | 1.0 during response window (steps 80–124), else 0.0                        |

**Catch trials:** On catch trials, none of the stimulus channels (u[0]–u[3]) fire — the network simply sees silence during the stimulus period. There is no explicit catch channel because the mouse receives no stimulus on catch trials; the absence of input is the signal. The network should learn to withhold licking when no stimulus is presented.

**Critical design note — no explicit context input.** There is no channel telling the network which target is currently rewarded. The network must infer the current block rule from the conjunction of reward (`u[4]`) and own-lick (`u[5]`) feedback across trials, and maintain this belief in its recurrent hidden state across the quiescent period.

**Reward logic:** On regular trials, reward is delivered immediately when a lick occurs during the response window (z > 0.5 at any timestep) AND the presented stimulus is the currently rewarded target (vis1 in vis1-rewarded blocks, aud1 in aud1-rewarded blocks). The reward magnitude in the real experiment is 3–5 µL of water (variable due to solenoid valve timing); for the RNN, the reward input channel is binary (0 or 1). The own-lick channel (`u[5]`) fires on any lick regardless of whether the stimulus was rewarded — this allows the network to distinguish "licked and got reward" (context-confirming) from "licked and got nothing" (context-disconfirming).

Exception: On instruction trials, reward is delivered non-contingently (regardless of whether the network licked). For `late_autoreward` instruction trials (the default), a non-contingent reward is scheduled for the end of the response window, but if the network licks during the response window, the scheduled reward is cancelled and a contingent reward is delivered immediately instead.

#### Output (z): 1 channel (scalar)

- `z(t) = σ(w_out · y(t))` — sigmoid of a linear readout from the hidden state
- Interpretation: lick probability (go/no-go)
- Target during response window: z = 1.0 if the stimulus is the currently rewarded target, z = 0.0 otherwise
- Target outside response window: z = 0.0 (no lick)
- Loss is computed only during the response window

### RNN Architecture

```python
class DynamicRoutingRNN(nn.Module):
    """
    Ground-truth RNN for the dynamic routing task.
    
    Dynamics (continuous-time, Euler discretized):
        y_{t+1} = (1 - alpha) * y_t + alpha * f(W_rec @ y_t + W_in @ u_t + b_rec) + sigma * sqrt(2 * alpha) * noise_t
    
    where:
        alpha = dt / tau
        f = ReLU
        noise_t ~ N(0, I)
    
    Output:
        z_t = sigmoid(w_out @ y_t + b_out)
    
    Hidden state y persists across all trials within a session, including across block boundaries.
    Hidden state is reset to zero only at the start of each session (not at block boundaries).
    The network must learn to update its context belief at block transitions using reward feedback.
    """
```

**Hyperparameters:**
- `N = 200` units
- `tau = 100 ms` (time constant)
- `dt = 20 ms` (discretization step, so alpha = dt/tau = 0.2)
- `sigma_rec = 0.15` (recurrent noise magnitude)
- `n_inputs = 7`
- `n_outputs = 1`

**Weight constraints:**
- `W_in`: element-wise non-negative (`W_in = |W_in_raw|`)
- `W_rec`: unconstrained (Dale's law optional for first implementation; if used, follow the `W_rec = |W_raw| @ D` pattern with 80% E / 20% I split)
- `w_out`: unconstrained (1 × N readout vector)

**Multi-trial training:** The network processes entire blocks as single training sequences. A block of K trials (~90) has K × 125 = ~11,250 timesteps. A full session (6 blocks) is ~67,500 timesteps. The hidden state carries over between trials within a block but gradients are truncated using windowed BPTT (see Training section).

### Training Curriculum

The curriculum mirrors the actual mouse training protocol. The RNN advances through stages automatically based on performance metrics computed on held-out evaluation batches.

#### Stage 0 — Auto-rewards (output bias shaping)

**Purpose:** Teach the network that licking leads to reward.

**Structure:**
- 150 trials per training session
- Only the rewarded target stimulus is presented (no distractors)
- Reward is delivered on every trial regardless of network output (non-contingent)
- No quiescent period enforcement, no catch trials

**Inputs active:** 1 stimulus channel + reward + own lick + trial phase
**Loss:** BCE on z(t) during the response window. Target is always z = 1 (lick).
**Advancement:** Fixed number of training steps (e.g., 500 steps). No performance gate — this stage just initializes the output bias.

#### Stage 1 — Visual discrimination

**Purpose:** Learn to lick to vis1, withhold licking to vis2.

**Structure:**
- Trials present either vis1 (rewarded) or vis2 (unrewarded)
- Context is constant (vis1 is always rewarded) — no block switching
- After a false alarm (z > 0.5 on a vis2 trial), insert 3 extra vis2 trials before the next normal trial (timeout analog)
- No auditory stimuli, no catch trials

**Inputs active:** vis1, vis2 channels + reward + own lick + trial phase (5 of 7 channels)
**Loss:** BCE on z(t) during response window. Extra loss weight (×3) on false-alarm trials (analog of timeout punishment).
**Advancement criterion:** d' > 1.5 for 2 consecutive evaluation checkpoints, where:
```
d' = Z(hit_rate_vis1) - Z(false_alarm_rate_vis2)
```
**Regression criterion:** If hits < 10 per evaluation session for 2 consecutive checkpoints, revert to Stage 0.

#### Stage 2 — Auditory discrimination

**Purpose:** Learn to lick to aud1, withhold licking to aud2.

**Structure:**
- Identical to Stage 1 but with aud1 (rewarded) and aud2 (unrewarded)
- Visual weights continue to receive gradients but visual stimuli are never presented, so they are effectively frozen
- After a false alarm on aud2, insert 3 extra aud2 trials

**Inputs active:** aud1, aud2 channels + reward + own lick + trial phase
**Loss:** Same as Stage 1 with ×3 FA weight.
**Advancement criterion:** d' > 1.5 for 2 consecutive evaluation checkpoints.

#### Stage 3 — Block switching with timeout

**Purpose:** Learn to use reward feedback to track which target is currently rewarded, and switch behavior accordingly.

**Structure:**
- 6 blocks per session, ~90 trials per block
- Rewarded target alternates each block (vis1 → aud1 → vis1 → ...)
- First block rewarded target is counterbalanced across training sessions
- 5 instruction trials at the start of each block (rewarded target, non-contingent reward)
- After instruction trials: all 4 stimuli + catch trials (10%), shuffled in sub-blocks of 20
- After a false alarm, insert 3 extra unrewarded-stimulus trials (timeout analog)

**Inputs active:** All 7 channels
**Hidden state:** Persists across trials within the entire session (no reset at block boundaries). The network must learn to update its context belief when instruction trials signal a block change.
**Loss:** BCE on z(t) during response window, with ×3 weight on false-alarm trials (timeout analog modeling the 3-second dark-screen timeout in Stages 1–3).
**BPTT:** Truncated backpropagation through time with a window of 15–20 trials. Hidden state is carried forward across the full block, but gradients are detached at window boundaries. This keeps memory requirements manageable (~1,875–2,500 timesteps per window) while allowing the network to learn from reward feedback within the gradient window.

**Advancement criterion:** Compute the following for each block in an evaluation session:
- Intra-modal d': hit rate on rewarded target vs FA rate on same-modality non-target
  - For vis1-rewarded blocks: d'_intra = Z(hit_vis1) - Z(FA_vis2)
  - For aud1-rewarded blocks: d'_intra = Z(hit_aud1) - Z(FA_aud2)
- Inter-modal d': hit rate on rewarded target vs FA rate on other-modality target
  - For vis1-rewarded blocks: d'_inter = Z(hit_vis1) - Z(FA_aud1)
  - For aud1-rewarded blocks: d'_inter = Z(hit_aud1) - Z(FA_vis1)

Pass criterion: both d'_intra > 1.5 and d'_inter > 1.5 for at least 4 of 6 blocks, across 2 consecutive evaluation sessions.

#### Stage 4 — Final task (no timeout, matches ephys conditions)

**Purpose:** Fine-tune to the exact conditions used during neural recordings.

**Structure:** Same as Stage 3 except:
- Remove the ×3 false-alarm loss weight (no timeout)
- Reduce learning rate by 10× (e.g., 1e-4 instead of 1e-3)
- Continue training until convergence (stable performance across 5 consecutive evaluation sessions)

**This is the model used for all subsequent analyses.** Record its activity for Phase 2.

### Training Hyperparameters

- **Optimizer:** Adam
- **Learning rate:** 1e-3 for Stages 0–3, 1e-4 for Stage 4
- **Gradient clipping:** Max norm = 1.0 (essential for BPTT over long sequences)
- **Regularization:**
  - L2 on firing rates: `λ_rate * (1/NT) Σ ||y_t||^2`, with `λ_rate = 1e-3`
  - L2 on weights: `λ_weight * (||W_rec||_F^2 + ||W_in||_F^2)`, with `λ_weight = 1e-4`
- **Batch size:** Number of independent sessions per gradient step. Start with 16; reduce if memory constrained.
- **Evaluation frequency:** Every 100 training steps, run 5 evaluation sessions (no noise, no gradient) and compute advancement metrics.
- **Ensemble:** Train 50 RNNs with different random seeds (reduced from 200 for tractability given the multi-trial curriculum; expand later if needed).

### Validation Checks After Training

1. **Hit rate and FA rate by block type:** For vis1-rewarded blocks, plot hit rate (P(lick|vis1)) and FA rates (P(lick|vis2), P(lick|aud1), P(lick|aud2), P(lick|catch)). Repeat for aud1-rewarded blocks with appropriate relabeling. Hits should be high (~0.8+), FAs should be low (~0.2 or less).

2. **Block transition dynamics:** Plot P(lick|rewarded target) across trial position within a block, separately for the first few blocks. Performance should start low at the block boundary, rise sharply during/after instruction trials, and plateau.

3. **Context inference speed:** After how many trials into a new block does d' exceed 1.0? This characterizes how quickly the network uses reward feedback to update its context belief.

4. **Heterogeneous mixed selectivity:** Visualize single-unit responses across conditions. Units should show mixed tuning to stimulus identity, block context, and lick/no-lick — not clean categorical selectivity.

5. **PCA on trial-averaged responses:** Confirm that ~6–10 PCs capture >90% of task-related variance (averaged across stimulus × context conditions).

6. **No leakage test:** On catch trials, the network's lick probability should be near zero in both block types (it has no stimulus to respond to).

---

## Phase 2: Latent Circuit Model

### Architecture

```python
class LatentCircuit(nn.Module):
    """
    Low-dimensional recurrent circuit fitted to RNN activity.
    
    Embedding:      y_hat = Q @ x        (Q is N × n, orthonormal columns)
    Dynamics:       x_{t+1} = (1 - alpha) * x_t + alpha * f(w_rec @ x_t + w_in @ u_t) + sigma * sqrt(2*alpha) * noise_t
    Output:         z_hat = sigmoid(w_out @ x_t)
    
    where f = ReLU, and alpha, sigma match the ground-truth RNN.
    """
```

**Latent circuit dimensions:**
- `n = 9` latent nodes with prescribed identity:

| Node | Identity             | Driven by input channel     | Reads out to         |
|------|----------------------|-----------------------------|----------------------|
| 0    | vis1 sensory         | `u[0]` (vis1 stimulus)      | —                    |
| 1    | vis2 sensory         | `u[1]` (vis2 stimulus)      | —                    |
| 2    | aud1 sensory         | `u[2]` (aud1 stimulus)      | —                    |
| 3    | aud2 sensory         | `u[3]` (aud2 stimulus)      | —                    |
| 4    | reward processing    | `u[4]` (reward)             | —                    |
| 5    | lick efference copy  | `u[5]` (own lick)           | —                    |
| 6    | trial phase / timing | `u[6]` (trial phase)        | —                    |
| 7    | context belief       | — (no direct input)         | —                    |
| 8    | lick decision        | —                           | z (lick probability) |

Nodes 7 (context belief) and 8 (lick decision) receive no direct external input — their activity arises entirely from recurrent interactions. This means the context belief node must be driven by reward and lick-efference nodes through `w_rec`, and the lick decision node must be driven by sensory and context nodes through `w_rec`. The structure of `w_rec` connecting these nodes is the circuit mechanism we want to infer.

**Node identity is enforced through input/output sparsity:**

`w_in` (9 × 7 matrix) has the following sparsity pattern:
```
         u0   u1   u2   u3   u4   u5   u6
node 0 [ *    0    0    0    0    0    0  ]   vis1 stim → vis1 node
node 1 [ 0    *    0    0    0    0    0  ]   vis2 stim → vis2 node
node 2 [ 0    0    *    0    0    0    0  ]   aud1 stim → aud1 node
node 3 [ 0    0    0    *    0    0    0  ]   aud2 stim → aud2 node
node 4 [ 0    0    0    0    *    0    0  ]   reward → reward node
node 5 [ 0    0    0    0    0    *    0  ]   own lick → lick eff. node
node 6 [ 0    0    0    0    0    0    *  ]   trial phase → timing node
node 7 [ 0    0    0    0    0    0    0  ]   context: no direct input
node 8 [ 0    0    0    0    0    0    0  ]   lick decision: no direct input
```
(* = learnable scalar, 0 = fixed at zero)

`w_out` (1 × 9 vector) has the following sparsity pattern:
```
       node0  node1  node2  node3  node4  node5  node6  node7  node8
w_out [  0      0      0      0      0      0      0      0      *   ]
```
Only node 8 (lick decision) reads out to the behavioral output.

`w_rec` (9 × 9 matrix) is **fully unconstrained** — all 81 entries are learnable. This is the connectivity we want to infer. The scientifically interesting connections include:
- `w_rec[7, 4]` and `w_rec[7, 5]`: reward and lick-efference → context belief (how context is updated)
- `w_rec[8, 0]` and `w_rec[8, 2]`: vis1 and aud1 sensory → lick decision (sensory-motor mapping)
- `w_rec[8, 7]`: context belief → lick decision (context-dependent gating)
- `w_rec[0, 7]` and `w_rec[2, 7]`: context belief → sensory nodes (suppression of irrelevant sensory responses)

### Parameters to optimize

- `Q` (N × n): embedding matrix, must satisfy `Q^T Q = I_n`
- `w_rec` (n × n): recurrent connectivity, unconstrained
- `w_in` (n × n_inputs): input connectivity, sparse (see above)
- `w_out` (n_outputs × n): output connectivity, sparse (see above)

**Orthonormality constraint on Q:**
- `Q` must satisfy `Q^T Q = I_n` (columns are orthonormal)
- Implementation (in order of preference):
  1. **Cayley retraction:** After each gradient step on Q, retract back to the Stiefel manifold using the Cayley transform
  2. **QR reparameterization:** Parameterize `Q = QR_factor(Q_raw)`, extract Q from QR decomposition of a learnable N × n matrix
  3. **Penalty method:** Add `λ_Q * ||Q^T Q - I||_F^2` to the loss. Simpler but less exact.

### Loss Function

```
L = L_neural + λ_behav * L_behavior

L_neural   = (1 / (K * T * N)) * Σ_{k,t} ||y_{k,t} - Q @ x_{k,t}||^2
L_behavior = (1 / (K * T_resp))  * Σ_{k, t ∈ response_window} BCE(z_{k,t}, z_hat_{k,t})
```

where:
- `y_{k,t}` = ground-truth RNN activity on trial k at time t (frozen data, not differentiated through)
- `x_{k,t}` = latent circuit state, obtained by forward-integrating the latent circuit dynamics with the same inputs `u_{k,t}`
- `z_{k,t}` = ground-truth RNN output on trial k at time t
- `z_hat_{k,t} = sigmoid(w_out @ x_{k,t})` = latent circuit's predicted output
- K = number of trials, T = total timesteps, T_resp = timesteps in response window, N = number of RNN units
- `λ_behav` = weighting on behavioral loss (tune so both terms contribute roughly equally)

**Important:** The latent circuit dynamics are integrated forward in time during fitting. Gradients flow through the latent circuit's ODE integration (backprop through time on the latent circuit), NOT through the ground-truth RNN — the RNN activity is frozen data.

**Multi-trial fitting:** Since the RNN maintains hidden state across trials, the latent circuit must also be run across multi-trial sequences to capture the cross-trial context dynamics. Run the latent circuit on the same trial sequences used during RNN training (blocks of ~90 trials), carrying the latent state `x` across trials. Use the same truncated BPTT windowing as RNN training (15–20 trials per gradient window).

### Fitting Procedure

For each trained RNN in the ensemble:
1. Record activity `y_{k,t}` and outputs `z_{k,t}` across ~20 evaluation sessions (train/test split 80/20). Each session = 6 blocks × ~90 trials. Include instruction trials and block transitions.
2. Initialize 50 latent circuits with random parameters (random orthonormal Q, small random w_rec, diagonal w_in, sparse w_out)
3. Optimize each with Adam (lr = 1e-3) for ~5000 epochs, with gradient clipping (max norm = 1.0)
4. Evaluate fit quality on test data:
   ```
   r² = 1 - Σ||y - Q @ x||² / Σ||y - mean(y)||²
   ```
5. Select top 10 latent circuits by test r² (these are "converged solutions")
6. Check that converged solutions have highly correlated `w_rec` (Pearson r > 0.90 expected across the top solutions for the same RNN)

---

## Phase 3: Validation Analyses

### Analysis 1: Connectivity conjugation

Validate that the inferred latent circuit mechanism exists in the actual RNN connectivity:

```python
# Extract the ground-truth RNN weight matrices
W_rec_rnn = rnn.get_W_rec()   # N × N
W_in_rnn = rnn.get_W_in()     # N × n_inputs

# Conjugate with inferred embedding Q (from latent circuit)
w_rec_conjugated = Q.T @ W_rec_rnn @ Q   # n × n
w_in_conjugated = Q.T @ W_in_rnn          # n × n_inputs

# Compare with inferred latent connectivity
r_rec = pearson_correlation(w_rec_conjugated.flatten(), w_rec_inferred.flatten())
r_in = pearson_correlation(w_in_conjugated.flatten(), w_in_inferred.flatten())
# Expect r_rec > 0.85, r_in > 0.85
```

This tests whether the latent subspace defined by Q is approximately an invariant subspace of the full RNN connectivity.

### Analysis 2: Activity projections showing dynamic routing

Project RNN activity onto the columns of Q to visualize latent node dynamics:

```python
# Context belief axis: Q[:, 7]
context_proj = Q[:, 7].T @ y
# Should show sustained separation between vis1-rewarded and aud1-rewarded blocks
# Should flip at block transitions (after instruction trials)

# Vis1 sensory axis: Q[:, 0]
vis1_proj = Q[:, 0].T @ y
# Should show transient activation when vis1 is presented
# Key test: compare vis1 response amplitude in vis1-rewarded vs aud1-rewarded blocks
# If inhibitory gating: vis1 response should be SUPPRESSED in aud1-rewarded blocks
# If gain modulation: vis1 response should be AMPLIFIED in vis1-rewarded blocks

# Aud1 sensory axis: Q[:, 2]
aud1_proj = Q[:, 2].T @ y
# Symmetric test: aud1 response amplitude in aud1-rewarded vs vis1-rewarded blocks

# Lick decision axis: Q[:, 8]
lick_proj = Q[:, 8].T @ y
# Should separate by lick vs no-lick during response window
# Should show ramping activity toward lick threshold on go trials

# Reward processing axis: Q[:, 4]
reward_proj = Q[:, 4].T @ y
# Should show transient response after reward delivery
# Should be followed by context belief update (check cross-correlation with context node)
```

Plot these as time courses across the full trial, grouped by stimulus × block context condition. The suppression or amplification pattern of irrelevant sensory representations is the key signature distinguishing different gating mechanisms.

### Analysis 3: Perturbation experiments

#### Perturbation A: Disable context-dependent gating of sensory responses

If the latent circuit shows that the context belief node inhibits irrelevant sensory nodes, remove those connections and verify the predicted behavioral effect.

```python
# Example: suppose w_rec[0, 7] < 0 (context node 7 inhibits vis1 node 0)
# and w_rec[2, 7] > 0 (context node 7 excites aud1 node 2)
# This would mean: when context = "aud1-rewarded", vis1 is suppressed

# In latent circuit: zero out the context→sensory connections
w_rec_perturbed = w_rec.clone()
w_rec_perturbed[0, 7] = 0   # remove context → vis1 connection
w_rec_perturbed[1, 7] = 0   # remove context → vis2 connection

# In RNN: apply rank-one perturbation (Eq. 6 from Langdon & Engel)
delta_W = -w_rec[0, 7] * torch.outer(Q[:, 0], Q[:, 7]) \
        - w_rec[1, 7] * torch.outer(Q[:, 1], Q[:, 7])
W_rec_perturbed = W_rec_rnn + delta_W

# Run both perturbed models on test sessions
# Expected: in aud1-rewarded blocks, the network should now respond to vis1
# (irrelevant visual stimulus is no longer suppressed)
# Measure: increase in P(lick|vis1) on aud1-rewarded blocks
```

#### Perturbation B: Disable reward → context update pathway

Remove the connection from the reward-processing node to the context belief node.

```python
# In latent circuit: zero out w_rec[7, 4] (reward node 4 → context node 7)
# In RNN: delta_W = -w_rec[7, 4] * torch.outer(Q[:, 7], Q[:, 4])

# Expected: the network should fail to update its context belief after block transitions
# Measure: d' should remain low for many trials after a block switch
# (the network is stuck on the old block rule because it can't process reward feedback)
```

#### Perturbation C: Disable sensory → lick-decision mapping for one modality

Remove the connection from a specific sensory node to the lick decision node.

```python
# In latent circuit: zero out w_rec[8, 2] (aud1 sensory node 2 → lick decision node 8)
# In RNN: delta_W = -w_rec[8, 2] * torch.outer(Q[:, 8], Q[:, 2])

# Expected: on aud1-rewarded blocks, hit rate on aud1 should drop dramatically
# while vis1-rewarded block performance should be unaffected
```

### Analysis 4: Comparison with correlation-based decoder

Fit a standard linear regression decoder D that maps neural activity to task variables:

```python
# Task variables: stimulus identity (4-dim one-hot), block context (1-dim), lick (1-dim)
# D: N × 6, fitted by least-squares: D = Y @ X_task.T @ (X_task @ X_task.T)^{-1}

# Compare stimulation along D axes vs Q axes:
# 1. Add perturbation along D_context axis during baseline period → measure effect on P(lick)
# 2. Add perturbation along Q[:, 7] (context belief axis) during baseline period → measure effect on P(lick)
# Q axes should produce larger, more predictable behavioral effects than D axes
# because Q axes are constrained to lie in a dynamically invariant subspace
```

### Analysis 5: Block transition dynamics in the latent circuit

This analysis is unique to the dynamic routing task (not present in Langdon & Engel's Mante task analysis) because of the reward-based context inference.

```python
# 1. Run the latent circuit across a full session (6 blocks)
# 2. Track the context belief node (x[7]) across time
# 3. Align to block transitions and average

# Expected trajectory of context node:
# - Sustained at one level during block k
# - At block boundary: begins to shift only AFTER the first instruction trial delivers reward
# - Reaches new steady state within 5–10 trials
# - The transition speed characterizes the "context update gain" of the circuit

# 4. Compare transition dynamics between the latent circuit and the ground-truth RNN
#    (project RNN activity onto Q[:, 7] and compare with latent x[7])
```

### Analysis 6: Space of circuit solutions

For the ensemble of 50 trained RNNs:

```python
# 1. Fit latent circuits to each RNN (take the best-fitting solution per RNN)
# 2. Collect all w_rec matrices (9 × 9 = 81-dimensional vectors)
# 3. PCA on the collection of flattened w_rec vectors
# 4. Visualize in PC1-PC2 space
# 5. Check for qualitative motifs shared across solutions:
#    - Do all solutions show inhibitory context → irrelevant sensory connections?
#    - Do all solutions show excitatory sensory → lick-decision connections?
#    - Do all solutions show reward → context update pathway?
# 6. Quantify: fraction of solutions with sign-consistent connections for each edge
```

---

## Implementation Notes

### File Structure

```
latent_circuit_dynamic_routing/
├── models/
│   ├── rnn.py                 # DynamicRoutingRNN class
│   ├── latent_circuit.py      # LatentCircuit class
│   └── stiefel.py             # Orthonormality utilities for Q (Cayley retraction)
├── tasks/
│   ├── dynamic_routing.py     # Trial generation, stimulus sampling, target computation
│   ├── session.py             # Session/block/trial sequence generation
│   └── curriculum.py          # Stage advancement logic, d' computation
├── training/
│   ├── train_rnn.py           # Phase 1: train ensemble of RNNs through curriculum
│   └── fit_latent.py          # Phase 2: fit latent circuits to trained RNNs
├── analysis/
│   ├── connectivity.py        # Analysis 1: conjugation validation
│   ├── projections.py         # Analysis 2: activity projections
│   ├── perturbations.py       # Analysis 3: perturbation experiments
│   ├── decoder_compare.py     # Analysis 4: Q vs D comparison
│   ├── block_transitions.py   # Analysis 5: context dynamics at block switches
│   └── solution_space.py      # Analysis 6: ensemble PCA
├── utils/
│   ├── plotting.py            # Hit/FA rates, transition curves, connectivity matrices
│   └── metrics.py             # d', r², correlation coefficients
└── notebooks/
    └── tutorial.ipynb          # End-to-end walkthrough
```

### Key Implementation Pitfalls to Avoid

1. **Do not train the RNN and latent circuit jointly.** The RNN is trained first (Phase 1), its activity is recorded, then the latent circuit is fit to that recorded activity as frozen data (Phase 2).

2. **Hidden state must persist across trials within a block.** This is the most important difference from the Langdon & Engel reference implementation. The RNN (and the latent circuit during fitting) processes sequences of ~90 trials without resetting the hidden state. If you reset between trials, the network cannot maintain a context belief.

3. **Reward feedback is closed-loop during RNN training.** The reward input `u[4]` depends on the network's own output `z(t)`. During training, threshold z at 0.5 at each timestep during the response window. On the first timestep where z > 0.5, fire the own-lick channel `u[5]`; if the stimulus is also the currently rewarded target, fire the reward channel `u[4]` simultaneously. Both pulse for 2 timesteps (40 ms). Use a straight-through estimator if gradients need to flow through the thresholding operation.

4. **Q orthonormality must be strictly maintained.** If using the penalty method, use a large λ_Q (≥ 10). Cayley retraction is preferred.

5. **The latent circuit integrates its own dynamics forward.** It does NOT receive RNN states as input — it receives only the same task inputs u and must reconstruct the RNN activity through its own recurrent dynamics.

6. **Noise:** Both the RNN (during training) and the latent circuit (during fitting) should use recurrent noise with the same statistical magnitude (σ = 0.15). The specific noise realization will differ.

7. **Node identity in the latent circuit is enforced through the sparsity pattern of w_in and w_out,** not through any constraint on w_rec. The recurrent connectivity is free to take any values — that is what reveals the mechanism.

8. **Truncated BPTT window size matters.** If the window is too short (< 5 trials), the network cannot learn from reward feedback (which arrives at the end of each trial and must influence the next trial's behavior). If too long (> 30 trials), memory and vanishing gradients become problematic. Start with 15–20 trials.

9. **Instruction trials must be implemented correctly.** On instruction trials, the reward channel fires regardless of the network's output. This is the primary signal that tells the network the block rule has changed. If instruction trials are omitted or implemented incorrectly, the network will never learn to switch contexts.

10. **Curriculum stage transitions should be logged.** Record the training step, stage, and advancement metrics at each evaluation checkpoint. This allows post-hoc analysis of learning dynamics and troubleshooting if the network gets stuck at a stage.