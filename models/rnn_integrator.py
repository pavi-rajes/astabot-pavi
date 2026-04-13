"""
IntegratorRNN: vanilla DynamicRoutingRNN augmented with a leaky reward integrator.

A leaky integrator accumulates reward history across trials and feeds it back
as an extra input channel (channel 7). This externalizes context maintenance
so the RNN only needs to learn to USE the context signal, not maintain it.

Integrator dynamics (per trial):
    delta = +1.0  if reward delivered
            -0.5  if miss (rewarded target, no lick)
             0.0  otherwise
    c_{t+1} = gamma * c_t + (1 - gamma) * delta

After 5 VIS1 instruction trials:  c → +1  (VIS1 context)
After block switch + AUD1 instr:   c → -1  (AUD1 context)

The RNN sees c_t as a constant input on channel 7 throughout each trial,
giving it a context belief signal that persists across the full session.

Architecture: identical to DynamicRoutingRNN but with n_inputs=8.
Stage 2 weights load cleanly — the extra W_in column (channel 7) initializes
near zero so behavior starts context-blind and gradient descent teaches usage.
"""

import torch
import torch.nn as nn
import numpy as np

from tasks.dynamic_routing import (
    RESP_START, RESP_END, STIM_ON, STIM_OFF, REWARD_DURATION,
    N_INPUTS, N_OUTPUTS, STEPS_PER_TRIAL
)

N_INPUTS_WITH_INTEGRATOR = N_INPUTS + 1  # 8


class IntegratorRNN(nn.Module):
    """
    Vanilla RNN with leaky reward integrator as extra input (channel 7).

    Identical to DynamicRoutingRNN except n_inputs=8.
    The integrator value is injected externally per trial via
    forward_trial_closedloop(integrator_value=c).
    """

    def __init__(self, n_units=100, n_outputs=1,
                 tau=0.100, dt=0.020, sigma_rec=0.15):
        super().__init__()
        n_inputs = N_INPUTS_WITH_INTEGRATOR  # 8
        self.N = n_units
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.tau = tau
        self.dt = dt
        self.alpha = dt / tau
        self.sigma_rec = sigma_rec

        self.W_in_raw = nn.Parameter(torch.randn(n_units, n_inputs) * 0.05)
        W_rec_init = torch.randn(n_units, n_units) / np.sqrt(n_units)
        sr = float(torch.linalg.eigvals(W_rec_init).abs().max())
        W_rec_init = W_rec_init * (0.8 / sr)
        self.W_rec = nn.Parameter(W_rec_init)
        self.b_rec = nn.Parameter(torch.zeros(n_units))
        self.w_out = nn.Parameter(torch.randn(n_outputs, n_units) * 0.1)
        self.b_out = nn.Parameter(torch.zeros(n_outputs))

    @property
    def W_in(self):
        return torch.abs(self.W_in_raw)

    def load_from_vanilla(self, state_dict):
        """
        Load Stage 2 weights from a DynamicRoutingRNN (7-input) checkpoint.
        The extra W_in column (channel 7 = integrator) initializes near zero.
        """
        own = self.state_dict()
        for k in ['W_rec', 'b_rec', 'w_out', 'b_out']:
            if k in state_dict:
                own[k].copy_(state_dict[k])
        # W_in_raw: copy first 7 columns, leave column 7 near zero
        if 'W_in_raw' in state_dict:
            own['W_in_raw'][:, :N_INPUTS].copy_(state_dict['W_in_raw'])
            own['W_in_raw'][:, N_INPUTS].fill_(0.0)
        self.load_state_dict(own)

    def forward_trial_closedloop(self, y0, stimulus_idx, rewarded_target_idx,
                                  instruction=False, add_noise=True, device=None,
                                  integrator_value=0.0):
        """
        Run one trial with integrator channel injected as a constant input.

        Extra parameter vs DynamicRoutingRNN:
            integrator_value : float
                Current integrator state c_t; constant throughout trial on channel 7.
        """
        if device is None:
            device = next(self.parameters()).device

        T = STEPS_PER_TRIAL
        is_rewarded = (stimulus_idx is not None) and (stimulus_idx == rewarded_target_idx)

        y = y0.to(device)

        u_trial = torch.zeros(T, self.n_inputs, device=device)
        if stimulus_idx is not None:
            u_trial[STIM_ON:STIM_OFF, stimulus_idx] = 1.0
        u_trial[RESP_START:RESP_END, 6] = 1.0          # trial phase
        u_trial[:, 7] = float(integrator_value)         # integrator — constant

        target = np.zeros(T, dtype=np.float32)
        if is_rewarded:
            target[RESP_START:RESP_END] = 1.0

        mask = np.zeros(T, dtype=bool)
        mask[RESP_START:RESP_END] = True

        noise_scale = self.sigma_rec * float(np.sqrt(2 * self.alpha)) if add_noise else 0.0

        with torch.no_grad():
            wu = u_trial @ self.W_in.T + self.b_rec   # (T, N)
            w_lick   = self.W_in[:, 5]
            w_reward = self.W_in[:, 4]

            y_state = y
            z_list = []
            lick_occurred = False
            reward_delivered = False

            auto_start = RESP_END - REWARD_DURATION

            for t in range(T):
                if instruction and not reward_delivered and t == auto_start:
                    wu[t:RESP_END] = wu[t:RESP_END] + w_reward.unsqueeze(0)
                    u_trial[t:RESP_END, 4] = 1.0
                    reward_delivered = True

                pre = y_state @ self.W_rec.T + wu[t]
                if add_noise:
                    pre = pre + noise_scale * torch.randn_like(pre)
                y_state = (1 - self.alpha) * y_state + self.alpha * torch.relu(pre)

                z_t = float(torch.sigmoid(y_state @ self.w_out.T + self.b_out).squeeze())
                z_list.append(z_t)

                if RESP_START <= t < RESP_END and not lick_occurred:
                    if z_t > 0.5:
                        lick_occurred = True
                        lick_end = min(t + REWARD_DURATION, T)
                        wu[t:lick_end] = wu[t:lick_end] + w_lick.unsqueeze(0)
                        u_trial[t:lick_end, 5] = 1.0
                        if is_rewarded:
                            wu[t:lick_end] = wu[t:lick_end] + w_reward.unsqueeze(0)
                            u_trial[t:lick_end, 4] = 1.0
                            reward_delivered = True

        z_seq = np.array(z_list, dtype=np.float32)
        u_np  = u_trial.cpu().numpy()

        return u_np, target, mask, lick_occurred, reward_delivered, z_seq, y_state.detach()


def make_integrator_rnn_fn(rnn, gamma=0.99, add_noise=True, device='cpu'):
    """
    Create a session-compatible rnn_fn that tracks the leaky reward integrator.

    The integrator state c resets to 0 at each session start (when h=None).

    Parameters
    ----------
    gamma : float
        Decay factor. 0.99 → τ ≈ 100 trials (slow, robust to noise).
                      0.95 → τ ≈ 20 trials  (fast, sharp block transitions).
    """
    c = [0.0]  # mutable integrator state (reset per session)

    def rnn_fn(h, stim, rewarded, instr, rng):
        if h is None:
            h = torch.zeros(rnn.N, device=device)
            c[0] = 0.0  # reset integrator at session start

        u, target, mask, lick, reward, z_seq, h_new = \
            rnn.forward_trial_closedloop(
                h, stim, rewarded, instr,
                add_noise=add_noise, device=device,
                integrator_value=c[0],
            )

        # Update integrator after observing trial outcome
        is_target = (stim is not None) and (stim == rewarded)
        if reward:
            delta = 1.0
        elif is_target and not lick:   # miss
            delta = -0.5
        else:
            delta = 0.0
        c[0] = float(np.tanh(gamma * c[0] + (1 - gamma) * delta))

        return u, target, mask, lick, reward, z_seq, h_new

    return rnn_fn
