"""
GatedDynamicRoutingRNN: GRU-style gated variant of DynamicRoutingRNN.

Motivation: vanilla RNN has fixed leak rate alpha=0.2, causing context signals
to decay as 0.98^t per timestep (~0.8% after 1 trial). A learned per-unit gate
lets the network hold context indefinitely.

Dynamics:
    gate = sigmoid(W_gate @ y_t + W_gate_in @ u_t + b_gate)
    y_{t+1} = gate * y_t + (1 - gate) * relu(W_rec @ y_t + W_in @ u_t + b_rec)
              + sigma * sqrt(2 * mean_alpha) * noise_t

where mean_alpha = mean(1 - gate) ≈ alpha initially (from b_gate initialization).

Initialization: b_gate = logit(alpha) ≈ -1.386 so gate ≈ 0.2 initially,
matching vanilla RNN behavior at initialization.
"""

import torch
import torch.nn as nn
import numpy as np

from tasks.dynamic_routing import (
    RESP_START, RESP_END, STIM_ON, STIM_OFF, REWARD_DURATION,
    N_INPUTS, N_OUTPUTS, STEPS_PER_TRIAL, VIS1, VIS2, AUD1, AUD2
)


class GatedDynamicRoutingRNN(nn.Module):
    """
    GRU-style gated variant of DynamicRoutingRNN.

    Extra parameters vs. DynamicRoutingRNN:
        W_gate     : (N, N)    recurrent gate weights
        W_gate_in  : (N, n_inputs)  input gate weights
        b_gate     : (N,)      gate bias (initialized to logit(alpha))

    All other parameters (W_in_raw, W_rec, b_rec, w_out, b_out) are identical
    and can be loaded from a DynamicRoutingRNN checkpoint.
    """

    def __init__(self, n_units=200, n_inputs=7, n_outputs=1,
                 tau=0.100, dt=0.020, sigma_rec=0.15):
        super().__init__()
        self.N = n_units
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.tau = tau
        self.dt = dt
        self.alpha = dt / tau  # nominal alpha (for noise scaling only)
        self.sigma_rec = sigma_rec

        # ── Shared parameters (same as DynamicRoutingRNN) ─────────────────────
        self.W_in_raw = nn.Parameter(torch.randn(n_units, n_inputs) * 0.05)
        W_rec_init = torch.randn(n_units, n_units) / np.sqrt(n_units)
        sr = float(torch.linalg.eigvals(W_rec_init).abs().max())
        W_rec_init = W_rec_init * (0.8 / sr)
        self.W_rec = nn.Parameter(W_rec_init)
        self.b_rec = nn.Parameter(torch.zeros(n_units))
        self.w_out = nn.Parameter(torch.randn(n_outputs, n_units) * 0.1)
        self.b_out = nn.Parameter(torch.zeros(n_outputs))

        # ── Gate parameters (new) ─────────────────────────────────────────────
        # Initialize small so gate ≈ sigmoid(logit(alpha)) = alpha = 0.2 initially
        gate_bias_init = float(np.log(self.alpha / (1 - self.alpha)))  # logit(0.2) ≈ -1.386
        self.W_gate    = nn.Parameter(torch.randn(n_units, n_units) * 0.01)
        self.W_gate_in = nn.Parameter(torch.randn(n_units, n_inputs) * 0.01)
        self.b_gate    = nn.Parameter(torch.full((n_units,), gate_bias_init))

    @property
    def W_in(self):
        return torch.abs(self.W_in_raw)

    def load_from_vanilla(self, state_dict):
        """Load shared weights from a DynamicRoutingRNN state_dict."""
        shared_keys = ['W_in_raw', 'W_rec', 'b_rec', 'w_out', 'b_out']
        own = self.state_dict()
        for k in shared_keys:
            if k in state_dict:
                own[k].copy_(state_dict[k])
        self.load_state_dict(own)

    def forward(self, y0, u_sequence, add_noise=True):
        """Same interface as DynamicRoutingRNN.forward()."""
        squeeze = u_sequence.dim() == 2
        if squeeze:
            u_sequence = u_sequence.unsqueeze(0)
            y0 = y0.unsqueeze(0)

        batch, T, _ = u_sequence.shape
        device = u_sequence.device

        W_in  = self.W_in
        W_rec = self.W_rec
        b_rec = self.b_rec

        y = y0
        y_list = []
        z_list = []

        noise_scale = self.sigma_rec * np.sqrt(2 * self.alpha) if add_noise else 0.0

        for t in range(T):
            u_t = u_sequence[:, t, :]  # (batch, n_inputs)

            pre    = (y @ W_rec.T) + (u_t @ W_in.T) + b_rec        # (batch, N)
            g_pre  = (y @ self.W_gate.T) + (u_t @ self.W_gate_in.T) + self.b_gate
            gate   = torch.sigmoid(g_pre)                            # (batch, N) in (0,1)

            new_state = torch.relu(pre)
            noise = noise_scale * torch.randn_like(y) if add_noise else 0.0
            y = gate * y + (1 - gate) * new_state + noise

            y_list.append(y)
            z_t = torch.sigmoid(y @ self.w_out.T + self.b_out)
            z_list.append(z_t)

        y_seq = torch.stack(y_list, dim=1)
        z_seq = torch.stack(z_list, dim=1)

        if squeeze:
            y_seq = y_seq.squeeze(0)
            z_seq = z_seq.squeeze(0)
            y = y.squeeze(0)

        return y_seq, z_seq, y

    def forward_trial_closedloop(self, y0, stimulus_idx, rewarded_target_idx,
                                  instruction=False, add_noise=True, device=None):
        """Same interface as DynamicRoutingRNN.forward_trial_closedloop()."""
        if device is None:
            device = next(self.parameters()).device

        T = STEPS_PER_TRIAL
        is_rewarded = (stimulus_idx is not None) and (stimulus_idx == rewarded_target_idx)

        y = y0.to(device)

        u_trial = torch.zeros(T, self.n_inputs, device=device)
        if stimulus_idx is not None:
            u_trial[STIM_ON:STIM_OFF, stimulus_idx] = 1.0
        u_trial[RESP_START:RESP_END, 6] = 1.0

        target = np.zeros(T, dtype=np.float32)
        if is_rewarded:
            target[RESP_START:RESP_END] = 1.0

        mask = np.zeros(T, dtype=bool)
        mask[RESP_START:RESP_END] = True

        noise_scale = self.sigma_rec * float(np.sqrt(2 * self.alpha)) if add_noise else 0.0

        with torch.no_grad():
            wu     = u_trial @ self.W_in.T + self.b_rec   # (T, N) static recurrent drive
            wu_g   = u_trial @ self.W_gate_in.T + self.b_gate  # (T, N) static gate drive
            w_lick   = self.W_in[:, 5]
            w_reward = self.W_in[:, 4]
            w_lick_g   = self.W_gate_in[:, 5]
            w_reward_g = self.W_gate_in[:, 4]

            y_state = y
            z_list = []
            lick_occurred = False
            reward_delivered = False

            auto_start = RESP_END - REWARD_DURATION

            for t in range(T):
                if instruction and not reward_delivered and t == auto_start:
                    wu[t:RESP_END]   = wu[t:RESP_END]   + w_reward.unsqueeze(0)
                    wu_g[t:RESP_END] = wu_g[t:RESP_END] + w_reward_g.unsqueeze(0)
                    u_trial[t:RESP_END, 4] = 1.0
                    reward_delivered = True

                pre   = y_state @ self.W_rec.T + wu[t]
                g_pre = y_state @ self.W_gate.T + wu_g[t]
                if add_noise:
                    pre = pre + noise_scale * torch.randn_like(pre)
                gate    = torch.sigmoid(g_pre)
                y_state = gate * y_state + (1 - gate) * torch.relu(pre)

                z_t = float(torch.sigmoid(y_state @ self.w_out.T + self.b_out).squeeze())
                z_list.append(z_t)

                if RESP_START <= t < RESP_END and not lick_occurred:
                    if z_t > 0.5:
                        lick_occurred = True
                        lick_end = min(t + REWARD_DURATION, T)
                        wu[t:lick_end]   = wu[t:lick_end]   + w_lick.unsqueeze(0)
                        wu_g[t:lick_end] = wu_g[t:lick_end] + w_lick_g.unsqueeze(0)
                        u_trial[t:lick_end, 5] = 1.0
                        if is_rewarded:
                            wu[t:lick_end]   = wu[t:lick_end]   + w_reward.unsqueeze(0)
                            wu_g[t:lick_end] = wu_g[t:lick_end] + w_reward_g.unsqueeze(0)
                            u_trial[t:lick_end, 4] = 1.0
                            reward_delivered = True

        z_seq = np.array(z_list, dtype=np.float32)
        u_np  = u_trial.cpu().numpy()

        return u_np, target, mask, lick_occurred, reward_delivered, z_seq, y_state.detach()


def make_gated_rnn_fn(rnn_model, add_noise=True, device=None):
    """Create a callable compatible with session.generate_session()."""
    if device is None:
        device = next(rnn_model.parameters()).device

    def rnn_fn(h, stim, rewarded, instr, rng):
        if h is None:
            h = torch.zeros(rnn_model.N, device=device)
        u, target, mask, lick, reward, z_seq, h_new = \
            rnn_model.forward_trial_closedloop(
                h, stim, rewarded, instr, add_noise=add_noise, device=device
            )
        return u, target, mask, lick, reward, z_seq, h_new

    return rnn_fn
