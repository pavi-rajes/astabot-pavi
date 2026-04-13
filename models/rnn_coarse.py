"""
DynamicRoutingRNNCoarse: same architecture as DynamicRoutingRNN but with
dt=100ms, tau=500ms (alpha=0.2).

Context retention per trial: 0.98^25 = 60% (vs 0.98^125 = 0.8% at dt=20ms).
Training speed: 5x faster (25 steps/trial vs 125 steps/trial).
"""

import torch
import torch.nn as nn
import numpy as np

from tasks.dynamic_routing_coarse import (
    RESP_START, RESP_END, STIM_ON, STIM_OFF, REWARD_DURATION,
    N_INPUTS, N_OUTPUTS, STEPS_PER_TRIAL, VIS1, VIS2, AUD1, AUD2,
    DT, TAU, ALPHA
)


class DynamicRoutingRNNCoarse(nn.Module):
    """
    Vanilla RNN for dynamic routing task at dt=100ms, tau=500ms.
    Identical to DynamicRoutingRNN except for time constants and step indices.
    """

    def __init__(self, n_units=100, n_inputs=N_INPUTS, n_outputs=N_OUTPUTS,
                 sigma_rec=0.15):
        super().__init__()
        self.N = n_units
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.tau = TAU
        self.dt = DT
        self.alpha = ALPHA
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
        """Load shared weights from a DynamicRoutingRNN (dt=20ms) checkpoint."""
        own = self.state_dict()
        for k in ['W_in_raw', 'W_rec', 'b_rec', 'w_out', 'b_out']:
            if k in state_dict:
                own[k].copy_(state_dict[k])
        self.load_state_dict(own)

    def forward_trial_closedloop(self, y0, stimulus_idx, rewarded_target_idx,
                                  instruction=False, add_noise=True, device=None):
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
            wu = u_trial @ self.W_in.T + self.b_rec
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


def make_rnn_fn_coarse(rnn, add_noise=True, device=None):
    if device is None:
        device = next(rnn.parameters()).device

    def rnn_fn(h, stim, rewarded, instr, rng):
        if h is None:
            h = torch.zeros(rnn.N, device=device)
        u, target, mask, lick, reward, z_seq, h_new = \
            rnn.forward_trial_closedloop(
                h, stim, rewarded, instr, add_noise=add_noise, device=device
            )
        return u, target, mask, lick, reward, z_seq, h_new

    return rnn_fn
