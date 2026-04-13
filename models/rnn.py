"""
DynamicRoutingRNN: Ground-truth RNN for the dynamic routing task.

Dynamics (continuous-time, Euler discretized):
    y_{t+1} = (1 - alpha) * y_t + alpha * f(W_rec @ y_t + W_in @ u_t + b_rec)
              + sigma * sqrt(2 * alpha) * noise_t

where:
    alpha = dt / tau = 0.2
    f = ReLU
    noise_t ~ N(0, I)

Output:
    z_t = sigmoid(w_out @ y_t + b_out)

Hidden state persists across all trials within a session (no reset at block boundaries).
"""

import torch
import torch.nn as nn
import numpy as np

from tasks.dynamic_routing import (
    RESP_START, RESP_END, STIM_ON, STIM_OFF, REWARD_DURATION,
    N_INPUTS, N_OUTPUTS, STEPS_PER_TRIAL, VIS1, VIS2, AUD1, AUD2
)


class DynamicRoutingRNN(nn.Module):
    """
    Ground-truth RNN for the dynamic routing task.

    Parameters
    ----------
    n_units : int, default 200
    n_inputs : int, default 7
    n_outputs : int, default 1
    tau : float, default 0.100  (seconds)
    dt : float, default 0.020   (seconds)
    sigma_rec : float, default 0.15
    """

    def __init__(self, n_units=200, n_inputs=7, n_outputs=1,
                 tau=0.100, dt=0.020, sigma_rec=0.15):
        super().__init__()
        self.N = n_units
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.tau = tau
        self.dt = dt
        self.alpha = dt / tau
        self.sigma_rec = sigma_rec

        # Learnable parameters
        # W_in is constrained to be non-negative: W_in = |W_in_raw|
        self.W_in_raw = nn.Parameter(torch.randn(n_units, n_inputs) * 0.05)
        # Initialize W_rec with spectral radius ~0.8 for stability
        W_rec_init = torch.randn(n_units, n_units) / np.sqrt(n_units)
        sr = float(torch.linalg.eigvals(W_rec_init).abs().max())
        W_rec_init = W_rec_init * (0.8 / sr)
        self.W_rec    = nn.Parameter(W_rec_init)
        self.b_rec    = nn.Parameter(torch.zeros(n_units))
        self.w_out    = nn.Parameter(torch.randn(n_outputs, n_units) * 0.1)
        self.b_out    = nn.Parameter(torch.zeros(n_outputs))

    @property
    def W_in(self):
        """Non-negative input weights."""
        return torch.abs(self.W_in_raw)

    def get_W_rec(self):
        return self.W_rec.detach()

    def get_W_in(self):
        return self.W_in.detach()

    def forward(self, y0, u_sequence, add_noise=True):
        """
        Run the RNN forward on a sequence of inputs.

        Parameters
        ----------
        y0 : torch.Tensor, shape (batch, N) or (N,)
            Initial hidden state.
        u_sequence : torch.Tensor, shape (batch, T, n_inputs) or (T, n_inputs)
            Input sequence.
        add_noise : bool
            Whether to add recurrent noise.

        Returns
        -------
        y_seq : torch.Tensor, shape (batch, T, N)
            Hidden state sequence.
        z_seq : torch.Tensor, shape (batch, T, n_outputs)
            Output sequence.
        y_final : torch.Tensor, shape (batch, N)
            Final hidden state.
        """
        squeeze = u_sequence.dim() == 2
        if squeeze:
            u_sequence = u_sequence.unsqueeze(0)
            y0 = y0.unsqueeze(0)

        batch, T, _ = u_sequence.shape
        device = u_sequence.device

        W_in  = self.W_in
        W_rec = self.W_rec
        b_rec = self.b_rec

        y = y0  # (batch, N)
        y_list = []
        z_list = []

        noise_scale = self.sigma_rec * np.sqrt(2 * self.alpha) if add_noise else 0.0

        for t in range(T):
            u_t = u_sequence[:, t, :]  # (batch, n_inputs)

            # Recurrent input
            pre = (y @ W_rec.T) + (u_t @ W_in.T) + b_rec  # (batch, N)

            # Euler step
            noise = noise_scale * torch.randn_like(y) if add_noise else 0.0
            y = (1 - self.alpha) * y + self.alpha * torch.relu(pre) + noise

            y_list.append(y)

            z_t = torch.sigmoid(y @ self.w_out.T + self.b_out)  # (batch, 1)
            z_list.append(z_t)

        y_seq = torch.stack(y_list, dim=1)  # (batch, T, N)
        z_seq = torch.stack(z_list, dim=1)  # (batch, T, n_outputs)

        if squeeze:
            y_seq = y_seq.squeeze(0)
            z_seq = z_seq.squeeze(0)
            y = y.squeeze(0)

        return y_seq, z_seq, y

    def forward_trial_closedloop(self, y0, stimulus_idx, rewarded_target_idx,
                                  instruction=False, add_noise=True, device=None):
        """
        Run one trial with closed-loop reward/lick feedback.
        Optimized: precomputes W_in @ u for the static input portion.

        Returns
        -------
        u_trial : np.ndarray, shape (STEPS_PER_TRIAL, N_INPUTS)
        target  : np.ndarray, shape (STEPS_PER_TRIAL,)
        mask    : np.ndarray, shape (STEPS_PER_TRIAL,), bool
        lick_occurred : bool
        reward_delivered : bool
        z_seq   : np.ndarray, shape (STEPS_PER_TRIAL,)
        y_final : torch.Tensor, shape (N,) — detached
        """
        if device is None:
            device = next(self.parameters()).device

        T = STEPS_PER_TRIAL
        is_rewarded = (stimulus_idx is not None) and (stimulus_idx == rewarded_target_idx)

        y = y0.to(device)  # (N,)

        # Build static u_trial (without lick/reward channels)
        u_trial = torch.zeros(T, self.n_inputs, device=device)
        if stimulus_idx is not None:
            u_trial[STIM_ON:STIM_OFF, stimulus_idx] = 1.0
        u_trial[RESP_START:RESP_END, 6] = 1.0  # trial phase

        target = np.zeros(T, dtype=np.float32)
        if is_rewarded:
            target[RESP_START:RESP_END] = 1.0

        mask = np.zeros(T, dtype=bool)
        mask[RESP_START:RESP_END] = True

        noise_scale = self.sigma_rec * float(np.sqrt(2 * self.alpha)) if add_noise else 0.0

        with torch.no_grad():
            # Precompute W_in @ u for the full trial (static channels only)
            wu = u_trial @ self.W_in.T + self.b_rec  # (T, N)
            # Columns for lick/reward channels (to add when they fire)
            w_lick   = self.W_in[:, 5]  # (N,) — weight column for own-lick
            w_reward = self.W_in[:, 4]  # (N,) — weight column for reward

            y_state = y
            z_list = []
            lick_occurred = False
            reward_delivered = False

            auto_start = RESP_END - REWARD_DURATION  # timestep when auto-reward fires

            for t in range(T):
                # Instruction trial: fire auto-reward into wu at auto_start if no lick yet
                # (must happen before the forward step so the network processes it)
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

                # Closed-loop: detect first lick in response window
                if RESP_START <= t < RESP_END and not lick_occurred:
                    if z_t > 0.5:
                        lick_occurred = True
                        lick_end = min(t + REWARD_DURATION, T)
                        # Update wu for lick channel
                        wu[t:lick_end] = wu[t:lick_end] + w_lick.unsqueeze(0)
                        u_trial[t:lick_end, 5] = 1.0
                        if is_rewarded:
                            wu[t:lick_end] = wu[t:lick_end] + w_reward.unsqueeze(0)
                            u_trial[t:lick_end, 4] = 1.0
                            reward_delivered = True

        z_seq = np.array(z_list, dtype=np.float32)
        u_np  = u_trial.cpu().numpy()

        return u_np, target, mask, lick_occurred, reward_delivered, z_seq, y_state.detach()


def make_rnn_fn(rnn_model, add_noise=True, device=None):
    """
    Create a callable compatible with session.generate_session().

    Returns a function: (h, stim, rewarded, instr, rng) ->
        (u, target, mask, lick, reward, z_seq, h_new)
    where h is a torch.Tensor or None (initializes to zeros).
    """
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
