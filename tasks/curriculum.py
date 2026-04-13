"""
Curriculum stage management for the dynamic routing task.

Stages:
  0 — Auto-rewards (output bias shaping)
  1 — Visual discrimination (vis1 vs vis2)
  2 — Auditory discrimination (aud1 vs aud2)
  3 — Block switching with timeout (×3 FA loss weight)
  4 — Final task (no timeout, matches ephys conditions)
"""

import numpy as np
from tasks.dynamic_routing import compute_dprime, VIS1, VIS2, AUD1, AUD2


class CurriculumManager:
    """
    Tracks the current training stage and evaluates advancement/regression criteria.
    """

    def __init__(self):
        self.stage = 0
        # History of (step, stage, metrics) tuples
        self.history = []
        # How many consecutive evaluation checkpoints pass/fail criteria
        self._consec_pass = 0
        self._consec_fail = 0

    # ------------------------------------------------------------------
    def evaluate(self, session_results, training_step):
        """
        Compute performance metrics and check stage advancement/regression.

        Parameters
        ----------
        session_results : list of dicts
            Each dict is one evaluation session (output of generate_session).
        training_step : int

        Returns
        -------
        metrics : dict
            Contains d' values and other per-stage metrics.
        advanced : bool
            Whether the stage advanced.
        regressed : bool
            Whether the stage regressed.
        """
        metrics = {}
        advanced = False
        regressed = False

        if self.stage == 0:
            metrics = self._stage0_metrics(session_results)
            # Fixed-step advancement — handled externally
        elif self.stage == 1:
            metrics = self._stage12_metrics(session_results, rewarded=VIS1, non_target=VIS2)
            advanced, regressed = self._check_stage12(metrics)
        elif self.stage == 2:
            metrics = self._stage12_metrics(session_results, rewarded=AUD1, non_target=AUD2)
            advanced, regressed = self._check_stage12(metrics)
        elif self.stage in (3, 4):
            metrics = self._stage34_metrics(session_results)
            if self.stage == 3:
                advanced, regressed = self._check_stage3(metrics)
            else:
                advanced, regressed = self._check_stage4(metrics)

        self.history.append({'step': training_step, 'stage': self.stage, 'metrics': metrics})

        if advanced:
            self.stage = min(self.stage + 1, 4)
            self._consec_pass = 0
            self._consec_fail = 0
        elif regressed:
            self.stage = max(self.stage - 1, 0)
            self._consec_pass = 0
            self._consec_fail = 0

        return metrics, advanced, regressed

    # ------------------------------------------------------------------
    def _stage0_metrics(self, sessions):
        hit_rates = []
        for sess in sessions:
            licks = np.array(sess['licks'])
            hit_rates.append(float(np.mean(licks)) if len(licks) > 0 else 0.0)
        return {'mean_hit_rate': float(np.mean(hit_rates))}

    def _stage12_metrics(self, sessions, rewarded, non_target):
        hits, fas = [], []
        for sess in sessions:
            stims = np.array(sess['stimulus'], dtype=object)
            licks = np.array(sess['licks'])
            masks = ~np.array(sess['instruction'])

            stims_reg = stims[masks]
            licks_reg = licks[masks]

            hit_mask = stims_reg == rewarded
            fa_mask  = stims_reg == non_target

            hr = float(np.mean(licks_reg[hit_mask])) if hit_mask.sum() > 0 else 0.5
            fa = float(np.mean(licks_reg[fa_mask]))  if fa_mask.sum()  > 0 else 0.5
            hits.append(hr)
            fas.append(fa)

        hr_mean = float(np.mean(hits))
        fa_mean = float(np.mean(fas))
        dp = compute_dprime(hr_mean, fa_mean)
        return {
            'hit_rate': hr_mean,
            'fa_rate': fa_mean,
            'dprime': dp,
            'n_hits': int(sum(h > 0 for h in hits)),
        }

    def _stage34_metrics(self, sessions):
        """Per-block d' computation for Stages 3/4."""
        block_metrics = []
        for sess in sessions:
            stims    = np.array(sess['stimulus'], dtype=object)
            licks    = np.array(sess['licks'])
            rewarded = np.array(sess['rewarded'])
            instr    = np.array(sess['instruction'])
            masks    = ~instr  # exclude instruction trials

            # Group by block rewarded target
            for rew_target in [VIS1, AUD1]:
                block_mask = masks & (rewarded == rew_target)
                if block_mask.sum() == 0:
                    continue

                s_b = stims[block_mask]
                l_b = licks[block_mask]

                if rew_target == VIS1:
                    target_stim     = VIS1
                    intra_distract  = VIS2
                    inter_distract  = AUD1
                else:
                    target_stim     = AUD1
                    intra_distract  = AUD2
                    inter_distract  = VIS1

                hm = s_b == target_stim
                fa_intra_m = s_b == intra_distract
                fa_inter_m = s_b == inter_distract

                hr = float(np.mean(l_b[hm]))           if hm.sum() > 0          else 0.5
                fa_intra = float(np.mean(l_b[fa_intra_m])) if fa_intra_m.sum() > 0 else 0.5
                fa_inter = float(np.mean(l_b[fa_inter_m])) if fa_inter_m.sum() > 0 else 0.5

                dp_intra = compute_dprime(hr, fa_intra)
                dp_inter = compute_dprime(hr, fa_inter)

                block_metrics.append({
                    'rew_target': rew_target,
                    'hit_rate': hr,
                    'fa_intra': fa_intra,
                    'fa_inter': fa_inter,
                    'dprime_intra': dp_intra,
                    'dprime_inter': dp_inter,
                })

        if not block_metrics:
            return {'dprime_intra': 0.0, 'dprime_inter': 0.0, 'n_pass_blocks': 0}

        dp_intras  = [m['dprime_intra'] for m in block_metrics]
        dp_inters  = [m['dprime_inter'] for m in block_metrics]
        n_pass = sum(
            (di > 1.5 and de > 1.5) for di, de in zip(dp_intras, dp_inters)
        )
        return {
            'dprime_intra': float(np.mean(dp_intras)),
            'dprime_inter': float(np.mean(dp_inters)),
            'n_pass_blocks': n_pass,
            'n_total_blocks': len(block_metrics),
            'block_metrics': block_metrics,
        }

    # ------------------------------------------------------------------
    def _check_stage12(self, metrics):
        dp = metrics.get('dprime', 0.0)
        n_hits = metrics.get('n_hits', 0)
        advanced = regressed = False

        if dp > 1.5:
            self._consec_pass += 1
            self._consec_fail = 0
        elif n_hits == 0:
            self._consec_fail += 1
            self._consec_pass = 0
        else:
            self._consec_pass = 0
            self._consec_fail = 0

        if self._consec_pass >= 2:
            advanced = True
        if self._consec_fail >= 2:
            regressed = True
        return advanced, regressed

    def _check_stage3(self, metrics):
        n_pass = metrics.get('n_pass_blocks', 0)
        n_total = metrics.get('n_total_blocks', 1)
        advanced = regressed = False

        # Pass criterion: ≥ 4/6 blocks pass for 2 consecutive evaluations
        frac_pass = n_pass / max(n_total, 1)
        if frac_pass >= (4 / 6):
            self._consec_pass += 1
            self._consec_fail = 0
        else:
            self._consec_pass = 0
            self._consec_fail = 0

        if self._consec_pass >= 2:
            advanced = True
        return advanced, regressed

    def _check_stage4(self, metrics):
        # Stage 4: converge when performance is stable across 5 consecutive evals
        # Return advanced=True to signal convergence (handled externally as "done")
        n_pass = metrics.get('n_pass_blocks', 0)
        n_total = metrics.get('n_total_blocks', 1)
        frac_pass = n_pass / max(n_total, 1)

        if frac_pass >= (4 / 6):
            self._consec_pass += 1
            self._consec_fail = 0
        else:
            self._consec_pass = 0
            self._consec_fail = 0

        advanced = self._consec_pass >= 5
        return advanced, False

    @property
    def fa_loss_weight(self):
        """Loss multiplier for false alarm trials."""
        return 3.0 if self.stage == 3 else 1.0

    @property
    def learning_rate(self):
        return 1e-4 if self.stage == 4 else 1e-3
