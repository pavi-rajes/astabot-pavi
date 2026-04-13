"""
Plotting utilities for the dynamic routing task.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_training_curves(log_path, save_path=None):
    """
    Plot training loss and d' curves over training steps from a CSV log file.
    Columns expected: step, stage, loss, dprime_intra, dprime_inter
    """
    if not HAS_MPL:
        print("matplotlib not available; skipping plot.")
        return

    import pandas as pd
    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(df['step'], df['loss'], lw=1)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss')
    ax.set_title('Training loss')

    ax = axes[1]
    if 'dprime_intra' in df.columns:
        ax.plot(df['step'], df['dprime_intra'], label="d' intra")
    if 'dprime_inter' in df.columns:
        ax.plot(df['step'], df['dprime_inter'], label="d' inter")
    ax.axhline(1.5, color='k', linestyle='--', lw=0.8, label='threshold')
    ax.set_xlabel('Training step')
    ax.set_ylabel("d'")
    ax.set_title("d' over training")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_hit_fa_rates(sessions, rewarded_target, distractor_intra,
                      distractor_inter=None, title='', save_path=None):
    """
    Bar chart of hit rate and FA rates across sessions.

    Parameters
    ----------
    sessions : list of session dicts
    rewarded_target : int
    distractor_intra : int  (same modality)
    distractor_inter : int or None  (other modality target)
    """
    if not HAS_MPL:
        print("matplotlib not available; skipping plot.")
        return

    from utils.metrics import hit_and_fa_rates

    hrs, fa_intras, fa_inters = [], [], []
    for sess in sessions:
        hr, fa_i, _, _ = hit_and_fa_rates(sess, rewarded_target, distractor_intra)
        hrs.append(hr)
        fa_intras.append(fa_i)
        if distractor_inter is not None:
            _, fa_e, _, _ = hit_and_fa_rates(sess, rewarded_target, distractor_inter)
            fa_inters.append(fa_e)

    labels = ['Hit rate', 'FA (intra)']
    values = [np.mean(hrs), np.mean(fa_intras)]
    errs   = [np.std(hrs), np.std(fa_intras)]

    if distractor_inter is not None:
        labels.append('FA (inter)')
        values.append(np.mean(fa_inters))
        errs.append(np.std(fa_inters))

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errs, capsize=4, color=['steelblue', 'coral', 'orange'][:len(labels)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Rate')
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_block_transition(sessions, rewarded_target, save_path=None):
    """
    Plot P(lick | rewarded target) as a function of trial position within a block.
    """
    if not HAS_MPL:
        print("matplotlib not available; skipping plot.")
        return

    from tasks.dynamic_routing import VIS1, VIS2, AUD1, AUD2

    max_trials = 100
    counts = np.zeros(max_trials, dtype=float)
    totals = np.zeros(max_trials, dtype=float)

    for sess in sessions:
        block_ids = np.array(sess['block_ids'])
        stims     = np.array(sess['stimulus'], dtype=object)
        licks     = np.array(sess['licks'])
        rewarded  = np.array(sess['rewarded'])

        for b in np.unique(block_ids):
            block_mask = (block_ids == b) & (rewarded == rewarded_target)
            if block_mask.sum() == 0:
                continue
            idx = np.where(block_mask)[0]
            for pos, i in enumerate(idx):
                if pos >= max_trials:
                    break
                if stims[i] == rewarded_target:
                    counts[pos] += float(licks[i])
                    totals[pos] += 1.0

    valid = totals > 0
    p_lick = np.where(valid, counts / totals, np.nan)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.where(valid)[0], p_lick[valid], marker='o', ms=3)
    ax.axhline(0.5, color='k', linestyle='--', lw=0.8)
    ax.set_xlabel('Trial position in block')
    ax.set_ylabel('P(lick | rewarded target)')
    ax.set_title(f'Block transition dynamics (rewarded={rewarded_target})')
    ax.set_xlim(0, int(valid.sum()))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)
