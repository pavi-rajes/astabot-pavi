import sys, os, glob, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rnn_integrator_coarse import IntegratorRNNCoarse, make_integrator_coarse_rnn_fn
from tasks.session_coarse import generate_session_coarse
from tasks.dynamic_routing_coarse import VIS1, VIS2, AUD1, AUD2

rnn = IntegratorRNNCoarse(n_units=100)

# Load latest checkpoint
ckpts = glob.glob('checkpoints_integrator_coarse/seed_42/*.pt')
if ckpts:
    latest = max(ckpts, key=os.path.getmtime)
    sd = torch.load(latest, map_location='cpu')
    # Try as full state dict first, fall back to vanilla load
    try:
        rnn.load_state_dict(sd)
        print(f"Loaded full state dict: {latest}")
    except Exception:
        rnn.load_from_vanilla(sd)
        print(f"Loaded via load_from_vanilla: {latest}")
else:
    rnn.load_from_vanilla(torch.load('checkpoints_v24/seed_42/rnn_stage2.pt', map_location='cpu'))
    print("Using Stage 2 weights (no integrator_coarse checkpoint found)")

rng = np.random.default_rng(42)
rnn_fn = make_integrator_coarse_rnn_fn(rnn, gamma=0.90, add_noise=True)

results = {
    'vis1_block': {VIS1: [], VIS2: [], AUD1: [], AUD2: []},
    'aud1_block': {VIS1: [], VIS2: [], AUD1: [], AUD2: []},
}
stim_names = {VIS1: 'VIS1', VIS2: 'VIS2', AUD1: 'AUD1', AUD2: 'AUD2'}

for _ in range(5):
    sess = generate_session_coarse(rnn_fn, stage=3, rng=rng)
    for i, stim in enumerate(sess['stimulus']):
        if sess['instruction'][i] or stim is None:
            continue
        rew = sess['rewarded'][i]
        key = 'vis1_block' if rew == VIS1 else 'aud1_block'
        if stim in results[key]:
            results[key][stim].append(int(sess['licks'][i]))

print()
print("Lick rates by block context:")
print("{:>8} | {:>10} | {:>10} | {:>14}".format(
    "Stimulus", "VIS1-block", "AUD1-block", "Context effect"))
print("-" * 52)
for stim in [VIS1, VIS2, AUD1, AUD2]:
    v = np.mean(results['vis1_block'][stim]) if results['vis1_block'][stim] else float('nan')
    a = np.mean(results['aud1_block'][stim]) if results['aud1_block'][stim] else float('nan')
    diff = v - a
    arrow = "<-- rewarded" if stim in (VIS1, AUD1) else ""
    print("{:>8} | {:>10.3f} | {:>10.3f} | {:>+.3f}  {}".format(
        stim_names[stim], v, a, diff, arrow))

print()
print("Expected if context-sensitive:")
print("  VIS1 hit rate: high in VIS1-block, LOW in AUD1-block  (d'_inter > 0)")
print("  AUD1 hit rate: LOW in VIS1-block, high in AUD1-block  (d'_inter > 0)")
print()

# Also check overall lick rate
all_licks = [l for sess_licks in [results['vis1_block'][s] + results['aud1_block'][s]
             for s in [VIS1,VIS2,AUD1,AUD2]] for l in sess_licks]
print(f"Overall lick rate: {np.mean(all_licks):.3f}  (0=never licks, 1=always licks)")
print(f"Total trials analyzed: {len(all_licks)}")
