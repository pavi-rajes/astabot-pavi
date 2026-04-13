 Why Training Is Stuck: A Systematic Analysis                                                                                  
                                                                                                                                
  The Core Problem                                                                                                              
                                                                                                                                
  The task requires the network to maintain a context belief (VIS1-block vs AUD1-block) for ~90 trials based only on reward     
  feedback. Three separate obstacles interact:                                                                                  
                                                                                                                                
  1. Signal decay: context signal in RNN hidden state vanishes over trials                                                      
  2. Gradient blindness: BPTT windows don't span block boundaries → gradient never sees "context switch"                        
  3. Attractor bootstrap: network needs correct licking to get context gradient, but needs context to lick correctly            
                                                                                                                                
  ---                                                                                                                           
  Iteration History                                                                                                             
                                                                                                                                
  ┌───────────────────────────────┬───────────┬────────────────────────────────┬───────────────────────────────────────────────┐
  │           Approach            │ d'_inter  │         What Improved          │              Why Still Failed                 │
  │                               │   Peak    │                                │                                               │
  ├───────────────────────────────┼───────────┼────────────────────────────────┼───────────────────────────────────────────────┤
  │ Stages 0–2 baseline           │ 0.00      │ d'_intra works (within-block   │ No mechanism to hold context over trials      │
  │                               │           │ discrimination)                │                                               │
  ├───────────────────────────────┼───────────┼────────────────────────────────┼───────────────────────────────────────────────┤
  │ Teacher forcing (v25)         │ 0.22      │ First real context signal;     │ Hint fades → RNN retains only 0.8%/trial;     │
  │                               │           │ bootstrapped attractor init    │ attractor never crystallizes                  │
  ├───────────────────────────────┼───────────┼────────────────────────────────┼───────────────────────────────────────────────┤
  │ GRU gating                    │ 0.00      │ Theoretical 88% retention/     │ Bootstrap deadlock: needs context gradient to │
  │                               │           │ trial                          │ learn gating, needs gating to get context     │
  ├───────────────────────────────┼───────────┼────────────────────────────────┼───────────────────────────────────────────────┤
  │ Coarse dt (dt=100ms)          │ 0.19      │ 60% retention/trial; 5×        │ 60%/trial → ~0 over 45 trials; single-block  │
  │                               │           │ faster training                │ BPTT blind to block boundary                 │
  ├───────────────────────────────┼───────────┼────────────────────────────────┼───────────────────────────────────────────────┤
  │ Coarse + 2-block BPTT         │ 0.19      │ Gradient technically spans     │ No persistent signal; hidden state ~0 at      │
  │                               │           │ block boundary                 │ block boundary regardless                    │
  ├───────────────────────────────┼───────────┼────────────────────────────────┼───────────────────────────────────────────────┤
  │ Integrator standalone         │ 0.00      │ c≈0.41 after 5 instruction     │ Bug: W_in[:,7] init to 0 → no gradient        │
  │ (γ=0.90)                      │           │ trials — strong persistent     │ foothold; integrator completely ignored       │
  │                               │           │ signal                         │                                               │
  ├───────────────────────────────┼───────────┼────────────────────────────────┼───────────────────────────────────────────────┤
  │ Integrator + Coarse           │ 0.36      │ Bug fixed; c=0.41 persists;    │ BPTT windows still split by block; gradient   │
  │ (current)                     │           │ 60%/trial RNN retention        │ can't learn to USE sign flip of c             │
  ├───────────────────────────────┼───────────┼────────────────────────────────┼───────────────────────────────────────────────┤
  │ Integrator + Coarse +         │ 0.285     │ All 3 root causes addressed;   │ Gradient variance too high (batch=1);         │
  │ 2-block BPTT + ctx loss (v1)  │ osc       │ ctx loss forces context repr   │ λ_ctx=0.2 too weak; no upward trend over      │
  │                               │           │                                │ 2000 steps (mean=0.055, std=0.097)            │
  └───────────────────────────────┴───────────┴────────────────────────────────┴───────────────────────────────────────────────┘

  ---
  CRITICAL FINDING: Explicit Context Ablation Succeeded (2026-04-07)

  Injecting ground-truth context (±1.0) as channel 7 — bypassing the integrator — produced
  rapid, monotonic Stage 3 learning:

    step 350: d'_inter=0.48, step 500: d'_inter=1.00, step 750: d'_inter=2.32 → ADVANCED
    Stage 4 done at step 1000 (d'_inter=2.74). Total time: ~10 hours (batch=4, bptt=2-block).

  This proves:
    ✅ The architecture (IntegratorRNNCoarse) CAN solve the task
    ✅ The behavioral loss (BCE + cross-modal FA penalty) IS sufficient
    ✅ The 2-block BPTT + ctx loss design IS correct
    ❌ The ONLY remaining bottleneck: learning to infer context from reward history

  ---
  What "Oscillating at 0.285" Means (v1 diagnosis)

  The v1 run (integrator_coarse_ctx) had d'_intra ≈ 4.0 but d'_inter stuck oscillating.
  Statistical analysis: mean=0.055, std=0.097, max=0.285 at step 1450, trend≈+0/step.
  The network briefly achieved n_pass_blocks=1 at step 1500, then lost it.

  Root cause: gradient variance. With batch_size=1, each gradient step sees one stochastic
  session. The ctx loss at λ=0.2 provides a signal too weak to consistently overcome the
  noise. The network randomly lands in the context attractor basin, then gets kicked out.

  ---
  Combinations Currently Running (v2)

  Integrator + Coarse + 2-block BPTT + ctx loss v2 (PID 22244, checkpoints_integrator_coarse_v2):
  - batch_size: 1 → 4       (4× variance reduction — main fix)
  - LR Stage 3: 1e-3 → 5e-4 (more stable optimization)
  - λ_ctx: 0.2 → 0.5        (stronger context supervision)
  - max_sr Stage 3: 1.0 → 1.2 (allows slight SR > 1 for attractor formation)
  - γ curriculum: 0.50→0.90 over 500 steps (c≈0.97 early → easier bootstrap)

  NOTE: v2 initially crashed due to torch.linalg.eigvals LAPACK failure at SR>1 threshold.
  Fixed by replacing eigvals with torch.linalg.matrix_norm(ord=2) (operator norm, stable SVD).
  Relaunched at 11:38AM 2026-04-07.

  ---
  Next If v2 Oscillates Too

  - Context teacher forcing: blend explicit (±1.0) with integrator c, fade explicit over 500 steps
    → bridges the gap between the two conditions we've now tested
  - Fixed γ=0.50 (no annealing): stronger signal throughout; less realistic but faster learning
  - λ_ctx=1.0: make context the dominant training objective; behavioral loss secondary
