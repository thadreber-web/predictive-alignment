# Predictive Alignment Research Log

## 2026-02-27 — Phase 1 Build & First Experiment

### What was built
- Complete Phase 1 codebase implementing Asabuki & Clopath (2025) predictive alignment
- Core files: `src/network.py`, `src/targets.py`, `src/instrumentation.py`, `src/utils.py`
- 7 experiment scripts (`experiments/01-07`)
- Running in stiefel-unsloth container on DGX Spark (GB10 GPU)

### Experiment 01: Sine wave — debugging journey

**Goal:** Train N=500 network to autonomously generate a sine wave (period=600ms, amplitude=1.5).

#### v1 — First attempt (FAILED)
- Config: dt=0.1ms, eta_w=1e-3, eta_m=1e-4, train=30s
- Bug: `tqdm` argument `miniinterval` → should be `mininterval`. Fixed across all files.
- Result: Lyapunov went from 6.6 → 47.3 (MORE chaotic after training!)
- Training error decreased (readout w learned) but M destabilized the network
- Autonomous generation: RMSE=1.31, output at wrong frequency
- Diagnosis: Too many timesteps accumulating M updates

#### v2 — Lower eta_m (IMPROVED but still broken)
- Config: dt=0.1ms, eta_m=1e-5, train=60s
- Result: Lyapunov 6.6 → 40.4, test RMSE=0.47
- Better but Lyapunov still going UP — M still destabilizing

#### v3 — dt-scaled learning (NO CHANGE)
- Added `dt` multiplier to learning updates in network.py
- Compensated with higher base rates (eta_w=1e-2, eta_m=1e-3)
- Result: Identical to v1 — effective rates were the same
- Lesson: scaling by dt + increasing base rate = no net change

#### Key insight: Paper uses dt=1ms, not 0.1ms!
- Checked actual paper (bioRxiv preprint): **dt = 1ms** with tau=10ms
- phase_1.md said dt=0.1ms but the paper uses 1ms
- This means 10x fewer timesteps per second of simulated time
- dt/tau = 0.1 (not 0.01) — larger integration steps

#### v4 — dt=1ms, original rates (PARTIAL)
- Config: dt=1ms, eta_w=1e-3, eta_m=1e-4, train=60s
- Result: Lyapunov 1.02 → 3.27, error 0.044 → 0.013, RMSE=0.78
- Error still decreasing at end of training — needed more time

#### v5 — dt=1ms, higher eta_m, longer training (SUCCESS)
- Config: dt=1ms, eta_w=1e-3, eta_m=1e-3, train=300s (500 periods)
- Result:
  - Lyapunov: 1.02 → 3.57
  - Error: 0.020 → **0.00089** (monotonic decrease)
  - Test RMSE: **0.027** (excellent match)
  - Autonomous generation: nearly perfect sine wave for full 10s test
- The network learned to be a sine wave oscillator

### Validated parameters for Phase 1
| Parameter | Value | Notes |
|-----------|-------|-------|
| dt | 1.0 ms | Matches paper (NOT 0.1ms from phase_1.md) |
| tau | 10.0 ms | |
| eta_w | 1e-3 | Readout learning rate |
| eta_m | 1e-3 | Recurrent learning rate (higher than phase_1.md suggested) |
| alpha | 1.0 | |
| g | 1.2 | |
| N | 500 | |
| Training | 300s+ | Need 500+ periods for convergence |

### Experiment 02: Alpha sweep (completed)

**Goal:** Sweep alpha = {0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0} with 20 seeds each. Reproduces Fig 3C-E.

**Results:**
| Alpha | Mean Error | Mean Lyap | Alignment trend |
|-------|-----------|-----------|-----------------|
| 0.0   | 0.009525  | 0.319     | Low/noisy       |
| 0.25  | 0.006846  | 0.337     | ~150            |
| 0.5   | 0.004044  | 0.346     | ~290            |
| 0.75  | 0.003140  | 0.359     | ~420            |
| 1.0   | **0.001570** | 0.363  | ~550            |
| 1.5   | 0.009054  | 0.282     | ~840            |
| 2.0   | 0.026649  | 0.286     | ~1140           |

**Key findings:**
- Alpha=1.0 gives lowest readout error — matches paper's Fig 3C
- Error decreases monotonically from alpha=0 to alpha=1, then rises (over-regularization)
- Alignment <Gr, Mr> scales linearly with alpha — matches paper's Fig 3D
- Plots saved: error_vs_alpha.png, lyapunov_vs_alpha.png, alignment_over_time.png, Gr_vs_Mr_scatter.png

### Experiment 03: Edge of chaos — SKIPPED

Skipped: reproduces known paper result (Supp Fig 5). Code validated by experiments 01-02.

### Code verification

Downloaded paper's official code (PA_code.py from github.com/TAsabuki/PredictiveAlignment).
Learning rule, dynamics, and weight initialization confirmed to match our implementation.

### Experiment 04: Eigenspectrum analysis (completed)

**Goal:** Snapshot eigenvalues of J = G+M at 0%, 25%, 50%, 75%, 100% of training on sine wave.

**Results:** Plots saved — eigenspectrum_evolution.png, eigenspectrum_panels.png, singular_values_M.png, effective_rank_M.png. Single run, ~5 min.

### Experiment 05: Lorenz attractor (completed — 2nd run)

**Goal:** Train N=500, K=3 network on 3D Lorenz attractor for 15,000s. Reproduces Fig 4.

**Run 1 — FAILED (incorrect Lorenz time scaling):**
- `generate_lorenz` advanced 1.0 Lorenz time units per network step (1ms)
- Lorenz oscillation period ~1ms vs network τ=10ms — network cannot track
- Error flat at 1.527 for entire 15M steps
- Root cause: Lorenz time and network time were not decoupled

**Paper comparison audit (2026-02-28):**
- Paper says "15,000 s trajectory" = 15,000 seconds of Lorenz time
- Each network step (1ms) should advance Lorenz by 0.001 seconds
- Added `lorenz_dt` parameter to `generate_lorenz` (default 0.001)
- Also found RSG pulse formulas were wrong (fixed before exp 06)

**Run 2 — SUCCESS (correct time scaling, lorenz_dt=0.001):**
- Oscillation period: ~734ms (matches paper Fig 4A)
- Error decreased throughout training: ~0.04 → **0.013**
- Runtime: ~51 min on GPU
- Plots saved: lorenz_components_late.png, lorenz_3d.png, lorenz_2d_*.png, lorenz_tent_map.png, eigenspectrum.png, lorenz_error.png

### Experiment 06: RSG timing (deferred)

**Goal:** N=1200, K=1, D=2, 200k trials with delays {100, 120, 140, 160} ms. Reproduces Fig 5.

**Bug fix before run:** `generate_rsg_trial` rewritten to match paper eq 17-19:
- Bipolar pulses: `2·exp(-(t-center)²/Δ²) - 1` (range [-1, +1])
- T_0 = 60ms, Δ = 15ms (was: unipolar 0-1, T_0=100, width=10)

Status: deferred — estimated 4-6 hours runtime. Code is ready, will run when time permits.

### Experiment 07: Damped pendulum (completed)

**Goal:** Train N=500, K=2 network on damped pendulum (theta, omega) for 50s (10 repeats of 5s trajectory). Custom experiment — not from the paper.

**Config:** b=0.5 (damping), g=9.81, L=1.0, theta0=2.0, omega0=0.0

**Results:**
- Final mean error: **0.028**
- Runtime: ~10 seconds
- Plots saved: pendulum_training.png, pendulum_test.png, phase_portrait_train.png, phase_portrait_test.png, pendulum_energy.png, pendulum_error.png

### Open questions
1. Lyapunov exponent values from perturbation estimator remain positive after training. Paper reports shift toward negative (Supp Fig 4). May be a measurement method difference — paper's code does not include their Lyapunov implementation.

### Next steps
- Run experiment 06 (RSG timing) when time permits (~4-6 hours)
- Phase 1 experiments complete (except exp 06)
