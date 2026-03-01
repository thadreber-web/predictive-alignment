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

## 2026-02-28 — Pendulum Ablations & Generalization Tests

### Experiment 07 ablations (07a, 07b, 07d)

Three ablations to understand what the pendulum network learned:

| Variant | Description | Final Error |
|---------|-------------|-------------|
| 07 (original) | N=500, 10 repeats of 5s | 0.028 |
| 07a | 50s continuous (no repeats) | **0.013** |
| 07b | N=100 (small network) | 0.176 |
| 07d | Input-driven, teacher forcing, self-feeding test | 0.020 |

**Findings:**
- 07a: Single long trajectory > repeated short trajectory for training
- 07b: N=100 lacks capacity, ~6x worse than N=500
- 07d: Teacher forcing trains well (0.020) but self-feeding at test diverges

### Generalization test: 07d vs 07a on novel ICs

Trained on (θ=2.0, ω=0.0), tested on novel ICs:

| IC | 07a error | 07d error |
|----|-----------|-----------|
| θ=1.0, ω=0.0 | 1.225 | 5.504 |
| θ=2.5, ω=1.0 | 2.598 | 5.772 |

**07d catastrophically fails:** Diverges to spurious fixed point (~θ≈3.1, ω≈4.2) within ~100ms. Classic autoregressive instability — network never saw its own errors during teacher-forced training.

**07a produces near-zero output:** Memorized the late (damped) portion of training trajectory. Lower error only because hovering near zero is closer to a damped pendulum than diverging.

Neither architecture generalizes. Both memorize trajectories, not physics.

### Experiment 07e: Multi-IC training for 07d

Trained 07d on 5 ICs: (0.5,0), (1.5,0), (2.0,0), (2.0,2.0), (3.0,-1.0). 5 repeats each, interleaved, network reset between trajectories.

**Training error:** 0.015 (good)

**Test on held-out ICs (self-feeding):**
| IC | Error |
|----|-------|
| θ=1.0, ω=0.0 (interp) | 2.253 |
| θ=2.5, ω=1.0 (interp) | 2.987 |
| θ=0.3, ω=0.5 (extrap) | 2.176 |
| θ=2.0, ω=0.0 (train sanity) | 2.748 |

**Improvement over single-IC 07d** (5.5 → 2.3) but still bad. No longer diverges to fixed point — instead enters fast spurious limit cycle (~200ms period vs true ~650ms). The network oscillates at its own natural timescale, not the pendulum's. Even the training IC sanity check fails (2.748), confirming self-feeding loop instability is the core problem, not IC coverage.

**Diagnosis:** Exposure bias. Teacher forcing trains on smooth ground-truth inputs. Self-feeding produces noisy inputs the network never learned to handle. Standard fix: scheduled sampling (gradually replace teacher forcing with self-feeding during training).

### Phase 1 conclusions

1. Predictive alignment successfully learns autonomous trajectory generation (sine, Lorenz, pendulum)
2. It does NOT generalize to novel initial conditions — it memorizes trajectories
3. Input-driven variant (07d) fails at test time due to train/test mismatch (exposure bias)
4. Multi-IC training reduces but does not fix the self-feeding instability
5. **Next: Phase 2 — scheduled sampling to bridge the teacher forcing → self-feeding gap**

## 2026-02-28 — Phase 2: Scheduled Sampling & BPTT Baseline

### Experiment 2.1: PA + Scheduled Sampling (FAILED)

N=500, K=2, D=2. 500 epochs, random ICs from θ∈[-2.5,2.5], ω∈[-3.0,3.0]. Linear anneal p_tf from 1.0→0.0 over 400 epochs.

**Training error stayed low** (~0.012–0.023) even as self-feeding increased — the network handled mixed inputs during training.

**But self-feeding checkpoints showed no improvement:**
| Epoch | p_tf | Self-feeding Error |
|-------|------|--------------------|
| 0 | 1.00 | 1.74 |
| 150 | 0.63 | 3.85 |
| 250 | 0.38 | 1.57 |
| 400 | 0.00 | 3.43 |
| 499 | 0.00 | 3.38 |

Error bounced randomly between 1.6–3.9 with no downward trend across 500 epochs.

**Final generalization (all held-out ICs): mean error = 3.606**

Failure mode: diverges to fixed point (θ≈2.7, ω≈-1.9) within ~100ms. Same catastrophic instability as 07d. Scheduled sampling changed which attractor the network finds, but didn't prevent divergence.

**Root cause:** PA's learning rule has no gradient signal through the self-feeding loop. The M update (`rec_error = Q@z - (M - αG)@r`) is purely local and one-step — it can't learn to correct multi-step autoregressive error accumulation. Self-feeding noise during training just makes M updates noisier, not more robust.

### Experiment 2.0: BPTT Baseline (PARTIAL SUCCESS)

Same architecture (N=500, K=2, D=2, sparse G + trainable M) but trained with truncated BPTT (T=50 steps, Adam, lr=1e-3, grad clip=1.0). Same scheduled sampling + multi-IC setup as exp 2.1.

**Checkpoint stability showed clear downward trend:**
| Epoch | p_tf | Self-feeding Error |
|-------|------|--------------------|
| 0 | 1.00 | 1.35 |
| 100 | 0.75 | 1.34 |
| 250 | 0.38 | 1.21 |
| 400 | 0.00 | 1.24 |
| 499 | 0.00 | 1.22 |

Converged to ~1.2 by epoch 300 and stayed stable. PA never achieved this.

**Final generalization on held-out ICs:**
| IC | BPTT | PA (2.1) |
|----|------|----------|
| θ=1.0, ω=0.0 | 1.225 | 3.378 |
| θ=2.5, ω=1.0 | 2.581 | 3.737 |
| θ=0.3, ω=0.5 | **0.437** | 3.412 |
| θ=-1.5, ω=2.0 | 1.911 | 3.644 |
| θ=2.0, ω=-2.5 | 2.381 | 3.859 |
| **Mean** | **1.707** | **3.606** |

**BPTT is 2x better than PA** on the same task. However, the time series plots reveal the BPTT network damps to near-zero output quickly — it found the trivial stable fixed point (output ≈ 0) rather than learning dynamics. Low error for small-amplitude ICs (extrap_tiny: 0.437) because zero is already close to a damped signal. Neither method learned physics.

### Key conclusions from Phase 2 so far

1. **Scheduled sampling does NOT fix PA's self-feeding instability.** PA's local learning rule cannot propagate error through the autoregressive chain. This is a fundamental limitation, not a hyperparameter issue.

2. **BPTT partially solves it** — gradient flow through the self-feeding loop lets it find stable (if trivial) solutions. This confirms the problem is partly PA-specific.

3. **Neither method learns generalizable dynamics** within 500 epochs. Both find degenerate solutions: PA diverges, BPTT collapses to zero.

4. **The state-to-state prediction framing may need rethinking.** The RNN dynamics (τ=10ms) operate on a much faster timescale than the pendulum (~650ms period). The readout z = w@r maps a 500-dim state through a rank-2 bottleneck — this may not be expressive enough for the self-feeding loop to maintain oscillatory dynamics.

### Experiment 2.5: Neural ODE Baseline (SUCCESS)

Architecture: small MLP (2→64→64→64→2, **8,642 params** vs RNN's 252k) learns dstate/dt = f(state). Integrated with differentiable RK4. Trained on 50-step segments with backprop through the solver, cosine LR schedule, 500 epochs.

**Checkpoint error (θ=1.0, ω=0.0) during training:**
| Epoch | Error | Notes |
|-------|-------|-------|
| 0 | 1.007 | Random init |
| 50 | 0.065 | Already below 0.5 threshold |
| 250 | 0.035 | Still improving |
| 350 | 0.023 | |
| 400 | 0.013 | |
| 499 | 0.014 | Converged |

Training time: 8,286s (~2.3 hours). Loss reached ~0 by epoch 150.

**Final generalization on held-out ICs:**

| IC | Neural ODE | BPTT | PA (2.1) |
|----|-----------|------|----------|
| θ=1.0, ω=0.0 | **0.014** | 1.225 | 3.378 |
| θ=2.5, ω=1.0 | **0.279** | 2.581 | 3.737 |
| θ=0.3, ω=0.5 | **0.006** | 0.437 | 3.412 |
| θ=-1.5, ω=2.0 | **0.047** | 1.911 | 3.644 |
| θ=2.0, ω=-2.5 | **0.080** | 2.381 | 3.859 |
| **Mean** | **0.085** | **1.707** | **3.606** |

Neural ODE is **20x better than BPTT** and **42x better than PA** on mean generalization error. All 5 held-out ICs are below the 0.5 success threshold. Time series plots show near-perfect tracking of ground truth for 5 full seconds on novel ICs.

The interp_hard IC (θ=2.5, ω=1.0) has the largest error (0.279) — this is near the edge of the training range and involves large-angle nonlinearity (sin(θ) deviates significantly from θ). Still tracks correctly but with slight phase drift by 5s.

**Vector field quality:** The learned derivatives match the true pendulum ODE closely:
```
Point (θ, ω)     True (dθ/dt, dω/dt)    Learned
(+1.0, +0.0)     (+0.000, -8.255)        (+0.012, -8.268)
(+0.0, +2.0)     (+2.000, -1.000)        (+2.000, -1.017)
(-1.0, -1.0)     (-1.000, +8.755)        (-0.982, +8.721)
(+2.5, +1.0)     (+1.000, -6.371)        (+0.968, -6.561)
```
The MLP has effectively learned dθ/dt ≈ ω and dω/dt ≈ -bω - (g/L)sin(θ) from trajectory data alone.

**Why Neural ODE works where RNNs fail:**
1. **Correct inductive bias:** Learns the derivative f(state) directly, not state→state mapping. The ODE solver handles time integration, so the network only needs to approximate the local vector field.
2. **No self-feeding instability:** There is no autoregressive loop — the ODE solver calls the network at each integration step with the solver's own state, not the network's previous output.
3. **Tiny model, huge gain:** 8,642 params vs 252,000 — the right architecture matters far more than scale.
4. **Generalizes by construction:** Learning dθ/dt = f(θ, ω) is a universal function approximation of the dynamics. Any IC just starts the solver at a different point in the same learned vector field.

**What this means for predictive alignment:**
- PA is well-suited for autonomous trajectory generation (memorizing attractors) but fundamentally cannot learn generalizable state-to-state dynamics via input-driven self-feeding
- The bottleneck is not capacity (N=500 has plenty) but the learning rule: PA's local Hebbian-like updates have no mechanism to optimize multi-step rollout stability
- A derivative-predicting variant of PA (if one could be designed) might bridge this gap, but it would require fundamentally different error signals than the current Q@z feedback

### Phase 2 overall conclusions

| Method | Params | Mean Gen. Error | Failure Mode |
|--------|--------|----------------|--------------|
| PA teacher forcing (07d) | 252k | 5.5 | Diverges to fixed point |
| PA multi-IC (07e) | 252k | 2.3 | Spurious limit cycle |
| PA + sched. sampling (2.1) | 252k | 3.6 | Diverges to fixed point |
| BPTT + sched. sampling (2.0) | 252k | 1.7 | Collapses to zero |
| **Neural ODE (2.5)** | **8.6k** | **0.085** | **None — tracks ground truth** |

The gap is 20x between Neural ODE and the best RNN method (BPTT), and 42x vs the best PA method. This is not a tuning issue — it's an architectural mismatch. The Neural ODE has the right inductive bias: learn the derivative, let the solver integrate. RNNs with readout bottlenecks and self-feeding loops are the wrong tool for learning generalizable dynamics from state observations.

### Open questions
1. Lyapunov exponent values from perturbation estimator remain positive after training. Paper reports shift toward negative (Supp Fig 4). May be a measurement method difference.
2. Could a PA-trained RNN learn the *derivative* instead of the next state? Would require modifying the target signal and readout interpretation.
3. Is the Neural ODE result robust to noise in training data? Real physical measurements have sensor noise.
4. Can Neural ODE scale to harder systems (Lotka-Volterra, double pendulum)?
5. The interp_hard IC (θ=2.5, ω=1.0) had 0.279 error — near the training range boundary. How does error scale with distance from training distribution?

### Experiment 2.3: Neural ODE on Lotka-Volterra (RUNNING)

Conservative 2D predator-prey system with multiplicative coupling:
```
dx/dt = αx - βxy,  dy/dt = δxy - γy
α=1.0, β=0.1, δ=0.075, γ=1.5, fixed point (20, 10)
```

Key differences from pendulum:
- Conservative (no damping) — orbits are closed, never decay
- Multiplicative xy coupling — harder nonlinearity than sin(θ)
- Positive-only state space — negative populations are nonsense
- Has conserved quantity V = δx - γln(x) + βy - αln(y)

Setup: Same MLP (2→64→64→64→2, 8.6k params). Training on random ICs from x∈[2,20], y∈[2,20], 30s trajectories (~3-4 full cycles). 500 epochs.

Test ICs (held out): near fixed point (15,8), small orbit (5,5), large orbit (25,15), low predator (10,3), high predator (3,18).

**Results: PARTIAL SUCCESS** — vector field learned accurately, but long-horizon integration drifts.

| IC | x₀ | y₀ | Error | Conservation drift |
|----|----|----|-------|--------------------|
| near_fp | 15.0 | 8.0 | 1.16 | 0.028 |
| small_orbit | 5.0 | 5.0 | 8.14 | 0.541 |
| large_orbit | 25.0 | 15.0 | 0.92 | 0.026 |
| low_predator | 10.0 | 3.0 | 4.78 | 0.306 |
| high_predator | 3.0 | 18.0 | 17.61 | 1.145 |
| **Mean** | | | **6.52** | |

Vector field quality (instantaneous derivatives) — learned accurately:
```
(10, 5): true=(+5.0, -3.75) learned=(+4.74, -3.59)
(30,15): true=(-15.0, +11.25) learned=(-15.27, +11.56)
(15, 3): true=(+10.5, -1.13) learned=(+10.51, -1.14)
```

**Key findings:**
- Near the training IC range center (near_fp, large_orbit): errors < 1.2, conservation drift < 0.03 — excellent
- Far from center or extreme predator-prey ratios (small_orbit, high_predator): errors 4.8–17.6, conservation leak up to 1.14
- The network *learned the vector field* (derivatives are accurate) but small per-step errors compound over 30s of integration (3000 steps)
- Lotka-Volterra is harder than pendulum: conservative orbits never converge to a fixed point, so integration errors accumulate without self-correction. Pendulum's damping acts as error-correcting — trajectories converge regardless of small deviations
- Mean error 6.52 vs pendulum's 0.085 — **77x worse**, confirming conservative systems are fundamentally harder for Neural ODEs without explicit symplectic structure

Training: 500 epochs, 83 min. Loss noisy early (spike to 39.5 at epoch 150) but converged. Checkpoint error: 47.8 → 1.16.

### Exp 2.4 — Double Pendulum (Neural ODE, 4D chaotic)

4D state (θ₁, θ₂, ω₁, ω₂), equal masses/lengths, g=9.81. MLP 4→128→128→128→4 (34k params). 500 epochs, dt=0.01s, 10s trajectories. Training ICs: θ∈[-1.5,1.5], ω∈[-2,2].

**Results: MODERATE SUCCESS** — short-horizon accurate, long-horizon diverges (expected for chaos).

| IC | θ₁ | θ₂ | ω₁ | ω₂ | Err 2s | Err 10s | Energy drift |
|----|----|----|----|----|--------|---------|-------------|
| small_angle | 0.5 | 0.5 | 0 | 0 | 0.41 | 1.48 | 7.6 |
| medium_angle | 1.0 | -0.5 | 0 | 0 | 0.36 | 2.48 | 8.0 |
| large_angle | 1.5 | 1.5 | 0 | 0 | 1.59 | 3.13 | 5.5 |
| chaotic_start | 2.0 | 2.0 | 0 | 0 | 2.76 | 3.22 | 7.4 |
| with_velocity | 0.8 | 0.3 | 1.0 | -1.0 | 0.55 | 4.36 | 23.4 |
| **Mean** | | | | | **1.13** | **2.94** | |

**Key findings:**
- Short-horizon (2s, ~2 oscillation periods) mean error 1.13 — usable for short-term prediction
- Long-horizon (10s) mean error 2.94 — better than Lotka-Volterra (6.52) despite being 4D and chaotic
- Chaotic IC (θ=2.0,2.0) shows expected rapid divergence: err_2s=2.76 is high because chaos amplifies small errors exponentially
- Vector field learned reasonably well at test points
- Energy drift significant (5–23) — no conservation structure enforced
- Training: 26 min, loss noisy due to chaotic trajectories

### Exp 2.3b — Hamiltonian Neural Network for Lotka-Volterra

Physics-informed architecture: MLP outputs scalar H(u,v) in log-coordinates (u=ln(x), v=ln(y)). Dynamics derived via torch.autograd: du/dt = -∂H/∂v, dv/dt = ∂H/∂u, transformed back to (x,y). Structurally guarantees H is conserved.

Training: derivative matching (Greydanus et al. 2019 approach) — train on finite-difference derivatives, not integrated trajectories. 500 epochs, 256 points per batch, 4 min training.

**Results: CONSERVATION WORKS, ACCURACY DOESN'T** — Hamiltonian structure reduces conservation drift but trajectory accuracy is worse than vanilla Neural ODE.

| IC | x₀ | y₀ | HNN err | NODE err | HNN drift | NODE drift |
|----|----|-----|---------|----------|-----------|------------|
| near_fp | 15.0 | 8.0 | 5.29 | 1.16 | **0.0007** | 0.028 |
| small_orbit | 5.0 | 5.0 | 26.17 | 8.14 | 0.453 | 0.541 |
| large_orbit | 25.0 | 15.0 | 7.03 | 0.92 | 0.315 | 0.026 |
| low_predator | 10.0 | 3.0 | 20.04 | 4.78 | 0.264 | 0.306 |
| high_predator | 3.0 | 18.0 | 59.39 | 17.61 | 0.802 | 1.145 |
| **Mean** | | | **23.58** | **6.52** | | |

**Key findings:**
- near_fp conservation drift: 0.0007 vs 0.028 — **40x improvement**. The Hamiltonian structure works.
- But trajectory errors are 3.6x worse (23.6 vs 6.5). The derivative matching training didn't converge well enough — loss still noisy.
- The log-coordinate transformation is physically correct (LV is canonical Hamiltonian in (ln(x), ln(y))), but the MLP struggles to learn the exponential structure H = δeᵘ - γu + βeᵛ - αv.
- Derivative matching trains faster (4 min vs 83 min) but gives worse results because it trains on pointwise derivatives rather than trajectory-level integration. The vanilla NODE integrates through segments during training, which provides stronger gradients.
- **Lesson**: Hamiltonian structure is the right idea but needs trajectory-level training (backprop through integrator) to get accurate enough derivatives. The original slow approach was correct — it just needs optimization (e.g., shorter segments, mixed precision, or adjoint method).

### Next steps
- Consider hybrid approach: HNN with trajectory-level training (short segments, efficient autograd)
- Run experiment 06 (RSG timing) when time permits (~4-6 hours)
- Consider PA + Neural ODE hybrid architecture
