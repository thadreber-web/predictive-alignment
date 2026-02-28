# Phase 1: Predictive Alignment — Foundation Build

## What You're Building

A recurrent neural network (RNN) that learns to simulate dynamical systems using the Predictive Alignment learning rule from Asabuki & Clopath (2025). By the end of Phase 1, you will have:

1. A working implementation of predictive alignment in PyTorch
2. Instrumentation tools to see inside the network while it learns
3. A validated reproduction of the paper's core results
4. Your first neural simulator — a network that *is* a pendulum

---

## Prerequisites

### Hardware
- DGX Spark (GB10, 128GB unified memory)
- Phase 1 is small-scale (N ≤ 1000 neurons), so CPU is fine for all experiments
- GPU becomes relevant in Phase 4 when you scale to N = 50,000

### Software
```bash
# Create a conda environment
conda create -n pred_align python=3.11 -y
conda activate pred_align

# Core dependencies
pip install torch numpy scipy matplotlib seaborn tqdm

# For instrumentation and analysis
pip install jupyter ipywidgets tensorboard

# For phase portraits and dynamical systems
pip install scikit-learn
```

### Paper Reference
- Asabuki & Clopath, "Taming the chaos gently: a predictive alignment learning rule in recurrent neural networks", Nature Communications 16, 6784 (2025)
- DOI: 10.1038/s41467-025-61309-9
- Keep this open while implementing — you'll reference equations frequently

---

## Project Structure

```
predictive_alignment/
├── README.md
├── src/
│   ├── __init__.py
│   ├── network.py          # Core RNN + predictive alignment
│   ├── targets.py           # Target signal generators
│   ├── instrumentation.py   # Visualization and analysis tools
│   └── utils.py             # Helpers (Lyapunov, eigenspectrum, etc.)
├── experiments/
│   ├── 01_sine_wave.py
│   ├── 02_break_alpha.py
│   ├── 03_edge_of_chaos.py
│   ├── 04_eigenspectrum.py
│   ├── 05_lorenz.py
│   ├── 06_rsg_timing.py
│   └── 07_pendulum.py
├── notebooks/
│   ├── exploration.ipynb
│   └── analysis.ipynb
├── results/
│   └── (generated plots and data)
└── requirements.txt
```

---

## Step 1: Implement the Core Network

### File: `src/network.py`

This is the heart of everything. Build a class called `PredictiveAlignmentRNN` that contains:

### 1.1 — Network State

These are the data structures you need:

| Variable | Shape | Description |
|----------|-------|-------------|
| `x` | (N,) | Membrane potentials of all neurons |
| `r` | (N,) | Firing rates = tanh(x) |
| `G` | (N, N) | Fixed strong connections (generates chaos) |
| `M` | (N, N) | Plastic weak connections (learns) |
| `w` | (K, N) | Readout weights (K = number of outputs) |
| `Q` | (N, K) | Random feedback matrix (fixed) |
| `W_in` | (N, D) | Input weights (D = input dimension, fixed) |

### 1.2 — Initialization

This matters a lot. Get it wrong and the network either does nothing or explodes.

**G (fixed connections):**
- Sparse: only 10% of entries are nonzero
- Nonzero entries drawn from Gaussian with mean=0, std = g / sqrt(p * N)
- Default: g = 1.2, p = 0.1
- g > 1.0 is what makes the network chaotic before learning

**M (plastic connections):**
- Dense: 100% connectivity
- Entries drawn from Gaussian with mean=0, std = g_m / sqrt(N)
- Default: g_m = 0.5
- Starts weak — will grow during learning

**w (readout weights):**
- Initialize to zeros or very small random values
- Shape: (K, N) where K is number of readout units

**Q (feedback matrix):**
- Entries drawn uniformly from [-3/sqrt(K), 3/sqrt(K)]
- Fixed throughout training, never modified

**W_in (input weights):**
- Only needed for tasks with external input (RSG, input-output mapping)
- Entries drawn from Gaussian, fixed throughout training

### 1.3 — Forward Step (Network Dynamics)

One timestep of the network. This runs every dt (typically 0.1 ms):

```
Pseudocode for one timestep:
─────────────────────────────
r = tanh(x)                           # firing rates from potentials
J = G + M                             # total recurrent connectivity
input_current = J @ r                 # recurrent input to each neuron
if external_input exists:
    input_current += W_in @ external_input
dx = (-x + input_current) * (dt / tau)  # Euler integration
x = x + dx                            # update membrane potentials
z = w @ r                             # readout output (K values)
```

**Key parameters:**
- tau = 10 ms (membrane time constant)
- dt = 0.1 ms (integration timestep — needs to be much smaller than tau)

### 1.4 — Learning Step (The Predictive Alignment Update)

This runs every timestep during training, after the forward step:

```
Pseudocode for one learning step:
──────────────────────────────────
r = tanh(x)                           # current firing rates
z = w @ r                             # current output (K values)
f = target(t)                         # target signal at current time

# --- Readout weight update (standard delta rule) ---
output_error = f - z                   # shape: (K,)
dw = eta_w * outer(output_error, r)    # shape: (K, N)
w = w + dw

# --- Recurrent weight update (PREDICTIVE ALIGNMENT) ---
feedback = Q @ z                       # shape: (N,) — output fed back to neurons
J_hat_r = (M - alpha * G) @ r          # shape: (N,) — regularized prediction
rec_error = feedback - J_hat_r         # shape: (N,) — per-neuron prediction error
dM = eta_m * outer(rec_error, r)       # shape: (N, N)
M = M + dM
```

**Key parameters:**
- eta_w = learning rate for readout weights (start with 1e-3, tune later)
- eta_m = learning rate for recurrent weights (start with 1e-4, tune later)
- alpha = 1.0 (alignment regularization strength)

### 1.5 — Important Implementation Notes

**outer(a, b):** This produces an (N, N) matrix from two (N,) vectors. In PyTorch: `torch.outer(a, b)`. At N=500, this is a 500×500 = 250,000 element matrix computed every timestep. At N=500 this is trivial. At N=50,000 (Phase 4) this becomes the bottleneck.

**dt vs tau ratio:** The paper uses tau=10ms. Your dt should be 0.1ms or smaller. Larger dt means faster simulation but less accurate dynamics. If the network blows up (x values going to infinity), try smaller dt first.

**Learning rates:** The paper doesn't give exact values for all experiments. You'll need to tune these. Start with eta_w = 1e-3 and eta_m = 1e-4. If readout error plateaus high, increase eta_w. If the network becomes unstable during training, decrease eta_m.

**Simulation duration:** For periodic targets, train for at least 50-100 periods of the target signal. For a sine wave with period 600ms, that's 30-60 seconds of simulated time = 300,000-600,000 timesteps at dt=0.1ms.

---

## Step 2: Implement Instrumentation Tools

### File: `src/instrumentation.py`

Build these tools before running any experiments. You'll use them constantly.

### 2.1 — Training Monitor

A class that records quantities at regular intervals during training:

**Record every timestep (or every 10th):**
- Readout output z(t)
- Target signal f(t)
- Readout error |f(t) - z(t)|

**Record every N_snapshot timesteps (e.g., every 1000):**
- Snapshot of M (for eigenspectrum analysis)
- Alignment correlation: sum of elementwise (G @ r) * (M @ r) across neurons
- Frobenius norm of M: sqrt(sum of M_ij^2) — tracks how much M has grown
- Sample neural activities (10-20 random neurons' x and r values)

**Record once at start and end:**
- Full eigenspectrum of J = G + M
- Lyapunov exponent estimate

### 2.2 — Lyapunov Exponent Estimator

Measures whether the network is chaotic (positive) or stable (negative).

```
Pseudocode:
───────────
1. Take the current network state x
2. Create a copy x_perturbed = x + tiny random perturbation (magnitude 1e-6)
3. Run both copies forward for 1000 steps with identical input
4. At each step, measure distance = ||x - x_perturbed||
5. Before distance saturates, renormalize: x_perturbed = x + (x_perturbed - x) * (1e-6 / distance)
6. Lyapunov exponent = average of log(distance / 1e-6) across all steps
```

If Lyapunov > 0: chaotic. If Lyapunov < 0: stable. The learning should drive it from positive to negative.

### 2.3 — Eigenspectrum Plotter

```
1. Compute J = G + M
2. Compute eigenvalues of J (complex-valued in general)
3. Plot real part vs imaginary part in the complex plane
4. Before learning: should be a uniform disk (random matrix theory)
5. After learning: disk plus outlier eigenvalues with large real parts
```

Those outliers are the learned dynamics. Their imaginary parts relate to oscillation frequency, their real parts relate to stability/growth.

### 2.4 — Phase Portrait Plotter

For multi-output tasks (Lorenz, pendulum), plot output dimensions against each other:
- z1 vs z2
- z2 vs z3
- 3D trajectory (z1, z2, z3)

Compare network output trajectories to target trajectories.

### 2.5 — PCA State Space Visualizer

```
1. Collect neural activity vectors r(t) over many timesteps
2. Stack into matrix R of shape (T, N)
3. Compute PCA: get top 3 principal components
4. Project r(t) into 3D PCA space
5. Plot the trajectory — this shows the network's internal dynamics
```

Before learning: messy, space-filling chaotic trajectory.
After learning: clean, structured trajectory (limit cycle for periodic targets, strange attractor for chaotic targets).

---

## Step 3: Target Signal Generators

### File: `src/targets.py`

Implement functions that generate target signals for training.

### 3.1 — Sine Wave

```python
def sine_target(t, period=600.0, amplitude=1.5):
    """Single sine wave. Period in ms."""
    return amplitude * np.sin(2 * np.pi * t / period)
```

### 3.2 — Multiple Frequencies

```python
def multi_sine_target(t, frequencies, amplitudes):
    """Sum of sine waves at different frequencies."""
    return sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(frequencies, amplitudes))
```

### 3.3 — Lorenz Attractor

Generate by integrating the Lorenz equations ahead of time:

```
dx1/dt = 10 * (x2 - x1)
dx2/dt = 28 * x1 - x2 - x1 * x3
dx3/dt = x1 * x2 - (8/3) * x3
```

Use scipy.integrate.solve_ivp to generate a long trajectory. Store as a lookup table. Scale output by 1/10 (as the paper does) to keep values in a reasonable range for tanh neurons.

Three readout units, one per Lorenz variable.

### 3.4 — Ready-Set-Go Task

Two input channels send Gaussian pulses with variable delay T_delay between them. Target output is a Gaussian pulse delayed by T_delay after the second input pulse. This is trial-based: each trial has a random delay sampled from {100, 120, 140, 160} ms.

### 3.5 — Damped Pendulum

Generate by integrating:

```
dθ/dt = ω
dω/dt = -(b * ω) - (g/L) * sin(θ)
```

Parameters: b=0.5 (damping), g=9.81, L=1.0. Initial conditions: θ=2.0 rad, ω=0.0.

Two readout units: one for θ(t), one for ω(t).

---

## Step 4: Run the Experiments

### Experiment 01 — Single Sine Wave (validates core implementation)

**File:** `experiments/01_sine_wave.py`

**Setup:**
- N = 500 neurons
- K = 1 readout
- Target: sine wave, period = 600ms (60 * tau)
- Train for 30 seconds simulated time
- dt = 0.1ms, tau = 10ms
- eta_w = 1e-3, eta_m = 1e-4, alpha = 1.0

**What to run:**
1. Initialize network
2. Run training loop for 300,000 timesteps
3. Turn off plasticity (stop updating w and M)
4. Run for another 10 seconds to test autonomous generation

**What to plot:**
1. Output z(t) vs target f(t) during early training (first 600ms)
2. Output z(t) vs target f(t) during late training (last 600ms)
3. Output z(t) after plasticity off — does it still produce the sine wave?
4. Root mean squared error over time (should decrease monotonically)
5. 10 example neuron traces: early (chaotic) vs late (coherent)

**Success criteria:**
- Error decreases monotonically during training
- Output matches target well in late training
- Output persists after plasticity is turned off
- Neuron traces transition from chaotic to structured

**If it doesn't work:**
- Network blows up (NaN values) → reduce dt, reduce eta_m, check initialization
- Error doesn't decrease → increase eta_w, check that Q and feedback are connected correctly
- Output matches during training but fails after plasticity off → train longer, or M hasn't grown enough relative to G
- Output is flat/dead → g might be too low (network not chaotic enough) or too high (chaos too strong to tame)

---

### Experiment 02 — Break the Alignment (tests the alpha parameter)

**File:** `experiments/02_break_alpha.py`

**Setup:** Same as Experiment 01, but run 20 independent simulations for each of: alpha = 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0

**What to measure for each run:**
- Final readout error after training
- Lyapunov exponent after training
- Alignment correlation ⟨Gr, Mr⟩ after training

**What to plot:**
1. Mean ± std of final error vs alpha (reproduces Figure 3C)
2. Mean ± std of Lyapunov exponent vs alpha
3. Alignment correlation over training time for alpha=0 vs alpha=1 (reproduces Figure 3D)
4. Joint distribution scatter plot of (Gr)_i vs (Mr)_i for alpha=0 vs alpha=1 (reproduces Figure 3E)

**What this teaches you:**
- alpha=0: M predicts feedback but doesn't align with G. Works but noisy, unstable.
- alpha=1: M predicts feedback AND aligns with G. Chaos suppressed more efficiently.
- alpha too large: M is forced to mimic G so closely it can't learn new dynamics.
- There's a sweet spot. Finding it teaches you about regularization trade-offs.

---

### Experiment 03 — Edge of Chaos Sweep (tests the role of chaos)

**File:** `experiments/03_edge_of_chaos.py`

**Setup:** Same target (sine wave), but vary the gain g of the fixed connections G.

Run 20 independent simulations for each of: g = 0.5, 0.7, 0.9, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0

**What to measure:**
- Final readout error
- Final recurrent prediction error ||Qz - Mr||
- Frobenius norm of w after training (large = fragile)
- Frobenius norm of M after training (large = fragile)
- Lyapunov exponent before training (should cross zero near g=1.0)

**What to plot:**
1. Error vs g (should show minimum near g=1.0-1.1)
2. Weight norms vs g (should show minimum near edge of chaos)
3. Lyapunov exponent before training vs g (should cross zero near g=1.0)
4. The "Efficiency" metric from the paper: eigenvalue entropy / sqrt(participation ratio) vs g

**What this teaches you:**
- g < 1: Network is subcritical. Activity decays. Not enough basis functions for learning.
- g ≈ 1: Edge of chaos. Rich but structured dynamics. Optimal learning.
- g >> 1: Deep chaos. Too much disorder, can't be tamed.
- This is a fundamental principle of dynamical systems used for computation.

---

### Experiment 04 — Eigenspectrum Analysis (look inside the weight matrix)

**File:** `experiments/04_eigenspectrum.py`

**Setup:** Train a network on sine wave (Experiment 01 setup).

**What to do:**
1. Before training: compute eigenvalues of J = G + M, plot in complex plane
2. At 25%, 50%, 75% of training: snapshot J, compute and plot eigenvalues
3. After training: final eigenvalue plot

**What you should see:**
- Before: eigenvalues fill a circle of radius ~g in the complex plane (Circular Law from random matrix theory)
- After: most eigenvalues still in the circle, but a few outliers appear with large real parts
- Those outliers correspond to the learned low-dimensional dynamics
- For a sine wave target: expect one conjugate pair of outliers with imaginary part ≈ 2π/period

**Additional analysis:**
- Compute rank of M at each snapshot (use singular value decay — how many singular values are significant?)
- Plot singular value spectrum of M before and after training
- If M ends up low-rank, that tells you the learned dynamics are low-dimensional even though M is a full N×N matrix

---

### Experiment 05 — Lorenz Attractor (complex chaotic target)

**File:** `experiments/05_lorenz.py`

**Setup:**
- N = 500 neurons, K = 3 readouts
- Pre-generate Lorenz trajectory: 15,000 seconds at dt=0.1ms, scaled by 1/10
- Train for full 15,000 seconds
- Then turn off plasticity and let run

**What to plot:**
1. Each Lorenz component: target vs output during late training (Figure 4A)
2. Three 2D projections: (z1,z2), (z2,z3), (z1,z3) for target and output (Figure 4C-E)
3. 3D trajectory plot for target and output (Figure 4F-G)
4. Tent map: plot successive local maxima of z3 component (Figure 4H)
5. Error over time during training

**What to expect after plasticity off:**
- Output will track target for a while, then diverge (it's a chaotic system — small errors grow)
- But the statistical properties should match — the attractor shape, the tent map, the oscillation characteristics
- The network learned the *attractor*, not the specific trajectory

**This is your first real result:** a 500-neuron network that has internalized the dynamics of a chaotic system and can autonomously generate trajectories on the Lorenz attractor.

---

### Experiment 06 — Ready-Set-Go Timing Task (temporal memory)

**File:** `experiments/06_rsg_timing.py`

**Setup:**
- N = 1200 neurons, K = 1 readout, D = 2 input channels
- Delay values: {100, 120, 140, 160} ms
- Train for 200,000 trials (trial-based, not continuous)
- Each trial: present two input pulses with delay T_delay, target is output pulse at 2*T_delay after first pulse

**What to plot:**
1. Output for each trained delay value (Figure 5B colored squares)
2. Test interpolation: present delays the network never saw (110, 130, 150 ms). Does it produce the right output timing?
3. Test extrapolation: present delays outside training range (80, 180, 200 ms). Does it fail? (Figure 5C)
4. PCA of network activity for different delays (Figure 5D)

**What this teaches you:**
- The network can do temporal reasoning — measure a time interval, hold it in memory, reproduce it
- It can interpolate (generalize within training range) but not extrapolate (generalize beyond)
- PCA reveals the network learned a manifold where delay is encoded as a linear shift along a specific direction
- This is working memory implemented in recurrent dynamics

---

### Experiment 07 — Damped Pendulum (your first neural simulator)

**File:** `experiments/07_pendulum.py`

**Setup:**
- N = 500 neurons, K = 2 readouts (θ and ω)
- Generate pendulum trajectory from initial condition (θ=2.0, ω=0.0) for 5 seconds
- Train for 50 seconds (show the same trajectory 10 times)
- Test: after training, kick off from the same initial condition with plasticity off

**What to plot:**
1. θ(t) target vs output
2. ω(t) target vs output
3. Phase portrait: θ vs ω for target and output (should see spiral toward equilibrium)
4. Energy over time: E = 0.5*ω² + g/L*(1-cos(θ)) for both target and output (should decrease due to damping)

**Why the pendulum matters:**
- It's a real physical system with meaningful variables
- The phase portrait tells you instantly whether the network captured the dynamics
- Energy conservation (or dissipation) is a physics constraint the network has to implicitly learn
- It's your bridge to Phase 2, where you'll train on progressively harder physical systems

**Extension (if time permits):**
- Train on multiple initial conditions simultaneously
- Test on initial conditions the network never saw
- Train on pendulums with different lengths L (using input signals to encode L)

---

## Step 5: Analysis and Documentation

After running all experiments, create a summary notebook (`notebooks/analysis.ipynb`) that:

### 5.1 — Compiles key results

For each experiment, the essential plot and the one-sentence finding.

### 5.2 — Documents surprises

What didn't match your expectations? What broke? What parameter tuning was required?

### 5.3 — Records the parameter settings that worked

Create a reference table of hyperparameters that produced good results for each experiment. You'll need this in later phases.

### 5.4 — Identifies open questions

What do you want to explore further? What seemed promising? What seemed like a dead end?

### 5.5 — Outlines Phase 2 modifications

Based on what you learned, what changes does the codebase need before Phase 2 (dynamical system identification with Lotka-Volterra, double pendulum)?

---

## Estimated Timeline

| Task | Duration | Notes |
|------|----------|-------|
| Environment setup | 1 hour | conda, packages, project structure |
| Core network class | 4-6 hours | The most important code you'll write |
| Instrumentation tools | 3-4 hours | Worth the investment — used in every experiment |
| Target generators | 2-3 hours | Straightforward numerical integration |
| Experiment 01 (sine) | 2-3 hours | Includes debugging time |
| Experiment 02 (alpha) | 2-3 hours | Mostly automated once 01 works |
| Experiment 03 (edge of chaos) | 2-3 hours | Mostly automated once 01 works |
| Experiment 04 (eigenspectrum) | 2-3 hours | Analysis-heavy |
| Experiment 05 (Lorenz) | 3-4 hours | Harder target, may need tuning |
| Experiment 06 (RSG) | 4-6 hours | Trial-based architecture, most complex |
| Experiment 07 (pendulum) | 2-3 hours | Straightforward if Lorenz works |
| Analysis notebook | 3-4 hours | Synthesis and documentation |
| **Total** | **~30-40 hours** | **Roughly 2 weeks at research-sprint pace** |

---

## Key Equations Reference Card

Keep this visible while implementing.

### Network dynamics (Eq. 1)
```
τ * dx/dt = -x + (G + M) @ r + W_in @ I + σξ
r = tanh(x)
z = w @ r
```

### Readout weight update (Eq. 4)
```
Δw = η_w * (f - z) ⊗ r      (outer product, shape K×N)
```

### Recurrent weight update — Predictive Alignment (Eq. 6)
```
Δm = η_m * (Q @ z - Ĵ @ r) ⊗ r   (outer product, shape N×N)

where Ĵ = M - α * G
```

### Cost function for recurrent weights (Eq. 5, for understanding only — you implement the gradient, not the cost)
```
L_rec = (1/2T) ∫ ||Qz - Mr||² dt  −  (α/T) ∫ (Gr)ᵀ(Mr) dt
         ↑ predict feedback              ↑ align with chaos
```

### Lyapunov exponent (Eq. 23-25)
```
λ = average over k of: log(γ_k / γ_0)

where γ_k = ||x_perturbed(k) - x(k)||
      γ_0 = 1e-6 (initial perturbation magnitude)
```

---

## Troubleshooting Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| x values explode to NaN | dt too large, or eta_m too large | Reduce dt to 0.05ms; reduce eta_m by 10x |
| Error doesn't decrease at all | Learning rates too small, or Q wired incorrectly | Increase eta_w; verify Q @ z has the right shape |
| Error decreases then increases | eta_m too large (M overshoots) | Reduce eta_m |
| Output matches during training, dies after plasticity off | M hasn't learned stable dynamics yet | Train longer; increase alpha |
| All neurons converge to same activity | G initialization wrong (not sparse, or g too low) | Check sparsity and gain of G |
| Network is stable before training (should be chaotic) | g < 1.0 | Increase g to 1.2 (default) |
| Training takes forever | tau/dt ratio too large; N too large for CPU | Use dt=0.1ms with tau=10ms; N=500 is fine for Phase 1 |
| Lorenz output matches trajectory but not attractor shape | Need to train longer — 15,000 seconds per the paper | Be patient; Lorenz needs a lot of training data |

---

## What Success Looks Like

When Phase 1 is done, you should be able to:

1. **Show someone a plot** of chaotic neural activity transforming into coherent patterns during training
2. **Explain why alpha matters** with your own experimental data
3. **Demonstrate the edge of chaos** with a parameter sweep showing optimal learning near g=1.0
4. **Show eigenvalue outliers** that correspond to learned dynamics emerging from random matrix structure
5. **Generate a Lorenz attractor** from a trained network running autonomously
6. **Show a network that tells time** by interpolating between trained delay intervals
7. **Run a neural pendulum** and compare its phase portrait to real pendulum physics

Each of these is a concrete, visual, demonstrable result that proves you understand what predictive alignment does and have the tools to extend it in Phase 2.