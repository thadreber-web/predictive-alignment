"""Experiment 3.6: Capacity limit — how many tasks before forgetting emerges?

Sweep number of tasks (4, 8, 12, 16, 20, 30, 50) at N=500, 1000, 2000.
Task pool: diverse K=1 waveforms (sine, sawtooth, triangle, square, chirp
at various frequencies/params). Find the threshold where max forgetting
ratio exceeds some threshold (e.g. 1.5x).

Plot: capacity curve — max tasks before forgetting vs N.
"""

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

import torch
import numpy as np
import math
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from src.utils import set_seed

# ── Config ───────────────────────────────────────────────────────────
N_VALUES = [500, 1000, 2000]
TASK_COUNTS = [4, 8, 12, 16, 20, 30, 50]

TAU = 10.0
DT = 1.0
ETA_W = 1e-3
ETA_M = 1e-3
ALPHA = 1.0
G_GAIN = 1.2
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_MS = 15_000.0
TRAIN_STEPS = int(TRAIN_MS / DT)

TEST_MS = 5_000.0  # shorter test to keep runtime manageable
TEST_STEPS = int(TEST_MS / DT)

RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "exp3_6_capacity_limit")

# ── Waveform generators (all K=1, all return 1D arrays) ─────────────

def gen_sine(t, period, amplitude=1.5):
    return amplitude * np.sin(2 * np.pi * t / period)

def gen_sawtooth(t, period, amplitude=1.5):
    return amplitude * (2.0 * np.mod(t / period, 1.0) - 1.0)

def gen_triangle(t, period, amplitude=1.5):
    phase = np.mod(t / period, 1.0)
    return amplitude * (4.0 * np.abs(phase - 0.5) - 1.0)

def gen_square(t, period, amplitude=1.5):
    return amplitude * np.sign(np.sin(2 * np.pi * t / period))

def gen_chirp(t, period_start, period_end, amplitude=1.5):
    """Linear chirp: frequency sweeps from 1/period_start to 1/period_end."""
    f0 = 1.0 / period_start
    f1 = 1.0 / period_end
    t_max = t[-1] if hasattr(t, '__len__') else t
    rate = (f1 - f0) / t_max
    phase = 2 * np.pi * (f0 * t + 0.5 * rate * t**2)
    return amplitude * np.sin(phase)

def gen_am_sine(t, carrier_period, mod_period, amplitude=1.5):
    """Amplitude-modulated sine."""
    carrier = np.sin(2 * np.pi * t / carrier_period)
    modulator = 0.5 * (1.0 + np.sin(2 * np.pi * t / mod_period))
    return amplitude * carrier * modulator

def gen_rectified_sine(t, period, amplitude=1.5):
    """Full-wave rectified sine (always positive, shifted to zero mean)."""
    return amplitude * (np.abs(np.sin(2 * np.pi * t / period)) - 2/np.pi)


def build_task_pool(n_steps, dt=1.0):
    """Build a pool of 50 diverse K=1 waveforms.

    Returns list of (name, data_array) tuples.
    """
    t = np.arange(n_steps) * dt
    tasks = []

    # Sine waves at different periods
    for period in [200, 400, 600, 800, 1000, 1200, 1500, 2000]:
        tasks.append((f"sine_p{period}", gen_sine(t, period)))

    # Sawtooth at different periods
    for period in [300, 600, 900, 1200, 1800]:
        tasks.append((f"saw_p{period}", gen_sawtooth(t, period)))

    # Triangle at different periods
    for period in [250, 500, 750, 1000, 1500]:
        tasks.append((f"tri_p{period}", gen_triangle(t, period)))

    # Square waves at different periods
    for period in [400, 800, 1200, 1600]:
        tasks.append((f"sq_p{period}", gen_square(t, period)))

    # Chirps
    tasks.append(("chirp_200_800", gen_chirp(t, 200, 800)))
    tasks.append(("chirp_400_1200", gen_chirp(t, 400, 1200)))
    tasks.append(("chirp_600_1800", gen_chirp(t, 600, 1800)))

    # AM sine waves
    tasks.append(("am_300_3000", gen_am_sine(t, 300, 3000)))
    tasks.append(("am_500_5000", gen_am_sine(t, 500, 5000)))
    tasks.append(("am_400_2000", gen_am_sine(t, 400, 2000)))

    # Rectified sine
    for period in [400, 800, 1200]:
        tasks.append((f"rect_p{period}", gen_rectified_sine(t, period)))

    # Multi-sine (sum of 2 frequencies)
    tasks.append(("ms_200_600", gen_sine(t, 200, 0.8) + gen_sine(t, 600, 0.7)))
    tasks.append(("ms_300_900", gen_sine(t, 300, 0.8) + gen_sine(t, 900, 0.7)))
    tasks.append(("ms_400_1200", gen_sine(t, 400, 0.8) + gen_sine(t, 1200, 0.7)))
    tasks.append(("ms_500_1500", gen_sine(t, 500, 0.8) + gen_sine(t, 1500, 0.7)))

    # Sine + sawtooth combos
    tasks.append(("sin_saw_400", gen_sine(t, 400, 0.8) + gen_sawtooth(t, 800, 0.5)))
    tasks.append(("sin_saw_600", gen_sine(t, 600, 0.8) + gen_sawtooth(t, 1200, 0.5)))

    # Clipped sine (soft-clipped)
    tasks.append(("clip_sine_400", 1.5 * np.tanh(2.0 * np.sin(2 * np.pi * t / 400))))
    tasks.append(("clip_sine_800", 1.5 * np.tanh(2.0 * np.sin(2 * np.pi * t / 800))))

    # Exponential decay bursts
    burst_period = 2000.0
    phase = np.mod(t, burst_period) / burst_period
    tasks.append(("burst_2000", 1.5 * np.exp(-5.0 * phase) * np.sin(2 * np.pi * phase * 10)))

    # More sine waves at different periods
    for period in [350, 550, 1400]:
        tasks.append((f"sine_p{period}", gen_sine(t, period)))

    # More triangle waves
    for period in [350, 2000]:
        tasks.append((f"tri_p{period}", gen_triangle(t, period)))

    # Phase-shifted sine pairs
    tasks.append(("cos_p600", gen_sine(t, 600) * 0 + 1.5 * np.cos(2 * np.pi * t / 600)))
    tasks.append(("cos_p1000", 1.5 * np.cos(2 * np.pi * t / 1000)))

    # More chirps
    tasks.append(("chirp_300_1500", gen_chirp(t, 300, 1500)))

    # Sawtooth + triangle combo
    tasks.append(("saw_tri_500", gen_sawtooth(t, 500, 0.8) + gen_triangle(t, 1000, 0.5)))
    tasks.append(("saw_tri_800", gen_sawtooth(t, 800, 0.8) + gen_triangle(t, 1600, 0.5)))

    assert len(tasks) >= 50, f"Only {len(tasks)} tasks, need 50"
    return tasks[:50]


# ── Network (reused from exp3.4/3.5) ────────────────────────────────

class MultiTaskPA:
    def __init__(self, N, g, tau, dt, eta_w, eta_m, alpha, device, seed):
        self.N = N
        self.g = g
        self.tau = tau
        self.dt = dt
        self.eta_w = eta_w
        self.eta_m = eta_m
        self.alpha = alpha
        self.device = torch.device(device)
        self.p = 0.1

        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        self.rng = rng

        std_g = g / math.sqrt(self.p * N)
        G = torch.randn(N, N, device=self.device, generator=rng) * std_g
        mask = torch.rand(N, N, device=self.device, generator=rng) < self.p
        self.G = G * mask.float()
        self.G.fill_diagonal_(0.0)

        std_m = 0.01 / math.sqrt(N)
        self.M = torch.randn(N, N, device=self.device, generator=rng) * std_m

        self.tasks = {}
        self.active_task = None
        self.x = torch.randn(N, device=self.device, generator=rng) * 0.1
        self.r = torch.tanh(self.x)

    def add_task(self, name, K=1):
        w = torch.zeros(K, self.N, device=self.device)
        q_bound = 3.0 / math.sqrt(K)
        Q = torch.empty(self.N, K, device=self.device).uniform_(-q_bound, q_bound)
        self.tasks[name] = {"w": w, "Q": Q, "K": K}

    def set_active_task(self, name):
        self.active_task = name

    def reset_state(self):
        self.x = torch.randn(self.N, device=self.device, generator=self.rng) * 0.1
        self.r = torch.tanh(self.x)

    def step(self):
        task = self.tasks[self.active_task]
        self.r = torch.tanh(self.x)
        J = self.G + self.M
        current = J @ self.r
        dx = (-self.x + current) * (self.dt / self.tau)
        self.x = self.x + dx
        z = task["w"] @ self.r
        return z

    def step_and_learn(self, target):
        task = self.tasks[self.active_task]
        z = self.step()
        output_error = target - z
        task["w"] = task["w"] + self.eta_w * torch.outer(output_error, self.r)
        feedback = task["Q"] @ z
        J_hat_r = (self.M - self.alpha * self.G) @ self.r
        rec_error = feedback - J_hat_r
        self.M = self.M + self.eta_m * torch.outer(rec_error, self.r)
        return z


def test_autonomous(net, task_name, test_steps):
    net.set_active_task(task_name)
    net.reset_state()
    zs = []
    for _ in range(test_steps):
        z = net.step()
        zs.append(z.detach().cpu().numpy())
    return np.array(zs)


def compute_error(predicted, truth):
    pred = predicted[:len(truth)]
    truth = truth[:len(pred)]
    return np.mean(np.abs(pred.ravel() - truth.ravel()))


def run_capacity_test(N, n_tasks, task_pool, device, log):
    """Train n_tasks sequentially, return max forgetting ratio."""
    tasks = task_pool[:n_tasks]
    task_names = [t[0] for t in tasks]

    net = MultiTaskPA(N=N, g=G_GAIN, tau=TAU, dt=DT,
                      eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
                      device=device, seed=SEED)
    for name, _ in tasks:
        net.add_task(name, K=1)

    # Test truth
    test_truth = {name: data[:TEST_STEPS].reshape(-1, 1) for name, data in tasks}

    # forgetting_matrix[i][j] = error on task j after training task i
    forgetting_matrix = np.zeros((n_tasks, n_tasks))

    for phase_idx, (name, data) in enumerate(tasks):
        net.set_active_task(name)
        for t in range(TRAIN_STEPS):
            target = torch.tensor([data[t]], device=device, dtype=torch.float32)
            net.step_and_learn(target)

        # Test all tasks after this phase
        for j, (test_name, _) in enumerate(tasks):
            pred = test_autonomous(net, test_name, TEST_STEPS)
            forgetting_matrix[phase_idx, j] = compute_error(pred, test_truth[test_name])

    # Compute forgetting ratios (skip last task — always 1.0)
    ratios = []
    for j in range(n_tasks - 1):  # exclude last task
        baseline = forgetting_matrix[j, j]
        final = forgetting_matrix[n_tasks - 1, j]
        if baseline > 0:
            ratios.append(final / baseline)

    max_ratio = max(ratios) if ratios else 1.0
    mean_ratio = np.mean(ratios) if ratios else 1.0

    return max_ratio, mean_ratio, forgetting_matrix


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(f"{RESULTS_DIR}/experiment.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("exp3.6")

    set_seed(SEED)
    device = torch.device(DEVICE)

    log.info("Exp 3.6: Capacity limit — tasks vs N scaling")
    log.info(f"N values: {N_VALUES}")
    log.info(f"Task counts: {TASK_COUNTS}")
    log.info(f"Train: {TRAIN_MS/1000:.0f}s per task, Test: {TEST_MS/1000:.0f}s")
    log.info(f"Device: {DEVICE}")

    # Build task pool (need enough steps for training + testing)
    total_steps = TRAIN_STEPS + TEST_STEPS
    task_pool = build_task_pool(total_steps, dt=DT)
    log.info(f"Task pool: {len(task_pool)} waveforms")
    log.info(f"Tasks: {[t[0] for t in task_pool]}")

    # Results storage
    # results[N][n_tasks] = (max_ratio, mean_ratio)
    results = {N: {} for N in N_VALUES}
    matrices = {}

    for N in N_VALUES:
        log.info(f"\n{'#'*70}")
        log.info(f"# N = {N}")
        log.info(f"{'#'*70}")

        for n_tasks in TASK_COUNTS:
            if n_tasks > len(task_pool):
                log.info(f"  Skipping n_tasks={n_tasks} (only {len(task_pool)} in pool)")
                continue

            t_start = time.time()
            log.info(f"  n_tasks={n_tasks}...")

            max_r, mean_r, fmat = run_capacity_test(N, n_tasks, task_pool, device, log)
            elapsed = time.time() - t_start

            results[N][n_tasks] = (max_r, mean_r)
            matrices[(N, n_tasks)] = fmat

            log.info(f"    max_ratio={max_r:.3f}, mean_ratio={mean_r:.3f}, time={elapsed:.1f}s")

    # ── Summary ──────────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("SUMMARY")
    log.info(f"{'='*70}")
    header = f"{'n_tasks':>8}" + "".join(f" | N={N:>5} max  mean" for N in N_VALUES)
    log.info(header)
    log.info("-" * len(header))
    for n_tasks in TASK_COUNTS:
        row = f"{n_tasks:>8}"
        for N in N_VALUES:
            if n_tasks in results[N]:
                max_r, mean_r = results[N][n_tasks]
                row += f" | {max_r:>10.3f} {mean_r:>5.3f}"
            else:
                row += f" | {'---':>10} {'---':>5}"
        log.info(row)

    # ── Plot 1: Max forgetting ratio vs n_tasks, one line per N ──────
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_N = {500: "#e74c3c", 1000: "#3498db", 2000: "#2ecc71"}
    for N in N_VALUES:
        counts = sorted(results[N].keys())
        max_ratios = [results[N][c][0] for c in counts]
        ax.plot(counts, max_ratios, "o-", lw=2, ms=7,
                color=colors_N[N], label=f"N={N}")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No forgetting")
    ax.axhline(y=1.5, color="orange", linestyle=":", alpha=0.5, label="1.5x threshold")
    ax.set_xlabel("Number of sequential tasks")
    ax.set_ylabel("Max forgetting ratio")
    ax.set_title("Exp 3.6: Capacity limit — max forgetting vs number of tasks")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/capacity_curve.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Mean forgetting ratio vs n_tasks ─────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for N in N_VALUES:
        counts = sorted(results[N].keys())
        mean_ratios = [results[N][c][1] for c in counts]
        ax.plot(counts, mean_ratios, "o-", lw=2, ms=7,
                color=colors_N[N], label=f"N={N}")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No forgetting")
    ax.set_xlabel("Number of sequential tasks")
    ax.set_ylabel("Mean forgetting ratio")
    ax.set_title("Exp 3.6: Mean forgetting ratio vs number of tasks")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/mean_forgetting_curve.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Capacity threshold vs N ──────────────────────────────
    # Find max tasks where max_ratio < 1.5 for each N
    fig, ax = plt.subplots(figsize=(8, 5))
    thresholds = {1.25: [], 1.5: [], 2.0: []}
    for thresh in thresholds:
        for N in N_VALUES:
            max_tasks = 0
            for n_tasks in sorted(results[N].keys()):
                if results[N][n_tasks][0] < thresh:
                    max_tasks = n_tasks
            thresholds[thresh].append(max_tasks)

    thresh_colors = {1.25: "#e74c3c", 1.5: "#f39c12", 2.0: "#27ae60"}
    for thresh, capacities in thresholds.items():
        ax.plot(N_VALUES, capacities, "o-", lw=2, ms=8,
                color=thresh_colors[thresh], label=f"threshold={thresh}x")

    ax.set_xlabel("Network size N")
    ax.set_ylabel("Max tasks before forgetting")
    ax.set_title("Exp 3.6: Capacity (max tasks) vs network size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/capacity_vs_N.png", dpi=150)
    plt.close(fig)

    # Save results
    save_dict = {
        "N_values": np.array(N_VALUES),
        "task_counts": np.array(TASK_COUNTS),
    }
    for N in N_VALUES:
        for n_tasks in results[N]:
            max_r, mean_r = results[N][n_tasks]
            save_dict[f"max_ratio_N{N}_T{n_tasks}"] = max_r
            save_dict[f"mean_ratio_N{N}_T{n_tasks}"] = mean_r
    np.savez(f"{RESULTS_DIR}/results.npz", **save_dict)

    log.info(f"\nAll results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
