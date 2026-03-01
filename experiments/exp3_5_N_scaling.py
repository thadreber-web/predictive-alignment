"""Experiment 3.5: N-scaling for 4-system continual learning.

Reuses exp3.4 protocol (sine→Lorenz→multi-sine→sawtooth, 15s each)
but sweeps N = [100, 200, 500, 1000, 2000].

Question: how does forgetting scale with network size N?
At small N, capacity should be insufficient and forgetting should emerge.
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

from src.targets import sine_target, generate_lorenz, multi_sine_target
from src.utils import set_seed

# ── Config ───────────────────────────────────────────────────────────
N_VALUES = [100, 200, 500, 1000, 2000]
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

TEST_MS = 10_000.0
TEST_STEPS = int(TEST_MS / DT)

RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "exp3_5_N_scaling")

TASKS = [
    {"name": "sine", "K": 1},
    {"name": "lorenz", "K": 3},
    {"name": "multi_sine", "K": 1},
    {"name": "sawtooth", "K": 1},
]

TASK_NAMES = [t["name"] for t in TASKS]
N_TASKS = len(TASKS)


# ── Reuse from exp3.4 ────────────────────────────────────────────────

class MultiTaskPA:
    """PA network that supports task-specific readouts sharing G and M."""

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

    def add_task(self, name, K):
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


def compute_error(predicted, truth, K):
    pred = predicted[:len(truth)]
    truth = truth[:len(pred)]
    if K == 1:
        return np.mean(np.abs(pred.ravel() - truth.ravel()))
    else:
        return np.mean(np.linalg.norm(pred - truth, axis=1))


def generate_sawtooth(t, period=800.0, amplitude=1.5):
    return amplitude * (2.0 * np.mod(t / period, 1.0) - 1.0)


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
    log = logging.getLogger("exp3.5")

    set_seed(SEED)
    device = torch.device(DEVICE)

    log.info("Exp 3.5: N-scaling for 4-system continual learning")
    log.info(f"N values: {N_VALUES}")
    log.info(f"Tasks: {TASK_NAMES}")
    log.info(f"Training: {TRAIN_MS/1000:.0f}s per task, Test: {TEST_MS/1000:.0f}s")
    log.info(f"Device: {DEVICE}")

    # ── Generate target data (shared across all N) ───────────────────
    total_steps = TRAIN_STEPS + TEST_STEPS
    t_all = np.arange(total_steps) * DT

    sine_data = sine_target(t_all, period=600.0, amplitude=1.5)
    lorenz_data = generate_lorenz(
        duration_ms=(TRAIN_MS + TEST_MS), dt=DT, scale=0.1,
    )
    multi_sine_data = multi_sine_target(
        t_all,
        frequencies=[1.0/600.0, 1.0/300.0, 1.0/900.0],
        amplitudes=[1.0, 0.5, 0.3],
    )
    sawtooth_data = generate_sawtooth(t_all, period=800.0, amplitude=1.5)

    task_data = {
        "sine": {"data": sine_data, "K": 1},
        "lorenz": {"data": lorenz_data, "K": 3},
        "multi_sine": {"data": multi_sine_data, "K": 1},
        "sawtooth": {"data": sawtooth_data, "K": 1},
    }

    test_truth = {}
    for name, td in task_data.items():
        if td["K"] == 1:
            test_truth[name] = td["data"][:TEST_STEPS].reshape(-1, 1)
        else:
            test_truth[name] = td["data"][:TEST_STEPS]

    # ── Sweep N values ───────────────────────────────────────────────
    # Results: forgetting_matrices[N] = (n_tasks, n_tasks) matrix
    #          forgetting_ratios[N] = dict {task_name: ratio}
    all_forgetting_matrices = {}
    all_forgetting_ratios = {}
    all_max_ratios = {}

    for N in N_VALUES:
        log.info(f"\n{'#'*70}")
        log.info(f"# N = {N}")
        log.info(f"{'#'*70}")

        t_start = time.time()

        net = MultiTaskPA(N=N, g=G_GAIN, tau=TAU, dt=DT,
                          eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
                          device=device, seed=SEED)
        for task in TASKS:
            net.add_task(task["name"], task["K"])

        forgetting_matrix = np.zeros((N_TASKS, N_TASKS))

        for phase_idx, task in enumerate(TASKS):
            name = task["name"]
            K = task["K"]
            log.info(f"  Phase {phase_idx+1}: Training on {name} (K={K})")

            net.set_active_task(name)
            data = task_data[name]["data"]

            for t in range(TRAIN_STEPS):
                if K == 1:
                    target = torch.tensor([data[t]], device=device, dtype=torch.float32)
                else:
                    target = torch.tensor(data[t], device=device, dtype=torch.float32)
                net.step_and_learn(target)

            # After this phase, test ALL tasks
            for j, test_task in enumerate(TASKS):
                test_name = test_task["name"]
                test_K = test_task["K"]
                pred = test_autonomous(net, test_name, TEST_STEPS)
                err = compute_error(pred, test_truth[test_name], test_K)
                forgetting_matrix[phase_idx, j] = err

            trained_errs = " | ".join(
                f"{TASK_NAMES[j]}={forgetting_matrix[phase_idx,j]:.4f}"
                for j in range(N_TASKS)
            )
            log.info(f"    Errors: {trained_errs}")

        elapsed = time.time() - t_start
        log.info(f"  N={N} done in {elapsed:.1f}s")

        # Compute forgetting ratios
        ratios = {}
        for j in range(N_TASKS):
            baseline = forgetting_matrix[j, j]
            final = forgetting_matrix[N_TASKS - 1, j]
            ratios[TASK_NAMES[j]] = final / baseline if baseline > 0 else float("nan")

        max_ratio = max(r for r in ratios.values() if not np.isnan(r))

        all_forgetting_matrices[N] = forgetting_matrix
        all_forgetting_ratios[N] = ratios
        all_max_ratios[N] = max_ratio

        log.info(f"  Forgetting ratios: {ratios}")
        log.info(f"  Max forgetting ratio: {max_ratio:.3f}")

    # ── Summary ──────────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("SUMMARY: N-scaling results")
    log.info(f"{'='*70}")
    header = f"{'N':>6} | {'max_ratio':>10}" + "".join(f" | {n:>12}" for n in TASK_NAMES)
    log.info(header)
    log.info("-" * len(header))
    for N in N_VALUES:
        ratios = all_forgetting_ratios[N]
        row = f"{N:>6} | {all_max_ratios[N]:>10.3f}"
        row += "".join(f" | {ratios[n]:>12.3f}" for n in TASK_NAMES)
        log.info(row)

    # ── Plot 1: Max forgetting ratio vs N ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(N_VALUES, [all_max_ratios[N] for N in N_VALUES], "ko-", lw=2, ms=8)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="No forgetting")
    ax.set_xlabel("Network size N")
    ax.set_ylabel("Max forgetting ratio (final / baseline)")
    ax.set_title("Exp 3.5: Max forgetting ratio vs network size")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/max_forgetting_vs_N.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Per-task forgetting ratio vs N ───────────────────────
    colors = {"sine": "#2ecc71", "lorenz": "#3498db", "multi_sine": "#e74c3c", "sawtooth": "#9b59b6"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in TASK_NAMES:
        ratios_for_task = [all_forgetting_ratios[N][name] for N in N_VALUES]
        ax.plot(N_VALUES, ratios_for_task, "o-", lw=2, ms=6,
                color=colors[name], label=name)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="No forgetting")
    ax.set_xlabel("Network size N")
    ax.set_ylabel("Forgetting ratio (final / baseline)")
    ax.set_title("Exp 3.5: Per-task forgetting ratio vs network size")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/per_task_forgetting_vs_N.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Forgetting matrices side-by-side ─────────────────────
    fig, axes = plt.subplots(1, len(N_VALUES), figsize=(4 * len(N_VALUES), 4))
    for idx, N in enumerate(N_VALUES):
        ax = axes[idx]
        mat = all_forgetting_matrices[N]
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(N_TASKS))
        ax.set_xticklabels([n[:4] for n in TASK_NAMES], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(N_TASKS))
        ax.set_yticklabels([f"after {n[:4]}" for n in TASK_NAMES], fontsize=7)
        ax.set_title(f"N={N}", fontsize=10)
        for i in range(N_TASKS):
            for j in range(N_TASKS):
                val = mat[i, j]
                color = "white" if val > mat.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=7)
    fig.suptitle("Exp 3.5: Forgetting matrices by network size", fontsize=12)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/forgetting_matrices_all_N.png", dpi=150)
    plt.close(fig)

    # Save results
    np.savez(f"{RESULTS_DIR}/results.npz",
             N_values=np.array(N_VALUES),
             max_ratios=np.array([all_max_ratios[N] for N in N_VALUES]),
             forgetting_ratios={N: all_forgetting_ratios[N] for N in N_VALUES},
             **{f"forgetting_matrix_N{N}": all_forgetting_matrices[N] for N in N_VALUES})

    log.info(f"\nAll results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
