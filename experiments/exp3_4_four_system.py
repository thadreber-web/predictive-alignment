"""Experiment 3.4: Four-system sequential stress test.

Sequential training: sine → Lorenz → multi-freq sine → sawtooth.
After each system, test ALL prior systems. Build forgetting matrix.

Question: does PA survive 4 sequential tasks without catastrophic forgetting?
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
N = 500
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

# Forgetting curve: test every N steps during each training phase
FORGET_TEST_EVERY = 500
FORGET_TEST_DURATION = 5_000.0
FORGET_TEST_STEPS = int(FORGET_TEST_DURATION / DT)

RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "exp3_4_four_system")

# Task definitions
TASKS = [
    {"name": "sine", "K": 1},
    {"name": "lorenz", "K": 3},
    {"name": "multi_sine", "K": 1},
    {"name": "sawtooth", "K": 1},
]


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

    def snapshot_M(self):
        return self.M.clone()


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
    """Sawtooth wave using np.mod."""
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
    log = logging.getLogger("exp3.4")

    set_seed(SEED)
    device = torch.device(DEVICE)

    log.info("Exp 3.4: Four-system sequential stress test")
    log.info(f"Tasks: {[t['name'] for t in TASKS]}")
    log.info(f"Training: {TRAIN_MS/1000:.0f}s per task, Test: {TEST_MS/1000:.0f}s")
    log.info(f"Device: {DEVICE}")

    # ── Generate all target data ──────────────────────────────────────
    total_steps = TRAIN_STEPS + TEST_STEPS + FORGET_TEST_STEPS
    t_all = np.arange(total_steps) * DT

    # 1. Sine
    sine_data = sine_target(t_all, period=600.0, amplitude=1.5)

    # 2. Lorenz
    lorenz_data = generate_lorenz(
        duration_ms=(TRAIN_MS + TEST_MS + FORGET_TEST_DURATION),
        dt=DT, scale=0.1,
    )

    # 3. Multi-freq sine (3 frequencies)
    multi_sine_data = multi_sine_target(
        t_all,
        frequencies=[1.0/600.0, 1.0/300.0, 1.0/900.0],  # cycles/ms
        amplitudes=[1.0, 0.5, 0.3],
    )

    # 4. Sawtooth
    sawtooth_data = generate_sawtooth(t_all, period=800.0, amplitude=1.5)

    # Map task names to data and K
    task_data = {
        "sine": {"data": sine_data, "K": 1},
        "lorenz": {"data": lorenz_data, "K": 3},
        "multi_sine": {"data": multi_sine_data, "K": 1},
        "sawtooth": {"data": sawtooth_data, "K": 1},
    }

    # Test truth for each task
    test_truth = {}
    for name, td in task_data.items():
        if td["K"] == 1:
            test_truth[name] = td["data"][:TEST_STEPS].reshape(-1, 1)
        else:
            test_truth[name] = td["data"][:TEST_STEPS]

    # ── Create network ────────────────────────────────────────────────
    net = MultiTaskPA(N=N, g=G_GAIN, tau=TAU, dt=DT,
                      eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
                      device=device, seed=SEED)
    for task in TASKS:
        net.add_task(task["name"], task["K"])

    # ── Sequential training with forgetting matrix ────────────────────
    task_names = [t["name"] for t in TASKS]
    n_tasks = len(task_names)

    # forgetting_matrix[i][j] = error on task j after training task i
    forgetting_matrix = np.zeros((n_tasks, n_tasks))

    # Per-system forgetting curves: curves[task_name] = list of (global_step, error)
    forgetting_curves = {name: [] for name in task_names}

    global_step = 0
    t_start = time.time()

    for phase_idx, task in enumerate(TASKS):
        name = task["name"]
        K = task["K"]
        log.info(f"\n{'='*60}")
        log.info(f"Phase {phase_idx+1}: Training on {name} (K={K})")
        log.info(f"{'='*60}")

        net.set_active_task(name)
        data = task_data[name]["data"]

        for t in range(TRAIN_STEPS):
            if K == 1:
                target = torch.tensor([data[t]], device=device, dtype=torch.float32)
            else:
                target = torch.tensor(data[t], device=device, dtype=torch.float32)
            net.step_and_learn(target)

            # Periodic forgetting test on all prior + current tasks
            if (t + 1) % FORGET_TEST_EVERY == 0:
                x_save = net.x.clone()
                r_save = net.r.clone()

                for test_name in task_names[:phase_idx + 1]:
                    test_K = task_data[test_name]["K"]
                    test_pred = test_autonomous(net, test_name, FORGET_TEST_STEPS)
                    if test_K == 1:
                        truth = task_data[test_name]["data"][:FORGET_TEST_STEPS].reshape(-1, 1)
                    else:
                        truth = task_data[test_name]["data"][:FORGET_TEST_STEPS]
                    err = compute_error(test_pred, truth, test_K)
                    forgetting_curves[test_name].append((global_step + t + 1, err))

                net.x = x_save
                net.r = r_save
                net.set_active_task(name)

        global_step += TRAIN_STEPS

        # After this phase, test ALL tasks seen so far
        log.info(f"\nTesting after Phase {phase_idx+1} ({name}):")
        for j, test_task in enumerate(TASKS):
            test_name = test_task["name"]
            test_K = test_task["K"]
            pred = test_autonomous(net, test_name, TEST_STEPS)
            err = compute_error(pred, test_truth[test_name], test_K)
            forgetting_matrix[phase_idx, j] = err
            trained = "trained" if j <= phase_idx else "untrained"
            log.info(f"  {test_name:>12} ({trained}): {err:.4f}")

    elapsed = time.time() - t_start
    log.info(f"\nTotal training time: {elapsed:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("FORGETTING MATRIX (error on task j after training through task i)")
    log.info(f"{'='*70}")
    header = f"{'Trained through':>18}" + "".join(f"{n:>12}" for n in task_names)
    log.info(header)
    log.info("-" * len(header))
    for i, name in enumerate(task_names):
        row = f"{name:>18}" + "".join(f"{forgetting_matrix[i,j]:>12.4f}" for j in range(n_tasks))
        log.info(row)

    # Compute forgetting ratios for trained tasks
    log.info(f"\nForgetting ratios (error after later training / error right after own training):")
    for j in range(n_tasks):
        baseline = forgetting_matrix[j, j]  # error right after own training
        if baseline > 0:
            final = forgetting_matrix[n_tasks - 1, j]  # error after all training
            ratio = final / baseline
            log.info(f"  {task_names[j]:>12}: {ratio:.3f}x (baseline={baseline:.4f}, final={final:.4f})")

    # ── Plots ─────────────────────────────────────────────────────────

    # 1. Forgetting matrix heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(forgetting_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels([f"after {n}" for n in task_names])
    ax.set_xlabel("Test Task")
    ax.set_ylabel("Trained Through")
    ax.set_title("Exp 3.4: Forgetting Matrix")
    for i in range(n_tasks):
        for j in range(n_tasks):
            val = forgetting_matrix[i, j]
            color = "white" if val > forgetting_matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10)
    plt.colorbar(im, ax=ax, label="Error")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/forgetting_matrix.png", dpi=150)
    plt.close(fig)

    # 2. Time series after all 4 tasks
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    t_test = np.arange(TEST_STEPS) * DT / 1000  # seconds

    for idx, task in enumerate(TASKS):
        name = task["name"]
        K = task["K"]
        ax = axes[idx]

        pred = test_autonomous(net, name, TEST_STEPS)
        err = forgetting_matrix[n_tasks - 1, idx]

        if K == 1:
            truth_plot = test_truth[name].ravel()
            pred_plot = pred.ravel()
            ax.plot(t_test, truth_plot, "k-", lw=1.5, label="Ground truth")
            ax.plot(t_test, pred_plot, "r-", lw=1, alpha=0.8, label=f"Predicted (err={err:.3f})")
        else:
            # For Lorenz, plot first component
            ax.plot(t_test, test_truth[name][:, 0], "k-", lw=1.5, label="Ground truth (x)")
            ax.plot(t_test, pred[:, 0], "r-", lw=1, alpha=0.8, label=f"Predicted (err={err:.3f})")

        ax.set_ylabel(name)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_title("Exp 3.4: All tasks after sequential training")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/time_series_all_tasks.png", dpi=150)
    plt.close(fig)

    # 3. Per-system forgetting curves throughout training
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {"sine": "#2ecc71", "lorenz": "#3498db", "multi_sine": "#e74c3c", "sawtooth": "#9b59b6"}
    phase_boundaries = [TRAIN_STEPS * (i + 1) for i in range(n_tasks)]

    for name in task_names:
        if forgetting_curves[name]:
            steps, errs = zip(*forgetting_curves[name])
            ax.plot(np.array(steps) * DT / 1000, errs, "-", lw=2,
                    color=colors.get(name, "gray"), label=name)

    for i, boundary in enumerate(phase_boundaries[:-1]):
        ax.axvline(x=boundary * DT / 1000, color="gray", linestyle="--", alpha=0.5)
        ax.text(boundary * DT / 1000, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1.0,
                f" → {task_names[i+1]}", fontsize=8, alpha=0.7, va="top")

    ax.set_xlabel("Total training time (s)")
    ax.set_ylabel("Error")
    ax.set_title("Exp 3.4: Per-system error throughout sequential training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/forgetting_curves.png", dpi=150)
    plt.close(fig)

    # Save results
    np.savez(f"{RESULTS_DIR}/results.npz",
             forgetting_matrix=forgetting_matrix,
             task_names=np.array(task_names))

    log.info(f"\nAll results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
