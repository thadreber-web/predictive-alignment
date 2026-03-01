"""Experiment 3.8: Q ablation — is Q orthogonality the key mechanism?

Three conditions on the same 4-task protocol:
1. Normal PA: independent random Q per task (baseline)
2. Shared Q: same Q matrix for all tasks
3. Shared Q, shared w: same Q AND same w for all tasks (extreme ablation)

If forgetting appears with shared Q: Q orthogonality is the key mechanism.
If shared Q still works: M partitions itself independently of Q.
"""

import sys
sys.path.insert(0, "/raid/predictive_alignment")

import torch
import numpy as np
import math
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
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

RESULTS_DIR = "/raid/predictive_alignment/results/exp3_8_Q_ablation"

TASKS = [
    {"name": "sine", "K": 1},
    {"name": "lorenz", "K": 3},
    {"name": "multi_sine", "K": 1},
    {"name": "sawtooth", "K": 1},
]
TASK_NAMES = [t["name"] for t in TASKS]
N_TASKS = len(TASKS)


class MultiTaskPA_QAblation:
    """PA network with configurable Q sharing."""

    def __init__(self, N, g, tau, dt, eta_w, eta_m, alpha, device, seed,
                 q_mode="independent"):
        self.N = N
        self.g = g
        self.tau = tau
        self.dt = dt
        self.eta_w = eta_w
        self.eta_m = eta_m
        self.alpha = alpha
        self.device = torch.device(device)
        self.p = 0.1
        self.q_mode = q_mode

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

        # Pre-generate shared Q matrices for each K value
        self._shared_Q = {}
        self._shared_w = {}

    def add_task(self, name, K):
        if self.q_mode == "independent":
            w = torch.zeros(K, self.N, device=self.device)
            q_bound = 3.0 / math.sqrt(K)
            Q = torch.empty(self.N, K, device=self.device).uniform_(-q_bound, q_bound)
        elif self.q_mode == "shared_Q":
            w = torch.zeros(K, self.N, device=self.device)
            if K not in self._shared_Q:
                q_bound = 3.0 / math.sqrt(K)
                self._shared_Q[K] = torch.empty(self.N, K, device=self.device).uniform_(-q_bound, q_bound)
            Q = self._shared_Q[K]  # same Q object shared across tasks with same K
        elif self.q_mode == "shared_Q_w":
            if K not in self._shared_Q:
                q_bound = 3.0 / math.sqrt(K)
                self._shared_Q[K] = torch.empty(self.N, K, device=self.device).uniform_(-q_bound, q_bound)
                self._shared_w[K] = torch.zeros(K, self.N, device=self.device)
            Q = self._shared_Q[K]
            w = self._shared_w[K]  # same w object shared
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
        # For shared_Q_w, this modifies the shared w in-place via +=
        if self.q_mode == "shared_Q_w":
            self._shared_w[task["K"]] = task["w"]
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


def run_condition(q_mode, task_data, test_truth, device, log):
    net = MultiTaskPA_QAblation(N=N, g=G_GAIN, tau=TAU, dt=DT,
                                 eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
                                 device=device, seed=SEED, q_mode=q_mode)
    for task in TASKS:
        net.add_task(task["name"], task["K"])

    forgetting_matrix = np.zeros((N_TASKS, N_TASKS))

    for phase_idx, task in enumerate(TASKS):
        name = task["name"]
        K = task["K"]
        net.set_active_task(name)
        data = task_data[name]["data"]

        for t in range(TRAIN_STEPS):
            if K == 1:
                target = torch.tensor([data[t]], device=device, dtype=torch.float32)
            else:
                target = torch.tensor(data[t], device=device, dtype=torch.float32)
            net.step_and_learn(target)

        for j, test_task in enumerate(TASKS):
            test_name = test_task["name"]
            test_K = test_task["K"]
            pred = test_autonomous(net, test_name, TEST_STEPS)
            forgetting_matrix[phase_idx, j] = compute_error(pred, test_truth[test_name], test_K)

    ratios = {}
    for j in range(N_TASKS):
        baseline = forgetting_matrix[j, j]
        final = forgetting_matrix[N_TASKS - 1, j]
        ratios[TASK_NAMES[j]] = final / baseline if baseline > 0 else float("nan")

    max_ratio = max(r for r in ratios.values() if not np.isnan(r))
    return forgetting_matrix, ratios, max_ratio


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
    log = logging.getLogger("exp3.8")

    set_seed(SEED)
    device = torch.device(DEVICE)

    log.info("Exp 3.8: Q ablation — is Q orthogonality the key mechanism?")
    log.info(f"N={N}, Tasks: {TASK_NAMES}")
    log.info(f"Device: {DEVICE}")

    # Generate target data
    total_steps = TRAIN_STEPS + TEST_STEPS
    t_all = np.arange(total_steps) * DT

    task_data = {
        "sine": {"data": sine_target(t_all, period=600.0, amplitude=1.5), "K": 1},
        "lorenz": {"data": generate_lorenz(duration_ms=(TRAIN_MS + TEST_MS), dt=DT, scale=0.1), "K": 3},
        "multi_sine": {"data": multi_sine_target(t_all, frequencies=[1/600, 1/300, 1/900], amplitudes=[1.0, 0.5, 0.3]), "K": 1},
        "sawtooth": {"data": generate_sawtooth(t_all, period=800.0, amplitude=1.5), "K": 1},
    }

    test_truth = {}
    for name, td in task_data.items():
        if td["K"] == 1:
            test_truth[name] = td["data"][:TEST_STEPS].reshape(-1, 1)
        else:
            test_truth[name] = td["data"][:TEST_STEPS]

    conditions = [
        ("independent", "Independent Q per task (baseline)"),
        ("shared_Q", "Shared Q (same feedback, separate w)"),
        ("shared_Q_w", "Shared Q AND shared w"),
    ]

    results = {}
    for q_mode, label in conditions:
        log.info(f"\n{'='*60}")
        log.info(f"Condition: {label} (q_mode={q_mode})")
        log.info(f"{'='*60}")

        t_start = time.time()
        fmat, ratios, max_r = run_condition(q_mode, task_data, test_truth, device, log)
        elapsed = time.time() - t_start

        results[q_mode] = {"matrix": fmat, "ratios": ratios, "max_ratio": max_r}

        log.info(f"\nForgetting matrix:")
        header = f"{'Trained through':>18}" + "".join(f"{n:>12}" for n in TASK_NAMES)
        log.info(header)
        for i, name in enumerate(TASK_NAMES):
            row = f"{name:>18}" + "".join(f"{fmat[i,j]:>12.4f}" for j in range(N_TASKS))
            log.info(row)

        log.info(f"\nForgetting ratios: {ratios}")
        log.info(f"Max forgetting ratio: {max_r:.3f}")
        log.info(f"Time: {elapsed:.1f}s")

    # ── Summary ──────────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("SUMMARY")
    log.info(f"{'='*70}")
    for q_mode, label in conditions:
        r = results[q_mode]
        log.info(f"\n{label}:")
        log.info(f"  Max ratio: {r['max_ratio']:.3f}")
        for name in TASK_NAMES:
            log.info(f"  {name:>12}: {r['ratios'][name]:.3f}")

    # ── Plot 1: Forgetting matrices side-by-side ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, (q_mode, label) in enumerate(conditions):
        ax = axes[idx]
        mat = results[q_mode]["matrix"]
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(N_TASKS))
        ax.set_xticklabels([n[:4] for n in TASK_NAMES], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(N_TASKS))
        ax.set_yticklabels([f"after {n[:4]}" for n in TASK_NAMES], fontsize=8)
        ax.set_title(label, fontsize=9)
        for i in range(N_TASKS):
            for j in range(N_TASKS):
                val = mat[i, j]
                color = "white" if val > mat.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Exp 3.8: Q ablation — Forgetting matrices", fontsize=12)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/forgetting_matrices.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Bar chart of forgetting ratios ───────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(N_TASKS)
    width = 0.25
    colors_cond = {"independent": "#2ecc71", "shared_Q": "#e74c3c", "shared_Q_w": "#3498db"}

    for idx, (q_mode, label) in enumerate(conditions):
        ratios_list = [results[q_mode]["ratios"][n] for n in TASK_NAMES]
        ax.bar(x + idx * width, ratios_list, width, label=label,
               color=colors_cond[q_mode], alpha=0.85)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Task")
    ax.set_ylabel("Forgetting ratio (final / baseline)")
    ax.set_title("Exp 3.8: Forgetting ratios by Q condition")
    ax.set_xticks(x + width)
    ax.set_xticklabels(TASK_NAMES)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/forgetting_ratios_bar.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Max ratio comparison ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [label for _, label in conditions]
    max_ratios = [results[q_mode]["max_ratio"] for q_mode, _ in conditions]
    bars = ax.bar(labels, max_ratios,
                  color=[colors_cond[q] for q, _ in conditions], alpha=0.85)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Max forgetting ratio")
    ax.set_title("Exp 3.8: Max forgetting ratio by Q condition")
    for bar, val in zip(bars, max_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/max_forgetting_comparison.png", dpi=150)
    plt.close(fig)

    np.savez(f"{RESULTS_DIR}/results.npz",
             **{f"matrix_{q}": results[q]["matrix"] for q, _ in conditions},
             **{f"max_ratio_{q}": results[q]["max_ratio"] for q, _ in conditions})

    log.info(f"\nAll results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
