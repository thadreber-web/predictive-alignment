"""Experiment 3.7: G ablation — does the fixed scaffold prevent forgetting?

Three conditions on the same 4-task protocol (sine→Lorenz→multi-sine→sawtooth):
1. Normal PA: G at g=1.2, M starts small — the baseline
2. G=0 ablation: G=0, M starts at g=1.2 strength (M replaces G)
3. G=0, M=0: G=0, M starts small (no initial connectivity)

If forgetting appears in condition 2 but not 1: G is essential.
If both work: Q separation alone is sufficient.
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

RESULTS_DIR = "/raid/predictive_alignment/results/exp3_7_G_ablation"

TASKS = [
    {"name": "sine", "K": 1},
    {"name": "lorenz", "K": 3},
    {"name": "multi_sine", "K": 1},
    {"name": "sawtooth", "K": 1},
]
TASK_NAMES = [t["name"] for t in TASKS]
N_TASKS = len(TASKS)


class MultiTaskPA:
    """PA network with configurable G initialization."""

    def __init__(self, N, g, tau, dt, eta_w, eta_m, alpha, device, seed,
                 g_mode="normal"):
        """
        g_mode:
          "normal" — standard: G sparse at g=1.2, M starts small
          "no_G"  — G=0, M starts at g=1.2 strength (sparse like G would be)
          "no_G_no_M" — G=0, M starts small (no initial connectivity)
        """
        self.N = N
        self.g = g
        self.tau = tau
        self.dt = dt
        self.eta_w = eta_w
        self.eta_m = eta_m
        self.alpha = alpha
        self.device = torch.device(device)
        self.p = 0.1
        self.g_mode = g_mode

        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        self.rng = rng

        std_g = g / math.sqrt(self.p * N)

        if g_mode == "normal":
            # Standard: G has connectivity, M starts small
            G = torch.randn(N, N, device=self.device, generator=rng) * std_g
            mask = torch.rand(N, N, device=self.device, generator=rng) < self.p
            self.G = G * mask.float()
            self.G.fill_diagonal_(0.0)

            std_m = 0.01 / math.sqrt(N)
            self.M = torch.randn(N, N, device=self.device, generator=rng) * std_m

        elif g_mode == "no_G":
            # G=0, M starts at G's strength
            # Generate G same way (for RNG consistency), but zero it
            G_dummy = torch.randn(N, N, device=self.device, generator=rng) * std_g
            mask = torch.rand(N, N, device=self.device, generator=rng) < self.p
            self.G = torch.zeros(N, N, device=self.device)

            # M starts with sparse connectivity at g=1.2 strength
            M = torch.randn(N, N, device=self.device, generator=rng) * std_g
            m_mask = torch.rand(N, N, device=self.device, generator=rng) < self.p
            self.M = M * m_mask.float()
            self.M.fill_diagonal_(0.0)

        elif g_mode == "no_G_no_M":
            # Both G=0 and M starts small
            G_dummy = torch.randn(N, N, device=self.device, generator=rng) * std_g
            mask = torch.rand(N, N, device=self.device, generator=rng) < self.p
            self.G = torch.zeros(N, N, device=self.device)

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


def run_condition(g_mode, task_data, test_truth, device, log):
    """Run 4-task sequential training under a given G condition."""
    net = MultiTaskPA(N=N, g=G_GAIN, tau=TAU, dt=DT,
                      eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
                      device=device, seed=SEED, g_mode=g_mode)
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

        # Test all tasks after this phase
        for j, test_task in enumerate(TASKS):
            test_name = test_task["name"]
            test_K = test_task["K"]
            pred = test_autonomous(net, test_name, TEST_STEPS)
            forgetting_matrix[phase_idx, j] = compute_error(pred, test_truth[test_name], test_K)

    # Compute forgetting ratios
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
    log = logging.getLogger("exp3.7")

    set_seed(SEED)
    device = torch.device(DEVICE)

    log.info("Exp 3.7: G ablation — does the fixed scaffold prevent forgetting?")
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

    # ── Run all three conditions ─────────────────────────────────────
    conditions = [
        ("normal", "Standard PA (G + M)"),
        ("no_G", "G=0, M starts at g=1.2"),
        ("no_G_no_M", "G=0, M starts small"),
    ]

    results = {}
    for g_mode, label in conditions:
        log.info(f"\n{'='*60}")
        log.info(f"Condition: {label} (g_mode={g_mode})")
        log.info(f"{'='*60}")

        t_start = time.time()
        fmat, ratios, max_r = run_condition(g_mode, task_data, test_truth, device, log)
        elapsed = time.time() - t_start

        results[g_mode] = {"matrix": fmat, "ratios": ratios, "max_ratio": max_r}

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
    for g_mode, label in conditions:
        r = results[g_mode]
        log.info(f"\n{label}:")
        log.info(f"  Max ratio: {r['max_ratio']:.3f}")
        for name in TASK_NAMES:
            log.info(f"  {name:>12}: {r['ratios'][name]:.3f}")

    # ── Plot 1: Forgetting matrices side-by-side ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, (g_mode, label) in enumerate(conditions):
        ax = axes[idx]
        mat = results[g_mode]["matrix"]
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
    fig.suptitle("Exp 3.7: G ablation — Forgetting matrices", fontsize=12)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/forgetting_matrices.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Bar chart of forgetting ratios ───────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(N_TASKS)
    width = 0.25
    colors_cond = {"normal": "#2ecc71", "no_G": "#e74c3c", "no_G_no_M": "#3498db"}

    for idx, (g_mode, label) in enumerate(conditions):
        ratios = [results[g_mode]["ratios"][n] for n in TASK_NAMES]
        ax.bar(x + idx * width, ratios, width, label=label,
               color=colors_cond[g_mode], alpha=0.85)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Task")
    ax.set_ylabel("Forgetting ratio (final / baseline)")
    ax.set_title("Exp 3.7: Forgetting ratios by condition")
    ax.set_xticks(x + width)
    ax.set_xticklabels(TASK_NAMES)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/forgetting_ratios_bar.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Max forgetting ratio comparison ──────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [label for _, label in conditions]
    max_ratios = [results[g_mode]["max_ratio"] for g_mode, _ in conditions]
    bars = ax.bar(labels, max_ratios,
                  color=[colors_cond[g] for g, _ in conditions], alpha=0.85)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No forgetting")
    ax.set_ylabel("Max forgetting ratio")
    ax.set_title("Exp 3.7: Max forgetting ratio by condition")
    for bar, val in zip(bars, max_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/max_forgetting_comparison.png", dpi=150)
    plt.close(fig)

    # Save
    np.savez(f"{RESULTS_DIR}/results.npz",
             **{f"matrix_{g}": results[g]["matrix"] for g, _ in conditions},
             **{f"max_ratio_{g}": results[g]["max_ratio"] for g, _ in conditions})

    log.info(f"\nAll results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
