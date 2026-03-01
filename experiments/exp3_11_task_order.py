"""Experiment 3.11: Task order permutation test.

Run same 4 tasks in multiple orders. If forgetting is truly zero regardless
of order, that strengthens the structural argument.
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
import itertools

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

RESULTS_DIR = "/raid/predictive_alignment/results/exp3_11_task_order"

ALL_TASKS = {
    "sine": {"K": 1},
    "lorenz": {"K": 3},
    "multi_sine": {"K": 1},
    "sawtooth": {"K": 1},
}
TASK_NAMES = list(ALL_TASKS.keys())


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
    log = logging.getLogger("exp3.11")

    set_seed(SEED)
    device = torch.device(DEVICE)

    log.info("Exp 3.11: Task order permutation test")
    log.info(f"N={N}, Device: {DEVICE}")

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

    # All 24 permutations of 4 tasks
    all_orders = list(itertools.permutations(TASK_NAMES))
    log.info(f"Testing all {len(all_orders)} permutations of {TASK_NAMES}")

    # Results: per-order final errors and forgetting ratios
    order_results = []

    for order_idx, order in enumerate(all_orders):
        order_label = "→".join(order)

        net = MultiTaskPA(N=N, g=G_GAIN, tau=TAU, dt=DT,
                          eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
                          device=device, seed=SEED)
        for name in TASK_NAMES:
            net.add_task(name, ALL_TASKS[name]["K"])

        n_tasks = len(order)
        forgetting_matrix = np.zeros((n_tasks, n_tasks))

        for phase_idx, name in enumerate(order):
            K = task_data[name]["K"]
            net.set_active_task(name)
            data = task_data[name]["data"]

            for t in range(TRAIN_STEPS):
                if K == 1:
                    target = torch.tensor([data[t]], device=device, dtype=torch.float32)
                else:
                    target = torch.tensor(data[t], device=device, dtype=torch.float32)
                net.step_and_learn(target)

            # Test all tasks in order
            for j, test_name in enumerate(order):
                test_K = task_data[test_name]["K"]
                pred = test_autonomous(net, test_name, TEST_STEPS)
                forgetting_matrix[phase_idx, j] = compute_error(
                    pred, test_truth[test_name], test_K)

        # Forgetting ratios
        ratios = {}
        for j in range(n_tasks):
            baseline = forgetting_matrix[j, j]
            final = forgetting_matrix[n_tasks - 1, j]
            task_name = order[j]
            ratios[task_name] = final / baseline if baseline > 0 else float("nan")

        max_ratio = max(r for r in ratios.values() if not np.isnan(r))
        mean_ratio = np.mean([r for r in ratios.values() if not np.isnan(r)])

        # Final errors (after all training)
        final_errors = {}
        for j, name in enumerate(order):
            final_errors[name] = forgetting_matrix[n_tasks - 1, j]

        order_results.append({
            "order": order,
            "label": order_label,
            "matrix": forgetting_matrix,
            "ratios": ratios,
            "max_ratio": max_ratio,
            "mean_ratio": mean_ratio,
            "final_errors": final_errors,
        })

        if order_idx % 6 == 0:
            log.info(f"  [{order_idx+1:>2}/{len(all_orders)}] {order_label}: max={max_ratio:.3f}, mean={mean_ratio:.3f}")

    # ── Summary ──────────────────────────────────────────────────────
    max_ratios = [r["max_ratio"] for r in order_results]
    mean_ratios = [r["mean_ratio"] for r in order_results]

    log.info(f"\n{'='*70}")
    log.info("SUMMARY across all 24 permutations")
    log.info(f"{'='*70}")
    log.info(f"Max forgetting ratio: min={min(max_ratios):.3f}, max={max(max_ratios):.3f}, "
             f"mean={np.mean(max_ratios):.3f}, std={np.std(max_ratios):.3f}")
    log.info(f"Mean forgetting ratio: min={min(mean_ratios):.3f}, max={max(mean_ratios):.3f}, "
             f"mean={np.mean(mean_ratios):.3f}, std={np.std(mean_ratios):.3f}")

    # Best and worst orders
    best_idx = np.argmin(max_ratios)
    worst_idx = np.argmax(max_ratios)
    log.info(f"\nBest order: {order_results[best_idx]['label']} (max ratio={max_ratios[best_idx]:.3f})")
    log.info(f"Worst order: {order_results[worst_idx]['label']} (max ratio={max_ratios[worst_idx]:.3f})")

    # Per-task final errors across all orders
    log.info(f"\nPer-task final error stats across all orders:")
    for name in TASK_NAMES:
        errs = [r["final_errors"][name] for r in order_results]
        log.info(f"  {name:>12}: mean={np.mean(errs):.4f}, std={np.std(errs):.4f}, "
                 f"min={min(errs):.4f}, max={max(errs):.4f}")

    # ── Plot 1: Distribution of max forgetting ratios ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(max_ratios, bins=15, color="#e74c3c", alpha=0.7, edgecolor="black")
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Max forgetting ratio")
    ax.set_ylabel("Count (of 24 permutations)")
    ax.set_title("Distribution of max forgetting ratio across all orders")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(mean_ratios, bins=15, color="#3498db", alpha=0.7, edgecolor="black")
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean forgetting ratio")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of mean forgetting ratio across all orders")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Exp 3.11: Task order effect on forgetting", fontsize=12)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/order_distributions.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Max ratio per order (sorted) ─────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    sorted_idx = np.argsort(max_ratios)
    sorted_labels = [order_results[i]["label"] for i in sorted_idx]
    sorted_max = [max_ratios[i] for i in sorted_idx]
    colors = ["#e74c3c" if v > 1.0 else "#2ecc71" for v in sorted_max]
    ax.barh(range(len(sorted_labels)), sorted_max, color=colors, alpha=0.8)
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=6)
    ax.set_xlabel("Max forgetting ratio")
    ax.set_title("Exp 3.11: Max forgetting ratio for all 24 task orders (sorted)")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/all_orders_sorted.png", dpi=150)
    plt.close(fig)

    # Save
    np.savez(f"{RESULTS_DIR}/results.npz",
             max_ratios=np.array(max_ratios),
             mean_ratios=np.array(mean_ratios),
             orders=np.array([r["label"] for r in order_results]))

    log.info(f"\nAll results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
