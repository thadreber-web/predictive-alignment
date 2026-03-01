"""Experiment 3.10: Task similarity — do similar tasks cause forgetting?

Train on sine waves at periods 200, 400, 600, 800, 1000, 1200ms.
Six tasks, each a slight variation. Tests whether forgetting resistance
depends on task diversity or holds for closely related tasks.

Also test a "very similar" condition: periods 580, 590, 600, 610, 620, 630ms.
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

from src.targets import sine_target
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

RESULTS_DIR = "/raid/predictive_alignment/results/exp3_10_task_similarity"


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


def run_similarity_test(periods, condition_name, device, log):
    """Run sequential training on sine waves at given periods."""
    total_steps = TRAIN_STEPS + TEST_STEPS
    t_all = np.arange(total_steps) * DT
    n_tasks = len(periods)

    task_names = [f"sine_p{p}" for p in periods]
    task_data = {name: sine_target(t_all, period=p, amplitude=1.5)
                 for name, p in zip(task_names, periods)}
    test_truth = {name: data[:TEST_STEPS].reshape(-1, 1)
                  for name, data in task_data.items()}

    net = MultiTaskPA(N=N, g=G_GAIN, tau=TAU, dt=DT,
                      eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
                      device=device, seed=SEED)
    for name in task_names:
        net.add_task(name, K=1)

    forgetting_matrix = np.zeros((n_tasks, n_tasks))

    for phase_idx, name in enumerate(task_names):
        net.set_active_task(name)
        data = task_data[name]

        for t in range(TRAIN_STEPS):
            target = torch.tensor([data[t]], device=device, dtype=torch.float32)
            net.step_and_learn(target)

        for j, test_name in enumerate(task_names):
            pred = test_autonomous(net, test_name, TEST_STEPS)
            forgetting_matrix[phase_idx, j] = compute_error(pred, test_truth[test_name])

    # Forgetting ratios (exclude last task)
    ratios = {}
    for j in range(n_tasks):
        baseline = forgetting_matrix[j, j]
        final = forgetting_matrix[n_tasks - 1, j]
        ratios[task_names[j]] = final / baseline if baseline > 0 else float("nan")

    max_ratio = max(r for r in ratios.values() if not np.isnan(r))
    mean_ratio = np.mean([r for r in ratios.values() if not np.isnan(r)])

    log.info(f"\n  Forgetting matrix for {condition_name}:")
    header = f"  {'':>12}" + "".join(f"  p={p:>5}" for p in periods)
    log.info(header)
    for i, name in enumerate(task_names):
        row = f"  {name:>12}" + "".join(f"  {forgetting_matrix[i,j]:>6.3f}" for j in range(n_tasks))
        log.info(row)
    log.info(f"  Max ratio: {max_ratio:.3f}, Mean ratio: {mean_ratio:.3f}")

    return forgetting_matrix, ratios, max_ratio, mean_ratio, task_names


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
    log = logging.getLogger("exp3.10")

    set_seed(SEED)
    device = torch.device(DEVICE)

    log.info("Exp 3.10: Task similarity — do similar tasks cause forgetting?")
    log.info(f"N={N}, Device: {DEVICE}")

    conditions = [
        ("diverse", [200, 400, 600, 800, 1000, 1200]),
        ("moderate", [400, 500, 600, 700, 800, 900]),
        ("similar", [550, 570, 590, 610, 630, 650]),
        ("very_similar", [580, 590, 600, 610, 620, 630]),
    ]

    results = {}
    for cond_name, periods in conditions:
        log.info(f"\n{'='*60}")
        log.info(f"Condition: {cond_name} — periods {periods}")
        log.info(f"{'='*60}")

        t_start = time.time()
        fmat, ratios, max_r, mean_r, tnames = run_similarity_test(
            periods, cond_name, device, log)
        elapsed = time.time() - t_start

        results[cond_name] = {
            "matrix": fmat, "ratios": ratios, "max_ratio": max_r,
            "mean_ratio": mean_r, "task_names": tnames, "periods": periods
        }
        log.info(f"  Time: {elapsed:.1f}s")

    # ── Summary ──────────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("SUMMARY")
    log.info(f"{'='*70}")
    for cond_name, periods in conditions:
        r = results[cond_name]
        log.info(f"\n{cond_name} (periods {periods}):")
        log.info(f"  Max ratio: {r['max_ratio']:.3f}, Mean ratio: {r['mean_ratio']:.3f}")
        for name, ratio in r["ratios"].items():
            log.info(f"    {name}: {ratio:.3f}")

    # ── Plot 1: Max and mean forgetting ratio by similarity ──────────
    fig, ax = plt.subplots(figsize=(10, 5))
    cond_labels = [f"{c}\n{p}" for c, p in conditions]
    max_ratios = [results[c]["max_ratio"] for c, _ in conditions]
    mean_ratios = [results[c]["mean_ratio"] for c, _ in conditions]

    x = np.arange(len(conditions))
    width = 0.35
    ax.bar(x - width/2, max_ratios, width, label="Max ratio", color="#e74c3c", alpha=0.85)
    ax.bar(x + width/2, mean_ratios, width, label="Mean ratio", color="#3498db", alpha=0.85)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Similarity condition")
    ax.set_ylabel("Forgetting ratio")
    ax.set_title("Exp 3.10: Forgetting vs task similarity")
    ax.set_xticks(x)
    ax.set_xticklabels([c for c, _ in conditions])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/similarity_comparison.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Forgetting matrices side-by-side ─────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for idx, (cond_name, periods) in enumerate(conditions):
        ax = axes[idx]
        mat = results[cond_name]["matrix"]
        n = len(periods)
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(p) for p in periods], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"after {p}" for p in periods], fontsize=7)
        ax.set_title(f"{cond_name}", fontsize=10)
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                color = "white" if val > mat.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=6)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Exp 3.10: Forgetting matrices by task similarity", fontsize=12)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/forgetting_matrices.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Per-task forgetting ratios ───────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for idx, (cond_name, periods) in enumerate(conditions):
        ax = axes[idx // 2][idx % 2]
        ratios = results[cond_name]["ratios"]
        names = list(ratios.keys())
        vals = [ratios[n] for n in names]
        colors = ["#e74c3c" if v > 1.0 else "#2ecc71" for v in vals]
        ax.bar(range(len(names)), vals, color=colors, alpha=0.85)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([str(p) for p in periods], rotation=45, fontsize=8)
        ax.set_ylabel("Forgetting ratio")
        ax.set_title(f"{cond_name}: max={results[cond_name]['max_ratio']:.3f}")
        ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle("Exp 3.10: Per-task forgetting ratios", fontsize=12)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/per_task_ratios.png", dpi=150)
    plt.close(fig)

    np.savez(f"{RESULTS_DIR}/results.npz",
             **{f"matrix_{c}": results[c]["matrix"] for c, _ in conditions},
             **{f"max_ratio_{c}": results[c]["max_ratio"] for c, _ in conditions},
             **{f"mean_ratio_{c}": results[c]["mean_ratio"] for c, _ in conditions})

    log.info(f"\nAll results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
