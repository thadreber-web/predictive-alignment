"""Experiment 3.9: BPTT forgetting comparison.

Same 4-task sequential protocol as exp 3.4 but with truncated BPTT training.
Architecture: same sparse G + trainable M, but per-task readout w is trained
via backprop instead of Hebbian rule. No PA learning rule for M — BPTT
optimizes M directly through gradient descent.

This gives the direct comparison: PA forgetting ratio vs BPTT forgetting ratio.
"""

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

import torch
import torch.nn as nn
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
G_GAIN = 1.2
P_SPARSE = 0.1
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_MS = 15_000.0
TRAIN_STEPS = int(TRAIN_MS / DT)

TEST_MS = 10_000.0
TEST_STEPS = int(TEST_MS / DT)

# BPTT config
T_BPTT = 50
LR = 1e-3
GRAD_CLIP = 1.0

RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "exp3_9_bptt_forgetting")

TASKS = [
    {"name": "sine", "K": 1},
    {"name": "lorenz", "K": 3},
    {"name": "multi_sine", "K": 1},
    {"name": "sawtooth", "K": 1},
]
TASK_NAMES = [t["name"] for t in TASKS]
N_TASKS = len(TASKS)


class BPTT_MultiTask(nn.Module):
    """RNN with shared G+M and per-task readout, trained with BPTT."""

    def __init__(self, N, g, tau, dt, p=0.1, seed=42):
        super().__init__()
        self.N = N
        self.tau = tau
        self.dt = dt

        rng = torch.Generator()
        rng.manual_seed(seed)

        # G: sparse fixed (not trainable)
        std_g = g / math.sqrt(p * N)
        G = torch.randn(N, N, generator=rng) * std_g
        mask = torch.rand(N, N, generator=rng) < p
        G = G * mask.float()
        G.fill_diagonal_(0.0)
        self.register_buffer("G", G)

        # M: dense trainable
        std_m = 0.01 / math.sqrt(N)
        self.M = nn.Parameter(torch.randn(N, N, generator=rng) * std_m)

        # Per-task readouts
        self.readouts = nn.ModuleDict()

    def add_task(self, name, K):
        layer = nn.Linear(self.N, K, bias=False)
        nn.init.zeros_(layer.weight)
        # Ensure readout is on same device as model
        device = self.G.device
        self.readouts[name] = layer.to(device)

    def forward_step(self, x, task_name):
        r = torch.tanh(x)
        J = self.G + self.M
        current = J @ r
        dx = (-x + current) * (self.dt / self.tau)
        x_new = x + dx
        z = self.readouts[task_name](r)
        return x_new, z

    def init_state(self, device):
        return torch.randn(self.N, device=device) * 0.1


def generate_sawtooth(t, period=800.0, amplitude=1.5):
    return amplitude * (2.0 * np.mod(t / period, 1.0) - 1.0)


def test_autonomous(model, task_name, test_steps, device):
    model.eval()
    with torch.no_grad():
        x = model.init_state(device)
        zs = []
        for _ in range(test_steps):
            x, z = model.forward_step(x, task_name)
            zs.append(z.detach().cpu().numpy())
    return np.array(zs)


def compute_error(predicted, truth, K):
    pred = predicted[:len(truth)]
    truth = truth[:len(pred)]
    if K == 1:
        return np.mean(np.abs(pred.ravel() - truth.ravel()))
    else:
        return np.mean(np.linalg.norm(pred - truth, axis=1))


def train_one_phase(model, task_name, K, data, train_steps, device, log):
    """Train on one task using truncated BPTT."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_chunks = train_steps // T_BPTT
    total_loss = 0.0

    x = model.init_state(device)

    for chunk_idx in range(n_chunks):
        t_start = chunk_idx * T_BPTT
        chunk_loss = 0.0

        # Detach state at chunk boundary to truncate gradients
        x = x.detach()

        for t_offset in range(T_BPTT):
            t = t_start + t_offset
            if t >= train_steps:
                break

            x, z = model.forward_step(x, task_name)

            if K == 1:
                target = torch.tensor([data[t]], device=device, dtype=torch.float32)
            else:
                target = torch.tensor(data[t], device=device, dtype=torch.float32)

            chunk_loss = chunk_loss + torch.mean((z - target) ** 2)

        optimizer.zero_grad()
        chunk_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += chunk_loss.item()

    avg_loss = total_loss / n_chunks
    log.info(f"    {task_name}: avg chunk loss = {avg_loss:.4f}")
    return avg_loss


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
    log = logging.getLogger("exp3.9")

    set_seed(SEED)
    device = torch.device(DEVICE)

    log.info("Exp 3.9: BPTT forgetting comparison")
    log.info(f"N={N}, T_BPTT={T_BPTT}, LR={LR}, Device: {DEVICE}")

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

    # ── BPTT model ────────────────────────────────────────────────────
    model = BPTT_MultiTask(N=N, g=G_GAIN, tau=TAU, dt=DT,
                            p=P_SPARSE, seed=SEED).to(device)
    for task in TASKS:
        model.add_task(task["name"], task["K"])

    forgetting_matrix_bptt = np.zeros((N_TASKS, N_TASKS))

    log.info("\n--- BPTT Sequential Training ---")
    t_start = time.time()

    for phase_idx, task in enumerate(TASKS):
        name = task["name"]
        K = task["K"]
        log.info(f"\n  Phase {phase_idx+1}: Training on {name} (K={K})")

        train_one_phase(model, name, K, task_data[name]["data"],
                        TRAIN_STEPS, device, log)

        # Test all tasks
        for j, test_task in enumerate(TASKS):
            test_name = test_task["name"]
            test_K = test_task["K"]
            pred = test_autonomous(model, test_name, TEST_STEPS, device)
            forgetting_matrix_bptt[phase_idx, j] = compute_error(
                pred, test_truth[test_name], test_K)

    bptt_time = time.time() - t_start

    # Forgetting ratios
    bptt_ratios = {}
    for j in range(N_TASKS):
        baseline = forgetting_matrix_bptt[j, j]
        final = forgetting_matrix_bptt[N_TASKS - 1, j]
        bptt_ratios[TASK_NAMES[j]] = final / baseline if baseline > 0 else float("nan")

    bptt_max = max(r for r in bptt_ratios.values() if not np.isnan(r))

    # ── PA baseline (same protocol) ───────────────────────────────────
    log.info("\n--- PA Sequential Training (baseline) ---")

    # Reuse the PA implementation inline
    rng_pa = torch.Generator(device=device)
    rng_pa.manual_seed(SEED)

    std_g = G_GAIN / math.sqrt(P_SPARSE * N)
    G_pa = torch.randn(N, N, device=device, generator=rng_pa) * std_g
    mask_pa = torch.rand(N, N, device=device, generator=rng_pa) < P_SPARSE
    G_pa = G_pa * mask_pa.float()
    G_pa.fill_diagonal_(0.0)

    std_m = 0.01 / math.sqrt(N)
    M_pa = torch.randn(N, N, device=device, generator=rng_pa) * std_m

    pa_tasks = {}
    for task in TASKS:
        K = task["K"]
        w = torch.zeros(K, N, device=device)
        q_bound = 3.0 / math.sqrt(K)
        Q = torch.empty(N, K, device=device).uniform_(-q_bound, q_bound)
        pa_tasks[task["name"]] = {"w": w, "Q": Q, "K": K}

    forgetting_matrix_pa = np.zeros((N_TASKS, N_TASKS))
    t_start = time.time()

    for phase_idx, task in enumerate(TASKS):
        name = task["name"]
        K = task["K"]
        log.info(f"\n  Phase {phase_idx+1}: Training on {name} (K={K})")

        # Train PA
        x_pa = torch.randn(N, device=device, generator=rng_pa) * 0.1
        r_pa = torch.tanh(x_pa)
        data = task_data[name]["data"]
        pt = pa_tasks[name]

        for t in range(TRAIN_STEPS):
            r_pa = torch.tanh(x_pa)
            J = G_pa + M_pa
            current = J @ r_pa
            dx = (-x_pa + current) * (DT / TAU)
            x_pa = x_pa + dx

            z = pt["w"] @ r_pa
            if K == 1:
                target = torch.tensor([data[t]], device=device, dtype=torch.float32)
            else:
                target = torch.tensor(data[t], device=device, dtype=torch.float32)

            output_error = target - z
            pt["w"] = pt["w"] + 1e-3 * torch.outer(output_error, r_pa)
            feedback = pt["Q"] @ z
            J_hat_r = (M_pa - 1.0 * G_pa) @ r_pa
            rec_error = feedback - J_hat_r
            M_pa = M_pa + 1e-3 * torch.outer(rec_error, r_pa)

        # Test all tasks
        for j, test_task in enumerate(TASKS):
            test_name = test_task["name"]
            test_K = test_task["K"]
            pt_test = pa_tasks[test_name]

            x_test = torch.randn(N, device=device, generator=rng_pa) * 0.1
            zs = []
            for _ in range(TEST_STEPS):
                r_test = torch.tanh(x_test)
                J = G_pa + M_pa
                dx = (-x_test + J @ r_test) * (DT / TAU)
                x_test = x_test + dx
                z = pt_test["w"] @ r_test
                zs.append(z.detach().cpu().numpy())
            pred = np.array(zs)
            forgetting_matrix_pa[phase_idx, j] = compute_error(
                pred, test_truth[test_name], test_K)

    pa_time = time.time() - t_start

    pa_ratios = {}
    for j in range(N_TASKS):
        baseline = forgetting_matrix_pa[j, j]
        final = forgetting_matrix_pa[N_TASKS - 1, j]
        pa_ratios[TASK_NAMES[j]] = final / baseline if baseline > 0 else float("nan")

    pa_max = max(r for r in pa_ratios.values() if not np.isnan(r))

    # ── Summary ──────────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("SUMMARY: BPTT vs PA forgetting")
    log.info(f"{'='*70}")

    log.info(f"\nBPTT forgetting matrix:")
    header = f"{'Trained through':>18}" + "".join(f"{n:>12}" for n in TASK_NAMES)
    log.info(header)
    for i, name in enumerate(TASK_NAMES):
        row = f"{name:>18}" + "".join(f"{forgetting_matrix_bptt[i,j]:>12.4f}" for j in range(N_TASKS))
        log.info(row)

    log.info(f"\nPA forgetting matrix:")
    log.info(header)
    for i, name in enumerate(TASK_NAMES):
        row = f"{name:>18}" + "".join(f"{forgetting_matrix_pa[i,j]:>12.4f}" for j in range(N_TASKS))
        log.info(row)

    log.info(f"\nForgetting ratios:")
    log.info(f"  {'Task':>12}  {'BPTT':>8}  {'PA':>8}")
    for name in TASK_NAMES:
        log.info(f"  {name:>12}  {bptt_ratios[name]:>8.3f}  {pa_ratios[name]:>8.3f}")

    log.info(f"\n  BPTT max ratio: {bptt_max:.3f}")
    log.info(f"  PA max ratio:   {pa_max:.3f}")
    log.info(f"\n  BPTT time: {bptt_time:.1f}s")
    log.info(f"  PA time:   {pa_time:.1f}s")

    # ── Plot 1: Forgetting matrices side-by-side ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for idx, (mat, label) in enumerate([
        (forgetting_matrix_bptt, "BPTT"),
        (forgetting_matrix_pa, "PA"),
    ]):
        ax = axes[idx]
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(N_TASKS))
        ax.set_xticklabels([n[:4] for n in TASK_NAMES], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(N_TASKS))
        ax.set_yticklabels([f"after {n[:4]}" for n in TASK_NAMES], fontsize=8)
        ax.set_title(f"{label}", fontsize=10)
        for i in range(N_TASKS):
            for j in range(N_TASKS):
                val = mat[i, j]
                color = "white" if val > mat.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Exp 3.9: BPTT vs PA — Forgetting matrices", fontsize=12)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/forgetting_matrices.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Bar chart comparison ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(N_TASKS)
    width = 0.35
    bptt_vals = [bptt_ratios[n] for n in TASK_NAMES]
    pa_vals = [pa_ratios[n] for n in TASK_NAMES]

    ax.bar(x - width/2, bptt_vals, width, label="BPTT", color="#e74c3c", alpha=0.85)
    ax.bar(x + width/2, pa_vals, width, label="PA", color="#2ecc71", alpha=0.85)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Task")
    ax.set_ylabel("Forgetting ratio (final / baseline)")
    ax.set_title("Exp 3.9: BPTT vs PA forgetting ratios")
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_NAMES)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/forgetting_comparison.png", dpi=150)
    plt.close(fig)

    np.savez(f"{RESULTS_DIR}/results.npz",
             bptt_matrix=forgetting_matrix_bptt,
             pa_matrix=forgetting_matrix_pa,
             bptt_max=bptt_max,
             pa_max=pa_max)

    log.info(f"\nAll results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
