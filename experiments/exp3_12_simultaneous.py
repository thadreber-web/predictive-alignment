"""Experiment 3.12: Simultaneous vs sequential training.

Train all 4 tasks simultaneously (interleaved, switching target every step)
instead of sequentially. Compare final performance on each task.

If simultaneous is better: sequential training is paying a cost even though
forgetting is zero. If equivalent: architecture truly doesn't care about order.
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

RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "exp3_12_simultaneous")

TASKS = [
    {"name": "sine", "K": 1},
    {"name": "lorenz", "K": 3},
    {"name": "multi_sine", "K": 1},
    {"name": "sawtooth", "K": 1},
]
TASK_NAMES = [t["name"] for t in TASKS]
N_TASKS = len(TASKS)


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
    log = logging.getLogger("exp3.12")

    set_seed(SEED)
    device = torch.device(DEVICE)

    log.info("Exp 3.12: Simultaneous vs sequential training")
    log.info(f"N={N}, Device: {DEVICE}")

    # Generate target data — need enough for both conditions
    # Sequential: 15k steps per task = 60k total
    # Simultaneous: 60k total steps, cycling through tasks
    total_data_steps = TRAIN_STEPS * N_TASKS + TEST_STEPS
    t_all = np.arange(total_data_steps) * DT

    task_data = {
        "sine": {"data": sine_target(t_all, period=600.0, amplitude=1.5), "K": 1},
        "lorenz": {"data": generate_lorenz(duration_ms=total_data_steps * DT, dt=DT, scale=0.1), "K": 3},
        "multi_sine": {"data": multi_sine_target(t_all, frequencies=[1/600, 1/300, 1/900], amplitudes=[1.0, 0.5, 0.3]), "K": 1},
        "sawtooth": {"data": generate_sawtooth(t_all, period=800.0, amplitude=1.5), "K": 1},
    }

    test_truth = {}
    for name, td in task_data.items():
        if td["K"] == 1:
            test_truth[name] = td["data"][:TEST_STEPS].reshape(-1, 1)
        else:
            test_truth[name] = td["data"][:TEST_STEPS]

    total_train_steps = TRAIN_STEPS * N_TASKS  # same total training time

    # ── Condition 1: Sequential (baseline from exp 3.4) ───────────────
    log.info(f"\n{'='*60}")
    log.info("Condition 1: Sequential training")
    log.info(f"{'='*60}")

    net_seq = MultiTaskPA(N=N, g=G_GAIN, tau=TAU, dt=DT,
                           eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
                           device=device, seed=SEED)
    for task in TASKS:
        net_seq.add_task(task["name"], task["K"])

    t_start = time.time()
    for phase_idx, task in enumerate(TASKS):
        name = task["name"]
        K = task["K"]
        net_seq.set_active_task(name)
        data = task_data[name]["data"]

        for t in range(TRAIN_STEPS):
            if K == 1:
                target = torch.tensor([data[t]], device=device, dtype=torch.float32)
            else:
                target = torch.tensor(data[t], device=device, dtype=torch.float32)
            net_seq.step_and_learn(target)

        log.info(f"  Phase {phase_idx+1}: {name} done")

    seq_time = time.time() - t_start

    # Test sequential
    seq_errors = {}
    for task in TASKS:
        name = task["name"]
        K = task["K"]
        pred = test_autonomous(net_seq, name, TEST_STEPS)
        seq_errors[name] = compute_error(pred, test_truth[name], K)
    log.info(f"  Sequential errors: {seq_errors}")

    # ── Condition 2: Simultaneous (round-robin, switch every step) ────
    log.info(f"\n{'='*60}")
    log.info("Condition 2: Simultaneous (round-robin per step)")
    log.info(f"{'='*60}")

    net_sim = MultiTaskPA(N=N, g=G_GAIN, tau=TAU, dt=DT,
                           eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
                           device=device, seed=SEED)
    for task in TASKS:
        net_sim.add_task(task["name"], task["K"])

    # Track per-task time index
    task_t = {name: 0 for name in TASK_NAMES}

    t_start = time.time()
    for global_t in range(total_train_steps):
        task_idx = global_t % N_TASKS
        task = TASKS[task_idx]
        name = task["name"]
        K = task["K"]
        t = task_t[name]

        net_sim.set_active_task(name)
        data = task_data[name]["data"]

        if K == 1:
            target = torch.tensor([data[t]], device=device, dtype=torch.float32)
        else:
            target = torch.tensor(data[t], device=device, dtype=torch.float32)
        net_sim.step_and_learn(target)
        task_t[name] += 1

    sim_time = time.time() - t_start

    # Test simultaneous
    sim_errors = {}
    for task in TASKS:
        name = task["name"]
        K = task["K"]
        pred = test_autonomous(net_sim, name, TEST_STEPS)
        sim_errors[name] = compute_error(pred, test_truth[name], K)
    log.info(f"  Simultaneous errors: {sim_errors}")

    # ── Condition 3: Simultaneous (block-interleaved, 500 steps each) ─
    log.info(f"\n{'='*60}")
    log.info("Condition 3: Simultaneous (block-interleaved, 500 steps per block)")
    log.info(f"{'='*60}")

    net_block = MultiTaskPA(N=N, g=G_GAIN, tau=TAU, dt=DT,
                             eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
                             device=device, seed=SEED)
    for task in TASKS:
        net_block.add_task(task["name"], task["K"])

    BLOCK_SIZE = 500
    task_t_block = {name: 0 for name in TASK_NAMES}
    n_blocks = total_train_steps // BLOCK_SIZE

    t_start = time.time()
    for block_idx in range(n_blocks):
        task_idx = block_idx % N_TASKS
        task = TASKS[task_idx]
        name = task["name"]
        K = task["K"]
        net_block.set_active_task(name)
        data = task_data[name]["data"]

        for step in range(BLOCK_SIZE):
            t = task_t_block[name]
            if K == 1:
                target = torch.tensor([data[t]], device=device, dtype=torch.float32)
            else:
                target = torch.tensor(data[t], device=device, dtype=torch.float32)
            net_block.step_and_learn(target)
            task_t_block[name] += 1

    block_time = time.time() - t_start

    # Test block-interleaved
    block_errors = {}
    for task in TASKS:
        name = task["name"]
        K = task["K"]
        pred = test_autonomous(net_block, name, TEST_STEPS)
        block_errors[name] = compute_error(pred, test_truth[name], K)
    log.info(f"  Block-interleaved errors: {block_errors}")

    # ── Summary ──────────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("SUMMARY")
    log.info(f"{'='*70}")

    log.info(f"\n  {'Task':>12}  {'Sequential':>12}  {'Round-robin':>12}  {'Block-500':>12}")
    for name in TASK_NAMES:
        log.info(f"  {name:>12}  {seq_errors[name]:>12.4f}  {sim_errors[name]:>12.4f}  {block_errors[name]:>12.4f}")

    seq_mean = np.mean(list(seq_errors.values()))
    sim_mean = np.mean(list(sim_errors.values()))
    block_mean = np.mean(list(block_errors.values()))
    log.info(f"  {'MEAN':>12}  {seq_mean:>12.4f}  {sim_mean:>12.4f}  {block_mean:>12.4f}")

    log.info(f"\n  Sequential time: {seq_time:.1f}s")
    log.info(f"  Round-robin time: {sim_time:.1f}s")
    log.info(f"  Block-500 time: {block_time:.1f}s")

    # ── Plot: Bar chart comparison ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(N_TASKS)
    width = 0.25
    ax.bar(x - width, [seq_errors[n] for n in TASK_NAMES], width,
           label="Sequential", color="#2ecc71", alpha=0.85)
    ax.bar(x, [sim_errors[n] for n in TASK_NAMES], width,
           label="Round-robin", color="#3498db", alpha=0.85)
    ax.bar(x + width, [block_errors[n] for n in TASK_NAMES], width,
           label="Block-500", color="#e74c3c", alpha=0.85)
    ax.set_xlabel("Task")
    ax.set_ylabel("Final error (MAE)")
    ax.set_title("Exp 3.12: Sequential vs simultaneous training — final errors")
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_NAMES)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/comparison.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Normalized comparison ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    # Normalize by sequential (baseline)
    sim_norm = {n: sim_errors[n] / seq_errors[n] for n in TASK_NAMES}
    block_norm = {n: block_errors[n] / seq_errors[n] for n in TASK_NAMES}

    ax.bar(x - width/2, [sim_norm[n] for n in TASK_NAMES], width,
           label="Round-robin / Sequential", color="#3498db", alpha=0.85)
    ax.bar(x + width/2, [block_norm[n] for n in TASK_NAMES], width,
           label="Block-500 / Sequential", color="#e74c3c", alpha=0.85)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Sequential baseline")
    ax.set_xlabel("Task")
    ax.set_ylabel("Error ratio (vs sequential)")
    ax.set_title("Exp 3.12: Simultaneous performance relative to sequential")
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_NAMES)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/normalized_comparison.png", dpi=150)
    plt.close(fig)

    np.savez(f"{RESULTS_DIR}/results.npz",
             seq_errors=np.array([seq_errors[n] for n in TASK_NAMES]),
             sim_errors=np.array([sim_errors[n] for n in TASK_NAMES]),
             block_errors=np.array([block_errors[n] for n in TASK_NAMES]))

    log.info(f"\nAll results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
