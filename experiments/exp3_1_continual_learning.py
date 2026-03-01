"""Experiment 3.1 & 3.2: Continual learning — sequential two-system test.

Phase 1: Train on sine wave (K=1, w₁, Q₁) for 30s.
Phase 2: Train on Lorenz (K=3, w₂, Q₂) for 15,000s, keeping M continuous.
         Periodically test sine recall using w₁ (forgetting curve = exp 3.2).

Controls: separate networks trained on sine-only and Lorenz-only.

Key question: does training on Lorenz destroy what M learned for sine?
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

from src.targets import sine_target, generate_lorenz
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

# Phase 1: sine
SINE_PERIOD = 600.0
SINE_AMP = 1.5
PHASE1_MS = 30_000.0
PHASE1_STEPS = int(PHASE1_MS / DT)

# Phase 2: Lorenz
PHASE2_MS = 15_000.0
PHASE2_STEPS = int(PHASE2_MS / DT)

# Forgetting curve: test sine every N steps during Lorenz training
FORGET_TEST_EVERY = 500  # steps
FORGET_TEST_DURATION = 5_000.0  # 5s sine test
FORGET_TEST_STEPS = int(FORGET_TEST_DURATION / DT)

# Final test
TEST_MS = 10_000.0
TEST_STEPS = int(TEST_MS / DT)

RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "exp3_1_continual_learning")


class MultiTaskPA:
    """PA network that supports task-specific readouts sharing G and M.

    Each task has its own (w, Q) pair. G and M are shared.
    During training, only the active task's w and Q are used.
    """

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

        # Shared: G, M
        std_g = g / math.sqrt(self.p * N)
        G = torch.randn(N, N, device=self.device, generator=rng) * std_g
        mask = torch.rand(N, N, device=self.device, generator=rng) < self.p
        self.G = G * mask.float()
        self.G.fill_diagonal_(0.0)

        std_m = 0.01 / math.sqrt(N)
        self.M = torch.randn(N, N, device=self.device, generator=rng) * std_m

        # Task-specific readouts: dict of {task_name: (w, Q, K)}
        self.tasks = {}
        self.active_task = None

        # State
        self.x = torch.randn(N, device=self.device, generator=rng) * 0.1
        self.r = torch.tanh(self.x)

    def add_task(self, name, K):
        """Register a new task with K readout units."""
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
        """Forward step using active task's readout."""
        task = self.tasks[self.active_task]
        self.r = torch.tanh(self.x)
        J = self.G + self.M
        current = J @ self.r
        dx = (-self.x + current) * (self.dt / self.tau)
        self.x = self.x + dx
        z = task["w"] @ self.r
        return z

    def step_and_learn(self, target):
        """Forward step + PA learning using active task's w and Q."""
        task = self.tasks[self.active_task]
        z = self.step()

        # Readout update (delta rule)
        output_error = target - z
        task["w"] = task["w"] + self.eta_w * torch.outer(output_error, self.r)

        # Recurrent update (predictive alignment)
        feedback = task["Q"] @ z
        J_hat_r = (self.M - self.alpha * self.G) @ self.r
        rec_error = feedback - J_hat_r
        self.M = self.M + self.eta_m * torch.outer(rec_error, self.r)

        return z

    def snapshot_M(self):
        return self.M.clone()


def test_autonomous(net, task_name, test_steps):
    """Run autonomous generation (no learning) for a task."""
    net.set_active_task(task_name)
    net.reset_state()
    zs = []
    for _ in range(test_steps):
        z = net.step()
        zs.append(z.detach().cpu().numpy())
    return np.array(zs)


def train_control(K, target_data, train_steps, device, seed):
    """Train a fresh PA network on a single task (control baseline)."""
    net = MultiTaskPA(N=N, g=G_GAIN, tau=TAU, dt=DT,
                      eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
                      device=device, seed=seed)
    net.add_task("ctrl", K)
    net.set_active_task("ctrl")

    n_traj = len(target_data)
    for t in range(train_steps):
        target = torch.tensor(target_data[t % n_traj], device=device, dtype=torch.float32)
        net.step_and_learn(target)

    return net


def compute_error(predicted, truth, K):
    """Mean L2 error between predicted and truth arrays."""
    pred = predicted[:len(truth)]
    truth = truth[:len(pred)]
    if K == 1:
        return np.mean(np.abs(pred.ravel() - truth.ravel()))
    else:
        return np.mean(np.linalg.norm(pred - truth, axis=1))


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
    log = logging.getLogger("exp3.1")

    set_seed(SEED)
    device = torch.device(DEVICE)

    log.info("Exp 3.1/3.2: Continual learning — sine → Lorenz")
    log.info(f"N={N}, α={ALPHA}, η_w={ETA_W}, η_m={ETA_M}, g={G_GAIN}")
    log.info(f"Phase 1: sine {PHASE1_MS/1000:.0f}s, Phase 2: Lorenz {PHASE2_MS/1000:.0f}s")
    log.info(f"Device: {DEVICE}")

    # ── Generate target data ─────────────────────────────────────────
    t_sine = np.arange(PHASE1_STEPS + TEST_STEPS) * DT
    sine_data = sine_target(t_sine, period=SINE_PERIOD, amplitude=SINE_AMP)

    lorenz_data = generate_lorenz(
        duration_ms=PHASE2_MS + TEST_MS + FORGET_TEST_DURATION,
        dt=DT, scale=0.1,
    )

    # ── Create multi-task network ────────────────────────────────────
    net = MultiTaskPA(N=N, g=G_GAIN, tau=TAU, dt=DT,
                      eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
                      device=device, seed=SEED)
    net.add_task("sine", K=1)
    net.add_task("lorenz", K=3)

    # ── Phase 1: Train on sine ───────────────────────────────────────
    log.info("\n--- Phase 1: Training on sine wave ---")
    net.set_active_task("sine")
    t_start = time.time()

    phase1_errors = []
    for t in range(PHASE1_STEPS):
        target = torch.tensor([sine_data[t]], device=device, dtype=torch.float32)
        z = net.step_and_learn(target)
        if t % 100 == 0:
            phase1_errors.append(abs(z[0].item() - sine_data[t]))

    M_after_phase1 = net.snapshot_M()
    elapsed1 = time.time() - t_start
    log.info(f"Phase 1 done in {elapsed1:.1f}s")

    # Test sine after Phase 1
    sine_pred_after_p1 = test_autonomous(net, "sine", TEST_STEPS)
    sine_truth = sine_data[:TEST_STEPS].reshape(-1, 1)
    sine_err_p1 = compute_error(sine_pred_after_p1, sine_truth, K=1)
    log.info(f"Sine error after Phase 1: {sine_err_p1:.4f}")

    # ── Phase 2: Train on Lorenz (forgetting curve = exp 3.2) ────────
    log.info("\n--- Phase 2: Training on Lorenz (measuring sine forgetting) ---")
    net.set_active_task("lorenz")
    t_start = time.time()

    phase2_errors = []
    forgetting_curve = []  # (step, sine_error)

    # Initial sine test before any Lorenz training
    sine_test_pred = test_autonomous(net, "sine", FORGET_TEST_STEPS)
    sine_test_truth = sine_data[:FORGET_TEST_STEPS].reshape(-1, 1)
    sine_err_init = compute_error(sine_test_pred, sine_test_truth, K=1)
    forgetting_curve.append((0, sine_err_init))
    log.info(f"  Step 0/{PHASE2_STEPS}: sine_err={sine_err_init:.4f}")

    # Switch back to Lorenz for training
    net.set_active_task("lorenz")

    for t in range(PHASE2_STEPS):
        target = torch.tensor(lorenz_data[t], device=device, dtype=torch.float32)
        z = net.step_and_learn(target)
        if t % 100 == 0:
            phase2_errors.append(np.linalg.norm(z.detach().cpu().numpy() - lorenz_data[t]))

        # Periodic sine test (forgetting curve)
        if (t + 1) % FORGET_TEST_EVERY == 0:
            # Save state, test sine, restore state
            x_save = net.x.clone()
            r_save = net.r.clone()

            sine_test_pred = test_autonomous(net, "sine", FORGET_TEST_STEPS)
            sine_err = compute_error(sine_test_pred, sine_test_truth, K=1)
            forgetting_curve.append((t + 1, sine_err))

            # Restore state and active task for continued Lorenz training
            net.x = x_save
            net.r = r_save
            net.set_active_task("lorenz")

            if (t + 1) % 2500 == 0:
                log.info(f"  Step {t+1}/{PHASE2_STEPS}: sine_err={sine_err:.4f}")

    M_after_phase2 = net.snapshot_M()
    elapsed2 = time.time() - t_start
    log.info(f"Phase 2 done in {elapsed2:.1f}s")

    # ── Final tests ──────────────────────────────────────────────────
    log.info("\n--- Final tests ---")

    # Sine after Phase 2
    sine_pred_after_p2 = test_autonomous(net, "sine", TEST_STEPS)
    sine_err_p2 = compute_error(sine_pred_after_p2, sine_truth, K=1)
    log.info(f"Sine error after Phase 2: {sine_err_p2:.4f}")

    # Lorenz after Phase 2
    lorenz_pred_after_p2 = test_autonomous(net, "lorenz", TEST_STEPS)
    lorenz_truth = lorenz_data[:TEST_STEPS]
    lorenz_err_p2 = compute_error(lorenz_pred_after_p2, lorenz_truth, K=3)
    log.info(f"Lorenz error after Phase 2: {lorenz_err_p2:.4f}")

    # M change analysis
    M_delta = M_after_phase2 - M_after_phase1
    M_frob = torch.norm(M_delta, p="fro").item()
    M_frob_p1 = torch.norm(M_after_phase1, p="fro").item()
    M_frob_p2 = torch.norm(M_after_phase2, p="fro").item()
    log.info(f"\n||M_phase1|| = {M_frob_p1:.2f}")
    log.info(f"||M_phase2|| = {M_frob_p2:.2f}")
    log.info(f"||ΔM|| = {M_frob:.2f}")
    log.info(f"||ΔM|| / ||M_phase1|| = {M_frob / M_frob_p1:.4f}")

    # ── Controls: train fresh networks on each task ──────────────────
    log.info("\n--- Control baselines (fresh networks) ---")

    # Sine-only control
    ctrl_sine = train_control(K=1, target_data=sine_data[:PHASE1_STEPS].reshape(-1, 1),
                              train_steps=PHASE1_STEPS, device=device, seed=SEED + 100)
    ctrl_sine_pred = test_autonomous(ctrl_sine, "ctrl", TEST_STEPS)
    ctrl_sine_err = compute_error(ctrl_sine_pred, sine_truth, K=1)
    log.info(f"Control sine error (fresh net, {PHASE1_MS/1000:.0f}s training): {ctrl_sine_err:.4f}")

    # Lorenz-only control
    ctrl_lorenz = train_control(K=3, target_data=lorenz_data[:PHASE2_STEPS],
                                train_steps=PHASE2_STEPS, device=device, seed=SEED + 200)
    ctrl_lorenz_pred = test_autonomous(ctrl_lorenz, "ctrl", TEST_STEPS)
    ctrl_lorenz_err = compute_error(ctrl_lorenz_pred, lorenz_truth, K=3)
    log.info(f"Control Lorenz error (fresh net, {PHASE2_MS/1000:.0f}s training): {ctrl_lorenz_err:.4f}")

    # ── Plots ────────────────────────────────────────────────────────

    # 1. Forgetting curve (exp 3.2)
    fc_steps, fc_errs = zip(*forgetting_curve)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(np.array(fc_steps) * DT / 1000, fc_errs, "r-", lw=2, label="Sine error during Lorenz training")
    ax.axhline(y=sine_err_p1, color="g", linestyle="--", alpha=0.7, label=f"Sine after Phase 1: {sine_err_p1:.3f}")
    ax.axhline(y=ctrl_sine_err, color="b", linestyle=":", alpha=0.7, label=f"Control (sine-only): {ctrl_sine_err:.3f}")
    ax.set_xlabel("Lorenz training time (s)")
    ax.set_ylabel("Sine error")
    ax.set_title("Exp 3.2: Forgetting Curve — Sine recall during Lorenz training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/forgetting_curve.png", dpi=150)
    plt.close(fig)

    # 2. Phase 1 training curve
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(phase1_errors)) * 100 * DT / 1000, phase1_errors, "b-", alpha=0.5, lw=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sine error")
    ax.set_title("Phase 1: Sine training error")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/phase1_training.png", dpi=150)
    plt.close(fig)

    # 3. Phase 2 training curve
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(phase2_errors)) * 100 * DT / 1000, phase2_errors, "b-", alpha=0.5, lw=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lorenz error")
    ax.set_title("Phase 2: Lorenz training error")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/phase2_training.png", dpi=150)
    plt.close(fig)

    # 4. Sine test comparison (after P1 vs after P2 vs control)
    t_test = np.arange(TEST_STEPS) * DT
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(t_test, sine_truth, "k-", lw=1.5, label="Ground truth")
    axes[0].plot(t_test, sine_pred_after_p1, "g-", lw=1, alpha=0.8, label=f"After Phase 1 (err={sine_err_p1:.3f})")
    axes[0].set_ylabel("Sine")
    axes[0].set_title("Sine recall: after Phase 1 (before Lorenz)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_test, sine_truth, "k-", lw=1.5, label="Ground truth")
    axes[1].plot(t_test, sine_pred_after_p2, "r-", lw=1, alpha=0.8, label=f"After Phase 2 (err={sine_err_p2:.3f})")
    axes[1].set_ylabel("Sine")
    axes[1].set_title("Sine recall: after Phase 2 (after Lorenz training)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_test, sine_truth, "k-", lw=1.5, label="Ground truth")
    axes[2].plot(t_test, ctrl_sine_pred, "b-", lw=1, alpha=0.8, label=f"Control (err={ctrl_sine_err:.3f})")
    axes[2].set_ylabel("Sine")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_title("Sine recall: control (fresh network)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/sine_comparison.png", dpi=150)
    plt.close(fig)

    # 5. Lorenz test
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    labels = ["x", "y", "z"]
    for i, (ax, lbl) in enumerate(zip(axes, labels)):
        ax.plot(t_test, lorenz_truth[:, i], "k-", lw=1.5, label="Ground truth")
        ax.plot(t_test, lorenz_pred_after_p2[:, i], "r-", lw=1, alpha=0.8, label="After Phase 2")
        ax.set_ylabel(lbl)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(f"Lorenz after Phase 2 (err={lorenz_err_p2:.3f})")
    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/lorenz_test.png", dpi=150)
    plt.close(fig)

    # ── Summary ──────────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"{'Metric':<40} {'Value':>10}")
    log.info("-" * 52)
    log.info(f"{'Sine error after Phase 1':<40} {sine_err_p1:>10.4f}")
    log.info(f"{'Sine error after Phase 2':<40} {sine_err_p2:>10.4f}")
    log.info(f"{'Sine forgetting ratio (P2/P1)':<40} {sine_err_p2/max(sine_err_p1,1e-8):>10.2f}x")
    log.info(f"{'Control sine error':<40} {ctrl_sine_err:>10.4f}")
    log.info(f"{'Lorenz error after Phase 2':<40} {lorenz_err_p2:>10.4f}")
    log.info(f"{'Control Lorenz error':<40} {ctrl_lorenz_err:>10.4f}")
    log.info(f"{'||ΔM|| (Frobenius)':<40} {M_frob:>10.2f}")
    log.info(f"{'||ΔM|| / ||M_phase1||':<40} {M_frob/M_frob_p1:>10.4f}")

    # Forgetting classification
    if sine_err_p2 < sine_err_p1 * 1.5:
        log.info("\nVerdict: MINIMAL FORGETTING — sine survives Lorenz training")
    elif sine_err_p2 < sine_err_p1 * 3.0:
        log.info("\nVerdict: MODERATE FORGETTING — sine degraded but recognizable")
    else:
        log.info("\nVerdict: CATASTROPHIC FORGETTING — sine destroyed by Lorenz training")

    np.savez(f"{RESULTS_DIR}/results.npz",
             phase1_errors=np.array(phase1_errors),
             phase2_errors=np.array(phase2_errors),
             forgetting_curve=np.array(forgetting_curve),
             sine_pred_p1=sine_pred_after_p1,
             sine_pred_p2=sine_pred_after_p2,
             lorenz_pred_p2=lorenz_pred_after_p2)
    log.info(f"\nAll results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
