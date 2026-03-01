"""Experiment 07: Damped pendulum — neural simulator.

N=500, K=2 (theta, omega), train 50s (10 repeats of 5s trajectory).
"""

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

import torch
import numpy as np
import logging
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.network import PredictiveAlignmentRNN
from src.targets import generate_pendulum
from src.instrumentation import plot_phase_portrait, TrainingMonitor
from src.utils import estimate_lyapunov, compute_eigenspectrum, set_seed

# ── Config ──────────────────────────────────────────────────────────
N = 500
K = 2        # theta, omega
TAU = 10.0
DT = 1.0
ETA_W = 1e-3
ETA_M = 1e-3
ALPHA = 1.0
G = 1.2

# Pendulum parameters
B = 0.5       # damping
GRAV = 9.81
L = 1.0
THETA0 = 2.0
OMEGA0 = 0.0

TRAJ_MS = 5_000.0      # 5 seconds per trajectory
N_REPEATS = 10          # show trajectory 10 times
TRAIN_MS = TRAJ_MS * N_REPEATS
TRAIN_STEPS = int(TRAIN_MS / DT)
TRAJ_STEPS = int(TRAJ_MS / DT)

TEST_MS = 5_000.0
TEST_STEPS = int(TEST_MS / DT)

RECORD_EVERY = 10
SEED = 42

RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "07_pendulum")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


LOG_FILE = f"{RESULTS_DIR}/experiment.log"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("exp07")
    log.info(f"Config: N={N}, K={K}, g={G}, dt={DT}, eta_w={ETA_W}, eta_m={ETA_M}")
    log.info(f"Train: {TRAIN_STEPS} steps ({N_REPEATS} repeats of {TRAJ_MS/1000:.0f}s)")
    log.info(f"Log: tail -f {LOG_FILE}")

    set_seed(SEED)

    log.info("Generating pendulum trajectory...")
    pend_traj = generate_pendulum(
        duration_ms=TRAJ_MS, dt=DT,
        b=B, g=GRAV, L=L, theta0=THETA0, omega0=OMEGA0,
    )
    print(f"Pendulum trajectory shape: {pend_traj.shape}")

    net = PredictiveAlignmentRNN(
        N=N, K=K, g=G, tau=TAU, dt=DT,
        eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
        device=torch.device(DEVICE), seed=SEED,
    )

    # ── Training ────────────────────────────────────────────────────
    log.info(f"Training for {TRAIN_STEPS} steps...")
    errors = []
    z_history = []
    f_history = []

    for step in tqdm(range(TRAIN_STEPS), desc="Training", mininterval=2.0):
        # Cycle through trajectory
        traj_idx = step % TRAJ_STEPS
        target_np = pend_traj[traj_idx]
        target = torch.tensor(target_np, device=DEVICE, dtype=torch.float32)

        z = net.step_and_learn(target)

        if step % RECORD_EVERY == 0:
            err = torch.norm(target - z).item()
            errors.append(err)

        # Record last repeat
        if step >= (N_REPEATS - 1) * TRAJ_STEPS:
            z_history.append(z.detach().cpu().numpy())
            f_history.append(target_np)

    z_history = np.array(z_history)
    f_history = np.array(f_history)
    errors = np.array(errors)

    log.info(f"Training complete. Final mean error: {np.mean(errors[-100:]):.6f}")

    # ── Test (plasticity off, same initial condition) ───────────────
    log.info(f"Testing for {TEST_STEPS} steps (plasticity off)...")
    test_z = []
    test_f = []

    # Reset network to start of trajectory
    # (The network should reproduce from its internal dynamics)
    for step in tqdm(range(TEST_STEPS), desc="Testing", mininterval=2.0):
        traj_idx = step % TRAJ_STEPS
        if traj_idx < len(pend_traj):
            target_np = pend_traj[traj_idx]
        else:
            target_np = pend_traj[-1]

        z = net.step()  # no learning
        test_z.append(z.detach().cpu().numpy())
        test_f.append(target_np)

    test_z = np.array(test_z)
    test_f = np.array(test_f)

    # ── Plots ───────────────────────────────────────────────────────

    # 1. theta(t) target vs output
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t_train = np.arange(len(z_history)) * DT * RECORD_EVERY if len(z_history) != TRAJ_STEPS else np.arange(len(z_history)) * DT
    t_plot = np.arange(len(z_history)) * DT

    axes[0].plot(t_plot, f_history[:, 0], "k-", alpha=0.7, label="target", linewidth=0.8)
    axes[0].plot(t_plot, z_history[:, 0], "r-", alpha=0.7, label="output", linewidth=0.8)
    axes[0].set_ylabel("theta (rad)")
    axes[0].set_title("Exp 07: Last Training Repeat")
    axes[0].legend()

    axes[1].plot(t_plot, f_history[:, 1], "k-", alpha=0.7, label="target", linewidth=0.8)
    axes[1].plot(t_plot, z_history[:, 1], "r-", alpha=0.7, label="output", linewidth=0.8)
    axes[1].set_ylabel("omega (rad/s)")
    axes[1].set_xlabel("Time (ms)")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/pendulum_training.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/pendulum_training.png")
    plt.close(fig)

    # 2. Test output
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t_test = np.arange(len(test_z)) * DT
    axes[0].plot(t_test, test_f[:, 0], "k-", alpha=0.7, label="target", linewidth=0.8)
    axes[0].plot(t_test, test_z[:, 0], "r-", alpha=0.7, label="output (autonomous)", linewidth=0.8)
    axes[0].set_ylabel("theta (rad)")
    axes[0].set_title("Exp 07: Autonomous Generation (Plasticity Off)")
    axes[0].legend()

    axes[1].plot(t_test, test_f[:, 1], "k-", alpha=0.7, label="target", linewidth=0.8)
    axes[1].plot(t_test, test_z[:, 1], "r-", alpha=0.7, label="output (autonomous)", linewidth=0.8)
    axes[1].set_ylabel("omega (rad/s)")
    axes[1].set_xlabel("Time (ms)")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/pendulum_test.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/pendulum_test.png")
    plt.close(fig)

    # 3. Phase portrait: theta vs omega
    plot_phase_portrait(
        [f_history, z_history],
        labels=["Target", "Output (training)"],
        dims=(0, 1),
        save_path=f"{RESULTS_DIR}/phase_portrait_train.png",
        title="Exp 07: Phase Portrait (Training)",
    )
    plot_phase_portrait(
        [test_f, test_z],
        labels=["Target", "Output (autonomous)"],
        dims=(0, 1),
        save_path=f"{RESULTS_DIR}/phase_portrait_test.png",
        title="Exp 07: Phase Portrait (Autonomous)",
    )

    # 4. Energy over time
    def pendulum_energy(theta, omega):
        return 0.5 * omega ** 2 + (GRAV / L) * (1 - np.cos(theta))

    E_target = pendulum_energy(test_f[:, 0], test_f[:, 1])
    E_output = pendulum_energy(test_z[:, 0], test_z[:, 1])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_test, E_target, "k-", label="Target energy", alpha=0.7)
    ax.plot(t_test, E_output, "r-", label="Output energy", alpha=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Energy")
    ax.set_title("Exp 07: Pendulum Energy (should decrease due to damping)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/pendulum_energy.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/pendulum_energy.png")
    plt.close(fig)

    # 5. Training error
    fig, ax = plt.subplots(figsize=(12, 4))
    window = max(1, len(errors) // 500)
    smoothed = np.convolve(errors, np.ones(window) / window, mode="valid")
    ax.plot(smoothed, "b-", alpha=0.7, linewidth=0.5)
    ax.set_xlabel("Recording step")
    ax.set_ylabel("Error ||f - z||")
    ax.set_title("Exp 07: Training Error")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/pendulum_error.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/pendulum_error.png")
    plt.close(fig)

    # Save data
    np.savez(f"{RESULTS_DIR}/results.npz",
             errors=errors, z_history=z_history, f_history=f_history,
             test_z=test_z, test_f=test_f)
    print(f"Saved: {RESULTS_DIR}/results.npz")
    print("Done!")


if __name__ == "__main__":
    main()
