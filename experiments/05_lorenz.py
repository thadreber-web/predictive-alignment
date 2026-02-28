"""Experiment 05: Lorenz attractor — complex chaotic target.

N=500, K=3 readouts, train on pre-generated Lorenz trajectory for 15,000s.
"""

import sys
sys.path.insert(0, "/raid/predictive_alignment")

import torch
import numpy as np
import logging
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from src.network import PredictiveAlignmentRNN
from src.targets import generate_lorenz
from src.instrumentation import (plot_phase_portrait, plot_3d_trajectory,
                                  plot_eigenspectrum, TrainingMonitor)
from src.utils import estimate_lyapunov, compute_eigenspectrum, set_seed

# ── Config ──────────────────────────────────────────────────────────
N = 500
K = 3
TAU = 10.0
DT = 1.0
ETA_W = 1e-3
ETA_M = 1e-3
ALPHA = 1.0
G = 1.2

TRAIN_MS = 15_000_000.0   # 15,000 seconds
TEST_MS = 5_000.0          # 5 seconds test

TRAIN_STEPS = int(TRAIN_MS / DT)
TEST_STEPS = int(TEST_MS / DT)

RECORD_EVERY = 100
SEED = 42

RESULTS_DIR = "/raid/predictive_alignment/results/05_lorenz"
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
    log = logging.getLogger("exp05")
    log.info(f"Config: N={N}, K={K}, g={G}, dt={DT}, eta_w={ETA_W}, eta_m={ETA_M}")
    log.info(f"Train: {TRAIN_MS/1000:.0f}s ({TRAIN_STEPS} steps)")
    log.info(f"Log: tail -f {LOG_FILE}")

    set_seed(SEED)

    log.info("Generating Lorenz trajectory...")
    # Generate trajectory: need TRAIN_STEPS + TEST_STEPS points
    total_steps = TRAIN_STEPS + TEST_STEPS
    lorenz_traj = generate_lorenz(
        duration_ms=(total_steps + 1) * DT,
        dt=DT, scale=0.1,
    )
    log.info(f"Lorenz trajectory shape: {lorenz_traj.shape}")

    # Truncate to needed length
    if len(lorenz_traj) > total_steps:
        lorenz_traj = lorenz_traj[:total_steps]

    net = PredictiveAlignmentRNN(
        N=N, K=K, g=G, tau=TAU, dt=DT,
        eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
        device=torch.device(DEVICE), seed=SEED,
    )

    # Eigenspectrum before
    eigs_before = compute_eigenspectrum(net.get_J())

    # ── Training ────────────────────────────────────────────────────
    errors = []
    z_late = []
    f_late = []

    log.info(f"Training for {TRAIN_STEPS} steps ({TRAIN_MS/1000:.0f}s simulated)...")

    for step in tqdm(range(TRAIN_STEPS), desc="Training", mininterval=5.0):
        target_np = lorenz_traj[step]
        target = torch.tensor(target_np, device=DEVICE, dtype=torch.float32)
        z = net.step_and_learn(target)

        if step % RECORD_EVERY == 0:
            err = torch.norm(target - z).item()
            errors.append(err)

        if step > 0 and step % 1_000_000 == 0:
            mean_err = np.mean(errors[-1000:])
            log.info(f"  step={step}/{TRAIN_STEPS} ({step*DT/1000:.0f}s): mean_err={mean_err:.6f}")

        # Record late training output (last 1% of training)
        if step >= int(0.99 * TRAIN_STEPS):
            z_late.append(z.detach().cpu().numpy())
            f_late.append(target_np)

    z_late = np.array(z_late)
    f_late = np.array(f_late)
    errors = np.array(errors)

    eigs_after = compute_eigenspectrum(net.get_J())

    # ── Testing (plasticity off) ────────────────────────────────────
    log.info(f"Training complete. Final mean error: {np.mean(errors[-100:]):.6f}")
    log.info(f"Testing for {TEST_STEPS} steps (plasticity off)...")
    test_z = []
    test_f = []

    for step in tqdm(range(TEST_STEPS), desc="Testing", mininterval=2.0):
        idx = TRAIN_STEPS + step
        if idx < len(lorenz_traj):
            target_np = lorenz_traj[idx]
        else:
            break
        z = net.step()
        test_z.append(z.detach().cpu().numpy())
        test_f.append(target_np)

    test_z = np.array(test_z)
    test_f = np.array(test_f)

    # ── Plots ───────────────────────────────────────────────────────

    # 1. Late training: each component
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    n_plot = min(5000, len(z_late))
    labels_dim = ["x", "y", "z"]
    for i in range(3):
        axes[i].plot(f_late[-n_plot:, i], "k-", alpha=0.7, label="target", linewidth=0.5)
        axes[i].plot(z_late[-n_plot:, i], "r-", alpha=0.7, label="output", linewidth=0.5)
        axes[i].set_ylabel(f"Lorenz {labels_dim[i]}")
        if i == 0:
            axes[i].legend()
    axes[-1].set_xlabel("Step")
    fig.suptitle("Exp 05: Lorenz Components (Late Training)")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/lorenz_components_late.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/lorenz_components_late.png")
    plt.close(fig)

    # 2. 2D projections
    for dims, name in [((0, 1), "xy"), ((1, 2), "yz"), ((0, 2), "xz")]:
        plot_phase_portrait(
            [f_late[-n_plot:], z_late[-n_plot:]],
            labels=["Target", "Output"],
            dims=dims,
            save_path=f"{RESULTS_DIR}/lorenz_2d_{name}.png",
            title=f"Exp 05: Lorenz {name.upper()} Projection",
        )

    # 3. 3D trajectory
    plot_3d_trajectory(
        [f_late[-n_plot:], z_late[-n_plot:]],
        labels=["Target", "Output"],
        save_path=f"{RESULTS_DIR}/lorenz_3d.png",
        title="Exp 05: Lorenz 3D Trajectory",
    )

    # 4. Error over training
    fig, ax = plt.subplots(figsize=(12, 4))
    window = max(1, len(errors) // 500)
    if window > 1:
        smoothed = np.convolve(errors, np.ones(window) / window, mode="valid")
    else:
        smoothed = errors
    ax.plot(smoothed, "b-", alpha=0.7, linewidth=0.5)
    ax.set_xlabel("Recording step")
    ax.set_ylabel("Error ||f - z||")
    ax.set_title("Exp 05: Training Error")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/lorenz_error.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/lorenz_error.png")
    plt.close(fig)

    # 5. Tent map: successive maxima of z3 component
    fig, ax = plt.subplots(figsize=(6, 6))
    for data, label, color in [(f_late[:, 2], "Target", "black"), (z_late[:, 2], "Output", "red")]:
        # Find local maxima
        maxima = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                maxima.append(data[i])
        if len(maxima) > 2:
            ax.scatter(maxima[:-1], maxima[1:], s=2, alpha=0.4, label=label, color=color)
    ax.set_xlabel("z3 max(n)")
    ax.set_ylabel("z3 max(n+1)")
    ax.set_title("Exp 05: Tent Map (Successive Maxima of z3)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/lorenz_tent_map.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/lorenz_tent_map.png")
    plt.close(fig)

    # 6. Eigenspectrum
    plot_eigenspectrum(
        [eigs_before, eigs_after],
        labels=["Before", "After"],
        save_path=f"{RESULTS_DIR}/eigenspectrum.png",
        title="Exp 05: Eigenspectrum Before/After Lorenz Training",
    )

    # Save data
    np.savez(f"{RESULTS_DIR}/results.npz",
             errors=errors, z_late=z_late, f_late=f_late,
             test_z=test_z, test_f=test_f)
    print(f"Saved: {RESULTS_DIR}/results.npz")
    print("Done!")


if __name__ == "__main__":
    main()
