"""Experiment 2.1: Scheduled sampling on damped pendulum.

Multi-IC training with linear annealing from teacher forcing to self-feeding.
Tests generalization to held-out initial conditions.

Architecture: N=500, K=2 (theta, omega), D=2 (state input)
Training: 500 epochs, each epoch = 1 random IC trajectory (5s)
Schedule: p(epoch) = max(0, 1 - epoch/N_ANNEAL) where p=prob of teacher forcing
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
import time

from src.network import PredictiveAlignmentRNN
from src.targets import generate_pendulum
from src.utils import set_seed

# ── Config ───────────────────────────────────────────────────────────
N = 500
K = 2
D = 2
TAU = 10.0
DT = 1.0
ETA_W = 1e-3
ETA_M = 1e-3
ALPHA = 1.0
G = 1.2
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pendulum
B = 0.5
GRAV = 9.81
L_PEND = 1.0

# Training
N_EPOCHS = 500
TRAJ_MS = 5_000.0
TRAJ_STEPS = int(TRAJ_MS / DT)
N_ANNEAL = 400           # epochs to fully transition to self-feeding

# IC sampling ranges
THETA_RANGE = (-2.5, 2.5)
OMEGA_RANGE = (-3.0, 3.0)

# Test ICs (held out)
TEST_ICS = [
    (1.0, 0.0, "interp_mild"),
    (2.5, 1.0, "interp_hard"),
    (0.3, 0.5, "extrap_tiny"),
    (-1.5, 2.0, "negative_theta"),
    (2.0, -2.5, "negative_omega"),
]
TEST_MS = 5_000.0
TEST_STEPS = int(TEST_MS / DT)

# Checkpoints: run self-feeding test every N epochs to monitor stability
CHECKPOINT_EVERY = 50

RECORD_EVERY = 10
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "exp2_1_scheduled_sampling")


def pendulum_energy(theta, omega):
    return 0.5 * omega ** 2 + (GRAV / L_PEND) * (1 - np.cos(theta))


def test_self_feeding(net, ic_theta, ic_omega, test_steps):
    """Run self-feeding test from given IC."""
    net.reset_state()
    prev_z = torch.tensor([ic_theta, ic_omega], device=DEVICE, dtype=torch.float32)
    zs = []
    for _ in range(test_steps):
        z = net.step(external_input=prev_z)
        prev_z = z.detach().clone()
        zs.append(z.detach().cpu().numpy())
    return np.array(zs)


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
    log = logging.getLogger("exp2.1")

    log.info(f"Exp 2.1: Scheduled sampling pendulum")
    log.info(f"N={N}, K={K}, D={D}, epochs={N_EPOCHS}, anneal={N_ANNEAL}")
    log.info(f"θ range: {THETA_RANGE}, ω range: {OMEGA_RANGE}")
    log.info(f"Device: {DEVICE}")

    set_seed(SEED)
    rng = np.random.RandomState(SEED)

    net = PredictiveAlignmentRNN(
        N=N, K=K, D=D, g=G, tau=TAU, dt=DT,
        eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
        device=torch.device(DEVICE), seed=SEED,
    )

    # ── Training ─────────────────────────────────────────────────────
    epoch_errors = []
    epoch_tf_ratios = []     # actual teacher forcing ratio each epoch
    checkpoint_results = []  # (epoch, ic_label, error)

    t_start = time.time()

    for epoch in range(N_EPOCHS):
        # Scheduled sampling probability
        p_tf = max(0.0, 1.0 - epoch / N_ANNEAL)

        # Sample random IC
        theta0 = rng.uniform(*THETA_RANGE)
        omega0 = rng.uniform(*OMEGA_RANGE)

        # Generate trajectory
        traj = generate_pendulum(
            duration_ms=TRAJ_MS, dt=DT, b=B, g=GRAV, L=L_PEND,
            theta0=theta0, omega0=omega0,
        )

        # Reset network state for new trajectory
        net.reset_state()

        errors = []
        tf_count = 0
        prev_z = torch.tensor([theta0, omega0], device=DEVICE, dtype=torch.float32)

        for t in range(TRAJ_STEPS - 1):
            # Target: next state
            target_np = traj[t + 1]
            target = torch.tensor(target_np, device=DEVICE, dtype=torch.float32)

            # Scheduled sampling: choose input
            if rng.random() < p_tf:
                # Teacher forcing: feed ground truth current state
                ext_input = torch.tensor(traj[t], device=DEVICE, dtype=torch.float32)
                tf_count += 1
            else:
                # Self-feeding: feed network's own previous prediction
                ext_input = prev_z

            z = net.step_and_learn(target, external_input=ext_input)
            prev_z = z.detach().clone()

            if t % RECORD_EVERY == 0:
                errors.append(torch.norm(target - z).item())

        epoch_err = np.mean(errors)
        epoch_errors.append(epoch_err)
        epoch_tf_ratios.append(tf_count / (TRAJ_STEPS - 1))

        if epoch % 25 == 0 or epoch == N_EPOCHS - 1:
            log.info(f"Epoch {epoch:4d}/{N_EPOCHS} | p_tf={p_tf:.3f} | "
                     f"actual_tf={epoch_tf_ratios[-1]:.3f} | error={epoch_err:.4f}")

        # Checkpoint: test self-feeding on one IC
        if epoch % CHECKPOINT_EVERY == 0 or epoch == N_EPOCHS - 1:
            ic_th, ic_om = 1.0, 0.0  # fixed checkpoint IC
            truth = generate_pendulum(
                duration_ms=TEST_MS, dt=DT, b=B, g=GRAV, L=L_PEND,
                theta0=ic_th, omega0=ic_om,
            )[:TEST_STEPS]
            z_test = test_self_feeding(net, ic_th, ic_om, TEST_STEPS)
            ckpt_err = np.mean(np.linalg.norm(truth - z_test, axis=1))
            checkpoint_results.append((epoch, ckpt_err))
            log.info(f"  Checkpoint (θ=1.0,ω=0.0) self-feeding error: {ckpt_err:.4f}")

    elapsed = time.time() - t_start
    log.info(f"Training done in {elapsed:.1f}s")

    # ── Plot: training error + TF ratio ──────────────────────────────
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(epoch_errors, "b-", alpha=0.5, linewidth=0.5, label="Epoch error")
    # Smooth
    w = max(1, len(epoch_errors) // 50)
    smoothed = np.convolve(epoch_errors, np.ones(w)/w, mode="valid")
    ax1.plot(np.arange(w-1, w-1+len(smoothed)), smoothed, "b-", linewidth=2, label="Smoothed error")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Error", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epoch_tf_ratios, "r-", alpha=0.3, linewidth=0.5)
    ax2.set_ylabel("Teacher Forcing Ratio", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.set_ylim(-0.05, 1.05)

    ax1.set_title("Exp 2.1: Training Error & Teacher Forcing Schedule")
    ax1.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/training_curve.png", dpi=150)
    plt.close(fig)

    # ── Plot: checkpoint self-feeding error over training ────────────
    ckpt_epochs, ckpt_errs = zip(*checkpoint_results)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ckpt_epochs, ckpt_errs, "go-", markersize=6, linewidth=2)
    ax.axhline(y=2.25, color="gray", linestyle="--", alpha=0.5, label="07e multi-IC baseline")
    ax.axhline(y=5.5, color="gray", linestyle=":", alpha=0.5, label="07d single-IC baseline")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Self-feeding Error (θ=1.0, ω=0.0)")
    ax.set_title("Exp 2.1: Self-feeding Stability During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/checkpoint_stability.png", dpi=150)
    plt.close(fig)

    # ── Final generalization test on all held-out ICs ────────────────
    log.info(f"\nFinal generalization test on {len(TEST_ICS)} held-out ICs:")
    summary = []

    for ic_theta, ic_omega, ic_label in TEST_ICS:
        truth = generate_pendulum(
            duration_ms=TEST_MS, dt=DT, b=B, g=GRAV, L=L_PEND,
            theta0=ic_theta, omega0=ic_omega,
        )[:TEST_STEPS]

        z_pred = test_self_feeding(net, ic_theta, ic_omega, TEST_STEPS)
        err = np.mean(np.linalg.norm(truth - z_pred, axis=1))
        summary.append((ic_label, ic_theta, ic_omega, err))
        log.info(f"  {ic_label:20s} θ={ic_theta:+.1f} ω={ic_omega:+.1f} → error={err:.4f}")

        t = np.arange(TEST_STEPS) * DT

        # Time series
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        axes[0].plot(t, truth[:, 0], "k-", lw=1.5, label="Ground truth")
        axes[0].plot(t, z_pred[:, 0], "r-", lw=1, alpha=0.8, label=f"Predicted (err={err:.3f})")
        axes[0].set_ylabel("θ (rad)")
        axes[0].set_title(f"Exp 2.1: θ₀={ic_theta}, ω₀={ic_omega} ({ic_label})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, truth[:, 1], "k-", lw=1.5, label="Ground truth")
        axes[1].plot(t, z_pred[:, 1], "r-", lw=1, alpha=0.8, label="Predicted")
        axes[1].set_ylabel("ω (rad/s)")
        axes[1].set_xlabel("Time (ms)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/test_{ic_label}.png", dpi=150)
        plt.close(fig)

        # Phase portrait
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, data, label, color in [
            (axes[0], truth, "Ground truth", "k"),
            (axes[1], z_pred, "Predicted", "r"),
        ]:
            ax.plot(data[:, 0], data[:, 1], color=color, lw=0.8, alpha=0.8)
            ax.plot(data[0, 0], data[0, 1], "o", color=color, ms=8)
            ax.set_xlabel("θ")
            ax.set_ylabel("ω")
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Phase — θ₀={ic_theta}, ω₀={ic_omega}", fontsize=13)
        plt.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/phase_{ic_label}.png", dpi=150)
        plt.close(fig)

        # Energy
        E_truth = pendulum_energy(truth[:, 0], truth[:, 1])
        E_pred = pendulum_energy(z_pred[:, 0], z_pred[:, 1])
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t, E_truth, "k-", lw=1.5, label="Ground truth")
        ax.plot(t, E_pred, "r-", lw=1, alpha=0.8, label="Predicted")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Energy")
        ax.set_title(f"Energy — θ₀={ic_theta}, ω₀={ic_omega}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/energy_{ic_label}.png", dpi=150)
        plt.close(fig)

    # ── Summary ──────────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"{'IC':<20} {'θ₀':>6} {'ω₀':>6} {'Error':>10}")
    log.info("-" * 45)
    for label, th, om, err in summary:
        log.info(f"{label:<20} {th:>+6.1f} {om:>+6.1f} {err:>10.4f}")
    mean_err = np.mean([e for _, _, _, e in summary])
    log.info(f"\nMean generalization error: {mean_err:.4f}")
    log.info(f"Success threshold: < 0.5")
    log.info(f"07e baseline: ~2.3")
    log.info(f"07d baseline: ~5.5")

    # Save data
    np.savez(f"{RESULTS_DIR}/results.npz",
             epoch_errors=np.array(epoch_errors),
             epoch_tf_ratios=np.array(epoch_tf_ratios),
             checkpoint_results=np.array(checkpoint_results),
             summary=np.array([(th, om, err) for _, th, om, err in summary]))
    log.info(f"\nAll results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
