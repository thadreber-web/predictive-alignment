"""Experiment 07e: Train 07d on multiple ICs, test generalization.

Training ICs (5 trajectories, 5s each, 5 repeats each = 125k steps):
  (0.5, 0.0), (1.5, 0.0), (2.0, 0.0), (2.0, 2.0), (3.0, -1.0)

Test ICs (held out):
  (1.0, 0.0)  — interpolation, mild
  (2.5, 1.0)  — interpolation, hard
  (0.3, 0.5)  — extrapolation, tiny

Interleaves trajectories during training so the network sees varied
dynamics each epoch rather than memorizing one sequence.
"""

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.network import PredictiveAlignmentRNN
from src.targets import generate_pendulum
from src.utils import set_seed

# ── Config ───────────────────────────────────────────────────────────
B = 0.5
GRAV = 9.81
L = 1.0
DT = 1.0
TAU = 10.0
ETA_W = 1e-3
ETA_M = 1e-3
ALPHA = 1.0
G = 1.2
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "07e_multiIC")

N = 500
K = 2
D = 2

TRAJ_MS = 5_000.0
TRAJ_STEPS = int(TRAJ_MS / DT)
N_REPEATS = 5

TEST_MS = 5_000.0
TEST_STEPS = int(TEST_MS / DT)

RECORD_EVERY = 10

# Training ICs — spread across phase space
TRAIN_ICS = [
    (0.5, 0.0),
    (1.5, 0.0),
    (2.0, 0.0),
    (2.0, 2.0),
    (3.0, -1.0),
]

# Held-out test ICs
TEST_ICS = [
    (1.0, 0.0, "interp_mild"),
    (2.5, 1.0, "interp_hard"),
    (0.3, 0.5, "extrap_tiny"),
]


def pendulum_energy(theta, omega):
    return 0.5 * omega ** 2 + (GRAV / L) * (1 - np.cos(theta))


def test_self_feeding(net, ic_theta, ic_omega, test_steps):
    """Self-feeding test: seed with IC, feed own output back."""
    prev_z = torch.tensor([ic_theta, ic_omega], device=DEVICE, dtype=torch.float32)
    zs = []
    for _ in range(test_steps):
        z = net.step(external_input=prev_z)
        prev_z = z.detach().clone()
        zs.append(z.detach().cpu().numpy())
    return np.array(zs)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    set_seed(SEED)

    # ── Generate all training trajectories ───────────────────────────
    print(f"Generating {len(TRAIN_ICS)} training trajectories...")
    trajs = []
    for theta0, omega0 in TRAIN_ICS:
        traj = generate_pendulum(
            duration_ms=TRAJ_MS, dt=DT, b=B, g=GRAV, L=L,
            theta0=theta0, omega0=omega0,
        )
        trajs.append(traj)
        print(f"  IC ({theta0:+.1f}, {omega0:+.1f}): {traj.shape}")

    n_trajs = len(trajs)
    total_steps = TRAJ_STEPS * N_REPEATS * n_trajs

    # ── Build network ────────────────────────────────────────────────
    net = PredictiveAlignmentRNN(
        N=N, K=K, D=D, g=G, tau=TAU, dt=DT,
        eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
        device=torch.device(DEVICE), seed=SEED,
    )

    # ── Training: interleave trajectories ────────────────────────────
    # Each "epoch" cycles through all trajectories in random order.
    # Within each trajectory, present TRAJ_STEPS sequential steps.
    print(f"\nTraining: {n_trajs} trajs x {N_REPEATS} repeats x {TRAJ_STEPS} steps = {total_steps} total")

    errors = []
    rng = np.random.RandomState(SEED)

    for repeat in range(N_REPEATS):
        order = rng.permutation(n_trajs)
        for traj_i in order:
            traj = trajs[traj_i]
            desc = f"Rep {repeat+1}/{N_REPEATS}, traj {traj_i} IC={TRAIN_ICS[traj_i]}"

            # Reset network state at start of each trajectory so it doesn't
            # carry hidden state from a different IC
            net.reset_state()

            for t in range(TRAJ_STEPS):
                target_np = traj[t]
                target = torch.tensor(target_np, device=DEVICE, dtype=torch.float32)

                if t > 0:
                    ext_input = torch.tensor(traj[t - 1], device=DEVICE, dtype=torch.float32)
                else:
                    # First step: feed the IC itself as input
                    ext_input = torch.tensor(
                        [TRAIN_ICS[traj_i][0], TRAIN_ICS[traj_i][1]],
                        device=DEVICE, dtype=torch.float32,
                    )

                z = net.step_and_learn(target, external_input=ext_input)

                if t % RECORD_EVERY == 0:
                    errors.append(torch.norm(target - z).item())

        print(f"  Repeat {repeat+1}/{N_REPEATS} done — recent error: {np.mean(errors[-500:]):.4f}")

    errors = np.array(errors)
    final_err = np.mean(errors[-500:])
    print(f"\nTraining complete. Final mean error: {final_err:.6f}")

    # ── Training error plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    window = max(1, len(errors) // 500)
    smoothed = np.convolve(errors, np.ones(window) / window, mode="valid")
    ax.plot(smoothed, "b-", alpha=0.7, linewidth=0.5)
    ax.set_xlabel("Recording step")
    ax.set_ylabel("Error ||f - z||")
    ax.set_title("07e: Training Error (multi-IC)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/training_error.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {RESULTS_DIR}/training_error.png")

    # ── Test on held-out ICs ─────────────────────────────────────────
    print(f"\nTesting on {len(TEST_ICS)} held-out ICs...")
    summary = []

    for ic_theta, ic_omega, ic_label in TEST_ICS:
        print(f"\n  IC: θ={ic_theta}, ω={ic_omega} ({ic_label})")

        truth = generate_pendulum(
            duration_ms=TEST_MS, dt=DT, b=B, g=GRAV, L=L,
            theta0=ic_theta, omega0=ic_omega,
        )[:TEST_STEPS]

        # Reset state before test
        net.reset_state()
        z_pred = test_self_feeding(net, ic_theta, ic_omega, TEST_STEPS)

        err = np.mean(np.linalg.norm(truth - z_pred, axis=1))
        summary.append((ic_label, ic_theta, ic_omega, err))
        print(f"    Mean error: {err:.4f}")

        t = np.arange(TEST_STEPS) * DT

        # Time series
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        axes[0].plot(t, truth[:, 0], "k-", lw=1.5, label="Ground truth")
        axes[0].plot(t, z_pred[:, 0], "r-", lw=1, alpha=0.8, label=f"07e multi-IC (err={err:.3f})")
        axes[0].set_ylabel("θ (rad)")
        axes[0].set_title(f"07e Multi-IC: θ₀={ic_theta}, ω₀={ic_omega} (held out)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, truth[:, 1], "k-", lw=1.5, label="Ground truth")
        axes[1].plot(t, z_pred[:, 1], "r-", lw=1, alpha=0.8, label="07e multi-IC")
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
            (axes[1], z_pred, "07e multi-IC", "r"),
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
        ax.plot(t, E_pred, "r-", lw=1, alpha=0.8, label="07e multi-IC")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Energy")
        ax.set_title(f"Energy — θ₀={ic_theta}, ω₀={ic_omega}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/energy_{ic_label}.png", dpi=150)
        plt.close(fig)

    # ── Also test on a training IC to confirm it still works ─────────
    print(f"\n  Sanity check: training IC θ=2.0, ω=0.0")
    truth_train = generate_pendulum(
        duration_ms=TEST_MS, dt=DT, b=B, g=GRAV, L=L,
        theta0=2.0, omega0=0.0,
    )[:TEST_STEPS]
    net.reset_state()
    z_train = test_self_feeding(net, 2.0, 0.0, TEST_STEPS)
    err_train = np.mean(np.linalg.norm(truth_train - z_train, axis=1))
    summary.append(("train_sanity", 2.0, 0.0, err_train))
    print(f"    Mean error: {err_train:.4f}")

    t = np.arange(TEST_STEPS) * DT
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(t, truth_train[:, 0], "k-", lw=1.5, label="Ground truth")
    axes[0].plot(t, z_train[:, 0], "r-", lw=1, alpha=0.8, label=f"07e (err={err_train:.3f})")
    axes[0].set_ylabel("θ (rad)")
    axes[0].set_title("07e Sanity: training IC θ₀=2.0, ω₀=0.0")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, truth_train[:, 1], "k-", lw=1.5, label="Ground truth")
    axes[1].plot(t, z_train[:, 1], "r-", lw=1, alpha=0.8, label="07e")
    axes[1].set_ylabel("ω (rad/s)")
    axes[1].set_xlabel("Time (ms)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/test_train_sanity.png", dpi=150)
    plt.close(fig)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'IC':<20} {'θ₀':>6} {'ω₀':>6} {'Error':>10}")
    print("-" * 45)
    for label, th, om, err in summary:
        print(f"{label:<20} {th:>6.1f} {om:>6.1f} {err:>10.4f}")
    print(f"\nPrevious single-IC 07d errors on these ICs: 5.504, 5.772")
    print(f"Results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
