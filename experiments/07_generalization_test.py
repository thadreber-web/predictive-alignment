"""Generalization test: 07d (input-driven) vs 07a (autonomous) on novel initial conditions.

Trains both networks on (θ=2.0, ω=0.0), then tests on:
  - (θ=1.0, ω=0.0)  — smaller amplitude
  - (θ=2.5, ω=1.0)  — larger amplitude + initial velocity

If 07d generalizes and 07a doesn't, 07d learned physics, not just a trajectory.
"""

import sys
sys.path.insert(0, "/raid/predictive_alignment")

import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from src.network import PredictiveAlignmentRNN
from src.targets import generate_pendulum
from src.utils import set_seed

# ── Shared config ────────────────────────────────────────────────────
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
RESULTS_DIR = "/raid/predictive_alignment/results/07_generalization"

# Training IC
TRAIN_THETA0 = 2.0
TRAIN_OMEGA0 = 0.0

# Novel ICs to test
TEST_ICS = [
    (1.0, 0.0, "mild"),
    (2.5, 1.0, "hard"),
]

TEST_MS = 5_000.0
TEST_STEPS = int(TEST_MS / DT)


def pendulum_energy(theta, omega):
    return 0.5 * omega ** 2 + (GRAV / L) * (1 - np.cos(theta))


def train_07a():
    """Train 07a: 50s continuous, autonomous."""
    N, K = 500, 2
    TRAJ_MS = 50_000.0
    TRAIN_STEPS = int(TRAJ_MS / DT)

    set_seed(SEED)
    traj = generate_pendulum(duration_ms=TRAJ_MS, dt=DT, b=B, g=GRAV, L=L,
                             theta0=TRAIN_THETA0, omega0=TRAIN_OMEGA0)

    net = PredictiveAlignmentRNN(
        N=N, K=K, g=G, tau=TAU, dt=DT,
        eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
        device=torch.device(DEVICE), seed=SEED,
    )

    for step in tqdm(range(TRAIN_STEPS), desc="07a Train", mininterval=2.0):
        target = torch.tensor(traj[step], device=DEVICE, dtype=torch.float32)
        net.step_and_learn(target)

    return net


def train_07d():
    """Train 07d: 10 repeats of 5s, input-driven with teacher forcing."""
    N, K, D = 500, 2, 2
    TRAJ_MS = 5_000.0
    N_REPEATS = 10
    TRAIN_STEPS = int(TRAJ_MS * N_REPEATS / DT)
    TRAJ_STEPS = int(TRAJ_MS / DT)

    set_seed(SEED)
    traj = generate_pendulum(duration_ms=TRAJ_MS, dt=DT, b=B, g=GRAV, L=L,
                             theta0=TRAIN_THETA0, omega0=TRAIN_OMEGA0)

    net = PredictiveAlignmentRNN(
        N=N, K=K, D=D, g=G, tau=TAU, dt=DT,
        eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
        device=torch.device(DEVICE), seed=SEED,
    )

    for step in tqdm(range(TRAIN_STEPS), desc="07d Train", mininterval=2.0):
        traj_idx = step % TRAJ_STEPS
        target = torch.tensor(traj[traj_idx], device=DEVICE, dtype=torch.float32)
        if traj_idx > 0:
            ext_input = torch.tensor(traj[traj_idx - 1], device=DEVICE, dtype=torch.float32)
        else:
            ext_input = torch.zeros(D, device=DEVICE, dtype=torch.float32)
        net.step_and_learn(target, external_input=ext_input)

    return net


def test_autonomous(net, test_steps):
    """Run 07a-style: no input, just free-running dynamics."""
    zs = []
    for _ in range(test_steps):
        z = net.step()
        zs.append(z.detach().cpu().numpy())
    return np.array(zs)


def test_self_feeding(net, ic_theta, ic_omega, test_steps):
    """Run 07d-style: feed own previous output as input."""
    K = 2
    prev_z = torch.tensor([ic_theta, ic_omega], device=DEVICE, dtype=torch.float32)
    zs = []
    for _ in range(test_steps):
        z = net.step(external_input=prev_z)
        prev_z = z.detach().clone()
        zs.append(z.detach().cpu().numpy())
    return np.array(zs)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Training 07a (autonomous, 50s continuous)...")
    net_a = train_07a()

    print("\nTraining 07d (input-driven, teacher forcing)...")
    net_d = train_07d()

    # Generate ground truth for each novel IC
    for ic_theta, ic_omega, ic_label in TEST_ICS:
        print(f"\n{'='*60}")
        print(f"Testing IC: θ={ic_theta}, ω={ic_omega} ({ic_label})")
        print(f"{'='*60}")

        truth = generate_pendulum(
            duration_ms=TEST_MS, dt=DT, b=B, g=GRAV, L=L,
            theta0=ic_theta, omega0=ic_omega,
        )[:TEST_STEPS]

        # 07a: autonomous (no way to inject novel IC — just free-runs)
        z_a = test_autonomous(net_a, TEST_STEPS)

        # 07d: self-feeding, seeded with novel IC
        z_d = test_self_feeding(net_d, ic_theta, ic_omega, TEST_STEPS)

        # Compute errors
        err_a = np.mean(np.linalg.norm(truth - z_a, axis=1))
        err_d = np.mean(np.linalg.norm(truth - z_d, axis=1))
        print(f"  07a mean error: {err_a:.4f}")
        print(f"  07d mean error: {err_d:.4f}")

        t = np.arange(TEST_STEPS) * DT

        # ── Plot: θ(t) and ω(t) ──
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        axes[0].plot(t, truth[:, 0], "k-", lw=1.5, label="Ground truth")
        axes[0].plot(t, z_a[:, 0], "b--", lw=1, alpha=0.8, label=f"07a autonomous (err={err_a:.3f})")
        axes[0].plot(t, z_d[:, 0], "r-", lw=1, alpha=0.8, label=f"07d self-feeding (err={err_d:.3f})")
        axes[0].set_ylabel("θ (rad)")
        axes[0].set_title(f"Novel IC: θ₀={ic_theta}, ω₀={ic_omega} — trained on θ₀=2.0, ω₀=0.0")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, truth[:, 1], "k-", lw=1.5, label="Ground truth")
        axes[1].plot(t, z_a[:, 1], "b--", lw=1, alpha=0.8, label="07a autonomous")
        axes[1].plot(t, z_d[:, 1], "r-", lw=1, alpha=0.8, label="07d self-feeding")
        axes[1].set_ylabel("ω (rad/s)")
        axes[1].set_xlabel("Time (ms)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/generalization_{ic_label}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {RESULTS_DIR}/generalization_{ic_label}.png")

        # ── Phase portrait ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, data, label, color in [
            (axes[0], truth, "Ground truth", "k"),
            (axes[1], z_a, "07a autonomous", "b"),
            (axes[2], z_d, "07d self-feeding", "r"),
        ]:
            ax.plot(data[:, 0], data[:, 1], color=color, lw=0.8, alpha=0.8)
            ax.plot(data[0, 0], data[0, 1], "o", color=color, ms=8)
            ax.set_xlabel("θ")
            ax.set_ylabel("ω")
            ax.set_title(label)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-8, 8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")

        fig.suptitle(f"Phase portraits — IC: θ₀={ic_theta}, ω₀={ic_omega}", fontsize=13)
        plt.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/phase_{ic_label}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {RESULTS_DIR}/phase_{ic_label}.png")

        # ── Energy ──
        E_truth = pendulum_energy(truth[:, 0], truth[:, 1])
        E_a = pendulum_energy(z_a[:, 0], z_a[:, 1])
        E_d = pendulum_energy(z_d[:, 0], z_d[:, 1])

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t, E_truth, "k-", lw=1.5, label="Ground truth")
        ax.plot(t, E_a, "b--", lw=1, alpha=0.8, label="07a autonomous")
        ax.plot(t, E_d, "r-", lw=1, alpha=0.8, label="07d self-feeding")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Energy")
        ax.set_title(f"Energy — IC: θ₀={ic_theta}, ω₀={ic_omega}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/energy_{ic_label}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {RESULTS_DIR}/energy_{ic_label}.png")

    print(f"\nAll results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
