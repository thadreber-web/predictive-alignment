"""Experiment 2.5: Neural ODE baseline for pendulum.

Learns dstate/dt = f_neural(state) where f_neural is a small MLP.
Trained with backprop through the ODE solver (adjoint method).
Tests generalization to held-out initial conditions.

This establishes the upper bound: can ANY method learn generalizable
pendulum dynamics from multi-IC training data?
"""

import sys
sys.path.insert(0, "/raid/predictive_alignment")

import torch
import torch.nn as nn
import numpy as np
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time

from src.targets import generate_pendulum
from src.utils import set_seed

# ── Config ───────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pendulum
B = 0.5
GRAV = 9.81
L_PEND = 1.0
DT = 1.0  # ms

# Training
N_EPOCHS = 500
TRAJ_MS = 5_000.0
TRAJ_STEPS = int(TRAJ_MS / DT)
DT_S = DT / 1000.0  # seconds for ODE integration
LR = 1e-3
SEGMENT_LEN = 50  # train on segments of 50 steps (backprop through solver)

# IC ranges (same as exp 2.0/2.1)
THETA_RANGE = (-2.5, 2.5)
OMEGA_RANGE = (-3.0, 3.0)

# Test ICs (same as exp 2.0/2.1)
TEST_ICS = [
    (1.0, 0.0, "interp_mild"),
    (2.5, 1.0, "interp_hard"),
    (0.3, 0.5, "extrap_tiny"),
    (-1.5, 2.0, "negative_theta"),
    (2.0, -2.5, "negative_omega"),
]
TEST_MS = 5_000.0
TEST_STEPS = int(TEST_MS / DT)

CHECKPOINT_EVERY = 50
RESULTS_DIR = "/raid/predictive_alignment/results/exp2_5_neural_ode"


class PendulumODEFunc(nn.Module):
    """MLP that learns dstate/dt = f(state).

    Input: (theta, omega) — shape (2,) or (batch, 2)
    Output: (dtheta/dt, domega/dt) — same shape
    """

    def __init__(self, hidden=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, t, state):
        return self.net(state)


def rk4_step(func, t, y, dt):
    """Single RK4 step."""
    k1 = func(t, y)
    k2 = func(t + dt/2, y + dt/2 * k1)
    k3 = func(t + dt/2, y + dt/2 * k2)
    k4 = func(t + dt, y + dt * k3)
    return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def integrate(func, y0, t_span, dt):
    """Integrate ODE with RK4, returns trajectory including y0."""
    n_steps = int((t_span[1] - t_span[0]) / dt)
    ys = [y0]
    y = y0
    t = t_span[0]
    for _ in range(n_steps):
        y = rk4_step(func, t, y, dt)
        ys.append(y)
        t = t + dt
    return torch.stack(ys)


def pendulum_energy(theta, omega):
    return 0.5 * omega ** 2 + (GRAV / L_PEND) * (1 - np.cos(theta))


def test_neural_ode(func, ic_theta, ic_omega, test_steps, device):
    """Generate trajectory from neural ODE."""
    func.eval()
    with torch.no_grad():
        y0 = torch.tensor([ic_theta, ic_omega], device=device, dtype=torch.float32)
        traj = integrate(func, y0, (0, test_steps * DT_S), DT_S)
    func.train()
    return traj.cpu().numpy()  # shape (test_steps+1, 2)


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
    log = logging.getLogger("exp2.5")

    set_seed(SEED)
    rng = np.random.RandomState(SEED)
    device = torch.device(DEVICE)

    func = PendulumODEFunc(hidden=64, n_layers=3).to(device)
    optimizer = torch.optim.Adam(func.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    n_params = sum(p.numel() for p in func.parameters())
    log.info(f"Exp 2.5: Neural ODE baseline")
    log.info(f"MLP: 2→64→64→64→2, {n_params:,} params")
    log.info(f"Epochs={N_EPOCHS}, segment_len={SEGMENT_LEN}, lr={LR}")
    log.info(f"Device: {DEVICE}")

    # ── Training ─────────────────────────────────────────────────────
    epoch_losses = []
    checkpoint_results = []
    t_start = time.time()

    for epoch in range(N_EPOCHS):
        # Sample random IC
        theta0 = rng.uniform(*THETA_RANGE)
        omega0 = rng.uniform(*OMEGA_RANGE)

        # Generate ground truth trajectory
        traj = generate_pendulum(
            duration_ms=TRAJ_MS, dt=DT, b=B, g=GRAV, L=L_PEND,
            theta0=theta0, omega0=omega0,
        )
        traj_t = torch.tensor(traj, device=device, dtype=torch.float32)

        # Train on random segments of the trajectory
        epoch_loss = 0.0
        n_segments = 0

        # Shuffle segment start points
        starts = list(range(0, TRAJ_STEPS - SEGMENT_LEN, SEGMENT_LEN // 2))
        rng.shuffle(starts)

        for seg_start in starts:
            seg_end = seg_start + SEGMENT_LEN
            y0 = traj_t[seg_start]
            target = traj_t[seg_start:seg_end + 1]  # +1 to include endpoint

            # Integrate neural ODE from y0
            pred = integrate(func, y0, (0, SEGMENT_LEN * DT_S), DT_S)

            # MSE loss over segment
            loss = torch.mean((pred - target) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(func.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_segments += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_segments, 1)
        epoch_losses.append(avg_loss)

        if epoch % 25 == 0 or epoch == N_EPOCHS - 1:
            log.info(f"Epoch {epoch:4d}/{N_EPOCHS} | loss={avg_loss:.6f} | "
                     f"lr={scheduler.get_last_lr()[0]:.6f}")

        # Checkpoint
        if epoch % CHECKPOINT_EVERY == 0 or epoch == N_EPOCHS - 1:
            ic_th, ic_om = 1.0, 0.0
            truth = generate_pendulum(
                duration_ms=TEST_MS, dt=DT, b=B, g=GRAV, L=L_PEND,
                theta0=ic_th, omega0=ic_om,
            )[:TEST_STEPS]
            pred = test_neural_ode(func, ic_th, ic_om, TEST_STEPS, device)[:TEST_STEPS]
            ckpt_err = np.mean(np.linalg.norm(truth - pred, axis=1))
            checkpoint_results.append((epoch, ckpt_err))
            log.info(f"  Checkpoint (θ=1.0,ω=0.0) error: {ckpt_err:.4f}")

    elapsed = time.time() - t_start
    log.info(f"Training done in {elapsed:.1f}s")

    # ── Plot: training loss ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(epoch_losses, "b-", alpha=0.5, linewidth=0.5, label="Epoch loss")
    w = max(1, len(epoch_losses) // 50)
    smoothed = np.convolve(epoch_losses, np.ones(w)/w, mode="valid")
    ax.semilogy(np.arange(w-1, w-1+len(smoothed)), smoothed, "b-", linewidth=2, label="Smoothed")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Exp 2.5: Neural ODE Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/training_loss.png", dpi=150)
    plt.close(fig)

    # ── Plot: checkpoint error ───────────────────────────────────────
    ckpt_epochs, ckpt_errs = zip(*checkpoint_results)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ckpt_epochs, ckpt_errs, "go-", markersize=6, linewidth=2, label="Neural ODE")
    ax.axhline(y=3.6, color="orange", linestyle="--", alpha=0.5, label="PA sched. sampling (2.1)")
    ax.axhline(y=1.7, color="blue", linestyle="--", alpha=0.5, label="BPTT (2.0)")
    ax.axhline(y=0.5, color="green", linestyle=":", alpha=0.5, label="Success threshold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Error (θ=1.0, ω=0.0)")
    ax.set_title("Exp 2.5: Neural ODE Generalization During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/checkpoint_error.png", dpi=150)
    plt.close(fig)

    # ── Final generalization test ────────────────────────────────────
    log.info(f"\nFinal generalization test on {len(TEST_ICS)} held-out ICs:")
    summary = []

    for ic_theta, ic_omega, ic_label in TEST_ICS:
        truth = generate_pendulum(
            duration_ms=TEST_MS, dt=DT, b=B, g=GRAV, L=L_PEND,
            theta0=ic_theta, omega0=ic_omega,
        )[:TEST_STEPS]

        pred = test_neural_ode(func, ic_theta, ic_omega, TEST_STEPS, device)[:TEST_STEPS]
        err = np.mean(np.linalg.norm(truth - pred, axis=1))
        summary.append((ic_label, ic_theta, ic_omega, err))
        log.info(f"  {ic_label:20s} θ={ic_theta:+.1f} ω={ic_omega:+.1f} → error={err:.4f}")

        t = np.arange(TEST_STEPS) * DT

        # Time series
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        axes[0].plot(t, truth[:, 0], "k-", lw=1.5, label="Ground truth")
        axes[0].plot(t, pred[:, 0], "r-", lw=1, alpha=0.8, label=f"Neural ODE (err={err:.3f})")
        axes[0].set_ylabel("θ (rad)")
        axes[0].set_title(f"Exp 2.5 Neural ODE: θ₀={ic_theta}, ω₀={ic_omega} ({ic_label})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, truth[:, 1], "k-", lw=1.5, label="Ground truth")
        axes[1].plot(t, pred[:, 1], "r-", lw=1, alpha=0.8, label="Neural ODE")
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
            (axes[1], pred, "Neural ODE", "r"),
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
        E_pred = pendulum_energy(pred[:, 0], pred[:, 1])
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t, E_truth, "k-", lw=1.5, label="Ground truth")
        ax.plot(t, E_pred, "r-", lw=1, alpha=0.8, label="Neural ODE")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Energy")
        ax.set_title(f"Energy — θ₀={ic_theta}, ω₀={ic_omega}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/energy_{ic_label}.png", dpi=150)
        plt.close(fig)

    # ── Summary ──────────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("SUMMARY — All methods comparison")
    log.info(f"{'='*70}")
    log.info(f"{'IC':<20} {'θ₀':>5} {'ω₀':>5} {'NeuODE':>8} {'BPTT':>8} {'PA 2.1':>8} {'PA 07e':>8}")
    log.info("-" * 70)
    bptt_errs = [1.225, 2.581, 0.437, 1.911, 2.381]
    pa21_errs = [3.378, 3.737, 3.412, 3.644, 3.859]
    for i, (label, th, om, err) in enumerate(summary):
        log.info(f"{label:<20} {th:>+5.1f} {om:>+5.1f} {err:>8.4f} {bptt_errs[i]:>8.4f} "
                 f"{pa21_errs[i]:>8.4f}")
    mean_err = np.mean([e for _, _, _, e in summary])
    log.info(f"\nNeural ODE mean: {mean_err:.4f}")
    log.info(f"BPTT mean:       1.707")
    log.info(f"PA sched. samp.: 3.606")
    log.info(f"Success target:  < 0.5")

    # Also test on learned vector field quality
    log.info(f"\n--- Vector field quality check ---")
    # Sample points in state space and compare learned vs true derivatives
    test_points = torch.tensor([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 2.0],
        [-1.0, -1.0], [2.5, 1.0], [0.5, -2.0],
    ], device=device, dtype=torch.float32)

    func.eval()
    with torch.no_grad():
        for pt in test_points:
            theta, omega = pt[0].item(), pt[1].item()
            # True derivatives (in seconds)
            true_dtheta = omega
            true_domega = -B * omega - (GRAV / L_PEND) * np.sin(theta)
            # Learned derivatives
            learned = func(0, pt)
            l_dtheta, l_domega = learned[0].item(), learned[1].item()
            log.info(f"  ({theta:+5.1f}, {omega:+5.1f}) | "
                     f"true=({true_dtheta:+6.3f}, {true_domega:+6.3f}) | "
                     f"learned=({l_dtheta:+6.3f}, {l_domega:+6.3f})")

    np.savez(f"{RESULTS_DIR}/results.npz",
             epoch_losses=np.array(epoch_losses),
             checkpoint_results=np.array(checkpoint_results),
             summary=np.array([(th, om, err) for _, th, om, err in summary]))
    log.info(f"\nAll results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
