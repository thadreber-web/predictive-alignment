"""Experiment 2.4: Neural ODE on the double pendulum (chaotic 4D system).

State: (θ₁, θ₂, ω₁, ω₂)
Equal masses m₁=m₂=1, equal lengths L₁=L₂=1, g=9.81.

Mass matrix form:
  [2, cos(d)] [α₁]   [ω₂²sin(d) - 2(g/L)sin(θ₁)]
  [cos(d), 1] [α₂] = [-ω₁²sin(d) - (g/L)sin(θ₂)]
  where d = θ₁ - θ₂

Tests Neural ODE on: 4D state, chaos, sensitivity to ICs.
"""

import sys
sys.path.insert(0, "/raid/predictive_alignment")

import torch
import torch.nn as nn
import numpy as np
import logging
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time

from src.utils import set_seed

# ── Physics ──────────────────────────────────────────────────────────
G_ACCEL = 9.81
L = 1.0
M = 1.0  # equal masses

# ── Config ───────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DT_S = 0.01          # 10ms integration step
TRAJ_DURATION = 10.0  # seconds
TRAJ_STEPS = int(TRAJ_DURATION / DT_S)

N_EPOCHS = 500
LR = 1e-3
SEGMENT_LEN = 50     # 0.5s segments for BPTT

# IC ranges for training — moderate angles (mix of regular and chaotic)
THETA_RANGE = (-1.5, 1.5)     # ~85 degrees
OMEGA_RANGE = (-2.0, 2.0)

# Held-out test ICs
TEST_ICS = [
    (0.5, 0.5, 0.0, 0.0, "small_angle"),        # regular, in-distribution
    (1.0, -0.5, 0.0, 0.0, "medium_angle"),       # moderate
    (1.5, 1.5, 0.0, 0.0, "large_angle"),         # near boundary of training
    (2.0, 2.0, 0.0, 0.0, "chaotic_start"),       # outside training, chaotic
    (0.8, 0.3, 1.0, -1.0, "with_velocity"),      # nonzero initial ω
]
TEST_DURATION = 10.0
TEST_STEPS = int(TEST_DURATION / DT_S)

CHECKPOINT_EVERY = 50
RESULTS_DIR = "/raid/predictive_alignment/results/exp2_4_double_pendulum"


def double_pendulum_rhs(t, state):
    """Double pendulum RHS for equal masses/lengths."""
    th1, th2, w1, w2 = state
    d = th1 - th2
    sin_d = np.sin(d)
    cos_d = np.cos(d)
    det = 2.0 - cos_d ** 2

    f1 = w2 ** 2 * sin_d - 2 * (G_ACCEL / L) * np.sin(th1)
    f2 = -w1 ** 2 * sin_d - (G_ACCEL / L) * np.sin(th2)

    alpha1 = (f1 - cos_d * f2) / det
    alpha2 = (2 * f2 - cos_d * f1) / det

    return [w1, w2, alpha1, alpha2]


def generate_dp_trajectory(th1_0, th2_0, w1_0, w2_0, duration, dt):
    t_eval = np.arange(0, duration, dt)
    sol = solve_ivp(double_pendulum_rhs, (0, duration),
                    [th1_0, th2_0, w1_0, w2_0],
                    t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-10)
    return sol.y.T  # (n_steps, 4)


def dp_energy(th1, th2, w1, w2):
    """Total energy of double pendulum (m=1, L=1)."""
    # KE = ½m(2L²ω₁² + L²ω₂² + 2L²ω₁ω₂cos(θ₁-θ₂))
    # PE = -mg(2Lcos(θ₁) + Lcos(θ₂))
    KE = 0.5 * (2 * w1**2 + w2**2 + 2 * w1 * w2 * np.cos(th1 - th2))
    PE = -G_ACCEL * (2 * np.cos(th1) + np.cos(th2))
    return KE + PE


class DoublePendulumODEFunc(nn.Module):
    """MLP that learns dstate/dt for 4D double pendulum."""

    def __init__(self, hidden=128, n_layers=3):
        super().__init__()
        layers = [nn.Linear(4, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, t, state):
        return self.net(state)


def rk4_step(func, t, y, dt):
    k1 = func(t, y)
    k2 = func(t + dt/2, y + dt/2 * k1)
    k3 = func(t + dt/2, y + dt/2 * k2)
    k4 = func(t + dt, y + dt * k3)
    return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def integrate(func, y0, n_steps, dt):
    ys = [y0]
    y = y0
    t = 0.0
    for _ in range(n_steps):
        y = rk4_step(func, t, y, dt)
        ys.append(y)
        t += dt
    return torch.stack(ys)


def test_neural_ode(func, ic, n_steps, dt, device):
    func.eval()
    with torch.no_grad():
        y0 = torch.tensor(ic, device=device, dtype=torch.float32)
        traj = integrate(func, y0, n_steps, dt)
    func.train()
    return traj.cpu().numpy()


def lyapunov_divergence(traj1, traj2, dt):
    """Compute trajectory divergence over time for Lyapunov-like analysis."""
    diff = np.linalg.norm(traj1 - traj2, axis=1)
    diff = np.maximum(diff, 1e-15)  # avoid log(0)
    return diff


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
    log = logging.getLogger("exp2.4")

    set_seed(SEED)
    rng = np.random.RandomState(SEED)
    device = torch.device(DEVICE)

    func = DoublePendulumODEFunc(hidden=128, n_layers=3).to(device)
    optimizer = torch.optim.Adam(func.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    n_params = sum(p.numel() for p in func.parameters())
    log.info(f"Exp 2.4: Neural ODE on Double Pendulum")
    log.info(f"m={M}, L={L}, g={G_ACCEL}")
    log.info(f"MLP: 4→128→128→128→4, {n_params:,} params")
    log.info(f"Epochs={N_EPOCHS}, segment_len={SEGMENT_LEN}, dt={DT_S}s")
    log.info(f"Trajectory: {TRAJ_DURATION}s = {TRAJ_STEPS} steps")
    log.info(f"Device: {DEVICE}")

    # ── Training ─────────────────────────────────────────────────────
    epoch_losses = []
    checkpoint_results = []
    t_start = time.time()

    for epoch in range(N_EPOCHS):
        th1_0 = rng.uniform(*THETA_RANGE)
        th2_0 = rng.uniform(*THETA_RANGE)
        w1_0 = rng.uniform(*OMEGA_RANGE)
        w2_0 = rng.uniform(*OMEGA_RANGE)

        traj = generate_dp_trajectory(th1_0, th2_0, w1_0, w2_0, TRAJ_DURATION, DT_S)
        traj_t = torch.tensor(traj, device=device, dtype=torch.float32)

        epoch_loss = 0.0
        n_segments = 0

        starts = list(range(0, len(traj) - SEGMENT_LEN, SEGMENT_LEN // 2))
        rng.shuffle(starts)

        for seg_start in starts:
            y0_seg = traj_t[seg_start]
            target = traj_t[seg_start:seg_start + SEGMENT_LEN + 1]

            pred = integrate(func, y0_seg, SEGMENT_LEN, DT_S)

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

        if epoch % CHECKPOINT_EVERY == 0 or epoch == N_EPOCHS - 1:
            # Checkpoint on small_angle IC
            truth = generate_dp_trajectory(0.5, 0.5, 0.0, 0.0, TEST_DURATION, DT_S)[:TEST_STEPS]
            pred_np = test_neural_ode(func, [0.5, 0.5, 0.0, 0.0], TEST_STEPS, DT_S, device)[:TEST_STEPS]
            ckpt_err = np.mean(np.linalg.norm(truth - pred_np, axis=1))
            checkpoint_results.append((epoch, ckpt_err))
            log.info(f"  Checkpoint (0.5,0.5,0,0) error: {ckpt_err:.4f}")

    elapsed = time.time() - t_start
    log.info(f"Training done in {elapsed:.1f}s")

    # ── Training loss plot ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(epoch_losses, "b-", alpha=0.5, linewidth=0.5, label="Epoch loss")
    w = max(1, len(epoch_losses) // 50)
    smoothed = np.convolve(epoch_losses, np.ones(w)/w, mode="valid")
    ax.semilogy(np.arange(w-1, w-1+len(smoothed)), smoothed, "b-", linewidth=2, label="Smoothed")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log)")
    ax.set_title("Exp 2.4: Double Pendulum Neural ODE Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/training_loss.png", dpi=150)
    plt.close(fig)

    # ── Checkpoint plot ──────────────────────────────────────────────
    ckpt_epochs, ckpt_errs = zip(*checkpoint_results)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ckpt_epochs, ckpt_errs, "go-", markersize=6, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Error (small_angle IC)")
    ax.set_title("Exp 2.4: Double Pendulum Generalization During Training")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/checkpoint_error.png", dpi=150)
    plt.close(fig)

    # ── Final generalization test ────────────────────────────────────
    log.info(f"\nFinal test on {len(TEST_ICS)} held-out ICs:")
    summary = []

    # Also compute short-horizon errors (first 2s) vs full horizon
    SHORT_STEPS = int(2.0 / DT_S)

    for th1, th2, w1, w2, ic_label in TEST_ICS:
        truth = generate_dp_trajectory(th1, th2, w1, w2, TEST_DURATION, DT_S)[:TEST_STEPS]
        pred = test_neural_ode(func, [th1, th2, w1, w2], TEST_STEPS, DT_S, device)[:TEST_STEPS]

        err_full = np.mean(np.linalg.norm(truth - pred, axis=1))
        err_short = np.mean(np.linalg.norm(truth[:SHORT_STEPS] - pred[:SHORT_STEPS], axis=1))

        summary.append((ic_label, th1, th2, w1, w2, err_short, err_full))
        log.info(f"  {ic_label:20s} ({th1:+.1f},{th2:+.1f},{w1:+.1f},{w2:+.1f}) | "
                 f"err_2s={err_short:.4f} err_10s={err_full:.4f}")

        t = np.arange(TEST_STEPS) * DT_S

        # Time series — all 4 state variables
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        labels = ["θ₁", "θ₂", "ω₁", "ω₂"]
        for i, (ax, lbl) in enumerate(zip(axes, labels)):
            ax.plot(t, truth[:, i], "k-", lw=1.5, label="Ground truth")
            ax.plot(t, pred[:, i], "r-", lw=1, alpha=0.8, label="Neural ODE")
            ax.set_ylabel(lbl)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[0].set_title(f"Exp 2.4: Double Pendulum ({ic_label}) — err_2s={err_short:.3f}, err_10s={err_full:.3f}")
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/test_{ic_label}.png", dpi=150)
        plt.close(fig)

        # Phase portraits: θ₁ vs θ₂, ω₁ vs ω₂
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for col, (data, title, color) in enumerate([
            (truth, "Ground truth", "k"), (pred, "Neural ODE", "r")
        ]):
            axes[0, col].plot(data[:, 0], data[:, 1], color=color, lw=0.5, alpha=0.7)
            axes[0, col].plot(data[0, 0], data[0, 1], "o", color=color, ms=6)
            axes[0, col].set_xlabel("θ₁")
            axes[0, col].set_ylabel("θ₂")
            axes[0, col].set_title(f"{title} — angles")
            axes[0, col].grid(True, alpha=0.3)

            axes[1, col].plot(data[:, 2], data[:, 3], color=color, lw=0.5, alpha=0.7)
            axes[1, col].plot(data[0, 2], data[0, 3], "o", color=color, ms=6)
            axes[1, col].set_xlabel("ω₁")
            axes[1, col].set_ylabel("ω₂")
            axes[1, col].set_title(f"{title} — velocities")
            axes[1, col].grid(True, alpha=0.3)

        fig.suptitle(f"Phase — {ic_label}", fontsize=13)
        plt.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/phase_{ic_label}.png", dpi=150)
        plt.close(fig)

        # Energy conservation
        E_truth = dp_energy(truth[:, 0], truth[:, 1], truth[:, 2], truth[:, 3])
        E_pred = dp_energy(pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3])
        E_drift = np.abs(E_pred[-1] - E_pred[0])

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t, E_truth, "k-", lw=1.5, label="Ground truth")
        ax.plot(t, E_pred, "r-", lw=1, alpha=0.8, label=f"Neural ODE (drift={E_drift:.4f})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Total Energy")
        ax.set_title(f"Energy — {ic_label}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/energy_{ic_label}.png", dpi=150)
        plt.close(fig)
        log.info(f"    Energy drift: {E_drift:.6f}")

        # Trajectory divergence (Lyapunov-like)
        div = lyapunov_divergence(truth, pred, DT_S)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.semilogy(t, div, "r-", lw=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("||truth - pred|| (log)")
        ax.set_title(f"Trajectory Divergence — {ic_label}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/divergence_{ic_label}.png", dpi=150)
        plt.close(fig)

    # ── Vector field check ───────────────────────────────────────────
    log.info(f"\n--- Vector field quality check ---")
    test_pts = [
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [1.0, -0.5, 1.0, -1.0],
        [0.0, 0.0, 1.0, 1.0],
    ]
    func.eval()
    with torch.no_grad():
        for pt in test_pts:
            true_deriv = double_pendulum_rhs(0, pt)
            pt_t = torch.tensor(pt, device=device, dtype=torch.float32)
            learned = func(0, pt_t).cpu().numpy()
            log.info(f"  ({pt[0]:+.1f},{pt[1]:+.1f},{pt[2]:+.1f},{pt[3]:+.1f}) | "
                     f"true=({true_deriv[0]:+6.3f},{true_deriv[1]:+6.3f},"
                     f"{true_deriv[2]:+6.3f},{true_deriv[3]:+6.3f}) | "
                     f"learned=({learned[0]:+6.3f},{learned[1]:+6.3f},"
                     f"{learned[2]:+6.3f},{learned[3]:+6.3f})")

    # ── Summary ──────────────────────────────────────────────────────
    log.info(f"\n{'='*80}")
    log.info("SUMMARY")
    log.info(f"{'='*80}")
    log.info(f"{'IC':<20} {'θ₁':>5} {'θ₂':>5} {'ω₁':>5} {'ω₂':>5} {'Err 2s':>8} {'Err 10s':>8}")
    log.info("-" * 70)
    for label, th1, th2, w1, w2, e_short, e_full in summary:
        log.info(f"{label:<20} {th1:>+5.1f} {th2:>+5.1f} {w1:>+5.1f} {w2:>+5.1f} "
                 f"{e_short:>8.4f} {e_full:>8.4f}")

    mean_short = np.mean([e for _, _, _, _, _, e, _ in summary])
    mean_full = np.mean([e for _, _, _, _, _, _, e in summary])
    log.info(f"\nMean error (2s):  {mean_short:.4f}")
    log.info(f"Mean error (10s): {mean_full:.4f}")
    log.info(f"Pendulum Neural ODE (2.5): 0.085")
    log.info(f"Lotka-Volterra Neural ODE (2.3): 6.52")

    np.savez(f"{RESULTS_DIR}/results.npz",
             epoch_losses=np.array(epoch_losses),
             checkpoint_results=np.array(checkpoint_results),
             summary=np.array([(th1, th2, w1, w2, es, ef)
                               for _, th1, th2, w1, w2, es, ef in summary]))
    log.info(f"\nAll results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
