"""Experiment 2.3: Neural ODE on Lotka-Volterra (predator-prey).

Conservative 2D system with multiplicative coupling and closed orbits.
Tests whether Neural ODE generalizes beyond dissipative pendulum dynamics.

dx/dt = αx - βxy      (prey)
dy/dt = δxy - γy       (predators)
α=1.0, β=0.1, δ=0.075, γ=1.5
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

# ── Lotka-Volterra parameters ────────────────────────────────────────
ALPHA = 1.0
BETA = 0.1
DELTA = 0.075
GAMMA = 1.5

# Fixed point
X_STAR = GAMMA / DELTA   # 20.0
Y_STAR = ALPHA / BETA    # 10.0

# ── Config ───────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DT_S = 0.01          # integration timestep in seconds
TRAJ_DURATION = 30.0  # seconds per trajectory
TRAJ_STEPS = int(TRAJ_DURATION / DT_S)

N_EPOCHS = 500
LR = 1e-3
SEGMENT_LEN = 100     # steps per BPTT segment (1s of simulation)

# IC ranges for training
X_RANGE = (2.0, 20.0)
Y_RANGE = (2.0, 20.0)

# Held-out test ICs
TEST_ICS = [
    (15.0, 8.0, "near_fp"),        # near fixed point (20, 10)
    (5.0, 5.0, "small_orbit"),     # small populations
    (25.0, 15.0, "large_orbit"),   # large populations, outside training x range
    (10.0, 3.0, "low_predator"),   # low predator
    (3.0, 18.0, "high_predator"),  # high predator, low prey
]
TEST_DURATION = 30.0
TEST_STEPS = int(TEST_DURATION / DT_S)

CHECKPOINT_EVERY = 50
RESULTS_DIR = "/raid/predictive_alignment/results/exp2_3_lotka_volterra"


def lotka_volterra_rhs(t, state):
    """True Lotka-Volterra RHS."""
    x, y = state
    dxdt = ALPHA * x - BETA * x * y
    dydt = DELTA * x * y - GAMMA * y
    return [dxdt, dydt]


def generate_lv_trajectory(x0, y0, duration, dt):
    """Generate Lotka-Volterra trajectory with solve_ivp."""
    t_eval = np.arange(0, duration, dt)
    sol = solve_ivp(lotka_volterra_rhs, (0, duration), [x0, y0],
                    t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-10)
    return sol.y.T  # (n_steps, 2)


def conservation_quantity(x, y):
    """Lotka-Volterra conserved quantity V = δx - γln(x) + βy - αln(y)."""
    # Clamp to avoid log(0)
    x_safe = np.maximum(x, 1e-10)
    y_safe = np.maximum(y, 1e-10)
    return DELTA * x_safe - GAMMA * np.log(x_safe) + BETA * y_safe - ALPHA * np.log(y_safe)


class LVODEFunc(nn.Module):
    """MLP that learns dstate/dt = f(state) for Lotka-Volterra."""

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


def test_neural_ode(func, x0, y0, n_steps, dt, device):
    func.eval()
    with torch.no_grad():
        y0_t = torch.tensor([x0, y0], device=device, dtype=torch.float32)
        traj = integrate(func, y0_t, n_steps, dt)
    func.train()
    return traj.cpu().numpy()


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
    log = logging.getLogger("exp2.3")

    set_seed(SEED)
    rng = np.random.RandomState(SEED)
    device = torch.device(DEVICE)

    func = LVODEFunc(hidden=64, n_layers=3).to(device)
    optimizer = torch.optim.Adam(func.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    n_params = sum(p.numel() for p in func.parameters())
    log.info(f"Exp 2.3: Neural ODE on Lotka-Volterra")
    log.info(f"α={ALPHA}, β={BETA}, δ={DELTA}, γ={GAMMA}")
    log.info(f"Fixed point: x*={X_STAR:.1f}, y*={Y_STAR:.1f}")
    log.info(f"MLP: 2→64→64→64→2, {n_params:,} params")
    log.info(f"Epochs={N_EPOCHS}, segment_len={SEGMENT_LEN}, dt={DT_S}s")
    log.info(f"Trajectory: {TRAJ_DURATION}s = {TRAJ_STEPS} steps")

    # ── Training ─────────────────────────────────────────────────────
    epoch_losses = []
    checkpoint_results = []
    t_start = time.time()

    for epoch in range(N_EPOCHS):
        x0 = rng.uniform(*X_RANGE)
        y0 = rng.uniform(*Y_RANGE)

        traj = generate_lv_trajectory(x0, y0, TRAJ_DURATION, DT_S)
        traj_t = torch.tensor(traj, device=device, dtype=torch.float32)

        epoch_loss = 0.0
        n_segments = 0

        starts = list(range(0, len(traj) - SEGMENT_LEN, SEGMENT_LEN // 2))
        rng.shuffle(starts)

        for seg_start in starts:
            seg_end = seg_start + SEGMENT_LEN
            y0_seg = traj_t[seg_start]
            target = traj_t[seg_start:seg_end + 1]

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
            truth = generate_lv_trajectory(15.0, 8.0, TEST_DURATION, DT_S)[:TEST_STEPS]
            pred_np = test_neural_ode(func, 15.0, 8.0, TEST_STEPS, DT_S, device)[:TEST_STEPS]
            ckpt_err = np.mean(np.linalg.norm(truth - pred_np, axis=1))
            checkpoint_results.append((epoch, ckpt_err))
            log.info(f"  Checkpoint (x=15,y=8) error: {ckpt_err:.4f}")

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
    ax.set_title("Exp 2.3: Lotka-Volterra Neural ODE Training Loss")
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
    ax.set_ylabel("Test Error (x=15, y=8)")
    ax.set_title("Exp 2.3: Lotka-Volterra Generalization During Training")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/checkpoint_error.png", dpi=150)
    plt.close(fig)

    # ── Final generalization test ────────────────────────────────────
    log.info(f"\nFinal test on {len(TEST_ICS)} held-out ICs:")
    summary = []
    all_truths = []
    all_preds = []

    for x0, y0, ic_label in TEST_ICS:
        truth = generate_lv_trajectory(x0, y0, TEST_DURATION, DT_S)[:TEST_STEPS]
        pred = test_neural_ode(func, x0, y0, TEST_STEPS, DT_S, device)[:TEST_STEPS]

        err = np.mean(np.linalg.norm(truth - pred, axis=1))
        summary.append((ic_label, x0, y0, err))
        all_truths.append(truth)
        all_preds.append(pred)
        log.info(f"  {ic_label:20s} x={x0:5.1f} y={y0:5.1f} → error={err:.4f}")

        # Check for negative populations
        neg_frac = np.mean((pred < 0).any(axis=1))
        if neg_frac > 0:
            log.info(f"    WARNING: {neg_frac*100:.1f}% of timesteps have negative populations")

        t = np.arange(TEST_STEPS) * DT_S

        # Time series
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        axes[0].plot(t, truth[:, 0], "k-", lw=1.5, label="Ground truth")
        axes[0].plot(t, pred[:, 0], "r-", lw=1, alpha=0.8, label=f"Neural ODE (err={err:.3f})")
        axes[0].set_ylabel("x (prey)")
        axes[0].set_title(f"Exp 2.3 Lotka-Volterra: x₀={x0}, y₀={y0} ({ic_label})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, truth[:, 1], "k-", lw=1.5, label="Ground truth")
        axes[1].plot(t, pred[:, 1], "r-", lw=1, alpha=0.8, label="Neural ODE")
        axes[1].set_ylabel("y (predators)")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/test_{ic_label}.png", dpi=150)
        plt.close(fig)

        # Phase portrait (individual)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, data, label, color in [
            (axes[0], truth, "Ground truth", "k"),
            (axes[1], pred, "Neural ODE", "r"),
        ]:
            ax.plot(data[:, 0], data[:, 1], color=color, lw=0.8, alpha=0.8)
            ax.plot(data[0, 0], data[0, 1], "o", color=color, ms=8)
            ax.set_xlabel("x (prey)")
            ax.set_ylabel("y (predators)")
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Phase — x₀={x0}, y₀={y0}", fontsize=13)
        plt.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/phase_{ic_label}.png", dpi=150)
        plt.close(fig)

        # Conservation quantity
        V_truth = conservation_quantity(truth[:, 0], truth[:, 1])
        V_pred = conservation_quantity(pred[:, 0], pred[:, 1])
        V_drift = np.abs(V_pred[-1] - V_pred[0])

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t, V_truth, "k-", lw=1.5, label="Ground truth")
        ax.plot(t, V_pred, "r-", lw=1, alpha=0.8, label=f"Neural ODE (drift={V_drift:.4f})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("V (conserved quantity)")
        ax.set_title(f"Conservation — x₀={x0}, y₀={y0}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/conservation_{ic_label}.png", dpi=150)
        plt.close(fig)
        log.info(f"    Conservation drift: {V_drift:.6f}")

    # ── Multi-orbit phase portrait overlay ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for ax, dataset, title in [(axes[0], all_truths, "Ground Truth"), (axes[1], all_preds, "Neural ODE")]:
        for i, (data, (x0, y0, label)) in enumerate(zip(dataset, TEST_ICS)):
            ax.plot(data[:, 0], data[:, 1], color=colors[i], lw=0.8, alpha=0.8, label=label)
            ax.plot(data[0, 0], data[0, 1], "o", color=colors[i], ms=6)
        ax.plot(X_STAR, Y_STAR, "k*", ms=12, label=f"Fixed pt ({X_STAR:.0f},{Y_STAR:.0f})")
        ax.set_xlabel("x (prey)")
        ax.set_ylabel("y (predators)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Lotka-Volterra: Multiple Orbits", fontsize=13)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/phase_overlay.png", dpi=150)
    plt.close(fig)

    # ── Vector field check ───────────────────────────────────────────
    log.info(f"\n--- Vector field quality check ---")
    test_points = torch.tensor([
        [X_STAR, Y_STAR],  # fixed point — should be (0, 0)
        [10.0, 5.0],
        [30.0, 15.0],
        [5.0, 20.0],
        [15.0, 3.0],
    ], device=device, dtype=torch.float32)

    func.eval()
    with torch.no_grad():
        for pt in test_points:
            x, y = pt[0].item(), pt[1].item()
            true_dx = ALPHA * x - BETA * x * y
            true_dy = DELTA * x * y - GAMMA * y
            learned = func(0, pt)
            l_dx, l_dy = learned[0].item(), learned[1].item()
            log.info(f"  ({x:5.1f}, {y:5.1f}) | "
                     f"true=({true_dx:+8.3f}, {true_dy:+8.3f}) | "
                     f"learned=({l_dx:+8.3f}, {l_dy:+8.3f})")

    # ── Summary ──────────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"{'IC':<20} {'x₀':>6} {'y₀':>6} {'Error':>10}")
    log.info("-" * 45)
    for label, x0, y0, err in summary:
        log.info(f"{label:<20} {x0:>6.1f} {y0:>6.1f} {err:>10.4f}")
    mean_err = np.mean([e for _, _, _, e in summary])
    log.info(f"\nMean generalization error: {mean_err:.4f}")
    log.info(f"Pendulum Neural ODE (exp 2.5): 0.085")

    np.savez(f"{RESULTS_DIR}/results.npz",
             epoch_losses=np.array(epoch_losses),
             checkpoint_results=np.array(checkpoint_results),
             summary=np.array([(x0, y0, err) for _, x0, y0, err in summary]))
    log.info(f"\nAll results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
