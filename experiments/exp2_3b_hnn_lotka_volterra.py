"""Experiment 2.3b: Hamiltonian Neural Network for Lotka-Volterra.

Lotka-Volterra is Hamiltonian in log-coordinates:
  u = ln(x), v = ln(y)
  H(u,v) = δ·eᵘ - γ·u + β·eᵛ - α·v
  du/dt = -∂H/∂v = α - β·eᵛ = α - βy
  dv/dt = +∂H/∂u = δ·eᵘ - γ = δx - γ

Training approach (Greydanus et al. 2019): train on derivatives directly.
Compute dx/dt from data via finite differences, then match the HNN's
predicted derivatives (obtained via autograd on the scalar H output).
This avoids expensive backprop through the integrator.

Testing: integrate the learned HNN forward with RK4.
"""

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

import torch
import torch.nn as nn
import numpy as np
import logging
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from src.utils import set_seed

# ── Lotka-Volterra parameters ────────────────────────────────────────
ALPHA = 1.0
BETA = 0.1
DELTA = 0.075
GAMMA = 1.5

X_STAR = GAMMA / DELTA   # 20.0
Y_STAR = ALPHA / BETA    # 10.0

# ── Config ───────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DT_S = 0.01
TRAJ_DURATION = 30.0
TRAJ_STEPS = int(TRAJ_DURATION / DT_S)

N_EPOCHS = 500
LR = 1e-3
BATCH_SIZE = 256      # points per batch for derivative matching

X_RANGE = (2.0, 20.0)
Y_RANGE = (2.0, 20.0)

TEST_ICS = [
    (15.0, 8.0, "near_fp"),
    (5.0, 5.0, "small_orbit"),
    (25.0, 15.0, "large_orbit"),
    (10.0, 3.0, "low_predator"),
    (3.0, 18.0, "high_predator"),
]
TEST_DURATION = 30.0
TEST_STEPS = int(TEST_DURATION / DT_S)

CHECKPOINT_EVERY = 50
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "exp2_3b_hnn_lotka_volterra")


def lotka_volterra_rhs(t, state):
    x, y = state
    dxdt = ALPHA * x - BETA * x * y
    dydt = DELTA * x * y - GAMMA * y
    return [dxdt, dydt]


def generate_lv_trajectory(x0, y0, duration, dt):
    t_eval = np.arange(0, duration, dt)
    sol = solve_ivp(lotka_volterra_rhs, (0, duration), [x0, y0],
                    t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-10)
    return sol.y.T


def conservation_quantity(x, y):
    x_safe = np.maximum(x, 1e-10)
    y_safe = np.maximum(y, 1e-10)
    return DELTA * x_safe - GAMMA * np.log(x_safe) + BETA * y_safe - ALPHA * np.log(y_safe)


def true_hamiltonian(u, v):
    """True H in log-coordinates: H = δeᵘ - γu + βeᵛ - αv."""
    return DELTA * np.exp(u) - GAMMA * u + BETA * np.exp(v) - ALPHA * v


class HamiltonianNet(nn.Module):
    """MLP that outputs a scalar H(u, v) where u=ln(x), v=ln(y)."""

    def __init__(self, hidden=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, uv):
        """Compute H(u, v). Input shape: (batch, 2) or (2,)."""
        return self.net(uv)

    def derivatives_batch(self, xy_batch):
        """Compute dx/dt, dy/dt for a batch of (x, y) points.

        Input: (batch, 2) tensor of [x, y] values.
        Output: (batch, 2) tensor of [dx/dt, dy/dt].
        """
        xy_batch = xy_batch.detach().requires_grad_(True)
        x = xy_batch[:, 0]
        y = xy_batch[:, 1]

        # Log-coordinates
        u = torch.log(torch.clamp(x, min=1e-6))
        v = torch.log(torch.clamp(y, min=1e-6))
        uv = torch.stack([u, v], dim=1)  # (batch, 2)
        uv.requires_grad_(True)

        H = self.forward(uv)  # (batch, 1)

        # Batch autograd: ∂H/∂uv for each sample
        dH = torch.autograd.grad(H.sum(), uv, create_graph=True)[0]  # (batch, 2)
        dH_du = dH[:, 0]
        dH_dv = dH[:, 1]

        # Symplectic: du/dt = -∂H/∂v, dv/dt = ∂H/∂u
        # Transform: dx/dt = x·du/dt, dy/dt = y·dv/dt
        dx_dt = x * (-dH_dv)
        dy_dt = y * dH_du

        return torch.stack([dx_dt, dy_dt], dim=1)

    def time_derivative_single(self, t, xy):
        """Single-point derivative for RK4 integration at test time."""
        xy_b = xy.unsqueeze(0)  # (1, 2)
        xy_b = xy_b.detach().requires_grad_(True)
        x = xy_b[:, 0]
        y = xy_b[:, 1]

        u = torch.log(torch.clamp(x, min=1e-6))
        v = torch.log(torch.clamp(y, min=1e-6))
        uv = torch.stack([u, v], dim=1)
        uv.requires_grad_(True)

        H = self.forward(uv)
        dH = torch.autograd.grad(H.sum(), uv, create_graph=False)[0]
        dH_du = dH[0, 0]
        dH_dv = dH[0, 1]

        dx_dt = x[0] * (-dH_dv)
        dy_dt = y[0] * dH_du

        return torch.stack([dx_dt, dy_dt])


def rk4_step(func, t, y, dt):
    k1 = func(t, y)
    k2 = func(t + dt/2, y + dt/2 * k1)
    k3 = func(t + dt/2, y + dt/2 * k2)
    k4 = func(t + dt, y + dt * k3)
    return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def test_hnn(hnn, x0, y0, n_steps, dt, device):
    """Integrate HNN forward for testing."""
    hnn.eval()
    y = torch.tensor([x0, y0], device=device, dtype=torch.float32)
    ys = [y.cpu().numpy()]
    t = 0.0
    with torch.no_grad():
        for _ in range(n_steps):
            with torch.enable_grad():
                y_next = rk4_step(hnn.time_derivative_single, t, y, dt)
            y = y_next.detach()
            ys.append(y.cpu().numpy())
            t += dt
    hnn.train()
    return np.array(ys)


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
    log = logging.getLogger("exp2.3b")

    set_seed(SEED)
    rng = np.random.RandomState(SEED)
    device = torch.device(DEVICE)

    hnn = HamiltonianNet(hidden=64, n_layers=3).to(device)
    optimizer = torch.optim.Adam(hnn.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    n_params = sum(p.numel() for p in hnn.parameters())
    log.info(f"Exp 2.3b: HNN on Lotka-Volterra (derivative matching)")
    log.info(f"α={ALPHA}, β={BETA}, δ={DELTA}, γ={GAMMA}")
    log.info(f"Fixed point: x*={X_STAR:.1f}, y*={Y_STAR:.1f}")
    log.info(f"HNN MLP: 2→64→64→64→1 (scalar H), {n_params:,} params")
    log.info(f"Epochs={N_EPOCHS}, batch_size={BATCH_SIZE}, dt={DT_S}s")
    log.info(f"Device: {DEVICE}")

    # ── Training: derivative matching ─────────────────────────────────
    # Instead of integrating through RK4 (slow with autograd), we:
    # 1. Sample (x,y) points from trajectories
    # 2. Compute true dx/dt, dy/dt via finite differences
    # 3. Train HNN's predicted derivatives to match
    epoch_losses = []
    checkpoint_results = []
    t_start = time.time()

    for epoch in range(N_EPOCHS):
        # Generate a trajectory from random IC
        x0 = rng.uniform(*X_RANGE)
        y0 = rng.uniform(*Y_RANGE)
        traj = generate_lv_trajectory(x0, y0, TRAJ_DURATION, DT_S)

        # Compute finite-difference derivatives
        # dx/dt ≈ (x[t+1] - x[t-1]) / (2*dt) — central differences
        derivs = np.zeros_like(traj[1:-1])
        derivs[:, 0] = (traj[2:, 0] - traj[:-2, 0]) / (2 * DT_S)
        derivs[:, 1] = (traj[2:, 1] - traj[:-2, 1]) / (2 * DT_S)
        points = traj[1:-1]  # corresponding state points

        # Sample random batch
        n_pts = len(points)
        indices = rng.choice(n_pts, size=min(BATCH_SIZE, n_pts), replace=False)

        xy_batch = torch.tensor(points[indices], device=device, dtype=torch.float32)
        deriv_target = torch.tensor(derivs[indices], device=device, dtype=torch.float32)

        # HNN predicted derivatives
        deriv_pred = hnn.derivatives_batch(xy_batch)

        loss = torch.mean((deriv_pred - deriv_target) ** 2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hnn.parameters(), 1.0)
        optimizer.step()

        scheduler.step()
        epoch_losses.append(loss.item())

        if epoch % 25 == 0 or epoch == N_EPOCHS - 1:
            log.info(f"Epoch {epoch:4d}/{N_EPOCHS} | loss={loss.item():.6f} | "
                     f"lr={scheduler.get_last_lr()[0]:.6f}")

        if epoch % CHECKPOINT_EVERY == 0 or epoch == N_EPOCHS - 1:
            truth = generate_lv_trajectory(15.0, 8.0, TEST_DURATION, DT_S)[:TEST_STEPS]
            pred_np = test_hnn(hnn, 15.0, 8.0, TEST_STEPS, DT_S, device)[:TEST_STEPS]
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
    ax.set_ylabel("Derivative MSE Loss (log)")
    ax.set_title("Exp 2.3b: HNN Lotka-Volterra Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/training_loss.png", dpi=150)
    plt.close(fig)

    # ── Checkpoint plot ──────────────────────────────────────────────
    ckpt_epochs, ckpt_errs = zip(*checkpoint_results)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ckpt_epochs, ckpt_errs, "go-", markersize=6, linewidth=2, label="HNN")
    ax.axhline(y=6.52, color="orange", linestyle="--", alpha=0.5, label="Vanilla NODE (2.3)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Error (x=15, y=8)")
    ax.set_title("Exp 2.3b: HNN Lotka-Volterra Generalization During Training")
    ax.legend()
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
        pred = test_hnn(hnn, x0, y0, TEST_STEPS, DT_S, device)[:TEST_STEPS]

        err = np.mean(np.linalg.norm(truth - pred, axis=1))
        summary.append((ic_label, x0, y0, err))
        all_truths.append(truth)
        all_preds.append(pred)
        log.info(f"  {ic_label:20s} x={x0:5.1f} y={y0:5.1f} → error={err:.4f}")

        t = np.arange(TEST_STEPS) * DT_S

        # Time series
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        axes[0].plot(t, truth[:, 0], "k-", lw=1.5, label="Ground truth")
        axes[0].plot(t, pred[:, 0], "r-", lw=1, alpha=0.8, label=f"HNN (err={err:.3f})")
        axes[0].set_ylabel("x (prey)")
        axes[0].set_title(f"Exp 2.3b HNN: x₀={x0}, y₀={y0} ({ic_label})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, truth[:, 1], "k-", lw=1.5, label="Ground truth")
        axes[1].plot(t, pred[:, 1], "r-", lw=1, alpha=0.8, label="HNN")
        axes[1].set_ylabel("y (predators)")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/test_{ic_label}.png", dpi=150)
        plt.close(fig)

        # Phase portrait
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, data, label, color in [
            (axes[0], truth, "Ground truth", "k"),
            (axes[1], pred, "HNN", "r"),
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

        # Conservation quantity — the key metric
        V_truth = conservation_quantity(truth[:, 0], truth[:, 1])
        V_pred = conservation_quantity(pred[:, 0], pred[:, 1])
        V_drift = np.abs(V_pred[-1] - V_pred[0])
        V_max_dev = np.max(np.abs(V_pred - V_pred[0]))

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t, V_truth, "k-", lw=1.5, label="Ground truth")
        ax.plot(t, V_pred, "r-", lw=1, alpha=0.8,
                label=f"HNN (drift={V_drift:.6f}, max_dev={V_max_dev:.6f})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("V (conserved quantity)")
        ax.set_title(f"Conservation — x₀={x0}, y₀={y0}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/conservation_{ic_label}.png", dpi=150)
        plt.close(fig)
        log.info(f"    Conservation drift: {V_drift:.6f}, max deviation: {V_max_dev:.6f}")

    # ── Multi-orbit phase portrait ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for ax, dataset, title in [(axes[0], all_truths, "Ground Truth"),
                                (axes[1], all_preds, "HNN")]:
        for i, (data, (x0, y0, label)) in enumerate(zip(dataset, TEST_ICS)):
            ax.plot(data[:, 0], data[:, 1], color=colors[i], lw=0.8, alpha=0.8, label=label)
            ax.plot(data[0, 0], data[0, 1], "o", color=colors[i], ms=6)
        ax.plot(X_STAR, Y_STAR, "k*", ms=12, label=f"Fixed pt ({X_STAR:.0f},{Y_STAR:.0f})")
        ax.set_xlabel("x (prey)")
        ax.set_ylabel("y (predators)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Lotka-Volterra HNN: Multiple Orbits", fontsize=13)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/phase_overlay.png", dpi=150)
    plt.close(fig)

    # ── Learned Hamiltonian landscape ────────────────────────────────
    log.info(f"\n--- Learned vs True Hamiltonian ---")
    hnn.eval()
    xg = np.linspace(1, 40, 80)
    yg = np.linspace(1, 25, 60)
    X, Y = np.meshgrid(xg, yg)
    U_grid = np.log(X)
    V_grid = np.log(Y)

    H_true = true_hamiltonian(U_grid, V_grid)

    uv_flat = torch.tensor(
        np.stack([U_grid.ravel(), V_grid.ravel()], axis=1),
        device=device, dtype=torch.float32
    )
    with torch.no_grad():
        H_learned_flat = hnn(uv_flat).cpu().numpy().ravel()
    H_learned = H_learned_flat.reshape(X.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, H, title in [(axes[0], H_true, "True H(u,v)"),
                          (axes[1], H_learned, "Learned H(u,v)")]:
        cs = ax.contour(X, Y, H, levels=20, cmap="viridis")
        ax.clabel(cs, inline=True, fontsize=7)
        ax.plot(X_STAR, Y_STAR, "r*", ms=12)
        ax.set_xlabel("x (prey)")
        ax.set_ylabel("y (predators)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Hamiltonian Contours (should match up to constant shift)", fontsize=12)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/hamiltonian_landscape.png", dpi=150)
    plt.close(fig)

    # ── Summary ──────────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("SUMMARY — HNN vs vanilla Neural ODE (exp 2.3)")
    log.info(f"{'='*70}")
    node_errs = [1.16, 8.14, 0.92, 4.78, 17.61]
    node_drifts = [0.028, 0.541, 0.026, 0.306, 1.145]
    log.info(f"{'IC':<20} {'x₀':>6} {'y₀':>6} {'HNN err':>9} {'NODE err':>9} {'HNN drift':>10} {'NODE drift':>11}")
    log.info("-" * 80)
    for i, (label, x0, y0, err) in enumerate(summary):
        pred_i = all_preds[i]
        V_p = conservation_quantity(pred_i[:, 0], pred_i[:, 1])
        drift_i = np.abs(V_p[-1] - V_p[0])
        log.info(f"{label:<20} {x0:>6.1f} {y0:>6.1f} {err:>9.4f} {node_errs[i]:>9.4f} "
                 f"{drift_i:>10.6f} {node_drifts[i]:>11.6f}")

    mean_err = np.mean([e for _, _, _, e in summary])
    log.info(f"\nHNN mean error:  {mean_err:.4f}")
    log.info(f"NODE mean error: 6.5212")

    np.savez(f"{RESULTS_DIR}/results.npz",
             epoch_losses=np.array(epoch_losses),
             checkpoint_results=np.array(checkpoint_results),
             summary=np.array([(x0, y0, err) for _, x0, y0, err in summary]))
    log.info(f"\nAll results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
