"""Experiment 2.3c: HNN with trajectory-level training for Lotka-Volterra.

Same HNN architecture as 2.3b (scalar H, symplectic structure in log-coords),
but trained with backprop through short RK4 integration segments — the same
approach that made the vanilla Neural ODE accurate (exp 2.3).

This combines:
- HNN structure → conservation guarantee
- Trajectory-level loss → tight derivative convergence

Key optimization: short segments (10 steps = 0.1s) keep autograd memory
reasonable. Fewer segments per epoch (20) keep runtime ~5-10s/epoch.
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
SEGMENT_LEN = 10       # 0.1s — short for fast autograd through RK4
SEGS_PER_EPOCH = 30    # subsample segments to control runtime

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
RESULTS_DIR = "/raid/predictive_alignment/results/exp2_3c_hnn_trajectory"


def lotka_volterra_rhs(t, state):
    x, y = state
    return [ALPHA * x - BETA * x * y, DELTA * x * y - GAMMA * y]


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
    return DELTA * np.exp(u) - GAMMA * u + BETA * np.exp(v) - ALPHA * v


class HamiltonianNet(nn.Module):
    """MLP: (u,v) → scalar H. Dynamics via symplectic autograd."""

    def __init__(self, hidden=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, uv):
        return self.net(uv)

    def time_derivative(self, t, xy, create_graph=True):
        """Compute dx/dt, dy/dt from Hamiltonian structure.

        create_graph=True for training (backprop through integrator).
        create_graph=False for testing (no graph needed).
        """
        x = xy[0]
        y = xy[1]

        u = torch.log(torch.clamp(x, min=1e-6))
        v = torch.log(torch.clamp(y, min=1e-6))

        uv = torch.stack([u, v])
        uv = uv.detach().requires_grad_(True)

        H = self.forward(uv)

        dH = torch.autograd.grad(H, uv, create_graph=create_graph)[0]

        # Symplectic: du/dt = -∂H/∂v, dv/dt = ∂H/∂u
        # Back to (x,y): dx/dt = x·du/dt, dy/dt = y·dv/dt
        dx_dt = x * (-dH[1])
        dy_dt = y * dH[0]

        return torch.stack([dx_dt, dy_dt])


def rk4_step(func, t, y, dt, **kwargs):
    k1 = func(t, y, **kwargs)
    k2 = func(t + dt/2, y + dt/2 * k1, **kwargs)
    k3 = func(t + dt/2, y + dt/2 * k2, **kwargs)
    k4 = func(t + dt, y + dt * k3, **kwargs)
    return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_hnn(hnn, y0, n_steps, dt, create_graph=True):
    """Integrate with RK4. create_graph controls autograd graph building."""
    ys = [y0]
    y = y0
    t = 0.0
    for _ in range(n_steps):
        y = rk4_step(hnn.time_derivative, t, y, dt, create_graph=create_graph)
        ys.append(y)
        t += dt
    return torch.stack(ys)


def test_hnn(hnn, x0, y0, n_steps, dt, device):
    """Integrate HNN for testing — no graph, detach each step for memory."""
    hnn.eval()
    y = torch.tensor([x0, y0], device=device, dtype=torch.float32)
    ys = [y.cpu().numpy()]
    t = 0.0
    for _ in range(n_steps):
        with torch.enable_grad():
            y_next = rk4_step(hnn.time_derivative, t, y, dt, create_graph=False)
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
    log = logging.getLogger("exp2.3c")

    set_seed(SEED)
    rng = np.random.RandomState(SEED)
    device = torch.device(DEVICE)

    hnn = HamiltonianNet(hidden=64, n_layers=3).to(device)
    optimizer = torch.optim.Adam(hnn.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    n_params = sum(p.numel() for p in hnn.parameters())
    log.info(f"Exp 2.3c: HNN + trajectory training on Lotka-Volterra")
    log.info(f"α={ALPHA}, β={BETA}, δ={DELTA}, γ={GAMMA}")
    log.info(f"Fixed point: x*={X_STAR:.1f}, y*={Y_STAR:.1f}")
    log.info(f"HNN MLP: 2→64→64→64→1 (scalar H), {n_params:,} params")
    log.info(f"Epochs={N_EPOCHS}, segment_len={SEGMENT_LEN}, segs/epoch={SEGS_PER_EPOCH}")
    log.info(f"Device: {DEVICE}")

    # ── Training: trajectory-level backprop through HNN ──────────────
    epoch_losses = []
    checkpoint_results = []
    t_start = time.time()

    for epoch in range(N_EPOCHS):
        x0 = rng.uniform(*X_RANGE)
        y0 = rng.uniform(*Y_RANGE)

        traj = generate_lv_trajectory(x0, y0, TRAJ_DURATION, DT_S)
        traj_t = torch.tensor(traj, device=device, dtype=torch.float32)

        # Subsample segment start points
        all_starts = list(range(0, len(traj) - SEGMENT_LEN - 1))
        rng.shuffle(all_starts)
        starts = all_starts[:SEGS_PER_EPOCH]

        epoch_loss = 0.0
        n_segments = 0

        for seg_start in starts:
            y0_seg = traj_t[seg_start]
            target = traj_t[seg_start:seg_start + SEGMENT_LEN + 1]

            # Integrate HNN through segment with autograd graph
            pred = integrate_hnn(hnn, y0_seg, SEGMENT_LEN, DT_S, create_graph=True)

            loss = torch.mean((pred - target) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hnn.parameters(), 1.0)
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
    ax.set_ylabel("MSE Loss (log)")
    ax.set_title("Exp 2.3c: HNN + Trajectory Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/training_loss.png", dpi=150)
    plt.close(fig)

    # ── Checkpoint plot ──────────────────────────────────────────────
    ckpt_epochs, ckpt_errs = zip(*checkpoint_results)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ckpt_epochs, ckpt_errs, "go-", markersize=6, linewidth=2, label="HNN+traj")
    ax.axhline(y=6.52, color="orange", linestyle="--", alpha=0.5, label="Vanilla NODE (2.3)")
    ax.axhline(y=23.6, color="red", linestyle=":", alpha=0.5, label="HNN deriv-match (2.3b)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Error (x=15, y=8)")
    ax.set_title("Exp 2.3c: HNN+Trajectory Generalization During Training")
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
        axes[0].set_title(f"Exp 2.3c HNN+Traj: x₀={x0}, y₀={y0} ({ic_label})")
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

        # Conservation quantity
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
                                (axes[1], all_preds, "HNN+Traj")]:
        for i, (data, (x0, y0, label)) in enumerate(zip(dataset, TEST_ICS)):
            ax.plot(data[:, 0], data[:, 1], color=colors[i], lw=0.8, alpha=0.8, label=label)
            ax.plot(data[0, 0], data[0, 1], "o", color=colors[i], ms=6)
        ax.plot(X_STAR, Y_STAR, "k*", ms=12, label=f"Fixed pt ({X_STAR:.0f},{Y_STAR:.0f})")
        ax.set_xlabel("x (prey)")
        ax.set_ylabel("y (predators)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Lotka-Volterra HNN+Trajectory: Multiple Orbits", fontsize=13)
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
    log.info(f"\n{'='*80}")
    log.info("SUMMARY — HNN+Traj vs HNN deriv-match vs vanilla NODE")
    log.info(f"{'='*80}")
    node_errs = [1.16, 8.14, 0.92, 4.78, 17.61]
    node_drifts = [0.028, 0.541, 0.026, 0.306, 1.145]
    hnn_dm_errs = [5.29, 26.17, 7.03, 20.04, 59.39]
    hnn_dm_drifts = [0.001, 0.453, 0.315, 0.264, 0.802]
    log.info(f"{'IC':<16} {'x₀':>5} {'y₀':>5} {'HNN+T err':>10} {'HNN-D err':>10} "
             f"{'NODE err':>9} {'HNN+T drift':>12} {'NODE drift':>11}")
    log.info("-" * 95)
    for i, (label, x0, y0, err) in enumerate(summary):
        pred_i = all_preds[i]
        V_p = conservation_quantity(pred_i[:, 0], pred_i[:, 1])
        drift_i = np.abs(V_p[-1] - V_p[0])
        log.info(f"{label:<16} {x0:>5.1f} {y0:>5.1f} {err:>10.4f} {hnn_dm_errs[i]:>10.4f} "
                 f"{node_errs[i]:>9.4f} {drift_i:>12.6f} {node_drifts[i]:>11.6f}")

    mean_err = np.mean([e for _, _, _, e in summary])
    log.info(f"\nHNN+Traj mean error:     {mean_err:.4f}")
    log.info(f"HNN deriv-match mean:    23.5839")
    log.info(f"Vanilla NODE mean:       6.5212")

    np.savez(f"{RESULTS_DIR}/results.npz",
             epoch_losses=np.array(epoch_losses),
             checkpoint_results=np.array(checkpoint_results),
             summary=np.array([(x0, y0, err) for _, x0, y0, err in summary]))
    log.info(f"\nAll results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
