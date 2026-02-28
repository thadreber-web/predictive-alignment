"""Experiment 2.0: BPTT baseline — same architecture, standard backprop.

Tests whether self-feeding instability is specific to predictive alignment
or a general RNN problem. Uses identical setup to exp2.1:
  - N=500, K=2, D=2, scheduled sampling, multi-IC training
  - But trained with truncated BPTT (PyTorch autograd) instead of PA rule

Truncated BPTT: unroll T_BPTT steps, backprop through that window.
This gives the optimizer gradient signal through the self-feeding loop,
which predictive alignment lacks.
"""

import sys
sys.path.insert(0, "/raid/predictive_alignment")

import torch
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time
import math

from src.targets import generate_pendulum
from src.utils import set_seed

# ── Config (matches exp2.1 exactly) ─────────────────────────────────
N = 500
K = 2
D = 2
TAU = 10.0
DT = 1.0
G_GAIN = 1.2
P_SPARSE = 0.1
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
N_ANNEAL = 400
LR = 1e-3
T_BPTT = 50  # truncated backprop window

# IC ranges
THETA_RANGE = (-2.5, 2.5)
OMEGA_RANGE = (-3.0, 3.0)

# Test ICs (same as exp2.1)
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
RECORD_EVERY = 10
RESULTS_DIR = "/raid/predictive_alignment/results/exp2_0_bptt_baseline"


class BPTT_RNN(nn.Module):
    """RNN with same architecture as PredictiveAlignmentRNN but trained with BPTT.

    Same structure: sparse fixed G, dense trainable M, readout w, input W_in.
    Dynamics: tau * dx/dt = -x + (G + M) @ tanh(x) + W_in @ input
    Output: z = w @ tanh(x)
    """

    def __init__(self, N, K, D, g, tau, dt, p=0.1, seed=42):
        super().__init__()
        self.N = N
        self.K = K
        self.D = D
        self.tau = tau
        self.dt = dt

        rng = torch.Generator()
        rng.manual_seed(seed)

        # G: sparse fixed (not trainable)
        std_g = g / math.sqrt(p * N)
        G = torch.randn(N, N, generator=rng) * std_g
        mask = torch.rand(N, N, generator=rng) < p
        G = G * mask.float()
        G.fill_diagonal_(0.0)
        self.register_buffer("G", G)

        # M: dense trainable
        std_m = 0.5 / math.sqrt(N)
        self.M = nn.Parameter(torch.randn(N, N, generator=rng) * std_m)

        # w: readout (trainable)
        self.w = nn.Parameter(torch.zeros(K, N))

        # W_in: input weights (trainable)
        self.W_in = nn.Parameter(torch.randn(N, D, generator=rng) * (1.0 / math.sqrt(D)))

    def forward_step(self, x, ext_input):
        """One timestep. Returns (new_x, z)."""
        r = torch.tanh(x)
        J = self.G + self.M
        current = J @ r + self.W_in @ ext_input
        dx = (-x + current) * (self.dt / self.tau)
        x_new = x + dx
        z = self.w @ r
        return x_new, z

    def init_state(self, device):
        return torch.randn(self.N, device=device) * 0.1


def pendulum_energy(theta, omega):
    return 0.5 * omega ** 2 + (GRAV / L_PEND) * (1 - np.cos(theta))


def test_self_feeding(model, ic_theta, ic_omega, test_steps, device):
    """Self-feeding test (no grad)."""
    model.eval()
    with torch.no_grad():
        x = model.init_state(device)
        prev_z = torch.tensor([ic_theta, ic_omega], device=device, dtype=torch.float32)
        zs = []
        for _ in range(test_steps):
            x, z = model.forward_step(x, prev_z)
            prev_z = z.clone()
            zs.append(z.cpu().numpy())
    model.train()
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
    log = logging.getLogger("exp2.0")

    log.info(f"Exp 2.0: BPTT baseline")
    log.info(f"N={N}, K={K}, D={D}, epochs={N_EPOCHS}, anneal={N_ANNEAL}, T_BPTT={T_BPTT}")
    log.info(f"θ range: {THETA_RANGE}, ω range: {OMEGA_RANGE}")
    log.info(f"LR={LR}, Device: {DEVICE}")

    set_seed(SEED)
    rng = np.random.RandomState(SEED)

    device = torch.device(DEVICE)
    model = BPTT_RNN(N, K, D, G_GAIN, TAU, DT, P_SPARSE, SEED).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable params: {n_params:,} (M: {N*N:,}, w: {K*N:,}, W_in: {N*D:,})")

    # ── Training ─────────────────────────────────────────────────────
    epoch_errors = []
    epoch_tf_ratios = []
    checkpoint_results = []

    t_start = time.time()

    for epoch in range(N_EPOCHS):
        p_tf = max(0.0, 1.0 - epoch / N_ANNEAL)

        theta0 = rng.uniform(*THETA_RANGE)
        omega0 = rng.uniform(*OMEGA_RANGE)

        traj = generate_pendulum(
            duration_ms=TRAJ_MS, dt=DT, b=B, g=GRAV, L=L_PEND,
            theta0=theta0, omega0=omega0,
        )
        traj_t = torch.tensor(traj, device=device, dtype=torch.float32)

        x = model.init_state(device)
        prev_z = torch.tensor([theta0, omega0], device=device, dtype=torch.float32)

        epoch_loss = 0.0
        tf_count = 0
        n_chunks = 0

        # Process trajectory in BPTT chunks
        for chunk_start in range(0, TRAJ_STEPS - 1, T_BPTT):
            chunk_end = min(chunk_start + T_BPTT, TRAJ_STEPS - 1)
            chunk_loss = torch.tensor(0.0, device=device)

            # Detach x at chunk boundary to truncate gradients
            x = x.detach()
            prev_z = prev_z.detach()

            for t in range(chunk_start, chunk_end):
                target = traj_t[t + 1]

                if rng.random() < p_tf:
                    ext_input = traj_t[t]
                    tf_count += 1
                else:
                    ext_input = prev_z

                x, z = model.forward_step(x, ext_input)
                prev_z = z
                chunk_loss = chunk_loss + torch.sum((target - z) ** 2)

            # Backprop through this chunk
            chunk_len = chunk_end - chunk_start
            avg_loss = chunk_loss / chunk_len
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += chunk_loss.item()
            n_chunks += 1

            # Detach for next chunk
            x = x.detach()
            prev_z = prev_z.detach()

        epoch_err = math.sqrt(epoch_loss / (TRAJ_STEPS - 1))
        epoch_errors.append(epoch_err)
        epoch_tf_ratios.append(tf_count / (TRAJ_STEPS - 1))

        if epoch % 25 == 0 or epoch == N_EPOCHS - 1:
            log.info(f"Epoch {epoch:4d}/{N_EPOCHS} | p_tf={p_tf:.3f} | "
                     f"actual_tf={epoch_tf_ratios[-1]:.3f} | rmse={epoch_err:.4f}")

        # Checkpoint
        if epoch % CHECKPOINT_EVERY == 0 or epoch == N_EPOCHS - 1:
            ic_th, ic_om = 1.0, 0.0
            truth = generate_pendulum(
                duration_ms=TEST_MS, dt=DT, b=B, g=GRAV, L=L_PEND,
                theta0=ic_th, omega0=ic_om,
            )[:TEST_STEPS]
            z_test = test_self_feeding(model, ic_th, ic_om, TEST_STEPS, device)
            ckpt_err = np.mean(np.linalg.norm(truth - z_test, axis=1))
            checkpoint_results.append((epoch, ckpt_err))
            log.info(f"  Checkpoint (θ=1.0,ω=0.0) self-feeding error: {ckpt_err:.4f}")

    elapsed = time.time() - t_start
    log.info(f"Training done in {elapsed:.1f}s")

    # ── Plot: training curve + TF ratio ──────────────────────────────
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(epoch_errors, "b-", alpha=0.5, linewidth=0.5, label="Epoch RMSE")
    w = max(1, len(epoch_errors) // 50)
    smoothed = np.convolve(epoch_errors, np.ones(w)/w, mode="valid")
    ax1.plot(np.arange(w-1, w-1+len(smoothed)), smoothed, "b-", linewidth=2, label="Smoothed RMSE")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training RMSE", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epoch_tf_ratios, "r-", alpha=0.3, linewidth=0.5)
    ax2.set_ylabel("Teacher Forcing Ratio", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.set_ylim(-0.05, 1.05)

    ax1.set_title("Exp 2.0: BPTT Baseline — Training RMSE & TF Schedule")
    ax1.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/training_curve.png", dpi=150)
    plt.close(fig)

    # ── Plot: checkpoint stability ───────────────────────────────────
    ckpt_epochs, ckpt_errs = zip(*checkpoint_results)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ckpt_epochs, ckpt_errs, "go-", markersize=6, linewidth=2, label="BPTT")
    ax.axhline(y=3.6, color="orange", linestyle="--", alpha=0.5, label="PA sched. sampling (exp2.1)")
    ax.axhline(y=2.25, color="gray", linestyle="--", alpha=0.5, label="PA multi-IC (07e)")
    ax.axhline(y=0.5, color="green", linestyle=":", alpha=0.5, label="Success threshold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Self-feeding Error (θ=1.0, ω=0.0)")
    ax.set_title("Exp 2.0: BPTT Self-feeding Stability During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/checkpoint_stability.png", dpi=150)
    plt.close(fig)

    # ── Final generalization test ────────────────────────────────────
    log.info(f"\nFinal generalization test on {len(TEST_ICS)} held-out ICs:")
    summary = []

    for ic_theta, ic_omega, ic_label in TEST_ICS:
        truth = generate_pendulum(
            duration_ms=TEST_MS, dt=DT, b=B, g=GRAV, L=L_PEND,
            theta0=ic_theta, omega0=ic_omega,
        )[:TEST_STEPS]

        z_pred = test_self_feeding(model, ic_theta, ic_omega, TEST_STEPS, device)
        err = np.mean(np.linalg.norm(truth - z_pred, axis=1))
        summary.append((ic_label, ic_theta, ic_omega, err))
        log.info(f"  {ic_label:20s} θ={ic_theta:+.1f} ω={ic_omega:+.1f} → error={err:.4f}")

        t = np.arange(TEST_STEPS) * DT

        # Time series
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        axes[0].plot(t, truth[:, 0], "k-", lw=1.5, label="Ground truth")
        axes[0].plot(t, z_pred[:, 0], "r-", lw=1, alpha=0.8, label=f"BPTT (err={err:.3f})")
        axes[0].set_ylabel("θ (rad)")
        axes[0].set_title(f"Exp 2.0 BPTT: θ₀={ic_theta}, ω₀={ic_omega} ({ic_label})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, truth[:, 1], "k-", lw=1.5, label="Ground truth")
        axes[1].plot(t, z_pred[:, 1], "r-", lw=1, alpha=0.8, label="BPTT")
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
            (axes[1], z_pred, "BPTT predicted", "r"),
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
        ax.plot(t, E_pred, "r-", lw=1, alpha=0.8, label="BPTT")
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
    log.info("SUMMARY — BPTT vs Predictive Alignment")
    log.info(f"{'='*60}")
    log.info(f"{'IC':<20} {'θ₀':>6} {'ω₀':>6} {'BPTT':>10} {'PA 2.1':>10} {'PA 07e':>10}")
    log.info("-" * 65)
    pa_errors = [3.378, 3.737, 3.412, 3.644, 3.859]  # from exp2.1
    for i, (label, th, om, err) in enumerate(summary):
        log.info(f"{label:<20} {th:>+6.1f} {om:>+6.1f} {err:>10.4f} {pa_errors[i]:>10.4f}")
    mean_err = np.mean([e for _, _, _, e in summary])
    log.info(f"\nBPTT mean generalization error: {mean_err:.4f}")
    log.info(f"PA scheduled sampling (exp2.1): 3.606")
    log.info(f"PA multi-IC (07e): ~2.3")
    log.info(f"Success threshold: < 0.5")

    np.savez(f"{RESULTS_DIR}/results.npz",
             epoch_errors=np.array(epoch_errors),
             epoch_tf_ratios=np.array(epoch_tf_ratios),
             checkpoint_results=np.array(checkpoint_results),
             summary=np.array([(th, om, err) for _, th, om, err in summary]))
    log.info(f"\nAll results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
