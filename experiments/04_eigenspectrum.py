"""Experiment 04: Eigenspectrum analysis — look inside the weight matrix.

Train on sine wave, snapshot eigenvalues of J at 0%, 25%, 50%, 75%, 100%.
Also analyze singular value spectrum of M.
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
from src.targets import sine_target
from src.utils import compute_eigenspectrum, compute_singular_values, set_seed
from src.instrumentation import plot_eigenspectrum

# ── Config ──────────────────────────────────────────────────────────
N = 500
K = 1
PERIOD = 600.0
AMPLITUDE = 1.5
TAU = 10.0
DT = 1.0
ETA_W = 1e-3
ETA_M = 1e-3
ALPHA = 1.0
G = 1.2

TRAIN_MS = 300_000.0
TRAIN_STEPS = int(TRAIN_MS / DT)

SNAPSHOT_FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.0]
SEED = 42

RESULTS_DIR = "/raid/predictive_alignment/results/04_eigenspectrum"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)

    net = PredictiveAlignmentRNN(
        N=N, K=K, g=G, tau=TAU, dt=DT,
        eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
        device=torch.device(DEVICE), seed=SEED,
    )

    snapshot_steps = [int(f * TRAIN_STEPS) for f in SNAPSHOT_FRACTIONS]
    eig_snapshots = []
    sv_snapshots = []
    labels = []

    # Before training snapshot
    eig_snapshots.append(compute_eigenspectrum(net.get_J()))
    sv_snapshots.append(compute_singular_values(net.M))
    labels.append("0% (before)")

    print(f"Training for {TRAIN_STEPS} steps...")
    next_snap_idx = 1  # already took 0%

    for step in tqdm(range(TRAIN_STEPS), desc="Training", mininterval=2.0):
        t = step * DT
        f_val = sine_target(t, period=PERIOD, amplitude=AMPLITUDE)
        target = torch.tensor([f_val], device=DEVICE, dtype=torch.float32)
        net.step_and_learn(target)

        if next_snap_idx < len(snapshot_steps) and step + 1 >= snapshot_steps[next_snap_idx]:
            frac = SNAPSHOT_FRACTIONS[next_snap_idx]
            eig_snapshots.append(compute_eigenspectrum(net.get_J()))
            sv_snapshots.append(compute_singular_values(net.M))
            labels.append(f"{frac:.0%}")
            print(f"  Snapshot at {frac:.0%}")
            next_snap_idx += 1

    # ── Plot 1: Eigenspectrum evolution ─────────────────────────────
    plot_eigenspectrum(
        eig_snapshots, labels=labels,
        save_path=f"{RESULTS_DIR}/eigenspectrum_evolution.png",
        title="Exp 04: Eigenspectrum of J During Training",
    )

    # ── Plot 2: Individual eigenspectrum panels ─────────────────────
    fig, axes = plt.subplots(1, len(eig_snapshots), figsize=(4 * len(eig_snapshots), 4))
    for i, (eigs, label) in enumerate(zip(eig_snapshots, labels)):
        ax = axes[i]
        ax.scatter(eigs.real, eigs.imag, s=2, alpha=0.4)
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(G * np.cos(theta), G * np.sin(theta), "k--", alpha=0.3)
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.set_xlabel("Real")
        if i == 0:
            ax.set_ylabel("Imaginary")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/eigenspectrum_panels.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/eigenspectrum_panels.png")
    plt.close(fig)

    # ── Plot 3: Singular value spectrum of M ────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for svs, label in zip(sv_snapshots, labels):
        ax.plot(svs[:50], label=label, alpha=0.7)
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title("Exp 04: Singular Values of M (top 50)")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/singular_values_M.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/singular_values_M.png")
    plt.close(fig)

    # ── Plot 4: Effective rank of M ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for svs, label in zip(sv_snapshots, labels):
        # Effective rank = exp(entropy of normalized singular values)
        s_norm = svs / svs.sum()
        s_norm = s_norm[s_norm > 1e-10]
        entropy = -np.sum(s_norm * np.log(s_norm))
        eff_rank = np.exp(entropy)
        ax.bar(label, eff_rank, alpha=0.7)
    ax.set_ylabel("Effective rank")
    ax.set_title("Exp 04: Effective Rank of M During Training")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/effective_rank_M.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/effective_rank_M.png")
    plt.close(fig)

    # Save data
    np.savez(f"{RESULTS_DIR}/results.npz",
             eig_snapshots=[e for e in eig_snapshots],
             sv_snapshots=[s for s in sv_snapshots],
             labels=labels)
    print(f"Saved: {RESULTS_DIR}/results.npz")


if __name__ == "__main__":
    main()
