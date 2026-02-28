"""Experiment 03: Edge of chaos sweep — tests the role of initial chaos.

Sweeps g = {0.5, 0.7, 0.9, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0} with 20 seeds each.
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
from src.utils import estimate_lyapunov, frobenius_norm, compute_eigenspectrum, participation_ratio, set_seed

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

TRAIN_MS = 300_000.0
TRAIN_STEPS = int(TRAIN_MS / DT)

G_VALUES = [0.5, 0.7, 0.9, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0]
N_SEEDS = 20

RESULTS_DIR = "/raid/predictive_alignment/results/03_edge_of_chaos"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_single(g_val, seed):
    """Run one training session with given g, return metrics."""
    set_seed(seed)
    net = PredictiveAlignmentRNN(
        N=N, K=K, g=g_val, tau=TAU, dt=DT,
        eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
        device=torch.device(DEVICE), seed=seed,
    )

    # Lyapunov before training
    lyap_before = estimate_lyapunov(net, n_steps=2000)

    errors = []
    for step in range(TRAIN_STEPS):
        t = step * DT
        f_val = sine_target(t, period=PERIOD, amplitude=AMPLITUDE)
        target = torch.tensor([f_val], device=DEVICE, dtype=torch.float32)
        z = net.step_and_learn(target)
        if step % 1000 == 0:
            errors.append(abs(f_val - z[0].item()))

    final_error = np.mean(errors[-10:])
    w_norm = frobenius_norm(net.w)
    M_norm = frobenius_norm(net.M)

    # Recurrent prediction error ||Qz - Mr||
    r = net.r.detach()
    z_out = net.z.detach()
    rec_pred_err = torch.norm(net.Q @ z_out - net.M @ r).item()

    return {
        "final_error": final_error,
        "lyap_before": lyap_before,
        "w_norm": w_norm,
        "M_norm": M_norm,
        "rec_pred_err": rec_pred_err,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}
    for g_val in G_VALUES:
        print(f"\n=== g = {g_val} ===")
        results = []
        for seed in tqdm(range(N_SEEDS), desc=f"g={g_val}"):
            r = run_single(g_val, seed)
            results.append(r)
        all_results[g_val] = results

    # ── Plot 1: Error vs g ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [np.mean([r["final_error"] for r in all_results[g]]) for g in G_VALUES]
    stds = [np.std([r["final_error"] for r in all_results[g]]) for g in G_VALUES]
    ax.errorbar(G_VALUES, means, yerr=stds, fmt="o-", capsize=5)
    ax.set_xlabel("g (chaos gain)")
    ax.set_ylabel("Final Readout Error")
    ax.set_title("Exp 03: Error vs Chaos Gain g")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/error_vs_g.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/error_vs_g.png")
    plt.close(fig)

    # ── Plot 2: Weight norms vs g ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, (key, ylabel) in enumerate([("w_norm", "||w||_F"), ("M_norm", "||M||_F")]):
        means = [np.mean([r[key] for r in all_results[g]]) for g in G_VALUES]
        stds = [np.std([r[key] for r in all_results[g]]) for g in G_VALUES]
        axes[i].errorbar(G_VALUES, means, yerr=stds, fmt="o-", capsize=5, color="green")
        axes[i].set_xlabel("g")
        axes[i].set_ylabel(ylabel)
        axes[i].set_title(f"{ylabel} vs g")
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/weight_norms_vs_g.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/weight_norms_vs_g.png")
    plt.close(fig)

    # ── Plot 3: Lyapunov before training vs g ───────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [np.mean([r["lyap_before"] for r in all_results[g]]) for g in G_VALUES]
    stds = [np.std([r["lyap_before"] for r in all_results[g]]) for g in G_VALUES]
    ax.errorbar(G_VALUES, means, yerr=stds, fmt="o-", capsize=5, color="red")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("g")
    ax.set_ylabel("Lyapunov Exponent (before training)")
    ax.set_title("Exp 03: Pre-training Chaos vs g")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/lyapunov_before_vs_g.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/lyapunov_before_vs_g.png")
    plt.close(fig)

    print(f"\nAll plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
