"""Experiment 02: Break the alignment — tests the alpha parameter.

Sweeps alpha = {0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0} with 20 seeds each.
Reproduces Figure 3C-E of the paper.
"""

import sys
sys.path.insert(0, "/raid/predictive_alignment")

import torch
import numpy as np
import logging
import os
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.network import PredictiveAlignmentRNN
from src.targets import sine_target
from src.utils import estimate_lyapunov, alignment_correlation, set_seed

# ── Config ──────────────────────────────────────────────────────────
N = 500
K = 1
PERIOD = 600.0
AMPLITUDE = 1.5
TAU = 10.0
DT = 1.0
ETA_W = 1e-3
ETA_M = 1e-3
G = 1.2

TRAIN_MS = 300_000.0
TRAIN_STEPS = int(TRAIN_MS / DT)

ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
N_SEEDS = 20

RESULTS_DIR = "/raid/predictive_alignment/results/02_break_alpha"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_single(alpha, seed):
    """Run one training session, return final metrics."""
    set_seed(seed)
    net = PredictiveAlignmentRNN(
        N=N, K=K, g=G, tau=TAU, dt=DT,
        eta_w=ETA_W, eta_m=ETA_M, alpha=alpha,
        device=torch.device(DEVICE), seed=seed,
    )

    errors = []
    align_history = []

    for step in range(TRAIN_STEPS):
        t = step * DT
        f_val = sine_target(t, period=PERIOD, amplitude=AMPLITUDE)
        target = torch.tensor([f_val], device=DEVICE, dtype=torch.float32)
        z = net.step_and_learn(target)

        if step % 1000 == 0:
            errors.append(abs(f_val - z[0].item()))
            align_history.append(alignment_correlation(net.G, net.M, net.r))

    # Final metrics
    lyap = estimate_lyapunov(net, n_steps=2000)
    final_error = np.mean(errors[-10:])
    final_align = np.mean(align_history[-10:])

    # Gr vs Mr scatter (sample 200 neurons)
    r = net.r.detach()
    Gr = (net.G @ r).cpu().numpy()
    Mr = (net.M @ r).cpu().numpy()

    return {
        "final_error": final_error,
        "lyapunov": lyap,
        "alignment": final_align,
        "errors": np.array(errors),
        "align_history": np.array(align_history),
        "Gr": Gr,
        "Mr": Mr,
    }


LOG_FILE = f"{RESULTS_DIR}/experiment.log"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("exp02")
    log.info(f"Config: N={N}, K={K}, g={G}, dt={DT}, eta_w={ETA_W}, eta_m={ETA_M}")
    log.info(f"Alpha values: {ALPHA_VALUES}, seeds per alpha: {N_SEEDS}")
    log.info(f"Train: {TRAIN_MS/1000:.0f}s ({TRAIN_STEPS} steps per run)")
    log.info(f"Total runs: {len(ALPHA_VALUES) * N_SEEDS}")
    log.info(f"Log: tail -f {LOG_FILE}")

    all_results = {}
    for alpha in ALPHA_VALUES:
        log.info(f"\n=== alpha = {alpha} ===")
        results = []
        for seed in tqdm(range(N_SEEDS), desc=f"alpha={alpha}"):
            r = run_single(alpha, seed)
            results.append(r)
            if seed == 0 or (seed + 1) % 5 == 0:
                log.info(f"  alpha={alpha} seed={seed}: error={r['final_error']:.6f}, "
                         f"lyap={r['lyapunov']:.4f}, align={r['alignment']:.4f}")
        mean_err = np.mean([r["final_error"] for r in results])
        mean_lyap = np.mean([r["lyapunov"] for r in results])
        log.info(f"  alpha={alpha} SUMMARY: mean_error={mean_err:.6f}, mean_lyap={mean_lyap:.4f}")
        all_results[alpha] = results

    # ── Plot 1: Final error vs alpha ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [np.mean([r["final_error"] for r in all_results[a]]) for a in ALPHA_VALUES]
    stds = [np.std([r["final_error"] for r in all_results[a]]) for a in ALPHA_VALUES]
    ax.errorbar(ALPHA_VALUES, means, yerr=stds, fmt="o-", capsize=5)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Final Readout Error")
    ax.set_title("Exp 02: Error vs Alpha (Fig 3C)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/error_vs_alpha.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/error_vs_alpha.png")
    plt.close(fig)

    # ── Plot 2: Lyapunov vs alpha ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [np.mean([r["lyapunov"] for r in all_results[a]]) for a in ALPHA_VALUES]
    stds = [np.std([r["lyapunov"] for r in all_results[a]]) for a in ALPHA_VALUES]
    ax.errorbar(ALPHA_VALUES, means, yerr=stds, fmt="o-", capsize=5, color="red")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Lyapunov Exponent")
    ax.set_title("Exp 02: Lyapunov vs Alpha")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/lyapunov_vs_alpha.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/lyapunov_vs_alpha.png")
    plt.close(fig)

    # ── Plot 3: Alignment over training for alpha=0 vs alpha=1 ──────
    fig, ax = plt.subplots(figsize=(8, 5))
    for alpha_plot in [0.0, 1.0]:
        histories = [r["align_history"] for r in all_results[alpha_plot]]
        mean_h = np.mean(histories, axis=0)
        std_h = np.std(histories, axis=0)
        t_h = np.arange(len(mean_h)) * 1000 * DT  # every 1000 steps
        ax.plot(t_h, mean_h, label=f"alpha={alpha_plot}")
        ax.fill_between(t_h, mean_h - std_h, mean_h + std_h, alpha=0.2)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Alignment <Gr, Mr>")
    ax.set_title("Exp 02: Alignment Over Training (Fig 3D)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/alignment_over_time.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/alignment_over_time.png")
    plt.close(fig)

    # ── Plot 4: Gr vs Mr scatter for alpha=0 vs alpha=1 ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, alpha_plot in enumerate([0.0, 1.0]):
        r0 = all_results[alpha_plot][0]
        axes[i].scatter(r0["Gr"], r0["Mr"], s=1, alpha=0.3)
        axes[i].set_xlabel("(G @ r)_i")
        axes[i].set_ylabel("(M @ r)_i")
        axes[i].set_title(f"alpha={alpha_plot} (Fig 3E)")
        axes[i].set_aspect("equal")
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/Gr_vs_Mr_scatter.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/Gr_vs_Mr_scatter.png")
    plt.close(fig)

    # Save numeric results
    np.savez(f"{RESULTS_DIR}/results.npz",
             alpha_values=ALPHA_VALUES,
             final_errors={str(a): [r["final_error"] for r in all_results[a]] for a in ALPHA_VALUES},
             lyapunovs={str(a): [r["lyapunov"] for r in all_results[a]] for a in ALPHA_VALUES})
    print(f"Saved: {RESULTS_DIR}/results.npz")


if __name__ == "__main__":
    main()
