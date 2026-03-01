"""Experiment 3.3: Alpha sweep for continual learning.

Sine→Lorenz protocol from 3.1, but with varying alpha during Phase 2.
Phase 1 always uses alpha=1.0. Phase 2 uses alpha in [0.0, 0.5, 1.0, 2.0, 4.0].

Question: does alpha modulate forgetting? Higher alpha = more M regularization
toward G, which might protect prior knowledge in M.
"""

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

import torch
import numpy as np
import math
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from src.targets import sine_target, generate_lorenz
from src.utils import set_seed

# ── Config ───────────────────────────────────────────────────────────
N = 500
TAU = 10.0
DT = 1.0
ETA_W = 1e-3
ETA_M = 1e-3
G_GAIN = 1.2
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SINE_PERIOD = 600.0
SINE_AMP = 1.5
PHASE1_MS = 30_000.0
PHASE1_STEPS = int(PHASE1_MS / DT)

PHASE2_MS = 15_000.0
PHASE2_STEPS = int(PHASE2_MS / DT)

FORGET_TEST_EVERY = 500
FORGET_TEST_DURATION = 5_000.0
FORGET_TEST_STEPS = int(FORGET_TEST_DURATION / DT)

TEST_MS = 10_000.0
TEST_STEPS = int(TEST_MS / DT)

ALPHA_VALUES = [0.0, 0.5, 1.0, 2.0, 4.0]

RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "exp3_3_alpha_sweep")


class MultiTaskPA:
    """PA network that supports task-specific readouts sharing G and M."""

    def __init__(self, N, g, tau, dt, eta_w, eta_m, alpha, device, seed):
        self.N = N
        self.g = g
        self.tau = tau
        self.dt = dt
        self.eta_w = eta_w
        self.eta_m = eta_m
        self.alpha = alpha
        self.device = torch.device(device)
        self.p = 0.1

        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        self.rng = rng

        std_g = g / math.sqrt(self.p * N)
        G = torch.randn(N, N, device=self.device, generator=rng) * std_g
        mask = torch.rand(N, N, device=self.device, generator=rng) < self.p
        self.G = G * mask.float()
        self.G.fill_diagonal_(0.0)

        std_m = 0.01 / math.sqrt(N)
        self.M = torch.randn(N, N, device=self.device, generator=rng) * std_m

        self.tasks = {}
        self.active_task = None

        self.x = torch.randn(N, device=self.device, generator=rng) * 0.1
        self.r = torch.tanh(self.x)

    def add_task(self, name, K):
        w = torch.zeros(K, self.N, device=self.device)
        q_bound = 3.0 / math.sqrt(K)
        Q = torch.empty(self.N, K, device=self.device).uniform_(-q_bound, q_bound)
        self.tasks[name] = {"w": w, "Q": Q, "K": K}

    def set_active_task(self, name):
        self.active_task = name

    def reset_state(self):
        self.x = torch.randn(self.N, device=self.device, generator=self.rng) * 0.1
        self.r = torch.tanh(self.x)

    def step(self):
        task = self.tasks[self.active_task]
        self.r = torch.tanh(self.x)
        J = self.G + self.M
        current = J @ self.r
        dx = (-self.x + current) * (self.dt / self.tau)
        self.x = self.x + dx
        z = task["w"] @ self.r
        return z

    def step_and_learn(self, target):
        task = self.tasks[self.active_task]
        z = self.step()

        output_error = target - z
        task["w"] = task["w"] + self.eta_w * torch.outer(output_error, self.r)

        feedback = task["Q"] @ z
        J_hat_r = (self.M - self.alpha * self.G) @ self.r
        rec_error = feedback - J_hat_r
        self.M = self.M + self.eta_m * torch.outer(rec_error, self.r)

        return z

    def snapshot_M(self):
        return self.M.clone()


def test_autonomous(net, task_name, test_steps):
    net.set_active_task(task_name)
    net.reset_state()
    zs = []
    for _ in range(test_steps):
        z = net.step()
        zs.append(z.detach().cpu().numpy())
    return np.array(zs)


def compute_error(predicted, truth, K):
    pred = predicted[:len(truth)]
    truth = truth[:len(pred)]
    if K == 1:
        return np.mean(np.abs(pred.ravel() - truth.ravel()))
    else:
        return np.mean(np.linalg.norm(pred - truth, axis=1))


def run_one_alpha(alpha_p2, device, sine_data, lorenz_data, log):
    """Run sine→Lorenz with given alpha during Phase 2. Returns metrics dict."""
    log.info(f"\n{'='*60}")
    log.info(f"Running alpha={alpha_p2}")
    log.info(f"{'='*60}")

    net = MultiTaskPA(N=N, g=G_GAIN, tau=TAU, dt=DT,
                      eta_w=ETA_W, eta_m=ETA_M, alpha=1.0,  # Phase 1 always alpha=1
                      device=device, seed=SEED)
    net.add_task("sine", K=1)
    net.add_task("lorenz", K=3)

    # Phase 1: sine with alpha=1.0
    net.set_active_task("sine")
    for t in range(PHASE1_STEPS):
        target = torch.tensor([sine_data[t]], device=device, dtype=torch.float32)
        net.step_and_learn(target)

    M_after_phase1 = net.snapshot_M()

    # Test sine after Phase 1
    sine_truth = sine_data[:TEST_STEPS].reshape(-1, 1)
    sine_pred_p1 = test_autonomous(net, "sine", TEST_STEPS)
    sine_err_p1 = compute_error(sine_pred_p1, sine_truth, K=1)
    log.info(f"  Sine error after Phase 1: {sine_err_p1:.4f}")

    # Phase 2: Lorenz with varied alpha
    net.alpha = alpha_p2
    net.set_active_task("lorenz")

    forgetting_curve = []
    # Initial sine test
    sine_test_truth = sine_data[:FORGET_TEST_STEPS].reshape(-1, 1)
    sine_test_pred = test_autonomous(net, "sine", FORGET_TEST_STEPS)
    sine_err_init = compute_error(sine_test_pred, sine_test_truth, K=1)
    forgetting_curve.append((0, sine_err_init))

    net.set_active_task("lorenz")
    for t in range(PHASE2_STEPS):
        target = torch.tensor(lorenz_data[t], device=device, dtype=torch.float32)
        net.step_and_learn(target)

        if (t + 1) % FORGET_TEST_EVERY == 0:
            x_save = net.x.clone()
            r_save = net.r.clone()

            sine_test_pred = test_autonomous(net, "sine", FORGET_TEST_STEPS)
            sine_err = compute_error(sine_test_pred, sine_test_truth, K=1)
            forgetting_curve.append((t + 1, sine_err))

            net.x = x_save
            net.r = r_save
            net.set_active_task("lorenz")

    M_after_phase2 = net.snapshot_M()

    # Final tests
    sine_pred_p2 = test_autonomous(net, "sine", TEST_STEPS)
    sine_err_p2 = compute_error(sine_pred_p2, sine_truth, K=1)

    lorenz_truth = lorenz_data[:TEST_STEPS]
    lorenz_pred_p2 = test_autonomous(net, "lorenz", TEST_STEPS)
    lorenz_err_p2 = compute_error(lorenz_pred_p2, lorenz_truth, K=3)

    M_delta_norm = torch.norm(M_after_phase2 - M_after_phase1, p="fro").item()
    forgetting_ratio = sine_err_p2 / max(sine_err_p1, 1e-8)

    log.info(f"  Sine error after Phase 2: {sine_err_p2:.4f}")
    log.info(f"  Forgetting ratio: {forgetting_ratio:.3f}")
    log.info(f"  Lorenz error: {lorenz_err_p2:.4f}")
    log.info(f"  ||ΔM||: {M_delta_norm:.2f}")

    return {
        "alpha": alpha_p2,
        "sine_err_p1": sine_err_p1,
        "sine_err_p2": sine_err_p2,
        "forgetting_ratio": forgetting_ratio,
        "lorenz_err": lorenz_err_p2,
        "M_delta_norm": M_delta_norm,
        "forgetting_curve": forgetting_curve,
    }


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
    log = logging.getLogger("exp3.3")

    set_seed(SEED)
    device = torch.device(DEVICE)

    log.info("Exp 3.3: Alpha sweep for continual learning")
    log.info(f"Alpha values: {ALPHA_VALUES}")
    log.info(f"Device: {DEVICE}")

    # Generate target data (shared across all runs)
    t_sine = np.arange(PHASE1_STEPS + TEST_STEPS) * DT
    sine_data = sine_target(t_sine, period=SINE_PERIOD, amplitude=SINE_AMP)

    lorenz_data = generate_lorenz(
        duration_ms=PHASE2_MS + TEST_MS + FORGET_TEST_DURATION,
        dt=DT, scale=0.1,
    )

    t_start = time.time()
    results = []
    for alpha in ALPHA_VALUES:
        res = run_one_alpha(alpha, device, sine_data, lorenz_data, log)
        results.append(res)

    elapsed = time.time() - t_start
    log.info(f"\nTotal time: {elapsed:.1f}s")

    # ── Summary table ─────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"{'Alpha':>6} {'Sine P1':>9} {'Sine P2':>9} {'Forget':>8} {'Lorenz':>9} {'||ΔM||':>9}")
    log.info("-" * 55)
    for r in results:
        log.info(f"{r['alpha']:>6.1f} {r['sine_err_p1']:>9.4f} {r['sine_err_p2']:>9.4f} "
                 f"{r['forgetting_ratio']:>8.3f} {r['lorenz_err']:>9.4f} {r['M_delta_norm']:>9.2f}")

    # ── Plots ─────────────────────────────────────────────────────────
    alphas = [r["alpha"] for r in results]
    forget_ratios = [r["forgetting_ratio"] for r in results]
    lorenz_errs = [r["lorenz_err"] for r in results]
    delta_Ms = [r["M_delta_norm"] for r in results]

    # 1. Forgetting ratio vs alpha (bar chart)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c" if fr > 1.5 else "#f39c12" if fr > 1.0 else "#2ecc71" for fr in forget_ratios]
    bars = ax.bar(range(len(alphas)), forget_ratios, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"α={a}" for a in alphas])
    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="No forgetting (ratio=1)")
    ax.set_ylabel("Forgetting Ratio (P2/P1)")
    ax.set_title("Exp 3.3: Forgetting Ratio vs Alpha")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar, fr in zip(bars, forget_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{fr:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/forgetting_ratio_vs_alpha.png", dpi=150)
    plt.close(fig)

    # 2. Overlaid forgetting curves
    fig, ax = plt.subplots(figsize=(14, 5))
    cmap = plt.cm.viridis
    for i, r in enumerate(results):
        fc_steps, fc_errs = zip(*r["forgetting_curve"])
        color = cmap(i / max(len(results) - 1, 1))
        ax.plot(np.array(fc_steps) * DT / 1000, fc_errs, "-", lw=2, color=color,
                label=f"α={r['alpha']}")
    ax.set_xlabel("Lorenz training time (s)")
    ax.set_ylabel("Sine error")
    ax.set_title("Exp 3.3: Forgetting Curves — Sine recall during Lorenz training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/forgetting_curves_overlaid.png", dpi=150)
    plt.close(fig)

    # 3. ||ΔM|| vs alpha
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "#3498db"
    color2 = "#e74c3c"
    ax1.plot(alphas, delta_Ms, "o-", color=color1, lw=2, markersize=8, label="||ΔM||")
    ax1.set_xlabel("Alpha (Phase 2)")
    ax1.set_ylabel("||ΔM|| (Frobenius)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(alphas, forget_ratios, "s--", color=color2, lw=2, markersize=8, label="Forgetting ratio")
    ax2.set_ylabel("Forgetting Ratio", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.axhline(y=1.0, color=color2, linestyle=":", alpha=0.3)

    ax1.set_title("Exp 3.3: M Change and Forgetting vs Alpha")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/delta_M_vs_alpha.png", dpi=150)
    plt.close(fig)

    # 4. Combined: Lorenz error vs alpha
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, lorenz_errs, "o-", color="#9b59b6", lw=2, markersize=8)
    ax.set_xlabel("Alpha (Phase 2)")
    ax.set_ylabel("Lorenz Error")
    ax.set_title("Exp 3.3: Lorenz Learning Quality vs Alpha")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/lorenz_err_vs_alpha.png", dpi=150)
    plt.close(fig)

    # Save results
    np.savez(f"{RESULTS_DIR}/results.npz",
             alphas=np.array(alphas),
             forget_ratios=np.array(forget_ratios),
             lorenz_errs=np.array(lorenz_errs),
             delta_Ms=np.array(delta_Ms))

    log.info(f"\nAll results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
