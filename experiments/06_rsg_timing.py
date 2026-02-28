"""Experiment 06: Ready-Set-Go timing task — temporal memory.

N=1200, K=1, D=2 input channels. Trial-based training.
Delay values: {100, 120, 140, 160} ms, 200,000 trials.
"""

import sys
sys.path.insert(0, "/raid/predictive_alignment")

import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

from src.network import PredictiveAlignmentRNN
from src.targets import generate_rsg_trial
from src.utils import set_seed

# ── Config ──────────────────────────────────────────────────────────
N = 1200
K = 1
D = 2
TAU = 10.0
DT = 1.0
ETA_W = 1e-3
ETA_M = 1e-3
ALPHA = 1.0
G = 1.2

TRAIN_DELAYS = [100, 120, 140, 160]  # ms
N_TRIALS = 200_000
TEST_INTERP_DELAYS = [110, 130, 150]
TEST_EXTRAP_DELAYS = [80, 180, 200]

SEED = 42
RESULTS_DIR = "/raid/predictive_alignment/results/06_rsg_timing"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_trial(net, delay, learn=True):
    """Run a single RSG trial.

    Returns:
        output: Array of shape (n_steps,).
        target: Array of shape (n_steps,).
    """
    input_signals, target_signal, trial_dur = generate_rsg_trial(delay, dt=DT)
    n_steps = len(target_signal)

    outputs = []
    for step in range(n_steps):
        ext_input = torch.tensor(input_signals[step], device=DEVICE, dtype=torch.float32)
        tgt = torch.tensor([target_signal[step]], device=DEVICE, dtype=torch.float32)

        if learn:
            z = net.step_and_learn(tgt, external_input=ext_input)
        else:
            z = net.step(external_input=ext_input)

        outputs.append(z[0].item())

    return np.array(outputs), target_signal


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)

    net = PredictiveAlignmentRNN(
        N=N, K=K, D=D, g=G, tau=TAU, dt=DT,
        eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
        device=torch.device(DEVICE), seed=SEED,
    )

    # ── Training ────────────────────────────────────────────────────
    print(f"Training for {N_TRIALS} trials...")
    trial_errors = []

    for trial in tqdm(range(N_TRIALS), desc="Training", mininterval=5.0):
        delay = np.random.choice(TRAIN_DELAYS)
        output, target = run_trial(net, delay, learn=True)
        trial_errors.append(np.mean((output - target) ** 2))

        # Reset state between trials (partial reset — small perturbation)
        net.x = net.x * 0.1

    trial_errors = np.array(trial_errors)

    # ── Test: trained delays ────────────────────────────────────────
    print("Testing trained delays...")
    trained_outputs = {}
    for delay in TRAIN_DELAYS:
        outputs = []
        for _ in range(10):
            net.x = net.x * 0.1
            out, tgt = run_trial(net, delay, learn=False)
            outputs.append(out)
        trained_outputs[delay] = (np.mean(outputs, axis=0), tgt)

    # ── Test: interpolation ─────────────────────────────────────────
    print("Testing interpolation delays...")
    interp_outputs = {}
    for delay in TEST_INTERP_DELAYS:
        outputs = []
        for _ in range(10):
            net.x = net.x * 0.1
            out, tgt = run_trial(net, delay, learn=False)
            outputs.append(out)
        interp_outputs[delay] = (np.mean(outputs, axis=0), tgt)

    # ── Test: extrapolation ─────────────────────────────────────────
    print("Testing extrapolation delays...")
    extrap_outputs = {}
    for delay in TEST_EXTRAP_DELAYS:
        outputs = []
        for _ in range(10):
            net.x = net.x * 0.1
            out, tgt = run_trial(net, delay, learn=False)
            outputs.append(out)
        extrap_outputs[delay] = (np.mean(outputs, axis=0), tgt)

    # ── PCA of network activity ─────────────────────────────────────
    print("Computing PCA of network activity...")
    pca_activities = {}
    for delay in TRAIN_DELAYS:
        net.x = net.x * 0.1
        input_signals, target_signal, _ = generate_rsg_trial(delay, dt=DT)
        activity = []
        for step in range(len(target_signal)):
            ext_input = torch.tensor(input_signals[step], device=DEVICE, dtype=torch.float32)
            net.step(external_input=ext_input)
            activity.append(net.r.detach().cpu().numpy())
        pca_activities[delay] = np.array(activity)

    # Fit PCA on combined activities
    all_act = np.vstack(list(pca_activities.values()))
    pca = PCA(n_components=3)
    pca.fit(all_act)

    # ── Plots ───────────────────────────────────────────────────────

    # 1. Trained delay outputs
    fig, axes = plt.subplots(len(TRAIN_DELAYS), 1, figsize=(12, 3 * len(TRAIN_DELAYS)), sharex=False)
    for i, delay in enumerate(TRAIN_DELAYS):
        out, tgt = trained_outputs[delay]
        t = np.arange(len(out)) * DT
        axes[i].plot(t, tgt, "k-", label="target", alpha=0.7)
        axes[i].plot(t, out, "r-", label="output", alpha=0.7)
        axes[i].set_title(f"Delay = {delay} ms")
        axes[i].legend(fontsize=8)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Exp 06: Trained Delay Outputs")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/trained_delays.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/trained_delays.png")
    plt.close(fig)

    # 2. Interpolation test
    fig, axes = plt.subplots(len(TEST_INTERP_DELAYS), 1,
                             figsize=(12, 3 * len(TEST_INTERP_DELAYS)), sharex=False)
    for i, delay in enumerate(TEST_INTERP_DELAYS):
        out, tgt = interp_outputs[delay]
        t = np.arange(len(out)) * DT
        axes[i].plot(t, tgt, "k-", label="target", alpha=0.7)
        axes[i].plot(t, out, "b-", label="output (unseen)", alpha=0.7)
        axes[i].set_title(f"Interpolation: Delay = {delay} ms")
        axes[i].legend(fontsize=8)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Exp 06: Interpolation Test")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/interpolation_test.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/interpolation_test.png")
    plt.close(fig)

    # 3. Extrapolation test
    fig, axes = plt.subplots(len(TEST_EXTRAP_DELAYS), 1,
                             figsize=(12, 3 * len(TEST_EXTRAP_DELAYS)), sharex=False)
    for i, delay in enumerate(TEST_EXTRAP_DELAYS):
        out, tgt = extrap_outputs[delay]
        t = np.arange(len(out)) * DT
        axes[i].plot(t, tgt, "k-", label="target", alpha=0.7)
        axes[i].plot(t, out, "m-", label="output (unseen)", alpha=0.7)
        axes[i].set_title(f"Extrapolation: Delay = {delay} ms")
        axes[i].legend(fontsize=8)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Exp 06: Extrapolation Test")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/extrapolation_test.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/extrapolation_test.png")
    plt.close(fig)

    # 4. PCA of network activity
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(TRAIN_DELAYS)))
    for i, delay in enumerate(TRAIN_DELAYS):
        proj = pca.transform(pca_activities[delay])
        ax.plot(proj[:, 0], proj[:, 1], proj[:, 2], alpha=0.5,
                linewidth=0.5, color=colors[i], label=f"{delay} ms")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
    ax.set_title("Exp 06: PCA of Network Activity by Delay")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/pca_by_delay.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/pca_by_delay.png")
    plt.close(fig)

    # 5. Training error curve
    fig, ax = plt.subplots(figsize=(12, 4))
    window = max(1, len(trial_errors) // 500)
    smoothed = np.convolve(trial_errors, np.ones(window) / window, mode="valid")
    ax.plot(smoothed, "b-", alpha=0.7, linewidth=0.5)
    ax.set_xlabel("Trial")
    ax.set_ylabel("MSE")
    ax.set_title("Exp 06: Trial Error During Training")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/trial_error.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/trial_error.png")
    plt.close(fig)

    print(f"\nAll plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
