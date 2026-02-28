"""Experiment 01: Single sine wave — validates core implementation.

N=500, K=1, sine wave period=600ms, train 30s, test 10s autonomous.
"""

import sys
sys.path.insert(0, "/raid/predictive_alignment")

import torch
import numpy as np
import logging
import os
from tqdm import tqdm

from src.network import PredictiveAlignmentRNN
from src.targets import sine_target
from src.instrumentation import TrainingMonitor, plot_training_output, plot_neuron_traces
from src.utils import estimate_lyapunov, compute_eigenspectrum, set_seed

# ── Config ──────────────────────────────────────────────────────────
N = 500
K = 1
PERIOD = 600.0        # ms
AMPLITUDE = 1.5
TAU = 10.0
DT = 1.0              # paper uses dt=1ms (not 0.1ms)
ETA_W = 1e-3
ETA_M = 1e-3          # higher: fewer steps with dt=1ms need stronger learning
ALPHA = 1.0
G = 1.2

TRAIN_MS = 300_000.0  # 300 seconds (500 periods)
TEST_MS = 10_000.0     # 10 seconds

TRAIN_STEPS = int(TRAIN_MS / DT)
TEST_STEPS = int(TEST_MS / DT)

RECORD_EVERY = 10          # record output every 10 steps
SNAPSHOT_EVERY = 5000      # snapshot every 5000 steps

SEED = 42
RESULTS_DIR = "/raid/predictive_alignment/results/01_sine_wave"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Run ─────────────────────────────────────────────────────────────

LOG_FILE = f"{RESULTS_DIR}/experiment.log"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Set up file + console logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("exp01")

    set_seed(SEED)
    log.info(f"Device: {DEVICE}")
    log.info(f"Config: N={N}, K={K}, g={G}, tau={TAU}, dt={DT}")
    log.info(f"  eta_w={ETA_W}, eta_m={ETA_M}, alpha={ALPHA}")
    log.info(f"  train={TRAIN_MS/1000:.0f}s ({TRAIN_STEPS} steps), test={TEST_MS/1000:.0f}s")
    log.info(f"Log file: {LOG_FILE}")
    log.info(f"  Monitor with: tail -f {LOG_FILE}")

    net = PredictiveAlignmentRNN(
        N=N, K=K, g=G, tau=TAU, dt=DT,
        eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
        device=torch.device(DEVICE), seed=SEED,
    )

    monitor = TrainingMonitor(snapshot_interval=SNAPSHOT_EVERY)

    # Lyapunov before training
    lyap_before = estimate_lyapunov(net, n_steps=2000)
    log.info(f"Lyapunov exponent before training: {lyap_before:.4f}")

    # Eigenspectrum before training
    eigs_before = compute_eigenspectrum(net.get_J())

    # ── Training loop ───────────────────────────────────────────────
    log.info(f"Training for {TRAIN_STEPS} steps ({TRAIN_MS/1000:.0f}s simulated)...")
    LOG_INTERVAL = 10000  # log metrics every 10k steps
    recent_errors = []

    for step in tqdm(range(TRAIN_STEPS), desc="Training", mininterval=2.0):
        t = step * DT
        f_val = sine_target(t, period=PERIOD, amplitude=AMPLITUDE)
        target = torch.tensor([f_val], device=DEVICE, dtype=torch.float32)

        z = net.step_and_learn(target)

        if step % RECORD_EVERY == 0:
            error = abs(f_val - z[0].item())
            monitor.record_step(t, z[0].item(), f_val, error)
            recent_errors.append(error)

        if step % SNAPSHOT_EVERY == 0:
            monitor.record_snapshot(t, net)

        if step > 0 and step % LOG_INTERVAL == 0:
            mean_err = np.mean(recent_errors) if recent_errors else 0
            max_x = net.x.abs().max().item()
            pct = 100 * step / TRAIN_STEPS
            log.info(f"  step {step:>7d}/{TRAIN_STEPS} ({pct:5.1f}%) | "
                     f"mean_err={mean_err:.6f} | max|x|={max_x:.2f} | "
                     f"t={t:.1f}ms")
            recent_errors = []

    # ── Test (plasticity off) ───────────────────────────────────────
    log.info(f"Testing for {TEST_STEPS} steps ({TEST_MS/1000:.0f}s, plasticity off)...")
    test_z = []
    test_f = []
    test_t = []

    for step in tqdm(range(TEST_STEPS), desc="Testing", mininterval=2.0):
        t = TRAIN_MS + step * DT
        f_val = sine_target(t, period=PERIOD, amplitude=AMPLITUDE)

        z = net.step()  # no learning
        if step % RECORD_EVERY == 0:
            test_z.append(z[0].item())
            test_f.append(f_val)
            test_t.append(t)

    # Lyapunov after training
    lyap_after = estimate_lyapunov(net, n_steps=2000)
    log.info(f"Lyapunov exponent after training: {lyap_after:.4f}")

    # Eigenspectrum after training
    eigs_after = compute_eigenspectrum(net.get_J())

    # ── Plots ───────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.instrumentation import plot_eigenspectrum

    # 1. Training output
    plot_training_output(monitor, save_path=f"{RESULTS_DIR}/training_output.png",
                         title="Exp 01: Sine Wave Training")

    # 2. Test output (plasticity off)
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    test_t = np.array(test_t)
    test_z = np.array(test_z)
    test_f = np.array(test_f)
    ax.plot(test_t, test_f, "k-", label="target", alpha=0.7)
    ax.plot(test_t, test_z, "r-", label="output (plasticity off)", alpha=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Exp 01: Autonomous Generation (Plasticity Off)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/test_autonomous.png", dpi=150, bbox_inches="tight")
    log.info(f"Saved: {RESULTS_DIR}/test_autonomous.png")
    plt.close(fig)

    # 3. Eigenspectrum before vs after
    plot_eigenspectrum(
        [eigs_before, eigs_after],
        labels=["Before training", "After training"],
        save_path=f"{RESULTS_DIR}/eigenspectrum.png",
        title="Exp 01: Eigenspectrum Before/After Training",
    )

    # 4. Neuron traces
    plot_neuron_traces(monitor, save_path=f"{RESULTS_DIR}/neuron_traces.png",
                       title="Exp 01: Neuron Firing Rates")

    # ── Summary ─────────────────────────────────────────────────────
    errors = monitor.get_error_array()
    early_err = errors[:len(errors)//10].mean()
    late_err = errors[-len(errors)//10:].mean()
    test_rmse = np.sqrt(np.mean((test_z - test_f) ** 2))

    log.info("=== Results ===")
    log.info(f"Lyapunov before: {lyap_before:.4f}")
    log.info(f"Lyapunov after:  {lyap_after:.4f}")
    log.info(f"Early training error (mean): {early_err:.6f}")
    log.info(f"Late training error (mean):  {late_err:.6f}")
    log.info(f"Test RMSE (plasticity off):  {test_rmse:.6f}")
    log.info(f"Plots saved to {RESULTS_DIR}/")

    # Save numeric results
    np.savez(f"{RESULTS_DIR}/results.npz",
             lyap_before=lyap_before, lyap_after=lyap_after,
             early_err=early_err, late_err=late_err, test_rmse=test_rmse,
             eigs_before=eigs_before, eigs_after=eigs_after,
             test_t=test_t, test_z=test_z, test_f=test_f)
    log.info(f"Saved: {RESULTS_DIR}/results.npz")


if __name__ == "__main__":
    main()
