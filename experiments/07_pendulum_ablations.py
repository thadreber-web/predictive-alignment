"""Experiment 07 ablations: pendulum variants.

07a — Single continuous 50s trajectory (no repeats)
07b — Reduced network size (N=100)
07d — Input-driven one-step-ahead predictor (D=2, teacher forcing)
"""

import sys
sys.path.insert(0, "/raid/predictive_alignment")

import torch
import numpy as np
import logging
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from src.network import PredictiveAlignmentRNN
from src.targets import generate_pendulum
from src.instrumentation import plot_phase_portrait
from src.utils import set_seed

# ── Shared pendulum parameters ──────────────────────────────────────
B = 0.5
GRAV = 9.81
L = 1.0
THETA0 = 2.0
OMEGA0 = 0.0
DT = 1.0
TAU = 10.0
ETA_W = 1e-3
ETA_M = 1e-3
ALPHA = 1.0
G = 1.2
RECORD_EVERY = 10
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = "/raid/predictive_alignment/results"


def pendulum_energy(theta, omega):
    return 0.5 * omega ** 2 + (GRAV / L) * (1 - np.cos(theta))


def make_plots(results_dir, label, z_history, f_history, test_z, test_f,
               errors, dt=DT, record_every=RECORD_EVERY):
    """Generate standard pendulum plots."""
    # 1. Training output
    t_plot = np.arange(len(z_history)) * dt
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t_plot, f_history[:, 0], "k-", alpha=0.7, label="target", linewidth=0.8)
    axes[0].plot(t_plot, z_history[:, 0], "r-", alpha=0.7, label="output", linewidth=0.8)
    axes[0].set_ylabel("theta (rad)")
    axes[0].set_title(f"{label}: Training Output")
    axes[0].legend()
    axes[1].plot(t_plot, f_history[:, 1], "k-", alpha=0.7, label="target", linewidth=0.8)
    axes[1].plot(t_plot, z_history[:, 1], "r-", alpha=0.7, label="output", linewidth=0.8)
    axes[1].set_ylabel("omega (rad/s)")
    axes[1].set_xlabel("Time (ms)")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(f"{results_dir}/pendulum_training.png", dpi=150)
    plt.close(fig)

    # 2. Test output
    t_test = np.arange(len(test_z)) * dt
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t_test, test_f[:, 0], "k-", alpha=0.7, label="target", linewidth=0.8)
    axes[0].plot(t_test, test_z[:, 0], "r-", alpha=0.7, label="output (autonomous)", linewidth=0.8)
    axes[0].set_ylabel("theta (rad)")
    axes[0].set_title(f"{label}: Autonomous Test")
    axes[0].legend()
    axes[1].plot(t_test, test_f[:, 1], "k-", alpha=0.7, label="target", linewidth=0.8)
    axes[1].plot(t_test, test_z[:, 1], "r-", alpha=0.7, label="output (autonomous)", linewidth=0.8)
    axes[1].set_ylabel("omega (rad/s)")
    axes[1].set_xlabel("Time (ms)")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(f"{results_dir}/pendulum_test.png", dpi=150)
    plt.close(fig)

    # 3. Phase portraits
    plot_phase_portrait(
        [f_history, z_history],
        labels=["Target", "Output (training)"],
        dims=(0, 1),
        save_path=f"{results_dir}/phase_portrait_train.png",
        title=f"{label}: Phase Portrait (Training)",
    )
    plot_phase_portrait(
        [test_f, test_z],
        labels=["Target", "Output (autonomous)"],
        dims=(0, 1),
        save_path=f"{results_dir}/phase_portrait_test.png",
        title=f"{label}: Phase Portrait (Autonomous)",
    )

    # 4. Energy
    E_target = pendulum_energy(test_f[:, 0], test_f[:, 1])
    E_output = pendulum_energy(test_z[:, 0], test_z[:, 1])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_test, E_target, "k-", label="Target energy", alpha=0.7)
    ax.plot(t_test, E_output, "r-", label="Output energy", alpha=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Energy")
    ax.set_title(f"{label}: Pendulum Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{results_dir}/pendulum_energy.png", dpi=150)
    plt.close(fig)

    # 5. Training error
    fig, ax = plt.subplots(figsize=(12, 4))
    window = max(1, len(errors) // 500)
    smoothed = np.convolve(errors, np.ones(window) / window, mode="valid")
    ax.plot(smoothed, "b-", alpha=0.7, linewidth=0.5)
    ax.set_xlabel("Recording step")
    ax.set_ylabel("Error ||f - z||")
    ax.set_title(f"{label}: Training Error")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{results_dir}/pendulum_error.png", dpi=150)
    plt.close(fig)


def setup_logger(name, log_file):
    """Create a logger that writes to its own file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ═══════════════════════════════════════════════════════════════════
# 07a — Single continuous trajectory (no repeats)
# ═══════════════════════════════════════════════════════════════════
def run_07a():
    results_dir = f"{BASE_DIR}/07a_pendulum_continuous"
    os.makedirs(results_dir, exist_ok=True)
    log = setup_logger("exp07a", f"{results_dir}/experiment.log")

    N = 500
    K = 2
    TRAJ_MS = 50_000.0
    TRAIN_STEPS = int(TRAJ_MS / DT)
    TEST_MS = 50_000.0
    TEST_STEPS = int(TEST_MS / DT)

    log.info(f"07a: N={N}, K={K}, continuous 50s trajectory, {TRAIN_STEPS} train steps")
    set_seed(SEED)

    traj = generate_pendulum(
        duration_ms=TRAJ_MS, dt=DT, b=B, g=GRAV, L=L,
        theta0=THETA0, omega0=OMEGA0,
    )
    log.info(f"Trajectory shape: {traj.shape}")

    net = PredictiveAlignmentRNN(
        N=N, K=K, g=G, tau=TAU, dt=DT,
        eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
        device=torch.device(DEVICE), seed=SEED,
    )

    # Train: single pass through 50s trajectory (no cycling)
    errors = []
    z_history = []
    f_history = []

    for step in tqdm(range(TRAIN_STEPS), desc="07a Train", mininterval=2.0):
        target_np = traj[step]
        target = torch.tensor(target_np, device=DEVICE, dtype=torch.float32)
        z = net.step_and_learn(target)

        if step % RECORD_EVERY == 0:
            errors.append(torch.norm(target - z).item())

        # Record last 5s for plots
        if step >= TRAIN_STEPS - int(5000 / DT):
            z_history.append(z.detach().cpu().numpy())
            f_history.append(target_np)

    z_history = np.array(z_history)
    f_history = np.array(f_history)
    errors = np.array(errors)
    final_err = np.mean(errors[-100:])
    log.info(f"Training done. Final mean error: {final_err:.6f}")

    # Test: 50s autonomous
    test_z = []
    test_traj = generate_pendulum(
        duration_ms=TEST_MS, dt=DT, b=B, g=GRAV, L=L,
        theta0=THETA0, omega0=OMEGA0,
    )
    for step in tqdm(range(TEST_STEPS), desc="07a Test", mininterval=2.0):
        z = net.step()
        test_z.append(z.detach().cpu().numpy())

    test_z = np.array(test_z)
    test_f = test_traj[:TEST_STEPS]

    make_plots(results_dir, "Exp 07a", z_history, f_history, test_z, test_f, errors)
    np.savez(f"{results_dir}/results.npz",
             errors=errors, z_history=z_history, f_history=f_history,
             test_z=test_z, test_f=test_f)
    log.info(f"07a complete. Final error: {final_err:.6f}")
    return final_err


# ═══════════════════════════════════════════════════════════════════
# 07b — Reduced network size (N=100)
# ═══════════════════════════════════════════════════════════════════
def run_07b():
    results_dir = f"{BASE_DIR}/07b_pendulum_small"
    os.makedirs(results_dir, exist_ok=True)
    log = setup_logger("exp07b", f"{results_dir}/experiment.log")

    N = 100
    K = 2
    TRAJ_MS = 5_000.0
    N_REPEATS = 10
    TRAIN_STEPS = int(TRAJ_MS * N_REPEATS / DT)
    TRAJ_STEPS = int(TRAJ_MS / DT)
    TEST_MS = 5_000.0
    TEST_STEPS = int(TEST_MS / DT)

    log.info(f"07b: N={N}, K={K}, 10 repeats of 5s, {TRAIN_STEPS} train steps")
    set_seed(SEED)

    traj = generate_pendulum(
        duration_ms=TRAJ_MS, dt=DT, b=B, g=GRAV, L=L,
        theta0=THETA0, omega0=OMEGA0,
    )
    log.info(f"Trajectory shape: {traj.shape}")

    net = PredictiveAlignmentRNN(
        N=N, K=K, g=G, tau=TAU, dt=DT,
        eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
        device=torch.device(DEVICE), seed=SEED,
    )

    errors = []
    z_history = []
    f_history = []

    for step in tqdm(range(TRAIN_STEPS), desc="07b Train", mininterval=2.0):
        traj_idx = step % TRAJ_STEPS
        target_np = traj[traj_idx]
        target = torch.tensor(target_np, device=DEVICE, dtype=torch.float32)
        z = net.step_and_learn(target)

        if step % RECORD_EVERY == 0:
            errors.append(torch.norm(target - z).item())

        if step >= (N_REPEATS - 1) * TRAJ_STEPS:
            z_history.append(z.detach().cpu().numpy())
            f_history.append(target_np)

    z_history = np.array(z_history)
    f_history = np.array(f_history)
    errors = np.array(errors)
    final_err = np.mean(errors[-100:])
    log.info(f"Training done. Final mean error: {final_err:.6f}")

    # Test
    test_z = []
    test_f_list = []
    for step in tqdm(range(TEST_STEPS), desc="07b Test", mininterval=2.0):
        traj_idx = step % TRAJ_STEPS
        target_np = traj[traj_idx] if traj_idx < len(traj) else traj[-1]
        z = net.step()
        test_z.append(z.detach().cpu().numpy())
        test_f_list.append(target_np)

    test_z = np.array(test_z)
    test_f = np.array(test_f_list)

    make_plots(results_dir, "Exp 07b", z_history, f_history, test_z, test_f, errors)
    np.savez(f"{results_dir}/results.npz",
             errors=errors, z_history=z_history, f_history=f_history,
             test_z=test_z, test_f=test_f)
    log.info(f"07b complete. Final error: {final_err:.6f}")
    return final_err


# ═══════════════════════════════════════════════════════════════════
# 07d — Input-driven (one-step-ahead predictor)
# ═══════════════════════════════════════════════════════════════════
def run_07d():
    results_dir = f"{BASE_DIR}/07d_pendulum_inputdriven"
    os.makedirs(results_dir, exist_ok=True)
    log = setup_logger("exp07d", f"{results_dir}/experiment.log")

    N = 500
    K = 2
    D = 2
    TRAJ_MS = 5_000.0
    N_REPEATS = 10
    TRAIN_STEPS = int(TRAJ_MS * N_REPEATS / DT)
    TRAJ_STEPS = int(TRAJ_MS / DT)
    TEST_MS = 5_000.0
    TEST_STEPS = int(TEST_MS / DT)

    log.info(f"07d: N={N}, K={K}, D={D}, input-driven, {TRAIN_STEPS} train steps")
    set_seed(SEED)

    traj = generate_pendulum(
        duration_ms=TRAJ_MS, dt=DT, b=B, g=GRAV, L=L,
        theta0=THETA0, omega0=OMEGA0,
    )
    log.info(f"Trajectory shape: {traj.shape}")

    net = PredictiveAlignmentRNN(
        N=N, K=K, D=D, g=G, tau=TAU, dt=DT,
        eta_w=ETA_W, eta_m=ETA_M, alpha=ALPHA,
        device=torch.device(DEVICE), seed=SEED,
    )

    errors = []
    z_history = []
    f_history = []

    for step in tqdm(range(TRAIN_STEPS), desc="07d Train", mininterval=2.0):
        traj_idx = step % TRAJ_STEPS
        target_np = traj[traj_idx]
        target = torch.tensor(target_np, device=DEVICE, dtype=torch.float32)

        # Teacher forcing: feed actual target at t-1 as input
        if traj_idx > 0:
            ext_input = torch.tensor(traj[traj_idx - 1], device=DEVICE, dtype=torch.float32)
        else:
            # First step of each repeat: feed zeros
            ext_input = torch.zeros(D, device=DEVICE, dtype=torch.float32)

        z = net.step_and_learn(target, external_input=ext_input)

        if step % RECORD_EVERY == 0:
            errors.append(torch.norm(target - z).item())

        if step >= (N_REPEATS - 1) * TRAJ_STEPS:
            z_history.append(z.detach().cpu().numpy())
            f_history.append(target_np)

    z_history = np.array(z_history)
    f_history = np.array(f_history)
    errors = np.array(errors)
    final_err = np.mean(errors[-100:])
    log.info(f"Training done. Final mean error: {final_err:.6f}")

    # Test: self-feeding (feed network's own output at t-1 as input)
    test_z = []
    test_f_list = []
    prev_z = torch.zeros(K, device=DEVICE, dtype=torch.float32)

    for step in tqdm(range(TEST_STEPS), desc="07d Test", mininterval=2.0):
        traj_idx = step % TRAJ_STEPS
        target_np = traj[traj_idx] if traj_idx < len(traj) else traj[-1]

        z = net.step(external_input=prev_z)
        prev_z = z.detach().clone()

        test_z.append(z.detach().cpu().numpy())
        test_f_list.append(target_np)

    test_z = np.array(test_z)
    test_f = np.array(test_f_list)

    make_plots(results_dir, "Exp 07d", z_history, f_history, test_z, test_f, errors)
    np.savez(f"{results_dir}/results.npz",
             errors=errors, z_history=z_history, f_history=f_history,
             test_z=test_z, test_f=test_f)
    log.info(f"07d complete. Final error: {final_err:.6f}")
    return final_err


# ═══════════════════════════════════════════════════════════════════
# Main — run all three sequentially
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Experiment 07 Ablations: Pendulum Variants")
    print("=" * 60)

    results = {}

    print("\n--- 07a: Continuous trajectory (no repeats) ---")
    results["07a"] = run_07a()

    print("\n--- 07b: Small network (N=100) ---")
    results["07b"] = run_07b()

    print("\n--- 07d: Input-driven (teacher forcing / self-feeding) ---")
    results["07d"] = run_07d()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — Final mean errors:")
    for key, err in sorted(results.items()):
        print(f"  {key}: {err:.6f}")
    print("=" * 60)
    print("Done! Compare with original exp07 error (~0.028)")


if __name__ == "__main__":
    main()
