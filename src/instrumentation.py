"""Instrumentation tools for monitoring, visualization, and analysis.

Provides TrainingMonitor for live recording and plotting functions
for eigenspectra, phase portraits, PCA state space, and Lyapunov tracking.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import os


class TrainingMonitor:
    """Records training quantities at regular intervals.

    Args:
        snapshot_interval: Record snapshots every this many steps.
        trace_neurons: Number of random neurons to track.
    """

    def __init__(self, snapshot_interval=1000, trace_neurons=20):
        self.snapshot_interval = snapshot_interval
        self.trace_neurons = trace_neurons

        # Per-timestep records (or every few steps)
        self.z_history = []
        self.f_history = []
        self.error_history = []
        self.t_history = []

        # Snapshot records
        self.M_snapshots = []
        self.alignment_history = []
        self.M_norm_history = []
        self.neuron_traces_x = []
        self.neuron_traces_r = []
        self.snapshot_times = []

        # Bookkeeping
        self._trace_indices = None
        self._step = 0

    def record_step(self, t, z, f, error):
        """Record per-timestep quantities.

        Args:
            t: Current time in ms.
            z: Readout output (numpy or tensor).
            f: Target signal (numpy or tensor).
            error: Readout error scalar.
        """
        self.t_history.append(t)
        self.z_history.append(_to_numpy(z))
        self.f_history.append(_to_numpy(f))
        self.error_history.append(float(error))
        self._step += 1

    def record_snapshot(self, t, network):
        """Record a snapshot of network state.

        Args:
            t: Current time in ms.
            network: PredictiveAlignmentRNN instance.
        """
        import torch
        from . import utils

        if self._trace_indices is None:
            self._trace_indices = np.random.choice(network.N, self.trace_neurons, replace=False)

        self.snapshot_times.append(t)
        self.M_norm_history.append(utils.frobenius_norm(network.M))
        self.alignment_history.append(
            utils.alignment_correlation(network.G, network.M, network.r)
        )

        # Neuron traces
        x_np = network.x.detach().cpu().numpy()
        r_np = network.r.detach().cpu().numpy()
        self.neuron_traces_x.append(x_np[self._trace_indices].copy())
        self.neuron_traces_r.append(r_np[self._trace_indices].copy())

    def get_error_array(self):
        return np.array(self.error_history)

    def get_z_array(self):
        return np.array(self.z_history)

    def get_f_array(self):
        return np.array(self.f_history)


def _to_numpy(x):
    """Convert tensor or scalar to numpy."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().copy()
    return np.asarray(x).copy()


# ── Plotting functions ──────────────────────────────────────────────


def plot_training_output(monitor, save_path=None, title="Training Output"):
    """Plot output vs target during training with error curve.

    Args:
        monitor: TrainingMonitor instance.
        save_path: If given, save figure here.
        title: Figure title.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
    t = np.array(monitor.t_history)
    z = monitor.get_z_array()
    f = monitor.get_f_array()

    # Early training (first 5%)
    n = len(t)
    early = slice(0, n // 20)
    axes[0].plot(t[early], f[early], "k-", label="target", alpha=0.7)
    axes[0].plot(t[early], z[early], "r-", label="output", alpha=0.7)
    axes[0].set_title("Early Training")
    axes[0].legend()
    axes[0].set_ylabel("Amplitude")

    # Late training (last 5%)
    late = slice(-n // 20, None)
    axes[1].plot(t[late], f[late], "k-", label="target", alpha=0.7)
    axes[1].plot(t[late], z[late], "r-", label="output", alpha=0.7)
    axes[1].set_title("Late Training")
    axes[1].legend()
    axes[1].set_ylabel("Amplitude")

    # Error over time (smoothed)
    errors = monitor.get_error_array()
    window = max(1, len(errors) // 500)
    if window > 1:
        smoothed = np.convolve(errors, np.ones(window) / window, mode="valid")
        t_smooth = t[:len(smoothed)]
    else:
        smoothed = errors
        t_smooth = t
    axes[2].plot(t_smooth, smoothed, "b-", alpha=0.7)
    axes[2].set_title("Readout Error")
    axes[2].set_ylabel("Error")
    axes[2].set_xlabel("Time (ms)")

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_eigenspectrum(eigenvalues_list, labels=None, save_path=None, title="Eigenspectrum"):
    """Plot eigenvalues in the complex plane.

    Args:
        eigenvalues_list: List of complex eigenvalue arrays.
        labels: List of labels for each set.
        save_path: If given, save figure here.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(eigenvalues_list)))

    for i, eigs in enumerate(eigenvalues_list):
        label = labels[i] if labels else f"Set {i}"
        ax.scatter(eigs.real, eigs.imag, s=3, alpha=0.5, color=colors[i], label=label)

    # Unit circle for reference
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, label="Unit circle")

    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_phase_portrait(trajectories, labels=None, dims=(0, 1),
                        save_path=None, title="Phase Portrait"):
    """Plot 2D phase portrait of trajectories.

    Args:
        trajectories: List of arrays, each shape (T, K).
        labels: List of labels.
        dims: Tuple of two dimension indices to plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    colors = ["black", "red", "blue", "green"]

    for i, traj in enumerate(trajectories):
        label = labels[i] if labels else f"Traj {i}"
        ax.plot(traj[:, dims[0]], traj[:, dims[1]], alpha=0.6, linewidth=0.5,
                color=colors[i % len(colors)], label=label)

    ax.set_xlabel(f"Dim {dims[0]}")
    ax.set_ylabel(f"Dim {dims[1]}")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_3d_trajectory(trajectories, labels=None, save_path=None, title="3D Trajectory"):
    """Plot 3D trajectories (for Lorenz etc.).

    Args:
        trajectories: List of arrays, each shape (T, 3+).
        labels: List of labels.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["black", "red", "blue", "green"]

    for i, traj in enumerate(trajectories):
        label = labels[i] if labels else f"Traj {i}"
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.5, linewidth=0.3,
                color=colors[i % len(colors)], label=label)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_pca_state_space(activity_matrix, n_components=3, save_path=None,
                         title="PCA State Space"):
    """Project neural activity into PCA space and plot.

    Args:
        activity_matrix: Array of shape (T, N) — neural firing rates over time.
        n_components: Number of PCA components.
        save_path: If given, save figure here.
    """
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(activity_matrix)

    fig = plt.figure(figsize=(10, 8))
    if n_components >= 3:
        ax = fig.add_subplot(111, projection="3d")
        # Color by time
        colors = np.linspace(0, 1, len(projected))
        ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                   c=colors, cmap="viridis", s=0.5, alpha=0.3)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
    else:
        ax = fig.add_subplot(111)
        colors = np.linspace(0, 1, len(projected))
        ax.scatter(projected[:, 0], projected[:, 1], c=colors, cmap="viridis", s=0.5, alpha=0.3)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig, pca


def plot_neuron_traces(monitor, save_path=None, title="Neuron Traces"):
    """Plot tracked neuron activities over time.

    Args:
        monitor: TrainingMonitor instance with recorded neuron traces.
    """
    traces_r = np.array(monitor.neuron_traces_r)  # (n_snapshots, n_neurons)
    times = np.array(monitor.snapshot_times)

    n_neurons = min(10, traces_r.shape[1])
    fig, axes = plt.subplots(n_neurons, 1, figsize=(12, 2 * n_neurons), sharex=True)
    if n_neurons == 1:
        axes = [axes]

    for i in range(n_neurons):
        axes[i].plot(times, traces_r[:, i], "b-", alpha=0.7, linewidth=0.5)
        axes[i].set_ylabel(f"r[{monitor._trace_indices[i]}]", fontsize=8)

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig
