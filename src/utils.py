"""Helper utilities: Lyapunov exponent, eigenspectrum, participation ratio."""

import torch
import numpy as np


def estimate_lyapunov(network, n_steps=1000, warmup_steps=500, perturbation=1e-6):
    """Estimate the maximum Lyapunov exponent of the network.

    Follows Asabuki & Clopath (2025) Eq. 23-25:
    - Generate two copies of network state, separated by perturbation
    - Run warmup_steps WITHOUT inputs to eliminate transients
    - Run n_steps WITHOUT inputs, recording log-stretch and renormalizing
    - Return average log-stretch per step

    Args:
        network: PredictiveAlignmentRNN instance.
        n_steps: Number of measurement steps (paper uses 1000).
        warmup_steps: Transient elimination steps (paper uses 500).
        perturbation: Initial perturbation γ₀ (paper uses 1e-6).

    Returns:
        lyapunov: Estimated maximum Lyapunov exponent (per step).
    """
    x_orig = network.x.clone()
    x_save = x_orig.clone()  # save to restore later

    # Perturbed state
    delta = torch.randn_like(x_orig)
    delta = delta / delta.norm() * perturbation
    x_pert = x_orig + delta

    # Phase 1: Warmup — run both copies forward WITHOUT inputs
    # to eliminate transient effects (paper: 500 steps)
    for _ in range(warmup_steps):
        # Step original (no inputs)
        r_orig = torch.tanh(x_orig)
        J = network.G + network.M
        dx = (-x_orig + J @ r_orig) * (network.dt / network.tau)
        x_orig = x_orig + dx

        # Step perturbed (no inputs)
        r_pert = torch.tanh(x_pert)
        dx = (-x_pert + J @ r_pert) * (network.dt / network.tau)
        x_pert = x_pert + dx

        # Renormalize during warmup to prevent blowup
        diff = x_pert - x_orig
        dist = diff.norm().item()
        if dist > 0:
            x_pert = x_orig + diff * (perturbation / dist)

    # Phase 2: Measurement — record log-stretches (paper: 1000 steps)
    log_stretches = []
    for _ in range(n_steps):
        # Step original (no inputs)
        r_orig = torch.tanh(x_orig)
        J = network.G + network.M
        dx = (-x_orig + J @ r_orig) * (network.dt / network.tau)
        x_orig = x_orig + dx

        # Step perturbed (no inputs)
        r_pert = torch.tanh(x_pert)
        dx = (-x_pert + J @ r_pert) * (network.dt / network.tau)
        x_pert = x_pert + dx

        # Measure divergence
        diff = x_pert - x_orig
        dist = diff.norm().item()

        if dist > 0:
            log_stretches.append(np.log(dist / perturbation))
            # Renormalize (Eq. 24)
            x_pert = x_orig + diff * (perturbation / dist)

    # Restore original state
    network.x = x_save
    network.r = torch.tanh(x_save)

    if len(log_stretches) == 0:
        return 0.0

    # Eq. 25: λ = <log(γ_k/γ_0)>_k  (per-step exponent)
    return np.mean(log_stretches)


def compute_eigenspectrum(J):
    """Compute eigenvalues of connectivity matrix J.

    Args:
        J: Tensor of shape (N, N).

    Returns:
        eigenvalues: Complex numpy array of eigenvalues.
    """
    J_np = J.detach().cpu().numpy()
    return np.linalg.eigvals(J_np)


def compute_singular_values(M):
    """Compute singular values of matrix M.

    Args:
        M: Tensor of shape (N, N).

    Returns:
        singular_values: Numpy array of singular values (descending).
    """
    M_np = M.detach().cpu().numpy()
    return np.linalg.svd(M_np, compute_uv=False)


def participation_ratio(eigenvalues):
    """Compute participation ratio from eigenvalues.

    PR = (sum |lambda_i|^2)^2 / sum |lambda_i|^4

    Args:
        eigenvalues: Complex numpy array.

    Returns:
        pr: Participation ratio (1 = single mode, N = uniform).
    """
    mags_sq = np.abs(eigenvalues) ** 2
    return mags_sq.sum() ** 2 / (mags_sq ** 2).sum()


def alignment_correlation(G, M, r):
    """Compute alignment between G@r and M@r.

    Args:
        G, M: Weight matrices (N, N).
        r: Firing rate vector (N,).

    Returns:
        correlation: Scalar alignment value.
    """
    Gr = G @ r
    Mr = M @ r
    return (Gr * Mr).sum().item()


def frobenius_norm(M):
    """Frobenius norm of a matrix."""
    return torch.sqrt((M ** 2).sum()).item()


def set_seed(seed, device=None):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed(seed)
