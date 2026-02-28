"""Core RNN with Predictive Alignment learning rule.

Implements Asabuki & Clopath (2025) "Taming the chaos gently",
Nature Communications 16, 6784.
"""

import torch
import torch.nn as nn
import math


class PredictiveAlignmentRNN:
    """Recurrent neural network trained with predictive alignment.

    Args:
        N: Number of neurons.
        K: Number of readout units.
        D: Input dimension (0 for autonomous tasks).
        g: Gain of fixed connections G (controls chaos strength).
        g_m: Initial gain of plastic connections M.
        p: Sparsity of G (fraction of nonzero entries).
        tau: Membrane time constant in ms.
        dt: Integration timestep in ms.
        eta_w: Learning rate for readout weights.
        eta_m: Learning rate for recurrent weights.
        alpha: Alignment regularization strength.
        device: Torch device.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        N=500,
        K=1,
        D=0,
        g=1.2,
        g_m=0.5,
        p=0.1,
        tau=10.0,
        dt=0.1,
        eta_w=1e-3,
        eta_m=1e-4,
        alpha=1.0,
        device=None,
        seed=42,
    ):
        self.N = N
        self.K = K
        self.D = D
        self.g = g
        self.g_m = g_m
        self.p = p
        self.tau = tau
        self.dt = dt
        self.eta_w = eta_w
        self.eta_m = eta_m
        self.alpha = alpha
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)

        self._init_weights()
        self.reset_state()

    def _init_weights(self):
        """Initialize all weight matrices."""
        N, K, D = self.N, self.K, self.D
        dev = self.device
        rng = self.rng

        # G: sparse fixed connections, std = g / sqrt(p * N)
        std_g = self.g / math.sqrt(self.p * N)
        G = torch.randn(N, N, device=dev, generator=rng) * std_g
        mask = torch.rand(N, N, device=dev, generator=rng) < self.p
        self.G = G * mask.float()
        self.G.fill_diagonal_(0.0)  # no self-connections

        # M: dense plastic connections, std = g_m / sqrt(N)
        std_m = self.g_m / math.sqrt(N)
        self.M = torch.randn(N, N, device=dev, generator=rng) * std_m

        # w: readout weights, initialized to zeros
        self.w = torch.zeros(K, N, device=dev)

        # Q: fixed random feedback, uniform [-3/sqrt(K), 3/sqrt(K)]
        q_bound = 3.0 / math.sqrt(K)
        self.Q = torch.empty(N, K, device=dev).uniform_(-q_bound, q_bound)

        # W_in: input weights (only if D > 0)
        if D > 0:
            self.W_in = torch.randn(N, D, device=dev, generator=rng) * (1.0 / math.sqrt(D))
        else:
            self.W_in = None

    def reset_state(self):
        """Reset membrane potentials to small random values."""
        self.x = torch.randn(self.N, device=self.device, generator=self.rng) * 0.1
        self.r = torch.tanh(self.x)
        self.z = self.w @ self.r

    def step(self, external_input=None):
        """Run one forward timestep (no learning).

        Args:
            external_input: Optional tensor of shape (D,).

        Returns:
            z: Readout output of shape (K,).
        """
        self.r = torch.tanh(self.x)
        J = self.G + self.M
        current = J @ self.r
        if external_input is not None and self.W_in is not None:
            current = current + self.W_in @ external_input
        dx = (-self.x + current) * (self.dt / self.tau)
        self.x = self.x + dx
        self.z = self.w @ self.r
        return self.z

    def step_and_learn(self, target, external_input=None):
        """Run one forward timestep and apply learning updates.

        Args:
            target: Target signal tensor of shape (K,).
            external_input: Optional tensor of shape (D,).

        Returns:
            z: Readout output of shape (K,).
        """
        # Forward step
        z = self.step(external_input)

        # Readout weight update (delta rule)
        output_error = target - z
        self.w = self.w + self.eta_w * torch.outer(output_error, self.r)

        # Recurrent weight update (predictive alignment)
        feedback = self.Q @ z                          # (N,)
        J_hat_r = (self.M - self.alpha * self.G) @ self.r  # (N,)
        rec_error = feedback - J_hat_r                 # (N,)
        self.M = self.M + self.eta_m * torch.outer(rec_error, self.r)

        return z

    def get_J(self):
        """Return total connectivity matrix J = G + M."""
        return self.G + self.M

    def state_dict(self):
        """Return all network parameters as a dict."""
        d = {
            "x": self.x.clone(),
            "G": self.G.clone(),
            "M": self.M.clone(),
            "w": self.w.clone(),
            "Q": self.Q.clone(),
        }
        if self.W_in is not None:
            d["W_in"] = self.W_in.clone()
        return d

    def load_state_dict(self, d):
        """Load network parameters from a dict."""
        self.x = d["x"].to(self.device)
        self.G = d["G"].to(self.device)
        self.M = d["M"].to(self.device)
        self.w = d["w"].to(self.device)
        self.Q = d["Q"].to(self.device)
        if "W_in" in d and self.W_in is not None:
            self.W_in = d["W_in"].to(self.device)
        self.r = torch.tanh(self.x)
        self.z = self.w @ self.r
