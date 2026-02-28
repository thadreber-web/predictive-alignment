"""Target signal generators for predictive alignment experiments.

All time values are in milliseconds. Targets return numpy arrays
that can be indexed by timestep.
"""

import numpy as np
from scipy.integrate import solve_ivp
import numba


def sine_target(t, period=600.0, amplitude=1.5):
    """Single sine wave target.

    Args:
        t: Time in ms (scalar or array).
        period: Period in ms.
        amplitude: Peak amplitude.
    """
    return amplitude * np.sin(2 * np.pi * t / period)


def multi_sine_target(t, frequencies, amplitudes):
    """Sum of sine waves.

    Args:
        t: Time in ms (scalar or array).
        frequencies: List of frequencies in cycles/ms.
        amplitudes: List of amplitudes.
    """
    return sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(frequencies, amplitudes))


@numba.njit(fastmath=True)
def _lorenz_rk4_sub(n_total, sub_steps, h, sigma, rho, beta, x0, y0, z0):
    """JIT-compiled RK4 integration of Lorenz system with sub-stepping."""
    traj = np.empty((n_total, 3))
    x, y, z = x0, y0, z0
    for i in range(n_total):
        traj[i, 0] = x
        traj[i, 1] = y
        traj[i, 2] = z
        for _ in range(sub_steps):
            dx1 = sigma * (y - x)
            dy1 = x * (rho - z) - y
            dz1 = x * y - beta * z

            x2 = x + 0.5 * h * dx1
            y2 = y + 0.5 * h * dy1
            z2 = z + 0.5 * h * dz1
            dx2 = sigma * (y2 - x2)
            dy2 = x2 * (rho - z2) - y2
            dz2 = x2 * y2 - beta * z2

            x3 = x + 0.5 * h * dx2
            y3 = y + 0.5 * h * dy2
            z3 = z + 0.5 * h * dz2
            dx3 = sigma * (y3 - x3)
            dy3 = x3 * (rho - z3) - y3
            dz3 = x3 * y3 - beta * z3

            x4 = x + h * dx3
            y4 = y + h * dy3
            z4 = z + h * dz3
            dx4 = sigma * (y4 - x4)
            dy4 = x4 * (rho - z4) - y4
            dz4 = x4 * y4 - beta * z4

            x += (h / 6) * (dx1 + 2*dx2 + 2*dx3 + dx4)
            y += (h / 6) * (dy1 + 2*dy2 + 2*dy3 + dy4)
            z += (h / 6) * (dz1 + 2*dz2 + 2*dz3 + dz4)
    return traj


def generate_lorenz(duration_ms, dt=1.0, scale=0.1, sigma=10.0, rho=28.0, beta=8.0/3.0,
                    x0=None, transient_ms=5000.0, lorenz_dt=0.001):
    """Pre-generate Lorenz attractor trajectory using numba-JIT RK4.

    Lorenz time and network time are decoupled: each output point
    (spaced dt ms apart in network time) advances the Lorenz system
    by lorenz_dt seconds. Paper uses lorenz_dt=0.001 so 1ms network
    time = 0.001 Lorenz seconds, giving ~750-1000ms oscillation period.

    Args:
        duration_ms: Total duration in ms (network time).
        dt: Network timestep in ms (controls number of output points).
        scale: Scale factor for output (paper uses 1/10).
        sigma, rho, beta: Lorenz parameters.
        x0: Initial condition [x, y, z]. Defaults to [1, 1, 1].
        transient_ms: Transient to discard in ms (network time).
        lorenz_dt: Lorenz time advance per output point (seconds).

    Returns:
        trajectory: Array of shape (n_steps, 3) scaled by `scale`.
    """
    if x0 is None:
        x0 = [1.0, 1.0, 1.0]

    total_ms = duration_ms + transient_ms
    n_total = int(total_ms / dt)
    # RK4 step size for Lorenz integration (h=0.001 is stable for standard Lorenz)
    h = lorenz_dt
    sub_steps = 1

    traj = _lorenz_rk4_sub(n_total, sub_steps, h, sigma, rho, beta, x0[0], x0[1], x0[2])

    # Discard transient
    skip = int(transient_ms / dt)
    return traj[skip:] * scale


def generate_pendulum(duration_ms, dt=0.1, b=0.5, g=9.81, L=1.0,
                      theta0=2.0, omega0=0.0):
    """Pre-generate damped pendulum trajectory.

    Args:
        duration_ms: Total duration in ms.
        dt: Timestep in ms.
        b: Damping coefficient.
        g: Gravitational acceleration.
        L: Pendulum length.
        theta0: Initial angle in radians.
        omega0: Initial angular velocity.

    Returns:
        trajectory: Array of shape (n_steps, 2) with columns [theta, omega].
    """
    # Pendulum dynamics are in seconds, but our dt is in ms
    dt_s = dt / 1000.0
    duration_s = duration_ms / 1000.0
    t_span = (0, duration_s)
    t_eval = np.arange(0, duration_s, dt_s)

    def pendulum(t, state):
        theta, omega = state
        return [omega, -b * omega - (g / L) * np.sin(theta)]

    sol = solve_ivp(pendulum, t_span, [theta0, omega0], t_eval=t_eval,
                    method="RK45", rtol=1e-8, atol=1e-8)

    return sol.y.T  # shape (n_steps, 2)


def generate_rsg_trial(t_delay, dt=1.0, T_0=60.0, delta=15.0):
    """Generate a single Ready-Set-Go trial (paper eq 17-19).

    Bipolar Gaussian pulses: 2*exp(-(t-center)^2/delta^2) - 1
    Baseline is -1, peak is +1.

    Args:
        t_delay: Delay between pulses in ms.
        dt: Timestep in ms.
        T_0: Uniform time offset for first pulse (paper: 60 ms).
        delta: Gaussian width parameter in ms (paper: 15 ms).

    Returns:
        input_signals: Array of shape (n_steps, 2) — two input channels.
        target_signal: Array of shape (n_steps,) — output target.
        trial_duration_ms: Total trial duration.
    """
    # Paper eq 17-19: s1 at T_0, s2 at T_0+T_delay, output at T_0+2*T_delay
    go_time = T_0 + 2 * t_delay
    trial_duration = go_time + 200.0

    n_steps = int(trial_duration / dt)
    t = np.arange(n_steps) * dt

    # Paper: s(t) = 2*exp(-(t - center)^2 / delta^2) - 1
    input1 = 2.0 * np.exp(-((t - T_0) ** 2) / (delta ** 2)) - 1.0
    input2 = 2.0 * np.exp(-((t - T_0 - t_delay) ** 2) / (delta ** 2)) - 1.0
    target_signal = 2.0 * np.exp(-((t - T_0 - 2 * t_delay) ** 2) / (delta ** 2)) - 1.0

    input_signals = np.stack([input1, input2], axis=1)  # (n_steps, 2)

    return input_signals, target_signal, trial_duration
