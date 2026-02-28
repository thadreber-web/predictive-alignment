"""Target signal generators for predictive alignment experiments.

All time values are in milliseconds. Targets return numpy arrays
that can be indexed by timestep.
"""

import numpy as np
from scipy.integrate import solve_ivp


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


def generate_lorenz(duration_ms, dt=0.1, scale=0.1, sigma=10.0, rho=28.0, beta=8.0/3.0,
                    x0=None, transient_ms=5000.0):
    """Pre-generate Lorenz attractor trajectory.

    Args:
        duration_ms: Total duration in ms.
        dt: Timestep in ms.
        scale: Scale factor for output (paper uses 1/10).
        sigma, rho, beta: Lorenz parameters.
        x0: Initial condition [x, y, z]. Defaults to [1, 1, 1].
        transient_ms: Time to discard as transient (ms).

    Returns:
        trajectory: Array of shape (n_steps, 3) scaled by `scale`.
    """
    if x0 is None:
        x0 = [1.0, 1.0, 1.0]

    total_ms = duration_ms + transient_ms
    t_span = (0, total_ms)
    t_eval = np.arange(0, total_ms, dt)

    def lorenz(t, state):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    sol = solve_ivp(lorenz, t_span, x0, t_eval=t_eval, method="RK45",
                    rtol=1e-8, atol=1e-8)

    # Discard transient
    skip = int(transient_ms / dt)
    traj = sol.y[:, skip:].T  # shape (n_steps, 3)

    return traj * scale


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


def generate_rsg_trial(t_delay, dt=0.1, pulse_width=10.0, pulse_amplitude=1.0):
    """Generate a single Ready-Set-Go trial.

    Two input pulses separated by t_delay ms. Target is an output pulse
    at t_delay ms after the second input pulse.

    Args:
        t_delay: Delay between pulses in ms.
        dt: Timestep in ms.
        pulse_width: Width of Gaussian pulses in ms (std dev).
        pulse_amplitude: Amplitude of pulses.

    Returns:
        input_signals: Array of shape (n_steps, 2) — two input channels.
        target_signal: Array of shape (n_steps,) — output target.
        trial_duration_ms: Total trial duration.
    """
    # Trial layout: ready at t=100ms, set at t=100+t_delay, go target at t=100+2*t_delay
    # Add 200ms buffer at end
    ready_time = 100.0
    set_time = ready_time + t_delay
    go_time = set_time + t_delay
    trial_duration = go_time + 200.0

    n_steps = int(trial_duration / dt)
    t = np.arange(n_steps) * dt

    # Gaussian pulse helper
    def gaussian_pulse(t, center, width, amp):
        return amp * np.exp(-0.5 * ((t - center) / width) ** 2)

    # Input channel 1: ready pulse
    input1 = gaussian_pulse(t, ready_time, pulse_width, pulse_amplitude)
    # Input channel 2: set pulse
    input2 = gaussian_pulse(t, set_time, pulse_width, pulse_amplitude)

    input_signals = np.stack([input1, input2], axis=1)  # (n_steps, 2)
    target_signal = gaussian_pulse(t, go_time, pulse_width, pulse_amplitude)

    return input_signals, target_signal, trial_duration
