import numpy as np
import torch
from pdstl.base import Belief


def normal_cdf(z):
    """Cumulative distribution function for standard normal distribution"""
    return 0.5 * (1 + torch.erf(z / torch.sqrt(torch.tensor(2.0))))


def constant_input(t):
    """Control input function u(t)."""
    return -0.5


def sinusoidial_input(t):
    """A sinusoidal control input function u(t)."""
    return 15 * np.sin(1 * np.pi * t)


def noisy_stock_input(t):
    """A noisy stock price-like input function u(t)."""
    np.random.seed(int(t * 100) % 10000)
    drift = 0.01 * t
    noise = 50.0 * np.random.randn()
    jitter = 0.2 * np.random.randn()
    return drift + noise + jitter


def piecewise_input(t):
    """
    Piecewise constant input for STL verification.
    """
    if t < 2:
        return 0.0
    elif t < 4:
        return 20.0
    elif t < 6:
        return -25.0
    elif t < 8:
        return 30.0
    else:
        return -5.0


def linear_system(a, b, g, q, mu, P, t, control_func=constant_input):
    """Propagate the belief state (mu, P) through one time step."""
    mean_trace = np.zeros(len(t))
    var_trace = np.zeros(len(t))

    mean_trace[0] = mu
    var_trace[0] = P
    Q = g**2 + q  # combined process noise covariance
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        u = control_func(t[i - 1])  # control input at time t[i-1]

        Phi = np.exp(a * dt)
        int_u = dt * b * u  # integral of b*u from t[i-1] to t[i]
        mean_trace[i] = Phi * mean_trace[i - 1] + int_u

        # Variance update
        var_trace[i] = (Phi**2) * var_trace[i - 1] + Q * dt
    return mean_trace, var_trace


def piecewise_signal(n_steps=7):
    """
    Discrete piecewise constant signal for STL verification.
    """
    t = np.arange(n_steps, dtype=float)

    default_values = [
        (45, 4),
        (55, 4),
        (60, 4),
        (48, 4),
        (42, 9),
        (58, 4),
        (52, 4),
    ]

    mean_trace = np.array([s[0] for s in default_values], dtype=float)
    var_trace = np.array([s[1] for s in default_values], dtype=float)

    return t, mean_trace, var_trace


class GaussianBelief(Belief):
    def __init__(self, mean, var, confidence_level=2.0):
        self.mean = mean
        self.var = var
        self.confidence_level = confidence_level

    def value(self):
        """Return mean (representative state)"""
        return self.mean

    def lower_bound(self):
        """Conservative lower bound: μ - k*σ"""
        std = torch.sqrt(self.var)
        return self.mean - self.confidence_level * std

    def upper_bound(self):
        """Conservative upper bound: μ + k*σ"""
        std = torch.sqrt(self.var)
        return self.mean + self.confidence_level * std

    def probability_of(self, residual):
        """Probability that residual >= 0"""
        std = torch.sqrt(self.var)
        z = residual / (std)
        return normal_cdf(z)
