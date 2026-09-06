import math

import numpy as np
import torch
from pdstl.base import Belief


def normal_cdf(z):
    """Cumulative distribution function for standard normal distribution"""
    return 0.5 * (1 + torch.erf(z / math.sqrt(2.0)))


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
    """Gaussian belief over the state at one prediction step.

    mean : [B, D]
    var  : [B, D] per-component variances, or [B, D, D] covariance.

    What the returned interval means
    -------------------------------
    For the projected component s = x[dim] ~ N(m, v) and the event {s >= c},
    write z = (m - c) / sqrt(v). With k = confidence_level this returns

        lower = Phi(z - k)      upper = Phi(z + k)      (exact would be Phi(z))

    which is exactly the range of the event probability as the mean varies over
    |m' - m| <= k * sqrt(v) with the variance held fixed -- an ambiguity set of
    means, k standard deviations wide. It is NOT a confidence interval and NOT
    uncertainty about the variance, so a state range of mean +/- k*sigma does
    not on its own justify it: the ambiguity set is the modelling assumption
    being asserted here.

    k = 0 collapses this to the exact probability as a singleton interval.
    """

    def __init__(self, mean, var, confidence_level=2.0):
        self.mean = mean
        self.var = var
        self.confidence_level = confidence_level

    def value(self):
        """Return mean (representative state), [B, D]"""
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
        """Probability that residual >= 0, for a residual scaled by this belief's σ.

        Gaussian-specific helper. Not part of the Belief contract: predicates go
        through probability_bounds(), which states the event explicitly instead
        of relying on the caller to build a residual with the right convention.
        """
        std = torch.sqrt(self.var)
        z = residual / (std)
        return normal_cdf(z)

    def _project(self, dim):
        """Mean and variance of the scalar component x[dim], each [B]."""
        m = self.mean[..., dim]
        if self.var.ndim == self.mean.ndim + 1:  # [B, D, D] covariance
            v = self.var[..., dim, dim]
        else:  # [B, D] per-component variances
            v = self.var[..., dim]
        if v.shape != m.shape:
            # [B, D] and [D, D] are indistinguishable by rank alone when B == D,
            # so catch the disagreement here rather than letting it broadcast
            # into a zero variance and a non-finite probability.
            raise ValueError(
                f"GaussianBelief: mean {tuple(self.mean.shape)} and var "
                f"{tuple(self.var.shape)} disagree; expected mean [B, D] with var "
                f"[B, D] or [B, D, D]."
            )
        return m, v

    def probability_bounds(self, predicate):
        """Lower/upper probability of the predicate's event, [B, 2]."""
        sense = getattr(predicate, "sense", None)
        if sense not in (">=", "<="):
            raise ValueError(
                f"GaussianBelief evaluates comparison predicates (sense '>=' or "
                f"'<='); it cannot evaluate {predicate!r}."
            )

        m, v = self._project(predicate.dim)
        std = torch.sqrt(v)
        # Keep the threshold on the belief's own dtype/device so gradients,
        # precision and placement all follow the incoming tensors.
        c = torch.as_tensor(predicate.threshold, dtype=m.dtype, device=m.device)
        k = self.confidence_level

        if sense == ">=":
            lower = normal_cdf(((m - k * std) - c) / std)
            upper = normal_cdf(((m + k * std) - c) / std)
        else:
            lower = normal_cdf((c - (m + k * std)) / std)
            upper = normal_cdf((c - (m - k * std)) / std)

        return torch.stack([lower, upper], dim=-1)  # [B, 2]
