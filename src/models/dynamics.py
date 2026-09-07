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

    A known mean and covariance determines an affine event probability exactly,
    so the bounds come back as [p, p].
    """

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def value(self):
        """Return mean (representative state), [B, D]"""
        return self.mean

    def _project(self, predicate):
        """Mean and variance of the scalar w·x named by the predicate, each [B]."""
        full_cov = self.var.ndim == self.mean.ndim + 1  # [B, D, D] vs [B, D]
        w = getattr(predicate, "weights", None)

        if w is None:  # axis-aligned event, w = e_dim
            i = predicate.dim
            m = self.mean[..., i]
            v = self.var[..., i, i] if full_cov else self.var[..., i]
        else:
            w = torch.as_tensor(w, dtype=self.mean.dtype, device=self.mean.device)
            m = (self.mean * w).sum(-1)
            if full_cov:
                v = torch.einsum("...i,...ij,...j->...", w, self.var, w)  # wᵀΣw
            else:
                # A per-component variance carries no cross terms; that
                # representation is itself the independence assumption.
                v = (self.var * w**2).sum(-1)

        if v.shape != m.shape:
            # [B, D] and [D, D] have the same rank when B == D, so catch the
            # disagreement here instead of letting it broadcast.
            raise ValueError(
                f"GaussianBelief: mean {tuple(self.mean.shape)} and var "
                f"{tuple(self.var.shape)} disagree; expected mean [B, D] with var "
                f"[B, D] or [B, D, D]."
            )
        return m, v

    def probability_bounds(self, predicate):
        """Exact probability of the predicate's event as [p, p], shape [B, 2]."""
        sense = getattr(predicate, "sense", None)
        if sense not in (">=", "<="):
            raise ValueError(
                f"GaussianBelief evaluates comparison predicates (sense '>=' or "
                f"'<='); it cannot evaluate {predicate}."
            )

        m, v = self._project(predicate)
        if bool((v < 0).any()):
            raise ValueError(
                "GaussianBelief: negative projected variance; the covariance is "
                "not positive semi-definite."
            )

        # Keep the threshold on the belief's own dtype/device so precision,
        # placement and autograd all follow the incoming tensors.
        c = torch.as_tensor(predicate.threshold, dtype=m.dtype, device=m.device)
        margin = (m - c) if sense == ">=" else (c - m)

        # Zero variance is a deterministic, inclusive comparison. Evaluate the
        # CDF on a safe variance so the unused branch cannot send NaN backward.
        positive = v > 0
        v_safe = torch.where(positive, v, torch.ones_like(v))
        p = torch.where(
            positive,
            normal_cdf(margin / torch.sqrt(v_safe)),
            (margin >= 0).to(m.dtype),
        )
        return torch.stack([p, p], dim=-1)  # [B, 2]
