"""Dynamics and Gaussian belief propagation used by the demos in ``main.py``.

Two propagators live here:

``linear_system``
    A scalar (1-D) numpy propagator for the introductory signal examples --
    no autograd, no control optimization.
``linear_gaussian_rollout``
    A differentiable torch propagator for the whole-pipeline verification,
    in explicit ``mu_{k+1} = A mu_k + B u_k``,
    ``Sigma_{k+1} = A Sigma_k A^T + Q`` form.

Linear-Gaussian dynamics are used because they give transparent, exact
mean/covariance propagation, which is what makes a verification example
checkable by hand. **pdSTL itself requires none of this**: it consumes atomic
event probabilities from whatever provider produces them, and assumes nothing
about linearity, Gaussianity, or mean/covariance beliefs.
"""

import numpy as np
import torch


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


def bound_controls(v, u_max):
    """Map unconstrained parameters ``v`` to controls in ``[-u_max, u_max]``.

    ``u = u_max * tanh(v)``. This is an *optimizer/application* choice -- a way
    to impose a box constraint on the controls without a projection step -- and
    is deliberately kept out of :func:`linear_gaussian_rollout` so that the
    dynamics stay a plain linear map. It is not part of the pdSTL semantics in
    any sense.
    """
    return u_max * torch.tanh(v)


def linear_gaussian_rollout(controls, x0_mean, x0_cov, A, B, Q):
    """Propagate a Gaussian belief through ``x_{k+1} = A x_k + B u_k + w_k``.

    For open-loop controls the mean and covariance recursions decouple::

        mu_{k+1}    = A mu_k + B u_k
        Sigma_{k+1} = A Sigma_k A^T + Q

    so no sampling is involved and the result is exact. Every operation is a
    differentiable torch op, so gradients flow from the returned belief back to
    ``controls`` (and to whatever produced them).

    Parameters
    ----------
    controls : torch.Tensor
        Shape ``[N, M]``, already in physical units -- apply
        :func:`bound_controls` first if the controls are box-constrained.
    x0_mean : torch.Tensor
        Shape ``[D]``.
    x0_cov : torch.Tensor
        Shape ``[D, D]``. May be exactly zero for a known initial state.
    A : torch.Tensor
        Shape ``[D, D]``.
    B : torch.Tensor
        Shape ``[D, M]``.
    Q : torch.Tensor
        Shape ``[D, D]``, the per-step process-noise covariance.

    Returns
    -------
    mean : torch.Tensor
        Shape ``[1, N+1, D]``.
    covariance : torch.Tensor
        Shape ``[1, N+1, D, D]``.

    Notes
    -----
    The leading singleton batch axis matches the ``[B, T, D]`` /
    ``[B, T, D, D]`` contract of
    :func:`pdstl.gaussian.gaussian_atom_traces`, so the result can be handed
    straight to the atomic probability provider.
    """
    mean_k = x0_mean
    cov_k = x0_cov
    means = [mean_k]
    covs = [cov_k]

    for k in range(controls.shape[0]):
        mean_k = A @ mean_k + B @ controls[k]
        cov_k = A @ cov_k @ A.T + Q
        means.append(mean_k)
        covs.append(cov_k)

    return torch.stack(means, dim=0).unsqueeze(0), torch.stack(covs, dim=0).unsqueeze(0)


class GaussianBelief:
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
