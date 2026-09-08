import math

import numpy as np
import torch
import torch.nn as nn

from pdstl.base import Belief, BeliefTrajectory


# Controlled dynamics


class Dynamics(nn.Module):
    """Base class for controlled dynamics."""

    def __init__(self, dt, u_max, device="cpu"):
        super().__init__()
        self.dt = dt
        self.u_max = u_max
        self.device = device

    def bound_control(self, v):
        """Smoothly bound an unconstrained control to ``[-u_max, u_max]``."""
        return self.u_max * torch.tanh(v)

    def step(self, x, P, u):
        """
        Propagates state x and covariance P one step forward with control u.
        x: [Dim]
        P: [Dim, Dim]
        u: [Control Dim]
        Returns: (x_next, P_next)
        """
        raise NotImplementedError

    def forward(self, v_sequence, x0_mean, x0_cov):
        raise NotImplementedError


class SingleIntegrator(Dynamics):
    """
    Standard Position-Velocity model defined in the PDF.

    State:   [x, y]
    Control: [vx, vy]
    """

    def __init__(self, dt=0.2, u_max=1.0, q_std=0.05, device="cpu"):
        super().__init__(dt, u_max, device)

        # Process Noise Covariance Q (Additive)
        # We assume diagonal noise for simplicity: Q = diag(q_std^2)
        self.Q = torch.eye(2, device=self.device) * q_std**2

    def step(self, x, P, u):
        # x_next = x + u * dt
        # P_next = P + Q
        return x + u * self.dt, P + self.Q

    def forward(self, v_sequence, x0_mean, x0_cov):
        """
        Rolls out the trajectory from t=0 to T.

        Args:
            v_sequence: Tensor [T, 2] (Unconstrained controls)
            x0_mean:    Tensor [2]    (Initial position)
            x0_cov:     Tensor [2, 2] (Initial uncertainty)

        Returns:
            mean_stack: [1, T+1, 2]
            cov_stack:  [1, T+1, 2, 2]
        """
        T = v_sequence.shape[0]

        # Storage for the trajectory
        means = [x0_mean]
        covs = [x0_cov]

        curr_mu = x0_mean
        curr_sigma = x0_cov

        for t in range(T):
            # 1. Squash the optimization variable to get physical control
            u = self.bound_control(v_sequence[t])

            # 2. Update Mean (Differentiable)
            curr_mu = curr_mu + u * self.dt

            # 3. Update Covariance (Open Loop Uncertainty Growth)
            curr_sigma = curr_sigma + self.Q

            means.append(curr_mu)
            covs.append(curr_sigma)

        # Stack results into tensors
        # Output shape: [Batch=1, Time, Dim]
        mean_stack = torch.stack(means).unsqueeze(0)
        cov_stack = torch.stack(covs).unsqueeze(0)

        return mean_stack, cov_stack


class DoubleIntegrator(Dynamics):
    """
    Alternative Physics-based model (Acceleration control).

    State:   [px, py, vx, vy]
    Control: [ax, ay]
    """

    def __init__(self, dt=0.2, u_max=1.0, q_std=0.02, device="cpu"):
        super().__init__(dt, u_max, device)

        # State Transition Matrix A
        self.A = torch.tensor(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
        )

        # Control Matrix B
        self.B = torch.tensor(
            [[0.5 * dt**2, 0.0], [0.0, 0.5 * dt**2], [dt, 0.0], [0.0, dt]],
            device=device,
        )

        # Process Noise Q
        self.Q = torch.eye(4, device=device) * q_std**2

    def step(self, x, P, u):
        # x_next = A x + B u
        # P_next = A P A^T + Q
        x_next = self.A @ x + self.B @ u
        P_next = self.A @ P @ self.A.t() + self.Q
        return x_next, P_next

    def forward(self, v_sequence, x0_mean, x0_cov):
        T = v_sequence.shape[0]
        means = [x0_mean]
        covs = [x0_cov]

        curr_mu = x0_mean
        curr_sigma = x0_cov

        for t in range(T):
            # 1. Bound Control
            u = self.bound_control(v_sequence[t])

            # 2. Update Mean
            curr_mu = self.A @ curr_mu + self.B @ u

            # 3. Update Covariance (Full Linear Update)
            curr_sigma = self.A @ curr_sigma @ self.A.t() + self.Q

            means.append(curr_mu)
            covs.append(curr_sigma)

        mean_stack = torch.stack(means).unsqueeze(0)
        cov_stack = torch.stack(covs).unsqueeze(0)

        return mean_stack, cov_stack


# Offline scalar model


def sinusoidal_input(t):
    """Sinusoidal scalar control used by the original pdSTL example."""
    return 15.0 * np.sin(np.pi * t)


def linear_system(a, b, g, q, mu, P, t, control_func):
    """Propagate the scalar Gaussian model used in the offline examples."""
    t = np.asarray(t, dtype=float)
    mean_trace = np.zeros(len(t), dtype=float)
    var_trace = np.zeros(len(t), dtype=float)
    mean_trace[0], var_trace[0] = mu, P
    process_variance = g**2 + q

    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        transition = np.exp(a * dt)
        mean_trace[i] = transition * mean_trace[i - 1] + dt * b * control_func(t[i - 1])
        var_trace[i] = transition**2 * var_trace[i - 1] + process_variance * dt
    return mean_trace, var_trace


def piecewise_signal(values=None):
    """Return a configurable discrete scalar mean/variance signal."""
    if values is None:
        values = ((45, 4), (55, 4), (60, 4), (48, 4), (42, 9), (58, 4), (52, 4))
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("piecewise values must have shape [T, 2] as (mean, variance)")
    return np.arange(len(values), dtype=float), values[:, 0], values[:, 1]


def create_signal_trace(config):
    """Create ``(time, mean, variance)`` from a signal configuration."""
    signal_type = config.get("type")

    if signal_type == "linear":
        if config.get("control") != "sinusoidal":
            raise ValueError("linear signal control must be 'sinusoidal'")
        parameters = config["parameters"]
        time = np.linspace(0.0, parameters["t_end"], parameters["n_steps"])
        mean, variance = linear_system(
            **{
                name: parameters[name]
                for name in ("a", "b", "g", "q", "mu", "P")
            },
            t=time,
            control_func=sinusoidal_input,
        )
        return time, mean, variance

    if signal_type == "piecewise":
        return piecewise_signal(config["values"])

    raise ValueError("signal type must be 'linear' or 'piecewise'")


# Gaussian belief


class GaussianBelief(Belief):
    """Gaussian belief with sigma-displaced probability endpoints."""

    def __init__(self, mean, variance, sigma_multiplier):
        sigma_multiplier = float(sigma_multiplier)
        if not math.isfinite(sigma_multiplier) or sigma_multiplier < 0:
            raise ValueError("sigma_multiplier must be finite and non-negative")
        self.mean = mean
        self.var = variance
        self.sigma_multiplier = sigma_multiplier
        self._validate_shapes()

    def _validate_shapes(self):
        if not torch.is_tensor(self.mean) or self.mean.ndim != 2:
            raise ValueError("GaussianBelief mean must have shape [B,D]")
        if not torch.is_tensor(self.var):
            raise ValueError("GaussianBelief variance must be a tensor")

        batch, state_dim = self.mean.shape
        if self.var.ndim == 2:
            if self.var.shape != self.mean.shape:
                raise ValueError(
                    "GaussianBelief diagonal variance must match mean shape [B,D]"
                )
            component_variance = self.var
        elif self.var.ndim == 3:
            if self.var.shape[0] != batch:
                raise ValueError(
                    "GaussianBelief mean and covariance batch dimensions differ"
                )
            if self.var.shape[1] != self.var.shape[2]:
                raise ValueError("GaussianBelief covariance matrices must be square")
            if self.var.shape[1] != state_dim:
                raise ValueError(
                    "GaussianBelief covariance state dimension must match mean"
                )
            component_variance = self.var.diagonal(dim1=-2, dim2=-1)
        else:
            raise ValueError("GaussianBelief variance must be [B,D] or [B,D,D]")

        if bool((component_variance < 0).any()):
            raise ValueError("GaussianBelief variance must be non-negative")

    def value(self):
        return self.mean

    def _component_variance(self):
        if self.var.ndim == 2:
            return self.var
        return self.var.diagonal(dim1=-2, dim2=-1)

    def lower_bound(self):
        return self.mean - self.sigma_multiplier * torch.sqrt(
            self._component_variance()
        )

    def upper_bound(self):
        return self.mean + self.sigma_multiplier * torch.sqrt(
            self._component_variance()
        )

    def _project(self, predicate):
        full_covariance = self.var.ndim == 3
        dim = getattr(predicate, "dim", 0)
        if not isinstance(dim, int) or not 0 <= dim < self.mean.shape[1]:
            raise ValueError(f"predicate dimension {dim} is outside the state")
        mean = self.mean[:, dim]
        variance = self.var[:, dim, dim] if full_covariance else self.var[:, dim]
        return mean, variance

    def probability_bounds(self, predicate):
        """Return an inclusive comparison probability interval, shaped [B,2]."""
        sense = getattr(predicate, "sense", None)
        if sense not in (">=", "<="):
            raise ValueError(f"GaussianBelief cannot evaluate {predicate}")
        mean, variance = self._project(predicate)
        threshold = torch.as_tensor(
            predicate.threshold, dtype=mean.dtype, device=mean.device
        )
        positive = variance > 0
        sigma = torch.sqrt(variance)
        safe_sigma = torch.where(positive, sigma, torch.ones_like(sigma))
        displacement = self.sigma_multiplier * sigma
        lower_state = mean - displacement
        upper_state = mean + displacement
        if sense == ">=":
            lower = torch.special.ndtr((lower_state - threshold) / safe_sigma)
            upper = torch.special.ndtr((upper_state - threshold) / safe_sigma)
            deterministic = mean >= threshold
        else:
            lower = torch.special.ndtr((threshold - upper_state) / safe_sigma)
            upper = torch.special.ndtr((threshold - lower_state) / safe_sigma)
            deterministic = mean <= threshold
        lower = torch.where(positive, lower, deterministic.to(mean.dtype))
        upper = torch.where(positive, upper, deterministic.to(mean.dtype))
        return torch.stack((lower, upper), dim=-1)


def create_gaussian_belief_trajectory(
    mean_trace, variance_trace, sigma_multiplier, dtype=None, device=None
):
    """Build a trajectory from scalar, vector, or batched Gaussian traces."""
    mean = torch.as_tensor(mean_trace, dtype=dtype, device=device)
    variance = torch.as_tensor(variance_trace, dtype=dtype, device=device)

    if mean.ndim == 1:
        if variance.shape != mean.shape:
            raise ValueError("scalar trace variance must exactly match mean shape [T]")
        mean = mean.unsqueeze(-1)
        variance = variance.unsqueeze(-1)
    if mean.ndim == 2:
        diagonal = variance.ndim == 2 and variance.shape == mean.shape
        covariance = variance.ndim == 3 and variance.shape == (
            mean.shape[0],
            mean.shape[1],
            mean.shape[1],
        )
        if not (diagonal or covariance):
            raise ValueError(
                "vector trace variance must exactly match [T,D] or [T,D,D]"
            )
        beliefs = [
            GaussianBelief(
                mean[t : t + 1], variance[t : t + 1], sigma_multiplier
            )
            for t in range(mean.shape[0])
        ]
    elif mean.ndim == 3:
        diagonal = variance.ndim == 3 and variance.shape == mean.shape
        covariance = variance.ndim == 4 and variance.shape == (
            mean.shape[0],
            mean.shape[1],
            mean.shape[2],
            mean.shape[2],
        )
        if not (diagonal or covariance):
            raise ValueError(
                "batched trace variance must exactly match [B,T,D] or [B,T,D,D]"
            )
        beliefs = [
            GaussianBelief(mean[:, t], variance[:, t], sigma_multiplier)
            for t in range(mean.shape[1])
        ]
    else:
        raise ValueError("mean trace must have shape [T], [T,D], or [B,T,D]")
    return BeliefTrajectory(beliefs)
