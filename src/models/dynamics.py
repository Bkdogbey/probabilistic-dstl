import numpy as np
import torch
import torch.nn as nn

from pdstl.base import Belief, BeliefTrajectory


# Controlled dynamics
#
# Dynamics defines one linear step, one control convention, and one rollout;
# concrete models only set A, B, Q. The same step also propagates a state
# enclosure [L, U] instead of a point mean -- see step_enclosure.


class Dynamics(nn.Module):
    """Linear controlled dynamics: x' = A x + B u,  P' = A P A^T + Q.

    Subclasses register ``A``, ``B``, ``Q`` as buffers in ``__init__``. ``Q``
    is the process noise added over one discrete step (no ``dt`` factor).
    """

    def __init__(self, dt, u_max, device="cpu"):
        super().__init__()
        self.dt = dt
        self.u_max = u_max
        self.device = device

    def bound_control(self, v):
        """Smoothly bound an unconstrained control to ``[-u_max, u_max]``.

        The one control convention: every physical control, inside a rollout
        or read externally for a cost, goes through this.
        """
        return self.u_max * torch.tanh(v)

    def step(self, x, P, u):
        """One step of the mean/covariance. x: [D], P: [D,D], u: [control dim]."""
        x_next = self.A @ x + self.B @ u
        P_next = self.A @ P @ self.A.t() + self.Q
        return x_next, P_next

    def forward(self, v_sequence, x0_mean, x0_cov):
        """Roll out mean and covariance for an unconstrained control sequence.

        v_sequence: [T, control dim]   x0_mean: [D]   x0_cov: [D,D]
        Returns mean [1,T+1,D], covariance [1,T+1,D,D].
        """
        means, covs = [x0_mean], [x0_cov]
        for v in v_sequence:
            mean, cov = self.step(means[-1], covs[-1], self.bound_control(v))
            means.append(mean)
            covs.append(cov)
        return torch.stack(means).unsqueeze(0), torch.stack(covs).unsqueeze(0)

    def step_enclosure(self, lower, upper, cov, u, d_lower, d_upper):
        """One step of a state enclosure [lower, upper] plus covariance.

        A is split into its non-negative and non-positive parts so the bound
        propagation stays sound under any sign of A's entries:

            L' = A+ L + A- U + B u + d_lower
            U' = A- L + A+ U + B u + d_upper
            P' = A P A^T + Q

        d_lower, d_upper bound an unknown deterministic offset for this step.
        lower, upper: [D]   cov: [D,D]   u: [control dim]
        """
        A_pos, A_neg = self.A.clamp(min=0), self.A.clamp(max=0)
        Bu = self.B @ u
        next_lower = A_pos @ lower + A_neg @ upper + Bu + d_lower
        next_upper = A_neg @ lower + A_pos @ upper + Bu + d_upper
        next_cov = self.A @ cov @ self.A.t() + self.Q
        return next_lower, next_upper, next_cov

    def rollout_enclosure(self, v_sequence, lower0, upper0, cov0, d_lower=None, d_upper=None):
        """Roll out an enclosure and covariance for an unconstrained control sequence.

        d_lower, d_upper: [D], default zero (no unknown offset).
        Returns lower [1,T+1,D], upper [1,T+1,D], covariance [1,T+1,D,D].
        """
        if d_lower is None:
            d_lower = torch.zeros_like(lower0)
        if d_upper is None:
            d_upper = torch.zeros_like(upper0)

        lowers, uppers, covs = [lower0], [upper0], [cov0]
        for v in v_sequence:
            lower, upper, cov = self.step_enclosure(
                lowers[-1], uppers[-1], covs[-1], self.bound_control(v), d_lower, d_upper
            )
            lowers.append(lower)
            uppers.append(upper)
            covs.append(cov)
        return (
            torch.stack(lowers).unsqueeze(0),
            torch.stack(uppers).unsqueeze(0),
            torch.stack(covs).unsqueeze(0),
        )


class SingleIntegrator(Dynamics):
    """Position-velocity model. State: [x, y]. Control: [vx, vy]."""

    def __init__(self, dt=0.2, u_max=1.0, q_std=0.05, device="cpu"):
        super().__init__(dt, u_max, device)
        self.register_buffer("A", torch.eye(2, device=device))
        self.register_buffer("B", dt * torch.eye(2, device=device))
        self.register_buffer("Q", torch.eye(2, device=device) * q_std**2)


class DoubleIntegrator(Dynamics):
    """Acceleration-controlled model. State: [px, py, vx, vy]. Control: [ax, ay]."""

    def __init__(self, dt=0.2, u_max=1.0, q_std=0.02, device="cpu"):
        super().__init__(dt, u_max, device)
        self.register_buffer(
            "A",
            torch.tensor(
                [
                    [1.0, 0.0, dt, 0.0],
                    [0.0, 1.0, 0.0, dt],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
            ),
        )
        self.register_buffer(
            "B",
            torch.tensor(
                [[0.5 * dt**2, 0.0], [0.0, 0.5 * dt**2], [dt, 0.0], [0.0, dt]],
                device=device,
            ),
        )
        self.register_buffer("Q", torch.eye(4, device=device) * q_std**2)


# Offline scalar model


def piecewise_signal(values=None):
    """Return a configurable discrete scalar mean/variance signal."""
    if values is None:
        values = ((45, 4), (55, 4), (60, 4), (48, 4), (42, 9), (58, 4), (52, 4))
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("piecewise values must have shape [T, 2] as (mean, variance)")
    return np.arange(len(values), dtype=float), values[:, 0], values[:, 1]


# Gaussian belief
#
#   X = x* + E,   x* in [lower, upper],   E ~ N(0, covariance)
#
# The location enclosure and the residual covariance are separate,
# independently supplied uncertainties -- one is never derived from the other.


class GaussianBelief(Belief):
    """A location known only within [lower, upper], plus Gaussian residual noise.

    lower, upper : [B, D]
        Admissible bounds on the location, supplied directly.
    covariance : [B, D] (diagonal) or [B, D, D] (full)
        Covariance of the residual. Only its diagonal is used; see _project.
    """

    def __init__(self, lower, upper, covariance):
        self.lower = lower
        self.upper = upper
        self.covariance = covariance
        self._validate_shapes()

    def _validate_shapes(self):
        if not torch.is_tensor(self.lower) or self.lower.ndim != 2:
            raise ValueError("GaussianBelief lower must have shape [B,D]")
        if not torch.isfinite(self.lower).all():
            raise ValueError("GaussianBelief lower must be finite")
        if not torch.is_tensor(self.upper) or self.upper.shape != self.lower.shape:
            raise ValueError("GaussianBelief upper must match lower's shape")
        if not torch.isfinite(self.upper).all():
            raise ValueError("GaussianBelief upper must be finite")
        if bool((self.lower > self.upper).any()):
            raise ValueError("GaussianBelief requires lower <= upper")

        batch, state_dim = self.lower.shape
        if not torch.is_tensor(self.covariance):
            raise ValueError("GaussianBelief covariance must be a tensor")
        if not torch.isfinite(self.covariance).all():
            raise ValueError("GaussianBelief covariance must be finite")
        if self.covariance.ndim == 2:
            if self.covariance.shape != self.lower.shape:
                raise ValueError("diagonal covariance must match lower's shape [B,D]")
            component_variance = self.covariance
        elif self.covariance.ndim == 3:
            if self.covariance.shape != (batch, state_dim, state_dim):
                raise ValueError("full covariance must have shape [B,D,D]")
            # A @ P @ A^T accumulates float roundoff, so this is a tolerance
            # check, not exact equality.
            if not torch.allclose(
                self.covariance, self.covariance.transpose(-1, -2), atol=1e-5
            ):
                raise ValueError("full covariance must be symmetric")
            eigenvalues = torch.linalg.eigvalsh(self.covariance)
            if bool((eigenvalues < -1e-6).any()):
                raise ValueError("full covariance must be positive semi-definite")
            component_variance = self.covariance.diagonal(dim1=-2, dim2=-1)
        else:
            raise ValueError("GaussianBelief covariance must be [B,D] or [B,D,D]")

        if bool((component_variance < 0).any()):
            raise ValueError("GaussianBelief covariance must be non-negative")

    def value(self):
        """Midpoint of [lower, upper], [B,D] -- a representative descriptor,
        not a substitute for enclosure-based predicate evaluation."""
        return (self.lower + self.upper) / 2

    def _project(self, predicate):
        full_covariance = self.covariance.ndim == 3
        dim = getattr(predicate, "dim", 0)
        if not isinstance(dim, int) or not 0 <= dim < self.lower.shape[1]:
            raise ValueError(f"predicate dimension {dim} is outside the state")
        lower = self.lower[:, dim]
        upper = self.upper[:, dim]
        variance = self.covariance[:, dim, dim] if full_covariance else self.covariance[:, dim]
        return lower, upper, variance

    def probability_bounds(self, predicate):
        """Enclose the predicate's probability over every admissible location, [B,2].

        For X >= c the tail probability increases with the location; for
        X <= c it decreases. Either way the extrema sit at the enclosure
        endpoints, so the same CDF evaluated there brackets every value the
        map takes in between.
        """
        sense = getattr(predicate, "sense", None)
        if sense not in (">=", "<="):
            raise ValueError(f"GaussianBelief cannot evaluate {predicate}")

        lower, upper, variance = self._project(predicate)
        threshold = torch.as_tensor(
            predicate.threshold, dtype=lower.dtype, device=lower.device
        )

        positive = variance > 0
        sigma = torch.sqrt(variance)
        safe_sigma = torch.where(positive, sigma, torch.ones_like(sigma))

        if sense == ">=":
            bound_lower = torch.special.ndtr((lower - threshold) / safe_sigma)
            bound_upper = torch.special.ndtr((upper - threshold) / safe_sigma)
            deterministic_lower = lower >= threshold
            deterministic_upper = upper >= threshold
        else:
            bound_lower = torch.special.ndtr((threshold - upper) / safe_sigma)
            bound_upper = torch.special.ndtr((threshold - lower) / safe_sigma)
            deterministic_lower = upper <= threshold
            deterministic_upper = lower <= threshold

        bound_lower = torch.where(positive, bound_lower, deterministic_lower.to(lower.dtype))
        bound_upper = torch.where(positive, bound_upper, deterministic_upper.to(lower.dtype))
        return torch.stack((bound_lower, bound_upper), dim=-1)


def create_gaussian_belief_trajectory(lower, upper, covariance, dtype=None, device=None):
    """Build a trajectory of GaussianBelief, one per step.

    lower, upper : [T] (scalar state) or [T, D]
    covariance   : [T] / [T, D] (diagonal) or [T, D, D] (full), matching lower's D
    """
    lower = torch.as_tensor(lower, dtype=dtype, device=device)
    upper = torch.as_tensor(upper, dtype=lower.dtype, device=lower.device)
    covariance = torch.as_tensor(covariance, dtype=lower.dtype, device=lower.device)

    if lower.ndim == 1:
        lower, upper = lower.unsqueeze(-1), upper.unsqueeze(-1)
        if covariance.ndim == 1:
            covariance = covariance.unsqueeze(-1)

    if lower.ndim != 2 or upper.shape != lower.shape:
        raise ValueError("lower and upper traces must have matching shape [T] or [T,D]")
    if covariance.shape[0] != lower.shape[0]:
        raise ValueError("covariance must have the same number of steps as lower/upper")

    return BeliefTrajectory(
        [
            GaussianBelief(lower[t : t + 1], upper[t : t + 1], covariance[t : t + 1])
            for t in range(lower.shape[0])
        ]
    )
