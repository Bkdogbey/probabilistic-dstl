"""Precise Gaussian beliefs and their atomic probability evaluations."""

import torch

from pdstl.base import Belief, BeliefTrajectory
from pdstl.predicates import InsideRectangle, OutsideRectangle


def _validate_covariance(owner, covariance, batch, state_dim):
    """Validate a [B,D] diagonal or [B,D,D] full covariance."""
    if not torch.is_tensor(covariance):
        raise ValueError(f"{owner} covariance must be a tensor")
    if not torch.isfinite(covariance).all():
        raise ValueError(f"{owner} covariance must be finite")
    if covariance.ndim == 2:
        if covariance.shape != (batch, state_dim):
            raise ValueError("diagonal covariance must have shape [B,D]")
        component_variance = covariance
    elif covariance.ndim == 3:
        if covariance.shape != (batch, state_dim, state_dim):
            raise ValueError("full covariance must have shape [B,D,D]")
        # A @ P @ A^T accumulates float roundoff, so this is a tolerance
        # check, not exact equality.
        if not torch.allclose(covariance, covariance.transpose(-1, -2), atol=1e-5):
            raise ValueError("full covariance must be symmetric")
        eigenvalues = torch.linalg.eigvalsh(covariance)
        if bool((eigenvalues < -1e-6).any()):
            raise ValueError("full covariance must be positive semi-definite")
        component_variance = covariance.diagonal(dim1=-2, dim2=-1)
    else:
        raise ValueError(f"{owner} covariance must be [B,D] or [B,D,D]")

    if bool((component_variance < 0).any()):
        raise ValueError(f"{owner} covariance must be non-negative")


def _marginal_variance(covariance, dim, state_dim):
    """The [B] variance of state component ``dim``."""
    if not isinstance(dim, int) or not 0 <= dim < state_dim:
        raise ValueError(f"predicate dimension {dim} is outside the state")
    return covariance[:, dim, dim] if covariance.ndim == 3 else covariance[:, dim]


def _tail_probability(owner, predicate, location, variance):
    """P(X sense threshold) for X ~ N(location, variance), [B].

    Zero variance is an inclusive deterministic comparison.
    """
    sense = getattr(predicate, "sense", None)
    if sense not in (">=", "<="):
        raise ValueError(f"{owner} cannot evaluate {predicate}")

    threshold = torch.as_tensor(
        predicate.threshold, dtype=location.dtype, device=location.device
    )
    positive = variance > 0
    safe_sigma = torch.where(positive, torch.sqrt(variance), torch.ones_like(variance))

    if sense == ">=":
        probability = _normal_cdf((location - threshold) / safe_sigma)
        deterministic = location >= threshold
    else:
        probability = _normal_cdf((threshold - location) / safe_sigma)
        deterministic = location <= threshold
    return torch.where(positive, probability, deterministic.to(location.dtype))


def _normal_cdf(z):
    """Phi(z), accurate far into the lower tail in float32."""
    return torch.exp(torch.special.log_ndtr(z))


def _interval_probability(low, high, location, variance):
    """P(low <= X <= high) for X ~ N(location, variance), [B].

    The difference is taken between the two smaller tails, so an interval deep
    in either tail keeps float32 resolution. Zero variance is an inclusive
    deterministic test.
    """
    low = torch.as_tensor(low, dtype=location.dtype, device=location.device)
    high = torch.as_tensor(high, dtype=location.dtype, device=location.device)
    positive = variance > 0
    sigma = torch.sqrt(torch.where(positive, variance, torch.ones_like(variance)))
    a, b = (low - location) / sigma, (high - location) / sigma

    probability = torch.where(
        a > 0,
        _normal_cdf(-a) - _normal_cdf(-b),  # interval right of the mean
        _normal_cdf(b) - _normal_cdf(a),
    ).clamp(0.0, 1.0)
    deterministic = (location >= low) & (location <= high)
    return torch.where(positive, probability, deterministic.to(location.dtype))


class GaussianBelief(Belief):
    """One precise Gaussian state belief, X ~ N(mean, covariance).

    mean : [B, D]
    covariance : [B, D] (diagonal) or [B, D, D] (full)
        Only the diagonal enters a one-dimensional predicate's marginal.
    """

    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance
        self._validate_shapes()

    def _validate_shapes(self):
        if not torch.is_tensor(self.mean) or self.mean.ndim != 2:
            raise ValueError("GaussianBelief mean must have shape [B,D]")
        if not torch.isfinite(self.mean).all():
            raise ValueError("GaussianBelief mean must be finite")
        _validate_covariance("GaussianBelief", self.covariance, *self.mean.shape)

    def value(self):
        """The mean, [B,D]."""
        return self.mean

    def probability_bounds(self, predicate):
        """Probability interval of the predicate's event, [B,2].

        x_j >= h:  exact, p = Phi((mu_j - h) / sigma_j), returned as [p, p]
        x_j <= h:  exact, p = Phi((h - mu_j) / sigma_j), returned as [p, p]
        InsideRectangle / OutsideRectangle:  see _rectangle_bounds
        """
        if isinstance(predicate, (InsideRectangle, OutsideRectangle)):
            return self._rectangle_bounds(predicate)

        dim = getattr(predicate, "dim", 0)
        variance = _marginal_variance(self.covariance, dim, self.mean.shape[1])
        p = _tail_probability("GaussianBelief", predicate, self.mean[:, dim], variance)
        return torch.stack((p, p), dim=-1)

    def _rectangle_bounds(self, predicate):
        """Bounds on P(X in R) from the two marginals, without assuming independence.

        p_x = P(x_min <= X_i <= x_max),  p_y = P(y_min <= X_j <= y_max)
        Inside:   [max(0, p_x + p_y - 1), min(p_x, p_y)]   (Frechet)
        Outside:  [1 - U, 1 - L]                            (exact complement)
        """
        x_dim, y_dim = predicate.dims
        p_x = self._marginal_interval(x_dim, predicate.x_range)
        p_y = self._marginal_interval(y_dim, predicate.y_range)
        lower = torch.clamp(p_x + p_y - 1.0, min=0.0)
        upper = torch.minimum(p_x, p_y)
        if isinstance(predicate, OutsideRectangle):
            lower, upper = 1.0 - upper, 1.0 - lower
        return torch.stack((lower, upper), dim=-1)

    def _marginal_interval(self, dim, bounds):
        """P(bounds[0] <= X_dim <= bounds[1]), [B]."""
        variance = _marginal_variance(self.covariance, dim, self.mean.shape[1])
        return _interval_probability(*bounds, self.mean[:, dim], variance)


def create_gaussian_belief_trajectory(mean, covariance, dtype=None, device=None):
    """Build a trajectory of precise GaussianBelief, one per step.

    mean       : [T] (scalar state), [T, D], or [B, T, D] (batched)
    covariance : matching mean -- [T]; [T, D] or [T, D, D]; [B, T, D] or [B, T, D, D]
    """
    mean = torch.as_tensor(mean, dtype=dtype, device=device)
    covariance = torch.as_tensor(covariance, dtype=mean.dtype, device=mean.device)

    if mean.ndim == 1:
        mean = mean.unsqueeze(-1)
        if covariance.ndim == 1:
            covariance = covariance.unsqueeze(-1)
    if mean.ndim == 2:
        mean, covariance = mean.unsqueeze(0), covariance.unsqueeze(0)

    if mean.ndim != 3:
        raise ValueError("mean trace must have shape [T], [T,D] or [B,T,D]")
    if covariance.ndim < 2 or covariance.shape[:2] != mean.shape[:2]:
        raise ValueError("covariance must have the same batch and number of steps as mean")

    return BeliefTrajectory(
        [GaussianBelief(mean[:, t], covariance[:, t]) for t in range(mean.shape[1])]
    )
