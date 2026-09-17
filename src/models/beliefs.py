"""Gaussian beliefs and the probabilities of pdSTL events under them."""

import torch

from pdstl.base import Belief, BeliefTrajectory
from pdstl.predicates import AxisInterval, HalfSpace


def _validate_covariance(covariance, batch, state_dim):
    """Check a [B,D] diagonal or [B,D,D] full covariance."""
    if not torch.is_tensor(covariance):
        raise ValueError("GaussianBelief covariance must be a tensor")
    if not torch.isfinite(covariance).all():
        raise ValueError("GaussianBelief covariance must be finite")
    if covariance.ndim == 2:
        if covariance.shape != (batch, state_dim):
            raise ValueError("diagonal covariance must have shape [B,D]")
        variance = covariance
    elif covariance.ndim == 3:
        if covariance.shape != (batch, state_dim, state_dim):
            raise ValueError("full covariance must have shape [B,D,D]")
        # Tolerance, not equality: A P A^T accumulates round-off.
        if not torch.allclose(covariance, covariance.transpose(-1, -2), atol=1e-5):
            raise ValueError("full covariance must be symmetric")
        if bool((torch.linalg.eigvalsh(covariance) < -1e-6).any()):
            raise ValueError("full covariance must be positive semi-definite")
        variance = covariance.diagonal(dim1=-2, dim2=-1)
    else:
        raise ValueError("GaussianBelief covariance must be [B,D] or [B,D,D]")
    if bool((variance < 0).any()):
        raise ValueError("GaussianBelief covariance must be non-negative")


def _normal_cdf(z):
    """Phi(z), accurate far into the lower tail in float32."""
    return torch.exp(torch.special.log_ndtr(z))


def _standard_scores(location, variance, *bounds):
    """(bound - location) / sigma for each bound, plus the positive-variance mask."""
    positive = variance > 0
    sigma = torch.sqrt(torch.where(positive, variance, torch.ones_like(variance)))
    scores = [
        (torch.as_tensor(b, dtype=location.dtype, device=location.device) - location) / sigma
        for b in bounds
    ]
    return positive, scores


def _tail_probability(predicate, location, variance):
    """P(X >= c) or P(X <= c), [B]; zero variance compares deterministically."""
    sense = getattr(predicate, "sense", None)
    if sense not in (">=", "<="):
        raise ValueError(f"GaussianBelief cannot evaluate {predicate}")
    positive, (z,) = _standard_scores(location, variance, predicate.threshold)
    if sense == ">=":
        probability, deterministic = _normal_cdf(-z), location >= predicate.threshold
    else:
        probability, deterministic = _normal_cdf(z), location <= predicate.threshold
    return torch.where(positive, probability, deterministic.to(location.dtype))


def _interval_probability(low, high, location, variance):
    """P(low <= X <= high), [B]; subtracts the smaller tails to keep float32 resolution."""
    positive, (a, b) = _standard_scores(location, variance, low, high)
    probability = torch.where(
        a > 0, _normal_cdf(-a) - _normal_cdf(-b), _normal_cdf(b) - _normal_cdf(a)
    ).clamp(0.0, 1.0)
    deterministic = (location >= low) & (location <= high)
    return torch.where(positive, probability, deterministic.to(location.dtype))


class GaussianBelief(Belief):
    """X ~ N(mean, covariance); mean [B,D], covariance [B,D] or [B,D,D]."""

    def __init__(self, mean, covariance):
        if not torch.is_tensor(mean) or mean.ndim != 2:
            raise ValueError("GaussianBelief mean must have shape [B,D]")
        if not torch.isfinite(mean).all():
            raise ValueError("GaussianBelief mean must be finite")
        _validate_covariance(covariance, *mean.shape)
        self.mean = mean
        self.covariance = covariance

    def value(self):
        return self.mean

    def probability_bounds(self, predicate):
        """Exact [p, p] for affine, threshold and axis-interval events.

        Rectangles are not handled here: they are conjunctions of two AxisIntervals and are
        combined by the pdSTL operators, which is what lets `scale` relax them.
        """
        if isinstance(predicate, HalfSpace):
            return self._half_space_bounds(predicate)
        if isinstance(predicate, AxisInterval):
            variance = self._variance(predicate.dim)  # validates the index first
            p = _interval_probability(
                predicate.lower, predicate.upper, self.mean[:, predicate.dim], variance
            )
            return torch.stack((p, p), dim=-1)
        dim = getattr(predicate, "dim", 0)
        variance = self._variance(dim)
        p = _tail_probability(predicate, self.mean[:, dim], variance)
        return torch.stack((p, p), dim=-1)

    def _half_space_bounds(self, predicate):
        """P(a^T X <= b) = Phi((b - a^T mu) / sqrt(a^T Sigma a))."""
        a = self.mean.new_tensor(predicate.a)
        if a.shape != (self.mean.shape[1],):
            raise ValueError("half-space normal must match the state dimension")
        location = self.mean @ a
        if self.covariance.ndim == 2:
            variance = (self.covariance * a.square()).sum(dim=-1)
        else:
            variance = torch.einsum("i,bij,j->b", a, self.covariance, a)
        # A PSD covariance can produce tiny negative projections by round-off.
        positive, (z,) = _standard_scores(location, variance.clamp_min(0), predicate.b)
        p = torch.where(positive, _normal_cdf(z), (location <= predicate.b).to(location.dtype))
        return torch.stack((p, p), dim=-1)

    def _variance(self, dim):
        """[B] marginal variance of state component dim."""
        if not isinstance(dim, int) or not 0 <= dim < self.mean.shape[1]:
            raise ValueError(f"predicate dimension {dim} is outside the state")
        cov = self.covariance
        return cov[:, dim, dim] if cov.ndim == 3 else cov[:, dim]


def create_gaussian_belief_trajectory(mean, covariance, dtype=None, device=None):
    """One GaussianBelief per step; mean [T], [T,D] or [B,T,D] with matching covariance."""
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
