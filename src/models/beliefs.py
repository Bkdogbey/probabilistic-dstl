"""Precise Gaussian beliefs and their atomic probability evaluations."""

import torch

from pdstl.base import Belief, BeliefTrajectory


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


def _marginal(covariance, predicate, state_dim):
    """The predicate's state dimension and that component's [B] variance."""
    dim = getattr(predicate, "dim", 0)
    if not isinstance(dim, int) or not 0 <= dim < state_dim:
        raise ValueError(f"predicate dimension {dim} is outside the state")
    variance = covariance[:, dim, dim] if covariance.ndim == 3 else covariance[:, dim]
    return dim, variance


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
        """Exact marginal probability of the predicate, returned as [p, p], [B,2].

        x_j >= h:  p = Phi((mu_j - h) / sigma_j)
        x_j <= h:  p = Phi((h - mu_j) / sigma_j)
        """
        dim, variance = _marginal(self.covariance, predicate, self.mean.shape[1])
        p = _tail_probability("GaussianBelief", predicate, self.mean[:, dim], variance)
        return torch.stack((p, p), dim=-1)


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
