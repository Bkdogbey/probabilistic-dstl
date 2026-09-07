"""Gaussian state beliefs and affine predicate probabilities."""

import math
import torch
from pdstl.base import Belief


def normal_cdf(z):
    """Cumulative distribution function for standard normal distribution"""
    return 0.5 * (1 + torch.erf(z / math.sqrt(2.0)))


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
