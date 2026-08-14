"""Differentiable probability primitives shared by pdSTL implementations."""

from __future__ import annotations

import math

import torch


def gaussian_residual_probability(mean, variance):
    """Return ``P(R >= 0)`` for ``R ~ N(mean, variance)``.

    Positive variance uses the exact Gaussian CDF. Zero variance uses the
    deterministic convention that equality satisfies the non-strict event.
    """

    if not torch.is_tensor(mean):
        mean = torch.as_tensor(mean, dtype=torch.get_default_dtype())
    variance = torch.as_tensor(variance, device=mean.device, dtype=mean.dtype)
    mean, variance = torch.broadcast_tensors(mean, variance)

    if torch.any(variance < 0):
        raise ValueError("Gaussian variance must be nonnegative.")

    positive_variance = variance > 0
    safe_variance = torch.where(
        positive_variance, variance, torch.ones_like(variance)
    )
    standardized_mean = mean / torch.sqrt(safe_variance)
    gaussian_probability = 0.5 * (
        1.0 + torch.erf(standardized_mean / math.sqrt(2.0))
    )
    deterministic_probability = (mean >= 0).to(dtype=gaussian_probability.dtype)
    return torch.where(
        positive_variance, gaussian_probability, deterministic_probability
    )
