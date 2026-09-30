"""Sampling and full-state Gaussian measurement updates for lane simulation."""

import torch


def sample_gaussian(mean, covariance, generator=None):
    """Sample independent batched Gaussians, including singular covariances."""
    values, vectors = torch.linalg.eigh(covariance)
    factor = vectors * values.clamp_min(0).sqrt().unsqueeze(-2)
    noise = torch.randn(
        mean.shape, dtype=mean.dtype, device=mean.device, generator=generator
    )
    return mean + (factor @ noise.unsqueeze(-1)).squeeze(-1)


def observe_gaussian(mean, covariance, observation, observation_covariance):
    """Condition a predicted Gaussian on z = x + v, v ~ N(0, R).

    R must be positive definite. The Joseph covariance update preserves
    symmetry and positive semidefiniteness in finite precision.
    """
    innovation_cov = covariance + observation_covariance
    gain = torch.linalg.solve(
        innovation_cov, covariance.transpose(-1, -2)
    ).transpose(-1, -2)
    posterior_mean = mean + (
        gain @ (observation - mean).unsqueeze(-1)
    ).squeeze(-1)
    residual = torch.eye(mean.shape[-1], device=mean.device, dtype=mean.dtype)
    residual = residual - gain
    posterior_cov = residual @ covariance @ residual.transpose(
        -1, -2
    ) + gain @ observation_covariance @ gain.transpose(-1, -2)
    return posterior_mean, (
        posterior_cov + posterior_cov.transpose(-1, -2)
    ) / 2
