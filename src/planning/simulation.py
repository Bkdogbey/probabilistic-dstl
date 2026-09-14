import torch


def simulate_gaussian_step(dynamics, mean, cov, u):
    """Executed step: predicted mean plus sampled process noise, predicted covariance."""
    mean_next, cov_next = dynamics.step(mean, cov, u)
    noise = torch.distributions.MultivariateNormal(
        torch.zeros_like(mean_next), dynamics.Q
    ).sample()
    return mean_next + noise, cov_next
