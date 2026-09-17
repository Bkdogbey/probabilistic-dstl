"""Events scored from a belief's moments rather than through `Belief.probability_bounds`.

These reach into `belief.value()` and `belief.covariance` directly, so they only work for
Gaussian beliefs and they bypass the contract every other event obeys. They exist because
circular and moving obstacles have no axis-interval form yet; when they get one, this module
goes away. Nothing in the reach-avoid path uses them.
"""

import math

import torch

from pdstl.operators import STL_Formula


def extract_trajectory_stats(belief_trajectory, diagonal_only=True):
    """Stack means [B,T,D] and variances [B,T,D] (or full covariances) over the trajectory."""
    means, vars_ = [], []
    for belief in belief_trajectory:
        means.append(belief.value())
        if diagonal_only and belief.covariance.ndim > 2:
            vars_.append(torch.diagonal(belief.covariance, dim1=-2, dim2=-1))
        else:
            vars_.append(belief.covariance)
    return torch.stack(means, dim=1), torch.stack(vars_, dim=1)


def normal_cdf(value, mean, var):
    """P(X <= value) for X ~ N(mean, var)."""
    z = (value - mean) / torch.sqrt(var + 1e-6)
    return 0.5 * (1 + torch.erf(z / math.sqrt(2)))


class CircularObstaclePredicate(STL_Formula):
    """P(||x - center|| > radius), using the variance projected on the radial direction."""

    def __init__(self, region):
        super().__init__()
        self.name = region.name
        self.center = torch.as_tensor(region.center, dtype=torch.float32)
        self.radius = float(region.radius)

    def robustness_trace(self, belief_trajectory, **kwargs):
        mu, cov = extract_trajectory_stats(belief_trajectory, diagonal_only=False)
        diff = mu - self.center.to(mu.device)
        dist = torch.norm(diff, dim=-1)
        direction = diff / (dist.unsqueeze(-1) + 1e-6)
        if cov.ndim == 3:
            radial_var = torch.sum(direction**2 * cov, dim=-1)
        else:
            radial_var = torch.einsum("bti,btij,btj->bt", direction, cov, direction)
        p_safe = 1.0 - normal_cdf(self.radius, dist, radial_var)
        return torch.stack([p_safe, p_safe], dim=-1)


class MovingRectangularObstaclePredicate(STL_Formula):
    """Max of the four one-sided probabilities of being outside a moving rectangle."""

    def __init__(self, region):
        super().__init__()
        self.name = region.name
        centers = torch.as_tensor(region.centers, dtype=torch.float32)
        self.x_traj, self.y_traj = centers[..., 0], centers[..., 1]
        self.width, self.height = float(region.width), float(region.height)

    def robustness_trace(self, belief_trajectory, **kwargs):
        mu, var = extract_trajectory_stats(belief_trajectory)
        mu_x, mu_y, var_x, var_y = mu[..., 0], mu[..., 1], var[..., 0], var[..., 1]
        half_w, half_h = self.width / 2.0, self.height / 2.0
        x_traj, y_traj = self.x_traj.to(mu.device), self.y_traj.to(mu.device)
        p_safe = torch.stack([
            normal_cdf(x_traj - half_w, mu_x, var_x),
            1.0 - normal_cdf(x_traj + half_w, mu_x, var_x),
            normal_cdf(y_traj - half_h, mu_y, var_y),
            1.0 - normal_cdf(y_traj + half_h, mu_y, var_y),
        ], dim=0).max(dim=0).values
        return torch.stack([p_safe, p_safe], dim=-1)
