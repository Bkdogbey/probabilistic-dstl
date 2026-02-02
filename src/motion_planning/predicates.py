"""
Spatial Predicates for 2D Motion Planning
==========================================
STL predicates for obstacle avoidance and goal reaching.
"""

import torch
import numpy as np
from pdstl.operators import STL_Formula


class ObstacleAvoidance(STL_Formula):
    """
    Rectangular obstacle avoidance: P(position OUTSIDE obstacle).

    Uses Gaussian CDF for proper probability computation.
    Returns conservative probability intervals [lower, upper].

    Args:
        x_min, x_max: Obstacle X-bounds
        y_min, y_max: Obstacle Y-bounds
    """

    def __init__(self, x_min, x_max, y_min, y_max):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.eps = 1e-6
        self.sqrt2 = np.sqrt(2.0)

    def _normal_cdf(self, z):
        """Standard normal CDF"""
        return 0.5 * (1 + torch.erf(z / self.sqrt2))

    def robustness_trace(self, belief_trajectory, **kwargs):
        probs_lower = []
        probs_upper = []

        for t in range(len(belief_trajectory)):
            belief = belief_trajectory[t]

            mu = belief.mean  # [B, 1, 2]
            var = belief.var  # [B, 1, 2]

            mu_x = mu[:, :, 0:1]
            mu_y = mu[:, :, 1:2]
            var_x = var[:, :, 0:1]
            var_y = var[:, :, 1:2]

            std_x = torch.sqrt(var_x + self.eps)
            std_y = torch.sqrt(var_y + self.eps)

            x_min_t = torch.tensor(self.x_min, dtype=mu.dtype, device=mu.device)
            x_max_t = torch.tensor(self.x_max, dtype=mu.dtype, device=mu.device)
            y_min_t = torch.tensor(self.y_min, dtype=mu.dtype, device=mu.device)
            y_max_t = torch.tensor(self.y_max, dtype=mu.dtype, device=mu.device)

            # P(inside) = P(x_min < x < x_max) * P(y_min < y < y_max)
            p_x_inside = self._normal_cdf((x_max_t - mu_x) / std_x) - self._normal_cdf(
                (x_min_t - mu_x) / std_x
            )
            p_y_inside = self._normal_cdf((y_max_t - mu_y) / std_y) - self._normal_cdf(
                (y_min_t - mu_y) / std_y
            )
            p_inside = p_x_inside * p_y_inside

            # Lower bound: Pessimistic (add penalty if uncertainty overlaps)
            lower_bound = belief.lower_bound()
            upper_bound = belief.upper_bound()

            overlap_x = (lower_bound[:, :, 0:1] < x_max_t) & (
                upper_bound[:, :, 0:1] > x_min_t
            )
            overlap_y = (lower_bound[:, :, 1:2] < y_max_t) & (
                upper_bound[:, :, 1:2] > y_min_t
            )
            overlap = overlap_x.float() * overlap_y.float()

            p_inside_pessimistic = torch.clamp(p_inside + 0.2 * overlap, 0.0, 1.0)
            p_outside_lower = torch.clamp(
                1.0 - p_inside_pessimistic, self.eps, 1.0 - self.eps
            )

            # Upper bound: Optimistic (reduce penalty if clearly outside)
            mean_outside_x = (mu_x < x_min_t - std_x) | (mu_x > x_max_t + std_x)
            mean_outside_y = (mu_y < y_min_t - std_y) | (mu_y > y_max_t + std_y)
            clearly_outside = (mean_outside_x.float() + mean_outside_y.float()).clamp(
                0, 1
            )

            p_inside_optimistic = p_inside * (1.0 - 0.2 * clearly_outside)
            p_outside_upper = torch.clamp(
                1.0 - p_inside_optimistic, self.eps, 1.0 - self.eps
            )

            probs_lower.append(p_outside_lower)
            probs_upper.append(p_outside_upper)

        lower = torch.cat(probs_lower, dim=1).squeeze(2)
        upper = torch.cat(probs_upper, dim=1).squeeze(2)

        return torch.stack([lower, upper], dim=-1)

    def __str__(self):
        return f"ObstacleAvoid(x∈[{self.x_min:.1f},{self.x_max:.1f}], y∈[{self.y_min:.1f},{self.y_max:.1f}])"


class GoalReaching(STL_Formula):
    """
    Goal reaching: P(distance to goal <= radius).

    Uses CDF-based probability computation for proper calibration.

    Args:
        goal_x, goal_y: Goal coordinates
        radius: Goal region radius
        temperature: Sigmoid temperature (default: 1.0)
    """

    def __init__(self, goal_x, goal_y, radius, temperature=1.0):
        super().__init__()
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.radius = radius
        self.temperature = temperature
        self.eps = 1e-6
        self.sqrt2 = np.sqrt(2.0)

    def _normal_cdf(self, z):
        """Standard normal CDF"""
        return 0.5 * (1 + torch.erf(z / self.sqrt2))

    def robustness_trace(self, belief_trajectory, **kwargs):
        probs_lower = []
        probs_upper = []

        for t in range(len(belief_trajectory)):
            belief = belief_trajectory[t]
            mu = belief.mean  # [B, 1, 2]
            var = belief.var  # [B, 1, 2]

            # Distance from mean to goal
            dx = mu[:, :, 0:1] - self.goal_x
            dy = mu[:, :, 1:2] - self.goal_y
            dist_mean = torch.sqrt(dx**2 + dy**2 + self.eps)

            # Approximate distance uncertainty (using sum of variances)

            var_dist = (dx**2 / (dist_mean**2 + self.eps)) * var[:, :, 0:1] + (
                dy**2 / (dist_mean**2 + self.eps)
            ) * var[:, :, 1:2]
            std_dist = torch.sqrt(var_dist + self.eps)

            # P(distance <= radius) using CDF
            # Lower bound: Pessimistic (assume farther, use mean + k*σ)
            k = 2.0
            dist_pessimistic = dist_mean + k * std_dist
            # P(d <= r) when actual distance is dist_pessimistic
            z_lower = (self.radius - dist_pessimistic) / (std_dist + self.eps)
            prob_lower = self._normal_cdf(z_lower)
            prob_lower = torch.clamp(prob_lower, self.eps, 1.0 - self.eps)

            # Upper bound: Optimistic (assume closer, use mean - k*σ)
            dist_optimistic = torch.clamp(dist_mean - k * std_dist, min=0.0)
            z_upper = (self.radius - dist_optimistic) / (std_dist + self.eps)
            prob_upper = self._normal_cdf(z_upper)
            prob_upper = torch.clamp(prob_upper, self.eps, 1.0 - self.eps)

            # Ensure lower <= upper
            prob_lower = torch.minimum(prob_lower, prob_upper)

            probs_lower.append(prob_lower)
            probs_upper.append(prob_upper)

        lower = torch.cat(probs_lower, dim=1).squeeze(-1)
        upper = torch.cat(probs_upper, dim=1).squeeze(-1)

        return torch.stack([lower, upper], dim=-1)

    def __str__(self):
        return f"GoalReach(({self.goal_x:.1f},{self.goal_y:.1f}), r={self.radius:.1f})"
