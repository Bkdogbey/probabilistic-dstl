import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse as MPLEllipse

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dynamics import GaussianBelief
from pdstl.base import BeliefTrajectory
from pdstl.operators import Always, Eventually


# =============================================================================
#  GOAL PREDICATE
# =============================================================================


class GoalReached(nn.Module):
    def __init__(self, x_goal, epsilon_g, beta_0=0.05, temperature=1.0):
        super(GoalReached, self).__init__()
        self.x_goal = x_goal
        self.epsilon_g = epsilon_g
        self.beta_0 = beta_0
        self.temperature = temperature
        self.eps = 1e-6

    def robustness_trace(self, belief_trajectory, **kwargs):
        """Compute goal reach probability for each timestep with conservative bounds."""
        probs_lower = []
        probs_upper = []

        for t in range(len(belief_trajectory)):
            belief = belief_trajectory[t]
            mu_t = belief.mean  # [B, 1, 2]
            var_t = belief.var  # [B, 1, 2]

            x_goal = self.x_goal.unsqueeze(1) if self.x_goal.dim() == 2 else self.x_goal

            # Distance from mean to goal
            dist_mean = torch.norm(mu_t - x_goal, p=2, dim=-1, keepdim=True)

            # Standard deviation (uncertainty in distance)
            std_t = torch.sqrt(var_t.sum(dim=-1, keepdim=True) + self.eps)

            # LOWER BOUND: Pessimistic - assume we're FARTHER from goal
            # Worst case: distance could be (dist_mean + k*std)
            k_pessimistic = 2.0  # 2-sigma confidence
            dist_pessimistic = dist_mean + k_pessimistic * std_t
            margin_lower = self.epsilon_g - dist_pessimistic

            # Compute probability with additional uncertainty factor
            beta = std_t + self.beta_0
            logit_lower = margin_lower / (beta * self.temperature + self.eps)
            prob_lower = torch.sigmoid(logit_lower)
            prob_lower = torch.clamp(prob_lower, self.eps, 1.0 - self.eps)

            # UPPER BOUND: Optimistic - assume we're CLOSER to goal
            # Best case: distance could be max(0, dist_mean - k*std)
            k_optimistic = 2.0
            dist_optimistic = torch.clamp(dist_mean - k_optimistic * std_t, min=0.0)
            margin_upper = self.epsilon_g - dist_optimistic

            logit_upper = margin_upper / (beta * self.temperature + self.eps)
            prob_upper = torch.sigmoid(logit_upper)
            prob_upper = torch.clamp(prob_upper, self.eps, 1.0 - self.eps)

            # Ensure lower <= upper (numerical safety)
            prob_lower = torch.minimum(prob_lower, prob_upper)

            probs_lower.append(prob_lower)
            probs_upper.append(prob_upper)

        lower_tensor = torch.cat(probs_lower, dim=1).squeeze(2)
        upper_tensor = torch.cat(probs_upper, dim=1).squeeze(2)

        return torch.stack([lower_tensor, upper_tensor], dim=-1)

    def forward(self, belief_trajectory, **kwargs):
        return self.robustness_trace(belief_trajectory, **kwargs)


# =============================================================================
#  SAFETY PREDICATE - CORRECTED
# =============================================================================


class ObstacleAvoidance(nn.Module):
    """
    Obstacle avoidance with CORRECTED conservative probability intervals.
    """

    def __init__(self, obstacle_bounds, conservative_factor=1.0):
        super(ObstacleAvoidance, self).__init__()
        self.x_min = obstacle_bounds["x"][0]
        self.x_max = obstacle_bounds["x"][1]
        self.y_min = obstacle_bounds["y"][0]
        self.y_max = obstacle_bounds["y"][1]
        self.conservative_factor = conservative_factor
        self.eps = 1e-6
        self.sqrt2 = np.sqrt(2.0)

    def _normal_cdf(self, z):
        """Standard normal CDF"""
        return 0.5 * (1 + torch.erf(z / self.sqrt2))

    def robustness_trace(self, belief_trajectory, **kwargs):
        """Compute obstacle avoidance probability for each timestep with conservative bounds."""
        probs_lower = []
        probs_upper = []

        for t in range(len(belief_trajectory)):
            belief = belief_trajectory[t]

            # Get mean and variance
            mu = belief.mean
            var = belief.var

            mu_x = mu[:, :, 0:1]
            mu_y = mu[:, :, 1:2]
            var_x = var[:, :, 0:1]
            var_y = var[:, :, 1:2]

            std_x = torch.sqrt(var_x + self.eps)
            std_y = torch.sqrt(var_y + self.eps)

            # Convert obstacle bounds to tensors
            x_min_t = torch.tensor(self.x_min, dtype=mu.dtype, device=mu.device)
            x_max_t = torch.tensor(self.x_max, dtype=mu.dtype, device=mu.device)
            y_min_t = torch.tensor(self.y_min, dtype=mu.dtype, device=mu.device)
            y_max_t = torch.tensor(self.y_max, dtype=mu.dtype, device=mu.device)

            # LOWER BOUND: Pessimistic estimate
            # For safety: assume we're MORE likely to be inside obstacle
            # Use conservative approach: if lower bound of position distribution
            # could be inside, reduce the probability of being outside

            # Check if mean position is close to obstacle
            lower_bound = belief.lower_bound()  # [B, 1, 2]
            upper_bound = belief.upper_bound()  # [B, 1, 2]

            lb_x = lower_bound[:, :, 0:1]
            lb_y = lower_bound[:, :, 1:2]
            ub_x = upper_bound[:, :, 0:1]
            ub_y = upper_bound[:, :, 1:2]

            # P(x inside obstacle) - use mean for nominal calculation
            p_x_less_max = self._normal_cdf((x_max_t - mu_x) / std_x)
            p_x_less_min = self._normal_cdf((x_min_t - mu_x) / std_x)
            p_x_inside = p_x_less_max - p_x_less_min

            p_y_less_max = self._normal_cdf((y_max_t - mu_y) / std_y)
            p_y_less_min = self._normal_cdf((y_min_t - mu_y) / std_y)
            p_y_inside = p_y_less_max - p_y_less_min

            # Assuming independence (valid for diagonal covariance)
            p_inside_nominal = p_x_inside * p_y_inside

            # LOWER BOUND: Pessimistic (higher chance of being inside)
            # If uncertainty bounds overlap obstacle, increase p_inside for lower bound
            overlap_x = (lb_x < x_max_t) & (ub_x > x_min_t)
            overlap_y = (lb_y < y_max_t) & (ub_y > y_min_t)
            overlap = (
                overlap_x.float() * overlap_y.float()
            )  # 1 if overlaps, 0 otherwise

            # Increase p_inside by uncertainty factor when overlapping
            uncertainty_penalty = 0.3 * overlap  # Add up to 30% uncertainty penalty
            p_inside_pessimistic = torch.clamp(
                p_inside_nominal + uncertainty_penalty, 0.0, 1.0
            )
            p_outside_lower = (1.0 - p_inside_pessimistic) * self.conservative_factor
            p_outside_lower = torch.clamp(p_outside_lower, self.eps, 1.0 - self.eps)

            # UPPER BOUND: Optimistic (lower chance of being inside)
            # If mean is clearly outside, give higher confidence
            mean_outside_x = (mu_x < x_min_t - std_x) | (mu_x > x_max_t + std_x)
            mean_outside_y = (mu_y < y_min_t - std_y) | (mu_y > y_max_t + std_y)
            clearly_outside = (mean_outside_x.float() + mean_outside_y.float()).clamp(
                0, 1
            )

            # Reduce p_inside when clearly outside
            p_inside_optimistic = p_inside_nominal * (1.0 - 0.3 * clearly_outside)
            p_outside_upper = (1.0 - p_inside_optimistic) * self.conservative_factor
            p_outside_upper = torch.clamp(p_outside_upper, self.eps, 1.0 - self.eps)

            probs_lower.append(p_outside_lower)
            probs_upper.append(p_outside_upper)

        lower_tensor = torch.cat(probs_lower, dim=1).squeeze(2)
        upper_tensor = torch.cat(probs_upper, dim=1).squeeze(2)

        return torch.stack([lower_tensor, upper_tensor], dim=-1)

    def forward(self, belief_trajectory, **kwargs):
        return self.robustness_trace(belief_trajectory, **kwargs)


# =============================================================================
# DYNAMICS MODEL
# =============================================================================


class Dynamics2D:
    """2D stochastic single-integrator dynamics"""

    def __init__(self, dt=0.1, sigma=0.1):
        self.dt = dt
        self.sigma = sigma
        self.Q_d = (sigma**2) * dt * torch.eye(2).unsqueeze(0).unsqueeze(0)

    def step(self, mu_t, Sigma_t, u_t):
        mu_next = mu_t + self.dt * u_t
        Sigma_next = Sigma_t + self.Q_d
        return mu_next, Sigma_next


# =============================================================================
# OPTIMIZER WITH CORRECTED LOSS
# =============================================================================


class ProbabilisticSTLOptimizer:
    """Gradient-based trajectory optimizer with CORRECTED loss computation"""

    def __init__(
        self,
        dynamics,
        spec_safe,
        spec_goal,
        x_goal,
        horizon,
        dt=0.1,
        optimizer_type="adam",
        lr=0.01,
        device="cpu",
        obstacle_bounds=None,
    ):
        self.dynamics = dynamics
        self.spec_safe = spec_safe
        self.spec_goal = spec_goal
        self.x_goal = x_goal.to(device)
        self.N = horizon
        self.dt = dt
        self.device = device
        self.obstacle_for_init = obstacle_bounds

        # Initialize controls with intelligent waypoint routing
        init_controls = torch.zeros(1, horizon - 1, 2, device=device)

        if x_goal is not None and obstacle_bounds is not None:
            x_goal_np = x_goal.cpu().numpy().squeeze()
            obs = obstacle_bounds

            x_start = np.array([0.0, 0.0])
            obs_center = np.array(
                [(obs["x"][0] + obs["x"][1]) / 2, (obs["y"][0] + obs["y"][1]) / 2]
            )

            # Calculate clearances on all sides
            clearance_right = abs(x_goal_np[0] - obs["x"][1])
            clearance_left = abs(x_goal_np[0] - obs["x"][0])
            clearance_top = abs(x_goal_np[1] - obs["y"][1])
            clearance_bottom = abs(x_goal_np[1] - obs["y"][0])

            max_clearance = max(
                clearance_right, clearance_left, clearance_top, clearance_bottom
            )

            # Choose waypoint based on maximum clearance with proper bounds checking
            margin = 1.5  # Increased margin for safety
            if max_clearance == clearance_right and x_goal_np[0] > obs["x"][1]:
                waypoint = np.array([obs["x"][1] + margin, obs_center[1]])
            elif max_clearance == clearance_left and x_goal_np[0] < obs["x"][0]:
                waypoint = np.array([obs["x"][0] - margin, obs_center[1]])
            elif max_clearance == clearance_top and x_goal_np[1] > obs["y"][1]:
                waypoint = np.array([obs_center[0], obs["y"][1] + margin])
            elif max_clearance == clearance_bottom and x_goal_np[1] < obs["y"][0]:
                waypoint = np.array([obs_center[0], obs["y"][0] - margin])
            else:
                # Default: go around right side
                waypoint = np.array([obs["x"][1] + margin, obs_center[1]])

            # Two-phase initialization: start -> waypoint -> goal
            n_first = int(horizon * 0.6)
            n_second = horizon - 1 - n_first

            time_to_waypoint = n_first * dt
            v_to_waypoint = (waypoint - x_start) / time_to_waypoint
            init_controls[:, :n_first, :] = torch.tensor(
                v_to_waypoint, device=device
            ).view(1, 1, 2)

            time_to_goal = n_second * dt
            v_to_goal = (x_goal_np - waypoint) / time_to_goal
            init_controls[:, n_first:, :] = torch.tensor(v_to_goal, device=device).view(
                1, 1, 2
            )

        self.U = nn.Parameter(init_controls)

        # Loss weights
        self.weights = {
            "safe": 100.0,  # Safety critical
            "goal": 100.0,  # Goal achievement
            "terminal": 50.0,  # Terminal cost
            "control": 0.01,  # Control regularization
            "smoothness": 0.1,  # Smoothness
        }

        self.alpha = 0.95  # Target probability threshold

        # Optimizer
        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam([self.U], lr=lr)
        elif optimizer_type == "lbfgs":
            self.optimizer = torch.optim.LBFGS([self.U], lr=lr, max_iter=20)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        self.optimizer_type = optimizer_type

    def forward_simulate(self, x0, Sigma0, U):
        """Propagate Gaussian belief through dynamics."""
        beliefs = []

        if isinstance(x0, np.ndarray):
            x0 = torch.tensor(x0, dtype=torch.float32, device=self.device)
        if isinstance(Sigma0, np.ndarray):
            Sigma0 = torch.tensor(Sigma0, dtype=torch.float32, device=self.device)

        if x0.dim() == 1:
            mu = x0.unsqueeze(0).unsqueeze(0)
        elif x0.dim() == 2:
            mu = x0.unsqueeze(1)
        else:
            mu = x0

        if Sigma0.dim() == 2:
            Sigma = Sigma0.unsqueeze(0).unsqueeze(0)
        elif Sigma0.dim() == 3:
            Sigma = Sigma0.unsqueeze(0)
        else:
            Sigma = Sigma0

        for t in range(self.N):
            # Extract diagonal variances for GaussianBelief
            var_diag = torch.stack([Sigma[:, :, 0, 0], Sigma[:, :, 1, 1]], dim=-1)
            belief = GaussianBelief(
                mean=mu.clone(), var=var_diag.clone(), confidence_level=2.0
            )
            beliefs.append(belief)

            if t < self.N - 1:
                u_t = U[:, t : t + 1, :]
                mu, Sigma = self.dynamics.step(mu, Sigma, u_t)

        return BeliefTrajectory(beliefs)

    def compute_loss(self, belief_traj):
        """Compute hinge loss for STL constraints with CORRECTED aggregation."""
        # Evaluate STL specifications - both return [B, T, 2]
        safe_trace = self.spec_safe(belief_traj)  # [B, T, 2]
        goal_trace = self.spec_goal(belief_traj)  # [B, T, 2]

        # Always: Take minimum over time (worst-case constraint violation)
        # Use LOWER bound (pessimistic) for safety constraints
        # safe_trace[:, :, 0] is [B, T] lower bound probabilities
        p_safe = safe_trace[:, :, 0].min(dim=1)[0].mean()

        # Eventually: Take maximum over time (best-case goal satisfaction)
        # Use LOWER bound (pessimistic) for goal achievement
        # goal_trace[:, :, 0] is [B, T] lower bound probabilities
        p_goal = goal_trace[:, :, 0].max(dim=1)[0].mean()

        # STL hinge losses - penalize when probability < threshold
        L_safe = torch.relu(self.alpha - p_safe)
        L_goal = torch.relu(self.alpha - p_goal)

        # Terminal cost - encourage reaching goal position
        mu_final = belief_traj[-1].mean
        L_terminal = torch.norm(mu_final - self.x_goal.unsqueeze(1), p=2) ** 2

        # Control regularization - penalize large controls
        L_control = torch.sum(self.U**2)

        # Smoothness - penalize jerky motions
        if self.U.shape[1] > 1:
            L_smoothness = torch.sum((self.U[:, 1:, :] - self.U[:, :-1, :]) ** 2)
        else:
            L_smoothness = torch.tensor(0.0, device=self.device)

        # Total loss
        loss = (
            self.weights["safe"] * L_safe
            + self.weights["goal"] * L_goal
            + self.weights["terminal"] * L_terminal
            + self.weights["control"] * L_control
            + self.weights["smoothness"] * L_smoothness
        )

        metrics = {
            "total": loss.item(),
            "safe": L_safe.item(),
            "goal": L_goal.item(),
            "terminal": L_terminal.item(),
            "control": L_control.item(),
            "smoothness": L_smoothness.item(),
            "p_safe": p_safe.item(),
            "p_goal": p_goal.item(),
        }

        return loss, metrics

    def optimize(
        self,
        x0,
        Sigma0,
        max_iters=200,
        verbose=True,
        print_every=10,
        early_stop_threshold=None,
    ):
        """Run gradient-based optimization with monitoring."""
        x0 = torch.tensor(x0, dtype=torch.float32, device=self.device)
        Sigma0 = torch.tensor(Sigma0, dtype=torch.float32, device=self.device)

        history = []

        if verbose:
            print("=" * 80)
            print("Starting Probabilistic STL Trajectory Optimization (CORRECTED)")
            print("=" * 80)
            print(f"Horizon: {self.N}, dt: {self.dt}, Device: {self.device}")
            print(
                f"Optimizer: {self.optimizer_type}, LR: {self.optimizer.param_groups[0]['lr']}"
            )
            print(f"Target probability: α = {self.alpha}")
            print("-" * 80)

        for iteration in range(max_iters):
            if self.optimizer_type == "lbfgs":

                def closure():
                    self.optimizer.zero_grad()
                    belief_traj = self.forward_simulate(x0, Sigma0, self.U)
                    loss, _ = self.compute_loss(belief_traj)
                    loss.backward()
                    return loss

                self.optimizer.step(closure)

                with torch.no_grad():
                    belief_traj = self.forward_simulate(x0, Sigma0, self.U)
                    _, metrics = self.compute_loss(belief_traj)
            else:
                self.optimizer.zero_grad()
                belief_traj = self.forward_simulate(x0, Sigma0, self.U)
                loss, metrics = self.compute_loss(belief_traj)
                loss.backward()

                # Gradient monitoring
                if iteration == 0 or (iteration % print_every == 0 and verbose):
                    grad_norm = self.U.grad.norm().item()
                    if grad_norm < 1e-8:
                        print(
                            f"  ⚠️  WARNING: Vanishing gradients (norm={grad_norm:.2e})"
                        )
                    elif grad_norm > 100:
                        print(f"  ⚠️  WARNING: Large gradients (norm={grad_norm:.2e})")

                torch.nn.utils.clip_grad_norm_([self.U], max_norm=10.0)
                self.optimizer.step()

            history.append(metrics)

            if verbose and (iteration % print_every == 0 or iteration == max_iters - 1):
                print(
                    f"Iter {iteration:4d} | "
                    f"Loss: {metrics['total']:8.4f} | "
                    f"P_safe: {metrics['p_safe']:.4f} | "
                    f"P_goal: {metrics['p_goal']:.4f} | "
                    f"L_safe: {metrics['safe']:6.3f} | "
                    f"L_goal: {metrics['goal']:6.3f}"
                )

            if early_stop_threshold is not None:
                if (
                    metrics["p_safe"] >= early_stop_threshold
                    and metrics["p_goal"] >= early_stop_threshold
                ):
                    if verbose:
                        print(f"\n✓ Early stopping at iteration {iteration}")
                        print(
                            f"  Achieved P_safe={metrics['p_safe']:.4f}, P_goal={metrics['p_goal']:.4f}"
                        )
                    break

        if verbose:
            print("-" * 80)
            print("Optimization complete!")
            print(f"Final P_safe: {history[-1]['p_safe']:.4f}")
            print(f"Final P_goal: {history[-1]['p_goal']:.4f}")
            print("=" * 80)

        return self.U.detach(), history

    def get_trajectory(self, x0, Sigma0, U):
        """Extract mean trajectory and covariances."""
        with torch.no_grad():
            belief_traj = self.forward_simulate(x0, Sigma0, U)

            means = []
            covs = []

            for t in range(len(belief_traj)):
                mu_t = belief_traj[t].mean.squeeze().cpu().numpy()
                var_t = belief_traj[t].var.squeeze().cpu().numpy()
                Sigma_t = np.diag(var_t)

                means.append(mu_t)
                covs.append(Sigma_t)

            means = np.array(means)
            covs = np.array(covs)

        return means, covs


# =============================================================================
# DIAGNOSTIC FUNCTIONS
# =============================================================================


def diagnose_predicate_calibration(optimizer, x0, Sigma0, x_goal, epsilon_g, obstacle):
    """
    Check if predicates are returning reasonable probabilities.
    This helps identify calibration issues.
    """
    print("\n" + "=" * 80)
    print("PREDICATE CALIBRATION DIAGNOSTIC")
    print("=" * 80)

    # Get trajectory
    belief_traj = optimizer.forward_simulate(x0, Sigma0, optimizer.U)
    means = []
    vars = []

    for t in range(len(belief_traj)):
        means.append(belief_traj[t].mean.squeeze().cpu().detach().numpy())
        vars.append(belief_traj[t].var.squeeze().cpu().detach().numpy())

    means = np.array(means)  # [T, 2]
    vars = np.array(vars)  # [T, 2]

    # Get predicate traces
    safe_pred_trace = optimizer.spec_safe.subformula(belief_traj)  # [B, T, 2]
    goal_pred_trace = optimizer.spec_goal.subformula(belief_traj)  # [B, T, 2]

    safe_pred_lower = safe_pred_trace[0, :, 0].detach().cpu().numpy()  # [T]
    safe_pred_upper = safe_pred_trace[0, :, 1].detach().cpu().numpy()  # [T]
    goal_pred_lower = goal_pred_trace[0, :, 0].detach().cpu().numpy()  # [T]
    goal_pred_upper = goal_pred_trace[0, :, 1].detach().cpu().numpy()  # [T]

    # Compute geometric metrics
    distances_to_goal = np.linalg.norm(means - x_goal.cpu().numpy().squeeze(), axis=1)

    # Check obstacle distances
    obs_x_min, obs_x_max = obstacle["x"]
    obs_y_min, obs_y_max = obstacle["y"]

    inside_obstacle = (
        (means[:, 0] >= obs_x_min)
        & (means[:, 0] <= obs_x_max)
        & (means[:, 1] >= obs_y_min)
        & (means[:, 1] <= obs_y_max)
    )

    # Print analysis
    print("\nGOAL PREDICATE ANALYSIS:")
    print("-" * 80)

    # Find closest approach to goal
    min_dist_idx = np.argmin(distances_to_goal)
    min_dist = distances_to_goal[min_dist_idx]
    goal_prob_lower_at_closest = goal_pred_lower[min_dist_idx]
    goal_prob_upper_at_closest = goal_pred_upper[min_dist_idx]

    print("Closest approach to goal:")
    print(f"  Time step: {min_dist_idx}")
    print(f"  Position: {means[min_dist_idx]}")
    print(f"  Distance: {min_dist:.4f} m")
    print(f"  Goal radius: {epsilon_g} m")
    print(
        f"  P_goal interval at this point: [{goal_prob_lower_at_closest:.4f}, {goal_prob_upper_at_closest:.4f}]"
    )

    if min_dist < epsilon_g:
        print("  ✓ Inside goal region!")
        if goal_prob_lower_at_closest < 0.7:
            print(
                f"  WARNING: Inside goal but P_goal_lower = {goal_prob_lower_at_closest:.4f} < 0.7"
            )
            print(
                "              This suggests conservative bounds are working (expected behavior)"
            )
    else:
        print(f"  ✗ Outside goal region (need {epsilon_g - min_dist:.4f} m closer)")

    print("\nFinal position:")
    print(f"  Distance to goal: {distances_to_goal[-1]:.4f} m")
    print(f"  P_goal interval: [{goal_pred_lower[-1]:.4f}, {goal_pred_upper[-1]:.4f}]")

    print("\nSAFETY PREDICATE ANALYSIS:")
    print("-" * 80)

    n_inside = np.sum(inside_obstacle)
    if n_inside > 0:
        print(f"⚠️  WARNING: {n_inside}/{len(means)} timesteps inside obstacle!")
        inside_indices = np.where(inside_obstacle)[0]
        print(
            f"  Timesteps inside: {inside_indices[:10]}..."
            if len(inside_indices) > 10
            else f"  Timesteps inside: {inside_indices}"
        )
        print(f"  P_safe at violations: {safe_pred_lower[inside_indices]}")
    else:
        print(" No timesteps inside obstacle!")

    min_safe_prob = np.min(safe_pred_lower)
    min_safe_idx = np.argmin(safe_pred_lower)

    print(
        f"\nLowest P_safe: [{safe_pred_lower[min_safe_idx]:.4f}, {safe_pred_upper[min_safe_idx]:.4f}] at t={min_safe_idx}"
    )
    print(f"  Position: {means[min_safe_idx]}")

    # Check interval widths
    goal_interval_widths = goal_pred_upper - goal_pred_lower
    safe_interval_widths = safe_pred_upper - safe_pred_lower

    print("\nINTERVAL WIDTH ANALYSIS:")
    print(
        f"  Goal interval width: mean={np.mean(goal_interval_widths):.4f}, max={np.max(goal_interval_widths):.4f}"
    )
    print(
        f"  Safe interval width: mean={np.mean(safe_interval_widths):.4f}, max={np.max(safe_interval_widths):.4f}"
    )
    print("  (Wider intervals = more uncertainty, expected with conservative bounds)")

    print("=" * 80 + "\n")

    return {
        "min_distance": min_dist,
        "goal_prob_lower_at_closest": goal_prob_lower_at_closest,
        "goal_prob_upper_at_closest": goal_prob_upper_at_closest,
        "n_inside_obstacle": n_inside,
        "min_safe_prob": min_safe_prob,
    }


def diagnose_shapes(optimizer, x0, Sigma0):
    """Check operator output shapes for debugging."""
    print("\n" + "=" * 80)
    print("OPERATOR OUTPUT SHAPE DIAGNOSTIC")
    print("=" * 80)

    belief_traj = optimizer.forward_simulate(x0, Sigma0, optimizer.U)

    safe_trace = optimizer.spec_safe(belief_traj)
    print(f"\nSafety operator output: {safe_trace.shape}")
    print(f"  Expected: [B, T, 2] = [1, {optimizer.N}, 2]")
    print(f"  Status: {'✓ CORRECT' if len(safe_trace.shape) == 3 else '✗ UNEXPECTED'}")

    goal_trace = optimizer.spec_goal(belief_traj)
    print(f"\nGoal operator output: {goal_trace.shape}")
    print(f"  Expected: [B, T, 2] = [1, {optimizer.N}, 2]")
    print(f"  Status: {'✓ CORRECT' if len(goal_trace.shape) == 3 else '✗ UNEXPECTED'}")

    print("\nSample values (first 5 timesteps):")
    print("  P_safe [lower, upper]:")
    for i in range(min(5, safe_trace.shape[1])):
        print(f"    t={i}: [{safe_trace[0, i, 0]:.4f}, {safe_trace[0, i, 1]:.4f}]")
    print("  P_goal [lower, upper]:")
    for i in range(min(5, goal_trace.shape[1])):
        print(f"    t={i}: [{goal_trace[0, i, 0]:.4f}, {goal_trace[0, i, 1]:.4f}]")
    print("=" * 80 + "\n")


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_trajectory_2d(
    means,
    covs,
    obstacles=None,
    goal=None,
    x0=None,
    controls=None,
    dt=0.1,
    save_path=None,
    figsize=(12, 10),
):
    """Visualize 2D trajectory with uncertainty ellipses."""
    N = len(means)
    time = np.arange(N) * dt

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
    ax_traj = fig.add_subplot(gs[0, :])
    ax_x = fig.add_subplot(gs[1, 0])
    ax_y = fig.add_subplot(gs[1, 1])

    # 2D Trajectory
    ax_traj.plot(
        means[:, 0], means[:, 1], "b-", linewidth=2, label="Mean trajectory", zorder=3
    )

    if x0 is not None:
        ax_traj.plot(x0[0], x0[1], "go", markersize=12, label="Start", zorder=4)

    if goal is not None:
        goal_circle = Circle(
            goal["position"],
            goal["radius"],
            color="green",
            alpha=0.2,
            label=f"Goal (r={goal['radius']})",
            zorder=1,
        )
        ax_traj.add_patch(goal_circle)
        ax_traj.plot(
            goal["position"][0], goal["position"][1], "g*", markersize=15, zorder=4
        )

    if obstacles is not None:
        for i, obs in enumerate(obstacles):
            x_min, x_max = obs["x"]
            y_min, y_max = obs["y"]
            width = x_max - x_min
            height = y_max - y_min

            rect = Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="red",
                alpha=0.3,
                label="Obstacle" if i == 0 else None,
                zorder=2,
            )
            ax_traj.add_patch(rect)

    # Uncertainty ellipses (95% confidence)
    k = np.sqrt(5.991)
    step = max(1, N // 10)

    for t in range(0, N, step):
        Sigma_t = covs[t]
        mu_t = means[t]

        eigenvalues, eigenvectors = np.linalg.eigh(Sigma_t)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        width = 2 * k * np.sqrt(eigenvalues[0])
        height = 2 * k * np.sqrt(eigenvalues[1])

        ellipse = MPLEllipse(
            mu_t,
            width,
            height,
            angle=angle,
            facecolor="blue",
            alpha=0.1,
            edgecolor="blue",
            linewidth=0.5,
            zorder=1,
        )
        ax_traj.add_patch(ellipse)

    ax_traj.set_xlabel("x (m)", fontsize=11)
    ax_traj.set_ylabel("y (m)", fontsize=11)
    ax_traj.set_title(
        "(a) 2D Trajectory with 95% Uncertainty Ellipses",
        fontsize=12,
        fontweight="bold",
    )
    ax_traj.legend(loc="best", fontsize=9)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_aspect("equal", adjustable="box")

    # X position vs time
    sigma_x = np.sqrt(covs[:, 0, 0])
    ax_x.fill_between(
        time,
        means[:, 0] - 2 * sigma_x,
        means[:, 0] + 2 * sigma_x,
        alpha=0.3,
        color="blue",
        label="±2σ",
    )
    ax_x.plot(time, means[:, 0], "b-", linewidth=2, label="μ_x(t)")

    if obstacles is not None:
        for obs in obstacles:
            x_min, x_max = obs["x"]
            ax_x.axhline(x_min, color="red", linestyle="--", linewidth=1, alpha=0.7)
            ax_x.axhline(x_max, color="red", linestyle="--", linewidth=1, alpha=0.7)
            ax_x.fill_between(time, x_min, x_max, alpha=0.1, color="red")

    ax_x.set_xlabel("Time (s)", fontsize=11)
    ax_x.set_ylabel("x (m)", fontsize=11)
    ax_x.set_title("(b) X Position vs Time", fontsize=12, fontweight="bold")
    ax_x.legend(loc="best", fontsize=9)
    ax_x.grid(True, alpha=0.3)

    # Y position vs time
    sigma_y = np.sqrt(covs[:, 1, 1])
    ax_y.fill_between(
        time,
        means[:, 1] - 2 * sigma_y,
        means[:, 1] + 2 * sigma_y,
        alpha=0.3,
        color="blue",
        label="±2σ",
    )
    ax_y.plot(time, means[:, 1], "b-", linewidth=2, label="μ_y(t)")

    if obstacles is not None:
        for obs in obstacles:
            y_min, y_max = obs["y"]
            ax_y.axhline(y_min, color="red", linestyle="--", linewidth=1, alpha=0.7)
            ax_y.axhline(y_max, color="red", linestyle="--", linewidth=1, alpha=0.7)
            ax_y.fill_between(time, y_min, y_max, alpha=0.1, color="red")

    ax_y.set_xlabel("Time (s)", fontsize=11)
    ax_y.set_ylabel("y (m)", fontsize=11)
    ax_y.set_title("(c) Y Position vs Time", fontsize=12, fontweight="bold")
    ax_y.legend(loc="best", fontsize=9)
    ax_y.grid(True, alpha=0.3)

    plt.suptitle(
        "Probabilistic STL Motion Planning with Conservative Bounds",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.tight_layout()
    plt.show()

    return fig


def plot_optimization_history(history, save_path=None):
    """Plot optimization convergence metrics."""
    iterations = range(len(history))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    total_loss = [h["total"] for h in history]
    stl_safe = [h["safe"] for h in history]
    stl_goal = [h["goal"] for h in history]
    p_safe = [h["p_safe"] for h in history]
    p_goal = [h["p_goal"] for h in history]

    # Total loss
    axes[0, 0].plot(iterations, total_loss, "b-", linewidth=2)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("(a) Total Loss", fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale("log")

    # STL violation penalties
    axes[0, 1].plot(iterations, stl_safe, "r-", linewidth=2, label="L_safe")
    axes[0, 1].plot(iterations, stl_goal, "g-", linewidth=2, label="L_goal")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("STL Hinge Loss")
    axes[0, 1].set_title("(b) STL Violation Penalties", fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Satisfaction probabilities
    axes[1, 0].plot(iterations, p_safe, "b-", linewidth=2, label="P_safe (lower)")
    axes[1, 0].plot(iterations, p_goal, "g-", linewidth=2, label="P_goal (lower)")
    axes[1, 0].axhline(0.95, color="red", linestyle="--", linewidth=1, label="α=0.95")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Probability")
    axes[1, 0].set_title("(c) STL Satisfaction Probabilities", fontweight="bold")
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Regularization terms
    terminal = [h["terminal"] for h in history]
    control = [h["control"] for h in history]
    smoothness = [h["smoothness"] for h in history]

    axes[1, 1].plot(
        iterations, terminal, "-", linewidth=1.5, label="Terminal", alpha=0.7
    )
    axes[1, 1].plot(iterations, control, "-", linewidth=1.5, label="Control", alpha=0.7)
    axes[1, 1].plot(
        iterations, smoothness, "-", linewidth=1.5, label="Smoothness", alpha=0.7
    )
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Loss Component")
    axes[1, 1].set_title("(d) Regularization Terms", fontweight="bold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale("log")

    plt.suptitle("Optimization Convergence", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Convergence plot saved to: {save_path}")

    plt.show()

    return fig


# =============================================================================
# MAIN EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CORRECTED: Probabilistic STL 2D Motion Planning Example")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    print()

    # Setup environment
    print("Setting up environment...")

    x0 = np.array([0.0, 0.0])
    Sigma0 = 0.1 * np.eye(2)

    x_goal = torch.tensor([[6.0, 8.0]], dtype=torch.float32, device=device)
    epsilon_g = 0.5

    obstacle = {"x": [1.0, 6.0], "y": [3.0, 5.0]}

    N = 100
    dt = 0.15

    print(f"  Start: {x0}")
    print(f"  Goal: {x_goal.cpu().numpy().squeeze()} (radius={epsilon_g})")
    print(f"  Obstacle: x ∈ {obstacle['x']}, y ∈ {obstacle['y']}")
    print(f"  Horizon: N={N}, dt={dt}\n")

    # Setup dynamics
    print("Initializing dynamics model...")

    sigma = 0.1
    dynamics = Dynamics2D(dt=dt, sigma=sigma)
    dynamics.Q_d = dynamics.Q_d.to(device)

    print(f"  Stochastic single-integrator with σ={sigma}")
    print(f"  Process noise: Q_d = {sigma**2 * dt:.4f} * I_2\n")

    # Construct STL specification
    print("Building STL specification...")

    phi_safe = ObstacleAvoidance(obstacle)
    spec_safe = Always(phi_safe, interval=None)

    phi_goal = GoalReached(x_goal, epsilon_g, beta_0=0.05, temperature=1.0)
    spec_goal = Eventually(phi_goal, interval=None)

    print("  Safety: □(avoid obstacle) - using CORRECTED conservative bounds")
    print("  Goal: ◇(reach goal) - using CORRECTED conservative bounds\n")

    # Create optimizer
    print("Creating optimizer...")

    optimizer = ProbabilisticSTLOptimizer(
        dynamics=dynamics,
        spec_safe=spec_safe,
        spec_goal=spec_goal,
        x_goal=x_goal,
        horizon=N,
        dt=dt,
        optimizer_type="adam",
        lr=0.02,
        device=device,
        obstacle_bounds=obstacle,
    )

    print(f"  Weights: {optimizer.weights}\n")

    # Diagnostic check
    diagnose_shapes(optimizer, x0, Sigma0)

    # Run optimization
    print("Running optimization...\n")

    U_opt, history = optimizer.optimize(
        x0=x0,
        Sigma0=Sigma0,
        max_iters=500,
        verbose=True,
        print_every=50,
        early_stop_threshold=0.95,
    )

    # Extract trajectory
    print("\nExtracting optimized trajectory...")

    means, covs = optimizer.get_trajectory(x0, Sigma0, U_opt)

    print(f"  Trajectory shape: {means.shape}")
    print(f"  Final position: {means[-1]}")
    print(
        f"  Distance to goal: {np.linalg.norm(means[-1] - x_goal.cpu().numpy().squeeze()):.4f}\n"
    )

    # Run calibration diagnostic
    print("Running post-optimization diagnostics...")
    stats = diagnose_predicate_calibration(
        optimizer, x0, Sigma0, x_goal, epsilon_g, obstacle
    )

    # Visualize results
    print("Generating visualizations...\n")

    plot_trajectory_2d(
        means=means,
        covs=covs,
        obstacles=[obstacle],
        goal={"position": x_goal.cpu().numpy().squeeze(), "radius": epsilon_g},
        x0=x0,
        controls=U_opt.cpu().numpy().squeeze(),
        dt=dt,
        save_path=None,
    )

    plot_optimization_history(history, save_path=None)

    print("=" * 80)
    print("Example complete!")
    print("=" * 80)
