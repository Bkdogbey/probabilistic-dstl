"""Dimension-aware planning for baseline and figure-eight experiments."""

from __future__ import annotations

import math
import pathlib
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from .utils import (
    DT,
    DETERMINISTIC_VIA_POINTS,
    END_XY,
    EXPERIMENT_DIR,
    FIG8_BASE_HEIGHT,
    FIG8_CHANCE_X_BOUNDS,
    FIG8_CHANCE_Y_BOUNDS,
    FIG8_CHANCE_Z_BOUNDS,
    FIG8_CENTER_X,
    FIG8_CENTER_Y,
    FIG8_CORRIDOR_RADIUS,
    FIG8_CROSSING_HEIGHT,
    FIG8_HALF_WIDTH,
    FIG8_OBSTACLES,
    FIG8_FLIGHT_POINTS,
    FIG8_T,
    FIG8_TOP_HEIGHT,
    FIG8_VERTICAL_HALF,
    FIG8_MAX_CONTROL_DELTA,
    FIG8_RETURN_TOLERANCE,
    FIG8_W_CORRIDOR,
    FIG8_W_PHI,
    FIG8_W_REF_XY,
    FIG8_W_REF_Z,
    FIG8_W_TERMINAL,
    FLIGHT_X_BOUNDS,
    FLIGHT_Y_BOUNDS,
    GOAL,
    INITIAL_VARIANCE,
    OBSTACLES,
    PLANNER_CONFIG,
    PLANNER_ALPHA,
    RESIDUAL_MEAN_PER_FAN,
    RESPONSE_TIME_CONSTANT,
    STATIONARY_RESIDUAL_VARIANCE_PER_FAN,
    START_XY,
    T,
    SAFE_PATH_FLIGHT_POINTS,
    U_MAX,
    Waypoint,
    Z_HEIGHT,
    PlanRepository,
    uncertainty_metadata,
    validate_waypoints,
)
from pdstl.base import BeliefTrajectory
from pdstl.operators import Always, And, Eventually, Or, STL_Formula
from planning.dynamics import Dynamics
from planning.environment import (
    CircularObstaclePredicate,
    Environment,
    extract_trajectory_stats,
    normal_cdf,
)
from planning.planner import Planner, TorchGaussianBelief


class PositionDynamics(Dynamics):
    """First-order command response with stationary empirical residuals.

    Controls move an internal commanded position. The Gaussian mean follows a
    continuous first-order velocity response, discretized exactly over each
    planner step. A zero time constant selects instantaneous response for an
    axis (currently z). The belief is exact at time zero; fan-conditioned
    residual mean and covariance are applied at future steps without random-
    walk accumulation.
    """

    def __init__(
        self, dimension: int, stationary_residual_variance,
        response_time_constant, residual_mean=None,
        dt=0.2, u_max=1.0,
        device='cpu', enforce_return: bool = False,
    ):
        super().__init__(dt=dt, u_max=u_max, device=device)
        variance = torch.as_tensor(
            stationary_residual_variance, dtype=torch.float32, device=device,
        )
        if variance.shape != (dimension,) or not torch.all(torch.isfinite(variance)):
            raise ValueError(
                f'stationary_residual_variance must be a finite {dimension}-vector.'
            )
        if torch.any(variance < 0):
            raise ValueError('stationary_residual_variance must be nonnegative.')
        time_constant = torch.as_tensor(
            response_time_constant, dtype=torch.float32, device=device,
        )
        if (
            time_constant.shape != (dimension,)
            or not torch.all(torch.isfinite(time_constant))
            or torch.any(time_constant < 0)
        ):
            raise ValueError(
                f'response_time_constant must be a finite nonnegative {dimension}-vector.'
            )
        bias = torch.zeros(dimension, dtype=torch.float32, device=device)
        if residual_mean is not None:
            bias = torch.as_tensor(residual_mean, dtype=torch.float32, device=device)
        if bias.shape != (dimension,) or not torch.all(torch.isfinite(bias)):
            raise ValueError(f'residual_mean must be a finite {dimension}-vector.')
        self.residual_covariance = torch.diag(variance)
        self.response_time_constant = time_constant
        self.residual_mean = bias
        # Some generic planner utilities sample ``dyn.Q``. Here Q is an
        # output-residual covariance, not an accumulating process covariance.
        self.Q = self.residual_covariance
        self.enforce_return = enforce_return

    def step(self, x, _covariance, control):
        """Single-step fallback with zero incoming response velocity.

        Full trajectory planning uses :meth:`forward`, which carries response
        velocity between steps. This method remains available to generic
        one-step callers and is exact for disabled (instantaneous) axes.
        """
        velocity = torch.zeros_like(control)
        response, _velocity = self._response_step(x, velocity, control)
        return response + self.residual_mean, self.residual_covariance

    def _response_step(self, position, velocity, control):
        """Apply the exact zero-order-hold solution of the response ODE."""
        enabled = self.response_time_constant > 0
        safe_tau = torch.where(
            enabled, self.response_time_constant, torch.ones_like(self.response_time_constant),
        )
        decay = torch.exp(-self.dt / safe_tau)
        response_displacement = (
            safe_tau * (1.0 - decay) * velocity
            + (self.dt - safe_tau * (1.0 - decay)) * control
        )
        next_position = torch.where(
            enabled,
            position + response_displacement,
            position + control * self.dt,
        )
        next_velocity = torch.where(
            enabled,
            decay * velocity + (1.0 - decay) * control,
            control,
        )
        return next_position, next_velocity

    def bound_control(self, v):
        controls = super().bound_control(v)
        if not self.enforce_return or controls.ndim != 2:
            return controls
        # A closed figure-eight must have zero integrated velocity. Centering
        # the whole sequence enforces that invariant exactly, then a common
        # scale preserves it while respecting the Euclidean speed bound.
        controls = controls - torch.mean(controls, dim=0, keepdim=True)
        max_speed = torch.linalg.vector_norm(controls, dim=1).max()
        scale = torch.clamp(self.u_max / (max_speed + 1e-9), max=1.0)
        return controls * scale

    def forward(self, v_sequence, x0_mean, x0_cov):
        """Roll out command response and the stationary residual belief."""
        controls = self.bound_control(v_sequence)
        response_position = x0_mean
        response_velocity = torch.zeros_like(x0_mean)
        means = [x0_mean]
        covariances = [x0_cov]
        for control in controls:
            response_position, response_velocity = self._response_step(
                response_position, response_velocity, control,
            )
            means.append(response_position + self.residual_mean)
            covariances.append(self.residual_covariance)
        return torch.stack(means).unsqueeze(0), torch.stack(covariances).unsqueeze(0)

    def commanded_trace(
        self, belief_mean_trace: torch.Tensor, controls: torch.Tensor,
    ) -> torch.Tensor:
        """Recover command waypoints from the optimized physical controls."""
        if controls.ndim != 2 or controls.shape[1] != belief_mean_trace.shape[-1]:
            raise ValueError('controls must have shape [T, dimension].')
        start = belief_mean_trace[:, :1, :]
        increments = torch.cumsum(controls * self.dt, dim=0).unsqueeze(0)
        return torch.cat([start, start + increments], dim=1)


class RectangularGoalPredicate(STL_Formula):
    """Probability of containment in a 2D or 3D box."""

    def __init__(self, region):
        super().__init__()
        self.x_min, self.x_max = region["x"]
        self.y_min, self.y_max = region["y"]
        self.z_min, self.z_max = region.get("z", (None, None))

    def robustness_trace(self, belief_trajectory, **kwargs):
        mu, var = extract_trajectory_stats(belief_trajectory)

        mu_x, mu_y = mu[..., 0], mu[..., 1]
        var_x, var_y = var[..., 0], var[..., 1]
        p_x = normal_cdf(self.x_max, mu_x, var_x) - normal_cdf(self.x_min, mu_x, var_x)
        p_y = normal_cdf(self.y_max, mu_y, var_y) - normal_cdf(self.y_min, mu_y, var_y)

        if self.z_min is not None:
            mu_z, var_z = mu[..., 2], var[..., 2]
            p_z = normal_cdf(self.z_max, mu_z, var_z) - normal_cdf(self.z_min, mu_z, var_z)
            p_goal = torch.clamp(p_x * p_y * p_z, min=0.0, max=1.0)
        else:
            p_goal = torch.clamp(p_x * p_y, min=0.0, max=1.0)

        return torch.stack([p_goal, p_goal], dim=-1)


class RectangularObstaclePredicate(STL_Formula):
    """Probability of being outside a 2D or 3D obstacle box.

    The planner propagates diagonal Gaussian covariance, so coordinate events
    are independent. Computing ``1 - P(inside)`` is smooth and exact for that
    model; selecting the largest single-face escape probability with max()
    created discontinuous gradients whenever the active obstacle face changed.
    """

    def __init__(self, region):
        super().__init__()
        self.x_min, self.x_max = region["x"]
        self.y_min, self.y_max = region["y"]
        self.z_min, self.z_max = region.get("z", (None, None))

    def robustness_trace(self, belief_trajectory, **kwargs):
        mu, var = extract_trajectory_stats(belief_trajectory)

        mu_x, mu_y = mu[..., 0], mu[..., 1]
        var_x, var_y = var[..., 0], var[..., 1]
        p_x_inside = normal_cdf(self.x_max, mu_x, var_x) - normal_cdf(
            self.x_min, mu_x, var_x,
        )
        p_y_inside = normal_cdf(self.y_max, mu_y, var_y) - normal_cdf(
            self.y_min, mu_y, var_y,
        )
        p_inside = p_x_inside * p_y_inside

        if self.z_min is not None:
            mu_z, var_z = mu[..., 2], var[..., 2]
            p_z_inside = normal_cdf(self.z_max, mu_z, var_z) - normal_cdf(
                self.z_min, mu_z, var_z,
            )
            p_inside = p_inside * p_z_inside

        p_safe = torch.clamp(1.0 - p_inside, min=0.0, max=1.0)
        return torch.stack([p_safe, p_safe], dim=-1)


class PositionEnvironment(Environment):
    """Environment with optional altitude bounds."""

    def __init__(self, device="cpu"):
        super().__init__(device=device)
        self.time_windowed_bounds = []
        self.bounds_interval = None

    def add_obstacle(self, x_range, y_range, z_range=None):
        obstacle = {"x": x_range, "y": y_range}
        if z_range is not None:
            obstacle["z"] = z_range
        self.obstacles.append(obstacle)

    def add_visit_region(self, x_range, y_range, z_range=None):
        region = {"x": x_range, "y": y_range}
        if z_range is not None:
            region["z"] = z_range
        self.visit_regions.append(region)

    def add_timed_visit_region(self, x_range, y_range, interval, z_range=None, label=None):
        region = {"x": x_range, "y": y_range, "interval": interval, "label": label}
        if z_range is not None:
            region["z"] = z_range
        self.timed_visit_regions.append(region)

    def add_time_windowed_bounds(self, x_range, y_range, z_range, interval, label=None):
        """Add bounds that must hold throughout a time interval."""
        self.time_windowed_bounds.append(
            {"x": x_range, "y": y_range, "z": z_range, "interval": interval, "label": label}
        )

    def set_goal(self, x_range, y_range, z_range=None, interval=None):
        self.goal = {"x": x_range, "y": y_range}
        if z_range is not None:
            self.goal["z"] = z_range
        self.goal_interval = interval

    def set_bounds(self, x_range, y_range, z_range=None, interval=None):
        self.bounds = {"x": x_range, "y": y_range}
        if z_range is not None:
            self.bounds["z"] = z_range
        self.bounds_interval = interval

    def get_predicates(self):
        preds = {
            "obstacles": [],
            "visit": [],
            "timed_visit": [],
            "choice_region_groups": [],
            "time_windowed_bounds": [],
            "goal": None,
        }

        if self.goal:
            preds["goal"] = RectangularGoalPredicate(self.goal)

        for region in self.visit_regions:
            preds["visit"].append(RectangularGoalPredicate(region))

        for region in self.timed_visit_regions:
            preds["timed_visit"].append(
                {
                    "predicate": RectangularGoalPredicate(region),
                    "interval": region["interval"],
                    "label": region.get("label", None),
                }
            )

        for group in self.choice_region_groups:
            group_preds = [RectangularGoalPredicate(region) for region in group["regions"]]
            preds["choice_region_groups"].append(
                {
                    "predicates": group_preds,
                    "interval": group.get("interval", None),
                    "label": group.get("label", None),
                }
            )

        for region in self.time_windowed_bounds:
            preds["time_windowed_bounds"].append(
                {
                    "predicate": RectangularGoalPredicate(region),
                    "interval": region["interval"],
                    "label": region.get("label", None),
                }
            )

        if self.obstacles or self.circle_obstacles or self.moving_obstacles:
            obs_preds = [RectangularObstaclePredicate(obs) for obs in self.obstacles]
            obs_preds.extend(
                CircularObstaclePredicate(obs, device=self.device)
                for obs in self.circle_obstacles
            )
            # Moving obstacles stay 2D-only (Crazyflie defines none; the shared
            # MovingRectangularObstaclePredicate isn't 3D-extended in this pass).
            preds["obstacles"] = obs_preds

        return preds

    def get_specification(self, T, t_goal_start=0, t_constraints_start=1):
        """Build the combined reach, avoid, and bounds specification."""
        preds = self.get_predicates()
        specs = []

        if preds["goal"]:
            goal_interval = self.goal_interval
            if goal_interval is None:
                goal_interval = [t_goal_start, T]
            specs.append(Eventually(preds["goal"], interval=goal_interval))

        for visit_pred in preds["visit"]:
            specs.append(Eventually(visit_pred, interval=[0, T]))

        for item in preds["timed_visit"]:
            specs.append(Eventually(item["predicate"], interval=item["interval"]))

        for group in preds["choice_region_groups"]:
            group_preds = group["predicates"]
            if not group_preds:
                continue
            interval = group.get("interval", None)
            if interval is None:
                interval = [0, T]
            choice_spec = Eventually(group_preds[0], interval=interval)
            for pred in group_preds[1:]:
                choice_spec = Or(choice_spec, Eventually(pred, interval=interval))
            specs.append(choice_spec)

        for item in preds["time_windowed_bounds"]:
            specs.append(Always(item["predicate"], interval=item["interval"]))

        if preds["obstacles"]:
            obs_preds = preds["obstacles"]
            current_safe_formula = obs_preds[0]
            for i in range(1, len(obs_preds)):
                current_safe_formula = And(current_safe_formula, obs_preds[i])
            specs.append(Always(current_safe_formula, interval=[t_constraints_start, T]))

        if self.bounds is not None:
            bounds_interval = self.bounds_interval
            if bounds_interval is None:
                bounds_interval = [t_constraints_start, T]
            specs.append(
                Always(RectangularGoalPredicate(self.bounds), interval=bounds_interval)
            )

        if not specs:
            raise ValueError("No constraints defined in environment.")

        combined_spec = specs[0]
        for i in range(1, len(specs)):
            combined_spec = And(combined_spec, specs[i])

        return combined_spec


class PositionPlanner(Planner):
    """Planner with dimension-aware guidance and optional path tracking."""

    def _goal_dist_loss(self, mean_trace):
        if self.env.goal is None:
            return torch.tensor(0.0, device=self.device)
        gx = sum(self.env.goal["x"]) / 2.0
        gy = sum(self.env.goal["y"]) / 2.0
        goal_z = self.env.goal.get("z")
        if goal_z is not None:
            gz = sum(goal_z) / 2.0
            goal_center = torch.tensor([[gx, gy, gz]], device=self.device)
            return torch.sum((mean_trace[:, -1, :3] - goal_center) ** 2)
        goal_center = torch.tensor([[gx, gy]], device=self.device)
        return torch.sum((mean_trace[:, -1, :2] - goal_center) ** 2)

    def _obs_repulsion_loss(self, mean_trace):
        loss = torch.tensor(0.0, device=self.device)
        margin = self.cfg["obs_margin"]

        for obs in self.env.obstacles:
            ranges = [obs["x"], obs["y"]]
            if obs.get("z") is not None:
                ranges.append(obs["z"])
            position = mean_trace[:, :, :len(ranges)]
            lower = torch.tensor(
                [axis[0] for axis in ranges], device=self.device, dtype=position.dtype,
            )
            upper = torch.tensor(
                [axis[1] for axis in ranges], device=self.device, dtype=position.dtype,
            )
            center = (lower + upper) / 2.0
            half_extent = (upper - lower) / 2.0
            relative = torch.abs(position - center) - half_extent
            outside_distance = torch.norm(torch.relu(relative), dim=2)
            inside_distance = torch.minimum(
                torch.max(relative, dim=2).values,
                torch.zeros_like(outside_distance),
            )
            signed_distance = outside_distance + inside_distance
            loss = loss + torch.sum(torch.relu(margin - signed_distance) ** 2)

        for obs in self.env.circle_obstacles:
            center = torch.tensor([obs["center"]], device=self.device)
            radius = obs["radius"] + margin
            pos_dim = center.shape[-1]
            dists = torch.norm(mean_trace[:, :, :pos_dim] - center, dim=2)
            loss = loss + torch.sum(torch.relu(radius - dists) ** 2)

        for obs in self.env.moving_obstacles:
            # Moving obstacles stay 2D-only; unused by this experiment.
            ox = torch.as_tensor(obs["x_traj"], device=self.device)
            oy = torch.as_tensor(obs["y_traj"], device=self.device)
            centers = torch.stack([ox, oy], dim=1).unsqueeze(0)
            radius = max(obs["width"], obs["height"]) / 2.0 + margin
            dists = torch.norm(mean_trace[:, :, :2] - centers, dim=2)
            loss = loss + torch.sum(torch.relu(radius - dists) ** 2)

        return loss

    def _guidance_loss(self, mean_trace, u_seq):
        """Add reference, corridor, and terminal quality to shared guidance."""
        loss = super()._guidance_loss(mean_trace, u_seq)
        ref = getattr(self, 'reference_trajectory', None)
        if ref is None:
            return loss

        ref = ref.to(device=self.device, dtype=mean_trace.dtype)
        ref_xy = ref[:, :2].unsqueeze(0)
        ref_z = ref[:, 2].unsqueeze(0)
        loss_ref_xy = torch.sum((mean_trace[:, :, :2] - ref_xy) ** 2)
        loss_ref_z = torch.sum((mean_trace[:, :, 2] - ref_z) ** 2)
        loss_terminal = torch.sum((mean_trace[:, -1, :3] - ref[0, :3]) ** 2)
        reference_distance = torch.norm(mean_trace[:, :, :3] - ref.unsqueeze(0), dim=2)
        corridor_radius = self.cfg.get('corridor_radius', float('inf'))
        loss_corridor = torch.sum(torch.relu(reference_distance - corridor_radius) ** 2)

        return (
            loss
            + self.cfg.get('w_ref_xy', 0.0) * loss_ref_xy
            + self.cfg.get('w_ref_z', 0.0) * loss_ref_z
            + self.cfg.get('w_terminal', 0.0) * loss_terminal
            + self.cfg.get('w_corridor', 0.0) * loss_corridor
        )


def nominal_baseline_waypoints(n_points: int = T + 1) -> list[tuple[float, float, float]]:
    if n_points < 2:
        raise ValueError('n_points must be at least 2.')
    y_span = END_XY[1] - START_XY[1]
    via_progress = np.array([(y - START_XY[1]) / y_span for _, y in DETERMINISTIC_VIA_POINTS])
    harmonics = np.arange(1, len(DETERMINISTIC_VIA_POINTS) + 1)
    def linear_x(s):
        return START_XY[0] + s * (END_XY[0] - START_XY[0])

    targets = np.array([
        x - linear_x(progress)
        for (x, _), progress in zip(DETERMINISTIC_VIA_POINTS, via_progress)
    ])
    coefficients = np.linalg.solve(
        np.sin(np.pi * np.outer(via_progress, harmonics)), targets,
    )
    progress = np.linspace(0.0, 1.0, n_points)
    x = linear_x(progress) + np.sin(np.pi * np.outer(progress, harmonics)) @ coefficients
    y = START_XY[1] + progress * y_span
    return [(float(px), float(py), Z_HEIGHT) for px, py in zip(x, y)]


def build_environment_baseline() -> PositionEnvironment:
    environment = PositionEnvironment()
    environment.set_bounds(FLIGHT_X_BOUNDS, FLIGHT_Y_BOUNDS)
    for obstacle in OBSTACLES:
        environment.add_obstacle(list(obstacle['x']), list(obstacle['y']))
    environment.set_goal(list(GOAL['x']), list(GOAL['y']))
    return environment


def build_planner_baseline(fan: int) -> tuple[PositionPlanner, PositionDynamics, PositionEnvironment]:
    dynamics = PositionDynamics(
        2,
        STATIONARY_RESIDUAL_VARIANCE_PER_FAN[fan][:2],
        RESPONSE_TIME_CONSTANT[:2],
        RESIDUAL_MEAN_PER_FAN[fan][:2],
        dt=DT,
        u_max=U_MAX,
    )
    environment = build_environment_baseline()
    planner = PositionPlanner(dynamics, environment, T, config=dict(PLANNER_CONFIG))
    return planner, dynamics, environment


def initial_belief_baseline(fan: int) -> tuple[torch.Tensor, torch.Tensor]:
    if fan not in STATIONARY_RESIDUAL_VARIANCE_PER_FAN:
        raise ValueError(f'Invalid fan setting {fan}.')
    mean = torch.tensor(START_XY, dtype=torch.float32)
    covariance = torch.eye(2, dtype=torch.float32) * INITIAL_VARIANCE
    return mean, covariance


def initial_controls_baseline() -> torch.Tensor:
    positions = np.array(nominal_baseline_waypoints(T + 1))[:, :2]
    return torch.tensor(np.diff(positions, axis=0) / DT, dtype=torch.float32)


def nominal_figure8_waypoints(n_points: int = FIG8_T + 1) -> list[tuple[float, float, float]]:
    """Return the smooth, altitude-varying vertical Gerono figure-eight."""
    if n_points < 2:
        raise ValueError('n_points must be at least 2')
    theta = np.linspace(np.pi, 3 * np.pi, n_points)
    x = FIG8_CENTER_X + FIG8_HALF_WIDTH * np.sin(2 * theta)
    y = FIG8_CENTER_Y + FIG8_VERTICAL_HALF * np.cos(theta)
    # Tie altitude to vertical progress through the figure: bottom -> crossing
    # -> top -> crossing -> bottom. A monotone cubic places both centre-crossing
    # passes at the configured intermediate height without abrupt reversals.
    vertical_progress = (y - (FIG8_CENTER_Y - FIG8_VERTICAL_HALF)) / (2 * FIG8_VERTICAL_HALF)
    crossing_gain = FIG8_CROSSING_HEIGHT - FIG8_BASE_HEIGHT
    top_gain = FIG8_TOP_HEIGHT - FIG8_BASE_HEIGHT
    quadratic = 8 * crossing_gain - top_gain
    cubic = 2 * top_gain - 8 * crossing_gain
    z = FIG8_BASE_HEIGHT + quadratic * vertical_progress**2 + cubic * vertical_progress**3
    return [(float(xi), float(yi), float(zi)) for xi, yi, zi in zip(x, y, z)]


def build_environment_figure8() -> PositionEnvironment:
    """Build the pdSTL Environment for the figure8 mission.

    The specification enforces workspace and obstacle avoidance. Reference
    tracking keeps the solution near the smooth 3D figure-eight.
    """
    env = PositionEnvironment()
    # Commanded waypoints use the narrower FIG8_*_BOUNDS. The probabilistic
    # workspace describes the physical room available to the uncertain state;
    # it must extend beyond a nominal curve that intentionally touches its
    # commanded y/z limits, otherwise boundary containment can never reach
    # alpha even for a perfectly closed mean trajectory.
    env.set_bounds(
        x_range=FIG8_CHANCE_X_BOUNDS,
        y_range=FIG8_CHANCE_Y_BOUNDS,
        z_range=FIG8_CHANCE_Z_BOUNDS,
    )
    for obs in FIG8_OBSTACLES:
        env.add_obstacle(x_range=list(obs['x']), y_range=list(obs['y']), z_range=list(obs['z']))
    return env


def build_planner_figure8(
    fan: int = 2,
) -> tuple[PositionPlanner, PositionDynamics, PositionEnvironment]:
    """Build a (Planner, dynamics, environment) for the figure8 mission.

    Attach the analytical path as the figure-eight reference trajectory.
    """
    dynamics = PositionDynamics(
        3,
        STATIONARY_RESIDUAL_VARIANCE_PER_FAN[fan],
        RESPONSE_TIME_CONSTANT,
        RESIDUAL_MEAN_PER_FAN[fan],
        dt=DT,
        u_max=U_MAX,
        enforce_return=True,
    )
    env = build_environment_figure8()
    planner = PositionPlanner(dynamics, env, FIG8_T, config={
        **PLANNER_CONFIG,
        'w_phi': FIG8_W_PHI,
        'w_ref_xy': FIG8_W_REF_XY,
        'w_ref_z': FIG8_W_REF_Z,
        'w_terminal': FIG8_W_TERMINAL,
        'w_corridor': FIG8_W_CORRIDOR,
        'corridor_radius': FIG8_CORRIDOR_RADIUS,
    })
    planner.reference_trajectory = torch.tensor(
        nominal_figure8_waypoints(FIG8_T + 1), dtype=torch.float32,
    )
    return planner, dynamics, env


def x0_belief_figure8(fan_speed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the known figure-eight start with exactly zero covariance."""
    if fan_speed not in STATIONARY_RESIDUAL_VARIANCE_PER_FAN:
        raise ValueError(f'Invalid fan setting {fan_speed}.')
    start = nominal_figure8_waypoints(n_points=2)[0]
    mean = torch.tensor(start, dtype=torch.float32)
    cov = torch.eye(3, dtype=torch.float32) * INITIAL_VARIANCE
    assert mean.shape == (3,), f'x0 mean must be 3D, got {tuple(mean.shape)}'
    assert cov.shape == (3, 3), f'x0 cov must be 3x3, got {tuple(cov.shape)}'
    return mean, cov


def nominal_initial_controls_figure8() -> torch.Tensor:
    """Return FIG8_T controls diffed from FIG8_T + 1 nominal states."""
    wps = np.array(nominal_figure8_waypoints(n_points=FIG8_T + 1))
    vels_xyz = np.diff(wps, axis=0) / DT
    init_u = torch.tensor(vels_xyz, dtype=torch.float32)
    assert init_u.shape == (FIG8_T, 3), f'init_u must be [{FIG8_T},3], got {tuple(init_u.shape)}'
    return init_u


def figure8_obstacle_clearance(curve_xyz: np.ndarray, obs: dict) -> float:
    """Minimum point-to-box distance; zero means collision."""
    def _axis_clearance(pos: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.maximum(np.maximum(lo - pos, 0.0), pos - hi)

    dx = _axis_clearance(curve_xyz[:, 0], *obs['x'])
    dy = _axis_clearance(curve_xyz[:, 1], *obs['y'])
    dz = _axis_clearance(curve_xyz[:, 2], *obs['z'])
    return float(np.min(np.sqrt(dx**2 + dy**2 + dz**2)))


def figure8_min_clearances(
    curve_xyz: np.ndarray, obstacles: list[dict] | None = None,
) -> dict[str, float]:
    """Return minimum clearance for each figure-eight obstacle."""
    obs_list = FIG8_OBSTACLES if obstacles is None else obstacles
    return {obs['name']: figure8_obstacle_clearance(curve_xyz, obs) for obs in obs_list}


def terminal_return_error(waypoints: list[tuple[float, float, float]]) -> float:
    """Euclidean distance between the final and initial waypoints."""
    return math.dist(waypoints[0], waypoints[-1])


def maximum_control_delta(
    waypoints: list[tuple[float, float, float]], dt: float = DT,
) -> float:
    """Largest change in consecutive Cartesian velocity commands."""
    if len(waypoints) < 3:
        return 0.0
    controls = np.diff(np.asarray(waypoints, dtype=float), axis=0) / dt
    return float(np.max(np.linalg.norm(np.diff(controls, axis=0), axis=1)))


def maximum_reference_deviation(
    waypoints: list[tuple[float, float, float]],
) -> float:
    """Maximum phase-aligned distance from the analytical figure-eight."""
    if not waypoints:
        return float('inf')
    actual = np.asarray(waypoints, dtype=float)
    reference = np.asarray(nominal_figure8_waypoints(len(waypoints)), dtype=float)
    return float(np.max(np.linalg.norm(actual - reference, axis=1)))


@dataclass(frozen=True)
class Scenario(ABC):
    name: str
    dimension: int
    horizon: int
    flight_points: int

    @property
    @abstractmethod
    def obstacles(self) -> list[dict]: ...

    @abstractmethod
    def nominal_waypoints(self, n_points: int | None = None) -> list[Waypoint]: ...

    @abstractmethod
    def build_planner(self, fan: int): ...

    @abstractmethod
    def initial_belief(self, fan: int): ...

    @abstractmethod
    def initial_controls(self): ...

    @abstractmethod
    def build_environment(self): ...

    def start(self) -> Waypoint:
        return self.nominal_waypoints(2)[0]

    def validate(self, waypoints: list[Waypoint], *, label: str = 'Waypoint') -> None:
        validate_waypoints(waypoints, scenario=self.name, label=label)


@dataclass(frozen=True)
class BaselineScenario(Scenario):
    name: str = 'baseline'
    dimension: int = 2
    horizon: int = T
    flight_points: int = SAFE_PATH_FLIGHT_POINTS
    obstacles: list[dict] = None

    def __post_init__(self):
        object.__setattr__(self, 'obstacles', OBSTACLES)

    def nominal_waypoints(self, n_points=None):
        return nominal_baseline_waypoints(n_points or self.horizon + 1)

    def build_planner(self, fan): return build_planner_baseline(fan)
    def initial_belief(self, fan): return initial_belief_baseline(fan)
    def initial_controls(self): return initial_controls_baseline()
    def build_environment(self): return build_environment_baseline()


@dataclass(frozen=True)
class FigureEightScenario(Scenario):
    name: str = 'figure8'
    dimension: int = 3
    horizon: int = FIG8_T
    flight_points: int = FIG8_FLIGHT_POINTS
    obstacles: list[dict] = None

    def __post_init__(self):
        object.__setattr__(self, 'obstacles', FIG8_OBSTACLES)

    def nominal_waypoints(self, n_points=None):
        return nominal_figure8_waypoints(n_points or self.horizon + 1)

    def validate(self, waypoints, *, label='Waypoint'):
        super().validate(waypoints, label=label)
        return_error = terminal_return_error(waypoints)
        if return_error > FIG8_RETURN_TOLERANCE:
            raise ValueError(
                f'{label} path does not close: return error {return_error:.3f} m '
                f'exceeds {FIG8_RETURN_TOLERANCE:.3f} m.'
            )
        reference_deviation = maximum_reference_deviation(waypoints)
        if reference_deviation > FIG8_CORRIDOR_RADIUS:
            raise ValueError(
                f'{label} path leaves figure-eight corridor: maximum deviation '
                f'{reference_deviation:.3f} m exceeds {FIG8_CORRIDOR_RADIUS:.3f} m.'
            )
        control_delta = maximum_control_delta(waypoints)
        if control_delta > FIG8_MAX_CONTROL_DELTA:
            raise ValueError(
                f'{label} path is not smooth: maximum control change '
                f'{control_delta:.3f} m/s exceeds {FIG8_MAX_CONTROL_DELTA:.3f} m/s.'
            )

    def build_planner(self, fan): return build_planner_figure8(fan)
    def initial_belief(self, fan): return x0_belief_figure8(fan)
    def initial_controls(self): return nominal_initial_controls_figure8()
    def build_environment(self): return build_environment_figure8()


def get_scenario(name: str) -> Scenario:
    scenarios = {'baseline': BaselineScenario(), 'figure8': FigureEightScenario()}
    try:
        return scenarios[name]
    except KeyError as exc:
        raise ValueError(f'Unknown Crazyflie scenario {name!r}.') from exc


def _satisfaction(planner, controls, mean, covariance) -> float:
    normalized = torch.clamp(controls / planner.dyn.u_max, -0.99, 0.99)
    unconstrained = 0.5 * torch.log((1 + normalized) / (1 - normalized))
    with torch.no_grad():
        mean_trace, covariance_trace = planner.dyn(unconstrained, mean, covariance)
        beliefs = [
            TorchGaussianBelief(mean_trace[:, step, :], covariance_trace[:, step])
            for step in range(planner.T + 1)
        ]
        return planner.env.get_specification(planner.T)(BeliefTrajectory(beliefs))[0, 0, 0].item()


class PlanningService:
    def __init__(self, plans: PlanRepository | None = None) -> None:
        self.plans = plans or PlanRepository()

    def run(self, *, fan: int, scenario: str, plot: bool = False) -> pathlib.Path:
        model = get_scenario(scenario)
        planner, dynamics, environment = model.build_planner(fan)
        initial_mean, initial_covariance = model.initial_belief(fan)
        initial_controls = model.initial_controls()
        rho_before = _satisfaction(planner, initial_controls, initial_mean, initial_covariance)
        best_mean, _covariance, controls, rho_after, _history = planner._optimize_window(
            initial_mean, initial_covariance, init_guess=initial_controls, verbose=True,
        )
        commanded_trace = dynamics.commanded_trace(best_mean, controls)
        positions = commanded_trace.squeeze(0).cpu().numpy()
        predicted_positions = best_mean.squeeze(0).cpu().numpy()
        if model.dimension == 2:
            waypoints = [(float(x), float(y), Z_HEIGHT) for x, y in positions]
        else:
            waypoints = [tuple(map(float, point)) for point in positions]
        if float(rho_after) < PLANNER_ALPHA:
            raise RuntimeError(
                f'Refusing to save fan-{fan} {scenario} plan: exact satisfaction '
                f'{float(rho_after):.4f} is below alpha={PLANNER_ALPHA:.4f}.'
            )
        model.validate(waypoints, label='Generated waypoint')
        geometry_metadata = {}
        if scenario == 'figure8':
            geometry_metadata = {
                'return_tolerance': FIG8_RETURN_TOLERANCE,
                'return_error': terminal_return_error(waypoints),
                'corridor_radius': FIG8_CORRIDOR_RADIUS,
                'max_reference_deviation': maximum_reference_deviation(waypoints),
                'max_control_delta_limit': FIG8_MAX_CONTROL_DELTA,
                'max_control_delta': maximum_control_delta(waypoints),
            }
        path = self.plans.save(fan, scenario, waypoints, {
            **uncertainty_metadata(fan),
            'rho_before': round(rho_before, 4),
            'rho_after': round(float(rho_after), 4),
            'alpha': PLANNER_ALPHA,
            'T': planner.T,
            'dt': DT,
            'generated': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            **geometry_metadata,
        })
        print(f'Wrote {len(waypoints)} waypoints to {path}')
        if plot:
            self._plot(
                model, environment, waypoints, predicted_positions,
                fan, float(rho_after),
            )
        return path

    @staticmethod
    def _plot(
        model: Scenario,
        environment,
        waypoints,
        predicted_positions,
        fan: int,
        satisfaction: float,
    ) -> None:
        import matplotlib.pyplot as plt

        from .utils import draw_environment_2d, draw_environment_3d
        from visualization.style import PALETTE, save_figure

        nominal = np.array(model.nominal_waypoints(model.flight_points))
        planned = np.array(waypoints)
        predicted = np.asarray(predicted_positions, dtype=float)
        standard_deviation = np.sqrt(
            np.asarray(STATIONARY_RESIDUAL_VARIANCE_PER_FAN[fan])
        )
        uncertainty_indices = np.linspace(1, len(planned) - 1, 8, dtype=int)
        if model.dimension == 2:
            figure, axis = plt.subplots(figsize=(7, 7))
            draw_environment_2d(axis, environment)
            axis.plot(
                nominal[:, 0], nominal[:, 1], '--',
                color=PALETTE['plan']['stroke'], alpha=0.65, label='nominal',
            )
            axis.plot(
                planned[:, 0], planned[:, 1], '-',
                color=PALETTE['ego']['stroke'], label='pdSTL command',
            )
            axis.plot(
                predicted[:, 0], predicted[:, 1], '-',
                color=PALETTE['lane']['stroke'], label='predicted actual mean',
            )
            axis.errorbar(
                predicted[uncertainty_indices, 0], predicted[uncertainty_indices, 1],
                xerr=standard_deviation[0], yerr=standard_deviation[1],
                fmt='none', ecolor=PALETTE['lane']['stroke'], alpha=0.35,
                capsize=2, label='1σ residual uncertainty',
            )
        else:
            figure = plt.figure(figsize=(8, 8))
            axis = figure.add_subplot(projection='3d')
            axis.view_init(elev=22, azim=-50)
            draw_environment_3d(
                axis, environment, fit_points=np.vstack([nominal, planned, predicted]),
            )
            axis.plot(
                *nominal.T, '--', color=PALETTE['plan']['stroke'],
                alpha=0.65, label='nominal',
            )
            axis.plot(
                *planned.T, '-', color=PALETTE['ego']['stroke'],
                label='pdSTL command',
            )
            axis.plot(
                *predicted.T, '-', color=PALETTE['lane']['stroke'],
                label='predicted actual mean',
            )
            axis.errorbar(
                predicted[uncertainty_indices, 0],
                predicted[uncertainty_indices, 1],
                predicted[uncertainty_indices, 2],
                xerr=standard_deviation[0],
                yerr=standard_deviation[1],
                zerr=standard_deviation[2],
                fmt='none', ecolor=PALETTE['lane']['stroke'], alpha=0.35,
                capsize=2, label='1σ residual uncertainty',
            )
        axis.set_title(
            f'{model.name} fan {fan}: P(sat)={satisfaction:.3f}, '
            f'α={PLANNER_ALPHA:.2f}'
        )
        axis.legend()
        figure.tight_layout()
        output = EXPERIMENT_DIR / 'plots' / f'planning_{model.name}_fan{fan:02d}'
        written = save_figure(figure, output, formats=('png', 'pdf'))
        plt.close(figure)
        print(f'Plots saved to {written["png"]} and {written["pdf"]}')
