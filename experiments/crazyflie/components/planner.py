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
    FIG8_CENTER_X,
    FIG8_CENTER_Y,
    FIG8_HALF_WIDTH,
    FIG8_MIDPOINT_T_END,
    FIG8_MIDPOINT_T_START,
    FIG8_MIDPOINT_Z,
    FIG8_OBSTACLES,
    FIG8_FLIGHT_POINTS,
    FIG8_T,
    FIG8_W_REF_XY,
    FIG8_W_REF_Z,
    FIG8_W_TERMINAL,
    FIG8_X_BOUNDS,
    FIG8_Y_BOUNDS,
    FIG8_Z_AMPLITUDE,
    FIG8_Z_BASE,
    FIG8_Z_BOUNDS,
    FLIGHT_X_BOUNDS,
    FLIGHT_Y_BOUNDS,
    GOAL,
    OBSTACLES,
    PLANNER_CONFIG,
    PLANNER_ALPHA,
    Q_STD,
    SIGMA0_PER_FAN,
    START_XY,
    T,
    SAFE_PATH_FLIGHT_POINTS,
    U_MAX,
    Waypoint,
    Z_HEIGHT,
    PlanRepository,
    validate_waypoints,
)
from pdstl.base import BeliefTrajectory
from pdstl.operators import Always, And, Eventually, Or, STL_Formula
from planning.dynamics import SingleIntegrator
from planning.environment import (
    CircularObstaclePredicate,
    Environment,
    extract_trajectory_stats,
    normal_cdf,
)
from planning.planner import Planner, TorchGaussianBelief


class PositionDynamics(SingleIntegrator):
    def __init__(self, dimension: int, dt=0.2, u_max=1.0, q_std=0.05, device='cpu'):
        super().__init__(dt=dt, u_max=u_max, q_std=q_std, device=device)
        self.Q = torch.eye(dimension, device=self.device) * q_std**2


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
    """Probability of being outside a 2D or 3D obstacle box."""

    def __init__(self, region):
        super().__init__()
        self.x_min, self.x_max = region["x"]
        self.y_min, self.y_max = region["y"]
        self.z_min, self.z_max = region.get("z", (None, None))

    def robustness_trace(self, belief_trajectory, **kwargs):
        mu, var = extract_trajectory_stats(belief_trajectory)

        mu_x, mu_y = mu[..., 0], mu[..., 1]
        var_x, var_y = var[..., 0], var[..., 1]
        probs = [
            normal_cdf(self.x_min, mu_x, var_x),
            1.0 - normal_cdf(self.x_max, mu_x, var_x),
            normal_cdf(self.y_min, mu_y, var_y),
            1.0 - normal_cdf(self.y_max, mu_y, var_y),
        ]

        if self.z_min is not None:
            mu_z, var_z = mu[..., 2], var[..., 2]
            probs.append(normal_cdf(self.z_min, mu_z, var_z))
            probs.append(1.0 - normal_cdf(self.z_max, mu_z, var_z))

        stacked_probs = torch.stack(probs, dim=0)
        p_safe, _ = torch.max(stacked_probs, dim=0)

        return torch.stack([p_safe, p_safe], dim=-1)


class PositionEnvironment(Environment):
    """Environment with optional altitude bounds."""

    def __init__(self, device="cpu"):
        super().__init__(device=device)
        self.time_windowed_bounds = []

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

    def set_bounds(self, x_range, y_range, z_range=None):
        self.bounds = {"x": x_range, "y": y_range}
        if z_range is not None:
            self.bounds["z"] = z_range

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
            specs.append(
                Always(RectangularGoalPredicate(self.bounds), interval=[t_constraints_start, T])
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
            cx = (obs["x"][0] + obs["x"][1]) / 2.0
            cy = (obs["y"][0] + obs["y"][1]) / 2.0
            radius = max(obs["x"][1] - obs["x"][0], obs["y"][1] - obs["y"][0]) / 2.0 + margin
            obs_z = obs.get("z")
            if obs_z is not None:
                cz = (obs_z[0] + obs_z[1]) / 2.0
                center = torch.tensor([[cx, cy, cz]], device=self.device)
                radius = max(radius, (obs_z[1] - obs_z[0]) / 2.0 + margin)
                dists = torch.norm(mean_trace[:, :, :3] - center, dim=2)
            else:
                center = torch.tensor([[cx, cy]], device=self.device)
                dists = torch.norm(mean_trace[:, :, :2] - center, dim=2)
            loss = loss + torch.sum(torch.relu(radius - dists) ** 2)

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

    def _compute_loss(self, mean_trace, u_seq, p_all, loss_fn):
        """Add figure-eight reference and terminal-return losses when configured."""
        loss = super()._compute_loss(mean_trace, u_seq, p_all, loss_fn)
        ref = getattr(self, 'reference_trajectory', None)
        if ref is None:
            return loss

        ref = ref.to(device=self.device, dtype=mean_trace.dtype)
        ref_xy = ref[:, :2].unsqueeze(0)
        ref_z = ref[:, 2].unsqueeze(0)
        loss_ref_xy = torch.sum((mean_trace[:, :, :2] - ref_xy) ** 2)
        loss_ref_z = torch.sum((mean_trace[:, :, 2] - ref_z) ** 2)
        loss_terminal = torch.sum((mean_trace[:, -1, :3] - ref[0, :3]) ** 2)

        return (
            loss
            + self.cfg.get('w_ref_xy', 0.0) * loss_ref_xy
            + self.cfg.get('w_ref_z', 0.0) * loss_ref_z
            + self.cfg.get('w_terminal', 0.0) * loss_terminal
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
    dynamics = PositionDynamics(2, dt=DT, u_max=U_MAX, q_std=Q_STD)
    environment = build_environment_baseline()
    planner = PositionPlanner(dynamics, environment, T, config=dict(PLANNER_CONFIG))
    return planner, dynamics, environment


def initial_belief_baseline(fan: int) -> tuple[torch.Tensor, torch.Tensor]:
    sigma0 = SIGMA0_PER_FAN[fan]
    mean = torch.tensor(START_XY, dtype=torch.float32)
    covariance = torch.eye(2, dtype=torch.float32) * sigma0
    return mean, covariance


def initial_controls_baseline() -> torch.Tensor:
    positions = np.array(nominal_baseline_waypoints(T + 1))[:, :2]
    return torch.tensor(np.diff(positions, axis=0) / DT, dtype=torch.float32)


def nominal_figure8_waypoints(n_points: int = FIG8_T + 1) -> list[tuple[float, float, float]]:
    """Return the smooth 3D Gerono figure-eight reference path."""
    if n_points < 2:
        raise ValueError('n_points must be at least 2')
    s = np.linspace(0.0, 1.0, n_points)
    q = 35 * s**4 - 84 * s**5 + 70 * s**6 - 20 * s**7
    theta = np.pi + 2 * np.pi * q
    x = FIG8_CENTER_X + FIG8_HALF_WIDTH * np.sin(2 * theta)
    y = FIG8_CENTER_Y + np.cos(theta)
    z = FIG8_Z_BASE + FIG8_Z_AMPLITUDE * np.sin(2 * np.pi * q) ** 2
    return [(float(xi), float(yi), float(zi)) for xi, yi, zi in zip(x, y, z)]


def build_environment_figure8() -> PositionEnvironment:
    """Build the pdSTL Environment for the figure8 mission.

    The specification enforces workspace, obstacle avoidance, and the
    midpoint altitude band. Reference tracking closes the loop.
    """
    env = PositionEnvironment()
    env.set_bounds(x_range=FIG8_X_BOUNDS, y_range=FIG8_Y_BOUNDS, z_range=FIG8_Z_BOUNDS)
    for obs in FIG8_OBSTACLES:
        env.add_obstacle(x_range=list(obs['x']), y_range=list(obs['y']), z_range=list(obs['z']))
    env.add_time_windowed_bounds(
        x_range=FIG8_X_BOUNDS, y_range=FIG8_Y_BOUNDS, z_range=list(FIG8_MIDPOINT_Z),
        interval=[FIG8_MIDPOINT_T_START, FIG8_MIDPOINT_T_END], label='midpoint_altitude',
    )
    return env


def build_planner_figure8() -> tuple[PositionPlanner, PositionDynamics, PositionEnvironment]:
    """Build a (Planner, dynamics, environment) for the figure8 mission.

    Attach the analytical path as the figure-eight reference trajectory.
    """
    dynamics = PositionDynamics(3, dt=DT, u_max=U_MAX, q_std=Q_STD)
    env = build_environment_figure8()
    planner = PositionPlanner(dynamics, env, FIG8_T, config={
        **PLANNER_CONFIG,
        'w_ref_xy': FIG8_W_REF_XY, 'w_ref_z': FIG8_W_REF_Z, 'w_terminal': FIG8_W_TERMINAL,
    })
    planner.reference_trajectory = torch.tensor(
        nominal_figure8_waypoints(FIG8_T + 1), dtype=torch.float32,
    )
    return planner, dynamics, env


def x0_belief_figure8(fan_speed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Initial belief using the approved fan covariance from config.yml."""
    sigma0 = SIGMA0_PER_FAN[fan_speed]
    start = nominal_figure8_waypoints(n_points=2)[0]
    mean = torch.tensor(start, dtype=torch.float32)
    cov = torch.diag(torch.tensor([sigma0, sigma0, sigma0], dtype=torch.float32))
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

    def build_planner(self, fan): return build_planner_figure8()
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
        planner, _dynamics, environment = model.build_planner(fan)
        initial_mean, initial_covariance = model.initial_belief(fan)
        initial_controls = model.initial_controls()
        rho_before = _satisfaction(planner, initial_controls, initial_mean, initial_covariance)
        best_mean, _covariance, _controls, rho_after, _history = planner._optimize_window(
            initial_mean, initial_covariance, init_guess=initial_controls, verbose=True,
        )
        positions = best_mean.squeeze(0).cpu().numpy()
        if model.dimension == 2:
            waypoints = [(float(x), float(y), Z_HEIGHT) for x, y in positions]
        else:
            waypoints = [tuple(map(float, point)) for point in positions]
        model.validate(waypoints, label='Generated waypoint')
        path = self.plans.save(fan, scenario, waypoints, {
            'sigma0': SIGMA0_PER_FAN[fan],
            'q_std': Q_STD,
            'rho_before': round(rho_before, 4),
            'rho_after': round(float(rho_after), 4),
            'alpha': PLANNER_ALPHA,
            'T': planner.T,
            'dt': DT,
            'generated': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        })
        print(f'Wrote {len(waypoints)} waypoints to {path}')
        if plot:
            self._plot(model, environment, waypoints, fan)
        return path

    @staticmethod
    def _plot(model: Scenario, environment, waypoints, fan: int) -> None:
        import matplotlib.pyplot as plt

        from .utils import draw_environment_2d, draw_environment_3d

        nominal = np.array(model.nominal_waypoints(model.flight_points))
        planned = np.array(waypoints)
        if model.dimension == 2:
            figure, axis = plt.subplots(figsize=(7, 7))
            draw_environment_2d(axis, environment)
            axis.plot(nominal[:, 0], nominal[:, 1], '--', label='nominal')
            axis.plot(planned[:, 0], planned[:, 1], '.-', label='pdSTL')
        else:
            figure = plt.figure(figsize=(8, 8))
            axis = figure.add_subplot(projection='3d')
            axis.view_init(elev=22, azim=-50)
            draw_environment_3d(axis, environment, fit_points=np.vstack([nominal, planned]))
            axis.plot(*nominal.T, '--', label='nominal')
            axis.plot(*planned.T, '.-', label='pdSTL')
        axis.legend()
        output = EXPERIMENT_DIR / 'plots' / f'{model.name}_fan{fan}_comparison.png'
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150, bbox_inches='tight')
        plt.close(figure)
        print(f'Plot saved to {output}')
