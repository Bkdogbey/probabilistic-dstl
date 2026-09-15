"""Planning world: geometry and its pdSTL specification (no probability computed here)."""

import math
from functools import reduce

import numpy as np
import torch

from pdstl.operators import Always, And, Eventually, STL_Formula
from pdstl.predicates import InsideRectangle, OutsideRectangle


class Environment:
    """Workspace, goal, obstacles and visit regions."""

    def __init__(self, device="cpu"):
        self.obstacles = []
        self.circle_obstacles = []
        self.moving_obstacles = []
        self.visit_regions = []
        self.lane_markings = []
        self.goal = None
        self.bounds = None
        self.device = device
        self.road = None
        self.lane_change = None
        self.success = None
        self.label = ""
        self.plot_xlim = None
        self.robot_dims = None

    def add_obstacle(self, x_range, y_range):
        self.obstacles.append({"x": x_range, "y": y_range})

    def add_circle_obstacle(self, center, radius):
        self.circle_obstacles.append({"center": center, "radius": radius})

    def add_moving_obstacle(self, x_traj, y_traj, width, height):
        """Rectangle whose center follows x_traj, y_traj over time."""
        self.moving_obstacles.append(
            {"x_traj": x_traj, "y_traj": y_traj, "width": width, "height": height}
        )

    def add_lane_marking(self, x_range, y_pos, style="dashed", color="white"):
        self.lane_markings.append({"x": x_range, "y": y_pos, "style": style, "color": color})

    def add_visit_region(self, x_range, y_range):
        self.visit_regions.append({"x": x_range, "y": y_range})

    def set_goal(self, x_range, y_range):
        self.goal = {"x": x_range, "y": y_range}

    def set_bounds(self, x_range, y_range):
        """Workspace the trajectory must always stay inside."""
        self.bounds = {"x": x_range, "y": y_range}

    def draw_on_ax(self, ax, **kwargs):
        from visualization.planning import draw_env_on_ax  # keeps matplotlib out of import

        draw_env_on_ax(ax, self, **kwargs)

    # --- Specification ---------------------------------------------------------

    def get_predicates(self):
        """Rectangles become belief-evaluated events; circles/moving obstacles are legacy."""
        obstacles = [OutsideRectangle(o["x"], o["y"]) for o in self.obstacles]
        obstacles += [CircularObstaclePredicate(o, device=self.device) for o in self.circle_obstacles]
        obstacles += [
            MovingRectangularObstaclePredicate(o, device=self.device) for o in self.moving_obstacles
        ]
        return {
            "goal": InsideRectangle(self.goal["x"], self.goal["y"]) if self.goal else None,
            "visit": [InsideRectangle(r["x"], r["y"]) for r in self.visit_regions],
            "obstacles": obstacles,
        }

    def get_specification(self, T, t_goal_start=0, t_constraints_start=1):
        """Always(safe) ∧ Eventually(goal) ∧ Eventually(visits) ∧ Always(in bounds)."""
        preds = self.get_predicates()
        specs = []
        if preds["obstacles"]:
            specs.append(Always(reduce(And, preds["obstacles"]), interval=[t_constraints_start, T]))
        if preds["goal"] is not None:
            specs.append(Eventually(preds["goal"], interval=[t_goal_start, T]))
        specs += [Eventually(visit, interval=[0, T]) for visit in preds["visit"]]
        if self.bounds is not None:
            inside = InsideRectangle(self.bounds["x"], self.bounds["y"])
            specs.append(Always(inside, interval=[t_constraints_start, T]))
        if not specs:
            raise ValueError("No constraints defined in environment.")
        return reduce(And, specs)

    # --- Lane change -------------------------------------------------------------

    def configure_lane_change(
        self, *, road, obstacle, goal, success, horizon, total_steps, dt,
        label="", plot_xlim=None, robot_dims=None,
    ):
        """Road markings, goal lane and a constant-speed moving obstacle."""
        self.road = dict(road)
        self.success = dict(success)
        self.label = label
        self.plot_xlim = plot_xlim
        self.robot_dims = tuple(robot_dims) if robot_dims is not None else None

        marking_x = road["marking_x_range"]
        self.add_lane_marking(x_range=marking_x, y_pos=road["lane_divider"], style="dashed")
        self.add_lane_marking(x_range=marking_x, y_pos=road["y_min"], style="solid")
        self.add_lane_marking(x_range=marking_x, y_pos=road["y_max"], style="solid")
        self.set_goal(**goal)

        times = np.arange(total_steps + horizon + 10) * dt
        obs_x = obstacle["x0"] + obstacle["speed"] * times
        obs_y = np.ones_like(times) * obstacle["y"]
        self.lane_change = {
            "obstacle": dict(obstacle),
            "horizon": horizon,
            "total_steps": total_steps,
            "dt": dt,
            "obs_x_global": obs_x,
            "obs_y_global": obs_y,
        }
        self.add_moving_obstacle(
            obs_x[: total_steps + 1], obs_y[: total_steps + 1],
            width=obstacle["width"], height=obstacle["height"],
        )

    def make_local_lane_change_window(self, step, curr_mean, cfg):
        """Local Environment for one lane-change MPC step."""
        if self.road is None or self.lane_change is None:
            raise ValueError("Lane-change local windows require configure_lane_change().")

        horizon = self.lane_change["horizon"]
        obstacle = self.lane_change["obstacle"]
        obs_x, obs_y = self.lane_change["obs_x_global"], self.lane_change["obs_y_global"]
        road = self.road
        curr_x = curr_mean.detach().cpu().numpy()[0]
        lookahead, width = cfg["mpc_goal_lookahead"], cfg["mpc_goal_window_width"]
        lane_margin = cfg["lane_boundary_margin"]

        env_local = Environment(device=self.device)
        env_local.set_goal(
            x_range=[curr_x + lookahead, curr_x + lookahead + width],
            y_range=[self.goal["y"][0] + cfg["goal_y_inset"], self.goal["y"][1] - cfg["goal_y_inset"]],
        )

        y_min_bound = road["y_min"] + lane_margin
        if curr_mean[1] > road["lane_divider"] - lane_margin:
            y_min_bound = road["lane_divider"]
        env_local.set_bounds(x_range=cfg["mpc_local_x_range"], y_range=[y_min_bound, road["y_max"]])

        idx_end = step + horizon + 1
        if idx_end <= len(obs_x):
            sl_x, sl_y = obs_x[step:idx_end], obs_y[step:idx_end]
        else:
            pad = idx_end - len(obs_x)
            sl_x = np.concatenate([obs_x[step:], np.full(pad, obs_x[-1])])
            sl_y = np.concatenate([obs_y[step:], np.full(pad, obs_y[-1])])
        env_local.add_moving_obstacle(sl_x, sl_y, width=obstacle["width"], height=obstacle["height"])
        return env_local

    def moving_obstacle_position(self, step):
        """Lane-change obstacle center at a global step."""
        if self.lane_change is None:
            return None
        obs_x, obs_y = self.lane_change["obs_x_global"], self.lane_change["obs_y_global"]
        idx = min(step, len(obs_x) - 1)
        return np.array([obs_x[idx], obs_y[idx]])

    def clip_moving_obstacles(self, num_points):
        for obs in self.moving_obstacles:
            obs["x_traj"] = obs["x_traj"][:num_points]
            obs["y_traj"] = obs["y_traj"][:num_points]


# --- LEGACY: Gaussian-specific predicates (circle / moving obstacles) -------------
# They read mean and covariance directly instead of Belief.probability_bounds.


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

    def __init__(self, circle_def, device="cpu"):
        super().__init__()
        self.center = torch.tensor(circle_def["center"], device=device, dtype=torch.float32)
        self.radius = circle_def["radius"]

    def robustness_trace(self, belief_trajectory, **kwargs):
        mu, cov = extract_trajectory_stats(belief_trajectory, diagonal_only=False)
        diff = mu - self.center
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

    def __init__(self, obs_def, device="cpu"):
        super().__init__()
        self.x_traj = torch.as_tensor(obs_def["x_traj"], device=device, dtype=torch.float32)
        self.y_traj = torch.as_tensor(obs_def["y_traj"], device=device, dtype=torch.float32)
        self.width = obs_def["width"]
        self.height = obs_def["height"]

    def robustness_trace(self, belief_trajectory, **kwargs):
        mu, var = extract_trajectory_stats(belief_trajectory)
        mu_x, mu_y, var_x, var_y = mu[..., 0], mu[..., 1], var[..., 0], var[..., 1]
        half_w, half_h = self.width / 2.0, self.height / 2.0
        p_safe = torch.stack([
            normal_cdf(self.x_traj - half_w, mu_x, var_x),
            1.0 - normal_cdf(self.x_traj + half_w, mu_x, var_x),
            normal_cdf(self.y_traj - half_h, mu_y, var_y),
            1.0 - normal_cdf(self.y_traj + half_h, mu_y, var_y),
        ], dim=0).max(dim=0).values
        return torch.stack([p_safe, p_safe], dim=-1)
