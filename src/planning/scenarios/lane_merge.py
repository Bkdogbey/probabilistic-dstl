"""Lane-merge scenario: road geometry, a moving obstacle, and its success condition.

Everything lane-specific lives here so the generic planner and the static reach-avoid
environment stay free of roads, vehicles and success counters. The mathematics is carried
over unchanged from `planning/environment.py` and `planning/planner.py` on RA_L-planning --
this module moves it, it does not redesign it.
"""

import math
from functools import reduce

import numpy as np
import torch

from pdstl.operators import Always, And, Eventually, STL_Formula
from planning.environment import Environment


# --- Gaussian-specific moving-obstacle predicate ------------------------------
# Reads mean and covariance directly instead of going through Belief.probability_bounds.
# Kept for lane merge only; the static reach-avoid path uses rectangle events.


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


# --- Environment --------------------------------------------------------------


class LaneMergeEnvironment(Environment):
    """Road, lane markings, a constant-speed moving obstacle and a lane-keeping goal."""

    def __init__(self, device="cpu"):
        super().__init__(device=device)
        self.moving_obstacles = []
        self.lane_markings = []
        self.road = None
        self.lane_change = None
        self.success = None
        self.label = ""
        self.plot_xlim = None
        self.robot_dims = None
        self.window_config = {}
        self._success_counter = 0

    # --- Construction ----------------------------------------------------------

    @classmethod
    def from_config(cls, config, device="cpu", window_config=None):
        env = cls(device=device)
        env.window_config = dict(window_config or {})
        env.configure(
            road=config["road"], obstacle=config["obstacle"], goal=config["goal"],
            success=config["success"], horizon=config["H"], total_steps=config["T_SIM"],
            dt=config["dt"], label=config.get("label", ""),
            plot_xlim=config.get("plot_xlim"), robot_dims=config.get("robot_dims"),
        )
        return env

    def add_moving_obstacle(self, x_traj, y_traj, width, height):
        """Rectangle whose center follows x_traj, y_traj over time."""
        self.moving_obstacles.append(
            {"x_traj": x_traj, "y_traj": y_traj, "width": width, "height": height}
        )

    def add_lane_marking(self, x_range, y_pos, style="dashed", color="white"):
        self.lane_markings.append({"x": x_range, "y": y_pos, "style": style, "color": color})

    def configure(
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
            "obstacle": dict(obstacle), "horizon": horizon, "total_steps": total_steps,
            "dt": dt, "obs_x_global": obs_x, "obs_y_global": obs_y,
        }
        self.add_moving_obstacle(
            obs_x[: total_steps + 1], obs_y[: total_steps + 1],
            width=obstacle["width"], height=obstacle["height"],
        )

    # --- Specification ---------------------------------------------------------

    def predicates(self, context=None):
        events = super().predicates(context)
        events["obstacles"] = [
            MovingRectangularObstaclePredicate(o, device=self.device)
            for o in self.moving_obstacles
        ]
        return events

    def specification(self, horizon, context=None, t_goal_start=0, t_constraints_start=1):
        """Preserved lane-merge form: safety, eventual goal and workspace as SEPARATE clauses.

        Deliberately not the static reach-avoid form, which folds the workspace into the
        safety Always. Frechet conjunction does not commute with the temporal min, so merging
        them would move lane-merge's numbers.
        """
        events = self.predicates(context)
        clauses = []
        if events["obstacles"]:
            clauses.append(
                Always(reduce(And, events["obstacles"]), interval=[t_constraints_start, horizon])
            )
        if events["goal"] is not None:
            clauses.append(Eventually(events["goal"], interval=[t_goal_start, horizon]))
        if events["workspace"] is not None:
            clauses.append(
                Always(events["workspace"], interval=[t_constraints_start, horizon])
            )
        if not clauses:
            raise ValueError("No constraints defined in environment.")
        return reduce(And, clauses)

    # --- Receding-horizon support ----------------------------------------------

    def window(self, step=0, belief=None):
        """Local one-window environment: goal slides ahead of the ego, obstacle slice advances."""
        if self.lane_change is None or belief is None:
            return self
        local = self.local_window(step, belief[0], self.window_config)
        local.window_config = self.window_config
        return local

    def local_window(self, step, current_mean, cfg):
        """Local Environment for one lane-merge replanning step."""
        if self.road is None or self.lane_change is None:
            raise ValueError("Lane-merge local windows require configure().")

        horizon = self.lane_change["horizon"]
        obstacle = self.lane_change["obstacle"]
        obs_x, obs_y = self.lane_change["obs_x_global"], self.lane_change["obs_y_global"]
        road = self.road
        curr_x = current_mean.detach().cpu().numpy()[0]
        lookahead, width = cfg["mpc_goal_lookahead"], cfg["mpc_goal_window_width"]
        lane_margin = cfg["lane_boundary_margin"]

        local = LaneMergeEnvironment(device=self.device)
        local.set_goal(
            x_range=[curr_x + lookahead, curr_x + lookahead + width],
            y_range=[self.goal["y"][0] + cfg["goal_y_inset"], self.goal["y"][1] - cfg["goal_y_inset"]],
        )

        y_min_bound = road["y_min"] + lane_margin
        if current_mean[1] > road["lane_divider"] - lane_margin:
            y_min_bound = road["lane_divider"]
        local.set_bounds(x_range=cfg["mpc_local_x_range"], y_range=[y_min_bound, road["y_max"]])

        idx_end = step + horizon + 1
        if idx_end <= len(obs_x):
            sl_x, sl_y = obs_x[step:idx_end], obs_y[step:idx_end]
        else:
            pad = idx_end - len(obs_x)
            sl_x = np.concatenate([obs_x[step:], np.full(pad, obs_x[-1])])
            sl_y = np.concatenate([obs_y[step:], np.full(pad, obs_y[-1])])
        local.add_moving_obstacle(sl_x, sl_y, width=obstacle["width"], height=obstacle["height"])
        return local

    def moving_obstacle_position(self, step):
        """Lane-merge obstacle center at a global step."""
        if self.lane_change is None:
            return None
        obs_x, obs_y = self.lane_change["obs_x_global"], self.lane_change["obs_y_global"]
        idx = min(step, len(obs_x) - 1)
        return np.array([obs_x[idx], obs_y[idx]])

    def clip_moving_obstacles(self, num_points):
        for obs in self.moving_obstacles:
            obs["x_traj"] = obs["x_traj"][:num_points]
            obs["y_traj"] = obs["y_traj"][:num_points]

    def reset_progress(self):
        self._success_counter = 0

    def is_complete(self, state, belief=None, context=None):
        """True once the ego has held the target lane for `consecutive_steps` in a row."""
        if self.success is None:
            return False
        y = float(state[1])
        inside = self.success["y_min"] <= y <= self.success["y_max"]
        self._success_counter = self._success_counter + 1 if inside else 0
        return self._success_counter >= self.success["consecutive_steps"]

    # --- Scenario-supplied shaping ---------------------------------------------

    def extra_loss(self, mean_trace, cfg):
        """Obstacle repulsion around the moving vehicle.

        Lane merge relies on this (`w_obs: 12.0` against `w_phi: 200.0`), so it is preserved --
        but it lives here, not in the generic objective, because it is the only user.
        """
        weight = cfg.get("w_obs", 0.0)
        if not weight or not self.moving_obstacles:
            return torch.zeros((), device=mean_trace.device)
        margin = cfg["obs_margin"]
        loss = torch.zeros((), device=mean_trace.device)
        for obs in self.moving_obstacles:
            ox = torch.as_tensor(obs["x_traj"], device=mean_trace.device)
            oy = torch.as_tensor(obs["y_traj"], device=mean_trace.device)
            centers = torch.stack([ox, oy], dim=1).unsqueeze(0)  # [1, T+1, 2]
            radius = max(obs["width"], obs["height"]) / 2.0 + margin
            dists = torch.norm(mean_trace[:, :, :2] - centers, dim=2)
            loss = loss + torch.sum(torch.relu(radius - dists) ** 2)
        return weight * loss
