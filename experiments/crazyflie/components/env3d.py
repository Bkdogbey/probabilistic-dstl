"""Crazyflie-only 3D extension of the shared pdSTL planning library.

Subclasses src/planning/{dynamics,environment,planner}.py rather than
modifying them: that shared library also backs ~6 other 2D scenario configs
under configs/scenarios/, so 3D support is kept isolated to this experiment.
Only the genuinely 2D-hardcoded pieces (process-noise covariance shape,
rectangular-predicate axis slicing, heuristic-loss centers) are overridden;
the STL operator chaining (Eventually/Or/And/Always) and the optimization
loop (Planner._optimize_window) are dimension-general already and reused
unchanged.
"""

from __future__ import annotations

import torch

from pdstl.operators import Always, And, Eventually, Or, STL_Formula
from planning.dynamics import SingleIntegrator
from planning.environment import (
    CircularObstaclePredicate,
    Environment,
    extract_trajectory_stats,
    normal_cdf,
)
from planning.planner import Planner


class SingleIntegrator3D(SingleIntegrator):
    """SingleIntegrator generalised to 3D position/velocity state.

    step()/forward() are inherited unchanged -- they're already shape-general
    tensor ops with no dimension-specific slicing. The only 2D-hardcoded piece
    was the process-noise covariance shape, overridden below.
    """

    def __init__(self, dt=0.2, u_max=1.0, q_std=0.05, device="cpu"):
        super().__init__(dt=dt, u_max=u_max, q_std=q_std, device=device)
        self.Q = torch.eye(3, device=self.device) * q_std**2
        assert self.Q.shape == (3, 3)


class RectangularGoalPredicate3D(STL_Formula):
    """3D rectangular goal containment via the product of per-axis probabilities.

    Extends RectangularGoalPredicate's p_x*p_y with a third p_z factor. Unlike
    the And operator's cross-predicate chaining (fixed to use the sound
    Frechet bound in pdstl.operators.And, since sub-formulas there can be
    correlated in ways the planner doesn't track), this per-axis product is
    exact -- not an approximation -- given this experiment's dynamics:
    SingleIntegrator3D's covariance is provably diagonal for all time
    (P_next = P + Q, both diagonal, no cross-axis coupling), so x, y, z are
    genuinely independent random variables under the belief model, and
    P(x∈X ∩ y∈Y ∩ z∈Z) = P(x∈X)·P(y∈Y)·P(z∈Z) exactly. This also matches the
    shared library's 2D RectangularGoalPredicate's own documented convention
    ("Combine using Product (Independence)... more accurate for a rectangular
    region than min()"), just extended to a third axis rather than a new
    assumption. With no z range, reduces to p_x * p_y (identical to the 2D
    predicate).
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
        p_x = normal_cdf(self.x_max, mu_x, var_x) - normal_cdf(self.x_min, mu_x, var_x)
        p_y = normal_cdf(self.y_max, mu_y, var_y) - normal_cdf(self.y_min, mu_y, var_y)

        if self.z_min is not None:
            mu_z, var_z = mu[..., 2], var[..., 2]
            p_z = normal_cdf(self.z_max, mu_z, var_z) - normal_cdf(self.z_min, mu_z, var_z)
            p_goal = torch.clamp(p_x * p_y * p_z, min=0.0, max=1.0)
        else:
            p_goal = torch.clamp(p_x * p_y, min=0.0, max=1.0)

        return torch.stack([p_goal, p_goal], dim=-1)


class RectangularObstaclePredicate3D(STL_Formula):
    """3D rectangular obstacle safety: safe if outside on ANY single axis (6-face max).

    The union bound P(union) >= max(p_i) is sound regardless of dependence --
    unlike the goal predicate's intersection case, no independence assumption
    is involved here (union always dominates any single member by
    monotonicity). Backward compatible: with no z range, behaves identically
    to the 4-face 2D RectangularObstaclePredicate. Flying above z_max (or
    below z_min) satisfies safety regardless of x/y -- this is what lets the
    planner route over a floor-mounted obstacle using its known height.
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


class Environment3D(Environment):
    """Environment generalised to 3D via optional z_range params.

    add_obstacle/set_goal/set_bounds accept an extra z_range (default None,
    matching the parent's 2D-only behaviour exactly when omitted).
    get_predicates()/get_specification() are overridden to build the 3D
    predicate classes above instead of the shared 2D ones; the STL chaining
    logic itself is copied near-verbatim from the parent since it isn't
    exposed as an independently overridable seam there.
    """

    def add_obstacle(self, x_range, y_range, z_range=None):
        obstacle = {"x": x_range, "y": y_range}
        if z_range is not None:
            obstacle["z"] = z_range
        self.obstacles.append(obstacle)

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
            "goal": None,
        }

        if self.goal:
            preds["goal"] = RectangularGoalPredicate3D(self.goal)

        for region in self.visit_regions:
            preds["visit"].append(RectangularGoalPredicate3D(region))

        for region in self.timed_visit_regions:
            preds["timed_visit"].append(
                {
                    "predicate": RectangularGoalPredicate3D(region),
                    "interval": region["interval"],
                    "label": region.get("label", None),
                }
            )

        for group in self.choice_region_groups:
            group_preds = [RectangularGoalPredicate3D(region) for region in group["regions"]]
            preds["choice_region_groups"].append(
                {
                    "predicates": group_preds,
                    "interval": group.get("interval", None),
                    "label": group.get("label", None),
                }
            )

        if self.obstacles or self.circle_obstacles or self.moving_obstacles:
            obs_preds = [RectangularObstaclePredicate3D(obs) for obs in self.obstacles]
            obs_preds.extend(
                CircularObstaclePredicate(obs, device=self.device)
                for obs in self.circle_obstacles
            )
            # Moving obstacles stay 2D-only (Crazyflie defines none; the shared
            # MovingRectangularObstaclePredicate isn't 3D-extended in this pass).
            preds["obstacles"] = obs_preds

        return preds

    def get_specification(self, T, t_goal_start=0, t_constraints_start=1):
        """Same Eventually/Or/And/Always structure as Environment.get_specification,
        with the 3D predicate classes substituted in (see class docstring)."""
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

        if preds["obstacles"]:
            obs_preds = preds["obstacles"]
            current_safe_formula = obs_preds[0]
            for i in range(1, len(obs_preds)):
                current_safe_formula = And(current_safe_formula, obs_preds[i])
            specs.append(Always(current_safe_formula, interval=[t_constraints_start, T]))

        if self.bounds is not None:
            specs.append(
                Always(RectangularGoalPredicate3D(self.bounds), interval=[t_constraints_start, T])
            )

        if not specs:
            raise ValueError("No constraints defined in environment.")

        combined_spec = specs[0]
        for i in range(1, len(specs)):
            combined_spec = And(combined_spec, specs[i])

        return combined_spec


class Planner3D(Planner):
    """Planner with 3D-aware heuristic guidance losses.

    _optimize_window itself is dimension-general (inherited, unchanged) --
    only the optional potential-field heuristic losses hardcode a 2D [:2]
    slice/center in the shared Planner. No override needed for:
      - _init_controls: its hardcoded-2 fallback only fires when init_guess
        is None; this experiment always passes an explicit init_guess.
      - _region_visit_loss/_visit_loss: this experiment's arena defines no
        visit regions.
      - _empty_u_trace: only used by the MPC loops, which this experiment's
        single-shot _optimize_window call never runs.
    """

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
