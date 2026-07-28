"""3D planning primitives, plus the 'gate' and 'figure8' scenarios.

The 2D baseline mission (components/planning_2d.py) never imports this file
-- it's planned with the plain 2D Environment/Planner/SingleIntegrator from
src/planning/. This module exists because 3D scenarios need a real third (z)
dimension.

Section 1 subclasses src/planning/{dynamics,environment,planner}.py rather
than modifying it, since that shared library also backs other 2D scenario
configs under configs/scenarios/. Only the genuinely 2D-hardcoded pieces
(process-noise covariance shape, rectangular-predicate axis slicing,
heuristic-loss centers) are overridden; STL operator chaining and the
optimization loop (Planner._optimize_window) are dimension-general already.
Section 1 is scenario-general -- SingleIntegrator3D/Environment3D/Planner3D
are reused unchanged by both scenarios below.

Section 2 (build_environment_3d/build_planner_3d/nominal_gate_waypoints/
x0_belief_3d/_nominal_init_guess_3d) is the gate scenario's construction
logic. The gate geometry *values* it builds from (GATE_*, POST_GATE_Z*)
live in config.yml.

Section 3 (build_environment_figure8/build_planner_figure8/
nominal_figure8_waypoints/x0_belief_figure8/_nominal_init_guess_figure8) is
the figure8 scenario's construction logic, mirroring Section 2's structure.
Figure8's pdSTL spec uses no visit regions -- only Always(workspace+altitude
envelope, avoid obstacles) and one tightened Always(mid-mission altitude
band). Without a visit region pulling the trajectory around the loop, the
spec alone would admit a trivial near-start solution, so Planner3D also
gains an optional reference-tracking objective (see _compute_loss below),
used only when a planner has a `reference_trajectory` attribute set --
gate's planner never sets it, so gate's optimization is unaffected.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# Imported first: components.config does the one sys.path.insert(src/) every
# planning.*/pdstl.* import in this file relies on -- importing it before
# them guarantees that regardless of which module gets imported first.
from components.config import (
    DT,
    END_XY,
    END_Z,
    FIG8_CENTER_X,
    FIG8_CENTER_Y,
    FIG8_HALF_WIDTH,
    FIG8_MIDPOINT_T_END,
    FIG8_MIDPOINT_T_START,
    FIG8_MIDPOINT_Z,
    FIG8_OBSTACLES,
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
    FLIGHT_Z_BOUNDS,
    GATE_CENTER_XY,
    GATE_T,
    GATE_T_DESCEND_START,
    GATE_T_END,
    GATE_T_START,
    GATE_X,
    GATE_Y,
    GATE_Z,
    GATE_Z_CENTER,
    GOAL,
    OBSTACLES,
    PLANNER_CONFIG,
    POST_GATE_Z,
    POST_GATE_Z_BAND,
    Q_STD,
    SAFE_PATH_VIA_POINTS,
    SIGMA0_PER_FAN,
    START_XY,
    START_Z,
    U_MAX,
    Z_HEIGHT,
    _pchip_eval,
    _pchip_slopes,
)
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

    Extends RectangularGoalPredicate's p_x*p_y with a third p_z factor. This
    per-axis product is exact -- not an approximation -- given this
    experiment's dynamics: SingleIntegrator3D's covariance is provably
    diagonal for all time (P_next = P + Q, both diagonal, no cross-axis
    coupling), so x, y, z are genuinely independent under the belief model,
    and P(x∈X ∩ y∈Y ∩ z∈Z) = P(x∈X)·P(y∈Y)·P(z∈Z) exactly. With no z range,
    reduces to p_x * p_y (identical to the 2D predicate).
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

    The union bound P(union) >= max(p_i) is sound regardless of dependence
    (union always dominates any single member by monotonicity), unlike the
    goal predicate's intersection case. This single predicate backs both
    floor-mounted and floating boxes uniformly -- the z-extent is just
    (z_min, z_max), whether it sits on the floor or hangs in the air. Flying
    above z_max (or below z_min) satisfies safety regardless of x/y, which is
    what lets the planner route over a floor-mounted obstacle (or under a
    floating one). The z_min is None branch remains only as a defensive
    fallback (Crazyflie obstacles always carry an explicit z after config.py
    canonicalisation); with no z range it reduces to the 4-face 2D predicate.
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
        """A rectangular region the trajectory must stay inside for an entire
        sub-interval (Always), unlike visit_regions/timed_visit_regions which
        only require being inside at some point (Eventually). Reuses the same
        containment predicate as goal/visit regions -- only the temporal
        operator wrapping it in get_specification() differs.
        """
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

        for region in self.time_windowed_bounds:
            preds["time_windowed_bounds"].append(
                {
                    "predicate": RectangularGoalPredicate3D(region),
                    "interval": region["interval"],
                    "label": region.get("label", None),
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
    slice/center in the shared Planner, overridden below. Not overridden:
    _init_controls (its 2D fallback only fires with no init_guess, and this
    experiment always passes one), _region_visit_loss/_visit_loss (no visit
    regions in this arena), _empty_u_trace (MPC-only; unused by this
    experiment's single-shot _optimize_window call).
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

    def _compute_loss(self, mean_trace, u_seq, p_all, loss_fn):
        """Adds an optional reference-tracking + terminal-return objective on
        top of the base Planner._compute_loss, active only when this planner
        has a `reference_trajectory` ([T+1, 3] tensor) attribute set.

        Used by the figure8 scenario, which defines no pdSTL visit region to
        pull the trajectory around the loop -- without this, obstacle
        avoidance plus the terminal-return term alone would admit a trivial
        near-start solution. gate never sets `reference_trajectory`, so
        `getattr` returns None and this method is byte-identical to the
        parent's _compute_loss for gate.
        """
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


# ═════════════════════════════════════════════════════════════════════════════
# 2. Gate scenario construction (3D-only). Geometry values (GATE_*,
#    POST_GATE_Z*) are imported from components/config.py -- see config.yml's
#    'gate:' section for the measurements and rationale behind each one.
# ═════════════════════════════════════════════════════════════════════════════
def nominal_gate_waypoints(n_points: int = GATE_T) -> list[tuple[float, float, float]]:
    """Deterministic climb-to-gate / pass-through / descend-and-avoid / land path.

    x(y) and z(y) are two independent PCHIP splines over the shared y-parameter
    (y increases monotonically along the whole route: START_XY -> gate -> via
    points -> END_XY). This is a warm start for the optimizer and (for the
    deterministic condition) a flyable path in its own right -- the z-profile
    nodes are shaped only to force a climb before the gate and a descent after
    it, not chosen to be obstacle-clear on their own.
    """
    xy_nodes = [START_XY, GATE_CENTER_XY, *SAFE_PATH_VIA_POINTS, END_XY]
    x_nodes = np.array([p[0] for p in xy_nodes], dtype=float)
    y_nodes = np.array([p[1] for p in xy_nodes], dtype=float)
    x_slopes = _pchip_slopes(y_nodes, x_nodes)

    # The z-profile must already be under POST_GATE_Z_BAND's ceiling by the y
    # the Always(altitude band) window starts (GATE_T_DESCEND_START, in planning-
    # horizon steps, independent of n_points) -- otherwise the warm start itself
    # violates that hard constraint right at the boundary, which (combined with
    # the other AND'd terms via the STL And operator's Frechet lower bound) can
    # crash the whole specification's P(sat) to 0 and stall the optimizer.
    y_descend = START_XY[1] + (END_XY[1] - START_XY[1]) * GATE_T_DESCEND_START / (GATE_T - 1)
    z_at_descend = 0.4  # comfortably under POST_GATE_Z_BAND[1]=0.5, on the way to POST_GATE_Z
    z_y_nodes = np.array([START_XY[1], -1.5, GATE_CENTER_XY[1], y_descend, -0.58, END_XY[1]])
    z_nodes = np.array([Z_HEIGHT, 0.75, GATE_Z_CENTER, z_at_descend, POST_GATE_Z, END_Z])
    z_slopes = _pchip_slopes(z_y_nodes, z_nodes)

    y_pos = np.linspace(START_XY[1], END_XY[1], n_points)
    x_pos = _pchip_eval(y_pos, y_nodes, x_nodes, x_slopes)
    z_pos = _pchip_eval(y_pos, z_y_nodes, z_nodes, z_slopes)
    return [(float(x), float(y), float(z)) for x, y, z in zip(x_pos, y_pos, z_pos)]


def build_environment_3d() -> Environment3D:
    """Build the pdSTL Environment for the 3D gate mission.

    Extends the same bounds/obstacles/goal as the 2D baseline. Each obstacle
    carries an explicit z-extent (obs['z'], canonicalised by config.py from
    either a floor-mounted 'height' shorthand or an explicit [z_min, z_max]);
    a floor-mounted box lets the optimizer clear it by flying over its top.
    Adds the timed gate visit region and a hard post-gate altitude ceiling
    (Always, not just a soft preference) -- without that ceiling, the
    obstacle-avoidance term alone is already satisfiable by flying above every
    obstacle's top for the whole mission, so nothing else would force a real
    post-gate descent.
    """
    env = Environment3D()
    env.set_bounds(x_range=FLIGHT_X_BOUNDS, y_range=FLIGHT_Y_BOUNDS, z_range=FLIGHT_Z_BOUNDS)
    for obs in OBSTACLES:
        env.add_obstacle(x_range=list(obs['x']), y_range=list(obs['y']), z_range=list(obs['z']))
    env.set_goal(x_range=list(GOAL['x']), y_range=list(GOAL['y']), z_range=list(GOAL['z']))
    env.add_timed_visit_region(
        x_range=list(GATE_X), y_range=list(GATE_Y),
        interval=[GATE_T_START, GATE_T_END], z_range=list(GATE_Z), label='gate',
    )
    env.add_time_windowed_bounds(
        x_range=FLIGHT_X_BOUNDS, y_range=FLIGHT_Y_BOUNDS, z_range=list(POST_GATE_Z_BAND),
        interval=[GATE_T_DESCEND_START, GATE_T], label='post_gate_low',
    )
    return env


def build_planner_3d() -> tuple[Planner3D, SingleIntegrator3D, Environment3D]:
    """Build a (Planner, dynamics, environment) for the 3D gate mission.

    Uses GATE_T (16) as the planning horizon instead of the 2D baseline's T
    (10) -- climb / gate / descend-and-avoid needs more steps than the
    baseline's single-phase mission.
    """
    dynamics = SingleIntegrator3D(dt=DT, u_max=U_MAX, q_std=Q_STD)
    env = build_environment_3d()
    planner = Planner3D(dynamics, env, GATE_T, config=dict(PLANNER_CONFIG))
    return planner, dynamics, env


def x0_belief_3d(fan_speed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Initial (mean, cov) belief for the gate mission at the given fan level.

    The paper's empirically characterized variance for this fan setting is
    placed on all three axes for this 3D extension.
    """
    sigma0 = SIGMA0_PER_FAN[fan_speed]
    mean = torch.tensor([*START_XY, START_Z], dtype=torch.float32)
    cov = torch.diag(torch.tensor(
        [sigma0, sigma0, sigma0], dtype=torch.float32,
    ))
    assert mean.shape == (3,), f'x0 mean must be 3D, got {tuple(mean.shape)}'
    assert cov.shape == (3, 3), f'x0 cov must be 3x3, got {tuple(cov.shape)}'
    return mean, cov


def _nominal_init_guess_3d() -> torch.Tensor:
    """Gate warm start: velocities from nominal_gate_waypoints (real z-velocity, not zeroed).

    Unlike the 2D baseline, the gate nominal path already climbs/descends, so
    its z-velocity is meaningful and kept as-is.
    """
    wps = np.array(nominal_gate_waypoints())
    vels_xyz = np.diff(wps, axis=0) / DT
    vels_xyz = np.vstack([vels_xyz, [[0.0, 0.0, 0.0]]])
    init_u = torch.tensor(vels_xyz, dtype=torch.float32)
    assert init_u.shape[-1] == 3, f'init_u must be [T, 3], got {tuple(init_u.shape)}'
    return init_u


# ═════════════════════════════════════════════════════════════════════════════
# 3. Figure8 scenario construction (3D-only). Geometry values (FIG8_*) are
#    imported from components/config.py -- see config.yml's 'figure8:'
#    section for the measurements and rationale behind each one.
# ═════════════════════════════════════════════════════════════════════════════
def nominal_figure8_waypoints(n_points: int = FIG8_T + 1) -> list[tuple[float, float, float]]:
    """Analytical Gerono figure-eight: x,y trace one closed loop via a 7th-order
    (quintic-blend) time law q(s), giving zero velocity/acceleration/jerk at
    s=0 and s=1. z is an independent, decoupled altitude profile (not tied to
    y like the gate's spline) with two symmetric humps and zero dz/ds at
    s=0, 0.5, 1 -- this is the deterministic path AND the pdSTL optimizer's
    warm start / soft reference (see Planner3D._compute_loss).
    """
    if n_points < 2:
        raise ValueError('n_points must be at least 2')
    s = np.linspace(0.0, 1.0, n_points)
    q = 35 * s**4 - 84 * s**5 + 70 * s**6 - 20 * s**7
    theta = np.pi + 2 * np.pi * q
    x = FIG8_CENTER_X + FIG8_HALF_WIDTH * np.sin(2 * theta)
    y = FIG8_CENTER_Y + np.cos(theta)
    z = FIG8_Z_BASE + FIG8_Z_AMPLITUDE * np.sin(2 * np.pi * q) ** 2
    return [(float(xi), float(yi), float(zi)) for xi, yi, zi in zip(x, y, z)]


def build_environment_figure8() -> Environment3D:
    """Build the pdSTL Environment for the figure8 mission.

    Deliberately no visit regions and no goal: the spec is Always(inside the
    figure8 workspace -- whose z bound [0.20, 0.70] already encodes the
    'airborne altitude' requirement -- AND avoid all 5 figure8-local
    obstacles), plus one tightened Always(altitude band) around the temporal
    midpoint via add_time_windowed_bounds (the same mechanism gate uses for
    its post-gate ceiling -- an Always band, not a spatial visit region).
    Returning to the start is handled entirely as a soft optimization
    objective (Planner3D._compute_loss), not part of this specification.
    """
    env = Environment3D()
    env.set_bounds(x_range=FIG8_X_BOUNDS, y_range=FIG8_Y_BOUNDS, z_range=FIG8_Z_BOUNDS)
    for obs in FIG8_OBSTACLES:
        env.add_obstacle(x_range=list(obs['x']), y_range=list(obs['y']), z_range=list(obs['z']))
    env.add_time_windowed_bounds(
        x_range=FIG8_X_BOUNDS, y_range=FIG8_Y_BOUNDS, z_range=list(FIG8_MIDPOINT_Z),
        interval=[FIG8_MIDPOINT_T_START, FIG8_MIDPOINT_T_END], label='midpoint_altitude',
    )
    return env


def build_planner_figure8() -> tuple[Planner3D, SingleIntegrator3D, Environment3D]:
    """Build a (Planner, dynamics, environment) for the figure8 mission.

    Uses FIG8_T (50) as the planning horizon. The planner's config merges in
    figure8's own reference-tracking weights (w_ref_xy/w_ref_z/w_terminal --
    kept out of the shared PLANNER_CONFIG so gate/baseline never see them),
    and the analytical curve is attached as `reference_trajectory` so
    Planner3D._compute_loss's reference/terminal terms activate.
    """
    dynamics = SingleIntegrator3D(dt=DT, u_max=U_MAX, q_std=Q_STD)
    env = build_environment_figure8()
    planner = Planner3D(dynamics, env, FIG8_T, config={
        **PLANNER_CONFIG,
        'w_ref_xy': FIG8_W_REF_XY, 'w_ref_z': FIG8_W_REF_Z, 'w_terminal': FIG8_W_TERMINAL,
    })
    planner.reference_trajectory = torch.tensor(
        nominal_figure8_waypoints(FIG8_T + 1), dtype=torch.float32,
    )
    return planner, dynamics, env


def x0_belief_figure8(fan_speed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Initial (mean, cov) belief for the figure8 mission at the given fan
    level. Mean is the analytical curve's own s=0 endpoint (figure8 has no
    separate measured START_XY -- the start IS the curve's output).

    Placeholder covariance model: SIGMA0_PER_FAN placed on all three axes,
    same as gate's x0_belief_3d. A later calibration phase will replace this
    with empirical fan-specific covariance traces Sigma_fan[t]; no fake
    loading from nonexistent calibration files is implemented here.
    """
    sigma0 = SIGMA0_PER_FAN[fan_speed]
    start = nominal_figure8_waypoints(n_points=2)[0]
    mean = torch.tensor(start, dtype=torch.float32)
    cov = torch.diag(torch.tensor([sigma0, sigma0, sigma0], dtype=torch.float32))
    assert mean.shape == (3,), f'x0 mean must be 3D, got {tuple(mean.shape)}'
    assert cov.shape == (3, 3), f'x0 cov must be 3x3, got {tuple(cov.shape)}'
    return mean, cov


def _nominal_init_guess_figure8() -> torch.Tensor:
    """Figure8 warm start: exactly FIG8_T controls diffed from FIG8_T+1
    nominal states, with NO zero-padding -- follows planning_2d.py's clean
    T+1-states -> T-controls pattern, not gate's _nominal_init_guess_3d
    (which builds only GATE_T waypoints, diffs to GATE_T-1 controls, then
    zero-pads one extra).
    """
    wps = np.array(nominal_figure8_waypoints(n_points=FIG8_T + 1))
    vels_xyz = np.diff(wps, axis=0) / DT
    init_u = torch.tensor(vels_xyz, dtype=torch.float32)
    assert init_u.shape == (FIG8_T, 3), f'init_u must be [{FIG8_T},3], got {tuple(init_u.shape)}'
    return init_u


def figure8_obstacle_clearance(curve_xyz: np.ndarray, obs: dict) -> float:
    """Min distance from any point on curve_xyz [N,3] to one figure8 obstacle
    box (0 = inside). Same dense point-to-AABB formula used for gate's
    plotting clearance annotations in waypoint_planning.py; kept local here
    (not imported from there) since planning_3d.py shouldn't depend on
    waypoint_planning.py.
    """
    def _axis_clearance(pos: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.maximum(np.maximum(lo - pos, 0.0), pos - hi)

    dx = _axis_clearance(curve_xyz[:, 0], *obs['x'])
    dy = _axis_clearance(curve_xyz[:, 1], *obs['y'])
    dz = _axis_clearance(curve_xyz[:, 2], *obs['z'])
    return float(np.min(np.sqrt(dx**2 + dy**2 + dz**2)))


def figure8_min_clearances(
    curve_xyz: np.ndarray, obstacles: list[dict] | None = None,
) -> dict[str, float]:
    """{obstacle name: min clearance} across all FIG8_OBSTACLES (or an
    override list) for a dense curve -- used by tests and by
    waypoint_planning.py's figure8 plan driver for a console sanity print.
    """
    obs_list = FIG8_OBSTACLES if obstacles is None else obstacles
    return {obs['name']: figure8_obstacle_clearance(curve_xyz, obs) for obs in obs_list}


def terminal_return_error(waypoints: list[tuple[float, float, float]]) -> float:
    """||last waypoint - first waypoint||_2 -- honest post-hoc report of how
    close the (optimized or nominal) trajectory returns to its own start.
    Not a test gate, not an optimizer hard constraint -- compare against
    FIG8_RETURN_TOLERANCE for reporting only.
    """
    return math.dist(waypoints[0], waypoints[-1])
