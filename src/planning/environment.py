"""Rectangular geometry and the environment the planner asks for a specification.

This module knows about rectangles and names. It does not sample trajectories, estimate
probabilities, compute robustness, or draw anything, and it has no opinion on whether the
planner uses the exact or the smooth semantics.
"""

from dataclasses import dataclass, field
from functools import reduce

import numpy as np
import torch

from pdstl.operators import Always, And, Eventually
from pdstl.predicates import InsideRectangle, OutsideRectangle, MovingRectangularObstaclePredicate

ROLES = ("workspace", "goal", "obstacle")


@dataclass
class RectangleRegion:
    """An axis-aligned rectangle: a name for formulas, a role for selection, a style for plots.

    `style` is carried but never interpreted here; reading it is visualization's job.
    """

    name: str
    role: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    style: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(
                f"region {self.name!r}: role must be one of {list(ROLES)}, got {self.role!r}"
            )
        if not self.xmin < self.xmax:
            raise ValueError(
                f"region {self.name!r}: needs xmin < xmax, got [{self.xmin}, {self.xmax}]"
            )
        if not self.ymin < self.ymax:
            raise ValueError(
                f"region {self.name!r}: needs ymin < ymax, got [{self.ymin}, {self.ymax}]"
            )

    @property
    def x(self):
        """The (min, max) pair, for callers that want the bounds together."""
        return (self.xmin, self.xmax)

    @property
    def y(self):
        return (self.ymin, self.ymax)


class Environment:
    """Named geometry and its reach-avoid specification."""

    def __init__(self):
        self.regions = {}

    def add_region(self, region):
        if region.name in self.regions:
            raise ValueError(f"region {region.name!r} already exists")
        self.regions[region.name] = region
        return region

    def region(self, name):
        try:
            return self.regions[name]
        except KeyError:
            raise ValueError(
                f"no region named {name!r}; have {sorted(self.regions)}"
            ) from None

    def by_role(self, role):
        """Every region with this role, in the order they were added."""
        return [region for region in self.regions.values() if region.role == role]

    def single_region(self, role):
        regions = self.by_role(role)
        if len(regions) != 1:
            raise ValueError(f"environment needs exactly one {role!r} region")
        return regions[0]

    def get_specification(self, horizon):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
        return reach_avoid_specification(self, horizon)


def reach_avoid_specification(environment, horizon):
    """G[1,H](inside workspace and outside every obstacle) and F[0,H](inside goal).

    Stay inside the workspace and clear of every obstacle from step 1 to the horizon, and
    reach the goal at least once between step 0 and the horizon.
    """
    events = reach_avoid_events(environment)
    safe = [events["workspace"], *events["obstacles"]]
    return Always(reduce(And, safe), interval=[1, horizon]) & Eventually(
        events["goal"], interval=[0, horizon]
    )


def reach_avoid_events(environment):
    """The environment's named atoms, shared by specification and diagnostic consumers."""
    workspace = environment.single_region("workspace")
    goal = environment.single_region("goal")
    return {
        "workspace": InsideRectangle(workspace.x, workspace.y, name=workspace.name),
        "goal": InsideRectangle(goal.x, goal.y, name=goal.name),
        "obstacles": [
            OutsideRectangle(o.x, o.y, name=o.name) for o in environment.by_role("obstacle")
        ],
    }


def _rectangle(entry, role, fallback_name):
    """One `{name, x: [min, max], y: [min, max], style}` block into a region."""
    if not isinstance(entry, dict):
        raise ValueError(f"{role} must be a mapping with x and y, got {entry!r}")
    name = entry.get("name", fallback_name)
    for axis in ("x", "y"):
        pair = entry.get(axis)
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise ValueError(f"region {name!r}: {axis} must be a [min, max] pair, got {pair!r}")
    (xmin, xmax), (ymin, ymax) = entry["x"], entry["y"]
    return RectangleRegion(
        name=name, role=role,
        xmin=float(xmin), xmax=float(xmax), ymin=float(ymin), ymax=float(ymax),
        style=dict(entry.get("style") or {}),
    )


def build_reach_avoid_environment(config):
    """Build the environment from a scenario file's geometry.

    Expects a `workspace` block, a `goal` block, and zero or more `obstacles`. Names, bounds
    and styles all come from configuration, so the example is changed without editing code.
    """
    for required in ("workspace", "goal"):
        if config.get(required) is None:
            raise ValueError(f"reach-avoid config needs a {required!r} block")

    environment = Environment()
    environment.add_region(_rectangle(config["workspace"], "workspace", "workspace"))
    environment.add_region(_rectangle(config["goal"], "goal", "goal"))

    obstacles = config.get("obstacles") or []
    if not isinstance(obstacles, (list, tuple)):
        raise ValueError(f"'obstacles' must be a list, got {obstacles!r}")
    for index, entry in enumerate(obstacles):
        environment.add_region(_rectangle(entry, "obstacle", f"obstacle_{index}"))
    return environment


@dataclass
class MovingRectangleRegion:
    """A rectangle whose centre follows a trajectory, one centre per prediction step."""

    name: str
    role: str
    centers: object  # [T, 2] tensor or array of centres
    width: float
    height: float
    style: dict = field(default_factory=dict)


@dataclass
class CircleRegion:
    """A disc obstacle.

    Supported by the deterministic baseline and visualization consumers.
    """

    name: str
    role: str
    center: tuple[float, float]
    radius: float
    style: dict = field(default_factory=dict)


class LaneMergeEnvironment(Environment):
    """An Environment that also carries the road description this scenario needs."""

    def __init__(self, metadata=None):
        super().__init__()
        self.metadata = dict(metadata or {})

    def get_specification(self, horizon):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        return lane_merge_specification(self, horizon)


def lane_merge_specification(environment, horizon):
    """Safety, eventual goal and workspace as SEPARATE clauses.

    Deliberately not the reach-avoid shape, which folds the workspace into the safety Always.
    Frechet conjunction does not commute with the temporal min, so merging them would move
    lane merge's numbers.
    """
    clauses = []
    obstacles = environment.by_role("obstacle")
    if obstacles:
        clauses.append(Always(
            reduce(And, [MovingRectangularObstaclePredicate(o) for o in obstacles]),
            interval=[1, horizon],
        ))
    goal = environment.region("goal")
    clauses.append(
        Eventually(InsideRectangle(goal.x, goal.y, name=goal.name), interval=[0, horizon])
    )
    workspace = environment.regions.get("workspace")
    if workspace is not None:
        clauses.append(
            Always(InsideRectangle(workspace.x, workspace.y, name=workspace.name),
                   interval=[1, horizon])
        )
    if not clauses:
        raise ValueError("lane merge needs at least a goal")
    return reduce(And, clauses)


def _lane_rectangle(name, role, block):
    (xmin, xmax), (ymin, ymax) = block["x_range"], block["y_range"]
    return RectangleRegion(
        name=name, role=role,
        xmin=float(xmin), xmax=float(xmax), ymin=float(ymin), ymax=float(ymax),
    )


def build_lane_merge_environment(config, device="cpu"):
    """Road markings, goal lane and a constant-speed moving vehicle, from the scenario file."""
    road, obstacle = config["road"], config["obstacle"]
    horizon, total_steps, dt = config["H"], config["T_SIM"], config["dt"]

    times = np.arange(total_steps + horizon + 10) * dt
    centers = np.stack(
        [obstacle["x0"] + obstacle["speed"] * times, np.full_like(times, obstacle["y"])], axis=1
    )
    marking_x = road["marking_x_range"]

    environment = LaneMergeEnvironment(
        metadata={
            "road": dict(road),
            "success": dict(config["success"]),
            "label": config.get("label", ""),
            "plot_xlim": config.get("plot_xlim"),
            "robot_dims": tuple(config["robot_dims"]) if config.get("robot_dims") else None,
            "horizon": horizon,
            "obstacle": dict(obstacle),
            "centers": centers,
            "device": device,
            "lane_markings": [
                {"x": marking_x, "y": road["lane_divider"], "style": "dashed", "color": "white"},
                {"x": marking_x, "y": road["y_min"], "style": "solid", "color": "white"},
                {"x": marking_x, "y": road["y_max"], "style": "solid", "color": "white"},
            ],
        },
    )
    environment.add_region(_lane_rectangle("goal", "goal", config["goal"]))
    environment.add_region(MovingRectangleRegion(
        name="vehicle", role="obstacle",
        centers=torch.as_tensor(centers[: total_steps + 1], dtype=torch.float32, device=device),
        width=float(obstacle["width"]), height=float(obstacle["height"]),
    ))
    return environment


def lane_local_window(environment, step, current_mean, config):
    """The one-window environment for a single replanning step.

    The goal slides ahead of the ego, the workspace floor lifts once the ego has committed to
    the lane change, and the vehicle trajectory is sliced to this window.
    """
    metadata = environment.metadata
    road, obstacle = metadata["road"], metadata["obstacle"]
    horizon, centers = metadata["horizon"], metadata["centers"]
    goal = environment.region("goal")

    ego_x = current_mean.detach().cpu().numpy()[0]
    lookahead, width = config["mpc_goal_lookahead"], config["mpc_goal_window_width"]
    margin, inset = config["lane_boundary_margin"], config["goal_y_inset"]

    window = LaneMergeEnvironment(
        metadata=metadata
    )
    window.add_region(RectangleRegion(
        name="goal", role="goal",
        xmin=ego_x + lookahead, xmax=ego_x + lookahead + width,
        ymin=goal.ymin + inset, ymax=goal.ymax - inset,
    ))

    floor = road["y_min"] + margin
    if current_mean[1] > road["lane_divider"] - margin:
        floor = road["lane_divider"]
    local_x = config["mpc_local_x_range"]
    window.add_region(RectangleRegion(
        name="workspace", role="workspace",
        xmin=float(local_x[0]), xmax=float(local_x[1]),
        ymin=float(floor), ymax=float(road["y_max"]),
    ))

    end = step + horizon + 1
    sliced = centers[step:end]
    if end > len(centers):  # hold the last centre past the end of the trajectory
        sliced = np.concatenate([sliced, np.repeat(centers[-1:], end - len(centers), axis=0)])
    window.add_region(MovingRectangleRegion(
        name="vehicle", role="obstacle",
        centers=torch.as_tensor(sliced, dtype=torch.float32, device=metadata["device"]),
        width=float(obstacle["width"]), height=float(obstacle["height"]),
    ))
    return window


def obstacle_position(environment, step):
    """The vehicle's centre at a global execution step."""
    centers = environment.metadata["centers"]
    return np.asarray(centers[min(step, len(centers) - 1)])
