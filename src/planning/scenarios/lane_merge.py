"""Lane merge: pre-existing scenario, kept working and kept out of the reach-avoid path.

This module is self-contained on purpose. It owns the region kinds only it uses, its own
formula, and its own execution helpers, so nothing here constrains the small reach-avoid
foundation in `planning/environment.py`. It is carried over as-is and is not redesigned here.
"""

from dataclasses import dataclass, field, replace
from functools import reduce

import numpy as np
import torch

from pdstl.operators import Always, And, Eventually
from pdstl.predicates import InsideRectangle
from planning.environment import Environment, RectangleRegion
from planning.scenarios.moment_predicates import MovingRectangularObstaclePredicate


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

    Retained for the `isinstance` branches in the planner, the deterministic baseline and the
    visualization layer. Nothing builds one at present -- see the note in the refactor report.
    """

    name: str
    role: str
    center: tuple[float, float]
    radius: float
    style: dict = field(default_factory=dict)


class LaneMergeEnvironment(Environment):
    """An Environment that also carries the road description this scenario needs."""

    def __init__(self, specification_builder, metadata=None):
        super().__init__(specification_builder)
        self.metadata = dict(metadata or {})


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


def _rectangle(name, role, block):
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
        specification_builder=lane_merge_specification,
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
    environment.add_region(_rectangle("goal", "goal", config["goal"]))
    environment.add_region(MovingRectangleRegion(
        name="vehicle", role="obstacle",
        centers=torch.as_tensor(centers[: total_steps + 1], dtype=torch.float32, device=device),
        width=float(obstacle["width"]), height=float(obstacle["height"]),
    ))
    return environment


def local_window(environment, step, current_mean, config):
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
        specification_builder=lane_merge_specification, metadata=metadata
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


def clip_moving_obstacles(environment, num_points):
    """Trim every moving region to `num_points` centres, for plotting an executed run."""
    for name, region in list(environment.regions.items()):
        if isinstance(region, MovingRectangleRegion):
            environment.regions[name] = replace(region, centers=region.centers[:num_points])


def success_reached(environment, mean, counter):
    """(updated counter, done): the ego must hold the target band for consecutive_steps."""
    success = environment.metadata.get("success")
    if success is None:
        return counter, False
    inside = success["y_min"] <= float(mean[1]) <= success["y_max"]
    counter = counter + 1 if inside else 0
    return counter, counter >= success["consecutive_steps"]
