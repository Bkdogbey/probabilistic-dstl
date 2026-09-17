"""The rectangular reach-avoid example: its geometry, and the formula that geometry implies."""

from functools import reduce

from pdstl.operators import Always, And, Eventually
from pdstl.predicates import InsideRectangle, OutsideRectangle
from planning.environment import Environment, RectangleRegion


def reach_avoid_specification(environment, horizon):
    """G[1,H](inside workspace and outside every obstacle) and F[0,H](inside goal).

    Stay inside the workspace and clear of every obstacle from step 1 to the horizon, and
    reach the goal at least once between step 0 and the horizon.
    """
    workspace = environment.region("workspace")
    goal = environment.region("goal")

    safe = [InsideRectangle(workspace.x, workspace.y, name=workspace.name)]
    safe += [
        OutsideRectangle(obstacle.x, obstacle.y, name=obstacle.name)
        for obstacle in environment.by_role("obstacle")
    ]

    return Always(reduce(And, safe), interval=[1, horizon]) & Eventually(
        InsideRectangle(goal.x, goal.y, name=goal.name), interval=[0, horizon]
    )


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

    environment = Environment(specification_builder=reach_avoid_specification)
    environment.add_region(_rectangle(config["workspace"], "workspace", "workspace"))
    environment.add_region(_rectangle(config["goal"], "goal", "goal"))

    obstacles = config.get("obstacles") or []
    if not isinstance(obstacles, (list, tuple)):
        raise ValueError(f"'obstacles' must be a list, got {obstacles!r}")
    for index, entry in enumerate(obstacles):
        environment.add_region(_rectangle(entry, "obstacle", f"obstacle_{index}"))
    return environment
