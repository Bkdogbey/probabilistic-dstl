"""Rectangular geometry, reach-avoid specifications, and lane construction.

The lane builder also creates moving geometry using NumPy and Torch. Probability
evaluation belongs to beliefs and pdSTL predicates, never this module.
"""

from dataclasses import dataclass, field
from functools import reduce
from math import ceil, floor

from pdstl.operators import Always, And, Eventually
from pdstl.predicates import (
    AxisInterval,
    HalfSpace,
    InsideRectangle,
    OutsideRectangle,
    RelativeAxisInterval,
)

ROLES = ("workspace", "goal", "obstacle")


# Rectangular geometry


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
        return [
            region for region in self.regions.values() if region.role == role
        ]

    def single_region(self, role):
        regions = self.by_role(role)
        if len(regions) != 1:
            raise ValueError(f"environment needs exactly one {role!r} region")
        return regions[0]

    def get_specification(self, horizon):
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or horizon < 1
        ):
            raise ValueError(
                f"horizon must be a positive integer, got {horizon!r}"
            )
        return reach_avoid_specification(self, horizon)


# Reach-avoid construction and specification


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
        "workspace": InsideRectangle(
            workspace.x, workspace.y, name=workspace.name
        ),
        "goal": InsideRectangle(goal.x, goal.y, name=goal.name),
        "obstacles": [
            OutsideRectangle(o.x, o.y, name=o.name)
            for o in environment.by_role("obstacle")
        ],
    }


def _rectangle(entry, role, fallback_name):
    """One `{name, x: [min, max], y: [min, max], style}` block into a region."""
    if not isinstance(entry, dict):
        raise ValueError(
            f"{role} must be a mapping with x and y, got {entry!r}"
        )
    name = entry.get("name", fallback_name)
    for axis in ("x", "y"):
        pair = entry.get(axis)
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise ValueError(
                f"region {name!r}: {axis} must be a [min, max] pair, got {pair!r}"
            )
    (xmin, xmax), (ymin, ymax) = entry["x"], entry["y"]
    return RectangleRegion(
        name=name,
        role=role,
        xmin=float(xmin),
        xmax=float(xmax),
        ymin=float(ymin),
        ymax=float(ymax),
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
    environment.add_region(
        _rectangle(config["workspace"], "workspace", "workspace")
    )
    environment.add_region(_rectangle(config["goal"], "goal", "goal"))

    obstacles = config.get("obstacles") or []
    if not isinstance(obstacles, (list, tuple)):
        raise ValueError(f"'obstacles' must be a list, got {obstacles!r}")
    for index, entry in enumerate(obstacles):
        environment.add_region(
            _rectangle(entry, "obstacle", f"obstacle_{index}")
        )
    return environment


# Lane-change construction


@dataclass
class MovingRectangleRegion:
    """A rectangle whose centre follows a trajectory, one centre per prediction step."""

    name: str
    role: str
    centers: object  # [T, 2] tensor or array of centres
    width: float
    height: float
    style: dict = field(default_factory=dict)


class LaneMergeEnvironment(Environment):
    """An Environment that also carries the road description this scenario needs."""

    def __init__(self, metadata=None):
        super().__init__()
        self.metadata = dict(metadata or {})

    def get_specification(self, horizon):
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or horizon < 1
        ):
            raise ValueError("horizon must be a positive integer")
        return lane_merge_specification(self, horizon)


def lane_merge_specification(environment, horizon):
    """Road and three relative collision checks, then timed target dwell."""
    return lane_subformulas(environment, horizon)["overall"]


def lane_subformulas(environment, horizon):
    metadata = environment.metadata
    collision = metadata["collision"]
    task = metadata["task"]
    step = metadata.get("step", 0)
    streak = metadata.get("streak", 0)
    names = [vehicle["name"] for vehicle in metadata["traffic"]]
    road_atom = lane_road_predicate(metadata)
    safety = {}
    for name in names:
        longitudinal = RelativeAxisInterval(
            name, -collision["longitudinal"], collision["longitudinal"], 0
        )
        lateral = RelativeAxisInterval(
            name, -collision["lateral"], collision["lateral"], 1
        )
        safety[name] = ~(longitudinal & lateral)
    all_safe = Always(reduce(And, [road_atom, *safety.values()]), [0, horizon])
    target = AxisInterval(
        task["target_center"] - task["target_tolerance"],
        task["target_center"] + task["target_tolerance"],
        dim=1,
        name="Target lane occupancy",
    )
    dwell = task["dwell_steps"]
    absolute_start, absolute_end = task["start_end_steps"]
    if step > absolute_end and streak > 0:
        witness = max(step - streak + 1, task["start_end_steps"][0])
        completion = Always(target, [0, max(0, witness + dwell - step)])
    else:
        start = max(0, absolute_start - step)
        end = max(0, absolute_end - step)
        completion = Eventually(Always(target, [0, dwell]), [start, end])
    if metadata.get("ramp") is not None:
        # By the ramp deadline the ego must be established in the main lane;
        # staying there prevents a valid short dwell followed by leaving it.
        completion = Always(target, [max(0, absolute_end - step), horizon])
    return {
        "road": road_atom,
        **{f"safety_{name}": predicate for name, predicate in safety.items()},
        "safe": all_safe,
        "complete": completion,
        "overall": all_safe & completion,
    }


def lane_road_predicate(metadata):
    """Use the outer road bounds; a taper adds its sloped lower edge."""
    road = metadata["road"]
    ramp = metadata.get("ramp")
    if ramp is None:
        return AxisInterval(
            road["y_min"], road["y_max"], dim=1, name="Road containment"
        )
    start, end = ramp["start_x"], ramp["end_x"]
    slope = (road["lane_divider"] - road["y_min"]) / (end - start)
    outer = AxisInterval(
        road["y_min"], road["y_max"], dim=1, name="Road outer bounds"
    )
    after_taper = AxisInterval(
        road["lane_divider"], road["y_max"], dim=1, name="Main lane"
    )
    above_taper = HalfSpace(
        (slope, -1.0, 0.0, 0.0),
        slope * start - road["y_min"],
        name="Above ramp edge",
    )
    return outer & (after_taper | above_taper)


def lane_contains_point(environment, x, y):
    """Physical road check used for the sampled executed trajectory."""
    road = environment.metadata["road"]
    if not road["y_min"] <= y <= road["y_max"]:
        return False
    ramp = environment.metadata.get("ramp")
    if ramp is None or y >= road["lane_divider"]:
        return True
    if x > ramp["end_x"]:
        return False
    slope = (road["lane_divider"] - road["y_min"]) / (
        ramp["end_x"] - ramp["start_x"]
    )
    lower = road["y_min"] + slope * (x - ramp["start_x"])
    return y >= max(road["y_min"], lower)


def _lane_rectangle(name, role, block):
    (xmin, xmax), (ymin, ymax) = block["x_range"], block["y_range"]
    return RectangleRegion(
        name=name,
        role=role,
        xmin=float(xmin),
        xmax=float(xmax),
        ymin=float(ymin),
        ymax=float(ymax),
    )


def build_lane_merge_environment(config, device="cpu"):
    """Road, target lane, and three constant-velocity traffic beliefs."""
    road = config["road"]
    ramp = config.get("ramp")
    if ramp is not None and not ramp["start_x"] < ramp["end_x"]:
        raise ValueError("ramp needs start_x < end_x")
    traffic = config["traffic"]
    if len(traffic) != 3 or len({car["name"] for car in traffic}) != 3:
        raise ValueError("lane traffic needs three uniquely named vehicles")
    task = dict(config["task"])
    dt = config["dt"]
    task["dwell_steps"] = ceil(task["dwell_seconds"] / dt)
    task["start_end_steps"] = (
        ceil(task["start_window_seconds"][0] / dt),
        floor(task["start_window_seconds"][1] / dt),
    )
    start, end = task["start_end_steps"]
    if (
        start < 0
        or end < start
        or task["dwell_steps"] < 1
        or config["H"] < end + task["dwell_steps"]
    ):
        raise ValueError(
            "lane horizon must cover the completion window and dwell time"
        )
    environment = LaneMergeEnvironment(
        metadata={
            "road": dict(road),
            "ramp": dict(ramp) if ramp is not None else None,
            "traffic": traffic,
            "ego_vehicle": dict(config["ego_vehicle"]),
            "collision": dict(config["collision"]),
            "task": task,
            "dt": dt,
            "device": device,
        },
    )
    environment.add_region(_lane_rectangle("goal", "goal", config["goal"]))
    return environment


def lane_local_window(environment, step, current_mean, config, streak=0):
    """Give a planning window its absolute time and observed dwell progress."""
    window = LaneMergeEnvironment(
        metadata={**environment.metadata, "step": step, "streak": streak}
    )
    window.regions = environment.regions
    return window
