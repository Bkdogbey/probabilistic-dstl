"""Rectangular geometry, the reach-avoid task, and lane construction.

Probability evaluation belongs to beliefs and pdSTL predicates, never here.
"""

from dataclasses import dataclass, field
from functools import reduce
from math import ceil, floor

from pdstl.operators import Always, And, Eventually, Or
from pdstl.predicates import (
    AxisInterval,
    HalfSpace,
    InsideRectangle,
    LessThan,
    OutsideRectangle,
    RelativeAxisInterval,
)

# workspace: stay inside; obstacle: keep out; goal: reach; target: visit.
ROLES = ("workspace", "goal", "obstacle", "target")


# Rectangular geometry


@dataclass
class RectangleRegion:
    """An axis-aligned rectangle with a name and a role."""

    name: str
    role: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(
                f"region {self.name!r}: role must be one of {list(ROLES)}"
            )
        if not (self.xmin < self.xmax and self.ymin < self.ymax):
            raise ValueError(
                f"region {self.name!r}: needs min < max on both axes"
            )

    @property
    def x(self):
        return (self.xmin, self.xmax)

    @property
    def y(self):
        return (self.ymin, self.ymax)


class Environment:
    """Named regions and, for reach-avoid, its visit groups."""

    def __init__(self):
        self.regions = {}
        self.visits = []  # [(dwell, [target names])]

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


# Reach-avoid task


def inside(region):
    return InsideRectangle(region.x, region.y, name=region.name)


def outside(region):
    return OutsideRectangle(region.x, region.y, name=f"¬{region.name}")


def safety_event(environment):
    """Inside the workspace and outside every obstacle."""
    return reduce(
        And,
        [
            inside(environment.single_region("workspace")),
            *map(outside, environment.by_role("obstacle")),
        ],
    )


def reach_avoid_specification(environment, horizon):
    """Stay safe, reach any goal, and dwell in one target of each visit group.

    Args:
        environment: Environment with a workspace, goals and visit groups.
        horizon: Last prediction step H; step 0 is the known initial belief.

    Returns:
        G[1,H] safe & F[1,H] (any goal) & F[1,H-d] (any target held d steps).
    """
    goals = environment.by_role("goal")
    if not goals:
        raise ValueError("reach-avoid needs at least one goal")
    parts = [
        Always(safety_event(environment), [1, horizon]),
        Eventually(reduce(Or, map(inside, goals)), [1, horizon]),
    ]
    for dwell, names in environment.visits:
        if not 0 <= dwell < horizon:
            raise ValueError(f"dwell {dwell} must lie in [0, H)")
        stays = [
            Always(inside(environment.region(name)), [0, dwell])
            for name in names
        ]
        parts.append(Eventually(reduce(Or, stays), [1, horizon - dwell]))
    return reduce(And, parts)


def _rectangles(block, role):
    """`{name: {x: [min, max], y: [min, max]}}` into regions."""
    if not isinstance(block, dict):
        raise ValueError(f"{role} regions must be a name -> {{x, y}} mapping")
    return [
        RectangleRegion(name, role, *map(float, (*box["x"], *box["y"])))
        for name, box in block.items()
    ]


def build_reach_avoid_environment(config):
    """Build the reach-avoid environment from a scenario config.

    Args:
        config: Mapping with `workspace` ({x, y}), `goals` and optional
            `obstacles` (name -> {x, y}), and optional `visit` groups
            ({dwell, regions: name -> {x, y}}).

    Returns:
        The Environment.
    """
    if "workspace" not in config or not config.get("goals"):
        raise ValueError("reach-avoid config needs a workspace and goals")
    environment = Environment()
    blocks = [
        ({"workspace": config["workspace"]}, "workspace"),
        (config.get("obstacles") or {}, "obstacle"),
        (config["goals"], "goal"),
    ]
    for block, role in blocks:
        for region in _rectangles(block, role):
            environment.add_region(region)
    for group in config.get("visit") or []:
        targets = _rectangles(group["regions"], "target")
        for region in targets:
            environment.add_region(region)
        environment.visits.append(
            (int(group.get("dwell", 0)), [r.name for r in targets])
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
    task = metadata["task"]
    step = metadata.get("step", 0)
    streak = metadata.get("streak", 0)
    road_atom = lane_road_predicate(metadata)
    safety = {}
    for vehicle in metadata["traffic"]:
        name = vehicle["name"]
        longitudinal, lateral = lane_collision_extents(metadata, vehicle)
        longitudinal = RelativeAxisInterval(
            name, -longitudinal, longitudinal, 0
        )
        lateral = RelativeAxisInterval(name, -lateral, lateral, 1)
        safety[name] = ~(longitudinal & lateral)
    safe_atom = reduce(And, [road_atom, *safety.values()])
    all_safe = Always(safe_atom, [0, horizon])
    target = AxisInterval(
        task["target_center"] - task["target_tolerance"],
        task["target_center"] + task["target_tolerance"],
        dim=1,
        name="Target lane occupancy",
    )
    dwell = task["dwell_steps"]  # number of elapsed transitions
    absolute_start, absolute_end = task["start_end_steps"]
    credit = lane_dwell_credit(metadata, step, streak)

    def completion_from_start(start, duration):
        finish = start + duration
        formula = Always(target, [start, finish])
        ramp = metadata.get("ramp")
        if ramp is not None:
            front_limit = ramp["end_x"] - metadata["ego_vehicle"]["width"] / 2
            before_end = LessThan(
                front_limit, dim=0, name="Front before ramp end"
            )
            formula = formula & Always(before_end, [finish, finish])
        return formula, finish

    if credit:
        remaining = dwell - (credit - 1)
        candidates = [completion_from_start(0, remaining)]
    else:
        start = max(0, absolute_start - step)
        latest_start = absolute_end - dwell - step
        if latest_start < start:
            raise ValueError("lane task has no remaining feasible dwell start")
        candidates = [
            completion_from_start(candidate, dwell)
            for candidate in range(start, latest_start + 1)
        ]
    completion = reduce(Or, [formula for formula, _finish in candidates])
    overall = reduce(
        Or,
        [
            Always(safe_atom, [0, finish]) & formula
            for formula, finish in candidates
        ],
    )
    return {
        "road": road_atom,
        **{f"safety_{name}": predicate for name, predicate in safety.items()},
        "safe": all_safe,
        "complete": completion,
        "overall": overall,
    }


def lane_road_predicate(metadata):
    """Contain the complete axis-aligned ego footprint inside the road."""
    road = metadata["road"]
    ramp = metadata.get("ramp")
    ego = metadata["ego_vehicle"]
    half_width = ego["width"] / 2
    half_height = ego["height"] / 2
    if ramp is None:
        return AxisInterval(
            road["y_min"] + half_height,
            road["y_max"] - half_height,
            dim=1,
            name="Road containment",
        )
    start, end = ramp["start_x"], ramp["end_x"]
    slope = (road["lane_divider"] - road["y_min"]) / (end - start)
    outer = AxisInterval(
        road["y_min"] + half_height,
        road["y_max"] - half_height,
        dim=1,
        name="Road outer bounds",
    )
    after_taper = AxisInterval(
        road["lane_divider"] + half_height,
        road["y_max"] - half_height,
        dim=1,
        name="Main lane",
    )
    above_taper = HalfSpace(
        (slope, -1.0, 0.0, 0.0),
        slope * start - road["y_min"] - half_height - slope * half_width,
        name="Above ramp edge",
    )
    return outer & (after_taper | above_taper)


def lane_collision_extents(metadata, vehicle):
    """Combined footprint half-extents plus the configured safety margins."""
    ego = metadata["ego_vehicle"]
    margin = metadata["safety_margin"]
    return (
        (ego["width"] + vehicle["width"]) / 2 + margin["longitudinal"],
        (ego["height"] + vehicle["height"]) / 2 + margin["lateral"],
    )


def lane_has_collision(environment, ego_position, traffic_positions):
    """Return true when any traffic safety envelope overlaps the ego."""
    for vehicle, position in zip(
        environment.metadata["traffic"], traffic_positions
    ):
        longitudinal, lateral = lane_collision_extents(
            environment.metadata, vehicle
        )
        if (
            abs(float(position[0]) - float(ego_position[0])) <= longitudinal
            and abs(float(position[1]) - float(ego_position[1])) <= lateral
        ):
            return True
    return False


def lane_contains_footprint(environment, x, y):
    """Physical road containment for every corner of the ego footprint."""
    metadata = environment.metadata
    road = metadata["road"]
    ego = metadata["ego_vehicle"]
    half_width, half_height = ego["width"] / 2, ego["height"] / 2
    if not (
        road["y_min"] <= y - half_height and y + half_height <= road["y_max"]
    ):
        return False
    ramp = metadata.get("ramp")
    if ramp is None or y - half_height >= road["lane_divider"]:
        return True
    slope = (road["lane_divider"] - road["y_min"]) / (
        ramp["end_x"] - ramp["start_x"]
    )
    front = min(max(x + half_width, ramp["start_x"]), ramp["end_x"])
    lower = road["y_min"] + slope * (front - ramp["start_x"])
    return y - half_height >= lower


def lane_contains_point(environment, x, y):
    """Backward-compatible alias for footprint-based road containment."""
    return lane_contains_footprint(environment, x, y)


def lane_target_contains(metadata, mean):
    task = metadata["task"]
    return (
        task["target_center"] - task["target_tolerance"]
        <= float(mean[1])
        <= task["target_center"] + task["target_tolerance"]
    )


def lane_dwell_credit(metadata, step, streak):
    """Number of consecutive in-window target samples available at this step."""
    start, _ = metadata["task"]["start_end_steps"]
    if not streak or step < start:
        return 0
    return min(streak, step - start + 1)


def lane_goal_reached(environment, mean, step, streak):
    """The sampled counterpart of the lane completion formula."""
    metadata = environment.metadata
    task = metadata["task"]
    start, end = task["start_end_steps"]
    dwell = task["dwell_steps"]
    if not (start + dwell <= step <= end and streak >= dwell + 1):
        return False
    ramp = metadata.get("ramp")
    return ramp is None or (
        float(mean[0]) + metadata["ego_vehicle"]["width"] / 2 <= ramp["end_x"]
    )


def lane_deadline_missed(environment, mean, step, streak):
    """Return true once no valid dwell can still finish by the deadline."""
    metadata = environment.metadata
    task = metadata["task"]
    start, end = task["start_end_steps"]
    credit = lane_dwell_credit(metadata, step, streak)
    remaining = (
        task["dwell_steps"] - (credit - 1) if credit else task["dwell_steps"]
    )
    earliest_finish = (
        step + remaining if credit else max(step, start) + remaining
    )
    ramp = metadata.get("ramp")
    past_ramp = ramp is not None and (
        float(mean[0]) + metadata["ego_vehicle"]["width"] / 2 > ramp["end_x"]
    )
    return earliest_finish > end or step >= end or past_ramp


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
    if start < 0 or end < start + task["dwell_steps"] or config["H"] < end:
        raise ValueError(
            "lane horizon must cover the completion window and dwell time"
        )
    environment = LaneMergeEnvironment(
        metadata={
            "road": dict(road),
            "ramp": dict(ramp) if ramp is not None else None,
            "traffic": traffic,
            "ego_vehicle": dict(config["ego_vehicle"]),
            "safety_margin": dict(
                config.get(
                    "safety_margin",
                    {"longitudinal": 0.5, "lateral": 0.2},
                )
            ),
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
