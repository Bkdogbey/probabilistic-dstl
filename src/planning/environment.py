"""Rectangular geometry, the reach-avoid task, and lane construction."""

import heapq
from dataclasses import dataclass
from functools import cache, reduce
from itertools import permutations, product
from math import ceil, floor, hypot, isfinite

import numpy as np

from pdstl.operators import Always, And, Eventually, Or
from pdstl.predicates import (
    AxisInterval,
    HalfSpace,
    InsideRectangle,
    LessThan,
    OutsideRectangle,
    RelativeAxisInterval,
)

ROLES = ("workspace", "goal", "obstacle", "target")


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

    @property
    def centre(self):
        return [(self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2]


class Environment:
    """Workspace bounds, obstacles, a goal and visit regions.

    `goal` is (interval, names) and `visits` is [(interval, dwell, names)];
    several names are alternatives, any one of which satisfies the task.
    """

    def __init__(self):
        self.regions = {}
        self.goal = None
        self.visits = []

    def set_bounds(self, x_range, y_range):
        """The workspace the trajectory must always stay inside."""
        self._add("bounds", "workspace", x_range, y_range)

    def add_obstacle(self, x_range, y_range, name=None):
        """A rectangle the trajectory must always stay out of."""
        count = len(self.by_role("obstacle"))
        self._add(
            name or f"obstacle {count + 1}", "obstacle", x_range, y_range
        )

    def set_goal(
        self, x_range=None, y_range=None, any_of=None, interval=None, name=None
    ):
        """A region to reach within `interval`, or one of `any_of`."""
        names = self._alternatives("goal", name, x_range, y_range, any_of)
        self.goal = (interval, names)

    def add_visit_region(
        self,
        x_range=None,
        y_range=None,
        any_of=None,
        dwell=0,
        interval=None,
        name=None,
    ):
        """A region to enter within `interval` and stay in for `dwell` steps."""
        label = f"visit {len(self.visits) + 1}"
        names = self._alternatives(label, name, x_range, y_range, any_of)
        self.visits.append((interval, int(dwell), names))

    def _alternatives(self, label, name, x_range, y_range, any_of):
        role = "goal" if label == "goal" else "target"
        if any_of is None:
            return [self._add(name or label, role, x_range, y_range).name]
        return [
            self._add(
                box.get("name", f"{label} {chr(ord('a') + k)}"),
                role,
                box["x_range"],
                box["y_range"],
            ).name
            for k, box in enumerate(any_of)
        ]

    def _add(self, name, role, x_range, y_range):
        (xmin, xmax), (ymin, ymax) = x_range, y_range
        return self.add_region(
            RectangleRegion(
                name, role, float(xmin), float(xmax), float(ymin), float(ymax)
            )
        )

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
        """The pdSTL task over prediction steps 0..horizon."""
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError(
                f"horizon must be a positive integer, got {horizon!r}"
            )
        return self._specification(horizon)

    def _specification(self, horizon):
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
    """Stay safe, reach the goal, and dwell in each visit region.

    Args:
        environment: Environment with bounds, a goal and visit regions.
        horizon: Last prediction step H; step 0 is the known initial belief.

    Returns:
        G[1,H] safe & F[goal interval] goal
        & F[visit interval] G[0,dwell] visit, for every visit region.
    """
    if environment.goal is None:
        raise ValueError("reach-avoid needs a goal")
    interval, names = environment.goal
    parts = [
        Always(safety_event(environment), [1, horizon]),
        Eventually(_any(environment, names, 0), _window(interval, horizon, 0)),
    ]
    for interval, dwell, names in environment.visits:
        parts.append(
            Eventually(
                _any(environment, names, dwell),
                _window(interval, horizon, dwell),
            )
        )
    return reduce(And, parts)


def _any(environment, names, dwell):
    """Inside one of the named regions for `dwell` more steps."""
    stays = [
        Always(inside(environment.region(name)), [0, dwell])
        if dwell
        else inside(environment.region(name))
        for name in names
    ]
    return reduce(Or, stays)


def _window(interval, horizon, dwell):
    """The requested window, defaulting to [1, H - dwell]."""
    start, end = interval or (1, horizon - dwell)
    if not 0 <= start <= end <= horizon - dwell:
        raise ValueError(
            f"interval {[start, end]} with dwell {dwell} must fit in [0, {horizon}]"
        )
    return [start, end]


def build_reach_avoid_environment(config):
    """Build the reach-avoid environment from a scenario config.

    Args:
        config: Mapping with `bounds`, `goal`, and optional `obstacles` and
            `visit_regions`; see configs/scenarios/reach_avoid/.

    Returns:
        The Environment.
    """
    environment = Environment()
    environment.set_bounds(**config["bounds"])
    for obstacle in config.get("obstacles") or []:
        environment.add_obstacle(**obstacle)
    environment.set_goal(**config["goal"])
    for visit in config.get("visit_regions") or []:
        environment.add_visit_region(**visit)
    return environment


# Shortest collision-free route: where the optimizer starts

_NEIGHBOURS = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]


def shortest_route(environment, start, clearance=0.05, resolution=0.05):
    """Shortest collision-free route from start through the task regions.

    Only a nominal start for the optimizer: `clearance` keeps it off the
    obstacles, and pdSTL adds any uncertainty-aware separation. The route
    visits the centre of the goal (or one of its alternatives) and of every
    visit region, in any order; the shortest combination wins.

    Args:
        environment: Reach-avoid Environment.
        start: [x, y] start position.
        clearance: Distance kept from obstacles and the workspace edge.
        resolution: Grid cell size of the search.

    Returns:
        [[x, y], ...] corner points after the start.
    """
    grid = _FreeGrid(environment, clearance, resolution)
    search = cache(grid.search)
    groups = [
        [environment.region(name) for name in names]
        for names in [environment.goal[1]]
        + [names for _, _, names in environment.visits]
    ]
    stops = [
        [grid.cell(start)] + [grid.cell(region.centre) for region in order]
        for choice in product(*groups)
        for order in permutations(choice)
    ]

    def length(cells):
        return sum(search(a)[0][b] for a, b in zip(cells, cells[1:]))

    best = min(stops, key=length)
    if not isfinite(length(best)):
        raise ValueError(
            f"no route keeps {clearance} m from every obstacle; "
            "lower route.clearance or move the obstacles"
        )
    route = []
    for a, b in zip(best, best[1:]):
        route += grid.shortcut(grid.trace(search(a)[1], a, b))
    return route


def _grown(x, y, region, margin):
    return (
        (x >= region.xmin - margin)
        & (x <= region.xmax + margin)
        & (y >= region.ymin - margin)
        & (y <= region.ymax + margin)
    )


class _FreeGrid:
    """Workspace cells that keep `clearance` from obstacles and the edge."""

    def __init__(self, environment, clearance, resolution):
        workspace = environment.single_region("workspace")
        self.step = resolution
        self.xs = np.arange(workspace.xmin, workspace.xmax, self.step)
        self.ys = np.arange(workspace.ymin, workspace.ymax, self.step)
        x, y = np.meshgrid(self.xs, self.ys, indexing="ij")
        self.free = _grown(x, y, workspace, -clearance)
        for obstacle in environment.by_role("obstacle"):
            self.free &= ~_grown(x, y, obstacle, clearance)

    def point(self, cell):
        return np.array([self.xs[cell[0]], self.ys[cell[1]]])

    def cell(self, point):
        """The free cell nearest to a point."""
        i, j = np.nonzero(self.free)
        k = np.argmin(
            (self.xs[i] - point[0]) ** 2 + (self.ys[j] - point[1]) ** 2
        )
        return int(i[k]), int(j[k])

    def search(self, source):
        """Dijkstra from one cell: distances to every cell, and parents."""
        distance = np.full(self.free.shape, np.inf)
        distance[source] = 0.0
        parent, queue = {}, [(0.0, source)]
        while queue:
            d, (i, j) = heapq.heappop(queue)
            if d > distance[i, j]:
                continue
            for di, dj in _NEIGHBOURS:
                neighbour = (i + di, j + dj)
                cost = d + hypot(di, dj)
                if self._open(neighbour) and cost < distance[neighbour]:
                    distance[neighbour], parent[neighbour] = cost, (i, j)
                    heapq.heappush(queue, (cost, neighbour))
        return distance, parent

    def _open(self, cell):
        (i, j), (nx, ny) = cell, self.free.shape
        return 0 <= i < nx and 0 <= j < ny and self.free[i, j]

    def trace(self, parent, source, target):
        cells = [target]
        while cells[-1] != source:
            cells.append(parent[cells[-1]])
        return cells[::-1]

    def shortcut(self, cells):
        """Keep only the corners: skip every cell a straight line can."""
        points = [self.point(cell) for cell in cells]
        corners, k = [], 0
        while k < len(points) - 1:
            m = len(points) - 1
            while m > k + 1 and not self._visible(points[k], points[m]):
                m -= 1
            corners.append(points[m].tolist())
            k = m
        return corners

    def _visible(self, p, q):
        samples = np.linspace(p, q, int(np.linalg.norm(q - p) / self.step) + 2)
        i = np.rint((samples[:, 0] - self.xs[0]) / self.step).astype(int)
        j = np.rint((samples[:, 1] - self.ys[0]) / self.step).astype(int)
        return bool(self.free[i, j].all())


# Lane-change construction


class LaneMergeEnvironment(Environment):
    """An Environment that also carries the road description this scenario needs."""

    def __init__(self, metadata=None):
        super().__init__()
        self.metadata = dict(metadata or {})

    def _specification(self, horizon):
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


def lane_local_window(environment, step, streak=0):
    """Give a planning window its absolute time and observed dwell progress."""
    window = LaneMergeEnvironment(
        metadata={**environment.metadata, "step": step, "streak": streak}
    )
    window.regions = environment.regions
    return window
