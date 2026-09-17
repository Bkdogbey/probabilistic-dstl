"""Rectangular geometry and the environment the planner asks for a specification.

This module knows about rectangles and names. It does not sample trajectories, estimate
probabilities, compute robustness, or draw anything, and it has no opinion on whether the
planner uses the exact or the smooth semantics.
"""

from dataclasses import dataclass, field

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
    """Named regions plus the specification builder the scenario injected."""

    def __init__(self, specification_builder):
        self.regions = {}
        self._specification_builder = specification_builder

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

    def get_specification(self, horizon):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
        return self._specification_builder(self, horizon)
