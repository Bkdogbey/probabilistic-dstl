"""Geometric events evaluated through belief probability bounds.

`AxisInterval` and `HalfSpace` are atoms a belief evaluates exactly. A rectangle is the
Boolean composition of its two axis intervals, so pdSTL's `beta` relaxes the conjunction
instead of the belief clamping it.
"""

import torch

from pdstl.base import BeliefTrajectory
from pdstl.operators import And, Negation, Predicate, STL_Formula

INF = float("inf")


def _ordered(label, low, high):
    low, high = float(low), float(high)
    if not low < high:
        raise ValueError(
            f"{label} must satisfy min < max, got [{low}, {high}]"
        )
    return low, high


def _state_index(dim):
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
        raise ValueError(
            f"dim must be a non-negative state index, got {dim!r}"
        )
    return dim


class AxisInterval(Predicate):
    """Event lower <= X[dim] <= upper. Either bound may be infinite."""

    def __init__(self, lower, upper, dim=0, name=None):
        lower, upper = _ordered("axis interval", lower, upper)
        dim = _state_index(dim)
        super().__init__(name=name or f"x[{dim}] in [{lower}, {upper}]")
        self.lower, self.upper, self.dim = lower, upper, dim


class GreaterThan(AxisInterval):
    """Event X[dim] >= threshold."""

    sense = ">="

    def __init__(self, threshold, dim=0, name=None):
        super().__init__(
            threshold, INF, dim, name=name or f"x[{dim}] >= {threshold}"
        )
        self.threshold = float(threshold)


class RelativeAxisInterval(AxisInterval):
    """Interval on traffic position minus ego position for one vehicle."""

    def __init__(self, vehicle, lower, upper, dim, name=None):
        super().__init__(
            lower,
            upper,
            dim=dim,
            name=name or f"{vehicle} relative axis {dim}",
        )
        self.vehicle = vehicle


class LessThan(AxisInterval):
    """Event X[dim] <= threshold."""

    sense = "<="

    def __init__(self, threshold, dim=0, name=None):
        super().__init__(
            -INF, threshold, dim, name=name or f"x[{dim}] <= {threshold}"
        )
        self.threshold = float(threshold)


class HalfSpace(Predicate):
    """Event a^T X <= b over the full state vector."""

    def __init__(self, a, b, name=None):
        a, b = tuple(float(v) for v in a), float(b)
        if not a or not any(a):
            raise ValueError("a must be a nonempty, nonzero normal vector")
        if not all(-INF < v < INF for v in (*a, b)):
            raise ValueError("half-space coefficients must be finite")
        super().__init__(name=name or f"HalfSpace({a} @ X <= {b})")
        self.a, self.b = a, b


class _Rectangle(STL_Formula):
    """Rectangle over two state components, as the conjunction of its two axis intervals.

    Keeps its geometry for plotting and delegates evaluation to `self.equivalent`, the way
    `Implies` delegates to its `Or` form.
    """

    negated = False

    def __init__(self, x_range, y_range, dims=(0, 1), name=None):
        super().__init__()
        self.x_range = _ordered("x_range", *x_range)
        self.y_range = _ordered("y_range", *y_range)
        if len(dims) != 2 or dims[0] == dims[1]:
            raise ValueError(
                f"dims must be two distinct state indices, got {tuple(dims)}"
            )
        self.dims = tuple(_state_index(d) for d in dims)

        self.name = name or (
            f"{type(self).__name__}(x[{self.dims[0]}] in {list(self.x_range)}, "
            f"x[{self.dims[1]}] in {list(self.y_range)})"
        )
        self.axes = (
            AxisInterval(*self.x_range, dim=self.dims[0]),
            AxisInterval(*self.y_range, dim=self.dims[1]),
        )
        inside = And(*self.axes)
        self.equivalent = Negation(inside) if self.negated else inside

    @property
    def is_pointwise(self):
        return self.equivalent.is_pointwise

    def robustness_trace(self, belief_trajectory, **kwargs):
        return self.equivalent(belief_trajectory, **kwargs)

    def __str__(self):
        return self.name


class InsideRectangle(_Rectangle):
    """Event X in R (closed)."""


class OutsideRectangle(_Rectangle):
    """Event X not in R, the complement of InsideRectangle."""

    negated = True


class MovingRectangularObstaclePredicate(STL_Formula):
    """Outside a rectangle whose geometry changes at each prediction step."""

    def __init__(self, region):
        super().__init__()
        self.name = region.name
        self.centers = torch.as_tensor(region.centers, dtype=torch.float32)
        if self.centers.ndim != 2 or self.centers.shape[1] != 2:
            raise ValueError(
                "moving rectangle centers must have shape [time, 2]"
            )
        self.width, self.height = float(region.width), float(region.height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("moving rectangle dimensions must be positive")
        self.events = torch.nn.ModuleList()
        for t, center in enumerate(self.centers):
            x, y = (float(v) for v in center)
            self.events.append(
                OutsideRectangle(
                    (x - self.width / 2, x + self.width / 2),
                    (y - self.height / 2, y + self.height / 2),
                    name=f"{self.name} at step {t}",
                )
            )

    @property
    def is_pointwise(self):
        return True

    def robustness_trace(self, belief_trajectory, **kwargs):
        if len(belief_trajectory) > len(self.events):
            raise ValueError("moving rectangle has fewer centers than beliefs")
        bounds = []
        for t, belief in enumerate(belief_trajectory):
            bounds.append(self.events[t](BeliefTrajectory([belief]), **kwargs))
        return torch.cat(bounds, dim=1)
