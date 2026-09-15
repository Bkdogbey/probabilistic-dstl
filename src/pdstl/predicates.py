"""Spatial events.

A predicate names an event; it holds geometry only. Each belief evaluates the
event's probability through ``Belief.probability_bounds(predicate)``, and the
pdSTL operators compose the resulting intervals over logic and time.
"""

from pdstl.operators import Predicate


class _Rectangle(Predicate):
    """Axis-aligned rectangle R = [x_min, x_max] x [y_min, y_max].

    ``dims`` names the two state components that play the roles of x and y.
    """

    def __init__(self, x_range, y_range, dims=(0, 1), name=None):
        x_range = _ordered("x_range", x_range)
        y_range = _ordered("y_range", y_range)
        dims = tuple(dims)
        if (
            len(dims) != 2
            or not all(isinstance(d, int) and d >= 0 for d in dims)
            or dims[0] == dims[1]
        ):
            raise ValueError(f"dims must be two distinct state indices, got {dims}")
        if name is None:
            name = (
                f"{type(self).__name__}(x[{dims[0]}] in {list(x_range)}, "
                f"x[{dims[1]}] in {list(y_range)})"
            )
        super().__init__(name=name)
        self.x_range = x_range
        self.y_range = y_range
        self.dims = dims


def _ordered(label, bounds):
    low, high = (float(b) for b in bounds)
    if not low < high:
        raise ValueError(f"{label} must satisfy min < max, got {list(bounds)}")
    return low, high


class InsideRectangle(_Rectangle):
    """The event X in R (closed rectangle)."""


class OutsideRectangle(_Rectangle):
    """The event X not in R: the exact complement of InsideRectangle."""
