"""Spatial events: geometry only; beliefs supply their probabilities."""

from pdstl.operators import Predicate


def _ordered(label, bounds):
    low, high = (float(b) for b in bounds)
    if not low < high:
        raise ValueError(f"{label} must satisfy min < max, got {list(bounds)}")
    return low, high


class _Rectangle(Predicate):
    """Rectangle [x_min, x_max] x [y_min, y_max] over state components dims."""

    def __init__(self, x_range, y_range, dims=(0, 1), name=None):
        x_range, y_range, dims = _ordered("x_range", x_range), _ordered("y_range", y_range), tuple(dims)
        if len(dims) != 2 or dims[0] == dims[1] or not all(isinstance(d, int) and d >= 0 for d in dims):
            raise ValueError(f"dims must be two distinct state indices, got {dims}")
        name = name or f"{type(self).__name__}(x[{dims[0]}] in {list(x_range)}, x[{dims[1]}] in {list(y_range)})"
        super().__init__(name=name)
        self.x_range, self.y_range, self.dims = x_range, y_range, dims


class InsideRectangle(_Rectangle):
    """Event X in R (closed)."""


class OutsideRectangle(_Rectangle):
    """Event X not in R, the complement of InsideRectangle."""
