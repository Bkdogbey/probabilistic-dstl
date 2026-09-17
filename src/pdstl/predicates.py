"""Spatial events: geometry only; beliefs supply their probabilities."""

from pdstl.operators import Predicate


class HalfSpace(Predicate):
    """Closed affine event a^T X <= b over the full state vector.

    Geometry only: the belief supplies the event probability. Coefficients are
    fixed, finite scalars; the normal must be nonzero.
    """

    def __init__(self, a, b, name=None):
        a, b = tuple(float(value) for value in a), float(b)
        if not a or not any(a):
            raise ValueError("a must be a nonempty, nonzero normal vector")
        if not all(-float("inf") < value < float("inf") for value in (*a, b)):
            raise ValueError("half-space coefficients must be finite")
        super().__init__(name=name or f"HalfSpace({a} @ X <= {b})")
        self.a, self.b = a, b


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
