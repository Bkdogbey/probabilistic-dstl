"""Spatial events: geometry only; beliefs supply their probabilities.

Rectangles are *compositions*, not atoms. A rectangle is the conjunction of two exact
one-dimensional `AxisInterval` events, so the Frechet conjunction lives in `operators.py`
where `scale` can relax it. Evaluating the conjunction inside the belief instead would clamp
its lower bound to zero before any smooth operator saw it, and no gradient would survive the
approach to a distant goal.

Hard evaluation is unchanged by this: `_conjunction` at `scale <= 0` is exactly
`[max(0, p_x + p_y - 1), min(p_x, p_y)]`, and `_negation` is exactly the complement.
"""

from pdstl.operators import And, Negation, Predicate, STL_Formula


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


def _state_index(dim):
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
        raise ValueError(f"dim must be a non-negative state index, got {dim!r}")
    return dim


class AxisInterval(Predicate):
    """Closed one-dimensional event lower <= X[dim] <= upper.

    The belief evaluates it exactly, as Phi((upper - mu)/sigma) - Phi((lower - mu)/sigma).
    This is the atom rectangles are built from.
    """

    def __init__(self, lower, upper, dim=0, name=None):
        lower, upper = _ordered("axis interval", (lower, upper))
        dim = _state_index(dim)
        super().__init__(name=name or f"x[{dim}] in [{lower}, {upper}]")
        self.lower, self.upper, self.dim = lower, upper, dim


class _Rectangle(STL_Formula):
    """Rectangle [x_min, x_max] x [y_min, y_max] over two state components.

    Holds the geometry (`x_range`, `y_range`, `dims`) for visualization and reporting, and
    delegates evaluation to the axis-interval composition built in `__init__`. This mirrors
    how `Implies` delegates to `self.equivalent`.
    """

    def __init__(self, x_range, y_range, dims=(0, 1), name=None):
        super().__init__()
        x_range, y_range = _ordered("x_range", x_range), _ordered("y_range", y_range)
        dims = tuple(dims)
        if len(dims) != 2 or dims[0] == dims[1]:
            raise ValueError(f"dims must be two distinct state indices, got {dims}")
        dims = tuple(_state_index(d) for d in dims)

        self.x_range, self.y_range, self.dims = x_range, y_range, dims
        self.name = name or (
            f"{type(self).__name__}(x[{dims[0]}] in {list(x_range)}, "
            f"x[{dims[1]}] in {list(y_range)})"
        )
        self.axes = (
            AxisInterval(*x_range, dim=dims[0]),
            AxisInterval(*y_range, dim=dims[1]),
        )
        self.equivalent = self._compose(And(*self.axes))

    @staticmethod
    def _compose(inside):
        raise NotImplementedError

    @property
    def is_pointwise(self):
        return self.equivalent.is_pointwise

    def robustness_trace(self, belief_trajectory, **kwargs):
        return self.equivalent(belief_trajectory, **kwargs)

    def __str__(self):
        return self.name


class InsideRectangle(_Rectangle):
    """Event X in R (closed): AxisInterval(x) AND AxisInterval(y)."""

    @staticmethod
    def _compose(inside):
        return inside


class OutsideRectangle(_Rectangle):
    """Event X not in R: NOT(AxisInterval(x) AND AxisInterval(y))."""

    @staticmethod
    def _compose(inside):
        return Negation(inside)
