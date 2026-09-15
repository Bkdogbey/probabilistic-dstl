"""Spatial pdSTL specifications built only from primitive atoms.

Each builder returns a formula; none of them sees a belief. Probabilities come
later, when a belief trajectory evaluates the atoms through its own
probability_bounds, and And/Or combine them with the pdSTL (Frechet) rules --
no product of marginals, no x/y independence assumption.

Rectangles are given as ``x_range = [min, max]``, ``y_range = [min, max]``.
"""

from pdstl.operators import Always, And, Eventually, GreaterThan, LessThan, Or


def _ordered(name, bounds):
    low, high = bounds
    if not low < high:
        raise ValueError(f"{name} must satisfy min < max, got {list(bounds)}")
    return low, high


def inside_rectangle(x_range, y_range, x_dim=0, y_dim=1):
    """(x >= x_min) and (x <= x_max) and (y >= y_min) and (y <= y_max)."""
    x_min, x_max = _ordered("x_range", x_range)
    y_min, y_max = _ordered("y_range", y_range)
    return And(
        And(GreaterThan(x_min, dim=x_dim), LessThan(x_max, dim=x_dim)),
        And(GreaterThan(y_min, dim=y_dim), LessThan(y_max, dim=y_dim)),
    )


def outside_rectangle(x_range, y_range, x_dim=0, y_dim=1):
    """(x <= x_min) or (x >= x_max) or (y <= y_min) or (y >= y_max).

    The complement geometry of the rectangle's open interior.
    """
    x_min, x_max = _ordered("x_range", x_range)
    y_min, y_max = _ordered("y_range", y_range)
    return Or(
        Or(LessThan(x_min, dim=x_dim), GreaterThan(x_max, dim=x_dim)),
        Or(LessThan(y_min, dim=y_dim), GreaterThan(y_max, dim=y_dim)),
    )


def reach_avoid(goal, obstacle, H, x_dim=0, y_dim=1):
    """Always[1, H](outside obstacle) and Eventually[1, H](inside goal).

    goal, obstacle: {"x_range": [min, max], "y_range": [min, max]}.
    Step 0 is the fixed initial belief, so both windows start at 1.
    """
    safe = outside_rectangle(**obstacle, x_dim=x_dim, y_dim=y_dim)
    reach = inside_rectangle(**goal, x_dim=x_dim, y_dim=y_dim)
    return And(Always(safe, interval=[1, H]), Eventually(reach, interval=[1, H]))
