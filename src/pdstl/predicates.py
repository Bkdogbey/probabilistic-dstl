"""Geometric events and the existing Gaussian-moment lane/baseline predicates.

`AxisInterval` and `HalfSpace` are atoms a belief evaluates exactly. A rectangle is the
Boolean composition of its two axis intervals, so pdSTL's `beta` relaxes the conjunction
instead of the belief clamping it.
"""

import math

import torch

from pdstl.operators import And, Negation, Predicate, STL_Formula

INF = float("inf")


def _ordered(label, low, high):
    low, high = float(low), float(high)
    if not low < high:
        raise ValueError(f"{label} must satisfy min < max, got [{low}, {high}]")
    return low, high


def _state_index(dim):
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
        raise ValueError(f"dim must be a non-negative state index, got {dim!r}")
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
        super().__init__(threshold, INF, dim, name=name or f"x[{dim}] >= {threshold}")
        self.threshold = float(threshold)


class LessThan(AxisInterval):
    """Event X[dim] <= threshold."""

    sense = "<="

    def __init__(self, threshold, dim=0, name=None):
        super().__init__(-INF, threshold, dim, name=name or f"x[{dim}] <= {threshold}")
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
            raise ValueError(f"dims must be two distinct state indices, got {tuple(dims)}")
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


# Legacy lane/baseline predicates below assume Gaussian moments. They do not
# change the probability_bounds contract of the geometric events above.

def extract_trajectory_stats(belief_trajectory, diagonal_only=True):
    """Stack means [B,T,D] and variances [B,T,D] (or full covariances) over the trajectory."""
    means, vars_ = [], []
    for belief in belief_trajectory:
        means.append(belief.value())
        if diagonal_only and belief.covariance.ndim > 2:
            vars_.append(torch.diagonal(belief.covariance, dim1=-2, dim2=-1))
        else:
            vars_.append(belief.covariance)
    return torch.stack(means, dim=1), torch.stack(vars_, dim=1)


def normal_cdf(value, mean, var):
    """P(X <= value) for X ~ N(mean, var)."""
    z = (value - mean) / torch.sqrt(var + 1e-6)
    return 0.5 * (1 + torch.erf(z / math.sqrt(2)))


class CircularObstaclePredicate(STL_Formula):
    """P(||x - center|| > radius), using the variance projected on the radial direction."""

    def __init__(self, region):
        super().__init__()
        self.name = region.name
        self.center = torch.as_tensor(region.center, dtype=torch.float32)
        self.radius = float(region.radius)

    def robustness_trace(self, belief_trajectory, **kwargs):
        mu, cov = extract_trajectory_stats(belief_trajectory, diagonal_only=False)
        diff = mu - self.center.to(mu.device)
        dist = torch.norm(diff, dim=-1)
        direction = diff / (dist.unsqueeze(-1) + 1e-6)
        if cov.ndim == 3:
            radial_var = torch.sum(direction**2 * cov, dim=-1)
        else:
            radial_var = torch.einsum("bti,btij,btj->bt", direction, cov, direction)
        p_safe = 1.0 - normal_cdf(self.radius, dist, radial_var)
        return torch.stack([p_safe, p_safe], dim=-1)


class MovingRectangularObstaclePredicate(STL_Formula):
    """Max of the four one-sided probabilities of being outside a moving rectangle."""

    def __init__(self, region):
        super().__init__()
        self.name = region.name
        centers = torch.as_tensor(region.centers, dtype=torch.float32)
        self.x_traj, self.y_traj = centers[..., 0], centers[..., 1]
        self.width, self.height = float(region.width), float(region.height)

    def robustness_trace(self, belief_trajectory, **kwargs):
        mu, var = extract_trajectory_stats(belief_trajectory)
        mu_x, mu_y, var_x, var_y = mu[..., 0], mu[..., 1], var[..., 0], var[..., 1]
        half_w, half_h = self.width / 2.0, self.height / 2.0
        x_traj, y_traj = self.x_traj.to(mu.device), self.y_traj.to(mu.device)
        p_safe = torch.stack([
            normal_cdf(x_traj - half_w, mu_x, var_x),
            1.0 - normal_cdf(x_traj + half_w, mu_x, var_x),
            normal_cdf(y_traj - half_h, mu_y, var_y),
            1.0 - normal_cdf(y_traj + half_h, mu_y, var_y),
        ], dim=0).max(dim=0).values
        return torch.stack([p_safe, p_safe], dim=-1)
