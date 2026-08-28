"""pdSTL formula core: predicates, Boolean operators, bounded temporal operators.

Every Formula is a torch.nn.Module. forward(source) queries a
ProbabilitySource and returns Tensor[B, T, 2] with [..., 0] = lower and
[..., 1] = upper. Boolean and temporal combinations use dependence-agnostic
Frechet bounds; no independence assumption, no multiplication of probabilities.

Two evaluation modes share one implementation:

``smooth=False`` (the default)
    Hard Frechet semantics. The output is a valid probability interval:
    0 <= lower <= upper <= 1. This is what monitoring and final certification
    must use.

``smooth=True``
    A differentiable surrogate for optimization only. The min/max/clamp
    reductions are replaced by softplus and log-sum-exp with temperature
    ``beta``, so gradients reach the probability source even where the hard
    reduction is flat. The result is *not* a certified bound; it approaches
    the hard result as ``beta`` increases. Rerun with ``smooth=False`` after
    optimization.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

__all__ = [
    "Always",
    "And",
    "Eventually",
    "Formula",
    "Not",
    "Or",
    "Predicate",
    "TemporalOperator",
]


# --- Numerical primitives ---------------------------------------------------
# Stable tensor building blocks, not semantics. Each operator writes its own
# Frechet equations out in full and calls these only for the reduction itself.


def _soft_relu(x, beta):
    """Smooth max(0, x): softplus(beta * x) / beta."""
    return F.softplus(beta * x) / beta


def _soft_max(x, beta, dim):
    """Smooth max over `dim`: logsumexp(beta * x) / beta."""
    return torch.logsumexp(beta * x, dim=dim) / beta


def _soft_min(x, beta, dim):
    """Smooth min over `dim`: -logsumexp(-beta * x) / beta."""
    return -torch.logsumexp(-beta * x, dim=dim) / beta


class Formula(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, source, *, smooth=False, beta=20.0):
        raise NotImplementedError

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)


class Predicate(Formula):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, source, *, smooth=False, beta=20.0):
        # A predicate is a table lookup: nothing to approximate, so `smooth`
        # and `beta` are accepted and ignored.
        return torch.stack([source.bounds(self, t) for t in range(len(source))], dim=1)

    def __str__(self):
        return self.name


class Not(Formula):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def forward(self, source, *, smooth=False, beta=20.0):
        # Negation is exact in both modes; only the child may need smoothing.
        bounds = self.child(source, smooth=smooth, beta=beta)
        lower_not = 1 - bounds[..., 1]
        upper_not = 1 - bounds[..., 0]
        return torch.stack([lower_not, upper_not], dim=-1)

    def __str__(self):
        return f"¬({self.child})"


class And(Formula):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def forward(self, source, *, smooth=False, beta=20.0):
        left_bounds = self.left(source, smooth=smooth, beta=beta)
        if self.left is self.right:
            return left_bounds
        right_bounds = self.right(source, smooth=smooth, beta=beta)

        left_lower, left_upper = left_bounds[..., 0], left_bounds[..., 1]
        right_lower, right_upper = right_bounds[..., 0], right_bounds[..., 1]

        # Frechet intersection: L = max(0, la + lb - 1), U = min(ua, ub).
        if smooth:
            lower = _soft_relu(left_lower + right_lower - 1, beta)
            upper = _soft_min(torch.stack([left_upper, right_upper], dim=-1), beta, -1)
        else:
            lower = torch.clamp(left_lower + right_lower - 1, min=0.0)
            upper = torch.minimum(left_upper, right_upper)
        return torch.stack([lower, upper], dim=-1)

    def __str__(self):
        return f"({self.left}) ∧ ({self.right})"


class Or(Formula):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def forward(self, source, *, smooth=False, beta=20.0):
        left_bounds = self.left(source, smooth=smooth, beta=beta)
        if self.left is self.right:
            return left_bounds
        right_bounds = self.right(source, smooth=smooth, beta=beta)

        left_lower, left_upper = left_bounds[..., 0], left_bounds[..., 1]
        right_lower, right_upper = right_bounds[..., 0], right_bounds[..., 1]

        # Frechet union: L = max(la, lb), U = min(1, ua + ub).
        if smooth:
            lower = _soft_max(torch.stack([left_lower, right_lower], dim=-1), beta, -1)
            total = left_upper + right_upper
            upper = _soft_min(torch.stack([torch.ones_like(total), total], dim=-1), beta, -1)
        else:
            lower = torch.maximum(left_lower, right_lower)
            upper = torch.clamp(left_upper + right_upper, max=1.0)
        return torch.stack([lower, upper], dim=-1)

    def __str__(self):
        return f"({self.left}) ∨ ({self.right})"


class TemporalOperator(Formula, ABC):
    """Shared mechanics for a bounded temporal operator over interval [a, b].

    The hidden state is the raw recent child bounds, Tensor[B, h, 2] with
    h <= b + 1 -- not an accumulated probability. A sliding window has to be
    able to drop an expired step, which a reduced [lower, upper] state cannot
    do. The state transition is exact and fully differentiable (cat + slice);
    only the operator-specific window reduction is hard or smooth.

    The state is an activation owned by the caller: `step` takes it in and
    hands it back, and nothing is stored on the module. The same `step` serves
    both the offline unroll in `forward` and later incremental online use.
    """

    def __init__(self, child, interval):
        super().__init__()
        a, b = interval
        if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int) or not isinstance(b, int):
            raise TypeError(f"interval endpoints must be integers, got {interval!r}")
        if a < 0:
            raise ValueError(f"interval lower endpoint must be >= 0, got {interval!r}")
        if a > b:
            raise ValueError(f"interval must satisfy a <= b, got {interval!r}")

        self.child = child
        self.a = a
        self.b = b

    @property
    def interval(self):
        return (self.a, self.b)

    def forward(self, source, *, smooth=False, beta=20.0):
        """Evaluate over the whole available trace.

        A child trace of shape [B, T, 2] yields [B, max(T - b, 0), 2]: the
        output at anchor k reduces child_trace[:, k + a : k + b + 1, :].
        Incomplete future windows are not padded -- a trace shorter than
        b + 1 simply returns an empty [B, 0, 2].
        """
        child_trace = self.child(source, smooth=smooth, beta=beta)

        state = None
        outputs = []
        for current_bounds in torch.unbind(child_trace, dim=1):
            output, state = self.step(current_bounds, state, smooth=smooth, beta=beta)
            if output is not None:
                outputs.append(output)

        if not outputs:
            # A slice, so dtype, device and the autograd graph all survive.
            return child_trace[:, :0, :]
        return torch.stack(outputs, dim=1)

    def step(self, current_bounds, state=None, *, smooth=False, beta=20.0):
        """Advance one time step.

        `current_bounds` is Tensor[B, 2], the child's bounds at this step, and
        `state` is the previous Tensor[B, h, 2] (or None to start). Returns
        (output, new_state): output is Tensor[B, 2] once b + 1 values have been
        seen, otherwise None while the window is still filling.
        """
        current = current_bounds.unsqueeze(1)
        new_state = current if state is None else torch.cat([state, current], dim=1)
        new_state = new_state[:, -(self.b + 1) :, :]

        if new_state.shape[1] < self.b + 1:
            return None, new_state

        window = new_state[:, self.a : self.b + 1, :]
        return self._reduce_window(window, smooth=smooth, beta=beta), new_state

    @abstractmethod
    def _reduce_window(self, window, *, smooth, beta):
        """Reduce Tensor[B, b - a + 1, 2] to Tensor[B, 2]."""
        raise NotImplementedError


class Always(TemporalOperator):
    """□[a,b]: intersection of the child events across the window."""

    def _reduce_window(self, window, *, smooth, beta):
        lower = window[..., 0]
        upper = window[..., 1]
        window_size = self.b - self.a + 1

        # Frechet intersection: L = max(0, sum(lower) - (n - 1)), U = min(upper).
        z = lower.sum(dim=1) - (window_size - 1)
        if smooth:
            lower_out = _soft_relu(z, beta)
            upper_out = _soft_min(upper, beta, 1)
        else:
            lower_out = torch.clamp(z, min=0.0)
            upper_out = upper.amin(dim=1)
        return torch.stack((lower_out, upper_out), dim=-1)

    def __str__(self):
        return f"□[{self.a},{self.b}]({self.child})"


class Eventually(TemporalOperator):
    """◇[a,b]: union of the child events across the window."""

    def _reduce_window(self, window, *, smooth, beta):
        lower = window[..., 0]
        upper = window[..., 1]

        # Frechet union: L = max(lower), U = min(1, sum(upper)).
        total = upper.sum(dim=1)
        if smooth:
            lower_out = _soft_max(lower, beta, 1)
            upper_candidates = torch.stack((torch.ones_like(total), total), dim=1)
            upper_out = _soft_min(upper_candidates, beta, 1)
        else:
            lower_out = lower.amax(dim=1)
            upper_out = torch.clamp(total, max=1.0)
        return torch.stack((lower_out, upper_out), dim=-1)

    def __str__(self):
        return f"◇[{self.a},{self.b}]({self.child})"
