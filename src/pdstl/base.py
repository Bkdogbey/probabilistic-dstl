"""Belief contract and trajectory containers for pdSTL.

A formula states a requirement; a belief evaluates it under its own uncertainty
model. Shapes: B batch, T trajectory length, D state dimension.

    Belief.probability_bounds(predicate) -> [B, 2]     lower, upper in [0, 1]
    Predicate.robustness_trace(trajectory) -> [B, T, 2]
"""

from abc import ABC, abstractmethod

import torch

# A CDF evaluation may land this far outside [0, 1] through round-off.
_ATOL = 1e-6


class Belief(ABC):
    """Uncertainty over the state at one prediction step."""

    @abstractmethod
    def probability_bounds(self, predicate):
        """Lower and upper probability of the predicate's event, [B, 2].

        Raise ValueError naming the predicate if this model cannot evaluate it.
        """

    def value(self):
        """Point estimate [B, D], for consumers that need a representative state."""
        raise NotImplementedError(f"{type(self).__name__} has no value()")


class BeliefTrajectory:
    """Ordered per-step beliefs. Organizes steps; computes nothing."""

    def __init__(self, beliefs):
        self.beliefs = list(beliefs)

    def __getitem__(self, t):
        return self.beliefs[t]

    def __len__(self):
        return len(self.beliefs)

    def __iter__(self):
        return iter(self.beliefs)

    def suffix(self, t):
        return type(self)(self.beliefs[t:])


class OnlineBeliefTrajectory(BeliefTrajectory):
    """Appendable container for streaming predictions.

    Not an incremental monitor: append() adds an input element, it does not
    advance any temporal-operator state, and re-evaluating a formula recomputes
    the whole trace.
    """

    def __init__(self, beliefs=None):
        super().__init__([] if beliefs is None else beliefs)

    def append(self, belief):
        self.beliefs.append(belief)

    @classmethod
    def from_list(cls, lst):
        return cls(lst)


class ProbabilityBelief(Belief):
    """A step whose bounds are supplied already computed, keyed by event name.

    bounds: {event name: [B, 2] tensor}. Tensors are kept as-is, so a tensor
    that requires grad keeps its graph.
    """

    def __init__(self, bounds, value=None):
        self.bounds = {}
        for name, b in bounds.items():
            b = torch.as_tensor(b)
            if b.ndim != 2 or b.shape[-1] != 2:
                raise ValueError(
                    f"bounds[{name!r}] must be [batch, 2], got {tuple(b.shape)}"
                )
            self.bounds[name] = b
        self._value = value

    def probability_bounds(self, predicate):
        name = getattr(predicate, "name", None)
        if name not in self.bounds:
            raise ValueError(
                f"no supplied bounds for event {name!r}; have {sorted(self.bounds)}"
            )
        return self.bounds[name]

    def value(self):
        return super().value() if self._value is None else self._value


def create_probability_belief_trajectory(predicate, bounds, dtype=None, device=None):
    """Build a trajectory from a supplied ``[T, 2]`` probability-bound trace.

    Column 0 is the lower and column 1 the upper probability of the predicate's
    event. Tensors are passed through rather than copied, so gradients survive.
    """
    bounds = torch.as_tensor(bounds, dtype=dtype, device=device)
    if bounds.ndim != 2 or bounds.shape[-1] != 2:
        raise ValueError(
            f"probability bounds must have shape [T, 2], got {tuple(bounds.shape)}"
        )
    if bounds.shape[0] < 1:
        raise ValueError("probability bounds must cover at least one step")
    check_probability_bounds(bounds.unsqueeze(0), predicate)

    return BeliefTrajectory(
        [
            ProbabilityBelief({predicate.name: bounds[t : t + 1]})
            for t in range(bounds.shape[0])
        ]
    )


def check_probability_bounds(trace, predicate=None):
    """Reject malformed bounds: non-finite, unordered, or outside [0, 1].

    Raises rather than clamping, which would silently change both the value and
    its gradient. Well-formedness is not coverage -- that claim belongs to
    whoever produced the numbers.
    """
    who = "predicate"
    if predicate is not None:
        who = getattr(predicate, "name", None) or type(predicate).__name__

    lower, upper = trace[..., 0], trace[..., 1]
    bad = (
        ~torch.isfinite(trace).all(dim=-1)
        | (lower > upper + _ATOL)
        | (trace < -_ATOL).any(dim=-1)
        | (trace > 1.0 + _ATOL).any(dim=-1)
    )  # [B, T]

    if bool(bad.any()):
        t = int(bad.any(dim=0).nonzero()[0])
        raise ValueError(
            f"{who}: malformed probability bounds at step {t}: {trace[:, t].tolist()}"
        )
