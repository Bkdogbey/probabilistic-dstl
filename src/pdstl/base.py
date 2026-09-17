"""Belief contract and trajectory containers: Belief.probability_bounds(event) -> [B, 2]."""

from abc import ABC, abstractmethod

import torch

_ATOL = 1e-6  # round-off allowed outside [0, 1]


class Belief(ABC):
    """Uncertainty over the state at one prediction step."""

    @abstractmethod
    def probability_bounds(self, predicate):
        """[B, 2] lower/upper probability of the event; ValueError if unsupported."""

    def value(self):
        """Representative state [B, D]."""
        raise NotImplementedError(f"{type(self).__name__} has no value()")


class BeliefTrajectory:
    """Ordered per-step beliefs."""

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
    """Appendable trajectory; formulas re-evaluate the whole trace (not incremental)."""

    def __init__(self, beliefs=None):
        super().__init__([] if beliefs is None else beliefs)

    def append(self, belief):
        self.beliefs.append(belief)

    @classmethod
    def from_list(cls, lst):
        return cls(lst)


class ProbabilityBelief(Belief):
    """Precomputed bounds per event name: {name: [B, 2] tensor}, graphs kept."""

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


def create_belief_trajectory(traces, dtype=None, device=None):
    """Trajectory from precomputed probability intervals, one [T, 2] trace per named event.

        create_belief_trajectory({"goal": goal_intervals, "safe": safe_intervals})

    Every trace must cover the same number of steps. Gradients are kept.
    """
    if not traces:
        raise ValueError("at least one event trace is required")

    tensors = {}
    for name, trace in traces.items():
        tensor = torch.as_tensor(trace, dtype=dtype, device=device)
        if tensor.ndim != 2 or tensor.shape[-1] != 2:
            raise ValueError(
                f"probability bounds for {name!r} must have shape [T, 2], "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.shape[0] < 1:
            raise ValueError("probability bounds must cover at least one step")
        check_probability_bounds(tensor.unsqueeze(0))
        tensors[name] = tensor

    steps = {name: tensor.shape[0] for name, tensor in tensors.items()}
    if len(set(steps.values())) > 1:
        raise ValueError(f"every trace must cover the same number of steps, got {steps}")

    return BeliefTrajectory(
        ProbabilityBelief({name: tensor[t : t + 1] for name, tensor in tensors.items()})
        for t in range(next(iter(steps.values())))
    )


def create_probability_belief_trajectory(predicate, bounds, dtype=None, device=None):
    """Trajectory from a [T, 2] (lower, upper) trace for one event; gradients kept."""
    return create_belief_trajectory({predicate.name: bounds}, dtype=dtype, device=device)


def check_probability_bounds(trace, predicate=None):
    """Raise (never clamp) on non-finite, unordered, or out-of-[0, 1] bounds."""
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
