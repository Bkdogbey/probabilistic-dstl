"""Probability-bound input interface for pdSTL.

A ProbabilitySource supplies bounds [lower, upper] on the satisfaction
probability of an atomic predicate at a discrete time, as a tensor of shape
[B, 2] with bounds[..., 0] = lower and bounds[..., 1] = upper, satisfying
0 <= lower <= upper <= 1. An exact probability p is represented as [p, p].
"""

from abc import ABC, abstractmethod

import torch

__all__ = ["ProbabilitySource", "OfflineSource", "OnlineSource", "validate_bounds"]


def validate_bounds(bounds, *, context=""):
    """Check bounds are a finite tensor with trailing dim 2 and lower <= upper."""
    where = f" ({context})" if context else ""

    if not isinstance(bounds, torch.Tensor):
        raise TypeError(f"bounds must be a torch.Tensor, got {type(bounds).__name__}{where}")
    if bounds.ndim == 0 or bounds.shape[-1] != 2:
        raise ValueError(f"bounds must have trailing dimension 2, got shape {tuple(bounds.shape)}{where}")
    if not torch.isfinite(bounds).all():
        raise ValueError(f"bounds contain non-finite values: {bounds}{where}")

    lower, upper = bounds[..., 0], bounds[..., 1]
    if bool((lower < 0.0).any()):
        raise ValueError(f"lower bound must be >= 0, got {bounds}{where}")
    if bool((upper > 1.0).any()):
        raise ValueError(f"upper bound must be <= 1, got {bounds}{where}")
    if bool((lower > upper).any()):
        raise ValueError(f"lower bound must be <= upper bound, got {bounds}{where}")

    return bounds


class ProbabilitySource(ABC):
    """Supplies atomic satisfaction-probability bounds [lower, upper]."""

    @abstractmethod
    def bounds(self, predicate, time):
        """Return bounds for `predicate` at `time`, shape [B, 2]."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Return the number of currently available time steps."""
        raise NotImplementedError


class OfflineSource(ProbabilitySource):
    """Complete offline traces, keyed by predicate: Tensor[B, T, 2]."""

    def __init__(self, traces):
        if not traces:
            raise ValueError("traces must contain at least one predicate")

        batch = length = None
        for predicate, trace in traces.items():
            validate_bounds(trace, context=str(predicate))
            if batch is None:
                batch, length = trace.shape[0], trace.shape[1]
            elif (trace.shape[0], trace.shape[1]) != (batch, length):
                raise ValueError(
                    f"all traces must share batch size and length, got "
                    f"{tuple(trace.shape[:2])} for {predicate}, expected {(batch, length)}"
                )

        self._traces = dict(traces)
        self._length = length

    def bounds(self, predicate, time):
        return self._traces[predicate][:, time, :]

    def __len__(self):
        return self._length


class OnlineSource(ProbabilitySource):
    """Growing source; each append() call adds one time step."""

    def __init__(self):
        self._steps = []
        self._predicates = None
        self._batch = None

    def append(self, bounds):
        predicates = frozenset(bounds)
        if self._predicates is not None and predicates != self._predicates:
            raise ValueError(f"expected predicates {self._predicates}, got {predicates}")

        batch = None
        for predicate, tensor in bounds.items():
            validate_bounds(tensor, context=str(predicate))
            if batch is None:
                batch = tensor.shape[0]
            elif tensor.shape[0] != batch:
                raise ValueError(f"inconsistent batch size within append: {tensor.shape[0]} != {batch}")

        if self._batch is not None and batch != self._batch:
            raise ValueError(f"batch size changed: expected {self._batch}, got {batch}")

        self._predicates = predicates
        self._batch = batch
        self._steps.append(dict(bounds))

    def bounds(self, predicate, time):
        return self._steps[time][predicate]

    def __len__(self):
        return len(self._steps)
