from abc import ABC, abstractmethod
from collections.abc import Mapping
from numbers import Real
import math

import torch


def validate_scale(scale):
    if isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale):
        raise ValueError("scale must be a finite real number")


def validate_interval_trace(trace, name="predicate"):
    """Validate a scalar event's bounds without copying or detaching its tensor."""
    if not isinstance(trace, torch.Tensor) or not trace.is_floating_point():
        raise TypeError(f"{name}: expected a floating-point torch.Tensor")
    if trace.ndim != 3 or trace.shape[-1] != 2 or min(trace.shape[:2]) == 0:
        raise ValueError(f"{name}: expected nonempty [batch, time, 2] bounds")
    if not torch.isfinite(trace).all():
        raise ValueError(f"{name}: bounds must be finite")
    lower, upper = trace.unbind(-1)
    if ((lower < 0) | (upper > 1) | (lower > upper)).any():
        raise ValueError(f"{name}: require 0 <= lower <= upper <= 1")
    return trace


class STL_Formula(torch.nn.Module):
    """Bounded, discrete-time interval score with complete-window outputs.

    ``formula(inputs, scale=...)`` returns [batch, time-horizon, 2].
    ``robustness(inputs)`` selects the origin. Positive scale approximates the
    endpoint equations; its output is not automatically a probability interval.
    Custom leaves have horizon zero and must return one scalar event per step.
    """

    @property
    def horizon(self):
        return 0

    def forward(self, inputs, scale=-1, keepdim=True, **kwargs):
        validate_scale(scale)
        if not keepdim:
            raise ValueError("traces retain their time axis; use robustness(..., keepdim=False)")
        if isinstance(inputs, Mapping):
            if not inputs:
                raise ValueError("predicate inputs cannot be empty")
            reference = None
            for name, value in inputs.items():
                trace = validate_interval_trace(value, str(name))
                layout = (trace.shape, trace.dtype, trace.device)
                if reference is not None and layout != reference:
                    raise ValueError("all predicate inputs must share batch, time, dtype, and device")
                reference = layout
            length = reference[0][1]
        else:
            length = len(inputs)
        if length <= self.horizon:
            raise ValueError(
                f"formula requires at least {self.horizon + 1} prediction steps; got {length}"
            )
        return self.robustness_trace(inputs, scale=scale, **kwargs)

    def robustness_trace(self, inputs, scale=-1, **kwargs):
        raise NotImplementedError

    def robustness(self, inputs, scale=-1, keepdim=True, **kwargs):
        """Return the score at the prediction origin: [B,1,2] or [B,2]."""
        origin = self(inputs, scale=scale, **kwargs)[:, :1, :]
        return origin if keepdim else origin.squeeze(1)

    def smoothing_error(self, scale):
        """Absolute endpoint-error bound; custom formulas must declare their own."""
        raise NotImplementedError("no smoothing-error bound declared for this formula")

    def __and__(self, other):
        from .operators import And
        return And(self, other)

    def __or__(self, other):
        from .operators import Or
        return Or(self, other)

    def __invert__(self):
        from .operators import Negation
        return Negation(self)


class Belief(ABC):
    """
    Abstract base class for any belief representation.
    Users must implement:
        - value() -> representative range of state (tensor)
        - probability_of(residual) -> probability (tensor)

    Legacy state-based adapter. New providers can pass interval tensors to
    Predicate directly. The legacy comparison operators additionally require
    lower_bound() and upper_bound(); these are not a universal belief interface.
    """

    @abstractmethod
    def value(self):
        """
        Return a representative range of state x(t)
        """
        raise NotImplementedError

    @abstractmethod
    def probability_of(self, residual):
        """
        Return P(residual >= 0)
        """
        raise NotImplementedError


class BeliefTrajectory:
    """
    Offline belief trajectory with list of beliefs
    """

    def __init__(self, beliefs):
        self.beliefs = beliefs

    def __getitem__(self, t):
        return self.beliefs[t]

    def __len__(self):
        return len(self.beliefs)

    def suffix(self, t):
        return BeliefTrajectory(self.beliefs[t:])


class OnlineBeliefTrajectory:
    """
    Legacy appendable belief list, not an incremental temporal monitor.
    Future-window evaluation still requires a complete prediction at one origin.
    """

    def __init__(self):
        self.beliefs = []

    def append(self, belief):
        self.beliefs.append(belief)

    def __getitem__(self, t):
        return self.beliefs[t]

    def suffix(self, t):
        return OnlineBeliefTrajectory.from_list(self.beliefs[t:])

    @classmethod
    def from_list(cls, lst):
        obj = cls()
        obj.beliefs = lst
        return obj

    def __len__(self):
        return len(self.beliefs)
