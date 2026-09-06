"""Distribution-independent inputs for scalar predicate probability intervals."""

from collections.abc import Mapping

from .base import STL_Formula, validate_interval_trace, validate_scale


class Predicate(STL_Formula):
    """Select named [batch, time, 2] bounds supplied by a probability provider.

    Providers own the mathematical justification of their bounds. This leaf
    checks numeric validity and returns the existing tensor, preserving autograd.
    Exact probabilities are represented by equal lower and upper endpoints.
    """

    def __init__(self, name):
        super().__init__()
        if not isinstance(name, str) or not name:
            raise ValueError("predicate name must be a nonempty string")
        self.name = name

    def robustness_trace(self, inputs, scale=-1, **kwargs):
        validate_scale(scale)
        if not isinstance(inputs, Mapping):
            raise TypeError("Predicate expects a mapping of names to interval tensors")
        if self.name not in inputs:
            raise KeyError(f"missing probability intervals for predicate {self.name!r}")
        return validate_interval_trace(inputs[self.name], self.name)

    def smoothing_error(self, scale):
        validate_scale(scale)
        return 0.0

    def __str__(self):
        return self.name
