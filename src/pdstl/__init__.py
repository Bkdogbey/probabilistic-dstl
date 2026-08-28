"""Probabilistic discrete-time Signal Temporal Logic (pdSTL).

The package provides atomic probability sources, pointwise Boolean formula
classes, and bounded temporal operators. A Formula is a torch.nn.Module:
calling it on a source returns Tensor[B, T, 2] of [lower, upper] probability
bounds. Pass ``smooth=True`` for a differentiable optimization surrogate;
the default ``smooth=False`` gives the hard, certifiable Frechet bounds.

Example
-------
>>> import torch
>>> a = Predicate("safe")
>>> b = Predicate("goal")
>>> source = OfflineSource(
...     {a: torch.tensor([[0.6, 0.9]]).unsqueeze(0), b: torch.tensor([[0.7, 0.95]]).unsqueeze(0)}
... )
>>> (a & b)(source)[0, 0].tolist()
[0.30000001192092896, 0.9]

A bounded temporal operator over [a, b] consumes b + 1 steps per output, so a
length-T trace yields max(T - b, 0) outputs:

>>> trace = torch.tensor([[0.9, 0.9], [0.8, 0.95], [0.85, 0.9]]).unsqueeze(0)
>>> Always(a, (0, 1))(OfflineSource({a: trace})).shape
torch.Size([1, 2, 2])
"""

from .base import (
    OfflineSource,
    OnlineSource,
    ProbabilitySource,
    validate_bounds,
)
from .operators import (
    Always,
    And,
    Eventually,
    Formula,
    Not,
    Or,
    Predicate,
    TemporalOperator,
    Until,
    UntilState,
)

__all__ = [
    "Always",
    "And",
    "Eventually",
    "Formula",
    "Not",
    "OfflineSource",
    "OnlineSource",
    "Or",
    "Predicate",
    "ProbabilitySource",
    "TemporalOperator",
    "Until",
    "UntilState",
    "validate_bounds",
]
