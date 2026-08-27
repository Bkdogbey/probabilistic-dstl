"""Probabilistic discrete-time Signal Temporal Logic (pdSTL).

The package provides atomic probability sources and pointwise Boolean
formula classes. A Formula is a torch.nn.Module: calling it on a source
returns Tensor[B, T, 2] of [lower, upper] probability bounds.

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
"""

from .base import (
    OfflineSource,
    OnlineSource,
    ProbabilitySource,
    validate_bounds,
)
from .operators import And, Formula, Not, Or, Predicate

__all__ = [
    "And",
    "Formula",
    "Not",
    "OfflineSource",
    "OnlineSource",
    "Or",
    "Predicate",
    "ProbabilitySource",
    "validate_bounds",
]
