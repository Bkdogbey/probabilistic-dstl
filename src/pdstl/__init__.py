"""Probabilistic discrete-time Signal Temporal Logic (pdSTL).

The package provides atomic probability sources, bounded STL formula classes,
and hard probability-bound evaluation.

Example
-------
>>> mu = Predicate(name="mu")
>>> source = TableProbabilitySource({(mu, 0): (0.9, 0.9), (mu, 1): (0.9, 0.9)})
>>> trace = evaluate(Always(mu, interval=[0, 1]), source)
>>> trace[0, 0].tolist()
[0.7999999523162842, 0.8999999761581421]
"""

from .base import ProbabilitySource, TableProbabilitySource, validate_bounds
from .operators import (
    Always,
    And,
    Eventually,
    Negation,
    Or,
    Predicate,
    STLFormula,
    TemporalOperator,
    Until,
)
from .propagate import evaluate

__all__ = [
    "Always",
    "And",
    "Eventually",
    "Negation",
    "Or",
    "Predicate",
    "ProbabilitySource",
    "STLFormula",
    "TableProbabilitySource",
    "TemporalOperator",
    "Until",
    "evaluate",
    "validate_bounds",
]
