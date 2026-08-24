"""Probabilistic discrete-time Signal Temporal Logic (pdSTL).

The package is deliberately small and splits three concerns:

``base.py``
    *What* probabilistic information a user supplies: the
    :class:`ProbabilitySource` contract and centralised bound validation.

``operators.py``
    *What* each STL operation means mathematically: the formula classes and the
    hard Frechet combination equations.

``propagate.py``
    *How* a formula is traversed, cached and evaluated over bounded discrete
    time.

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
    "evaluate",
    "validate_bounds",
]
