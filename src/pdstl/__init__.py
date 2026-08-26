"""Probabilistic discrete-time Signal Temporal Logic (pdSTL).

The package provides atomic probability sources, bounded STL formula classes,
and hard probability-bound evaluation. Three backends compute the same exact
hard semantics: the reference interpreter (:func:`evaluate`), the compiled
fold graph (:func:`compile_formula`), and the formula-structured recurrent
evaluator (:func:`compile_recurrent_formula`).

Example
-------
>>> mu = Predicate(name="mu")
>>> source = TableProbabilitySource({(mu, 0): (0.9, 0.9), (mu, 1): (0.9, 0.9)})
>>> trace = evaluate(Always(mu, interval=[0, 1]), source)
>>> trace[0, 0].tolist()
[0.7999999523162842, 0.8999999761581421]
"""

from .base import ProbabilitySource, TableProbabilitySource, validate_bounds
from .graph import CompiledFormula, compile_formula
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
from .recurrent import RecurrentFormula, compile_recurrent_formula

__all__ = [
    "Always",
    "And",
    "CompiledFormula",
    "Eventually",
    "Negation",
    "Or",
    "Predicate",
    "ProbabilitySource",
    "RecurrentFormula",
    "STLFormula",
    "TableProbabilitySource",
    "TemporalOperator",
    "Until",
    "compile_formula",
    "compile_recurrent_formula",
    "evaluate",
    "validate_bounds",
]
