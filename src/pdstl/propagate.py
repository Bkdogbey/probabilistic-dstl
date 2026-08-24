"""Bounded-time evaluation of pdSTL formulas.

Traverses a formula over discrete time, obtains atomic probability bounds from
a :class:`ProbabilitySource`, and delegates composition to the operators.
Structural event identity tracks exact repetitions and complements.
Event keys also share cached values between structurally identical formulas;
no broader Boolean simplification or dependence model is attempted.
The public result has shape ``[B, T_valid, 2]``.
"""

from __future__ import annotations

from collections.abc import Hashable

import torch

from .base import ProbabilitySource, validate_bounds
from .operators import (
    And,
    Negation,
    Or,
    Predicate,
    STLFormula,
    TemporalOperator,
)

__all__ = ["EvaluationContext", "evaluate"]


def _canonical_pair(key1: Hashable, key2: Hashable) -> tuple:
    """Return a deterministic ordering for a commutative event pair.

    Ordering by ``repr`` supports the mixed element types used in event keys.
    """
    return tuple(sorted((key1, key2), key=repr))


def _is_complement(key1: Hashable, key2: Hashable) -> bool:
    """Return True if one event key is the complement of the other.

    NNF restricts negation keys to wrapped atoms, so this check is structural.
    """
    return key1 == ("not", key2) or key2 == ("not", key1)


class EvaluationContext:
    """Evaluation state for a single ``evaluate`` call.

    Owns the source and per-evaluation caches for atoms, formula objects, and
    structural events. This class is internal to the propagation module.
    """

    def __init__(self, source: ProbabilitySource) -> None:
        self.source = source
        # (predicate.uid, time) -> [B, 2]
        self._atoms: dict[tuple[int, int], torch.Tensor] = {}
        # Preserve exact formula-object identity across repeated traversal.
        self._formula_cache: dict[
            tuple[STLFormula, int], tuple[torch.Tensor, Hashable]
        ] = {}
        # Preserve structural event identity across different formula objects.
        self._event_cache: dict[Hashable, torch.Tensor] = {}
        self._batch: int | None = None
        self.n_source_queries = 0

    def atomic_bounds(self, predicate: Predicate, time: int) -> torch.Tensor:
        """Return validated bounds for one atomic predicate-time event.

        Results have shape ``[B, 2]`` and each event queries the source at most
        once. Validation here applies the same boundary checks to every source.
        """
        cache_key = (predicate.uid, time)
        if cache_key in self._atoms:
            return self._atoms[cache_key]

        bounds = self.source.bounds(predicate, time)
        self.n_source_queries += 1
        validate_bounds(bounds, context=f"{predicate} at time {time}")

        batch = bounds.shape[0]
        if self._batch is None:
            self._batch = batch
        elif batch != self._batch:
            raise ValueError(
                f"inconsistent batch size from the probability source: "
                f"{predicate} at time {time} returned shape "
                f"{tuple(bounds.shape)} (batch {batch}), but an earlier query "
                f"in this evaluation returned batch {self._batch}; a source "
                f"must use one batch size throughout an evaluation"
            )

        self._atoms[cache_key] = bounds
        return bounds

    def evaluate(self, formula: STLFormula, time: int) -> torch.Tensor:
        """Return probability bounds for a formula at one discrete time.

        The result has shape ``[B, 2]`` with lower and upper bounds last.
        """
        value, _ = self._evaluate_with_key(formula, time)
        return value

    def _evaluate_with_key(
        self, formula: STLFormula, time: int
    ) -> tuple[torch.Tensor, Hashable]:
        """Evaluate one formula-time node and return its bounds and event key."""
        memo_key = (formula, time)
        if memo_key in self._formula_cache:
            return self._formula_cache[memo_key]

        result = self._dispatch(formula, time)
        self._formula_cache[memo_key] = result
        self._event_cache.setdefault(result[1], result[0])
        return result

    def _dispatch(
        self, formula: STLFormula, time: int
    ) -> tuple[torch.Tensor, Hashable]:
        """Evaluate one formula node according to its type."""
        if isinstance(formula, Predicate):
            return self.atomic_bounds(formula, time), ("atom", formula.uid, time)

        if isinstance(formula, Negation):
            # Negated atoms derive from the positive event; the source is not
            # queried independently for ~A.
            child_value, child_key = self._evaluate_with_key(
                formula.subformula, time
            )
            return formula.combine(child_value), ("not", child_key)

        if isinstance(formula, (And, Or)):
            return self._eval_binary(formula, time)

        if isinstance(formula, TemporalOperator):
            return self._eval_temporal(formula, time)

        raise TypeError(f"unsupported formula type: {type(formula).__name__}")

    def _eval_binary(
        self, formula: And | Or, time: int
    ) -> tuple[torch.Tensor, Hashable]:
        """Evaluate a Boolean binary operator at one time."""
        left_value, left_key = self._evaluate_with_key(formula.left, time)
        right_value, right_key = self._evaluate_with_key(formula.right, time)

        is_and = isinstance(formula, And)
        key = (formula.tag[0], _canonical_pair(left_key, right_key))

        # Exact repetition: A ∩ A = A and A ∪ A = A.
        if left_key == right_key:
            return left_value, key

        # Exact complements: A ∩ Aᶜ = ∅ and A ∪ Aᶜ = Ω.
        if _is_complement(left_key, right_key):
            constant = 0.0 if is_and else 1.0
            return torch.full_like(left_value, constant), key

        cached = self._event_cache.get(key)
        if cached is not None:
            return cached, key

        return formula.combine(left_value, right_value), key

    def _eval_temporal(
        self, formula: TemporalOperator, time: int
    ) -> tuple[torch.Tensor, Hashable]:
        """Evaluate a bounded temporal operator at one time."""
        a, b = formula.a, formula.b
        values: list[torch.Tensor] = []
        child_keys: list[Hashable] = []

        for offset in range(a, b + 1):
            value, child_key = self._evaluate_with_key(
                formula.subformula, time + offset
            )
            values.append(value)
            child_keys.append(child_key)

        key = (formula.tag[0], a, b, tuple(child_keys))

        cached = self._event_cache.get(key)
        if cached is not None:
            return cached, key

        stacked = torch.stack(values, dim=-2)  # [B, n, 2]
        return formula.combine(stacked), key


def evaluate(formula: STLFormula, source: ProbabilitySource) -> torch.Tensor:
    """Evaluate ``formula`` at every valid discrete time.

    Parameters
    ----------
    formula : STLFormula
        A bounded-time pdSTL formula.
    source : ProbabilitySource
        Supplies the atomic probability bounds.

    Returns
    -------
    torch.Tensor
        Shape ``[B, T_valid, 2]`` where
        ``T_valid = source.horizon - formula.horizon() + 1``, and
        the last dimension stores lower and upper probability bounds.

        ``trace[:, 0, :]`` is the enclosure at the initial time.

    Notes
    -----
    Times requiring data past ``source.horizon`` are omitted rather than padded.

    Raises
    ------
    ValueError
        If the source horizon is too short for the formula's lookahead.
    """
    if not isinstance(formula, STLFormula):
        raise TypeError(f"formula must be an STLFormula, got {type(formula).__name__}")
    if not isinstance(source, ProbabilitySource):
        raise TypeError(
            f"source must be a ProbabilitySource, got {type(source).__name__}"
        )

    required = formula.horizon()
    n_valid = source.horizon - required + 1
    if n_valid <= 0:
        raise ValueError(
            f"source horizon {source.horizon} is too short for {formula}, which "
            f"needs a lookahead of {required} steps; the source must cover at "
            f"least times 0 ... {required}"
        )

    context = EvaluationContext(source)
    per_time = [context.evaluate(formula, time) for time in range(n_valid)]
    return torch.stack(per_time, dim=-2)  # [B, T_valid, 2]
