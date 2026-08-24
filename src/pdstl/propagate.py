"""Bounded-time evaluation of a pdSTL formula against a probability source.

This module owns *how* a formula is traversed, cached and indexed over discrete
time. It contains no probability equations: every combination step is delegated
to the operator's own ``combine`` method in ``operators.py``, so the semantics
are defined in exactly one place.

Pipeline::

    ProbabilitySource
        -> atomic bounds [l_{i,k}, u_{i,k}]
        -> propagation graph (this module)
        -> Boolean / temporal composition (operators.py)
        -> [P_lower(phi, k), P_upper(phi, k)]

Event identity
--------------
Every ``(subformula, time)`` node is given a canonical hashable **event key**
built bottom-up. Keys do two jobs:

1. they let structurally identical events share a cached value;
2. they make two Boolean relationships exactly decidable, so the recursive
   Frechet bound is not needlessly weakened.

The two identities handled are *repetition* (``A and A = A``, ``A or A = A``)
and *complement* (``A and ~A = empty``, ``A or ~A = universe``). The complement
case matters because generic Frechet cannot recover it from interval marginals:
with ``A = [0.4, 0.7]`` and therefore ``~A = [0.3, 0.6]``, blind Frechet reports
``A and ~A = [0, 0.6]`` -- an upper bound of 0.6 on an impossible event.

Nothing more is attempted: no distributivity, no absorption, no correlation
model, and no canonicalisation of associativity. The recursive result stays
**sound**, but it is not claimed to be globally sharp for arbitrary formulas.
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

__all__ = ["PropagationContext", "evaluate"]


def _canonical_pair(key1: Hashable, key2: Hashable) -> tuple:
    """Order two event keys deterministically.

    ``And`` and ``Or`` are symmetric in their arguments, so ``A and B`` and
    ``B and A`` denote the same event and must share a key. Sorting by ``repr``
    gives a total order that never raises on the mixed ``str``/``int``/``tuple``
    elements that make up a key, unlike direct tuple comparison.
    """
    return tuple(sorted((key1, key2), key=repr))


def _is_complement(key1: Hashable, key2: Hashable) -> bool:
    """True when one key is exactly the Boolean complement of the other.

    Negation is restricted to atomic predicates, so the only ``("not", k)`` keys
    that exist wrap an atom. That makes complementarity a purely structural
    check rather than any form of dependence modelling.
    """
    return key1 == ("not", key2) or key2 == ("not", key1)


class PropagationContext:
    """Evaluation state for a single ``evaluate`` call.

    Owns the probability source, the atomic event cache and the formula cache.
    This class is internal: it is importable for inspection and testing but is
    deliberately not part of the public package API.
    """

    def __init__(self, source: ProbabilitySource) -> None:
        self.source = source
        # (predicate.uid, time) -> [B, 2]
        self._atoms: dict[tuple[int, int], torch.Tensor] = {}
        # (formula, time) -> ([B, 2], key); short-circuits repeated traversal
        # of the same subformula object.
        self._formula_cache: dict[
            tuple[STLFormula, int], tuple[torch.Tensor, Hashable]
        ] = {}
        # key -> [B, 2]; shares values between structurally identical events
        # reached through different objects.
        self._event_cache: dict[Hashable, torch.Tensor] = {}
        self._batch: int | None = None
        self.n_source_queries = 0

    def atom(self, predicate: Predicate, time: int) -> torch.Tensor:
        """Bounds for ``E_{i,k}``, querying the source at most once per event.

        Returns shape ``[B, 2]``. Validation runs here rather than inside each
        source, so *every* source is checked at the core boundary regardless of
        what it does internally.
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
        """Probability enclosure of ``formula`` at discrete ``time``.

        Returns shape ``[B, 2]`` holding
        ``[P_lower(phi, time), P_upper(phi, time)]``.
        """
        value, _ = self._eval_keyed(formula, time)
        return value

    def _eval_keyed(
        self, formula: STLFormula, time: int
    ) -> tuple[torch.Tensor, Hashable]:
        """Evaluate one node, returning its value and its canonical event key."""
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
        """Traverse one node.

        Dispatch lives here rather than as a method on each formula class: a
        ``formula._evaluate(context, time)`` double dispatch would be shorter,
        but it would push child traversal and time indexing back into
        ``operators.py`` and blur the responsibility boundary.
        """
        if isinstance(formula, Predicate):
            return self.atom(formula, time), ("atom", formula.uid, time)

        if isinstance(formula, Negation):
            # Derived from the positive atom. The source is never asked for a
            # negated predicate.
            child_value, child_key = self._eval_keyed(formula.subformula, time)
            return formula.combine(child_value), ("not", child_key)

        if isinstance(formula, (And, Or)):
            return self._eval_binary(formula, time)

        if isinstance(formula, TemporalOperator):
            return self._eval_temporal(formula, time)

        raise TypeError(f"unsupported formula type: {type(formula).__name__}")

    def _eval_binary(
        self, formula: And | Or, time: int
    ) -> tuple[torch.Tensor, Hashable]:
        left_value, left_key = self._eval_keyed(formula.left, time)
        right_value, right_key = self._eval_keyed(formula.right, time)

        is_and = isinstance(formula, And)
        key = (formula.tag[0], _canonical_pair(left_key, right_key))

        # Identity 1: repetition. A and A = A, A or A = A. Applying generic
        # Frechet here would weaken the interval for no reason.
        if left_key == right_key:
            return left_value, key

        # Identity 2: complement. A and ~A is impossible, A or ~A is certain.
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
        a, b = formula.a, formula.b
        values: list[torch.Tensor] = []
        child_keys: list[Hashable] = []

        for offset in range(a, b + 1):
            value, child_key = self._eval_keyed(formula.subformula, time + offset)
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

    This is the single canonical evaluation API for the package.

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
        ``trace[..., 0]`` / ``trace[..., 1]`` are the lower and upper
        probability bounds.

        ``trace[:, 0, :]`` is the probability enclosure of the formula, i.e.
        the quantity the paper reports.

    Notes
    -----
    Times whose evaluation would need source data past ``source.horizon`` are
    **not produced at all**. The tail is never padded and no value is
    fabricated; the returned trace is exactly as long as the source supports.

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

    context = PropagationContext(source)
    per_time = [context.evaluate(formula, time) for time in range(n_valid)]
    return torch.stack(per_time, dim=-2)  # [B, T_valid, 2]
