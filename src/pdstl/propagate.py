"""Bounded-time evaluation of pdSTL formulas.

Traverses a formula over discrete time, obtains atomic probability bounds from
a :class:`ProbabilitySource`, and composes them with the shared Frechet rules.
Structural event identity tracks exact repetitions and complements: a reduction
that leaves one event returns *that event's* key, so the identities compose
through nesting instead of holding only one level deep.
Event keys also share cached values between structurally identical formulas;
no broader Boolean simplification or dependence model is attempted.
The public result has shape ``[B, T_valid, 2]``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Hashable, Sequence

import torch

from .base import ProbabilitySource, validate_bounds
from .operators import (
    Always,
    And,
    Negation,
    Or,
    Predicate,
    STLFormula,
    TemporalOperator,
    Until,
    frechet_intersection,
    frechet_union,
)

__all__ = ["EvaluationContext", "evaluate"]

# One evaluated node: its probability bounds and its structural event key.
Entry = tuple[torch.Tensor, Hashable]


def _canonical_keys(keys: Sequence[Hashable]) -> tuple:
    """Return a deterministic ordering for a commutative set of event keys.

    Ordering by ``repr`` supports the mixed element types used in event keys.
    """
    return tuple(sorted(keys, key=repr))


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

        if isinstance(formula, Until):
            return self._eval_until(formula, time)

        if isinstance(formula, TemporalOperator):
            return self._eval_temporal(formula, time)

        raise TypeError(f"unsupported formula type: {type(formula).__name__}")

    def _combine_events(
        self,
        entries: Sequence[Entry],
        *,
        intersection: bool,
        make_key: Callable[[tuple[Hashable, ...]], Hashable],
    ) -> Entry:
        """Reduce operand events under set intersection or union.

        The identity rules are applied to the event *keys* before any Frechet
        combination, so exact structure is never discarded:

        1. duplicate events collapse (``A ∩ A = A``, ``A ∪ A = A``);
        2. a single surviving event is returned *under its own key*, which is
           what makes the identities compositional -- ``(A ∧ A) ∧ A`` reduces
           to ``A`` rather than to a compound event that merely happens to
           carry ``A``'s bounds. Singleton temporal windows land here too, so
           ``G[a,a] phi`` is the child event at ``k + a``;
        3. exact complements are recognised: ``A ∩ Aᶜ = ∅``, ``A ∪ Aᶜ = Ω``.

        Anything else falls back to the dependence-agnostic Frechet rule, under
        the compound key supplied by ``make_key``.
        """
        unique: list[Entry] = []
        seen: set[Hashable] = set()
        for value, key in entries:
            if key not in seen:
                seen.add(key)
                unique.append((value, key))

        if len(unique) == 1:
            return unique[0]

        keys = tuple(key for _, key in unique)
        key = make_key(keys)

        if any(_is_complement(k1, k2) for k1, k2 in itertools.combinations(keys, 2)):
            constant = 0.0 if intersection else 1.0
            return torch.full_like(unique[0][0], constant), key

        cached = self._event_cache.get(key)
        if cached is not None:
            return cached, key

        stacked = torch.stack([value for value, _ in unique], dim=-2)  # [B, n, 2]
        combine = frechet_intersection if intersection else frechet_union
        return combine(stacked), key

    def _eval_binary(
        self, formula: And | Or, time: int
    ) -> tuple[torch.Tensor, Hashable]:
        """Evaluate a Boolean binary operator at one time."""
        entries = [
            self._evaluate_with_key(operand, time)
            for operand in (formula.left, formula.right)
        ]
        tag = formula.tag[0]
        return self._combine_events(
            entries,
            intersection=isinstance(formula, And),
            make_key=lambda keys: (tag, _canonical_keys(keys)),
        )

    def _eval_temporal(
        self, formula: TemporalOperator, time: int
    ) -> tuple[torch.Tensor, Hashable]:
        """Evaluate a bounded temporal operator at one time.

        ``G`` intersects its window and ``F`` unions it. A singleton window
        reduces to the child event itself, by the identity rules in
        :meth:`_combine_events`.
        """
        a, b = formula.a, formula.b
        entries = [
            self._evaluate_with_key(formula.subformula, time + offset)
            for offset in range(a, b + 1)
        ]
        tag = formula.tag[0]
        return self._combine_events(
            entries,
            intersection=isinstance(formula, Always),
            make_key=lambda keys: (tag, a, b, keys),
        )

    def _eval_until(self, formula: Until, time: int) -> tuple[torch.Tensor, Hashable]:
        """Evaluate bounded strong until at one time.

        Builds the candidates ``C_j = E_{phi2, k+j} ∩ (∩_{r<j} E_{phi1, k+r})``
        for ``j = a ... b`` and unions them. The ``phi1`` prefix is grown lazily
        from ``r = 0``, so the source is queried for ``phi1`` only at offsets
        ``0 ... b-1`` and for ``phi2`` only at ``a ... b`` -- exactly the window
        that :meth:`Until.horizon` accounts for.
        """
        a, b = formula.a, formula.b
        prefix: list[Entry] = []  # E_{phi1, k+r}, r = 0, 1, ...
        candidates: list[Entry] = []

        def conjunction_key(keys: tuple[Hashable, ...]) -> Hashable:
            return ("and", _canonical_keys(keys))

        for j in range(b + 1):
            while len(prefix) < j:
                prefix.append(
                    self._evaluate_with_key(formula.left, time + len(prefix))
                )
            if j < a:
                continue
            right_entry = self._evaluate_with_key(formula.right, time + j)
            # j = 0 leaves the prefix empty, so C_0 is E_{phi2, k} itself and
            # the reduction hands back the right child's own event key.
            candidates.append(
                self._combine_events(
                    [right_entry, *prefix[:j]],
                    intersection=True,
                    make_key=conjunction_key,
                )
            )

        value, key = self._combine_events(
            candidates,
            intersection=False,
            # Not the "or" namespace: the value below carries the common-prefix
            # tightening, so it must not be aliased with a plain disjunction of
            # the same candidate events.
            make_key=lambda keys: ("until", a, b, keys),
        )

        if a > 0:
            # Every candidate contains the common prefix P_a = ∩_{r<a} E_{phi1,k+r},
            # so E_U ⊆ P_a and the union's upper bound cannot exceed P(P_a).
            # The lower bound is untouched, and no dependence is assumed.
            prefix_value, _ = self._combine_events(
                prefix[:a], intersection=True, make_key=conjunction_key
            )
            upper = torch.minimum(value[..., 1], prefix_value[..., 1])
            value = torch.stack([value[..., 0], upper], dim=-1)

        return value, key


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
        ``T_valid = len(source) - formula.horizon()``, and
        the last dimension stores lower and upper probability bounds.

        ``trace[:, 0, :]`` is the enclosure at the initial time.

    Notes
    -----
    Times requiring data past the available time steps are omitted rather
    than padded.

    Raises
    ------
    ValueError
        If the source has too few time steps for the formula's lookahead.
    """
    if not isinstance(formula, STLFormula):
        raise TypeError(f"formula must be an STLFormula, got {type(formula).__name__}")
    if not isinstance(source, ProbabilitySource):
        raise TypeError(
            f"source must be a ProbabilitySource, got {type(source).__name__}"
        )

    required = formula.horizon()
    n_valid = len(source) - required
    if n_valid <= 0:
        raise ValueError(
            f"source has {len(source)} time steps, too short for {formula}, which "
            f"needs a lookahead of {required} steps; the source must cover at "
            f"least {required + 1} time steps"
        )

    context = EvaluationContext(source)
    per_time = [context.evaluate(formula, time) for time in range(n_valid)]
    return torch.stack(per_time, dim=-2)  # [B, T_valid, 2]
