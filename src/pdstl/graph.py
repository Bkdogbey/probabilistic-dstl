"""Compiled hard-semantics execution backend for pdSTL.

This module is an execution-layer addition only: it does not redefine the
hard probability semantics in :mod:`operators` and :mod:`propagate`, which
remain the mathematical source of truth. Instead, it *compiles* a formula
once, for a fixed source horizon, into a small DAG of tensor operations that
can then be executed repeatedly against materialized atomic probability
traces, producing the complete ``[B, T_valid, 2]`` trace in one call rather
than one discrete time at a time.

Exactness contract
-------------------
For every supported formula and every valid atomic input, the trace returned
by :class:`CompiledFormula` is numerically identical to
``propagate.evaluate(formula, source)`` -- not merely a sound enclosure of
it. "Exact" here means exact with respect to the *hard interval semantics*;
it does not mean the returned interval equals the unknown true satisfaction
probability. The interval can remain conservative because dependence across
time and across sub-events is unspecified -- no independence assumption is
introduced anywhere in this module, and :mod:`propagate` remains the trusted
oracle this backend is checked against.

Design summary
---------------
Every ``(ast node, anchor)`` pair -- where ``anchor`` is an integer offset
relative to the *top-level* formula's own output time -- compiles to at most
one node in a shared DAG, exactly mirroring how
``propagate.EvaluationContext`` caches by ``(formula, time)`` and by
structural event key. Because every leaf key has the form
``("atom", uid, k + offset)`` for a formula-time offset fixed by operator
syntax, comparing two operands for exact duplication or exact complementation
does not depend on which absolute ``k`` the top-level formula happens to be
evaluated at -- only on their *offsets* relative to one another. This is what
licenses compiling the identity-aware reduction once and reusing it,
unchanged, for every output time in the vectorized trace.

Only atomic predicate nodes ever slice into raw input data (at
``[:, anchor : anchor + T_valid, :]`` of the materialized ``[B, N+1, 2]``
atom trace); every compound node combines already ``[B, T_valid, 2]``-shaped
operands directly, using the exact same
:func:`operators.frechet_intersection` / :func:`operators.frechet_union`
primitives, :func:`propagate._canonical_keys`, and
:func:`propagate._is_complement` helpers the reference interpreter uses, so
the reduction logic cannot silently drift from it.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Hashable, Mapping, Sequence

import torch

from .base import ProbabilitySource, validate_bounds
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
    frechet_intersection,
    frechet_union,
)
from .propagate import EvaluationContext, _canonical_keys, _is_complement

__all__ = ["CompiledFormula", "compile_formula", "materialize_atom_traces"]

# One compiled operand: where to read its value, and its structural event key.
_Entry = tuple["_Ref", Hashable]


class _NodeRef:
    """Read a node's tensor starting at a fixed offset."""

    __slots__ = ("node", "offset")

    def __init__(self, node: _Node, offset: int) -> None:
        self.node = node
        self.offset = offset


class _ConstantRef:
    """A TOP/BOTTOM outcome from exact-complement collapse: constant everywhere."""

    __slots__ = ("value",)

    def __init__(self, value: float) -> None:
        self.value = value


_Ref = _NodeRef | _ConstantRef


class _AtomNode:
    """Holds the raw materialized trace for one predicate, shape [B, N+1, 2].

    Shared by every ``(predicate, anchor)`` occurrence of this predicate --
    only the reading offset differs, not the underlying data.
    """

    __slots__ = ("tensor", "uid")

    def __init__(self, uid: int) -> None:
        self.uid = uid
        self.tensor: torch.Tensor | None = None

    def compute(self, resolve: Callable[[_Ref], torch.Tensor], atom_traces: Mapping[int, torch.Tensor]) -> None:
        self.tensor = atom_traces[self.uid]


class _NegationNode:
    """Pointwise complement of a child tensor; no offset shift."""

    __slots__ = ("ast_node", "child", "tensor")

    def __init__(self, child: _Ref, ast_node: Negation) -> None:
        self.child = child
        self.ast_node = ast_node
        self.tensor: torch.Tensor | None = None

    def compute(self, resolve: Callable[[_Ref], torch.Tensor], atom_traces: Mapping[int, torch.Tensor]) -> None:
        self.tensor = self.ast_node.combine(resolve(self.child))


class _FoldNode:
    """The one generic n-ary reduction node: a Frechet intersection or union.

    Used for And/Or's two-operand fold, each Always/Eventually window fold,
    and every Until candidate/union/prefix fold. Operands are already
    deduplicated and pairwise non-complementary by construction (see
    :meth:`_Compiler._combine`).
    """

    __slots__ = ("intersection", "operands", "tensor")

    def __init__(self, intersection: bool, operands: Sequence[_Ref]) -> None:
        self.intersection = intersection
        self.operands = list(operands)
        self.tensor: torch.Tensor | None = None

    def compute(self, resolve: Callable[[_Ref], torch.Tensor], atom_traces: Mapping[int, torch.Tensor]) -> None:
        stacked = torch.stack([resolve(operand) for operand in self.operands], dim=-2)
        combine = frechet_intersection if self.intersection else frechet_union
        self.tensor = combine(stacked)


class _UntilTightenNode:
    """Caps the Until union's upper bound at the common-prefix upper bound.

    The lower bound is untouched; see ``Until``'s common-prefix tightening in
    :mod:`propagate` for the justification (E_U is a subset of the common
    prefix P_a for a > 0, so this is subset containment, not a new
    dependence assumption).
    """

    __slots__ = ("prefix", "tensor", "union")

    def __init__(self, union: _Ref, prefix: _Ref) -> None:
        self.union = union
        self.prefix = prefix
        self.tensor: torch.Tensor | None = None

    def compute(self, resolve: Callable[[_Ref], torch.Tensor], atom_traces: Mapping[int, torch.Tensor]) -> None:
        union_value = resolve(self.union)
        prefix_value = resolve(self.prefix)
        lower = union_value[..., 0]
        upper = torch.minimum(union_value[..., 1], prefix_value[..., 1])
        self.tensor = torch.stack([lower, upper], dim=-1)


_Node = _AtomNode | _NegationNode | _FoldNode | _UntilTightenNode


def _validate_atom_traces(
    atom_traces: Mapping[int, torch.Tensor],
    *,
    referenced_uids: frozenset[int] | set[int],
    predicate_names: Mapping[int, str],
    horizon: int,
    context: str,
) -> tuple[int, torch.dtype, torch.device]:
    """Check a materialized ``{uid: [B, horizon+1, 2]}`` mapping and report ``B``/dtype/device.

    Shared by every backend that consumes the tensor-input contract -- the
    compiled graph here and the recurrent evaluator in :mod:`pdstl.recurrent`
    -- so the two cannot drift in what they accept. ``context`` only labels
    error messages.
    """
    missing = set(referenced_uids) - set(atom_traces)
    if missing:
        names = ", ".join(predicate_names.get(uid, f"uid={uid}") for uid in sorted(missing))
        raise KeyError(f"missing atom traces for: {names}")

    expected_time = horizon + 1
    batch: int | None = None
    dtype: torch.dtype | None = None
    device: torch.device | None = None
    for uid in sorted(referenced_uids):
        tensor = atom_traces[uid]
        name = predicate_names.get(uid, f"uid={uid}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"atom trace for {name} must be a torch.Tensor, got {type(tensor).__name__}"
            )
        if tensor.ndim != 3 or tensor.shape[1] != expected_time or tensor.shape[2] != 2:
            raise ValueError(
                f"atom trace for {name} must have shape [B, {expected_time}, 2], "
                f"got {tuple(tensor.shape)}"
            )
        validate_bounds(tensor.reshape(-1, 2), context=f"{name} ({context})")

        if batch is None:
            batch, dtype, device = tensor.shape[0], tensor.dtype, tensor.device
        else:
            if tensor.shape[0] != batch:
                raise ValueError(
                    f"inconsistent batch size: {name} has batch {tensor.shape[0]}, "
                    f"expected {batch}"
                )
            if tensor.dtype != dtype:
                raise ValueError(f"inconsistent dtype: {name} has {tensor.dtype}, expected {dtype}")
            if tensor.device != device:
                raise ValueError(f"inconsistent device: {name} has {tensor.device}, expected {device}")

    assert batch is not None and dtype is not None and device is not None
    return batch, dtype, device


def _dedupe_entries(entries: Sequence[_Entry]) -> list[_Entry]:
    """Drop exact key repeats, preserving first-seen order."""
    seen: set[Hashable] = set()
    unique: list[_Entry] = []
    for ref, key in entries:
        if key not in seen:
            seen.add(key)
            unique.append((ref, key))
    return unique


class _Compiler:
    """Builds the compiled DAG for one formula at a fixed top-level horizon.

    Every ``(id(ast_node), anchor)`` pair is compiled at most once, mirroring
    ``EvaluationContext._formula_cache``'s ``(formula, time)`` keying with
    ``anchor`` standing in for ``time`` (valid by translation invariance: the
    reduction decisions below depend only on relative offsets, never on which
    absolute output time the top-level formula is eventually evaluated at).
    A second cache, keyed by structural event key, mirrors
    ``EvaluationContext._event_cache`` and shares values across
    structurally-identical but object-distinct subformulas.
    """

    def __init__(self) -> None:
        self.nodes: list[_Node] = []
        self._atom_nodes: dict[int, _AtomNode] = {}
        self._formula_memo: dict[tuple[int, int], _Entry] = {}
        self._event_cache: dict[Hashable, _Ref] = {}
        self.referenced_uids: set[int] = set()
        self.predicate_names: dict[int, str] = {}

    def compile_at(self, ast_node: STLFormula, anchor: int) -> _Entry:
        """Compile ``ast_node`` as read at ``anchor`` steps past the top-level time."""
        memo_key = (id(ast_node), anchor)
        cached = self._formula_memo.get(memo_key)
        if cached is not None:
            return cached

        if isinstance(ast_node, Predicate):
            entry = self._compile_atom(ast_node, anchor)
        elif isinstance(ast_node, Negation):
            entry = self._compile_negation(ast_node, anchor)
        elif isinstance(ast_node, (And, Or)):
            entry = self._compile_binary(ast_node, anchor)
        elif isinstance(ast_node, Until):
            entry = self._compile_until(ast_node, anchor)
        elif isinstance(ast_node, TemporalOperator):
            entry = self._compile_temporal(ast_node, anchor)
        else:
            raise TypeError(f"unsupported formula type: {type(ast_node).__name__}")

        self._formula_memo[memo_key] = entry
        self._event_cache.setdefault(entry[1], entry[0])
        return entry

    def _compile_atom(self, ast_node: Predicate, anchor: int) -> _Entry:
        self.referenced_uids.add(ast_node.uid)
        self.predicate_names[ast_node.uid] = str(ast_node)
        atom_node = self._atom_nodes.get(ast_node.uid)
        if atom_node is None:
            atom_node = _AtomNode(ast_node.uid)
            self._atom_nodes[ast_node.uid] = atom_node
            self.nodes.append(atom_node)
        return _NodeRef(atom_node, anchor), ("atom", ast_node.uid, anchor)

    def _compile_negation(self, ast_node: Negation, anchor: int) -> _Entry:
        child_ref, child_key = self.compile_at(ast_node.subformula, anchor)
        node = _NegationNode(child_ref, ast_node)
        self.nodes.append(node)
        return _NodeRef(node, 0), ("not", child_key)

    def _compile_binary(self, ast_node: And | Or, anchor: int) -> _Entry:
        left = self.compile_at(ast_node.left, anchor)
        right = self.compile_at(ast_node.right, anchor)
        tag = ast_node.tag[0]
        return self._combine(
            [left, right],
            intersection=isinstance(ast_node, And),
            make_key=lambda keys: (tag, _canonical_keys(keys)),
        )

    def _compile_temporal(self, ast_node: Always | Eventually, anchor: int) -> _Entry:
        a, b = ast_node.a, ast_node.b
        entries = [self.compile_at(ast_node.subformula, anchor + offset) for offset in range(a, b + 1)]
        tag = ast_node.tag[0]
        return self._combine(
            entries,
            intersection=isinstance(ast_node, Always),
            make_key=lambda keys: (tag, a, b, keys),
        )

    def _compile_until(self, ast_node: Until, anchor: int) -> _Entry:
        a, b = ast_node.a, ast_node.b
        prefix: list[_Entry] = []  # E_{phi1, anchor+r}, r = 0, 1, ...
        candidates: list[_Entry] = []

        def conjunction_key(keys: tuple[Hashable, ...]) -> Hashable:
            return ("and", _canonical_keys(keys))

        for j in range(b + 1):
            while len(prefix) < j:
                r = len(prefix)
                prefix.append(self.compile_at(ast_node.left, anchor + r))
            if j < a:
                continue
            right_entry = self.compile_at(ast_node.right, anchor + j)
            candidates.append(
                self._combine([right_entry, *prefix[:j]], intersection=True, make_key=conjunction_key)
            )

        ref, key = self._combine(
            candidates,
            intersection=False,
            make_key=lambda keys: ("until", a, b, keys),
        )

        if a > 0:
            prefix_ref, _ = self._combine(prefix[:a], intersection=True, make_key=conjunction_key)
            node = _UntilTightenNode(union=ref, prefix=prefix_ref)
            self.nodes.append(node)
            ref = _NodeRef(node, 0)

        self._event_cache[key] = ref
        return ref, key

    def _combine(
        self,
        entries: Sequence[_Entry],
        *,
        intersection: bool,
        make_key: Callable[[tuple[Hashable, ...]], Hashable],
    ) -> _Entry:
        """Direct clone of ``EvaluationContext._combine_events`` over ``_Ref``s.

        Duplicate operands collapse first, then exact complements, then a
        single surviving operand is returned under its own ref/key (no new
        node -- this is what makes identities compositional through nesting),
        and only then does a generic Frechet fold get built.
        """
        unique = _dedupe_entries(entries)

        if len(unique) == 1:
            return unique[0]

        keys = tuple(key for _, key in unique)
        key = make_key(keys)

        if any(_is_complement(k1, k2) for k1, k2 in itertools.combinations(keys, 2)):
            constant = 0.0 if intersection else 1.0
            return _ConstantRef(constant), key

        cached = self._event_cache.get(key)
        if cached is not None:
            return cached, key

        node = _FoldNode(intersection, [ref for ref, _ in unique])
        self.nodes.append(node)
        ref = _NodeRef(node, 0)
        self._event_cache[key] = ref
        return ref, key


class CompiledFormula:
    """A compiled, reusable hard-probability computation graph for one formula.

    Construct via :func:`compile_formula`. Calling the instance executes the
    precompiled DAG against materialized atomic traces and returns the
    complete ``[B, T_valid, 2]`` enclosure trace -- numerically identical to
    ``propagate.evaluate(formula, source)`` for a source whose
    ``bounds(predicate, t)`` equals ``atom_traces[predicate.uid][:, t, :]``
    for every ``t`` in ``0 .. horizon``.
    """

    def __init__(
        self,
        formula: STLFormula,
        horizon: int,
        valid_length: int,
        root_ref: _Ref,
        nodes: list[_Node],
        referenced_uids: set[int],
        predicate_names: dict[int, str],
    ) -> None:
        self.formula = formula
        self.horizon = horizon
        self.valid_length = valid_length
        self._root_ref = root_ref
        self._nodes = nodes
        self._referenced_uids = frozenset(referenced_uids)
        self._predicate_names = dict(predicate_names)

    @property
    def n_nodes(self) -> int:
        """Number of distinct compiled DAG nodes (excludes aliased/constant results)."""
        return len(self._nodes)

    def __call__(self, atom_traces: Mapping[int, torch.Tensor]) -> torch.Tensor:
        """Execute the compiled graph. Returns shape ``[B, T_valid, 2]``."""
        batch, dtype, device = self._validate(atom_traces)

        def resolve(ref: _Ref) -> torch.Tensor:
            if isinstance(ref, _ConstantRef):
                return torch.full((batch, self.valid_length, 2), ref.value, dtype=dtype, device=device)
            return ref.node.tensor[:, ref.offset : ref.offset + self.valid_length, :]

        for node in self._nodes:
            node.compute(resolve, atom_traces)

        return resolve(self._root_ref)

    def _validate(self, atom_traces: Mapping[int, torch.Tensor]) -> tuple[int, torch.dtype, torch.device]:
        return _validate_atom_traces(
            atom_traces,
            referenced_uids=self._referenced_uids,
            predicate_names=self._predicate_names,
            horizon=self.horizon,
            context="compiled graph",
        )


def compile_formula(formula: STLFormula, *, horizon: int) -> CompiledFormula:
    """Compile ``formula`` once for a fixed source horizon.

    Parameters
    ----------
    formula : STLFormula
        A bounded-time pdSTL formula.
    horizon : int
        The largest discrete time index atomic traces will cover; valid
        times are ``0 .. horizon`` inclusive, matching
        ``ProbabilitySource.horizon``.

    Returns
    -------
    CompiledFormula
        A reusable callable; see :meth:`CompiledFormula.__call__`.

    Raises
    ------
    ValueError
        If ``horizon`` is too short for ``formula``'s lookahead.
    """
    if not isinstance(formula, STLFormula):
        raise TypeError(f"formula must be an STLFormula, got {type(formula).__name__}")
    if not isinstance(horizon, int) or isinstance(horizon, bool):
        raise TypeError(f"horizon must be an int, got {type(horizon).__name__}")
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")

    required = formula.horizon()
    valid_length = horizon - required + 1
    if valid_length <= 0:
        raise ValueError(
            f"horizon {horizon} is too short for {formula}, which needs a lookahead of "
            f"{required} steps; the horizon must cover at least times 0 ... {required}"
        )

    compiler = _Compiler()
    root_ref, _ = compiler.compile_at(formula, 0)
    return CompiledFormula(
        formula,
        horizon,
        valid_length,
        root_ref,
        compiler.nodes,
        compiler.referenced_uids,
        compiler.predicate_names,
    )


def _collect_predicates(formula: STLFormula, out: dict[int, Predicate] | None = None) -> dict[int, Predicate]:
    """Walk ``formula`` and return its distinct predicates, keyed by uid."""
    if out is None:
        out = {}

    if isinstance(formula, Predicate):
        out[formula.uid] = formula
    elif isinstance(formula, Negation):
        _collect_predicates(formula.subformula, out)
    elif isinstance(formula, (And, Or, Until)):
        _collect_predicates(formula.left, out)
        _collect_predicates(formula.right, out)
    elif isinstance(formula, TemporalOperator):
        _collect_predicates(formula.subformula, out)
    else:
        raise TypeError(f"unsupported formula type: {type(formula).__name__}")

    return out


def materialize_atom_traces(
    formula: STLFormula, source: ProbabilitySource, horizon: int
) -> dict[int, torch.Tensor]:
    """Build one ``[B, horizon+1, 2]`` tensor per predicate reachable in ``formula``.

    A small adapter from the existing :class:`~pdstl.base.ProbabilitySource`
    interface to the tensor-input contract :class:`CompiledFormula` expects;
    it does not change ``ProbabilitySource`` itself. Reuses
    ``EvaluationContext.atomic_bounds`` for the same validated, batch-checked,
    query-once atomic lookup the reference interpreter uses.
    """
    predicates = _collect_predicates(formula)
    context = EvaluationContext(source)
    traces: dict[int, torch.Tensor] = {}
    for predicate in predicates.values():
        rows = [context.atomic_bounds(predicate, t) for t in range(horizon + 1)]
        traces[predicate.uid] = torch.stack(rows, dim=1)  # [B, horizon+1, 2]
    return traces
