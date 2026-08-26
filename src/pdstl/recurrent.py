"""Formula-structured recurrent pdSTL evaluator.

This is a third execution layer for the *same* hard probability-interval
semantics defined by :mod:`pdstl.operators` and :mod:`pdstl.propagate`. It
adds no new semantics: for every supported bounded NNF formula and every
valid atomic input, this module's output is numerically identical -- up to
floating-point re-association -- to both ``propagate.evaluate`` (the reference
interpreter) and ``graph.CompiledFormula`` (the compiled fold DAG).

What "recurrent" means here
---------------------------
The forward-looking temporal operators ``G_[a,b]``, ``F_[a,b]`` and
``U_[a,b]`` are evaluated by a **backward-time scan** over discrete time,
``k = T_valid-1, ..., 0``, so a future-time STL window becomes causal with
respect to the reversed scan. Each temporal operator owns one recurrent cell
carrying a *bounded* state whose width depends on the operator's window, never
on the trace length.

This is a **non-learned recurrent realization** of the exact hard probability
semantics:

* there are no trainable parameters anywhere in this module -- see
  :meth:`RecurrentFormula.parameters`, which returns an empty tuple;
* ``torch.nn`` is never imported, and no ``LSTM``/``GRU``/learned RNN is
  involved;
* the recurrence relation is fixed by the *formula*, not fitted to data.

The quantities propagated through the cells remain probability intervals
``[P_lower, P_upper]`` derived from the atomic probability provider, so
recurrent probability-bound propagation stays differentiable end to end and
downstream synthesis can optimize ``P_lower`` directly.

The recurrent primitive: a backward ladder
------------------------------------------
Every temporal operator reduces to one bounded backward recurrence over its
child trace ``c``. For intersection, the state holds, for each ladder depth
``j``, the two totals the Frechet rule needs::

    sums[j] = sum_{r<j} lower(c[p + r])
    mins[j] = min_{r<j} upper(c[p + r])

and the backward step at scan position ``p`` is a shift-and-absorb::

    sums <- concat([0, sums[:D] + lower(c[p])])
    mins <- concat([1, min(mins[:D], upper(c[p]))])

which is exact because ``P_j(p) = E_c[p] intersect P_{j-1}(p+1)``. The union
ladder is the dual (running sum of uppers, running max of lowers). The state
is ``O(W)`` wide and each backward step costs ``O(1)`` tensor operations.

* ``G_[a,b] phi`` at anchor ``k`` reads ladder entry ``W = b-a+1`` at scan
  position ``p = k + a``, then applies the exact probability intersection
  ``lower = max(0, sum_i l_i - (m-1))``, ``upper = min_i u_i``. It is *not*
  a temporal minimum on both bounds.
* ``F_[a,b] phi`` reads the union ladder: ``lower = max_i l_i``,
  ``upper = min(1, sum_i u_i)``.
* ``phi U_[a,b] psi`` runs the intersection ladder over ``phi`` and reads
  *every* candidate prefix ``P_j`` for ``j = a..b`` out of the one shared
  state, so prefixes are never recomputed per candidate. The common-prefix
  upper cap ``P_a`` is read straight off the same state.

Boolean ``And``/``Or`` and ``Negation`` are pointwise in time and correctly
need no scan; they are applied to whole traces.

Structural reduction happens first
----------------------------------
Compilation performs the identity/complement reduction *before* any Frechet
aggregation, using the same ``propagate._canonical_keys`` /
``propagate._is_complement`` primitives the reference interpreter uses, and
the Frechet rules are applied through the shared equations in
:mod:`pdstl.operators`. The post-reduction operand count ``m`` -- never the
syntactic one -- is what enters ``max(0, sum l_i - (m-1))``.

Because every event key bottoms out in ``("atom", uid, t)`` with an absolute
time, keys shift rigidly with the anchor and reduction decisions depend only
on *relative* offsets. Structure is therefore analyzed once, at anchor 0, and
reused unchanged at every output time -- the same translation-invariance
argument :mod:`pdstl.graph` documents. Unlike :mod:`pdstl.graph`, which emits
one DAG node per ``(ast node, anchor)`` pair and folds whole traces
statically, this module emits **one cell per operator** and unfolds it over
time through the recurrence above.

Output contract
---------------
``RecurrentFormula(atom_traces)`` consumes ``dict[predicate.uid,
Tensor[B, N+1, 2]]`` -- exactly ``CompiledFormula``'s input -- and returns
``Tensor[B, T_valid, 2]`` with ``T_valid = N - H(phi) + 1``. Times that would
require data past the horizon are omitted, never padded.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

import torch

from .graph import _validate_atom_traces
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
    _frechet_intersection_from_totals,
    _frechet_union_from_totals,
    frechet_intersection,
    frechet_union,
)
from .propagate import _canonical_keys, _is_complement

__all__ = ["RecurrentFormula", "compile_recurrent_formula"]


# ---------------------------------------------------------------------------
# Event keys: the shift algebra
# ---------------------------------------------------------------------------


def _shift_key(key: Hashable, delta: int) -> Hashable:
    """Translate an event key forward in time by ``delta`` discrete steps.

    Keys are analyzed in each node's own frame (its anchor at 0), so a parent
    that reads a child at offset ``o`` must shift the child's key by ``o``
    before comparing it with its siblings. Shifting is exactly the identity
    that makes anchor-0 analysis valid for every anchor:
    ``_shift_key(_shift_key(k, o), a) == _shift_key(k, o + a)``, and
    ``_shift_key(node_key, A)`` reproduces the key
    :mod:`pdstl.graph` builds for that node at anchor ``A``.

    Commutative operand tuples are re-canonicalised *after* shifting, since
    ``_canonical_keys`` orders by ``repr`` and that order is not stable under
    translation. The window bounds ``a``/``b`` of a temporal key are operator
    parameters, not times, so they are deliberately left alone.
    """
    if delta == 0:
        return key

    tag = key[0]
    if tag == "atom":
        return ("atom", key[1], key[2] + delta)
    if tag == "not":
        return ("not", _shift_key(key[1], delta))
    if tag in ("and", "or"):
        return (tag, _canonical_keys([_shift_key(k, delta) for k in key[1]]))
    if tag in ("always", "eventually", "until"):
        return (tag, key[1], key[2], tuple(_shift_key(k, delta) for k in key[3]))
    raise ValueError(f"unrecognised event key tag: {tag!r}")


# ---------------------------------------------------------------------------
# References into computed traces
# ---------------------------------------------------------------------------


class _CellRef:
    """Read a cell's trace starting at a fixed offset."""

    __slots__ = ("cell", "offset")

    def __init__(self, cell: _Cell, offset: int) -> None:
        self.cell = cell
        self.offset = offset


class _ConstRef:
    """A TOP/BOTTOM outcome from exact-complement collapse: constant everywhere."""

    __slots__ = ("value",)

    def __init__(self, value: float) -> None:
        self.value = value


_Ref = _CellRef | _ConstRef

# One analyzed operand: where its value lives, and its structural event key.
_Entry = tuple[_Ref, Hashable]


def _shift_ref(ref: _Ref, delta: int) -> _Ref:
    """Move a reference ``delta`` steps later; constants are time-invariant."""
    if isinstance(ref, _ConstRef):
        return ref
    return _CellRef(ref.cell, ref.offset + delta)


def _read(ref: _Ref, length: int, batch: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Materialize ``[B, length, 2]`` for one reference."""
    if isinstance(ref, _ConstRef):
        return torch.full((batch, length, 2), ref.value, dtype=dtype, device=device)
    return ref.cell.trace[:, ref.offset : ref.offset + length, :]


# ---------------------------------------------------------------------------
# Identity-first structural reduction
# ---------------------------------------------------------------------------

# What a reduction decided, before any Frechet rule is consulted.
_SINGLE = "single"  # exactly one distinct event survives -- return it as-is
_CONSTANT = "constant"  # an exact complement pair was found -- empty set / whole space
_FOLD = "fold"  # nothing collapsed; a generic Frechet combination is required


def _reduce(
    entries: Sequence[_Entry], *, make_key: Callable[[tuple[Hashable, ...]], Hashable]
) -> tuple[str, list[_Entry], Hashable]:
    """Reduce operand *events* before any probability arithmetic happens.

    Mirrors ``propagate.EvaluationContext._combine_events`` and
    ``graph._Compiler._combine`` on the structural half of their work:

    1. duplicate events collapse (``A n A = A``, ``A u A = A``);
    2. a single survivor is returned under *its own* key, which is what makes
       the identities compositional through nesting and what makes a singleton
       temporal window equal to its child event;
    3. an exact complement pair is recognised (``A n A^c = empty``,
       ``A u A^c = Omega``).

    Returns the verdict, the surviving entries in first-seen order, and the
    resulting event key. The caller supplies the constant's value (0 for an
    intersection, 1 for a union) and builds whatever cell the ``_FOLD`` case
    needs; the *count* of survivors returned here is the post-reduction
    operand count ``m`` that must enter the Frechet intersection rule.
    """
    unique: list[_Entry] = []
    seen: set[Hashable] = set()
    for ref, key in entries:
        if key not in seen:
            seen.add(key)
            unique.append((ref, key))

    if len(unique) == 1:
        return _SINGLE, unique, unique[0][1]

    keys = tuple(key for _, key in unique)
    key = make_key(keys)

    if any(_is_complement(k1, k2) for k1, k2 in itertools.combinations(keys, 2)):
        return _CONSTANT, unique, key

    return _FOLD, unique, key


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


class _Cell:
    """One computation in the recurrent plan; produces ``[B, length, 2]``.

    ``length`` is assigned at compile time from the demands of every consumer,
    so each cell computes exactly the span its parents read and no more.
    """

    __slots__ = ("length", "trace")

    def __init__(self) -> None:
        self.length: int = 0
        self.trace: torch.Tensor | None = None

    def child_demands(self) -> Iterable[tuple[_Ref, int]]:
        """``(child ref, extra lookahead)`` pairs; child needs ``length + extra``."""
        return ()

    @property
    def state_width(self) -> int:
        """Width of the recurrent state, or 0 for a pointwise cell."""
        return 0

    def run(self, atom_traces, batch, dtype, device) -> None:
        raise NotImplementedError


class _AtomCell(_Cell):
    """Holds one predicate's materialized trace, shape ``[B, N+1, 2]``.

    Atoms are data, not a recurrence; every occurrence of the predicate reads
    this one tensor at its own offset.
    """

    __slots__ = ("uid",)

    def __init__(self, uid: int) -> None:
        super().__init__()
        self.uid = uid

    def run(self, atom_traces, batch, dtype, device) -> None:
        self.trace = atom_traces[self.uid]


class _NegationCell(_Cell):
    """Exact negation ``[l, u] -> [1-u, 1-l]``; pointwise in time."""

    __slots__ = ("ast_node", "child")

    def __init__(self, child: _Ref, ast_node: Negation) -> None:
        super().__init__()
        self.child = child
        self.ast_node = ast_node

    def child_demands(self):
        return ((self.child, 0),)

    def run(self, atom_traces, batch, dtype, device) -> None:
        child = _read(self.child, self.length, batch, dtype, device)
        self.trace = self.ast_node.combine(child)


class _StackFoldCell(_Cell):
    """A generic Frechet intersection/union over already-reduced operands.

    Serves the Boolean ``And``/``Or``, which are pointwise in time and so
    genuinely need no temporal recurrence. It also stands as the generic
    fallback for a temporal window whose survivor set is not the full
    contiguous window; that cannot arise (window operands are one child key
    shifted by *distinct* offsets, and shifting always changes an event), but
    routing to it costs nothing and keeps correctness unconditional.

    Operands arrive deduplicated and pairwise non-complementary, so the
    operand count here already *is* the post-reduction count ``m``.
    """

    __slots__ = ("intersection", "operands")

    def __init__(self, intersection: bool, operands: Sequence[_Ref]) -> None:
        super().__init__()
        self.intersection = intersection
        self.operands = list(operands)

    def child_demands(self):
        return tuple((operand, 0) for operand in self.operands)

    def run(self, atom_traces, batch, dtype, device) -> None:
        stacked = torch.stack(
            [_read(operand, self.length, batch, dtype, device) for operand in self.operands],
            dim=-2,
        )  # [B, length, m, 2]
        combine = frechet_intersection if self.intersection else frechet_union
        self.trace = combine(stacked)


class _LadderMixin:
    """Shared bookkeeping for the backward-scan cells.

    ``last_scan_indices`` records the scan positions visited by the most
    recent execution, in order, which is what makes "forward-time STL is
    evaluated by a backward scan" checkable rather than merely asserted.
    Concrete cells declare the slot themselves, so this mixin stays layout-free.
    """

    __slots__ = ()

    def _reset_scan(self) -> None:
        self.last_scan_indices: list[int] = []

    @property
    def n_state_updates(self) -> int:
        """Backward steps taken; linear in the trace length, not ``T * W``."""
        return len(self.last_scan_indices)


class _WindowCell(_Cell, _LadderMixin):
    """``G_[a,b]`` / ``F_[a,b]`` as a backward scan with bounded sliding state.

    The ladder state is ``W + 1`` entries wide for a window of width
    ``W = b - a + 1`` and is independent of the trace length. Output at anchor
    ``k`` reads ladder entry ``W`` at scan position ``p = k + a``; the scan
    starts at the far end of the child view, so that entry is always fully
    populated by the time it is read.
    """

    __slots__ = ("a", "b", "child", "intersection", "last_scan_indices")

    def __init__(self, intersection: bool, child: _Ref, a: int, b: int) -> None:
        super().__init__()
        self.intersection = intersection
        self.child = child
        self.a = a
        self.b = b
        self._reset_scan()

    def child_demands(self):
        return ((self.child, self.b),)

    @property
    def state_width(self) -> int:
        return self.b - self.a + 2

    def run(self, atom_traces, batch, dtype, device) -> None:
        a, b = self.a, self.b
        width = b - a + 1
        view = _read(self.child, self.length + b, batch, dtype, device)
        view_length = view.shape[1]

        zeros = torch.zeros((batch, 1), dtype=dtype, device=device)
        ones = torch.ones((batch, 1), dtype=dtype, device=device)
        # totals[j] and extrema[j] summarise the j child events at p .. p+j-1.
        totals = torch.zeros((batch, width + 1), dtype=dtype, device=device)
        if self.intersection:
            extrema = torch.ones((batch, width + 1), dtype=dtype, device=device)
        else:
            extrema = torch.zeros((batch, width + 1), dtype=dtype, device=device)

        outputs: list[torch.Tensor | None] = [None] * self.length
        self._reset_scan()
        for p in range(view_length - 1, a - 1, -1):
            self.last_scan_indices.append(p)
            lower = view[:, p, 0:1]
            upper = view[:, p, 1:2]
            if self.intersection:
                totals = torch.cat([zeros, totals[:, :width] + lower], dim=1)
                extrema = torch.cat([ones, torch.minimum(extrema[:, :width], upper)], dim=1)
            else:
                totals = torch.cat([zeros, totals[:, :width] + upper], dim=1)
                extrema = torch.cat([zeros, torch.maximum(extrema[:, :width], lower)], dim=1)

            k = p - a
            if 0 <= k < self.length:
                if self.intersection:
                    outputs[k] = _frechet_intersection_from_totals(
                        totals[:, width], extrema[:, width], width
                    )
                else:
                    outputs[k] = _frechet_union_from_totals(extrema[:, width], totals[:, width])

        self.trace = torch.stack(outputs, dim=1)  # [B, length, 2]


class _UntilCell(_Cell, _LadderMixin):
    """Bounded strong ``U_[a,b]`` as a backward scan over a shared prefix ladder.

    At anchor ``k`` the exact event is

        union_{j=a..b} ( E_psi,k+j  n  intersect_{r=0..j-1} E_phi,k+r )

    -- the ``phi`` prefix always starting at ``r = 0``, never at ``r = a``.
    One intersection ladder over ``phi`` supplies the totals of *every* prefix
    ``P_j`` simultaneously, so no candidate ever recomputes a prefix and the
    per-anchor cost is ``O(1)`` tensor operations over an ``O(W)``-wide state.

    Per-candidate structural reduction is resolved at compile time into three
    fixed vectors:

    ``counts``
        the post-reduction operand count ``m_j`` that enters the Frechet
        intersection rule -- not ``j`` and not the syntactic count;
    ``include_psi``
        0 when ``E_psi,k+j`` duplicates one prefix event and the reduction
        therefore drops that prefix element. The dropped element carries the
        *same* event and hence the same bounds as ``psi``, so removing it from
        the sum is exactly cancelling ``psi``'s own contribution, and the
        running minimum is unaffected;
    ``false_mask``
        candidates that contain an exact complement pair and are therefore the
        empty event ``[0, 0]``.

    For ``a > 0`` every candidate is contained in the common prefix
    ``P_a``, so the union's upper bound is capped by ``P_a``'s -- read
    directly off the ladder state at depth ``a``. The lower bound is
    untouched and no dependence is assumed.
    """

    __slots__ = (
        "_counts",
        "_false_mask",
        "_has_false",
        "_include_psi",
        "_offsets",
        "a",
        "b",
        "last_scan_indices",
        "left",
        "right",
    )

    def __init__(
        self,
        left: _Ref,
        right: _Ref,
        a: int,
        b: int,
        offsets: Sequence[int],
        counts: Sequence[int],
        include_psi: Sequence[float],
        false_flags: Sequence[bool],
    ) -> None:
        super().__init__()
        self.left = left
        self.right = right
        self.a = a
        self.b = b
        self._offsets = tuple(offsets)
        self._counts = tuple(counts)
        self._include_psi = tuple(include_psi)
        self._false_mask = tuple(false_flags)
        self._has_false = any(false_flags)
        self._reset_scan()

    def child_demands(self):
        return ((self.left, self.b - 1), (self.right, self.b))

    @property
    def state_width(self) -> int:
        return self.b + 1

    @property
    def n_prefix_updates(self) -> int:
        """Times a ``phi`` event was folded into the shared prefix ladder."""
        return len(self.last_scan_indices)

    def run(self, atom_traces, batch, dtype, device) -> None:
        a, b = self.a, self.b
        left = _read(self.left, self.length + b - 1, batch, dtype, device)
        right = _read(self.right, self.length + b, batch, dtype, device)
        left_length = left.shape[1]

        offsets = torch.tensor(self._offsets, dtype=torch.long, device=device)
        counts = torch.tensor(self._counts, dtype=dtype, device=device)
        include = torch.tensor(self._include_psi, dtype=dtype, device=device)
        false_mask = torch.tensor(self._false_mask, dtype=torch.bool, device=device).view(1, -1, 1)

        zeros = torch.zeros((batch, 1), dtype=dtype, device=device)
        ones = torch.ones((batch, 1), dtype=dtype, device=device)
        # sums[j] / mins[j] are the Frechet totals of the prefix P_j at anchor p.
        sums = torch.zeros((batch, b + 1), dtype=dtype, device=device)
        mins = torch.ones((batch, b + 1), dtype=dtype, device=device)

        outputs: list[torch.Tensor | None] = [None] * self.length
        self._reset_scan()
        for p in range(left_length - 1, -1, -1):
            self.last_scan_indices.append(p)
            sums = torch.cat([zeros, sums[:, :b] + left[:, p, 0:1]], dim=1)
            mins = torch.cat([ones, torch.minimum(mins[:, :b], left[:, p, 1:2])], dim=1)

            if p >= self.length:
                continue  # still warming the ladder; no valid anchor here yet

            psi = right.index_select(1, offsets + p)  # [B, J, 2]
            total_lower = sums.index_select(1, offsets) + include * psi[..., 0]
            min_upper = torch.minimum(mins.index_select(1, offsets), psi[..., 1])
            candidates = _frechet_intersection_from_totals(total_lower, min_upper, counts)
            if self._has_false:
                candidates = torch.where(false_mask, torch.zeros_like(candidates), candidates)

            value = _frechet_union_from_totals(
                candidates[..., 0].amax(dim=-1), candidates[..., 1].sum(dim=-1)
            )
            if a > 0:
                value = torch.stack(
                    [value[..., 0], torch.minimum(value[..., 1], mins[:, a])], dim=-1
                )
            outputs[p] = value

        self.trace = torch.stack(outputs, dim=1)  # [B, length, 2]


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


class _Compiler:
    """Analyzes a formula once, at anchor 0, into a plan of recurrent cells.

    Every node is analyzed in its *own* frame, so the result is independent of
    where the node appears; a parent reading a child at offset ``o`` shifts the
    child's ref and key by ``o``. Nodes are memoized both by object identity
    and by structural event key, so structurally identical subformulas share
    one cell -- and, crucially, so that two syntactically different operands
    whose *shifted* keys coincide are still recognised as the same event.
    """

    def __init__(self) -> None:
        self.cells: list[_Cell] = []
        self._atom_cells: dict[int, _AtomCell] = {}
        self._by_id: dict[int, _Entry] = {}
        self._by_key: dict[Hashable, _Entry] = {}
        self.referenced_uids: set[int] = set()
        self.predicate_names: dict[int, str] = {}

    def analyze(self, ast_node: STLFormula) -> _Entry:
        cached = self._by_id.get(id(ast_node))
        if cached is not None:
            return cached

        if isinstance(ast_node, Predicate):
            entry = self._analyze_atom(ast_node)
        elif isinstance(ast_node, Negation):
            entry = self._analyze_negation(ast_node)
        elif isinstance(ast_node, (And, Or)):
            entry = self._analyze_binary(ast_node)
        elif isinstance(ast_node, Until):
            entry = self._analyze_until(ast_node)
        elif isinstance(ast_node, TemporalOperator):
            entry = self._analyze_temporal(ast_node)
        else:
            raise TypeError(f"unsupported formula type: {type(ast_node).__name__}")

        shared = self._by_key.get(entry[1])
        if shared is not None:
            entry = shared  # same event: reuse the cell already computing it
        else:
            self._by_key[entry[1]] = entry
        self._by_id[id(ast_node)] = entry
        return entry

    def _add(self, cell: _Cell) -> _CellRef:
        self.cells.append(cell)
        return _CellRef(cell, 0)

    def _analyze_atom(self, ast_node: Predicate) -> _Entry:
        self.referenced_uids.add(ast_node.uid)
        self.predicate_names[ast_node.uid] = str(ast_node)
        cell = self._atom_cells.get(ast_node.uid)
        if cell is None:
            cell = _AtomCell(ast_node.uid)
            self._atom_cells[ast_node.uid] = cell
            self.cells.append(cell)
        return _CellRef(cell, 0), ("atom", ast_node.uid, 0)

    def _analyze_negation(self, ast_node: Negation) -> _Entry:
        child_ref, child_key = self.analyze(ast_node.subformula)
        return self._add(_NegationCell(child_ref, ast_node)), ("not", child_key)

    def _analyze_binary(self, ast_node: And | Or) -> _Entry:
        entries = [self.analyze(ast_node.left), self.analyze(ast_node.right)]
        tag = ast_node.tag[0]
        intersection = isinstance(ast_node, And)
        verdict, survivors, key = _reduce(
            entries, make_key=lambda keys: (tag, _canonical_keys(keys))
        )
        if verdict is _SINGLE:
            return survivors[0]
        if verdict is _CONSTANT:
            return _ConstRef(0.0 if intersection else 1.0), key
        return self._add(_StackFoldCell(intersection, [ref for ref, _ in survivors])), key

    def _analyze_temporal(self, ast_node: Always | Eventually) -> _Entry:
        a, b = ast_node.a, ast_node.b
        child_ref, child_key = self.analyze(ast_node.subformula)
        entries = [
            (_shift_ref(child_ref, offset), _shift_key(child_key, offset))
            for offset in range(a, b + 1)
        ]
        tag = ast_node.tag[0]
        intersection = isinstance(ast_node, Always)
        verdict, survivors, key = _reduce(entries, make_key=lambda keys: (tag, a, b, keys))

        if verdict is _SINGLE:
            # A singleton window is the child event itself, at offset a.
            return survivors[0]
        if verdict is _CONSTANT:
            return _ConstRef(0.0 if intersection else 1.0), key
        if len(survivors) == b - a + 1:
            return self._add(_WindowCell(intersection, child_ref, a, b)), key
        # Unreachable in practice (see _StackFoldCell); kept so correctness
        # never depends on that argument holding.
        return self._add(_StackFoldCell(intersection, [ref for ref, _ in survivors])), key

    def _analyze_until(self, ast_node: Until) -> _Entry:
        a, b = ast_node.a, ast_node.b
        left_ref, left_key = self.analyze(ast_node.left)
        right_ref, right_key = self.analyze(ast_node.right)

        def conjunction_key(keys: tuple[Hashable, ...]) -> Hashable:
            return ("and", _canonical_keys(keys))

        # E_{phi, k+r} for r = 0 .. b-1. The prefix always starts at r = 0.
        prefix = [
            (_shift_ref(left_ref, r), _shift_key(left_key, r)) for r in range(b)
        ]

        offsets: list[int] = []
        counts: list[int] = []
        include_psi: list[float] = []
        false_flags: list[bool] = []
        union_entries: list[_Entry] = []
        alias_by_key: dict[Hashable, _Ref | None] = {}

        for j in range(a, b + 1):
            right_entry = (_shift_ref(right_ref, j), _shift_key(right_key, j))
            verdict, survivors, key = _reduce(
                [right_entry, *prefix[:j]], make_key=conjunction_key
            )

            if verdict is _CONSTANT:
                offsets.append(j)
                counts.append(1)
                include_psi.append(1.0)
                false_flags.append(True)
                alias = _ConstRef(0.0)
            else:
                # psi is first in the operand list, so it is always retained;
                # prefix events are pairwise distinct, so at most one of them
                # can be dropped as a duplicate of psi.
                dropped = j + 1 - len(survivors)
                if dropped not in (0, 1):
                    raise AssertionError(
                        f"Until candidate j={j} of {ast_node} reduced {dropped} prefix "
                        "operands; at most one is structurally possible"
                    )
                offsets.append(j)
                counts.append(j + 1 - dropped)
                include_psi.append(0.0 if dropped else 1.0)
                false_flags.append(False)
                alias = survivors[0][0] if verdict is _SINGLE else None

            union_entries.append((alias if alias is not None else _ConstRef(0.0), key))
            alias_by_key[key] = alias

        verdict_u, survivors_u, key_u = _reduce(
            union_entries,
            # Not the "or" namespace: the value below carries the common-prefix
            # tightening, so it must not alias a plain disjunction of the same
            # candidate events.
            make_key=lambda keys: ("until", a, b, keys),
        )

        if verdict_u is _CONSTANT:
            return _ConstRef(1.0), key_u
        if verdict_u is _SINGLE:
            alias = alias_by_key[survivors_u[0][1]]
            if alias is not None:
                # A lone candidate that is itself a lone event: `phi U[0,0] psi`
                # is exactly `psi`, with psi's own identity. The common-prefix
                # cap is a no-op here -- that candidate already intersects every
                # prefix event, so its upper bound cannot exceed P_a's.
                return alias, key_u

        # Keep one slot per surviving candidate event, in candidate order.
        surviving = {key for _, key in survivors_u}
        keep: list[int] = []
        taken: set[Hashable] = set()
        for index, (_, key) in enumerate(union_entries):
            if key in surviving and key not in taken:
                taken.add(key)
                keep.append(index)

        cell = _UntilCell(
            left_ref,
            right_ref,
            a,
            b,
            offsets=[offsets[i] for i in keep],
            counts=[counts[i] for i in keep],
            include_psi=[include_psi[i] for i in keep],
            false_flags=[false_flags[i] for i in keep],
        )
        return self._add(cell), key_u


def _assign_lengths(
    cells: Sequence[_Cell], root_ref: _Ref, valid_length: int, horizon: int
) -> list[_Cell]:
    """Size every cell to what its consumers actually read, and drop the rest.

    Cells are created children-first, so walking the list in reverse visits
    every consumer before the cell it consumes. A cell nothing demands is
    unreachable -- it lost an operand slot to an event-identity reduction --
    and is pruned rather than executed.
    """
    demands: dict[int, int] = {id(cell): 0 for cell in cells}

    def require(ref: _Ref, length: int) -> None:
        if isinstance(ref, _ConstRef):
            return
        needed = ref.offset + length
        demands[id(ref.cell)] = max(demands[id(ref.cell)], needed)

    require(root_ref, valid_length)

    reachable: list[_Cell] = []
    for cell in reversed(cells):
        demand = demands[id(cell)]
        if demand == 0:
            continue
        if isinstance(cell, _AtomCell):
            if demand > horizon + 1:
                raise AssertionError(
                    f"atom uid={cell.uid} needs {demand} time steps but the horizon "
                    f"only provides {horizon + 1}"
                )
            cell.length = horizon + 1
        else:
            cell.length = demand
        reachable.append(cell)
        for child_ref, extra in cell.child_demands():
            require(child_ref, cell.length + extra)

    reachable.reverse()  # back to execution (children-first) order
    return reachable


class RecurrentFormula:
    """A reusable formula-structured recurrent evaluator for one formula.

    Construct via :func:`compile_recurrent_formula`. Calling the instance runs
    the plan's cells -- unfolding each temporal operator backward over time
    through its own bounded recurrent state -- and returns the complete
    ``[B, T_valid, 2]`` hard probability-bound trace, with
    ``[..., 0]`` the exact hard lower bound and ``[..., 1]`` the exact hard
    upper bound.

    There are no learned parameters; see :meth:`parameters`.
    """

    def __init__(
        self,
        formula: STLFormula,
        horizon: int,
        valid_length: int,
        root_ref: _Ref,
        cells: list[_Cell],
        referenced_uids: set[int],
        predicate_names: dict[int, str],
    ) -> None:
        self.formula = formula
        self.horizon = horizon
        self.valid_length = valid_length
        self._root_ref = root_ref
        self._cells = cells
        self._referenced_uids = frozenset(referenced_uids)
        self._predicate_names = dict(predicate_names)

    @property
    def cells(self) -> tuple[_Cell, ...]:
        """The plan's cells, in execution order (children before consumers)."""
        return tuple(self._cells)

    @property
    def n_cells(self) -> int:
        """Number of cells; one per surviving operator, not per (operator, time)."""
        return len(self._cells)

    @property
    def temporal_cells(self) -> tuple[_Cell, ...]:
        """The backward-scan cells: one per surviving ``G``/``F``/``U``."""
        return tuple(cell for cell in self._cells if isinstance(cell, _LadderMixin))

    @property
    def recurrent_state_size(self) -> int:
        """Total recurrent state width, in ladder entries.

        Depends only on the formula's temporal windows, never on the horizon.
        """
        return sum(cell.state_width for cell in self._cells)

    def parameters(self) -> tuple:
        """No learned parameters exist: this recurrence is defined by the formula."""
        return ()

    def __call__(self, atom_traces: Mapping[int, torch.Tensor]) -> torch.Tensor:
        """Execute the recurrent plan. Returns shape ``[B, T_valid, 2]``."""
        batch, dtype, device = _validate_atom_traces(
            atom_traces,
            referenced_uids=self._referenced_uids,
            predicate_names=self._predicate_names,
            horizon=self.horizon,
            context="recurrent evaluator",
        )

        for cell in self._cells:
            cell.run(atom_traces, batch, dtype, device)

        return _read(self._root_ref, self.valid_length, batch, dtype, device)


def compile_recurrent_formula(formula: STLFormula, *, horizon: int) -> RecurrentFormula:
    """Compile ``formula`` into a recurrent evaluator for a fixed source horizon.

    Parameters
    ----------
    formula : STLFormula
        A bounded-time pdSTL formula.
    horizon : int
        The largest discrete time index atomic traces will cover; valid times
        are ``0 .. horizon`` inclusive, matching ``ProbabilitySource.horizon``.

    Returns
    -------
    RecurrentFormula
        A reusable callable; see :meth:`RecurrentFormula.__call__`.

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
    root_ref, _ = compiler.analyze(formula)
    cells = _assign_lengths(compiler.cells, root_ref, valid_length, horizon)
    return RecurrentFormula(
        formula,
        horizon,
        valid_length,
        root_ref,
        cells,
        compiler.referenced_uids,
        compiler.predicate_names,
    )
