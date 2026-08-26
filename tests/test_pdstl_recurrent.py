"""The formula-structured recurrent hard-probability backend, ``pdstl.recurrent``.

The central theorem, in code::

    ReferenceHard == CompiledHard == RecurrentHard

``propagate.evaluate`` is the semantic oracle, ``graph.CompiledFormula`` is the
compiled fold DAG, and ``recurrent.RecurrentFormula`` unfolds each temporal
operator backward over time through a bounded, non-learned recurrent state.
This file checks the three agree over the *complete* valid trace -- not only at
``t = 0`` -- for every supported operator, every event-identity reduction, and
randomized formulas; that deterministic 0/1 atoms reproduce ordinary Boolean
STL; that autograd survives the recurrence; and that the temporal computation
really is a backward stateful scan rather than another static fold graph.
"""

import ast
import inspect
import random

import pytest
import torch

from pdstl import (
    Always,
    And,
    Eventually,
    Negation,
    Or,
    Predicate,
    TableProbabilitySource,
    Until,
    evaluate,
)
from pdstl import recurrent as recurrent_module
from pdstl.gaussian import GaussianHalfspace, gaussian_atom_traces
from pdstl.graph import compile_formula, materialize_atom_traces
from pdstl.recurrent import compile_recurrent_formula

from tests.pdstl_rollout import differentiable_rollout

SEED = 20260825
ATOL = 1e-6


# ---------------------------------------------------------------------------
# Three-way comparison helpers
# ---------------------------------------------------------------------------


def recurrent_trace(formula, source, horizon=None):
    """The recurrent backend's full trace, materializing atoms from ``source``."""
    if horizon is None:
        horizon = source.horizon
    traces = materialize_atom_traces(formula, source, horizon)
    return compile_recurrent_formula(formula, horizon=horizon)(traces)


def recurrent_enclosure(formula, source, horizon=None, time=0):
    return recurrent_trace(formula, source, horizon)[0, time].tolist()


def assert_all_backends_agree(formula, source, horizon=None, atol=ATOL):
    """Reference, compiled and recurrent hard semantics must all coincide."""
    if horizon is None:
        horizon = source.horizon

    reference = evaluate(formula, source)
    traces = materialize_atom_traces(formula, source, horizon)
    compiled = compile_formula(formula, horizon=horizon)(traces)
    recurrent = compile_recurrent_formula(formula, horizon=horizon)(traces)

    assert compiled.shape == reference.shape, f"{formula}: compiled shape"
    assert recurrent.shape == reference.shape, f"{formula}: recurrent shape"
    assert torch.allclose(compiled, reference, atol=atol), (
        f"{formula}\nreference={reference}\ncompiled={compiled}"
    )
    assert torch.allclose(recurrent, reference, atol=atol), (
        f"{formula}\nreference={reference}\nrecurrent={recurrent}"
    )
    assert torch.allclose(recurrent, compiled, atol=atol), (
        f"{formula}\ncompiled={compiled}\nrecurrent={recurrent}"
    )
    return recurrent


@pytest.fixture
def worked_example():
    """A0=[.80,.90], A1=[.70,.85], A2=[.90,.95], B0=[.60,.75], B1=[.80,.90], B2=[.50,.70]."""
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource(
        {
            (a, 0): (0.80, 0.90),
            (a, 1): (0.70, 0.85),
            (a, 2): (0.90, 0.95),
            (b, 0): (0.60, 0.75),
            (b, 1): (0.80, 0.90),
            (b, 2): (0.50, 0.70),
        },
        horizon=2,
    )
    return a, b, source


@pytest.fixture
def a_source():
    a = Predicate(name="A")
    return a, TableProbabilitySource({(a, k): (0.6, 0.9) for k in range(3)})


# ---------------------------------------------------------------------------
# Phase 9: per-operator equivalence over the complete trace
# ---------------------------------------------------------------------------


def test_atom_agrees(worked_example):
    a, _, source = worked_example
    assert_all_backends_agree(a, source)


def test_negated_atom_agrees(worked_example):
    a, _, source = worked_example
    out = assert_all_backends_agree(Negation(a), source)
    assert out[0, 0].tolist() == pytest.approx([0.10, 0.20])


def test_and_agrees(worked_example):
    a, b, source = worked_example
    out = assert_all_backends_agree(And(a, b), source)
    assert out[0, 0].tolist() == pytest.approx([0.40, 0.75])


def test_or_agrees(worked_example):
    a, b, source = worked_example
    out = assert_all_backends_agree(Or(a, b), source)
    assert out[0, 0].tolist() == pytest.approx([0.80, 1.00])


def test_always_is_exact_probability_intersection(worked_example):
    """Not a temporal min on both bounds: the lower bound is the Frechet one."""
    a, _, source = worked_example
    out = assert_all_backends_agree(Always(a, interval=[0, 2]), source)
    assert out[0, 0].tolist() == pytest.approx([0.40, 0.85])
    # A temporal-min implementation would report min(0.80, 0.70, 0.90) = 0.70.
    assert out[0, 0, 0].item() != pytest.approx(0.70)


def test_eventually_is_exact_probability_union(worked_example):
    a, _, source = worked_example
    out = assert_all_backends_agree(Eventually(a, interval=[0, 2]), source)
    assert out[0, 0].tolist() == pytest.approx([0.90, 1.00])


def test_nested_boolean_formulas_agree(worked_example):
    a, b, source = worked_example
    for formula in [
        And(Or(a, b), And(a, b)),
        Or(And(a, Negation(b)), And(Negation(a), b)),
        And(And(a, b), Or(b, Negation(a))),
    ]:
        assert_all_backends_agree(formula, source)


def test_nested_temporal_formulas_agree(worked_example):
    a, b, source = worked_example
    for formula in [
        Always(And(a, b), interval=[0, 1]),
        Eventually(Or(a, Always(b, [0, 1])), interval=[0, 1]),
        Always(Eventually(a, [0, 1]), interval=[0, 1]),
        Eventually(Always(b, [0, 1]), interval=[0, 1]),
    ]:
        assert_all_backends_agree(formula, source)


def test_mixed_formulas_agree():
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource(
        {(p, k): (0.5, 0.7) for p in (a, b) for k in range(7)}, horizon=6
    )
    for formula in [
        And(Always(a, [0, 2]), Eventually(b, [1, 3])),
        Or(Until(a, b, [0, 2]), Always(a, [1, 2])),
        Always(Until(a, b, [0, 1]), [0, 2]),
        Eventually(And(Always(a, [0, 1]), Negation(b)), [1, 2]),
    ]:
        assert_all_backends_agree(formula, source)


def test_singleton_windows_agree(a_source):
    a, source = a_source
    for formula in [
        Always(a, [0, 0]),
        Always(a, [2, 2]),
        Eventually(a, [1, 1]),
        And(Always(a, [0, 0]), Eventually(a, [0, 0])),
    ]:
        assert_all_backends_agree(formula, source)


# ---------------------------------------------------------------------------
# Phase 8: Until, all required edge cases
# ---------------------------------------------------------------------------


def test_until_00_is_the_right_operand():
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource({(a, 0): (0.50, 0.50), (b, 0): (0.30, 0.60)}, horizon=0)
    out = assert_all_backends_agree(Until(a, b, [0, 0]), source)
    assert out[0, 0].tolist() == pytest.approx([0.30, 0.60])


def test_until_00_identity_survives_into_surrounding_structure():
    """``phi U[0,0] psi == psi`` with psi's *own* event identity, not just its value."""
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource({(a, 0): (0.50, 0.50), (b, 0): (0.30, 0.60)}, horizon=0)
    assert recurrent_enclosure(Until(a, b, [0, 0]) & b, source) == pytest.approx([0.30, 0.60])
    assert recurrent_enclosure(Until(a, b, [0, 0]) & ~b, source) == pytest.approx([0.0, 0.0])


def test_until_11_agrees(worked_example):
    a, b, source = worked_example
    out = assert_all_backends_agree(Until(a, b, [1, 1]), source)
    assert out[0, 0].tolist() == pytest.approx([0.60, 0.90])


def test_until_01_agrees():
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource(
        {
            (a, 0): (0.50, 0.60),
            (a, 1): (0.0, 1.0),  # unread by U[0,1]; present only for materialization
            (b, 0): (0.10, 0.20),
            (b, 1): (0.30, 0.40),
        }
    )
    out = assert_all_backends_agree(Until(a, b, [0, 1]), source)
    assert out[0, 0].tolist() == pytest.approx([0.10, 0.60])


def test_until_12_keeps_the_common_prefix_upper_tightening(worked_example):
    a, b, source = worked_example
    out = assert_all_backends_agree(Until(a, b, [1, 2]), source)
    assert out[0, 0].tolist() == pytest.approx([0.60, 0.90])
    # Without the P_a cap the Frechet union alone would allow an upper of 1.00.
    assert out[0, 0].tolist() != pytest.approx([0.60, 1.00])


def test_until_22_agrees(worked_example):
    a, b, source = worked_example
    out = assert_all_backends_agree(Until(a, b, [2, 2]), source)
    assert out[0, 0].tolist() == pytest.approx([0.0, 0.70])


def test_until_agrees_over_the_full_trace_including_nesting():
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource({(p, k): (0.5, 0.7) for p in (a, b) for k in range(8)})
    for formula in [
        Until(a, b, [0, 0]),
        Until(a, b, [1, 1]),
        Until(a, b, [0, 1]),
        Until(a, b, [1, 2]),
        Until(a, b, [2, 2]),
        Until(a, b, [0, 3]),
        Until(Always(a, [0, 1]), b, [1, 2]),
        Until(a, Eventually(b, [0, 1]), [2, 4]),
        Until(Until(a, b, [0, 1]), b, [1, 2]),
        Until(a, Until(a, b, [0, 1]), [1, 2]),
    ]:
        assert_all_backends_agree(formula, source)


def test_until_prefix_starts_at_zero_not_at_a():
    """``phi`` must hold from ``r = 0``, so violating it before the window opens kills the formula."""
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource(
        {
            (a, 0): (0.0, 0.0),  # phi is false at r = 0, before the window opens
            (a, 1): (1.0, 1.0),
            (a, 2): (1.0, 1.0),
            (b, 0): (1.0, 1.0),
            (b, 1): (1.0, 1.0),
            (b, 2): (1.0, 1.0),
        },
        horizon=2,
    )
    out = assert_all_backends_agree(Until(a, b, [1, 2]), source)
    assert out[0, 0].tolist() == pytest.approx([0.0, 0.0])


# ---------------------------------------------------------------------------
# Phase 10: event-identity regressions
# ---------------------------------------------------------------------------


def test_repeated_conjunction_and_disjunction_preserve_identity(a_source):
    a, source = a_source
    assert recurrent_enclosure(a & a, source) == pytest.approx([0.6, 0.9])
    assert recurrent_enclosure(a | a, source) == pytest.approx([0.6, 0.9])
    assert_all_backends_agree(a & a, source)
    assert_all_backends_agree(a | a, source)


def test_nested_repetition_stays_exact(a_source):
    """``(A & A) & A == A``, not the naive Frechet fold of three copies."""
    a, source = a_source
    assert recurrent_enclosure((a & a) & a, source) == pytest.approx([0.6, 0.9])
    assert recurrent_enclosure((a | a) | a, source) == pytest.approx([0.6, 0.9])
    assert recurrent_enclosure((a & a) & a, source) != pytest.approx([0.0, 0.9])
    assert_all_backends_agree((a & a) & a, source)
    assert_all_backends_agree((a | a) | a, source)


def test_complement_identities_are_exact():
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, 0): (0.4, 0.7)}, horizon=0)
    assert recurrent_enclosure(a & ~a, source) == pytest.approx([0.0, 0.0])
    assert recurrent_enclosure(a | ~a, source) == pytest.approx([1.0, 1.0])
    assert recurrent_enclosure((a & a) & ~a, source) == pytest.approx([0.0, 0.0])
    assert recurrent_enclosure((a | a) | ~a, source) == pytest.approx([1.0, 1.0])
    assert recurrent_enclosure(~a & (a & a), source) == pytest.approx([0.0, 0.0])
    assert recurrent_enclosure((a & a) & ~a, source) != pytest.approx([0.0, 0.6])


def test_singleton_window_preserves_identity(a_source):
    a, source = a_source
    assert recurrent_enclosure(Always(a, [0, 0]) & a, source) == pytest.approx([0.6, 0.9])
    assert recurrent_enclosure(Eventually(a, [0, 0]) | a, source) == pytest.approx([0.6, 0.9])
    assert recurrent_enclosure(
        Always(a, [0, 0]) & Eventually(a, [0, 0]), source
    ) == pytest.approx([0.6, 0.9])


def test_distinct_offsets_are_not_collapsed(a_source):
    """Different offsets are different events; the generic Frechet fold applies."""
    a, source = a_source
    phi = Always(a, [0, 0]) & Always(a, [1, 1])
    assert recurrent_enclosure(phi, source) == pytest.approx([0.2, 0.9])
    assert_all_backends_agree(phi, source)


def test_temporal_repetition_collapsing_to_identical_event_keys(a_source):
    """``G[1,1]A`` and ``F[1,1]A`` are the same event, so they must not double-count."""
    a, source = a_source
    phi = Always(a, [1, 1]) & Eventually(a, [1, 1])
    assert recurrent_enclosure(phi, source) == pytest.approx([0.6, 0.9])
    assert_all_backends_agree(phi, source)


@pytest.fixture
def shifted_pair():
    """p, q on a 6-step horizon, all bounds [0.6, 0.9]."""
    p = Predicate(name="p")
    q = Predicate(name="q")
    source = TableProbabilitySource(
        {(z, k): (0.6, 0.9) for z in (p, q) for k in range(6)}, horizon=5
    )
    return p, q, source


def test_until_candidate_with_a_duplicate_event(shifted_pair):
    """``G[1,1]p U[1,1] p``: the right operand at j=1 *is* the prefix event at r=0.

    The candidate's post-reduction operand count is 1, not 2, so the exact
    answer is the event itself -- a naive Frechet fold over the syntactic two
    operands would report ``max(0, 1.2 - 1) = 0.2``.
    """
    p, _, source = shifted_pair
    formula = Until(Always(p, [1, 1]), p, [1, 1])
    out = assert_all_backends_agree(formula, source)
    assert out[0, 0].tolist() == pytest.approx([0.6, 0.9])
    assert out[0, 0].tolist() != pytest.approx([0.2, 0.9])


def test_until_candidate_with_a_complement_event(shifted_pair):
    """``G[1,1]p U[1,1] ~p``: the candidate contains an exact complement pair."""
    p, _, source = shifted_pair
    formula = Until(Always(p, [1, 1]), Negation(p), [1, 1])
    out = assert_all_backends_agree(formula, source)
    assert out[0, 0].tolist() == pytest.approx([0.0, 0.0])


def test_until_duplicate_and_complement_over_wider_windows(shifted_pair):
    """The duplicate/complement fires only at the offsets where it structurally can."""
    p, _, source = shifted_pair
    for formula in [
        Until(Always(p, [1, 1]), p, [0, 3]),
        Until(Always(p, [1, 1]), p, [2, 3]),
        Until(Always(p, [2, 2]), Negation(p), [0, 3]),
        Until(Always(p, [1, 1]), Negation(p), [1, 3]),
    ]:
        assert_all_backends_agree(formula, source)


def test_until_candidate_with_a_singleton_surviving_event(shifted_pair):
    """``U[1,1]`` whose lone candidate reduces to one event must keep that identity."""
    p, _, source = shifted_pair
    formula = Until(Always(p, [1, 1]), p, [1, 1])
    # The whole Until *is* the event ``p`` at offset 1, so conjoining it with
    # that same event changes nothing, and with its complement gives false.
    assert recurrent_enclosure(formula & Always(p, [1, 1]), source) == pytest.approx([0.6, 0.9])
    assert recurrent_enclosure(
        formula & Negation(p), source
    ) != pytest.approx([0.6, 0.9])
    assert_all_backends_agree(formula & Always(p, [1, 1]), source)


def test_event_identity_crosses_node_boundaries_via_shifting(shifted_pair):
    """``(G[1,1]p & G[1,1]q)`` and ``G[1,1](p & q)`` are the same event.

    They are syntactically different subformulas whose keys coincide only
    after shifting, so conjoining them must collapse rather than fold.
    """
    p, q, source = shifted_pair
    left = And(Always(p, [1, 1]), Always(q, [1, 1]))
    right = Always(And(p, q), [1, 1])
    alone = recurrent_enclosure(left, source)
    assert recurrent_enclosure(And(left, right), source) == pytest.approx(alone)
    assert_all_backends_agree(And(left, right), source)


def test_repeated_subexpression_shares_one_cell():
    """Structurally identical subformulas compile to the same cell, not copies."""
    a = Predicate(name="A")
    b = Predicate(name="B")
    shared = Always(And(a, b), [0, 2])
    formula = (shared & shared) & (shared | shared)

    baseline = compile_recurrent_formula(shared, horizon=4)
    wrapped = compile_recurrent_formula(formula, horizon=4)
    assert wrapped.n_cells == baseline.n_cells


# ---------------------------------------------------------------------------
# Phase 9 (continued): randomized differential testing
# ---------------------------------------------------------------------------


def _random_formula(rng, preds, depth, max_window=3):
    if depth <= 0 or rng.random() < 0.3:
        p = rng.choice(preds)
        return Negation(p) if rng.random() < 0.3 else p

    kind = rng.choice(["and", "or", "always", "eventually", "until"])
    if kind == "and":
        return And(_random_formula(rng, preds, depth - 1), _random_formula(rng, preds, depth - 1))
    if kind == "or":
        return Or(_random_formula(rng, preds, depth - 1), _random_formula(rng, preds, depth - 1))
    lo = rng.randint(0, max_window)
    hi = rng.randint(lo, max_window)
    if kind == "always":
        return Always(_random_formula(rng, preds, depth - 1), interval=[lo, hi])
    if kind == "eventually":
        return Eventually(_random_formula(rng, preds, depth - 1), interval=[lo, hi])
    return Until(
        _random_formula(rng, preds, depth - 1),
        _random_formula(rng, preds, depth - 1),
        [lo, hi],
    )


class _BatchedTable(TableProbabilitySource):
    """A TableProbabilitySource whose bounds are repeated across a batch."""

    def __init__(self, table, horizon, batch):
        super().__init__(table, horizon=horizon)
        self._repeat = batch

    def bounds(self, predicate, time):
        return super().bounds(predicate, time).repeat(self._repeat, 1)


def _random_source(rng, preds, horizon, batch):
    table = {}
    for p in preds:
        for t in range(horizon + 1):
            lo = round(rng.uniform(0.0, 0.9), 3)
            hi = round(rng.uniform(lo, 1.0), 3)
            table[(p, t)] = (lo, hi)
    if batch > 1:
        return _BatchedTable(table, horizon, batch)
    return TableProbabilitySource(table, horizon=horizon)


def test_randomized_differential_equivalence():
    """Random formulas: recurrent must match both other backends exactly."""
    rng = random.Random(SEED)
    horizon = 12
    failures = []

    trials = 0
    while trials < 150:
        preds = [Predicate(name=f"p{trials}_{j}") for j in range(3)]
        formula = _random_formula(rng, preds, depth=4)
        batch = rng.choice([1, 1, 1, 3])
        source = _random_source(rng, preds, horizon, batch)

        if horizon - formula.horizon() + 1 <= 0:
            continue
        trials += 1

        reference = evaluate(formula, source)
        traces = materialize_atom_traces(formula, source, horizon)
        compiled = compile_formula(formula, horizon=horizon)(traces)
        recurrent = compile_recurrent_formula(formula, horizon=horizon)(traces)

        if recurrent.shape != reference.shape:
            failures.append((formula, "shape", reference.shape, recurrent.shape))
        elif not torch.allclose(recurrent, reference, atol=1e-5):
            failures.append((formula, "vs reference", reference, recurrent))
        elif not torch.allclose(recurrent, compiled, atol=1e-5):
            failures.append((formula, "vs compiled", compiled, recurrent))

    assert not failures, f"{len(failures)} of {trials} random formulas mismatched: {failures[:2]}"


# ---------------------------------------------------------------------------
# Phase 11: deterministic 0/1 inputs reproduce ordinary Boolean STL
# ---------------------------------------------------------------------------


def boolean_stl(formula, signal, time):
    """Independent reference Boolean STL semantics; see test_pdstl_graph.py."""
    from pdstl.operators import TemporalOperator

    if isinstance(formula, Predicate):
        return bool(signal[formula.uid][time])
    if isinstance(formula, Negation):
        return not boolean_stl(formula.subformula, signal, time)
    if isinstance(formula, And):
        return boolean_stl(formula.left, signal, time) and boolean_stl(formula.right, signal, time)
    if isinstance(formula, Or):
        return boolean_stl(formula.left, signal, time) or boolean_stl(formula.right, signal, time)
    if isinstance(formula, Until):
        return any(
            boolean_stl(formula.right, signal, time + j)
            and all(boolean_stl(formula.left, signal, time + r) for r in range(j))
            for j in range(formula.a, formula.b + 1)
        )
    if isinstance(formula, TemporalOperator):
        window = range(time + formula.a, time + formula.b + 1)
        results = [boolean_stl(formula.subformula, signal, k) for k in window]
        return all(results) if isinstance(formula, Always) else any(results)
    raise TypeError(type(formula).__name__)


def test_deterministic_inputs_reproduce_boolean_stl():
    rng = random.Random(SEED + 1)
    horizon = 6

    a = Predicate(name="A")
    b = Predicate(name="B")

    formulas = [
        a & b,
        a | ~b,
        Always(a, [0, 2]),
        Eventually(b, [1, 3]),
        Until(a, b, [0, 0]),
        Until(a, b, [1, 3]),
        Until(a & b, b, [1, 2]),
        Always(Until(a, b, [0, 1]), [0, 1]),
        Until(a, b, [0, 2]) | Always(a, [0, 1]),
        Eventually(And(Always(a, [0, 1]), b), [0, 2]),
    ]

    for _ in range(20):
        signal = {
            a.uid: [rng.randint(0, 1) for _ in range(horizon + 1)],
            b.uid: [rng.randint(0, 1) for _ in range(horizon + 1)],
        }
        table = {}
        for predicate in (a, b):
            for k in range(horizon + 1):
                truth = float(signal[predicate.uid][k])
                table[(predicate, k)] = (truth, truth)
        source = TableProbabilitySource(table, horizon=horizon)

        for formula in formulas:
            out = recurrent_trace(formula, source, horizon)
            for time in range(out.shape[1]):
                lower, upper = out[0, time].tolist()
                expected = float(boolean_stl(formula, signal, time))
                assert lower == pytest.approx(upper), f"{formula} @ {time}"
                assert lower == pytest.approx(expected), f"{formula} @ {time}"


# ---------------------------------------------------------------------------
# Error handling parity with the compiled backend
# ---------------------------------------------------------------------------


def test_rejects_too_short_horizon():
    a = Predicate(name="A")
    with pytest.raises(ValueError, match="too short"):
        compile_recurrent_formula(Always(a, [0, 4]), horizon=2)


def test_rejects_missing_atom_trace():
    a = Predicate(name="A")
    b = Predicate(name="B")
    recurrent = compile_recurrent_formula(And(a, b), horizon=0)
    with pytest.raises(KeyError):
        recurrent({a.uid: torch.tensor([[[0.5, 0.5]]])})


def test_rejects_wrong_shape():
    a = Predicate(name="A")
    recurrent = compile_recurrent_formula(a, horizon=1)
    with pytest.raises(ValueError):
        recurrent({a.uid: torch.tensor([[0.5, 0.5]])})  # missing the time axis


def test_rejects_inconsistent_batch_size():
    a = Predicate(name="A")
    b = Predicate(name="B")
    recurrent = compile_recurrent_formula(And(a, b), horizon=0)
    with pytest.raises(ValueError, match="inconsistent batch size"):
        recurrent(
            {
                a.uid: torch.tensor([[[0.5, 0.5]]]),
                b.uid: torch.tensor([[[0.5, 0.5]], [[0.4, 0.6]]]),
            }
        )


def test_rejects_invalid_bounds():
    a = Predicate(name="A")
    recurrent = compile_recurrent_formula(a, horizon=0)
    with pytest.raises(ValueError):
        recurrent({a.uid: torch.tensor([[[0.7, 0.3]]])})  # lower > upper


def test_output_is_not_padded():
    """``T_valid = horizon - H(phi) + 1``; invalid tail times are omitted."""
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, k): (0.5, 0.6) for k in range(6)}, horizon=5)
    formula = Always(a, [0, 3])
    out = recurrent_trace(formula, source, 5)
    assert formula.horizon() == 3
    assert out.shape == (1, 5 - 3 + 1, 2)  # times 0..2 only; 3..5 are not padded in


def test_batched_inputs_agree(worked_example):
    a, b, source = worked_example
    formula = Until(a, b, [0, 2])
    traces = materialize_atom_traces(formula, source, 2)
    batched = {uid: torch.cat([t, t.flip(1)], dim=0) for uid, t in traces.items()}
    compiled = compile_formula(formula, horizon=2)(batched)
    recurrent = compile_recurrent_formula(formula, horizon=2)(batched)
    assert recurrent.shape[0] == 2
    assert torch.allclose(recurrent, compiled, atol=ATOL)


# ---------------------------------------------------------------------------
# Phase 12/13: autograd through the Gaussian provider and the recurrence
# ---------------------------------------------------------------------------

# Shared scenario, matching tests/test_pdstl_end_to_end.py: N=2, y0=0, safety
# margin M=0.6, q_std=0.3, v=0 init. Per-step safety probability is
# [1.0 (deterministic at k=0), 0.9772, 0.9214]; the Frechet-intersection
# pre-clamp lower is ~0.8986 (far from the 0 boundary) and the amin gap over
# the runner-up is ~0.056 -- both far larger than the h=1e-4 FD step, so a
# perturbation of that size cannot flip which branch of clamp/amin is active.
N = 2
D = 2
DT = 1.0
U_MAX = 1.0
Q_STD = 0.3
Y_MIN = -0.6
KINK_MARGIN_FLOOR = 0.01  # a comfortable multiple of h = 1e-4


def _scenario(v, *, y_min=Y_MIN, q_std=Q_STD, x0_std=0.0):
    x0_mean = torch.zeros(D)
    x0_cov = torch.diag(torch.tensor([x0_std**2, x0_std**2]))
    process_noise = torch.diag(torch.tensor([q_std**2, q_std**2]))
    mean, covariance = differentiable_rollout(
        v, x0_mean, x0_cov, dt=DT, u_max=U_MAX, process_noise=process_noise
    )

    mu_safe = Predicate(name="mu_safe")
    mu_goal = Predicate(name="mu_goal")  # unreferenced by the formula; zero-grad check
    halfspaces = [
        GaussianHalfspace(predicate=mu_safe, normal=torch.tensor([0.0, 1.0]), threshold=y_min),
        GaussianHalfspace(predicate=mu_goal, normal=torch.tensor([1.0, 0.0]), threshold=999.0),
    ]
    traces = gaussian_atom_traces(mean, covariance, halfspaces)

    formula = Always(mu_safe, interval=[0, N])
    return compile_recurrent_formula(formula, horizon=N), compile_formula(formula, horizon=N), traces


def test_recurrent_full_chain_produces_valid_bounds():
    v = torch.zeros(N, D, requires_grad=True)
    recurrent, _, traces = _scenario(v)
    out = recurrent(traces)

    assert out.shape == (1, 1, 2)
    lower, upper = out[0, 0].tolist()
    assert 0.0 <= lower <= upper <= 1.0


def test_recurrent_and_compiled_forward_values_match():
    v = torch.zeros(N, D, requires_grad=True)
    recurrent, compiled, traces = _scenario(v)
    assert torch.allclose(recurrent(traces), compiled(traces), atol=ATOL)


def test_gradient_flows_from_lower_bound_to_v():
    v = torch.zeros(N, D, requires_grad=True)
    recurrent, _, traces = _scenario(v)
    lower_bound = recurrent(traces)[0, 0, 0]
    (-lower_bound).backward()

    assert v.grad is not None
    assert torch.isfinite(v.grad).all()
    # The y-branch drives the safety atom, so it must carry real signal.
    assert torch.any(v.grad[:, 1] != 0.0)


def test_kink_margins_are_verified_before_finite_difference_check():
    """Check, don't assume, that the chosen point avoids Frechet kinks."""
    v = torch.zeros(N, D, requires_grad=True)
    _, _, traces = _scenario(v)

    ps = [trace[0, :, 0] for trace in traces.values()]
    # The safety trace is the one NOT saturated at 0 (mu_goal's threshold of
    # 999 drives it to exactly 0 everywhere).
    p = next(candidate for candidate in ps if candidate.max().item() > 0.0)

    n = p.shape[0]
    lower_margin = p.sum().item() - (n - 1)  # distance from the 0.0 clamp boundary
    sorted_p, _ = torch.sort(p)
    upper_margin = (sorted_p[1] - sorted_p[0]).item()  # winner vs. runner-up gap

    assert lower_margin > KINK_MARGIN_FLOOR, f"lower margin {lower_margin} too close to 0"
    assert upper_margin > KINK_MARGIN_FLOOR, f"upper margin {upper_margin} too close to the runner-up"


def test_finite_difference_matches_autograd_at_interior_point():
    v = torch.zeros(N, D, requires_grad=True)
    recurrent, _, traces = _scenario(v)
    recurrent(traces)[0, 0, 0].backward()
    analytic = v.grad.clone()

    h = 1e-4

    def lower_at(v_value):
        recurrent_h, _, traces_h = _scenario(v_value)
        return recurrent_h(traces_h)[0, 0, 0].item()

    numeric = torch.zeros_like(v)
    base = v.detach().clone()
    for k in range(N):
        for d in range(D):
            plus = base.clone()
            plus[k, d] += h
            minus = base.clone()
            minus[k, d] -= h
            numeric[k, d] = (lower_at(plus) - lower_at(minus)) / (2 * h)

    assert torch.allclose(analytic, numeric, atol=1e-3, rtol=1e-3), (
        f"analytic={analytic}\nnumeric={numeric}"
    )
    # mu_goal is unreferenced by Always(mu_safe, ...): the x-branch is exactly zero.
    assert torch.all(analytic[:, 0] == 0.0)
    assert torch.all(numeric[:, 0] == 0.0)


def test_recurrent_and_compiled_gradients_match_at_an_ordinary_point():
    """Away from clamp/amin ties both backends must select the same subgradient."""
    v_rec = torch.zeros(N, D, requires_grad=True)
    recurrent, _, traces_rec = _scenario(v_rec)
    recurrent(traces_rec)[0, 0, 0].backward()

    v_comp = torch.zeros(N, D, requires_grad=True)
    _, compiled, traces_comp = _scenario(v_comp)
    compiled(traces_comp)[0, 0, 0].backward()

    assert torch.allclose(v_rec.grad, v_comp.grad, atol=1e-6)


DIM = 2
TIE_MARGIN = 1e-3


def _gaussian_scenario(preds, controls, horizon):
    """A differentiable ``{uid: [1, N+1, 2]}`` provider driven by ``controls``."""
    v = controls.clone().requires_grad_(True)
    steps = torch.cumsum(torch.tanh(v), dim=0).unsqueeze(0)
    mean = torch.cat([torch.zeros(1, 1, DIM, dtype=torch.float64), steps], dim=1)
    covariance = torch.eye(DIM, dtype=torch.float64).expand(1, horizon + 1, DIM, DIM) * 0.25
    halfspaces = [
        GaussianHalfspace(
            predicate=p,
            normal=torch.tensor([1.0, 0.4 * i + 0.3]),
            threshold=0.25 * i - 0.1,
        )
        for i, p in enumerate(preds)
    ]
    return v, gaussian_atom_traces(mean, covariance, halfspaces)


def _is_an_ordinary_point(traces):
    """True when no atom probability sits on a tie or a saturation boundary.

    Where several operands of a ``max``/``min`` coincide exactly, or where a
    probability is pinned at 0 or 1, the two backends may legitimately select
    different subgradients -- their reduction orders differ. Those points are
    excluded from the gradient comparison rather than papered over.
    """
    for trace in traces.values():
        p = trace[0, :, 0]
        if p.min().item() < TIE_MARGIN or p.max().item() > 1.0 - TIE_MARGIN:
            return False
        gaps = (p.unsqueeze(0) - p.unsqueeze(1)).abs()
        gaps = gaps + torch.eye(p.shape[0], dtype=p.dtype)
        if gaps.min().item() < TIE_MARGIN:
            return False
    return True


def test_randomized_gradient_equivalence_with_the_compiled_backend():
    """Random formulas over differentiable Gaussian atoms: values *and* gradients match.

    Runs in float64 so the comparison is not masked by float32 noise, and
    differentiates ``sum(P_lower)`` over the whole valid trace rather than a
    single time, so every temporal cell contributes to the backward pass.
    Trials whose atomic probabilities land on a max/min tie or a saturation
    boundary are skipped -- see :func:`_is_an_ordinary_point`.
    """
    rng = random.Random(SEED + 2)
    horizon = 9
    mismatches = []
    forward_only_mismatches = []

    compared = 0
    attempts = 0
    while compared < 40 and attempts < 200:
        attempts += 1
        preds = [Predicate(name=f"q{attempts}_{j}") for j in range(3)]
        formula = _random_formula(rng, preds, depth=4)
        if horizon - formula.horizon() + 1 <= 0:
            continue

        controls = torch.tensor(
            [[rng.uniform(-0.8, 0.8) for _ in range(DIM)] for _ in range(horizon)],
            dtype=torch.float64,
        )
        _, probe = _gaussian_scenario(preds, controls, horizon)
        if not _is_an_ordinary_point(probe):
            continue
        compared += 1

        results = []
        for backend in ("compiled", "recurrent"):
            v, traces = _gaussian_scenario(preds, controls, horizon)
            evaluator = (
                compile_formula(formula, horizon=horizon)
                if backend == "compiled"
                else compile_recurrent_formula(formula, horizon=horizon)
            )
            out = evaluator(traces)
            out[..., 0].sum().backward()
            results.append((out.detach(), v.grad.clone()))

        if not torch.allclose(results[0][0], results[1][0], atol=1e-10):
            forward_only_mismatches.append(formula)
        elif not torch.allclose(results[0][1], results[1][1], atol=1e-8):
            mismatches.append(formula)

    assert compared >= 30, f"only {compared} ordinary points found in {attempts} attempts"
    assert not forward_only_mismatches, f"forward mismatches: {forward_only_mismatches[:2]}"
    assert not mismatches, f"{len(mismatches)} of {compared} gradients mismatched: {mismatches[:2]}"


def test_subgradient_selection_may_differ_only_at_exact_ties():
    """A fully tied union: values still agree exactly, gradients need not.

    Documents the one place the three backends are *not* required to coincide.
    With every atomic probability exactly 0.5, ``max_i l_i`` has a degenerate
    argmax, so the compiled fold's ``amax`` and the recurrent ladder's running
    ``maximum`` distribute the incoming gradient differently. The forward
    probability bounds remain identical, which is what the contract covers.
    """
    horizon = 6
    p = Predicate(name="tied")
    formula = Eventually(p, [0, 3])

    outs = []
    for factory in (compile_formula, compile_recurrent_formula):
        v = torch.zeros(horizon, DIM, dtype=torch.float64, requires_grad=True)
        steps = torch.cumsum(torch.tanh(v), dim=0).unsqueeze(0)
        mean = torch.cat([torch.zeros(1, 1, DIM, dtype=torch.float64), steps], dim=1)
        covariance = torch.eye(DIM, dtype=torch.float64).expand(
            1, horizon + 1, DIM, DIM
        ) * 0.25
        traces = gaussian_atom_traces(
            mean,
            covariance,
            [GaussianHalfspace(predicate=p, normal=torch.tensor([1.0, 0.0]), threshold=0.0)],
        )
        assert torch.allclose(
            traces[p.uid][0, :, 0], torch.full((horizon + 1,), 0.5, dtype=torch.float64)
        ), "scenario is meant to be exactly tied"
        out = factory(formula, horizon=horizon)(traces)
        out[..., 0].sum().backward()
        outs.append((out.detach(), v.grad.clone()))

    assert torch.allclose(outs[0][0], outs[1][0], atol=1e-12), "forward values must still agree"


def test_expected_hard_zero_gradient_region_is_unchanged():
    """Where the pre-clamp Frechet lower is negative, P_lower and its gradient are 0.

    This is the exact hard semantics behaving as specified, not a bug: with
    three per-step probabilities of 0.5, ``sum(p) - (n-1) = -0.5 < 0``, the
    clamp in ``frechet_intersection`` pins the lower bound at 0, and no
    gradient reaches ``v``. It is reproduced here deliberately and left as-is
    -- no margin, no smoothing, no surrogate.
    """
    v = torch.zeros(N, D, requires_grad=True)
    # Every step is genuinely stochastic (a non-degenerate initial covariance,
    # so no step takes the deterministic branch) and sits exactly on the
    # threshold, giving p = 0.5 at all three times. Each p is individually
    # differentiable in v; it is the clamp alone that zeroes the gradient.
    recurrent, compiled, traces = _scenario(v, y_min=0.0, q_std=0.3, x0_std=0.3)

    p = next(t[0, :, 0] for t in traces.values() if t[0, :, 0].max().item() > 0.0)
    pre_clamp = p.sum().item() - (p.shape[0] - 1)
    assert pre_clamp < 0.0, f"scenario is not in the clamped region: {pre_clamp}"
    assert p.min().item() > 0.0, "every step must be stochastic, not a hard 0"

    out = recurrent(traces)
    assert out[0, 0, 0].item() == 0.0
    assert torch.allclose(out, compiled(traces), atol=ATOL)

    out[0, 0, 0].backward()
    assert v.grad is not None
    assert torch.all(v.grad == 0.0)

    # The *upper* bound at the same point is unclamped and still carries
    # gradient, confirming the zero is the clamp's doing and not a severed graph.
    v_upper = torch.zeros(N, D, requires_grad=True)
    recurrent_u, _, traces_u = _scenario(v_upper, y_min=0.0, q_std=0.3, x0_std=0.3)
    recurrent_u(traces_u)[0, 0, 1].backward()
    assert torch.any(v_upper.grad != 0.0)


# ---------------------------------------------------------------------------
# Phase 14: the temporal computation really is a recurrent backward scan
# ---------------------------------------------------------------------------


def _temporal_cells(formula, horizon):
    return compile_recurrent_formula(formula, horizon=horizon).temporal_cells


def _run(formula, horizon, bounds=(0.5, 0.7)):
    predicates = {}

    def collect(node):
        if isinstance(node, Predicate):
            predicates[node.uid] = node
        elif isinstance(node, Negation):
            collect(node.subformula)
        elif isinstance(node, (And, Or, Until)):
            collect(node.left)
            collect(node.right)
        else:
            collect(node.subformula)

    collect(formula)
    source = TableProbabilitySource(
        {(p, k): bounds for p in predicates.values() for k in range(horizon + 1)},
        horizon=horizon,
    )
    compiled = compile_recurrent_formula(formula, horizon=horizon)
    compiled(materialize_atom_traces(formula, source, horizon))
    return compiled


def test_forward_time_operators_are_evaluated_by_a_backward_scan():
    a = Predicate(name="A")
    b = Predicate(name="B")
    for formula in [Always(a, [0, 2]), Eventually(a, [1, 3]), Until(a, b, [0, 3])]:
        compiled = _run(formula, horizon=8)
        cells = compiled.temporal_cells
        assert cells, f"{formula} produced no temporal cell"
        for cell in cells:
            order = cell.last_scan_indices
            assert len(order) > 1
            assert all(
                later < earlier for earlier, later in zip(order, order[1:])
            ), f"{formula}: scan order is not strictly decreasing: {order}"
            assert order[0] > order[-1]


def test_temporal_state_is_bounded_and_independent_of_the_horizon():
    a = Predicate(name="A")
    b = Predicate(name="B")
    for formula in [Always(a, [0, 3]), Eventually(a, [1, 4]), Until(a, b, [1, 4])]:
        widths = {
            horizon: compile_recurrent_formula(formula, horizon=horizon).recurrent_state_size
            for horizon in (10, 40, 160)
        }
        assert len(set(widths.values())) == 1, f"{formula}: state grew with the horizon: {widths}"


def test_window_state_width_tracks_the_window_not_the_trace():
    a = Predicate(name="A")
    for width in (1, 4, 9):
        (cell,) = _temporal_cells(Always(a, [0, width]), horizon=50)
        assert cell.state_width == width + 2  # ladder entries 0 .. W


def test_until_shares_prefix_state_recurrently():
    """The prefix ladder is advanced once per time step, not once per candidate."""
    a = Predicate(name="A")
    b = Predicate(name="B")
    horizon = 40
    window = 8
    formula = Until(a, b, [0, window])
    compiled = _run(formula, horizon=horizon)
    (cell,) = compiled.temporal_cells

    valid = horizon - formula.horizon() + 1
    assert cell.n_prefix_updates == valid + window - 1
    # A per-candidate recomputation would touch ~valid * (window+1) prefixes.
    assert cell.n_prefix_updates < valid * (window + 1) / 2
    assert cell.state_width == window + 1


def test_temporal_cell_count_is_one_per_operator_not_one_per_anchor():
    a = Predicate(name="A")
    b = Predicate(name="B")
    formula = And(Always(a, [0, 3]), Until(a, b, [1, 3]))
    for horizon in (12, 60):
        compiled = compile_recurrent_formula(formula, horizon=horizon)
        assert len(compiled.temporal_cells) == 2


def test_no_trainable_parameters_exist():
    a = Predicate(name="A")
    b = Predicate(name="B")
    compiled = _run(Eventually(And(a, Always(b, [1, 3])), [0, 3]), horizon=12)

    assert compiled.parameters() == ()
    for cell in compiled.cells:
        for slot in type(cell).__slots__:
            value = getattr(cell, slot, None)
            assert not isinstance(value, torch.nn.Module)
            if isinstance(value, torch.Tensor):
                assert not value.is_leaf or not value.requires_grad


def test_no_learned_recurrent_machinery_is_used():
    """Check the module's actual code, not its prose, for learned RNN machinery."""
    tree = ast.parse(inspect.getsource(recurrent_module))

    imported: set[str] = set()
    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.Name):
            identifiers.add(node.id)

    assert not any(name.split(".")[0:2] == ["torch", "nn"] for name in imported), imported
    for banned in ("nn", "LSTM", "GRU", "RNN", "Parameter", "Module"):
        assert banned not in identifiers, f"{banned!r} is referenced in pdstl/recurrent.py"


def test_public_api_is_exported():
    import pdstl

    assert pdstl.RecurrentFormula is recurrent_module.RecurrentFormula
    assert pdstl.compile_recurrent_formula is recurrent_module.compile_recurrent_formula
    # Low-level cells and state stay private.
    assert not hasattr(pdstl, "_UntilCell")
    assert not any(name.endswith("Cell") for name in pdstl.__all__)
