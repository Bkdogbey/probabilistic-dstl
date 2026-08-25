"""The compiled hard-probability execution backend, ``pdstl.graph``.

The other test files check the reference interpreter (``propagate.evaluate``)
against hand calculations and soundness. This file checks that the compiled
backend (``compile_formula`` / ``CompiledFormula`` / ``materialize_atom_traces``)
reproduces the reference *exactly*, over the complete valid trace, for every
supported operator -- including the event-identity reductions (repetition,
complement, singleton-window and Until-edge-case collapse) that make the
difference between a sound enclosure and the exact reference value.
"""

import itertools
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
from pdstl.graph import compile_formula, materialize_atom_traces

SEED = 20260824
N_TRIALS = 200


def compiled_trace(formula, source, horizon=None):
    """The compiled backend's full trace, materializing atoms from ``source``."""
    if horizon is None:
        horizon = source.horizon
    compiled = compile_formula(formula, horizon=horizon)
    traces = materialize_atom_traces(formula, source, horizon)
    return compiled(traces)


def compiled_enclosure(formula, source, horizon=None, time=0):
    return compiled_trace(formula, source, horizon)[0, time].tolist()


def assert_matches_reference(formula, source, horizon=None):
    """The compiled backend and the reference interpreter must agree exactly."""
    if horizon is None:
        horizon = source.horizon
    ref = evaluate(formula, source)
    out = compiled_trace(formula, source, horizon)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-6), f"{formula}\nref={ref}\nout={out}"
    return out


@pytest.fixture
def worked_example():
    """A0 = [.80, .90], A1 = [.70, .85], A2=[.90,.95], B0=[.60,.75], B1 = [.80, .90], B2 = [.50, .70]."""
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


# ---------------------------------------------------------------------------
# Primitive / formula equivalence, over the complete trace
# ---------------------------------------------------------------------------


def test_atom_matches_reference(worked_example):
    a, _, source = worked_example
    assert_matches_reference(a, source)


def test_negation_matches_reference(worked_example):
    a, _, source = worked_example
    out = assert_matches_reference(Negation(a), source)
    assert out[0, 0].tolist() == pytest.approx([0.10, 0.20])


def test_and_matches_reference(worked_example):
    a, b, source = worked_example
    out = assert_matches_reference(And(a, b), source)
    assert out[0, 0].tolist() == pytest.approx([0.40, 0.75])


def test_or_matches_reference(worked_example):
    a, b, source = worked_example
    out = assert_matches_reference(Or(a, b), source)
    assert out[0, 0].tolist() == pytest.approx([0.80, 1.00])


def test_always_matches_reference(worked_example):
    a, _, source = worked_example
    out = assert_matches_reference(Always(a, interval=[0, 2]), source)
    assert out[0, 0].tolist() == pytest.approx([0.40, 0.85])


def test_eventually_matches_reference(worked_example):
    a, _, source = worked_example
    out = assert_matches_reference(Eventually(a, interval=[0, 2]), source)
    assert out[0, 0].tolist() == pytest.approx([0.90, 1.00])


def test_nested_boolean_temporal_matches_reference(worked_example):
    a, b, source = worked_example
    out = assert_matches_reference(Always(And(a, b), interval=[0, 1]), source)
    assert out[0, 0].tolist() == pytest.approx([0.0, 0.75])


def test_nested_temporal_matches_reference(worked_example):
    a, b, source = worked_example
    assert_matches_reference(Eventually(Or(a, Always(b, [0, 1])), interval=[0, 1]), source)


# ---------------------------------------------------------------------------
# Until: analytical worked examples (also checked against the reference)
# ---------------------------------------------------------------------------


def test_until_11_matches_the_analytical_interval(worked_example):
    a, b, source = worked_example
    out = assert_matches_reference(Until(a, b, [1, 1]), source)
    assert out[0, 0].tolist() == pytest.approx([0.60, 0.90])


def test_until_12_matches_the_tightened_interval(worked_example):
    a, b, source = worked_example
    out = assert_matches_reference(Until(a, b, [1, 2]), source)
    assert out[0, 0].tolist() == pytest.approx([0.60, 0.90])
    assert out[0, 0].tolist() != pytest.approx([0.60, 1.00])


def test_until_22_matches_the_analytical_interval(worked_example):
    a, b, source = worked_example
    out = assert_matches_reference(Until(a, b, [2, 2]), source)
    assert out[0, 0].tolist() == pytest.approx([0.0, 0.70])


def test_until_00_is_the_right_operand():
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource(
        {(a, 0): (0.50, 0.50), (b, 0): (0.30, 0.60)}, horizon=0
    )
    out = assert_matches_reference(Until(a, b, [0, 0]), source)
    assert out[0, 0].tolist() == pytest.approx([0.30, 0.60])


def test_until_01_needs_no_tightening():
    a = Predicate(name="A")
    b = Predicate(name="B")
    # `materialize_atom_traces` eagerly fetches every predicate over the full
    # 0..horizon range (see graph.py's module docstring / task spec section
    # on atomic trace materialization), unlike `evaluate`'s lazy per-formula
    # querying -- so unlike test_pdstl_until.py's sparse table, every
    # predicate needs an entry at every time up to the source horizon, even
    # times this particular formula does not read.
    source = TableProbabilitySource(
        {
            (a, 0): (0.50, 0.60),
            (a, 1): (0.0, 1.0),  # unread by U[0,1], present only for materialization
            (b, 0): (0.10, 0.20),
            (b, 1): (0.30, 0.40),
        }
    )
    out = assert_matches_reference(Until(a, b, [0, 1]), source)
    assert out[0, 0].tolist() == pytest.approx([0.10, 0.60])


def test_until_matches_reference_over_the_full_trace():
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource(
        {(p, k): (0.5, 0.7) for p in (a, b) for k in range(6)}
    )
    for formula in [
        Until(a, b, [0, 0]),
        Until(a, b, [1, 2]),
        Until(a, b, [0, 3]),
        Until(Always(a, [0, 1]), b, [1, 2]),
        Until(a, Eventually(b, [0, 1]), [2, 4]),
    ]:
        assert_matches_reference(formula, source)


# ---------------------------------------------------------------------------
# Identity / dedup / complement -- numeric correctness, not just performance
# ---------------------------------------------------------------------------


@pytest.fixture
def a_source():
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, k): (0.6, 0.9) for k in range(3)})
    return a, source


def test_repeated_conjunction_and_disjunction_preserve_identity(a_source):
    a, source = a_source
    assert compiled_enclosure(a & a, source) == pytest.approx([0.6, 0.9])
    assert compiled_enclosure(a | a, source) == pytest.approx([0.6, 0.9])


def test_nested_repetition_stays_exact(a_source):
    """``(A ∧ A) ∧ A = A``, not the naive Frechet fold of three copies.

    Naive Frechet over three identical [0.6,0.9] intervals would give lower
    max(0, 1.8-2)=0, i.e. [0.0, 0.9] for the conjunction -- wrong.
    """
    a, source = a_source
    assert compiled_enclosure((a & a) & a, source) == pytest.approx([0.6, 0.9])
    assert compiled_enclosure((a | a) | a, source) == pytest.approx([0.6, 0.9])
    assert compiled_enclosure((a & a) & a, source) != pytest.approx([0.0, 0.9])
    assert_matches_reference((a & a) & a, source)
    assert_matches_reference((a | a) | a, source)


def test_exact_zero_lower_bound_counterexample():
    """The headline correctness case: repeated identical intervals must not
    collapse toward the naive (wrong) Frechet fold."""
    a = Predicate(name="a")
    source = TableProbabilitySource({(a, 0): (0.5, 0.5)}, horizon=0)

    out = compiled_enclosure((a & a) & a, source)
    assert out == pytest.approx([0.5, 0.5])
    assert out != pytest.approx([0.0, 0.5])


def test_nested_complement_identity_is_exact():
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, 0): (0.4, 0.7)}, horizon=0)

    assert compiled_enclosure((a & a) & ~a, source) == pytest.approx([0.0, 0.0])
    assert compiled_enclosure((a | a) | ~a, source) == pytest.approx([1.0, 1.0])
    assert compiled_enclosure(~a & (a & a), source) == pytest.approx([0.0, 0.0])
    assert compiled_enclosure(~a | (a | a), source) == pytest.approx([1.0, 1.0])
    assert compiled_enclosure((a & a) & ~a, source) != pytest.approx([0.0, 0.6])


def test_singleton_window_preserves_identity(a_source):
    a, source = a_source
    assert compiled_enclosure(Always(a, [0, 0]) & a, source) == pytest.approx([0.6, 0.9])
    assert compiled_enclosure(Eventually(a, [0, 0]) | a, source) == pytest.approx([0.6, 0.9])
    assert compiled_enclosure(
        Always(a, [0, 0]) & Eventually(a, [0, 0]), source
    ) == pytest.approx([0.6, 0.9])


def test_distinct_singleton_windows_are_not_collapsed(a_source):
    """Different offsets must not collapse; the generic Frechet fold applies."""
    a, source = a_source
    phi = Always(a, [0, 0]) & Always(a, [1, 1])
    assert compiled_enclosure(phi, source) == pytest.approx([0.2, 0.9])
    assert_matches_reference(phi, source)


def test_until_00_identity_survives_into_surrounding_structure():
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource(
        {(a, 0): (0.50, 0.50), (b, 0): (0.30, 0.60)}, horizon=0
    )
    assert compiled_enclosure(Until(a, b, [0, 0]) & b, source) == pytest.approx([0.30, 0.60])
    assert compiled_enclosure(Until(a, b, [0, 0]) & ~b, source) == pytest.approx([0.0, 0.0])


# ---------------------------------------------------------------------------
# DAG sharing: node count stays bounded rather than growing tree-like
# ---------------------------------------------------------------------------


def test_repeated_subexpression_is_compiled_once():
    a = Predicate(name="A")
    b = Predicate(name="B")
    shared = Always(And(a, b), [0, 2])
    formula = (shared & shared) & (shared | shared)

    baseline = compile_formula(shared, horizon=4)
    wrapped = compile_formula(formula, horizon=4)
    # `shared` occurs 4 times syntactically. Every And/Or combining it with
    # itself collapses by identity (same event, same key) to `shared`'s own
    # node, so wrapping it in redundant Boolean structure must not add a
    # single node over compiling `shared` alone.
    assert wrapped.n_nodes == baseline.n_nodes


def test_until_prefix_is_shared_across_candidates():
    """The left-prefix state should be built once and reused, not recomputed."""
    a = Predicate(name="A")
    b = Predicate(name="B")
    formula = Until(a, b, [0, 5])
    compiled = compile_formula(formula, horizon=5)
    # atom(A) + atom(B) + 5 prefix-conjunction folds (j=1..5, j=0 aliases B)
    # + 1 union fold = 8, not the O(W^2) node blowup a naive implementation
    # (recomputing each prefix from scratch per candidate) would produce.
    assert compiled.n_nodes <= 8


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_compile_formula_rejects_too_short_horizon():
    a = Predicate(name="A")
    with pytest.raises(ValueError, match="too short"):
        compile_formula(Always(a, [0, 4]), horizon=2)


def test_compiled_call_rejects_missing_atom_trace():
    a = Predicate(name="A")
    b = Predicate(name="B")
    compiled = compile_formula(And(a, b), horizon=0)
    only_a = {a.uid: torch.tensor([[[0.5, 0.5]]])}
    with pytest.raises(KeyError):
        compiled(only_a)


def test_compiled_call_rejects_wrong_shape():
    a = Predicate(name="A")
    compiled = compile_formula(a, horizon=1)
    wrong = {a.uid: torch.tensor([[0.5, 0.5]])}  # missing the time axis
    with pytest.raises(ValueError):
        compiled(wrong)


def test_compiled_call_rejects_inconsistent_batch_size():
    a = Predicate(name="A")
    b = Predicate(name="B")
    compiled = compile_formula(And(a, b), horizon=0)
    mismatched = {
        a.uid: torch.tensor([[[0.5, 0.5]]]),
        b.uid: torch.tensor([[[0.5, 0.5]], [[0.4, 0.6]]]),
    }
    with pytest.raises(ValueError, match="inconsistent batch size"):
        compiled(mismatched)


def test_compiled_call_rejects_invalid_bounds():
    a = Predicate(name="A")
    compiled = compile_formula(a, horizon=0)
    invalid = {a.uid: torch.tensor([[[0.7, 0.3]]])}  # lower > upper
    with pytest.raises(ValueError):
        compiled(invalid)


# ---------------------------------------------------------------------------
# Randomized differential testing against propagate.evaluate
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
    if kind == "always":
        lo = rng.randint(0, max_window)
        hi = rng.randint(lo, max_window)
        return Always(_random_formula(rng, preds, depth - 1), interval=[lo, hi])
    if kind == "eventually":
        lo = rng.randint(0, max_window)
        hi = rng.randint(lo, max_window)
        return Eventually(_random_formula(rng, preds, depth - 1), interval=[lo, hi])
    lo = rng.randint(0, max_window)
    hi = rng.randint(lo, max_window)
    return Until(
        _random_formula(rng, preds, depth - 1),
        _random_formula(rng, preds, depth - 1),
        [lo, hi],
    )


def _random_source(rng, preds, horizon, batch):
    table = {}
    for p in preds:
        for t in range(horizon + 1):
            lo = round(rng.uniform(0.0, 0.9), 3)
            hi = round(rng.uniform(lo, 1.0), 3)
            table[(p, t)] = (lo, hi)
    source = TableProbabilitySource(table, horizon=horizon)
    if batch > 1:
        # Repeat each predicate's row to synthesize a batched source.
        return _BatchedTable(table, horizon, batch)
    return source


class _BatchedTable(TableProbabilitySource):
    """A TableProbabilitySource whose bounds are repeated across a batch."""

    def __init__(self, table, horizon, batch):
        super().__init__(table, horizon=horizon)
        self._repeat = batch

    def bounds(self, predicate, time):
        row = super().bounds(predicate, time)
        return row.repeat(self._repeat, 1)


def test_randomized_differential_equivalence():
    rng = random.Random(SEED)
    horizon = 12
    failures = []

    trials = 0
    while trials < 150:
        preds = [Predicate(name=f"p{trials}_{j}") for j in range(3)]
        formula = _random_formula(rng, preds, depth=4)
        batch = rng.choice([1, 1, 1, 3])
        source = _random_source(rng, preds, horizon, batch)

        try:
            required = formula.horizon()
        except Exception:
            continue
        if horizon - required + 1 <= 0:
            continue
        trials += 1

        ref = evaluate(formula, source)
        out = compiled_trace(formula, source, horizon)
        if out.shape != ref.shape or not torch.allclose(out, ref, atol=1e-5):
            failures.append((formula, ref, out))

    assert not failures, f"{len(failures)} of {trials} random formulas mismatched"


# ---------------------------------------------------------------------------
# Deterministic 0/1 inputs reproduce ordinary Boolean STL
# ---------------------------------------------------------------------------


def boolean_stl(formula, signal, time):
    """Independent reference Boolean STL semantics; see test_pdstl_until.py."""
    from pdstl.operators import And as A_
    from pdstl.operators import Negation as N_
    from pdstl.operators import Or as O_
    from pdstl.operators import Predicate as P_
    from pdstl.operators import TemporalOperator

    if isinstance(formula, P_):
        return bool(signal[formula.uid][time])
    if isinstance(formula, N_):
        return not boolean_stl(formula.subformula, signal, time)
    if isinstance(formula, A_):
        return boolean_stl(formula.left, signal, time) and boolean_stl(formula.right, signal, time)
    if isinstance(formula, O_):
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
    ]

    for _ in range(30):
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
            out = compiled_trace(formula, source, horizon)
            for time in range(out.shape[1]):
                lower, upper = out[0, time].tolist()
                expected = float(boolean_stl(formula, signal, time))
                assert lower == pytest.approx(upper)
                assert lower == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Soundness against explicit joint distributions (reused fixtures)
# ---------------------------------------------------------------------------


def random_simplex(rng, n):
    weights = [rng.expovariate(1.0) for _ in range(n)]
    total = sum(weights)
    return [w / total for w in weights]


def marginal(joint, outcomes, index):
    return sum(p for outcome, p in zip(outcomes, joint) if outcome[index] == 1)


def assert_encloses(interval, truth, label):
    lower, upper = interval
    assert lower <= truth + 1e-6, f"{label}: lower {lower} exceeds truth {truth}"
    assert upper >= truth - 1e-6, f"{label}: upper {upper} below truth {truth}"


def test_compiled_and_or_enclose_the_true_probability():
    rng = random.Random(SEED + 2)
    outcomes = list(itertools.product([0, 1], repeat=2))

    a = Predicate(name="A")
    b = Predicate(name="B")

    for _ in range(N_TRIALS):
        joint = random_simplex(rng, 4)
        p_a = marginal(joint, outcomes, 0)
        p_b = marginal(joint, outcomes, 1)
        p_and = sum(p for o, p in zip(outcomes, joint) if o[0] == 1 and o[1] == 1)
        p_or = sum(p for o, p in zip(outcomes, joint) if o[0] == 1 or o[1] == 1)

        source = TableProbabilitySource({(a, 0): (p_a, p_a), (b, 0): (p_b, p_b)}, horizon=0)

        assert_encloses(compiled_enclosure(a & b, source), p_and, "A and B")
        assert_encloses(compiled_enclosure(a | b, source), p_or, "A or B")


def test_compiled_until_encloses_the_true_probability_under_dependence():
    """Reuses the Until[1,2] joint-distribution scenario from test_pdstl_until.py."""
    rng = random.Random(SEED + 3)
    outcomes = list(itertools.product([0, 1], repeat=4))

    a = Predicate(name="A")
    b = Predicate(name="B")

    for _ in range(N_TRIALS):
        joint = random_simplex(rng, 16)
        a0, a1, b1, b2 = (marginal(joint, outcomes, i) for i in range(4))
        truth = sum(
            p
            for o, p in zip(outcomes, joint)
            if (o[2] == 1 and o[0] == 1) or (o[3] == 1 and o[0] == 1 and o[1] == 1)
        )

        source = TableProbabilitySource(
            {
                (a, 0): (a0, a0),
                (a, 1): (a1, a1),
                (a, 2): (0.0, 1.0),  # unread by U[1,2]; present for materialization
                (b, 0): (0.0, 1.0),  # unread by U[1,2]; present for materialization
                (b, 1): (b1, b1),
                (b, 2): (b2, b2),
            },
            horizon=2,
        )

        assert_encloses(compiled_enclosure(Until(a, b, [1, 2]), source), truth, "A U[1,2] B")
