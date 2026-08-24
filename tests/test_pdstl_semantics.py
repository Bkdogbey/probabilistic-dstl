"""Boolean semantics: negation, Frechet bounds, event identities, and the
degenerate deterministic case.

Covers requirement groups C, D, E, F and I.
"""

import itertools
import random

import pytest

from pdstl import (
    Always,
    Eventually,
    Predicate,
    TableProbabilitySource,
    evaluate,
)


def enclosure(formula, source, time=0):
    """The ``[lower, upper]`` pair of ``formula`` at ``time``, as floats."""
    return evaluate(formula, source)[0, time].tolist()


# ---------------------------------------------------------------------------
# C. Negation
# ---------------------------------------------------------------------------


def test_negation_flips_and_complements_the_interval():
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, 0): (0.2, 0.7)}, horizon=0)

    assert enclosure(a, source) == pytest.approx([0.2, 0.7])
    assert enclosure(~a, source) == pytest.approx([0.3, 0.8])


def test_double_negation_is_not_silently_defined():
    """Negation is restricted to predicates; NNF is the intended discipline."""
    a = Predicate(name="A")
    with pytest.raises(TypeError, match="negation normal form"):
        _ = ~(~a)
    with pytest.raises(TypeError, match="negation normal form"):
        _ = ~Always(a, [0, 1])


# ---------------------------------------------------------------------------
# D / E. Conjunction and disjunction
# ---------------------------------------------------------------------------


@pytest.fixture
def ab_source():
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource(
        {(a, 0): (0.6, 0.9), (b, 0): (0.7, 0.95)}, horizon=0
    )
    return a, b, source


def test_conjunction_uses_frechet_bounds(ab_source):
    """lower = max(0, 0.6 + 0.7 - 1) = 0.3, upper = min(0.9, 0.95) = 0.9."""
    a, b, source = ab_source
    assert enclosure(a & b, source) == pytest.approx([0.3, 0.9])


def test_disjunction_uses_frechet_bounds(ab_source):
    """lower = max(0.6, 0.7) = 0.7, upper = min(1, 0.9 + 0.95) = 1.0."""
    a, b, source = ab_source
    assert enclosure(a | b, source) == pytest.approx([0.7, 1.0])


def test_conjunction_lower_bound_is_not_the_product(ab_source):
    """Guards against reintroducing the independence assumption.

    A product lower bound would give 0.6 * 0.7 = 0.42, which is not sound
    without independence. Frechet gives 0.3.
    """
    a, b, source = ab_source
    assert enclosure(a & b, source)[0] == pytest.approx(0.3)


def test_binary_operators_are_commutative(ab_source):
    a, b, source = ab_source
    assert enclosure(a & b, source) == pytest.approx(enclosure(b & a, source))
    assert enclosure(a | b, source) == pytest.approx(enclosure(b | a, source))


def test_commutative_operands_share_one_event_key(ab_source):
    """``A & B`` and ``B & A`` are the same event, so they share a cache entry."""
    from pdstl.propagate import PropagationContext

    a, b, source = ab_source
    context = PropagationContext(source)

    _, key_ab = context._eval_keyed(a & b, 0)
    _, key_ba = context._eval_keyed(b & a, 0)

    assert key_ab == key_ba


# ---------------------------------------------------------------------------
# F. Event identity
# ---------------------------------------------------------------------------


def test_repetition_identity_returns_the_original_interval():
    """``A & A = A`` and ``A | A = A``, exactly -- not weakened by Frechet."""
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, 0): (0.6, 0.9)}, horizon=0)

    assert enclosure(a & a, source) == pytest.approx([0.6, 0.9])
    assert enclosure(a | a, source) == pytest.approx([0.6, 0.9])

    # Blind Frechet would have given these weaker intervals.
    assert enclosure(a & a, source) != pytest.approx([0.2, 0.9])
    assert enclosure(a | a, source) != pytest.approx([0.6, 1.0])


def test_complement_identity_is_exact():
    """``A & ~A`` is impossible and ``A | ~A`` is certain.

    Generic Frechet cannot recover this from interval marginals alone: with
    ``A = [0.4, 0.7]`` and ``~A = [0.3, 0.6]`` it reports ``[0, 0.6]`` and
    ``[0.4, 1.0]``, putting positive upper probability on an impossible event.
    """
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, 0): (0.4, 0.7)}, horizon=0)

    assert enclosure(~a, source) == pytest.approx([0.3, 0.6])
    assert enclosure(a & ~a, source) == pytest.approx([0.0, 0.0])
    assert enclosure(a | ~a, source) == pytest.approx([1.0, 1.0])

    # The values blind Frechet would have produced.
    assert enclosure(a & ~a, source) != pytest.approx([0.0, 0.6])
    assert enclosure(a | ~a, source) != pytest.approx([0.4, 1.0])


def test_complement_identity_holds_in_both_operand_orders():
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, 0): (0.4, 0.7)}, horizon=0)

    assert enclosure(~a & a, source) == pytest.approx([0.0, 0.0])
    assert enclosure(~a | a, source) == pytest.approx([1.0, 1.0])


def test_identities_apply_to_compound_events_too():
    """Identity is keyed on the event, not on the object."""
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, k): (0.9, 0.95) for k in range(2)})

    g1 = Always(a, [0, 1])
    g2 = Always(a, [0, 1])  # structurally identical, different object

    assert enclosure(g1, source) == pytest.approx([0.8, 0.95])
    assert enclosure(g1 & g2, source) == pytest.approx([0.8, 0.95])
    assert enclosure(g1 | g2, source) == pytest.approx([0.8, 0.95])


def test_distinct_events_are_not_collapsed():
    """Only exact repetition collapses; different times stay independent events."""
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, k): (0.6, 0.9) for k in range(2)})

    # G[0,0] A reads time 0; G[1,1] A reads time 1. Different events.
    phi = Always(a, [0, 0]) & Always(a, [1, 1])
    assert enclosure(phi, source) == pytest.approx([0.2, 0.9])


# ---------------------------------------------------------------------------
# I. Deterministic consistency with ordinary Boolean STL
# ---------------------------------------------------------------------------


def boolean_stl(formula, signal, time):
    """Reference ordinary (non-probabilistic) discrete-time STL evaluator.

    ``signal`` maps ``predicate.uid -> list of 0/1 truth values``. This is the
    textbook semantics, written independently of the pdSTL implementation so
    that agreement is evidence rather than tautology.
    """
    from pdstl.operators import And, Negation, Or, TemporalOperator
    from pdstl.operators import Predicate as P

    if isinstance(formula, P):
        return bool(signal[formula.uid][time])
    if isinstance(formula, Negation):
        return not boolean_stl(formula.subformula, signal, time)
    if isinstance(formula, And):
        return boolean_stl(formula.left, signal, time) and boolean_stl(
            formula.right, signal, time
        )
    if isinstance(formula, Or):
        return boolean_stl(formula.left, signal, time) or boolean_stl(
            formula.right, signal, time
        )
    if isinstance(formula, TemporalOperator):
        window = range(time + formula.a, time + formula.b + 1)
        results = [boolean_stl(formula.subformula, signal, k) for k in window]
        return all(results) if isinstance(formula, Always) else any(results)
    raise TypeError(type(formula).__name__)


def test_zero_one_probabilities_reproduce_boolean_stl():
    """With 0/1 atoms the bounds must collapse onto ordinary Boolean STL."""
    rng = random.Random(20260824)
    horizon = 5

    a = Predicate(name="A")
    b = Predicate(name="B")

    formulas = [
        a,
        ~a,
        a & b,
        a | b,
        Always(a, [0, 2]),
        Eventually(a, [0, 2]),
        Always(a & b, [1, 3]),
        Eventually(~a | b, [0, 2]),
        Always(Eventually(a, [0, 1]), [0, 2]),
        Always(a, [0, 1]) & Eventually(b, [0, 2]),
    ]

    for _ in range(40):
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
            trace = evaluate(formula, source)
            for time in range(trace.shape[1]):
                lower, upper = trace[0, time].tolist()
                expected = float(boolean_stl(formula, signal, time))

                assert lower == pytest.approx(upper), (
                    f"{formula} at t={time} is not degenerate: [{lower}, {upper}]"
                )
                assert lower == pytest.approx(expected), (
                    f"{formula} at t={time}: pdSTL {lower} vs Boolean {expected}"
                )


def test_deterministic_case_covers_both_truth_values():
    """Sanity check that the reference comparison is not vacuous."""
    a = Predicate(name="A")
    seen = set()
    for bits in itertools.product([0.0, 1.0], repeat=3):
        source = TableProbabilitySource(
            {(a, k): (bit, bit) for k, bit in enumerate(bits)}
        )
        seen.add(tuple(evaluate(Always(a, [0, 2]), source)[0, 0].tolist()))
        seen.add(tuple(evaluate(Eventually(a, [0, 2]), source)[0, 0].tolist()))
    assert seen == {(0.0, 0.0), (1.0, 1.0)}
