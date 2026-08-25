"""Event-identity propagation through logical reductions.

``test_pdstl_semantics`` checks that ``A & A`` and ``A | A`` report ``A``'s
interval. This file checks the stronger property that makes those identities
*compositional*: a reduction that leaves exactly one event returns that
surviving child's event key, not a freshly minted compound key. Without it the
identity holds one level deep and then evaporates -- ``(A & A) & A`` falls back
to generic Frechet, and ``(A & A) & ~A`` no longer looks like a complement.

Singleton temporal windows are the same rule applied to a one-element window.
"""

import pytest

from pdstl import (
    Always,
    Eventually,
    Predicate,
    TableProbabilitySource,
    evaluate,
)
from pdstl.propagate import EvaluationContext


def enclosure(formula, source, time=0):
    """The ``[lower, upper]`` pair of ``formula`` at ``time``, as floats."""
    return evaluate(formula, source)[0, time].tolist()


@pytest.fixture
def a_source():
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, k): (0.6, 0.9) for k in range(3)})
    return a, source


# ---------------------------------------------------------------------------
# Repetition
# ---------------------------------------------------------------------------


def test_repetition_returns_the_surviving_child_key(a_source):
    a, source = a_source
    context = EvaluationContext(source)

    _, atom_key = context._evaluate_with_key(a, 0)

    assert context._evaluate_with_key(a & a, 0)[1] == atom_key
    assert context._evaluate_with_key(a | a, 0)[1] == atom_key


def test_nested_repetition_stays_exact(a_source):
    """``(A ∧ A) ∧ A = A`` and ``(A ∨ A) ∨ A = A``.

    With a compound key on the inner node the outer node would see two
    different events and apply generic Frechet, giving ``[0.2, 0.9]`` for the
    conjunction and ``[0.6, 1.0]`` for the disjunction.
    """
    a, source = a_source

    assert enclosure((a & a) & a, source) == pytest.approx([0.6, 0.9])
    assert enclosure((a | a) | a, source) == pytest.approx([0.6, 0.9])

    assert enclosure((a & a) & a, source) != pytest.approx([0.2, 0.9])
    assert enclosure((a | a) | a, source) != pytest.approx([0.6, 1.0])


def test_deeply_nested_repetition_stays_exact(a_source):
    a, source = a_source

    assert enclosure(((a & a) & (a & a)) & a, source) == pytest.approx([0.6, 0.9])
    assert enclosure(((a | a) | (a | a)) | a, source) == pytest.approx([0.6, 0.9])


def test_mixed_nested_repetition_stays_exact(a_source):
    """The reduction is on events, so the operators may be mixed."""
    a, source = a_source

    assert enclosure((a & a) | a, source) == pytest.approx([0.6, 0.9])
    assert enclosure((a | a) & a, source) == pytest.approx([0.6, 0.9])


# ---------------------------------------------------------------------------
# Complements
# ---------------------------------------------------------------------------


def test_nested_complement_identity_is_still_exact():
    """``(A ∧ A) ∧ ¬A = ∅`` and ``(A ∨ A) ∨ ¬A = Ω``.

    Reachable only because the inner node reports ``A``'s own key, which is
    what the complement check is looking for.
    """
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, 0): (0.4, 0.7)}, horizon=0)

    assert enclosure((a & a) & ~a, source) == pytest.approx([0.0, 0.0])
    assert enclosure((a | a) | ~a, source) == pytest.approx([1.0, 1.0])

    # Both operand orders.
    assert enclosure(~a & (a & a), source) == pytest.approx([0.0, 0.0])
    assert enclosure(~a | (a | a), source) == pytest.approx([1.0, 1.0])

    # What generic Frechet would have reported.
    assert enclosure((a & a) & ~a, source) != pytest.approx([0.0, 0.6])
    assert enclosure((a | a) | ~a, source) != pytest.approx([0.4, 1.0])


def test_complement_of_a_singleton_window_is_exact():
    """``G[0,0] A ∧ ¬A`` is impossible, because ``G[0,0] A`` *is* ``A``."""
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, 0): (0.4, 0.7)}, horizon=0)

    assert enclosure(Always(a, [0, 0]) & ~a, source) == pytest.approx([0.0, 0.0])
    assert enclosure(Eventually(a, [0, 0]) | ~a, source) == pytest.approx([1.0, 1.0])


# ---------------------------------------------------------------------------
# Singleton temporal windows
# ---------------------------------------------------------------------------


def test_singleton_window_preserves_the_child_event_key(a_source):
    """``G[a,a] phi`` and ``F[a,a] phi`` are ``phi`` evaluated at ``k + a``."""
    a, source = a_source
    context = EvaluationContext(source)

    _, shifted_key = context._evaluate_with_key(a, 1)

    assert context._evaluate_with_key(Always(a, [1, 1]), 0)[1] == shifted_key
    assert context._evaluate_with_key(Eventually(a, [1, 1]), 0)[1] == shifted_key


def test_singleton_window_of_a_compound_child_preserves_identity():
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource(
        {(p, k): (0.6, 0.9) for p in (a, b) for k in range(2)}
    )
    context = EvaluationContext(source)

    _, child_key = context._evaluate_with_key(a & b, 1)

    assert context._evaluate_with_key(Always(a & b, [1, 1]), 0)[1] == child_key
    assert context._evaluate_with_key(Eventually(a & b, [1, 1]), 0)[1] == child_key


def test_singleton_windows_collapse_against_their_child(a_source):
    """Identity survives into surrounding Boolean structure."""
    a, source = a_source

    assert enclosure(Always(a, [0, 0]) & a, source) == pytest.approx([0.6, 0.9])
    assert enclosure(Eventually(a, [0, 0]) | a, source) == pytest.approx([0.6, 0.9])
    assert enclosure(
        Always(a, [0, 0]) & Eventually(a, [0, 0]), source
    ) == pytest.approx([0.6, 0.9])

    # A non-singleton window is a genuinely different event and must not
    # collapse: G[0,1] A has lower max(0, 0.6 + 0.6 - 1) = 0.2.
    assert enclosure(Always(a, [0, 1]) & a, source) == pytest.approx([0.0, 0.9])


def test_distinct_singleton_windows_are_not_collapsed(a_source):
    """Only the same event collapses; different offsets stay distinct."""
    a, source = a_source

    phi = Always(a, [0, 0]) & Always(a, [1, 1])
    assert enclosure(phi, source) == pytest.approx([0.2, 0.9])
