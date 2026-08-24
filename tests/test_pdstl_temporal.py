"""Bounded temporal semantics and the valid evaluation horizon.

Covers requirement groups G (Always), H (Eventually) and J (formula horizon /
valid trace).
"""

import pytest

from pdstl import (
    Always,
    Eventually,
    Predicate,
    TableProbabilitySource,
    evaluate,
)

# ---------------------------------------------------------------------------
# G / H. Always and Eventually
# ---------------------------------------------------------------------------


@pytest.fixture
def two_step_source():
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, 0): (0.9, 0.9), (a, 1): (0.9, 0.9)})
    return a, source


def test_always_over_two_events(two_step_source):
    """lower = max(0, 0.9 + 0.9 - 1) = 0.8, upper = min(0.9, 0.9) = 0.9."""
    a, source = two_step_source
    assert evaluate(Always(a, [0, 1]), source)[0, 0].tolist() == pytest.approx(
        [0.8, 0.9]
    )


def test_eventually_over_two_events(two_step_source):
    """lower = max(0.9, 0.9) = 0.9, upper = min(1, 0.9 + 0.9) = 1.0."""
    a, source = two_step_source
    assert evaluate(Eventually(a, [0, 1]), source)[0, 0].tolist() == pytest.approx(
        [0.9, 1.0]
    )


def test_always_does_not_assume_temporal_independence(two_step_source):
    """An independence assumption would give 0.9 * 0.9 = 0.81, not 0.8."""
    a, source = two_step_source
    lower = evaluate(Always(a, [0, 1]), source)[0, 0, 0].item()
    assert lower == pytest.approx(0.8)
    assert lower != pytest.approx(0.81)


def test_always_lower_bound_saturates_at_zero():
    """With three weak events the union bound gives nothing, not a negative."""
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, k): (0.5, 0.6) for k in range(3)})

    # max(0, 1.5 - 2) = 0, min upper = 0.6
    assert evaluate(Always(a, [0, 2]), source)[0, 0].tolist() == pytest.approx(
        [0.0, 0.6]
    )


def test_eventually_upper_bound_saturates_at_one():
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, k): (0.2, 0.6) for k in range(3)})

    # max lower = 0.2, min(1, 1.8) = 1.0
    assert evaluate(Eventually(a, [0, 2]), source)[0, 0].tolist() == pytest.approx(
        [0.2, 1.0]
    )


def test_singleton_interval_is_the_atom_itself():
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, 0): (0.35, 0.85)}, horizon=0)

    assert evaluate(Always(a, [0, 0]), source)[0, 0].tolist() == pytest.approx(
        [0.35, 0.85]
    )
    assert evaluate(Eventually(a, [0, 0]), source)[0, 0].tolist() == pytest.approx(
        [0.35, 0.85]
    )


def test_offset_interval_reads_the_shifted_window():
    a = Predicate(name="A")
    source = TableProbabilitySource(
        {(a, 0): (0.0, 0.0), (a, 1): (0.9, 0.9), (a, 2): (0.9, 0.9)}
    )

    # G[1,2] at t=0 reads times 1 and 2 only, so the zero at t=0 is ignored.
    assert evaluate(Always(a, [1, 2]), source)[0, 0].tolist() == pytest.approx(
        [0.8, 0.9]
    )


# ---------------------------------------------------------------------------
# J. Formula horizon and the valid evaluation trace
# ---------------------------------------------------------------------------


def test_horizon_algebra():
    a = Predicate(name="A")
    b = Predicate(name="B")

    assert a.horizon() == 0
    assert (~a).horizon() == 0
    assert Always(a, [0, 2]).horizon() == 2
    assert Eventually(a, [1, 3]).horizon() == 3
    assert (Always(a, [0, 2]) & Eventually(b, [0, 4])).horizon() == 4
    assert (Always(a, [0, 2]) | Eventually(b, [0, 1])).horizon() == 2

    # Nesting adds the inner lookahead to the outer upper endpoint.
    assert Always(Eventually(a, [0, 2]), [0, 1]).horizon() == 3
    assert Eventually(Always(a, [1, 2]), [2, 3]).horizon() == 5


def test_valid_trace_length_matches_the_horizon():
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, k): (0.9, 0.95) for k in range(6)})
    assert source.horizon == 5

    for formula in [
        a,
        Always(a, [0, 1]),
        Eventually(a, [0, 3]),
        Always(Eventually(a, [0, 2]), [0, 1]),
    ]:
        trace = evaluate(formula, source)
        expected = source.horizon - formula.horizon() + 1
        assert trace.shape == (1, expected, 2), f"{formula} produced {trace.shape}"


def test_out_of_range_times_are_never_queried():
    """No fabricated tail: the source is only ever asked for times it covers."""
    queried = []

    class RecordingSource(TableProbabilitySource):
        def bounds(self, predicate, time):
            queried.append(time)
            return super().bounds(predicate, time)

    a = Predicate(name="A")
    source = RecordingSource({(a, k): (0.9, 0.95) for k in range(4)})

    evaluate(Always(a, [0, 2]), source)

    assert max(queried) <= source.horizon
    assert sorted(set(queried)) == [0, 1, 2, 3]


def test_source_too_short_for_the_formula_is_an_error():
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, 0): (0.9, 0.9), (a, 1): (0.9, 0.9)})

    # H(G[0,2]) = 2 needs times 0..2, but the source stops at 1.
    with pytest.raises(ValueError, match="too short"):
        evaluate(Always(a, [0, 5]), source)


def test_exactly_saturating_horizon_yields_a_single_time_step():
    a = Predicate(name="A")
    source = TableProbabilitySource({(a, k): (0.9, 0.95) for k in range(3)})

    trace = evaluate(Always(a, [0, 2]), source)
    assert trace.shape == (1, 1, 2)


def test_nested_temporal_trace_values_are_correct():
    """G[0,1] F[0,1] A over a source where the inner windows differ."""
    a = Predicate(name="A")
    source = TableProbabilitySource(
        {
            (a, 0): (0.0, 0.0),
            (a, 1): (0.5, 0.5),
            (a, 2): (0.5, 0.5),
        }
    )

    inner = Eventually(a, [0, 1])
    # F[0,1] at t=0 reads {0, 1}: lower max(0, 0.5)=0.5, upper min(1, 0.5)=0.5
    # F[0,1] at t=1 reads {1, 2}: lower max(0.5, 0.5)=0.5, upper min(1, 1.0)=1.0
    inner_trace = evaluate(inner, source)
    assert inner_trace[0, 0].tolist() == pytest.approx([0.5, 0.5])
    assert inner_trace[0, 1].tolist() == pytest.approx([0.5, 1.0])

    # G[0,1] over those two: lower max(0, 0.5+0.5-1)=0, upper min(0.5, 1.0)=0.5
    outer = Always(inner, [0, 1])
    assert outer.horizon() == 2
    outer_trace = evaluate(outer, source)
    assert outer_trace.shape == (1, 1, 2)
    assert outer_trace[0, 0].tolist() == pytest.approx([0.0, 0.5])
