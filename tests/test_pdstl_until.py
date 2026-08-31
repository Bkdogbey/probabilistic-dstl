"""Bounded strong-Until semantics and recurrent evaluation."""

import itertools
import random

import pytest
import torch

from pdstl import And, Eventually, OfflineSource, Predicate, Until


def _source(a, b, a_rows, b_rows):
    return OfflineSource(
        {
            a: torch.tensor(a_rows, dtype=torch.float64).unsqueeze(0),
            b: torch.tensor(b_rows, dtype=torch.float64).unsqueeze(0),
        }
    )


@pytest.fixture
def worked_example():
    a = Predicate("A")
    b = Predicate("B")
    source = _source(
        a,
        b,
        [[0.80, 0.90], [0.70, 0.85], [0.60, 0.80]],
        [[0.10, 0.20], [0.80, 0.90], [0.50, 0.70]],
    )
    return a, b, source


def test_until_single_candidate_matches_analytical_intersection(worked_example):
    a, b, source = worked_example

    assert Until(a, b, (1, 1))(source)[0, 0].tolist() == pytest.approx(
        [0.30, 0.85]
    )
    assert Until(a, b, (2, 2))(source)[0, 0].tolist() == pytest.approx(
        [0.00, 0.70]
    )


def test_until_candidate_union_uses_common_prefix_upper_bound(worked_example):
    a, b, source = worked_example

    # Candidate bounds are C1=[.30,.85], C2=[0,.70]. Their union upper would
    # be 1 without recognizing that both candidates require A0 and A1.
    assert Until(a, b, (1, 2))(source)[0, 0].tolist() == pytest.approx(
        [0.30, 0.85]
    )


def test_until_zero_lower_endpoint_still_uses_left_common_prefix(worked_example):
    a, b, source = worked_example

    # C0=[0,.20] and C1=[.30,.85] share A0, so the union upper is capped
    # by P(A0) <= .90 instead of the generic union upper 1.
    assert Until(a, b, (0, 1))(source)[0, 0].tolist() == pytest.approx(
        [0.30, 0.90]
    )


def test_until_zero_zero_requires_both_operands_at_the_witness(worked_example):
    a, b, source = worked_example

    torch.testing.assert_close(Until(a, b, (0, 0))(source), And(a, b)(source))


def test_until_prefix_starts_at_the_anchor_not_at_a():
    a = Predicate("A")
    b = Predicate("B")
    source = _source(
        a,
        b,
        [[0.0, 0.0], [1.0, 1.0]],
        [[0.0, 0.0], [0.9, 0.9]],
    )

    assert Until(a, b, (1, 1))(source)[0, 0].tolist() == pytest.approx([0.0, 0.0])


def test_until_requires_left_at_the_right_event_time():
    a = Predicate("A")
    b = Predicate("B")
    source = _source(
        a,
        b,
        [[1.0, 1.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 1.0]],
    )

    assert Until(a, b, (1, 1))(source)[0, 0].tolist() == pytest.approx([0.0, 0.0])


def test_until_step_matches_every_offline_output(worked_example):
    a, b, source = worked_example
    formula = Until(a, b, (1, 2))
    offline = formula(source)
    state = None
    outputs = []

    for time in range(len(source)):
        output, state = formula.step(
            source.bounds(a, time),
            source.bounds(b, time),
            state,
        )
        if output is not None:
            outputs.append(output)

    torch.testing.assert_close(torch.stack(outputs, dim=1), offline)
    assert state.left.shape == (1, 3, 2)
    assert state.right.shape == (1, 3, 2)


def test_until_incomplete_window_returns_no_output(worked_example):
    a, b, source = worked_example
    formula = Until(a, b, (1, 2))

    output, state = formula.step(source.bounds(a, 0), source.bounds(b, 0))
    assert output is None
    output, state = formula.step(source.bounds(a, 1), source.bounds(b, 1), state)
    assert output is None


def test_until_reproduces_boolean_semantics_for_deterministic_inputs():
    a = Predicate("A")
    b = Predicate("B")

    for a0, a1, a2, b1, b2 in itertools.product((0.0, 1.0), repeat=5):
        source = _source(
            a,
            b,
            [[a0, a0], [a1, a1], [a2, a2]],
            [[0.0, 0.0], [b1, b1], [b2, b2]],
        )
        result = Until(a, b, (1, 2))(source)[0, 0]
        expected = float(
            (bool(a0) and bool(a1) and bool(b1))
            or (bool(a0) and bool(a1) and bool(a2) and bool(b2))
        )

        assert result.tolist() == pytest.approx([expected, expected])


def test_true_until_matches_eventually():
    true = Predicate("true")
    b = Predicate("B")
    source = _source(
        true,
        b,
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        [[0.10, 0.20], [0.80, 0.90], [0.50, 0.70]],
    )

    torch.testing.assert_close(
        Until(true, b, (1, 2))(source),
        Eventually(b, (1, 2))(source),
    )


def test_until_false_right_is_false():
    a = Predicate("A")
    false = Predicate("false")
    source = _source(
        a,
        false,
        [[0.80, 0.90], [0.70, 0.85], [0.60, 0.80]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    )

    result = Until(a, false, (1, 2))(source)
    torch.testing.assert_close(result, torch.zeros_like(result))


def _random_simplex(rng, size):
    weights = [rng.expovariate(1.0) for _ in range(size)]
    total = sum(weights)
    return [weight / total for weight in weights]


def _marginal(joint, outcomes, index):
    return sum(p for outcome, p in zip(outcomes, joint) if outcome[index])


@pytest.mark.parametrize("maximum_width", [0.0, 0.25])
def test_until_bounds_enclose_arbitrary_joint_distributions(maximum_width):
    rng = random.Random(20260828)
    outcomes = list(itertools.product((0, 1), repeat=5))
    a = Predicate("A")
    b = Predicate("B")

    for _ in range(100):
        joint = _random_simplex(rng, len(outcomes))
        a0, a1, a2, b1, b2 = (
            _marginal(joint, outcomes, index) for index in range(5)
        )
        width = rng.uniform(0.0, maximum_width)

        def bounds(probability, delta=width):
            return [
                max(0.0, probability - delta),
                min(1.0, probability + delta),
            ]

        truth = sum(
            probability
            for outcome, probability in zip(outcomes, joint)
            if (outcome[0] and outcome[1] and outcome[3])
            or (outcome[0] and outcome[1] and outcome[2] and outcome[4])
        )
        source = _source(
            a,
            b,
            [bounds(a0), bounds(a1), bounds(a2)],
            [[0.0, 0.0], bounds(b1), bounds(b2)],
        )
        lower, upper = Until(a, b, (1, 2))(source)[0, 0].tolist()

        assert lower <= truth + 1e-12
        assert upper >= truth - 1e-12


def test_smooth_until_approaches_the_hard_result(worked_example):
    a, b, source = worked_example
    formula = Until(a, b, (1, 2))
    hard = formula(source)
    errors = [
        (formula(source, smooth=True, beta=beta) - hard).abs().max().item()
        for beta in (5.0, 20.0, 100.0, 500.0)
    ]

    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 1e-2


def test_smooth_until_backpropagates_to_both_operands():
    a = Predicate("A")
    b = Predicate("B")
    a_trace = torch.tensor(
        [[[0.80, 0.90], [0.70, 0.85], [0.60, 0.80]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    b_trace = torch.tensor(
        [[[0.10, 0.20], [0.80, 0.90], [0.50, 0.70]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    source = OfflineSource({a: a_trace, b: b_trace})

    Until(a, b, (1, 2))(source, smooth=True, beta=20.0).sum().backward()

    assert a_trace.grad is not None
    assert b_trace.grad is not None
    assert torch.isfinite(a_trace.grad).all()
    assert torch.isfinite(b_trace.grad).all()
    assert bool((a_trace.grad != 0).any())
    assert bool((b_trace.grad != 0).any())
