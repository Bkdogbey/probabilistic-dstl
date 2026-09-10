"""GaussianBelief: X = x* + E, x* in [lower, upper], E ~ N(0, covariance).

Kept small: analytic agreement, the zero-variance boundary, and the
structural guarantees that matter -- a tighter bound gives a tighter
probability interval, and a collapsed bound recovers the exact Gaussian.
"""

import pytest
import torch
from scipy.stats import norm

from models.dynamics import GaussianBelief, create_gaussian_belief_trajectory
from pdstl.operators import Always, GreaterThan, LessThan


def belief(lower, upper, variance):
    """One-step, one-dimensional belief from plain floats, shape [1,1]."""
    t = lambda v: torch.as_tensor([[v]], dtype=torch.float64)
    return GaussianBelief(t(lower), t(upper), t(variance))


@pytest.mark.parametrize("sense", [">=", "<="])
def test_endpoints_match_the_analytic_tail_probability(sense):
    m_lower, m_upper, sigma, c = 48.0, 52.0, 2.0, 50.0
    predicate = GreaterThan(c) if sense == ">=" else LessThan(c)

    got = belief(m_lower, m_upper, sigma**2).probability_bounds(predicate)[0]

    if sense == ">=":
        expected = (norm.cdf((m_lower - c) / sigma), norm.cdf((m_upper - c) / sigma))
    else:
        expected = (norm.cdf((c - m_upper) / sigma), norm.cdf((c - m_lower) / sigma))

    assert got[0].item() == pytest.approx(expected[0], abs=1e-12)
    assert got[1].item() == pytest.approx(expected[1], abs=1e-12)
    assert got[0] <= got[1]


def test_pessimism_flips_direction_with_the_sense():
    """The pessimistic (lower) bound for X >= c is the optimistic bound for
    X <= c, so the two intervals are reflections of one another."""
    traj_bounds = (45.0, 49.0, 4.0)  # asymmetric around the threshold

    gt = belief(*traj_bounds).probability_bounds(GreaterThan(50.0))[0]
    lt = belief(*traj_bounds).probability_bounds(LessThan(50.0))[0]

    assert lt[0].item() == pytest.approx(1.0 - gt[1].item(), abs=1e-9)
    assert lt[1].item() == pytest.approx(1.0 - gt[0].item(), abs=1e-9)


@pytest.mark.parametrize(
    "m_lower, m_upper, expect",
    [
        (51.0, 52.0, (1.0, 1.0)),  # entirely above the threshold
        (48.0, 49.0, (0.0, 0.0)),  # entirely below
        (49.0, 51.0, (0.0, 1.0)),  # straddles it
        (50.0, 50.0, (1.0, 1.0)),  # exactly at it, inclusive
    ],
)
def test_zero_variance_is_an_inclusive_deterministic_comparison(m_lower, m_upper, expect):
    got = belief(m_lower, m_upper, 0.0).probability_bounds(GreaterThan(50.0))[0]

    assert tuple(got.tolist()) == expect


def test_a_tighter_bound_gives_a_tighter_interval():
    """Shrinking [lower, upper] toward a point must never widen the interval:
    this is the containment guarantee the enclosure exists to provide."""
    wide = belief(46.0, 54.0, 4.0).probability_bounds(GreaterThan(50.0))[0]
    tight = belief(49.0, 51.0, 4.0).probability_bounds(GreaterThan(50.0))[0]

    assert wide[0] <= tight[0]
    assert tight[1] <= wide[1]


def test_collapsed_bound_recovers_a_single_gaussian_tail_probability():
    mean, variance, threshold = 51.0, 4.0, 50.0

    got = belief(mean, mean, variance).probability_bounds(GreaterThan(threshold))[0]
    expected = norm.cdf((mean - threshold) / (variance**0.5))

    assert got[0].item() == pytest.approx(expected, abs=1e-12)
    assert got[1].item() == pytest.approx(expected, abs=1e-12)


def test_gradients_reach_the_bounds():
    lower = torch.tensor([[48.0]], dtype=torch.float64, requires_grad=True)
    upper = torch.tensor([[52.0]], dtype=torch.float64, requires_grad=True)
    step = GaussianBelief(lower, upper, torch.tensor([[4.0]], dtype=torch.float64))

    step.probability_bounds(GreaterThan(50.0)).sum().backward()

    assert lower.grad is not None and upper.grad is not None
    assert torch.isfinite(lower.grad).all() and torch.isfinite(upper.grad).all()


def test_trajectory_factory_builds_one_belief_per_step_and_composes_with_always():
    lower = torch.tensor([[48.0], [49.0], [50.0]], dtype=torch.float64)
    upper = torch.tensor([[52.0], [53.0], [54.0]], dtype=torch.float64)
    variance = torch.full((3, 1), 4.0, dtype=torch.float64)

    trajectory = create_gaussian_belief_trajectory(lower, upper, variance)

    assert len(trajectory) == 3
    trace = Always(GreaterThan(50.0), interval=[0, 2])(trajectory)
    assert (trace[..., 0] <= trace[..., 1]).all()
    assert (trace >= 0.0).all() and (trace <= 1.0).all()


def test_trajectory_factory_accepts_a_scalar_trace():
    lower = torch.tensor([48.0, 49.0, 50.0], dtype=torch.float64)
    upper = torch.tensor([52.0, 53.0, 54.0], dtype=torch.float64)
    variance = torch.full((3,), 4.0, dtype=torch.float64)

    trajectory = create_gaussian_belief_trajectory(lower, upper, variance)

    assert len(trajectory) == 3
    assert trajectory[0].lower.shape == (1, 1)


def test_unevaluable_predicate_and_malformed_shapes_are_rejected():
    from pdstl.operators import Predicate

    with pytest.raises(ValueError, match="cannot evaluate"):
        belief(1.0, 2.0, 1.0).probability_bounds(Predicate("mystery"))

    with pytest.raises(ValueError, match="requires lower <= upper"):
        belief(2.0, 1.0, 1.0)

    with pytest.raises(ValueError, match="must be non-negative"):
        belief(1.0, 2.0, -1.0)
