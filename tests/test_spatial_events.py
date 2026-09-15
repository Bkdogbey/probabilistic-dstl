"""Spatial events at their three layers:

    pdstl.predicates   what event (geometry only)
    GaussianBelief     how one belief evaluates it (probability_bounds)
    Environment        which formula the planning world asks for
"""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import norm

import planning.environment as environment
from models.beliefs import GaussianBelief, create_gaussian_belief_trajectory
from pdstl.operators import (
    Always,
    And,
    Eventually,
    GreaterThan,
    LessThan,
    Predicate,
    _conjunction,
)
from pdstl.predicates import InsideRectangle, OutsideRectangle
from planning.environment import (
    CircularObstaclePredicate,
    Environment,
    MovingRectangularObstaclePredicate,
)

ROOT = Path(__file__).resolve().parents[1]
GOAL = ([10.0, 12.0], [2.0, 4.0])


def _interval(low, high, mu, sigma):
    return norm.cdf((high - mu) / sigma) - norm.cdf((low - mu) / sigma)


# --- Predicates: events, not probabilities ----------------------------------


@pytest.mark.parametrize("event_type", [InsideRectangle, OutsideRectangle])
def test_rectangle_events_store_geometry_only(event_type):
    event = event_type([4, 7], [3.8, 8.0])

    assert isinstance(event, Predicate)
    assert event.x_range == (4.0, 7.0)
    assert event.y_range == (3.8, 8.0)
    assert event.dims == (0, 1)
    assert event_type.robustness_trace is Predicate.robustness_trace
    assert list(event.parameters()) == [] and list(event.buffers()) == []
    assert event_type.__name__ in str(event)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x_range": [2.0, 1.0], "y_range": [0.0, 1.0]},
        {"x_range": [0.0, 1.0], "y_range": [1.0, 1.0]},
        {"x_range": [0.0, 1.0], "y_range": [0.0, 1.0], "dims": (1, 1)},
        {"x_range": [0.0, 1.0], "y_range": [0.0, 1.0], "dims": (0, -1)},
    ],
)
def test_rectangle_events_reject_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        InsideRectangle(**kwargs)


def test_predicates_module_imports_no_gaussian_or_model_code():
    tree = ast.parse((ROOT / "src/pdstl/predicates.py").read_text(encoding="utf-8"))
    modules = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    modules += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert modules and all(m.startswith("pdstl") for m in modules)

    identifiers = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    identifiers |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("torch", "mean", "covariance", "GaussianBelief", "norm", "erf"):
        assert forbidden not in identifiers


# --- GaussianBelief evaluates the events ------------------------------------


def test_scalar_predicates_are_unchanged():
    mean = torch.tensor([[0.3, -1.2]])
    covariance = torch.tensor([[[0.25, 0.1], [0.1, 0.49]]])
    belief = GaussianBelief(mean, covariance)

    ge = belief.probability_bounds(GreaterThan(0.5, dim=0))
    le = belief.probability_bounds(LessThan(-1.0, dim=1))

    expected_ge = norm.cdf((0.3 - 0.5) / 0.5)
    expected_le = norm.cdf((-1.0 + 1.2) / 0.7)
    np.testing.assert_allclose(ge.numpy(), [[expected_ge] * 2], atol=1e-6)
    np.testing.assert_allclose(le.numpy(), [[expected_le] * 2], atol=1e-6)


def test_inside_bounds_are_ordered_probabilities_for_random_beliefs():
    torch.manual_seed(0)
    mean = torch.randn(64, 2) * 2 + torch.tensor([11.0, 3.0])
    factor = torch.randn(64, 2, 2) * 0.6
    covariance = factor @ factor.transpose(-1, -2) + 1e-3 * torch.eye(2)
    bounds = GaussianBelief(mean, covariance).probability_bounds(InsideRectangle(*GOAL))

    assert bounds.shape == (64, 2)
    assert (bounds >= 0).all() and (bounds <= 1).all()
    assert (bounds[:, 0] <= bounds[:, 1] + 1e-6).all()


def test_diagonal_inside_uses_exact_marginals_and_brackets_the_independent_value():
    mu, sigma = np.array([10.4, 3.5]), np.array([0.6, 0.8])
    belief = GaussianBelief(
        torch.tensor(mu[None], dtype=torch.float64),
        torch.tensor(sigma[None] ** 2, dtype=torch.float64),
    )
    lower, upper = belief.probability_bounds(InsideRectangle(*GOAL))[0].tolist()

    p_x = _interval(*GOAL[0], mu[0], sigma[0])
    p_y = _interval(*GOAL[1], mu[1], sigma[1])
    assert lower == pytest.approx(max(0.0, p_x + p_y - 1.0), abs=1e-10)
    assert upper == pytest.approx(min(p_x, p_y), abs=1e-10)
    assert lower <= p_x * p_y <= upper  # independence is one member of the set


def test_correlated_inside_probability_lies_within_the_bounds():
    mean = torch.tensor([[10.8, 3.3]], dtype=torch.float64)
    covariance = torch.tensor([[[0.5, -0.45], [-0.45, 0.6]]], dtype=torch.float64)
    lower, upper = GaussianBelief(mean, covariance).probability_bounds(
        InsideRectangle(*GOAL)
    )[0].tolist()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        x = torch.distributions.MultivariateNormal(mean[0], covariance[0]).sample((200_000,))
    inside = ((x[:, 0] >= 10) & (x[:, 0] <= 12) & (x[:, 1] >= 2) & (x[:, 1] <= 4))
    p = inside.double().mean().item()

    assert lower - 0.01 <= p <= upper + 0.01
    assert upper - lower > 0.05  # the correlated case is genuinely an interval


def test_outside_is_the_exact_complement_of_inside():
    torch.manual_seed(2)
    mean = torch.randn(16, 2) * 2 + torch.tensor([5.5, 5.0])
    covariance = torch.diag_embed(torch.rand(16, 2) + 0.05)
    belief = GaussianBelief(mean, covariance)
    rectangle = ([4.0, 7.0], [3.8, 8.0])

    inside = belief.probability_bounds(InsideRectangle(*rectangle))
    outside = belief.probability_bounds(OutsideRectangle(*rectangle))

    torch.testing.assert_close(outside[:, 0], 1.0 - inside[:, 1])
    torch.testing.assert_close(outside[:, 1], 1.0 - inside[:, 0])


@pytest.mark.parametrize(
    "point, inside",
    [([11.0, 3.0], True), ([10.0, 4.0], True), ([9.99, 3.0], False), ([11.0, 4.01], False)],
)
def test_zero_variance_rectangle_is_a_deterministic_closed_test(point, inside):
    belief = GaussianBelief(torch.tensor([point]), torch.zeros(1, 2, 2))
    expected = float(inside)

    assert belief.probability_bounds(InsideRectangle(*GOAL)).tolist() == [[expected] * 2]
    assert belief.probability_bounds(OutsideRectangle(*GOAL)).tolist() == [[1 - expected] * 2]


def test_interval_deep_in_a_tail_keeps_float32_resolution():
    belief = GaussianBelief(torch.tensor([[0.0, 0.0]]), torch.ones(1, 2))
    far = belief.probability_bounds(InsideRectangle([4.0, 5.0], [-1e3, 1e3]))
    expected = _interval(4.0, 5.0, 0.0, 1.0)

    assert far[0, 1].item() == pytest.approx(expected, rel=1e-2)
    assert far[0, 1].item() > 0


def test_gradients_reach_the_mean_through_rectangle_events_and_pdstl():
    mean = torch.tensor(
        [[0.0, 5.0], [3.0, 4.0], [6.0, 3.5], [9.0, 3.0], [10.5, 3.0]], requires_grad=True
    )
    covariance = torch.eye(2).expand(5, 2, 2) * 0.2
    trajectory = create_gaussian_belief_trajectory(mean, covariance)
    spec = And(
        Always(OutsideRectangle([4.0, 7.0], [3.8, 8.0]), interval=[1, 4]),
        Eventually(InsideRectangle(*GOAL), interval=[1, 4]),
    )

    score = spec(trajectory, scale=-1)[0, 0, 0]
    score.backward()

    assert 0 < score.item() < 1
    assert torch.isfinite(mean.grad).all()
    assert mean.grad.abs().sum() > 0


def test_gaussian_belief_rejects_rectangle_dimensions_outside_the_state():
    belief = GaussianBelief(torch.zeros(1, 2), torch.ones(1, 2))
    with pytest.raises(ValueError, match="outside the state"):
        belief.probability_bounds(InsideRectangle([0, 1], [0, 1], dims=(0, 2)))


# --- Environment builds the formula from events -----------------------------


def _reach_avoid_environment():
    env = Environment()
    env.set_goal(*GOAL)
    env.add_obstacle([4.0, 7.0], [3.8, 8.0])
    return env


def test_environment_builds_always_outside_and_eventually_inside():
    H = 12
    spec = _reach_avoid_environment().get_specification(H, t_goal_start=1)

    assert isinstance(spec, And)
    safe, reach = spec.subformula1, spec.subformula2
    assert isinstance(safe, Always) and safe.interval == [1, H]
    assert isinstance(reach, Eventually) and reach.interval == [1, H]
    assert type(safe.subformula) is OutsideRectangle
    assert (safe.subformula.x_range, safe.subformula.y_range) == ((4.0, 7.0), (3.8, 8.0))
    assert type(reach.subformula) is InsideRectangle
    assert (reach.subformula.x_range, reach.subformula.y_range) == ((10.0, 12.0), (2.0, 4.0))


def test_environment_computes_no_probability_itself(monkeypatch):
    H = 4
    spec = _reach_avoid_environment().get_specification(H, t_goal_start=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy Gaussian probability path used")

    monkeypatch.setattr(environment, "normal_cdf", forbidden)
    monkeypatch.setattr(environment, "extract_trajectory_stats", forbidden)
    evaluated = []
    original = GaussianBelief.probability_bounds

    def spy(self, predicate):
        evaluated.append(type(predicate))
        return original(self, predicate)

    monkeypatch.setattr(GaussianBelief, "probability_bounds", spy)

    mean = torch.tensor([[0.0, 5.0], [3.0, 4.0], [6.0, 3.0], [9.0, 3.0], [11.0, 3.0]])
    trajectory = create_gaussian_belief_trajectory(mean, torch.eye(2).expand(5, 2, 2) * 0.1)
    got = spec(trajectory, scale=-1)[0, 0]

    assert sorted(set(t.__name__ for t in evaluated)) == ["InsideRectangle", "OutsideRectangle"]
    assert len(evaluated) == 2 * (H + 1)

    # Rebuild the value from the belief's own event bounds and the pdSTL rules.
    safe = torch.stack([original(b, OutsideRectangle([4.0, 7.0], [3.8, 8.0])) for b in trajectory], 1)
    goal = torch.stack([original(b, InsideRectangle(*GOAL)) for b in trajectory], 1)
    expected = _conjunction(
        safe[:, 1:].min(dim=1).values, goal[:, 1:].max(dim=1).values
    )[0]
    torch.testing.assert_close(got, expected)


def test_legacy_obstacle_kinds_keep_their_predicates():
    env = Environment()
    env.add_circle_obstacle([1.0, 1.0], 0.5)
    env.add_moving_obstacle([0.0, 1.0], [0.0, 0.0], 1.0, 1.0)
    kinds = [type(p) for p in env.get_predicates()["obstacles"]]
    assert kinds == [CircularObstaclePredicate, MovingRectangularObstaclePredicate]
