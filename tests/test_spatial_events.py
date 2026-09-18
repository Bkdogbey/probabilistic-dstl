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

from models.beliefs import GaussianBelief, create_gaussian_belief_trajectory
from pdstl.base import BeliefTrajectory
from pdstl.operators import Always, And, Eventually, Predicate, _conjunction, _negation
from pdstl.predicates import GreaterThan, HalfSpace, InsideRectangle, LessThan, OutsideRectangle
from planning.environment import CircleRegion, MovingRectangleRegion
from pdstl import predicates as moment_predicates
from pdstl.predicates import (
    CircularObstaclePredicate,
    MovingRectangularObstaclePredicate,
)
from planning.environment import build_reach_avoid_environment

ROOT = Path(__file__).resolve().parents[1]
GOAL = ([10.0, 12.0], [2.0, 4.0])


def _interval(low, high, mu, sigma):
    return norm.cdf((high - mu) / sigma) - norm.cdf((low - mu) / sigma)


def rect_bounds(belief, rectangle, beta=None):
    """Probability interval [B, 2] of a rectangle event under one belief.

    Rectangles are conjunctions of axis intervals rather than atoms, so they are evaluated
    through the operator layer instead of by `GaussianBelief.probability_bounds`.
    """
    return rectangle(BeliefTrajectory([belief]), beta=beta)[:, 0]


# --- Predicates: events, not probabilities ----------------------------------


@pytest.mark.parametrize("event_type", [InsideRectangle, OutsideRectangle])
def test_rectangle_events_store_geometry_only(event_type):
    event = event_type([4, 7], [3.8, 8.0])

    assert event.x_range == (4.0, 7.0)
    assert event.y_range == (3.8, 8.0)
    assert event.dims == (0, 1)
    assert list(event.parameters()) == [] and list(event.buffers()) == []
    assert event_type.__name__ in str(event)


@pytest.mark.parametrize("event_type", [InsideRectangle, OutsideRectangle])
def test_rectangle_is_composed_of_two_exact_axis_intervals(event_type):
    """The rectangle is a formula over two atoms, so `beta` reaches its conjunction."""
    event = event_type([4, 7], [3.8, 8.0])

    axis_x, axis_y = event.axes
    assert isinstance(axis_x, Predicate) and isinstance(axis_y, Predicate)
    assert (axis_x.lower, axis_x.upper, axis_x.dim) == (4.0, 7.0, 0)
    assert (axis_y.lower, axis_y.upper, axis_y.dim) == (3.8, 8.0, 1)
    assert event.is_pointwise, "a rectangle stays a pointwise event"


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
    assert modules and all(m.startswith("pdstl") or m in {"math", "torch"} for m in modules)

    # Only the explicit legacy moment predicates may inspect moments. Geometric
    # events continue to delegate probability evaluation to the belief contract.
    geometry = [n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name not in {"CircularObstaclePredicate", "MovingRectangularObstaclePredicate"}]
    nodes = [node for cls in geometry for node in ast.walk(cls)]
    identifiers = {n.id for n in nodes if isinstance(n, ast.Name)}
    identifiers |= {n.attr for n in nodes if isinstance(n, ast.Attribute)}
    for forbidden in ("torch", "mean", "covariance", "GaussianBelief", "norm", "erf"):
        assert forbidden not in identifiers


# --- GaussianBelief evaluates the events ------------------------------------


@pytest.mark.parametrize("full", [False, True])
def test_affine_probability_uses_projected_gaussian_in_three_dimensions(full):
    mean = torch.tensor([[0.2, -0.4, 0.6], [-0.3, 0.5, 0.8]], dtype=torch.float64)
    covariance = torch.tensor([[0.4, 0.1, -0.05], [0.1, 0.6, 0.2], [-0.05, 0.2, 0.8]],
                              dtype=torch.float64)
    cov = covariance.expand(2, 3, 3) if full else covariance.diag().expand(2, 3)
    a, b = np.array([1.0, -2.0, 0.5]), 0.3
    projected_cov = covariance.numpy() if full else np.diag(covariance.diag().numpy())
    expected = norm.cdf((b - mean.numpy() @ a) / np.sqrt(a @ projected_cov @ a))
    bounds = GaussianBelief(mean, cov).probability_bounds(HalfSpace(a, b))
    np.testing.assert_allclose(bounds.numpy(), np.repeat(expected[:, None], 2, axis=1))


def test_affine_singular_covariance_uses_closed_inequality_and_finite_gradients():
    mean = torch.tensor([[1.0, 1.0], [2.0, 1.0], [0.0, 1.0]], requires_grad=True)
    covariance = torch.ones(3, 2, 2, requires_grad=True)
    bounds = GaussianBelief(mean, covariance).probability_bounds(HalfSpace([1, -1], 0))
    assert bounds.tolist() == [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]
    bounds.sum().backward()
    assert torch.isfinite(mean.grad).all() and torch.isfinite(covariance.grad).all()


def test_affine_probability_gradients_reach_mean_and_covariance():
    mean = torch.tensor([[0.2, -0.1]], dtype=torch.float64, requires_grad=True)
    factor = torch.tensor([[[0.8, 0.0], [0.2, 0.6]]], dtype=torch.float64, requires_grad=True)
    event = HalfSpace([1, -2], 0.7)

    def probability(mu, chol):
        return GaussianBelief(mu, chol @ chol.transpose(-1, -2)).probability_bounds(event)

    assert torch.autograd.gradcheck(probability, (mean, factor))
    probability(mean, factor).sum().backward()
    assert mean.grad.abs().sum() > 0 and factor.grad.abs().sum() > 0


@pytest.mark.parametrize("a,b", [([], 0), ([0, 0], 1), ([float("nan"), 1], 0),
                                  ([1, float("inf")], 0), ([1], float("inf"))])
def test_half_space_rejects_invalid_geometry(a, b):
    with pytest.raises(ValueError):
        HalfSpace(a, b)


def test_half_space_rejects_wrong_state_dimension():
    with pytest.raises(ValueError, match="state dimension"):
        GaussianBelief(torch.zeros(1, 2), torch.ones(1, 2)).probability_bounds(HalfSpace([1], 0))


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
    bounds = rect_bounds(GaussianBelief(mean, covariance), InsideRectangle(*GOAL))

    assert bounds.shape == (64, 2)
    assert (bounds >= 0).all() and (bounds <= 1).all()
    assert (bounds[:, 0] <= bounds[:, 1] + 1e-6).all()


def test_diagonal_inside_uses_exact_marginals_and_brackets_the_independent_value():
    mu, sigma = np.array([10.4, 3.5]), np.array([0.6, 0.8])
    belief = GaussianBelief(
        torch.tensor(mu[None], dtype=torch.float64),
        torch.tensor(sigma[None] ** 2, dtype=torch.float64),
    )
    lower, upper = rect_bounds(belief, InsideRectangle(*GOAL))[0].tolist()

    p_x = _interval(*GOAL[0], mu[0], sigma[0])
    p_y = _interval(*GOAL[1], mu[1], sigma[1])
    assert lower == pytest.approx(max(0.0, p_x + p_y - 1.0), abs=1e-10)
    assert upper == pytest.approx(min(p_x, p_y), abs=1e-10)
    assert lower <= p_x * p_y <= upper  # independence is one member of the set


def test_correlated_inside_probability_lies_within_the_bounds():
    mean = torch.tensor([[10.8, 3.3]], dtype=torch.float64)
    covariance = torch.tensor([[[0.5, -0.45], [-0.45, 0.6]]], dtype=torch.float64)
    lower, upper = rect_bounds(
        GaussianBelief(mean, covariance),
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

    inside = rect_bounds(belief, InsideRectangle(*rectangle))
    outside = rect_bounds(belief, OutsideRectangle(*rectangle))

    torch.testing.assert_close(outside[:, 0], 1.0 - inside[:, 1])
    torch.testing.assert_close(outside[:, 1], 1.0 - inside[:, 0])


@pytest.mark.parametrize(
    "point, inside",
    [([11.0, 3.0], True), ([10.0, 4.0], True), ([9.99, 3.0], False), ([11.0, 4.01], False)],
)
def test_zero_variance_rectangle_is_a_deterministic_closed_test(point, inside):
    belief = GaussianBelief(torch.tensor([point]), torch.zeros(1, 2, 2))
    expected = float(inside)

    assert rect_bounds(belief, InsideRectangle(*GOAL)).tolist() == [[expected] * 2]
    assert rect_bounds(belief, OutsideRectangle(*GOAL)).tolist() == [[1 - expected] * 2]


def test_interval_deep_in_a_tail_keeps_float32_resolution():
    belief = GaussianBelief(torch.tensor([[0.0, 0.0]]), torch.ones(1, 2))
    far = rect_bounds(belief, InsideRectangle([4.0, 5.0], [-1e3, 1e3]))
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

    score = spec(trajectory, beta=None)[0, 0, 0]
    score.backward()

    assert 0 < score.item() < 1
    assert torch.isfinite(mean.grad).all()
    assert mean.grad.abs().sum() > 0


def test_gaussian_belief_rejects_rectangle_dimensions_outside_the_state():
    belief = GaussianBelief(torch.zeros(1, 2), torch.ones(1, 2))
    with pytest.raises(ValueError, match="outside the state"):
        rect_bounds(belief, InsideRectangle([0, 1], [0, 1], dims=(0, 2)))


# --- Environment builds the formula from events -----------------------------


def _reach_avoid_environment():
    return build_reach_avoid_environment({
        "workspace": {"x": [-50.0, 50.0], "y": [-50.0, 50.0]},
        "goal": {"x": list(GOAL[0]), "y": list(GOAL[1])},
        "obstacles": [{"name": "block", "x": [4.0, 7.0], "y": [3.8, 8.0]}],
    })


def test_environment_builds_always_outside_and_eventually_inside():
    H = 12
    spec = _reach_avoid_environment().get_specification(H)

    assert isinstance(spec, And)
    safe, reach = spec.subformula1, spec.subformula2
    assert isinstance(safe, Always) and safe.interval == [1, H]
    assert isinstance(reach, Eventually) and reach.interval == [0, H]

    # safety is one Always over the workspace conjoined with every obstacle
    workspace, block = safe.subformula.subformula1, safe.subformula.subformula2
    assert type(workspace) is InsideRectangle
    assert type(block) is OutsideRectangle
    assert (block.x_range, block.y_range) == ((4.0, 7.0), (3.8, 8.0))
    assert type(reach.subformula) is InsideRectangle
    assert (reach.subformula.x_range, reach.subformula.y_range) == ((10.0, 12.0), (2.0, 4.0))


def test_environment_computes_no_probability_itself(monkeypatch):
    H = 4
    spec = _reach_avoid_environment().get_specification(H)

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy Gaussian probability path used")

    monkeypatch.setattr(moment_predicates, "normal_cdf", forbidden)
    monkeypatch.setattr(moment_predicates, "extract_trajectory_stats", forbidden)
    evaluated = []
    original = GaussianBelief.probability_bounds

    def spy(self, predicate):
        evaluated.append(type(predicate))
        return original(self, predicate)

    monkeypatch.setattr(GaussianBelief, "probability_bounds", spy)

    mean = torch.tensor([[0.0, 5.0], [3.0, 4.0], [6.0, 3.0], [9.0, 3.0], [11.0, 3.0]])
    trajectory = create_gaussian_belief_trajectory(mean, torch.eye(2).expand(5, 2, 2) * 0.1)
    got = spec(trajectory, beta=None)[0, 0]

    # Rectangles are no longer atoms: the belief only ever sees exact axis intervals, and
    # the Frechet conjunction that turns two of them into a rectangle lives in operators.py.
    assert sorted(set(t.__name__ for t in evaluated)) == ["AxisInterval"]
    # One call per axis interval, not one per step: the trajectory scores all T+1 at once.
    assert len(evaluated) == 6  # three rectangles (workspace, block, goal), two axes each

    # Rebuild the value from the belief's own axis-interval bounds and the pdSTL rules:
    # conjoin the two axes, negate for "outside", then apply the temporal reductions.
    def rect(rectangle):
        axis_x, axis_y = rectangle.axes
        px = torch.stack([original(b, axis_x) for b in trajectory], 1)
        py = torch.stack([original(b, axis_y) for b in trajectory], 1)
        inside = _conjunction(px, py)
        return _negation(inside) if isinstance(rectangle, OutsideRectangle) else inside

    # safety conjoins the workspace with the block at each step, THEN takes the temporal min
    workspace = rect(InsideRectangle([-50.0, 50.0], [-50.0, 50.0]))
    block = rect(OutsideRectangle([4.0, 7.0], [3.8, 8.0]))
    safe = _conjunction(workspace, block)[:, 1:].min(dim=1).values
    goal = rect(InsideRectangle(*GOAL)).max(dim=1).values
    torch.testing.assert_close(got, _conjunction(safe, goal)[0])


def test_moment_predicates_are_built_from_their_regions():
    """Circles and moving rectangles are regions like any other; only their event differs."""
    import torch as _torch

    circle = CircleRegion(name="c", role="obstacle", center=(1.0, 1.0), radius=0.5)
    moving = MovingRectangleRegion(
        name="v", role="obstacle",
        centers=_torch.tensor([[0.0, 0.0], [1.0, 0.0]]), width=1.0, height=1.0,
    )
    assert CircularObstaclePredicate(circle).radius == 0.5
    assert MovingRectangularObstaclePredicate(moving).width == 1.0
