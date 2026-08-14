"""Audit Groups A-C: beliefs, Gaussian atoms, and rectangular predicates."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.stats import multivariate_normal, norm

from experiments.crazyflie.components.planner import (
    RectangularGoalPredicate as CrazyflieGoal,
)
from experiments.crazyflie.components.planner import (
    RectangularObstaclePredicate as CrazyflieObstacle,
)
from pdstl.base import BeliefTrajectory, OnlineBeliefTrajectory
from pdstl.operators import GreaterThan, LessThan
from pdstl.propagate import compute_bounds
from models.dynamics import GaussianBelief
from planning.environment import (
    CircularObstaclePredicate,
    MovingRectangularObstaclePredicate,
    RectangularGoalPredicate,
    RectangularObstaclePredicate,
    normal_cdf,
)
from planning.planner import TorchGaussianBelief

from conftest import MinimalBelief


def _scalar_gaussian(mean, variance, confidence_level=2.0):
    return GaussianBelief(
        torch.tensor([[[mean]]], dtype=torch.float64),
        torch.tensor([[[variance]]], dtype=torch.float64),
        confidence_level=confidence_level,
    )


def _torch_belief(mean, covariance):
    return TorchGaussianBelief(
        torch.as_tensor(mean, dtype=torch.float64).reshape(1, -1),
        torch.as_tensor(covariance, dtype=torch.float64).reshape(
            1, len(mean), len(mean)
        ),
    )


@pytest.mark.parametrize(
    "predicate",
    [
        GreaterThan(0.0),
        LessThan(0.0),
        pytest.param(
            RectangularGoalPredicate({"x": [-1, 1], "y": [-1, 1]}),
            marks=pytest.mark.xfail(
                strict=True, reason="A1: planning predicate requires undeclared fields"
            ),
        ),
        pytest.param(
            RectangularObstaclePredicate({"x": [-1, 1], "y": [-1, 1]}),
            marks=pytest.mark.xfail(
                strict=True, reason="A1: planning predicate requires undeclared fields"
            ),
        ),
        pytest.param(
            CircularObstaclePredicate({"center": [0, 0], "radius": 1}),
            marks=pytest.mark.xfail(
                strict=True, reason="A1: planning predicate requires undeclared fields"
            ),
        ),
        pytest.param(
            MovingRectangularObstaclePredicate(
                {"x_traj": [0], "y_traj": [0], "width": 1, "height": 1}
            ),
            marks=pytest.mark.xfail(
                strict=True, reason="A1: planning predicate requires undeclared fields"
            ),
        ),
    ],
    ids=["greater", "less", "box-goal", "box-obstacle", "circle", "moving-box"],
)
def test_a1_documented_belief_contract_supports_builtin_predicates(predicate):
    result = predicate(BeliefTrajectory([MinimalBelief()]))
    assert result.shape == (1, 1, 2)


@pytest.mark.parametrize("predicate", [GreaterThan(0.0), LessThan(0.0)])
@pytest.mark.xfail(
    strict=True, reason="A2: TorchGaussianBelief lacks scalar predicate methods"
)
def test_a2_torch_gaussian_works_with_scalar_predicates(predicate):
    result = predicate(BeliefTrajectory([_torch_belief([0.0], [[1.0]])]))
    assert result.shape == (1, 1, 2)


def test_a3_offline_online_trajectory_behavior_matches():
    beliefs = [_scalar_gaussian(-0.5, 1.0), _scalar_gaussian(0.5, 1.0)]
    offline = BeliefTrajectory(beliefs)
    online = OnlineBeliefTrajectory.from_list(beliefs.copy())
    assert len(offline) == len(online) == 2
    assert offline[1] is online[1]
    assert len(offline.suffix(1)) == len(online.suffix(1)) == 1
    assert torch.equal(GreaterThan(0.0)(offline), GreaterThan(0.0)(online))


@pytest.mark.parametrize(
    "predicate,mean,variance,expected",
    [
        (GreaterThan(0.0), 0.0, 1.0, 0.5),
        (GreaterThan(0.0), 1.0, 4.0, norm.cdf(0.5)),
        (LessThan(0.0), 0.0, 1.0, 0.5),
        (LessThan(0.0), 1.0, 4.0, norm.cdf(-0.5)),
    ],
    ids=["ge-standard", "ge-shifted", "le-standard", "le-shifted"],
)
def test_b1_scalar_gaussian_atom_is_exact(predicate, mean, variance, expected):
    observed = predicate(BeliefTrajectory([_scalar_gaussian(mean, variance)]))[0, 0]
    assert observed.numpy() == pytest.approx([expected, expected], abs=1e-7)


def test_b1_propagate_path_uses_exact_gaussian_probability_and_temporal_minimum():
    observed = compute_bounds(
        mean_trace=np.array([0.0, 1.0]),
        var_trace=np.array([1.0, 4.0]),
        time=np.array([0.0, 1.0]),
        threshold=0.0,
        start_time=0.0,
        end_time=1.0,
    )
    assert observed == pytest.approx((0.5, 0.5))


def test_b3_propagate_path_uses_deterministic_equality_convention():
    observed = compute_bounds(
        mean_trace=np.array([0.0]),
        var_trace=np.array([0.0]),
        time=np.array([0.0]),
        threshold=0.0,
        start_time=0.0,
        end_time=0.0,
    )
    assert observed == (1.0, 1.0)


@pytest.mark.parametrize(
    "predicate,expected",
    [
        (GreaterThan(0.0), [0.5, 0.5]),
        (LessThan(0.0), [0.5, 0.5]),
    ],
)
def test_b2_confidence_level_does_not_shift_atomic_probability(predicate, expected):
    observed = predicate(
        BeliefTrajectory([_scalar_gaussian(0.0, 1.0, confidence_level=1.0)])
    )[0, 0]
    assert observed.numpy() == pytest.approx(expected, abs=1e-7)


def test_b3_zero_variance_uses_deterministic_greater_equal_semantics():
    means = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    observed = normal_cdf(0.0, means, torch.zeros_like(means))
    greater_equal = 1.0 - observed
    # At equality P(X >= 0) is one under the audit's deterministic convention.
    greater_equal[1] = observed[1]
    assert greater_equal.numpy() == pytest.approx([0.0, 1.0, 1.0], abs=1e-12)


@pytest.mark.parametrize(
    "predicate,expected",
    [
        (GreaterThan(0.0), [[0, 0], [1, 1], [1, 1]]),
        (LessThan(0.0), [[1, 1], [1, 1], [0, 0]]),
    ],
)
def test_b3_scalar_atom_zero_variance_uses_deterministic_semantics(
    predicate, expected
):
    observed = [
        predicate(BeliefTrajectory([_scalar_gaussian(mean, 0.0)]))[0, 0]
        .detach()
        .numpy()
        for mean in [-1.0, 0.0, 1.0]
    ]
    np.testing.assert_allclose(observed, expected, atol=0.0)


@pytest.mark.parametrize("variance", [1e-16, 1e-12, 1e-8])
def test_b4_tiny_variance_is_finite_monotone_and_has_finite_gradient(variance):
    means = torch.tensor([-1e-4, 0.0, 1e-4], dtype=torch.float64, requires_grad=True)
    cdf = normal_cdf(0.0, means, torch.full_like(means, variance))
    probability_ge = 1.0 - cdf
    assert torch.isfinite(probability_ge).all()
    assert torch.all(probability_ge[1:] >= probability_ge[:-1])
    probability_ge.sum().backward()
    assert torch.isfinite(means.grad).all()


def test_c1_independent_box_goal_returns_frechet_interval_containing_product():
    belief = _torch_belief([0.0, 0.0], np.eye(2))
    observed = RectangularGoalPredicate({"x": [-1, 1], "y": [-1, 1]})(
        BeliefTrajectory([belief])
    )[0, 0]
    axis = norm.cdf(1) - norm.cdf(-1)
    expected = [max(0.0, 2.0 * axis - 1.0), axis]
    assert observed.numpy() == pytest.approx(expected, abs=1e-6)
    assert observed[0].item() <= axis**2 <= observed[1].item()


def test_c1_shared_obstacle_complements_box_frechet_interval():
    belief = _torch_belief([0.0, 0.0], np.eye(2))
    observed = RectangularObstaclePredicate({"x": [-1, 1], "y": [-1, 1]})(
        BeliefTrajectory([belief])
    )[0, 0]
    axis = norm.cdf(1) - norm.cdf(-1)
    inside_lower, inside_upper = max(0.0, 2.0 * axis - 1.0), axis
    expected = [1.0 - inside_upper, 1.0 - inside_lower]
    exact_independent_outside = 1.0 - axis**2
    assert observed.numpy() == pytest.approx(expected, abs=1e-6)
    assert observed[0].item() <= exact_independent_outside <= observed[1].item()


def test_c2_correlated_box_joint_probability_lies_in_generic_interval():
    covariance = np.array([[1.0, 0.9], [0.9, 1.0]])
    belief = _torch_belief([0.0, 0.0], covariance)
    observed = RectangularGoalPredicate({"x": [-1, 1], "y": [-1, 1]})(
        BeliefTrajectory([belief])
    )[0, 0]
    expected = multivariate_normal.cdf(
        [1, 1],
        mean=[0, 0],
        cov=covariance,
        lower_limit=[-1, -1],
        rng=np.random.default_rng(20260814),
    )
    axis = norm.cdf(1) - norm.cdf(-1)
    assert observed.numpy() == pytest.approx(
        [max(0.0, 2.0 * axis - 1.0), axis], abs=1e-6
    )
    assert expected == pytest.approx(0.596360, abs=1e-5)
    assert observed[0].item() <= expected <= observed[1].item()


def test_c3_correlated_probability_lies_inside_frechet_interval():
    covariance = np.array([[1.0, 0.9], [0.9, 1.0]])
    axis = norm.cdf(1) - norm.cdf(-1)
    lower, upper = max(0.0, 2.0 * axis - 1.0), axis
    joint = multivariate_normal.cdf(
        [1, 1], mean=[0, 0], cov=covariance, lower_limit=[-1, -1],
        rng=np.random.default_rng(20260814),
    )
    assert lower <= joint <= upper


def test_c3_off_diagonal_covariance_does_not_change_generic_bounds():
    region = {"x": [-1, 1], "y": [-1, 1]}
    diagonal = _torch_belief([0.0, 0.0], np.eye(2))
    correlated = _torch_belief([0.0, 0.0], [[1.0, 0.9], [0.9, 1.0]])
    diagonal_result = RectangularGoalPredicate(region)(
        BeliefTrajectory([diagonal])
    )
    correlated_result = RectangularGoalPredicate(region)(
        BeliefTrajectory([correlated])
    )
    torch.testing.assert_close(diagonal_result, correlated_result)


@pytest.mark.parametrize(
    "mean,inside",
    [
        ([0.0, 0.0], True),
        ([2.0, 0.0], False),
        ([-1.0, 0.0], True),
        ([1.0, 0.0], True),
        ([0.0, -1.0], True),
        ([0.0, 1.0], True),
    ],
    ids=["inside", "outside", "x-lower", "x-upper", "y-lower", "y-upper"],
)
def test_c3_deterministic_box_includes_boundaries(mean, inside):
    belief = _torch_belief(mean, np.zeros((2, 2)))
    trajectory = BeliefTrajectory([belief])
    region = {"x": [-1, 1], "y": [-1, 1]}
    goal = RectangularGoalPredicate(region)(trajectory)[0, 0]
    obstacle = RectangularObstaclePredicate(region)(trajectory)[0, 0]
    expected_goal = [1.0, 1.0] if inside else [0.0, 0.0]
    expected_obstacle = [0.0, 0.0] if inside else [1.0, 1.0]
    assert goal.tolist() == expected_goal
    assert obstacle.tolist() == expected_obstacle


def test_c3_generic_rectangle_interval_invariants_for_seeded_cases():
    rng = np.random.default_rng(20260814)
    region = {"x": [-1.5, 0.75], "y": [-0.25, 2.0]}
    for _ in range(100):
        mean = rng.normal(size=2)
        variances = rng.lognormal(mean=-1.0, sigma=1.0, size=2)
        trajectory = BeliefTrajectory([_torch_belief(mean, np.diag(variances))])
        for predicate in (
            RectangularGoalPredicate(region),
            RectangularObstaclePredicate(region),
        ):
            lower, upper = predicate(trajectory)[0, 0].tolist()
            assert 0.0 <= lower <= upper <= 1.0


def test_c4_crazyflie_diagonal_box_predicates_are_exact_products():
    belief = _torch_belief([0.0, 0.0], np.eye(2))
    traj = BeliefTrajectory([belief])
    region = {"x": [-1, 1], "y": [-1, 1]}
    axis = norm.cdf(1) - norm.cdf(-1)
    goal = CrazyflieGoal(region)(traj)[0, 0, 0].item()
    outside = CrazyflieObstacle(region)(traj)[0, 0, 0].item()
    generic_goal = RectangularGoalPredicate(region)(traj)[0, 0]
    generic_outside = RectangularObstaclePredicate(region)(traj)[0, 0]
    assert goal == pytest.approx(axis**2, abs=1e-6)
    assert outside == pytest.approx(1.0 - axis**2, abs=1e-6)
    assert generic_goal[0].item() <= goal <= generic_goal[1].item()
    assert generic_outside[0].item() <= outside <= generic_outside[1].item()


def test_c4_crazyflie_obstacle_product_lies_in_shared_frechet_interval():
    belief = _torch_belief([0.0, 0.0], np.eye(2))
    traj = BeliefTrajectory([belief])
    region = {"x": [-1, 1], "y": [-1, 1]}
    shared = RectangularObstaclePredicate(region)(traj)[0, 0]
    duplicate = CrazyflieObstacle(region)(traj)[0, 0, 0]
    assert shared[0] <= duplicate <= shared[1]
