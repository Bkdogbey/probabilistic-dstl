"""Patch 2 tests for the shared Gaussian residual probability primitive."""

from __future__ import annotations

import math

import pytest
import torch
from scipy.stats import norm

from models.dynamics import GaussianBelief
from pdstl.base import BeliefTrajectory
from pdstl.operators import GreaterThan, LessThan
from pdstl.probability import gaussian_residual_probability
from planning.environment import RectangularObstaclePredicate, normal_cdf
from planning.planner import TorchGaussianBelief


def _belief(mean, variance):
    return GaussianBelief(
        torch.tensor([[[mean]]], dtype=torch.float64),
        torch.tensor([[[variance]]], dtype=torch.float64),
    )


def test_shared_probability_zero_variance_and_equality():
    means = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    observed = gaussian_residual_probability(means, torch.zeros_like(means))
    assert observed.tolist() == [0.0, 1.0, 1.0]


@pytest.mark.parametrize("variance", [1e-16, 1e-12, 1e-8])
def test_shared_probability_tiny_positive_variance_is_exact(variance):
    std = math.sqrt(variance)
    means = torch.tensor([-std, 0.0, std], dtype=torch.float64)
    observed = gaussian_residual_probability(means, variance)
    expected = torch.tensor([norm.cdf(-1), 0.5, norm.cdf(1)], dtype=torch.float64)
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=1e-12)


def test_shared_probability_rejects_negative_variance():
    with pytest.raises(ValueError, match="nonnegative"):
        gaussian_residual_probability(torch.tensor(0.0), torch.tensor(-1.0))


def test_shared_probability_has_finite_positive_variance_gradients():
    mean = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    variance = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    probability = gaussian_residual_probability(mean, variance)
    probability.backward()
    assert torch.isfinite(probability)
    assert torch.isfinite(mean.grad)
    assert torch.isfinite(variance.grad)


@pytest.mark.parametrize(
    "predicate,mean,threshold,residual",
    [
        (GreaterThan, 1.0, 0.25, 0.75),
        (LessThan, 1.0, 0.25, -0.75),
    ],
)
def test_greater_less_orientation_matches_shared_residual(
    predicate, mean, threshold, residual
):
    observed = predicate(threshold)(BeliefTrajectory([_belief(mean, 0.5)]))[0, 0, 0]
    expected = gaussian_residual_probability(
        torch.tensor(residual, dtype=torch.float64),
        torch.tensor(0.5, dtype=torch.float64),
    )
    torch.testing.assert_close(observed, expected)


@pytest.mark.parametrize(
    "mean,variance,threshold",
    [(0.0, 1.0, 0.0), (1.0, 4.0, 0.0), (0.0, 0.0, 0.0)],
)
def test_planning_less_edge_matches_corrected_gaussian_belief(
    mean, variance, threshold
):
    planning_probability = normal_cdf(
        threshold,
        torch.tensor(mean, dtype=torch.float64),
        torch.tensor(variance, dtype=torch.float64),
    )
    atom_probability = LessThan(threshold)(
        BeliefTrajectory([_belief(mean, variance)])
    )[0, 0, 0]
    torch.testing.assert_close(planning_probability, atom_probability)


@pytest.mark.parametrize("mean", [-1.0, 0.0, 1.0])
def test_planning_greater_edge_matches_corrected_gaussian_belief_at_zero_variance(
    mean,
):
    planning_probability = gaussian_residual_probability(
        torch.tensor(mean, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)
    )
    atom_probability = GreaterThan(0.0)(
        BeliefTrajectory([_belief(mean, 0.0)])
    )[0, 0, 0]
    torch.testing.assert_close(planning_probability, atom_probability)


def test_generic_obstacle_complement_excludes_included_box_boundary():
    belief = TorchGaussianBelief(
        torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        torch.zeros(1, 2, 2, dtype=torch.float64),
    )
    predicate = RectangularObstaclePredicate({"x": [-1.0, 1.0], "y": [-1.0, 1.0]})
    observed = predicate(BeliefTrajectory([belief]))[0, 0]
    torch.testing.assert_close(observed, torch.zeros(2, dtype=torch.float64))
