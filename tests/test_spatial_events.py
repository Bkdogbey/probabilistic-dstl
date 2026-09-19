"""Geometric events delegate probability evaluation to beliefs."""

import numpy as np
import pytest
import torch
from scipy.stats import norm

from models.beliefs import GaussianBelief, ProbabilityBelief
from models.rollouts import create_gaussian_belief_trajectory
from pdstl.base import BeliefTrajectory
from pdstl.predicates import (
    HalfSpace,
    OutsideRectangle,
    MovingRectangularObstaclePredicate,
)
from planning.environment import MovingRectangleRegion


@pytest.mark.parametrize("full", [False, True])
def test_half_space_uses_projected_gaussian_variance(full):
    mean = torch.tensor([[0.2, -0.4, 0.6]], dtype=torch.float64)
    covariance = torch.tensor(
        [[0.4, 0.1, -0.05], [0.1, 0.6, 0.2], [-0.05, 0.2, 0.8]],
        dtype=torch.float64,
    )
    cov = covariance.unsqueeze(0) if full else covariance.diag().unsqueeze(0)
    normal, cutoff = np.array([1.0, -2.0, 0.5]), 0.3
    projected = covariance.numpy() if full else np.diag(covariance.diag())
    expected = norm.cdf(
        (cutoff - mean.numpy() @ normal) / np.sqrt(normal @ projected @ normal)
    )
    bounds = GaussianBelief(mean, cov).probability_bounds(
        HalfSpace(normal, cutoff)
    )
    np.testing.assert_allclose(bounds.numpy(), [[expected[0], expected[0]]])


def test_half_space_gradient_reaches_gaussian_parameters():
    mean = torch.tensor([[0.2, -0.1]], dtype=torch.float64, requires_grad=True)
    factor = torch.tensor(
        [[[0.8, 0.0], [0.2, 0.6]]], dtype=torch.float64, requires_grad=True
    )
    event = HalfSpace([1, -2], 0.7)
    bounds = GaussianBelief(mean, factor @ factor.transpose(-1, -2))
    bounds.probability_bounds(event).sum().backward()
    assert mean.grad.abs().sum() > 0 and factor.grad.abs().sum() > 0


def test_moving_rectangle_uses_probability_only_beliefs():
    centers = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    region = MovingRectangleRegion("vehicle", "obstacle", centers, 1.0, 1.0)
    moving = MovingRectangularObstaclePredicate(region)
    beliefs = []
    for center, x_bounds, y_bounds in (
        (centers[0], [0.2, 0.4], [0.3, 0.5]),
        (centers[1], [0.6, 0.8], [0.7, 0.9]),
    ):
        x, y = center.tolist()
        event = OutsideRectangle((x - 0.5, x + 0.5), (y - 0.5, y + 0.5))
        beliefs.append(
            ProbabilityBelief(
                {
                    event.axes[0].name: torch.tensor(
                        [x_bounds], requires_grad=True
                    ),
                    event.axes[1].name: torch.tensor(
                        [y_bounds], requires_grad=True
                    ),
                }
            )
        )
    trace = BeliefTrajectory(beliefs)
    hard = moving(trace)
    torch.testing.assert_close(hard[0], torch.tensor([[0.6, 1.0], [0.2, 0.7]]))
    moving(trace, beta=10)[..., 0].sum().backward()
    gradient = beliefs[1].bounds[next(iter(beliefs[1].bounds))].grad
    assert gradient is not None and gradient.abs().sum() > 0


def test_moving_rectangle_has_non_degenerate_valid_gaussian_bounds():
    region = MovingRectangleRegion(
        "vehicle", "obstacle", torch.tensor([[0.0, 0.0]]), 1.0, 1.0
    )
    beliefs = create_gaussian_belief_trajectory(
        torch.tensor([[0.1, 0.1]]), torch.eye(2).unsqueeze(0) * 0.4
    )
    lower, upper = MovingRectangularObstaclePredicate(region)(beliefs)[0, 0]
    assert 0 <= lower < upper <= 1
