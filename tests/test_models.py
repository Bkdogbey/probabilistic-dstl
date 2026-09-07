"""Controlled dynamics and the shared Gaussian belief boundary."""

import pytest
import torch
from models.beliefs import GaussianBelief
from models.dynamics import SingleIntegrator, DoubleIntegrator
from pdstl.base import BeliefTrajectory
from pdstl.operators import GreaterThan
from planning.environment import Environment, extract_trajectory_stats
from planning.planner import Planner


@pytest.mark.parametrize(
    "model_type, dimension", [(SingleIntegrator, 2), (DoubleIntegrator, 4)]
)
def test_rollout_matches_step_and_linear_prediction(model_type, dimension):
    model = model_type(dt=0.2, u_max=2.0, q_std=0.1)
    initial_mean = torch.arange(dimension, dtype=torch.float32)
    initial_covariance = torch.eye(dimension) * 0.3
    parameters = torch.tensor([[0.2, -0.1], [-0.1, 0.3]], requires_grad=True)
    controls = model.bound_control(parameters)
    mean, covariance = model(parameters, initial_mean, initial_covariance)
    transition = torch.eye(dimension)
    if dimension == 2:
        control_matrix = torch.eye(2) * 0.2
    else:
        transition[0, 2] = transition[1, 3] = 0.2
        control_matrix = torch.tensor([[0.02, 0], [0, 0.02], [0.2, 0], [0, 0.2]])
    expected_mean, expected_covariance = initial_mean, initial_covariance
    torch.testing.assert_close(mean[0, 0], initial_mean)
    torch.testing.assert_close(covariance[0, 0], initial_covariance)
    for k, control in enumerate(controls):
        next_mean, next_covariance = model.step(
            expected_mean, expected_covariance, control
        )
        expected_mean = transition @ expected_mean + control_matrix @ control
        expected_covariance = (
            transition @ expected_covariance @ transition.T
            + torch.eye(dimension) * 0.01
        )
        torch.testing.assert_close(next_mean, expected_mean)
        torch.testing.assert_close(next_covariance, expected_covariance)
        torch.testing.assert_close(mean[0, k + 1], expected_mean)
        torch.testing.assert_close(covariance[0, k + 1], expected_covariance)
    mean[0, -1].sum().backward()
    assert torch.isfinite(parameters.grad).all()
    assert parameters.grad.abs().sum() > 0


def test_planning_extracts_the_same_gaussian_beliefs():
    mean = torch.tensor([[1.0, 2.0]])
    covariance = torch.tensor([[[0.5, 0.2], [0.2, 0.8]]])
    trajectory = BeliefTrajectory([GaussianBelief(mean, covariance)] * 2)
    extracted_mean, extracted_covariance = extract_trajectory_stats(
        trajectory, diagonal_only=False
    )
    _, extracted_variance = extract_trajectory_stats(trajectory)
    torch.testing.assert_close(extracted_mean, mean.unsqueeze(1).expand(-1, 2, -1))
    torch.testing.assert_close(
        extracted_covariance, covariance.unsqueeze(1).expand(-1, 2, -1, -1)
    )
    torch.testing.assert_close(
        extracted_variance, extracted_covariance.diagonal(dim1=-2, dim2=-1)
    )
    assert GreaterThan(0.0)(trajectory).shape == (1, 2, 2)


def test_planner_accepts_the_shared_belief_in_a_small_window():
    environment = Environment()
    environment.set_goal([0.5, 1.5], [-0.5, 0.5])
    planner = Planner(
        SingleIntegrator(), environment, 3, config={"max_iters": 1, "scale": -1}
    )
    mean, covariance, controls, score, history = planner._optimize_window(
        torch.tensor([0.0, 0.0]),
        torch.eye(2) * 0.1,
        init_guess=torch.zeros(3, 2),
        verbose=False,
    )
    assert mean.shape == (1, 4, 2)
    assert covariance.shape == (1, 4, 2, 2)
    assert controls.shape == (3, 2)
    assert 0 <= score <= 1
    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history)).all()
