"""Controlled dynamics and the shared Gaussian belief boundary."""

import numpy as np
import pytest
import torch
from models.dynamics import (
    DoubleIntegrator,
    GaussianBelief,
    IntervalBelief,
    SingleIntegrator,
    create_gaussian_belief_trajectory,
    create_interval_belief_trajectory,
    linear_system,
    piecewise_signal,
    sinusoidal_input,
)
from pdstl.base import BeliefTrajectory
from pdstl.operators import GreaterThan, LessThan
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
    trajectory = BeliefTrajectory([GaussianBelief(mean, covariance, 0.0)] * 2)
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


def test_gaussian_trajectory_factory_preserves_supported_shapes_and_types():
    scalar_mean = np.array([1.0, 2.0], dtype=np.float64)
    scalar = create_gaussian_belief_trajectory(
        scalar_mean, np.array([0.25, 1.0]), sigma_multiplier=1.5
    )
    assert scalar[0].mean.shape == (1, 1)
    assert scalar[0].mean.dtype == torch.float64

    vector = create_gaussian_belief_trajectory(
        np.zeros((3, 2)), np.ones((3, 2, 2)), sigma_multiplier=1.0
    )
    assert len(vector) == 3 and vector[0].var.shape == (1, 2, 2)

    mean = torch.zeros(2, 3, 2, dtype=torch.float64, requires_grad=True)
    variance = torch.ones(2, 3, 2, dtype=torch.float64)
    batched = create_gaussian_belief_trajectory(
        mean, variance, sigma_multiplier=0.0
    )
    assert len(batched) == 3 and batched[0].mean.shape == (2, 2)
    assert batched[0].mean.device == mean.device
    GreaterThan(0.0)(batched).sum().backward()
    assert mean.grad is not None and torch.isfinite(mean.grad).all()


@pytest.mark.parametrize(
    "mean,variance",
    [
        (np.zeros(3), np.zeros((3, 1))),
        (np.zeros((3, 2)), np.zeros((3, 3))),
        (np.zeros((3, 2)), np.zeros((3, 2, 3))),
        (np.zeros((2, 3, 2)), np.zeros((2, 3, 3))),
        (np.zeros((2, 3, 2)), np.zeros((2, 3, 2, 3))),
    ],
)
def test_gaussian_trajectory_factory_rejects_inexact_shapes(mean, variance):
    with pytest.raises(ValueError, match="exactly match"):
        create_gaussian_belief_trajectory(mean, variance, sigma_multiplier=1.0)


# Bounded-state belief: an alternative upstream source, unused by the examples


@pytest.mark.parametrize(
    "lower,upper,message",
    [
        (torch.zeros(1, 2), torch.zeros(1, 3), "matching shapes"),
        (torch.zeros(3), torch.zeros(3), r"\[B,D\]"),
        (torch.tensor([[float("nan")]]), torch.tensor([[1.0]]), "finite"),
        (torch.tensor([[2.0]]), torch.tensor([[1.0]]), "lower <= upper"),
    ],
)
def test_interval_belief_rejects_malformed_bounds(lower, upper, message):
    with pytest.raises(ValueError, match=message):
        IntervalBelief(lower, upper)


def test_interval_belief_rejects_invalid_predicate_dimension():
    belief = IntervalBelief(torch.zeros(1, 1), torch.ones(1, 1))
    with pytest.raises(ValueError, match="outside the state"):
        belief.probability_bounds(GreaterThan(0.5, dim=5))


@pytest.mark.parametrize(
    "lower,upper,expected",
    [
        (5.0, 5.0, [1.0, 1.0]),  # x_lower == threshold -> guaranteed
        (6.0, 8.0, [1.0, 1.0]),  # entirely above -> guaranteed
        (3.0, 5.0, [0.0, 1.0]),  # x_upper == threshold -> not violated
        (3.0, 7.0, [0.0, 1.0]),  # crossing
        (1.0, 4.9, [0.0, 0.0]),  # entirely below -> violated
    ],
)
def test_interval_belief_greater_than_is_inclusive(lower, upper, expected):
    belief = IntervalBelief(torch.tensor([[lower]]), torch.tensor([[upper]]))
    got = belief.probability_bounds(GreaterThan(5.0))
    torch.testing.assert_close(got[0], torch.tensor(expected))


@pytest.mark.parametrize(
    "lower,upper,expected",
    [
        (5.0, 5.0, [1.0, 1.0]),  # x_upper == threshold -> guaranteed
        (1.0, 4.0, [1.0, 1.0]),  # entirely below -> guaranteed
        (5.0, 7.0, [0.0, 1.0]),  # x_lower == threshold -> not violated
        (3.0, 7.0, [0.0, 1.0]),  # crossing
        (5.1, 8.0, [0.0, 0.0]),  # entirely above -> violated
    ],
)
def test_interval_belief_less_than_is_inclusive(lower, upper, expected):
    belief = IntervalBelief(torch.tensor([[lower]]), torch.tensor([[upper]]))
    got = belief.probability_bounds(LessThan(5.0))
    torch.testing.assert_close(got[0], torch.tensor(expected))


def test_interval_trajectory_factory_builds_one_belief_per_step():
    trajectory = create_interval_belief_trajectory([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
    assert len(trajectory) == 3
    assert trajectory[0].lower.shape == (1, 1)
    assert trajectory[0].upper.shape == (1, 1)


def test_interval_trajectory_factory_rejects_mismatched_traces():
    with pytest.raises(ValueError, match="matching shapes"):
        create_interval_belief_trajectory([1.0, 2.0], [1.0, 2.0, 3.0])


# Generated scalar signals: alternative upstream sources for Gaussian beliefs


def test_sinusoidal_input_is_a_scalar_control():
    np.testing.assert_allclose(sinusoidal_input(np.array([0.0, 0.5])), [0.0, 15.0])


def test_linear_system_propagates_mean_and_variance():
    time = np.linspace(0.0, 1.0, 5)
    mean, variance = linear_system(
        a=0.01, b=1.0, g=2.0, q=2.5, mu=50.0, P=0.15, t=time,
        control_func=sinusoidal_input,
    )
    assert mean.shape == variance.shape == time.shape
    assert mean[0] == 50.0 and variance[0] == 0.15
    assert np.all(np.diff(variance) > 0)
    assert np.isfinite(mean).all()


def test_piecewise_signal_returns_time_mean_and_variance():
    time, mean, variance = piecewise_signal([[45.0, 4.0], [55.0, 9.0]])
    np.testing.assert_allclose(time, [0.0, 1.0])
    np.testing.assert_allclose(mean, [45.0, 55.0])
    np.testing.assert_allclose(variance, [4.0, 9.0])
    assert len(piecewise_signal()[0]) == 7


def test_piecewise_signal_rejects_non_pair_values():
    with pytest.raises(ValueError, match=r"\[T, 2\]"):
        piecewise_signal([[45.0, 4.0, 1.0]])
