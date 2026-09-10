"""Controlled dynamics and the shared Gaussian belief boundary."""

import numpy as np
import pytest
import torch
from models.dynamics import (
    DoubleIntegrator,
    GaussianBelief,
    SingleIntegrator,
    create_gaussian_belief_trajectory,
    piecewise_signal,
)
from pdstl.base import BeliefTrajectory
from pdstl.operators import Always, GreaterThan
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


def test_step_enclosure_matches_hand_computation_when_a_is_non_negative():
    model = SingleIntegrator(dt=0.5, u_max=1.0, q_std=0.1)
    lower, upper = torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])
    cov = torch.eye(2) * 0.2
    u = torch.tensor([0.4, -0.2])
    d_lower, d_upper = torch.tensor([-0.1, -0.1]), torch.tensor([0.1, 0.1])

    next_lower, next_upper, next_cov = model.step_enclosure(
        lower, upper, cov, u, d_lower, d_upper
    )

    # A = I is entirely non-negative, so A+ = I, A- = 0: no bound swapping.
    torch.testing.assert_close(next_lower, lower + model.dt * u + d_lower)
    torch.testing.assert_close(next_upper, upper + model.dt * u + d_upper)
    torch.testing.assert_close(next_cov, cov + model.Q)


def test_step_enclosure_swaps_bounds_where_a_is_negative():
    """A negative A entry must pull from the *other* bound (A- @ U for L, A- @
    L for U), or the propagated interval would stop being a valid enclosure."""
    model = SingleIntegrator(dt=1.0, u_max=1.0, q_std=0.0)
    model.A = torch.tensor([[1.0, -1.0], [0.0, 1.0]])  # reassigns the buffer
    zero = torch.zeros(2)

    lower, upper = torch.tensor([0.0, 2.0]), torch.tensor([1.0, 3.0])
    next_lower, next_upper, _ = model.step_enclosure(
        lower, upper, torch.zeros(2, 2), zero, zero, zero
    )

    # row 0 = [1, -1]: A+ row = [1, 0], A- row = [0, -1]
    #   L0' = 1*lower[0] + (-1)*upper[1] = 0 - 3 = -3
    #   U0' = 1*upper[0] + (-1)*lower[1] = 1 - 2 = -1
    torch.testing.assert_close(next_lower[0], torch.tensor(-3.0))
    torch.testing.assert_close(next_upper[0], torch.tensor(-1.0))
    assert (next_lower <= next_upper).all()


def test_rollout_enclosure_accumulates_covariance_with_no_dt_factor_and_reaches_gradients():
    model = SingleIntegrator(dt=0.3, u_max=1.0, q_std=0.1)
    v = torch.zeros(4, 2, requires_grad=True)
    lower0, upper0 = torch.tensor([0.0, 0.0]), torch.tensor([0.2, 0.2])

    lower, upper, cov = model.rollout_enclosure(v, lower0, upper0, torch.zeros(2, 2))

    assert lower.shape == upper.shape == (1, 5, 2)
    assert cov.shape == (1, 5, 2, 2)
    # Q is a per-step covariance: four steps accumulate exactly 4*Q, no dt scaling.
    torch.testing.assert_close(cov[0, -1], 4 * model.Q)

    (lower.sum() + upper.sum()).backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_rollout_enclosure_defaults_the_offset_to_zero():
    model = SingleIntegrator(dt=0.3, u_max=1.0, q_std=0.0)
    v = torch.zeros(2, 2)
    lower0 = upper0 = torch.tensor([1.0, 1.0])  # collapsed, zero control

    lower, upper, _ = model.rollout_enclosure(v, lower0, upper0, torch.zeros(2, 2))

    torch.testing.assert_close(lower, upper)
    torch.testing.assert_close(lower[0, 0], lower0)
    torch.testing.assert_close(lower[0, -1], lower0)


def test_planning_extracts_the_same_gaussian_beliefs():
    mean = torch.tensor([[1.0, 2.0]])
    covariance = torch.tensor([[[0.5, 0.2], [0.2, 0.8]]])
    trajectory = BeliefTrajectory([GaussianBelief(mean, mean, covariance)] * 2)
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
    scalar_lower = np.array([1.0, 2.0], dtype=np.float64)
    scalar = create_gaussian_belief_trajectory(
        scalar_lower, scalar_lower + 1.0, np.array([0.25, 1.0])
    )
    assert scalar[0].lower.shape == (1, 1)
    assert scalar[0].lower.dtype == torch.float64

    vector = create_gaussian_belief_trajectory(
        np.zeros((3, 2)), np.ones((3, 2)), np.ones((3, 2, 2))
    )
    assert len(vector) == 3 and vector[0].covariance.shape == (1, 2, 2)

    lower = torch.zeros(3, 2, dtype=torch.float64, requires_grad=True)
    trajectory = create_gaussian_belief_trajectory(lower, lower + 1.0, torch.ones(3, 2))
    assert len(trajectory) == 3 and trajectory[0].lower.shape == (1, 2)
    assert trajectory[0].lower.device == lower.device
    GreaterThan(0.0)(trajectory).sum().backward()
    assert lower.grad is not None and torch.isfinite(lower.grad).all()


@pytest.mark.parametrize(
    "lower,upper,covariance,message",
    [
        (np.zeros(3), np.zeros(2), np.zeros(3), "matching shape"),
        (np.zeros((3, 2)), np.zeros((3, 2)), np.zeros((2, 2)), "same number of steps"),
    ],
)
def test_gaussian_trajectory_factory_rejects_mismatched_shapes(lower, upper, covariance, message):
    with pytest.raises(ValueError, match=message):
        create_gaussian_belief_trajectory(lower, upper, covariance)


# --- Input validation --------------------------------------------------


@pytest.mark.parametrize(
    "lower,upper,covariance,message",
    [
        ([[float("nan"), 0.0]], [[1.0, 1.0]], [[1.0, 1.0]], "lower must be finite"),
        ([[0.0, 0.0]], [[float("inf"), 1.0]], [[1.0, 1.0]], "upper must be finite"),
        ([[0.0, 0.0]], [[1.0, 1.0]], [[float("-inf"), 1.0]], "covariance must be finite"),
    ],
)
def test_gaussian_belief_rejects_nonfinite_inputs(lower, upper, covariance, message):
    t = lambda v: torch.as_tensor(v, dtype=torch.float64)
    with pytest.raises(ValueError, match=message):
        GaussianBelief(t(lower), t(upper), t(covariance))


def test_gaussian_belief_rejects_an_asymmetric_full_covariance():
    lower = upper = torch.zeros(1, 2)
    covariance = torch.tensor([[[1.0, 2.0], [3.0, 1.0]]])

    with pytest.raises(ValueError, match="must be symmetric"):
        GaussianBelief(lower, upper, covariance)


def test_gaussian_belief_rejects_a_symmetric_non_positive_semidefinite_covariance():
    lower = upper = torch.zeros(1, 2)
    covariance = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]])  # eigenvalues -1, 3

    with pytest.raises(ValueError, match="positive semi-definite"):
        GaussianBelief(lower, upper, covariance)


def test_gaussian_belief_accepts_a_covariance_actually_propagated_by_rollout():
    """A regression guard: the new symmetry/PSD checks must not reject a
    covariance that came from real (float-roundoff-bearing) propagation."""
    model = DoubleIntegrator(dt=0.2, u_max=1.0, q_std=0.1)
    x, P = torch.zeros(4), torch.eye(4) * 0.3
    for _ in range(10):
        x, P = model.step(x, P, torch.tensor([0.3, -0.1]))

    GaussianBelief(x.unsqueeze(0), x.unsqueeze(0), P.unsqueeze(0))  # must not raise


def test_gradient_reaches_controls_through_the_full_enclosure_belief_path():
    """controls -> rollout_enclosure -> belief trajectory -> predicate
    probability, the general enclosure machinery end to end."""
    model = SingleIntegrator(dt=1.0, u_max=3.0, q_std=0.5)
    controls = torch.zeros(4, 2, requires_grad=True)
    lower0, upper0 = torch.tensor([48.0, 0.0]), torch.tensor([49.0, 0.0])
    covariance0 = torch.eye(2) * 4.0

    lower, upper, covariance = model.rollout_enclosure(controls, lower0, upper0, covariance0)
    traj = create_gaussian_belief_trajectory(lower[0], upper[0], covariance[0])

    formula = Always(GreaterThan(50.0), interval=[0, 1])
    formula(traj)[..., 0].sum().backward()

    assert controls.grad is not None
    assert torch.isfinite(controls.grad).all()
    assert controls.grad.abs().sum() > 0


def test_piecewise_signal_returns_time_mean_and_variance():
    time, mean, variance = piecewise_signal([[45.0, 4.0], [55.0, 9.0]])
    np.testing.assert_allclose(time, [0.0, 1.0])
    np.testing.assert_allclose(mean, [45.0, 55.0])
    np.testing.assert_allclose(variance, [4.0, 9.0])
    assert len(piecewise_signal()[0]) == 7


def test_piecewise_signal_rejects_non_pair_values():
    with pytest.raises(ValueError, match=r"\[T, 2\]"):
        piecewise_signal([[45.0, 4.0, 1.0]])
