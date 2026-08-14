"""Audit Groups G-J: smoothing, planner integration, evaluation, complexity."""

from __future__ import annotations

import inspect
import pathlib

import numpy as np
import pytest
import torch
import yaml

from evaluation.metrics import evaluate_rollouts
from evaluation.monte_carlo import rollout_controls
from pdstl.base import BeliefTrajectory
from pdstl.operators import Always, And, Eventually, Or, STL_Formula, Temporal_Operator, Until
from planning.dynamics import SingleIntegrator
from planning.environment import Environment
from planning.planner import Planner, TorchGaussianBelief

from conftest import ConstantFormula, REPO_ROOT


TRACE = np.array(
    [[0.15, 0.35], [0.75, 0.85], [0.30, 0.65], [0.90, 0.95]], dtype=float
)
OTHER = np.array(
    [[0.60, 0.80], [0.25, 0.55], [0.85, 0.90], [0.40, 0.70]], dtype=float
)
BETAS = [1, 5, 10, 50, 100, 500]


def _values(formula, scale):
    return formula(None, scale=scale).detach().cpu().numpy()


@pytest.mark.parametrize(
    "formula",
    [
        Always(ConstantFormula(TRACE), interval=[0, 3]),
        Eventually(ConstantFormula(TRACE), interval=[0, 3]),
        pytest.param(
            Until(ConstantFormula(TRACE), ConstantFormula(OTHER), interval=[0, 3]),
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "G1: smooth Until retains the exclusive pointwise-min path and "
                    "does not converge to corrected exact StoRI Until"
                ),
            ),
        ),
    ],
    ids=["always", "eventually", "until"],
)
def test_g1_smoothing_converges_to_corresponding_exact_implementation(formula):
    exact = _values(formula, -1)
    errors = [np.max(np.abs(_values(formula, beta) - exact)) for beta in BETAS]
    assert errors[-1] < errors[0]
    assert errors[-1] < 0.01


@pytest.mark.parametrize(
    "formula,scale",
    [
        (Always(ConstantFormula([[0.05, 0.05]] * 3), interval=[0, 2]), 10),
        (Eventually(ConstantFormula([[0.99, 0.99]] * 3), interval=[0, 2]), 10),
        (Eventually(ConstantFormula([[0.99, 0.99]] * 3), interval=[0, 2]), 50),
    ],
    ids=["smooth-min-beta10", "smooth-max-beta10", "smooth-max-beta50"],
)
@pytest.mark.xfail(
    strict=True, reason="G2: unnormalized log-sum-exp leaves probability interval"
)
def test_g2_smooth_temporal_output_stays_in_unit_interval(formula, scale):
    observed = _values(formula, scale)
    assert np.all(observed >= 0.0)
    assert np.all(observed <= 1.0)


@pytest.mark.parametrize("operator", [And, Or])
@pytest.mark.parametrize("scale", [-1, 10, 50])
def test_g3_boolean_nodes_ignore_requested_smoothing(operator, scale):
    formula = operator(ConstantFormula(TRACE), ConstantFormula(OTHER))
    assert _values(formula, scale) == pytest.approx(_values(formula, -1), abs=1e-7)


def _and_lower_gradient(lower, scale):
    left_lower = torch.tensor(lower, dtype=torch.float32, requires_grad=True)
    right_lower = torch.tensor(lower, dtype=torch.float32, requires_grad=True)
    left = torch.stack([left_lower, torch.tensor(0.9)]).reshape(1, 1, 2)
    right = torch.stack([right_lower, torch.tensor(0.9)]).reshape(1, 1, 2)
    output = And(ConstantFormula(left), ConstantFormula(right))(None, scale=scale)[0, 0, 0]
    output.backward()
    return output.item(), left_lower.grad.item(), right_lower.grad.item()


@pytest.mark.parametrize("lower", [0.4, 0.5, 0.6])
@pytest.mark.parametrize("scale", [-1, 50], ids=["exact", "requested-smooth"])
def test_g4_requested_smooth_and_retains_hard_frechet_gradient(lower, scale):
    output, grad_left, grad_right = _and_lower_gradient(lower, scale)
    expected_output = max(2.0 * lower - 1.0, 0.0)
    expected_gradient = 0.0 if lower < 0.5 else 1.0
    assert output == pytest.approx(expected_output, abs=1e-7)
    assert grad_left == pytest.approx(expected_gradient)
    assert grad_right == pytest.approx(expected_gradient)


def test_g4_smooth_temporal_gradient_is_finite():
    trace = torch.tensor(TRACE, dtype=torch.float32, requires_grad=True)
    output = Always(ConstantFormula(trace), interval=[0, 3])(None, scale=10)[0, 0, 0]
    output.backward()
    assert torch.isfinite(trace.grad).all()
    assert torch.count_nonzero(trace.grad[:, 0]) > 1


@pytest.mark.parametrize(
    "score,expected_gradient",
    [(-0.05, 0.0), (0.2, -5.0), (1.05, 0.0)],
)
def test_g5_planner_clamp_log_gradient(score, expected_gradient):
    value = torch.tensor(score, requires_grad=True)
    loss = -torch.log(torch.clamp(value, min=1e-6, max=1.0))
    loss.backward()
    assert value.grad.item() == pytest.approx(expected_gradient, abs=1e-6)


class _DummyDynamics(torch.nn.Module):
    device = "cpu"
    u_max = 1.0
    Q = torch.eye(2) * 0.01

    def bound_control(self, values):
        return torch.tanh(values)

    def forward(self, values, x0_mean, x0_cov):
        controls = self.bound_control(values)
        positions = [x0_mean]
        current = x0_mean
        for control in controls:
            current = current + 0.1 * control
            positions.append(current)
        mean = torch.stack(positions).unsqueeze(0)
        covariance = x0_cov.expand(len(positions), -1, -1).unsqueeze(0)
        return mean, covariance


class _DummyEnvironment:
    goal = None
    obstacles = []
    circle_obstacles = []
    moving_obstacles = []
    visit_regions = []
    timed_visit_regions = []
    choice_region_groups = []


class _ScaleSpyFormula(STL_Formula):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.values = []
        self.smooth_gradients = []

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        probability = torch.sigmoid(belief_trajectory[1].mean_full[:, 0])
        if scale > 0:
            probability = probability + 0.1
            if probability.requires_grad:
                probability.register_hook(
                    lambda gradient: self.smooth_gradients.append(gradient.detach().clone())
                )
        self.calls.append(scale)
        self.values.append(probability.detach().item())
        pair = torch.stack([probability, probability], dim=-1)
        return pair.unsqueeze(1).expand(-1, len(belief_trajectory), -1)


def _run_one_planner_iteration(smoothing_marker):
    torch.manual_seed(140826)
    config = {
        "max_iters": 1,
        "alpha": 2.0,
        "w_phi": 1.0,
        "w_u": 0.0,
        "w_du": 0.0,
        "w_dist": 0.0,
        "w_obs": 0.0,
        "w_visit": 0.0,
    }
    if smoothing_marker is not None:
        config["stl_smoothing_scale"] = smoothing_marker
    planner = Planner(_DummyDynamics(), _DummyEnvironment(), T=1, config=config)
    formula = _ScaleSpyFormula()
    result = planner._optimize_window(
        torch.zeros(2), torch.eye(2) * 0.01, spec=formula, verbose=False
    )
    return formula, result


def test_h1_smooth_objective_but_exact_selection_and_reporting_value():
    formula, result = _run_one_planner_iteration(10.0)
    assert formula.calls == [10.0, -1]
    assert formula.smooth_gradients
    assert result[3] == pytest.approx(formula.values[1])
    assert formula.values[0] > formula.values[1]


def test_h1_missing_smoothing_scale_selects_exact_path():
    formula, result = _run_one_planner_iteration(None)
    assert formula.calls == [-1.0]
    assert result[3] == pytest.approx(formula.values[0])


def test_h2_scenario_smoothing_configuration_inventory():
    with (REPO_ROOT / "configs/planning.yaml").open() as stream:
        planning = yaml.safe_load(stream)
    with (REPO_ROOT / "configs/scenarios/two_gap_reach_avoid.yaml").open() as stream:
        two_gap = yaml.safe_load(stream)
    with (REPO_ROOT / "experiments/crazyflie/components/config.yml").open() as stream:
        crazyflie = yaml.safe_load(stream)
    assert "stl_smoothing_scale" not in planning
    assert "stl_smoothing_scale" not in two_gap.get("planner", {})
    assert crazyflie["planner"]["config"]["stl_smoothing_scale"] == 50.0


@pytest.mark.xfail(
    strict=True, reason="I1: rollout initial states are cloned from x0_mean"
)
def test_i1_rollouts_sample_initial_covariance():
    dynamics = SingleIntegrator(q_std=0.05)
    rollouts = rollout_controls(
        dynamics,
        torch.zeros(2),
        torch.eye(2) * 0.25,
        torch.zeros(1, 2),
        num_rollouts=32,
        seed=140826,
    )
    assert torch.var(rollouts[:, 0, :], dim=0).min().item() > 0.0


@pytest.mark.xfail(
    strict=True, reason="I2: manual metrics ignore the goal time interval"
)
def test_i2_empirical_metrics_evaluate_same_timed_goal_formula():
    environment = Environment()
    environment.set_goal([1.0, 2.0], [1.0, 2.0], interval=[0, 0])
    rollout = torch.tensor([[[0.0, 0.0], [1.5, 1.5]]])
    metrics = evaluate_rollouts(rollout, environment)

    beliefs = [
        TorchGaussianBelief(point.unsqueeze(0), torch.zeros(1, 2, 2))
        for point in rollout[0]
    ]
    formula_score = environment.get_specification(T=1)(BeliefTrajectory(beliefs))[
        0, 0, 0
    ].item()
    formula_satisfied = formula_score >= 0.5
    assert metrics["satisfaction_rate"] == float(formula_satisfied)


@pytest.mark.xfail(
    strict=True, reason="I3: empirical satisfaction has no confidence interval"
)
def test_i3_empirical_satisfaction_includes_confidence_interval():
    environment = Environment()
    environment.set_goal([-1.0, 1.0], [-1.0, 1.0])
    metrics = evaluate_rollouts(torch.zeros(4, 2, 2), environment)
    assert "satisfaction_rate_ci_lower" in metrics
    assert "satisfaction_rate_ci_upper" in metrics


def test_j_dense_shift_and_repeated_until_prefix_structure_present():
    shift_source = inspect.getsource(Temporal_Operator._apply_shift)
    until_source = inspect.getsource(Until.robustness_trace)
    benchmark_source = (REPO_ROOT / "src/planning/benchmark_complexity.py").read_text()
    assert "torch.matmul" in shift_source
    assert "for t in range(T)" in until_source
    assert "for tau in range(start, end + 1)" in until_source
    assert "phi[:, t:tau, :]" in until_source
    assert "O(T)" in benchmark_source
