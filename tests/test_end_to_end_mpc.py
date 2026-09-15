"""Receding-horizon smoke test on the precise-Gaussian pdSTL pipeline:

    current belief -> H-step prediction -> pdSTL -> optimise controls
    -> execute only u_0 -> simulated step -> replan

Seeded, so the simulated process noise is reproducible.
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch

from models.beliefs import create_gaussian_belief_trajectory
from planning.examples import (
    end_to_end_setup,
    load_end_to_end_config,
    run_end_to_end_mpc_reach,
)
from models.rollouts import gaussian_rollout
from planning.planner import Planner
from utils import get_device
from visualization.robustness import plot_mpc_reach

NAME = "end_to_end_mpc_reach"


@pytest.fixture(scope="module")
def config():
    return load_end_to_end_config(NAME)


@pytest.fixture(scope="module")
def setup(config):
    cfg, _ = config
    return end_to_end_setup(cfg, get_device())


@pytest.fixture(scope="module")
def result():
    return run_end_to_end_mpc_reach(save=False, verbose=False)


def _replans(result):
    return len(result["plan_mean_traces"])


def _assert_valid_covariance(cov):
    torch.testing.assert_close(cov, cov.transpose(-1, -2), atol=1e-6, rtol=0)
    assert (torch.linalg.eigvalsh(cov) >= -1e-6).all()


# A
def test_planner_replans_multiple_times(result):
    assert _replans(result) >= 2
    assert len(result["hard_scores"]) == len(result["plan_controls"]) == _replans(result)


# B
def test_only_the_first_control_of_each_plan_is_executed(result, setup, config):
    cfg, _ = config
    dyn = setup[0]
    means, covs, applied = result["mean_trace"][0], result["cov_trace"][0], result["u_trace"][0]

    assert applied.shape == (_replans(result), 2)
    assert means.shape[0] == covs.shape[0] == _replans(result) + 1
    for k, plan_u in enumerate(result["plan_controls"]):
        assert plan_u.shape == (cfg["H"], 2)
        torch.testing.assert_close(applied[k], plan_u[0])
        predicted_mean, predicted_cov = dyn.step(means[k], covs[k], applied[k])
        torch.testing.assert_close(covs[k + 1], predicted_cov)
        # the executed step differs from the prediction only by process noise
        assert (means[k + 1] - predicted_mean).norm() < 5 * cfg["q_std"] * math.sqrt(2)


def test_without_noise_each_step_applies_exactly_the_first_planned_control(setup, config):
    cfg, planner_cfg = config
    dyn, x0_mean, x0_cov, _, spec = setup
    planner = Planner(dyn, None, cfg["H"], config=planner_cfg)

    result = planner.run_receding_horizon(
        (x0_mean, x0_cov),
        make_rollout=lambda state: gaussian_rollout(dyn, *state),
        execute=lambda state, u: dyn.step(*state, u),
        is_done=lambda state: False,
        spec=spec,
        max_steps=3,
        init_guess=torch.tensor(cfg["init_control"]).repeat(cfg["H"], 1),
    )

    assert result["stopped_reason"] == "max_steps"
    means = [mean for mean, _ in result["states"]]
    for k, best in enumerate(result["candidates"]):
        torch.testing.assert_close(means[k + 1], dyn.A @ means[k] + dyn.B @ best.controls[0])


# C
def test_control_limits_are_respected(result):
    u_max = result["u_max"]
    assert (result["u_trace"].abs() <= u_max).all()
    for plan_u in result["plan_controls"]:
        assert (plan_u.abs() <= u_max).all()


# D
def test_states_covariances_controls_and_scores_are_finite(result):
    for key in ("mean_trace", "cov_trace", "u_trace"):
        assert torch.isfinite(result[key]).all(), key
    assert all(math.isfinite(s) for s in result["hard_scores"])
    assert all(math.isfinite(j) for j in result["objectives"])
    for cov in result["cov_trace"][0]:
        _assert_valid_covariance(cov)
    for plan in result["plan_mean_traces"]:
        assert torch.isfinite(plan).all()


# E
def test_warm_start_is_the_previous_plan_shifted_by_one(result, config):
    cfg, _ = config
    first = result["warm_starts"][0]
    torch.testing.assert_close(first, torch.tensor(cfg["init_control"]).repeat(cfg["H"], 1))

    for previous, warm in zip(result["plan_controls"], result["warm_starts"][1:]):
        assert warm.shape == previous.shape == (cfg["H"], 2)
        torch.testing.assert_close(warm[:-1], previous[1:])
        torch.testing.assert_close(warm[-1], previous[-1])


# F, G
def test_executed_trajectory_reaches_the_target(result, config):
    cfg, _ = config
    executed_x = result["mean_trace"][0, :, cfg["dim"]]

    assert result["stopped_reason"] == "goal_reached"
    assert executed_x[-1] >= cfg["threshold"]
    assert (executed_x[:-1] < cfg["threshold"]).all()  # stopped at the first arrival
    assert executed_x[-1] - executed_x[0] >= 1.5


# H
def test_progress_comes_from_the_pdstl_objective(result, config):
    cfg, planner_cfg = config
    assert planner_cfg["w_dist"] == planner_cfg["w_obs"] == planner_cfg["w_visit"] == 0
    assert planner_cfg["w_phi"] > 0

    # the optimiser moved away from the initial guess in the very first window
    init = torch.tensor(cfg["init_control"])
    assert (result["plan_controls"][0][0] - init).abs().max() > 0.1
    assert result["hard_scores"][-1] > result["hard_scores"][0]


# I
def test_planning_windows_have_finite_nonzero_gradients(result, setup, config):
    cfg, planner_cfg = config
    dyn, _, _, _, spec = setup
    planner = Planner(dyn, None, cfg["H"], config=planner_cfg)
    means, covs = result["mean_trace"][0], result["cov_trace"][0]

    for k, warm in enumerate(result["warm_starts"]):
        v = planner._init_controls(warm)
        mean, cov = dyn(v, means[k], covs[k])
        robustness = spec(create_gaussian_belief_trajectory(mean[0], cov[0]))[0, 0, 0]
        robustness.backward()

        assert torch.isfinite(v.grad).all()
        if robustness.item() < 1 - 1e-6:  # not saturated in float32
            assert v.grad.abs().sum() > 0, k


# J
def test_predicted_plans_and_executed_trajectory_are_distinct(result, config):
    cfg, _ = config
    means = result["mean_trace"][0]

    for k, plan in enumerate(result["plan_mean_traces"]):
        assert plan.shape == (1, cfg["H"] + 1, 2)
        torch.testing.assert_close(plan[0, 0], means[k])  # each plan starts at the executed belief
        assert not torch.equal(plan[0, 1], means[k + 1])  # execution is not the prediction
    assert not result["mean_trace"].requires_grad


def test_mpc_plot_draws_state_score_and_control_panels(result, config):
    cfg, _ = config
    fig, axes = plot_mpc_reach(cfg["dt"], result, cfg["threshold"], dim=cfg["dim"])

    assert len(axes) == 3
    assert len(axes[0].lines) == _replans(result) + 2  # plans, executed, threshold
    fig.canvas.draw()
    plt.close(fig)
