"""The configured two-block reach-avoid pipeline and its presentation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch

from planning.runners import run_reach_avoid, setup_problem
from utils import load_config
from visualization.planning import visualize_reach_avoid


@pytest.fixture(scope="module")
def problem():
    return setup_problem(load_config("configs/scenarios/reach_avoid.yaml"),
                         device="cpu", with_environment=True)


@pytest.fixture(scope="module")
def result():
    return run_reach_avoid(show=False, save=False)


def test_default_geometry_and_custom_names(problem):
    assert len(problem.env.by_role("obstacle")) == 2
    cfg = dict(problem.cfg)
    cfg["workspace"] = {**cfg["workspace"], "name": "room"}
    cfg["goal"] = {**cfg["goal"], "name": "destination"}
    renamed = setup_problem(cfg, device="cpu", with_environment=True)
    spec = renamed.env.get_specification(cfg["H"])
    assert spec.subformula2.subformula.name == "destination"


def test_reach_avoid_returns_safe_bounded_replayable_plan(problem, result):
    spec = problem.env.get_specification(problem.cfg["H"])
    initial = problem.planner.evaluate_controls(problem.rollout, problem.init_guess, spec=spec)
    replay = problem.planner.evaluate_controls(problem.rollout, result.controls, spec=spec)
    assert result.smooth_lower > initial.smooth_lower
    assert result.hard_interval[0] > 0.5
    assert replay.hard_interval == pytest.approx(result.hard_interval, abs=1e-6)
    torch.testing.assert_close(replay.rollout.aux["mean_trace"], result.rollout.aux["mean_trace"])
    assert result.controls.abs().max() <= problem.dyn.u_max
    points = result.rollout.aux["mean_trace"][0]
    workspace, goal = problem.env.single_region("workspace"), problem.env.single_region("goal")

    def inside(region):
        return ((points[:, 0] >= region.xmin) & (points[:, 0] <= region.xmax)
                & (points[:, 1] >= region.ymin) & (points[:, 1] <= region.ymax))

    assert inside(workspace).all()
    assert inside(goal).any()
    assert all(not inside(obstacle).any() for obstacle in problem.env.by_role("obstacle"))
    crossing = points[(points[:, 0] >= 3) & (points[:, 0] <= 5)]
    assert len(crossing) > 0 and ((crossing[:, 1] > 4) & (crossing[:, 1] < 6)).all()


def test_gradient_reaches_controls(problem):
    parameters = problem.planner._control_parameters(problem.init_guess).requires_grad_()
    beliefs = problem.rollout(parameters).belief_trajectory
    spec = problem.env.get_specification(problem.cfg["H"])
    spec.smooth_lower(beliefs, 2.).backward()
    assert parameters.grad.abs().sum() > 0 and torch.isfinite(parameters.grad).all()


def test_three_views_are_saved_and_zero_obstacles_plot(problem, result, tmp_path):
    path = tmp_path / "reach.png"
    views = visualize_reach_avoid(result, problem.env, dt=problem.cfg["dt"],
                                  save_path=str(path), show=False)
    assert len(views) == 3
    assert all((tmp_path / name).exists() for name in
               ("reach.png", "reach_probabilities.png", "reach_optimization.png"))
    cfg = {**problem.cfg, "obstacles": []}
    empty = setup_problem(cfg, device="cpu", with_environment=True)
    visualize_reach_avoid(result, empty.env, dt=cfg["dt"], show=False)
    plt.close("all")


def test_plot_honors_configured_region_styles(problem, result):
    from matplotlib.colors import to_rgba
    from visualization.planning import geometry, plot_reach_avoid

    cfg = dict(problem.cfg)
    cfg["workspace"] = {**cfg["workspace"], "name": "room",
                        "style": {"edgecolor": "purple", "linestyle": ":"}}
    cfg["goal"] = {**cfg["goal"], "name": "destination", "style": {"color": "blue"}}
    styled = setup_problem(cfg, device="cpu", with_environment=True)
    assert geometry(styled.env).goal["style"]["color"] == "blue"
    fig, ax = plot_reach_avoid(None, None, result.rollout.aux["mean_trace"],
                               result.rollout.aux["cov_trace"], styled.env, show=False)
    workspace, goal = ax.patches[:2]
    assert workspace.get_edgecolor() == to_rgba("purple")
    assert goal.get_facecolor() == to_rgba("blue", alpha=0.4)
    plt.close(fig)
