"""The configured two-block reach-avoid pipeline and its presentation."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch
from scipy.stats import chi2

from planning.runners import run_reach_avoid, setup_problem
from utils import load_config
from visualization.planning import plot_reach_avoid


@pytest.fixture(scope="module")
def problem():
    return setup_problem(
        load_config("configs/scenarios/reach_avoid.yaml"),
        device="cpu",
        with_environment=True,
    )


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
    initial = problem.planner.evaluate_controls(
        problem.rollout, problem.init_guess, spec=spec
    )
    replay = problem.planner.evaluate_controls(
        problem.rollout, result.controls, spec=spec
    )
    initial_at_final_beta = spec.smooth_lower(
        initial.rollout.belief_trajectory, result.smoothing_beta
    )
    assert result.smooth_lower > initial_at_final_beta
    assert result.hard_interval[0] > 0.5
    assert replay.hard_interval == pytest.approx(
        result.hard_interval, abs=1e-6
    )
    torch.testing.assert_close(
        replay.rollout.aux["mean_trace"], result.rollout.aux["mean_trace"]
    )
    assert result.controls.abs().max() <= problem.dyn.u_max
    points = result.rollout.aux["mean_trace"][0]
    workspace, goal = (
        problem.env.single_region("workspace"),
        problem.env.single_region("goal"),
    )

    def inside(region):
        return (
            (points[:, 0] >= region.xmin)
            & (points[:, 0] <= region.xmax)
            & (points[:, 1] >= region.ymin)
            & (points[:, 1] <= region.ymax)
        )

    assert inside(workspace).all()
    assert inside(goal).any()
    assert all(
        not inside(obstacle).any()
        for obstacle in problem.env.by_role("obstacle")
    )
    crossing = points[(points[:, 0] >= 3) & (points[:, 0] <= 5)]
    assert (
        len(crossing) > 0
        and ((crossing[:, 1] > 4) & (crossing[:, 1] < 6)).all()
    )


def test_gradient_reaches_controls(problem):
    parameters = problem.planner._control_parameters(
        problem.init_guess
    ).requires_grad_()
    beliefs = problem.rollout(parameters).belief_trajectory
    spec = problem.env.get_specification(problem.cfg["H"])
    spec.smooth_lower(beliefs, 2.0).backward()
    assert (
        parameters.grad.abs().sum() > 0
        and torch.isfinite(parameters.grad).all()
    )


def test_publication_figure_is_saved_and_zero_obstacles_plot(
    problem, result, tmp_path
):
    path = tmp_path / "reach.png"
    fig, axes = plot_reach_avoid(
        result,
        problem.env,
        dt=problem.cfg["dt"],
        save_path=str(path),
        show=False,
    )
    assert len(axes) == 3
    assert path.exists() and path.with_suffix(".pdf").exists()
    for ax in axes:
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert len(labels) == len(set(labels))
    first_ellipse = next(
        patch
        for patch in axes[0].patches
        if patch.get_label() == "95% belief ellipse"
    )
    eigenvalue = (
        torch.linalg.eigvalsh(result.rollout.aux["cov_trace"][0, 0, :2, :2])
        .max()
        .item()
    )
    assert first_ellipse.width == pytest.approx(
        2 * (chi2.ppf(0.95, 2) * eigenvalue) ** 0.5
    )
    cfg = {**problem.cfg, "obstacles": []}
    empty = setup_problem(cfg, device="cpu", with_environment=True)
    empty_fig, empty_axes = plot_reach_avoid(
        result, empty.env, dt=cfg["dt"], show=False
    )
    labels = [
        text.get_text() for text in empty_axes[1].get_legend().get_texts()
    ]
    assert not any("Obstacle" in label for label in labels)
    plt.close(empty_fig)
    plt.close(fig)


def test_plot_honors_configured_region_styles(problem, result):
    from matplotlib.colors import to_rgba

    cfg = dict(problem.cfg)
    cfg["obstacles"] = [{**cfg["obstacles"][0], "style": {"color": "purple"}}]
    styled = setup_problem(cfg, device="cpu", with_environment=True)
    fig, axes = plot_reach_avoid(result, styled.env, dt=cfg["dt"], show=False)
    obstacle = next(p for p in axes[0].patches if p.get_label() == "Obstacles")
    assert obstacle.get_facecolor() == to_rgba("purple", alpha=0.35)
    plt.close(fig)
