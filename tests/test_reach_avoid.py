"""The configurable double-slit reach-avoid pipeline and presentation."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch
from scipy.stats import chi2

from planning.runners import run_reach_avoid, setup_problem
from planning.planner import IterationRecord
from utils import load_config
from visualization.animation import animate_reach_avoid
from visualization.live_plots import create_reach_avoid_live_view
from visualization.planning import (
    COLORS,
    plot_reach_avoid,
    plot_reach_avoid_pdstl,
    reach_avoid_certificate_trace,
)


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
    assert len(problem.env.by_role("obstacle")) == 3
    assert problem.cfg["x0_mean"] == [-3.0, 0.0]
    workspace = problem.env.single_region("workspace")
    goal = problem.env.single_region("goal")
    assert workspace.x == tuple(problem.cfg["workspace"]["x"])
    assert workspace.y == tuple(problem.cfg["workspace"]["y"])
    assert goal.x == tuple(problem.cfg["goal"]["x"])
    assert goal.y == tuple(problem.cfg["goal"]["y"])
    assert {
        region.name: (region.x, region.y)
        for region in problem.env.by_role("obstacle")
    } == {
        obstacle["name"]: (tuple(obstacle["x"]), tuple(obstacle["y"]))
        for obstacle in problem.cfg["obstacles"]
    }
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
    assert result.hard_interval[0] > 0.9
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
    crossing = points[(points[:, 0] >= 0) & (points[:, 0] <= 1.5)]
    middle = problem.env.region("middle_wall")
    top = problem.env.region("top_wall")
    assert (
        len(crossing) > 0
        and (
            (crossing[:, 1] > middle.ymax) & (crossing[:, 1] < top.ymin)
        ).all()
    )
    assert float(points[:, 1].max()) > 2.0


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
    fig, axis = plot_reach_avoid(
        result,
        problem.env,
        title="Double Slit",
        save_path=str(path),
        show=False,
    )
    assert len(fig.axes) == 1 and fig.axes[0] is axis
    assert path.exists() and path.with_suffix(".pdf").exists()
    assert axis.get_legend() is None
    assert axis.get_title() == (
        f"Double Slit | P↓(φ)={result.hard_interval[0]:.3f}"
    )
    assert axis.get_xlim() == pytest.approx(
        tuple(problem.cfg["workspace"]["x"])
    )
    assert axis.get_ylim() == pytest.approx(
        tuple(problem.cfg["workspace"]["y"])
    )
    assert {text.get_text() for text in axis.texts} >= {"G"}
    assert sum(p.get_label() == "Obstacles" for p in axis.patches) == 3
    assert {collection.get_label() for collection in axis.collections} >= {
        "Start",
        "Terminal belief mean",
    }
    first_ellipse = next(
        patch
        for patch in axis.patches
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
    empty_fig, empty_axis = plot_reach_avoid(result, empty.env, show=False)
    assert not any(
        patch.get_label() == "Obstacles" for patch in empty_axis.patches
    )
    plt.close(empty_fig)
    plt.close(fig)


def test_plot_honors_configured_region_styles(problem, result):
    from matplotlib.colors import to_rgba

    cfg = dict(problem.cfg)
    cfg["obstacles"] = [
        {
            **cfg["obstacles"][0],
            "name": "shifted_wall",
            "x": [-1.0, -0.25],
            "y": [2.0, 3.0],
            "style": {"color": "purple"},
        }
    ]
    styled = setup_problem(cfg, device="cpu", with_environment=True)
    fig, axis = plot_reach_avoid(result, styled.env, show=False)
    obstacle = next(p for p in axis.patches if p.get_label() == "Obstacles")
    assert styled.env.region("shifted_wall").x == (-1.0, -0.25)
    assert (obstacle.get_x(), obstacle.get_y()) == (-1.0, 2.0)
    assert (obstacle.get_width(), obstacle.get_height()) == (0.75, 1.0)
    assert obstacle.get_facecolor() == to_rgba("purple", alpha=0.45)
    assert obstacle.get_hatch() == "//"
    plt.close(fig)


def test_overall_pdstl_certificate_is_plotted_over_prediction_time(
    problem, result, tmp_path
):
    bounds = reach_avoid_certificate_trace(
        result.rollout.belief_trajectory, problem.env
    )
    assert bounds.shape == (problem.cfg["H"] + 1, 2)
    assert bounds[-1] == pytest.approx(result.hard_interval, abs=1e-6)
    path = tmp_path / "reach_avoid_pdstl.png"
    fig, axis = plot_reach_avoid_pdstl(
        result,
        problem.env,
        dt=problem.cfg["dt"],
        save_path=path,
        show=False,
    )
    assert path.exists() and path.with_suffix(".pdf").exists()
    assert len(fig.axes) == 1 and fig.axes[0] is axis
    assert axis.lines[0].get_color() == COLORS["score"]
    assert len(axis.lines[0].get_xdata()) == problem.cfg["H"] + 1
    assert "final P↓(φ)=" in axis.get_title()
    plt.close(fig)


def test_live_view_updates_path_and_overall_pdstl_trace(problem, result):
    spec = problem.env.get_specification(problem.cfg["H"])
    fig, axes, observe, finish = create_reach_avoid_live_view(
        problem.env,
        lambda controls: problem.planner.evaluate_controls(
            problem.rollout, controls, spec=spec
        ),
        dt=problem.cfg["dt"],
        title="Double Slit",
        max_iters=problem.planner.cfg["max_iters"],
    )
    record = IterationRecord(
        result.controls,
        result.final_loss,
        result.smooth_lower,
        result.hard_interval,
        result.smoothing_beta,
    )
    observe(0, record)
    assert len(axes[0].lines[0].get_xdata()) == problem.cfg["H"] + 1
    assert len(axes[1].lines[0].get_xdata()) == problem.cfg["H"] + 1
    assert axes[0].lines[0].get_color() == COLORS["mean"]
    assert axes[1].lines[0].get_color() == COLORS["score"]
    finish(result)
    plt.close(fig)


def test_animation_is_the_same_single_environment_view(problem, result):
    fig, axis, movie = animate_reach_avoid(
        result,
        problem.env,
        dt=problem.cfg["dt"],
        title="Double Slit",
        show=False,
    )
    assert len(fig.axes) == 1 and fig.axes[0] is axis
    assert axis.get_title() == (
        f"Double Slit | P↓(φ)={result.hard_interval[0]:.3f}"
    )
    movie._draw_next_frame(1, blit=False)
    path = axis.lines[1]
    assert len(path.get_xdata()) == 2
    assert movie is not None
    movie._draw_was_started = True
    plt.close(fig)
