"""The configurable reach-avoid planner and its two stlpy examples."""

from functools import reduce

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch
from scipy.stats import chi2

from models.rollouts import create_gaussian_belief_trajectory
from pdstl.operators import Always, And, Eventually, Or
from planning.environment import (
    Environment,
    RectangleRegion,
    build_reach_avoid_environment,
    inside,
    outside,
)
from planning.planner import IterationRecord
from planning.runners import run_reach_avoid, setup_problem
from utils import load_config
from visualization.animation import animate_reach_avoid
from visualization.live_plots import create_reach_avoid_live_view
from visualization.planning import plot_reach_avoid

EXAMPLES = ("narrow_passage", "either_or")
CONFIG = {
    "workspace": {"x": [0, 10], "y": [0, 10]},
    "obstacles": {"wall": {"x": [3, 5], "y": [4, 6]}},
    "goals": {
        "a": {"x": [7, 8], "y": [8, 9]},
        "b": {"x": [8, 9], "y": [1, 2]},
    },
    "visit": [
        {
            "dwell": 2,
            "regions": {
                "t1": {"x": [1, 2], "y": [6, 7]},
                "t2": {"x": [7, 8], "y": [4, 5]},
            },
        }
    ],
}


def _problem(name, **planner):
    cfg = load_config(f"configs/scenarios/{name}.yaml")
    cfg["planner"] = {**cfg["planner"], **planner}
    return setup_problem(cfg, device="cpu", with_environment=True)


def _inside(points, region):
    return (
        (points[:, 0] >= region.xmin)
        & (points[:, 0] <= region.xmax)
        & (points[:, 1] >= region.ymin)
        & (points[:, 1] <= region.ymax)
    )


@pytest.fixture(scope="module")
def short():
    """A 20-iteration either-or solve from its first route."""
    problem = _problem("either_or", max_iters=20)
    spec = problem.env.get_specification(problem.cfg["H"])
    result = problem.planner.optimize_window(
        problem.rollout, spec=spec, init_guess=problem.init_guess
    )
    return problem, spec, result


# --- Environment and specification -------------------------------------------


def test_regions_validate_their_role_bounds_and_names():
    with pytest.raises(ValueError, match="role must be one of"):
        RectangleRegion("r", "lava", 0, 1, 0, 1)
    with pytest.raises(ValueError, match="min < max"):
        RectangleRegion("r", "goal", 1, 1, 0, 1)
    environment = Environment()
    environment.add_region(RectangleRegion("g", "goal", 0, 1, 0, 1))
    with pytest.raises(ValueError, match="already exists"):
        environment.add_region(RectangleRegion("g", "goal", 2, 3, 2, 3))
    with pytest.raises(ValueError, match="no region named"):
        environment.region("nowhere")


def test_builder_reads_every_role_and_visit_group():
    environment = build_reach_avoid_environment(CONFIG)
    roles = {name: r.role for name, r in environment.regions.items()}
    assert roles == {
        "workspace": "workspace",
        "wall": "obstacle",
        "a": "goal",
        "b": "goal",
        "t1": "target",
        "t2": "target",
    }
    assert environment.visits == [(2, ["t1", "t2"])]
    with pytest.raises(ValueError, match="workspace and goals"):
        build_reach_avoid_environment({"workspace": CONFIG["workspace"]})


def test_specification_is_built_from_the_roles():
    environment = build_reach_avoid_environment(CONFIG)
    region = environment.region
    H = 12
    safe = And(inside(region("workspace")), outside(region("wall")))
    stays = Or(
        Always(inside(region("t1")), [0, 2]),
        Always(inside(region("t2")), [0, 2]),
    )
    expected = reduce(
        And,
        [
            Always(safe, [1, H]),
            Eventually(Or(inside(region("a")), inside(region("b"))), [1, H]),
            Eventually(stays, [1, H - 2]),
        ],
    )
    steps = torch.linspace(0, 1, H + 1).unsqueeze(-1)
    mean = torch.tensor([1.5, 1.5]) + steps * torch.tensor([6.0, 7.0])
    beliefs = create_gaussian_belief_trajectory(
        mean, torch.full_like(mean, 0.05)
    )
    spec = environment.get_specification(H)
    torch.testing.assert_close(
        spec.probability_interval(beliefs),
        expected.probability_interval(beliefs),
    )
    torch.testing.assert_close(
        spec.smooth_lower(beliefs, 50.0), expected.smooth_lower(beliefs, 50.0)
    )
    with pytest.raises(ValueError, match="horizon must be a positive"):
        environment.get_specification(0)


# --- Examples ------------------------------------------------------------------


@pytest.mark.parametrize("name", EXAMPLES)
def test_every_route_warm_start_follows_its_route(name):
    problem = _problem(name)
    spec = problem.env.get_specification(problem.cfg["H"])
    routes = problem.cfg["routes"].values()
    for route, guess in zip(routes, problem.init_guesses):
        assert guess.abs().max() <= problem.dyn.u_max
        plan = problem.planner.evaluate_controls(
            problem.rollout, guess, spec=spec
        )
        end = plan.rollout.aux["mean_trace"][0, -1, :2]
        torch.testing.assert_close(
            end, torch.tensor(route[-1]), atol=0.3, rtol=0
        )


def test_short_solve_is_bounded_improving_and_replayable(short):
    problem, spec, result = short
    initial = problem.planner.evaluate_controls(
        problem.rollout, problem.init_guess, spec=spec
    )
    replay = problem.planner.evaluate_controls(
        problem.rollout, result.controls, spec=spec
    )
    assert result.hard_interval[0] >= initial.hard_interval[0]
    assert result.controls.abs().max() <= problem.dyn.u_max
    assert replay.hard_interval == pytest.approx(result.hard_interval)
    assert len(result.hard_lower_history) == 21


def test_figure_legend_is_short_and_ellipses_use_the_confidence(
    short, tmp_path
):
    problem, _, result = short
    fig, (axis, controls, scores) = plot_reach_avoid(
        result,
        problem.env,
        control_unit="m/s²",
        title="Either–Or",
        ellipse_confidence=0.9,
        alternatives=[result],
        save_path=str(tmp_path / "plan.png"),
    )
    assert (tmp_path / "plan.png").exists()
    assert (tmp_path / "plan.pdf").exists()
    labels = [t.get_text() for t in axis.get_legend().get_texts()]
    assert labels[:-1] == [
        "Obstacle",
        "Goal",
        "Target",
        "Other routes",
        "Plan",
        "90% ellipse",
    ]
    assert labels[-1].startswith("Min P(safe)")
    assert controls.get_ylabel() == "control [m/s²]"
    assert "Exact pdSTL" in scores.get_title()
    ellipse = next(p for p in axis.patches if p.get_label() == "90% ellipse")
    variance = torch.linalg.eigvalsh(
        result.rollout.aux["cov_trace"][0, 0, :2, :2]
    ).max()
    assert ellipse.width == pytest.approx(
        2 * (chi2.ppf(0.9, 2) * variance.item()) ** 0.5, rel=1e-5
    )
    plt.close(fig)


def test_live_view_and_animation(short):
    problem, spec, result = short
    fig, axes, observe, finish = create_reach_avoid_live_view(
        problem.env,
        lambda controls: problem.planner.evaluate_controls(
            problem.rollout, controls, spec=spec
        ),
        dt=problem.cfg["dt"],
        title="Either–Or",
        max_iters=problem.planner.cfg["max_iters"],
        alpha=problem.planner.cfg["alpha"],
    )
    observe(
        0,
        IterationRecord(
            result.controls,
            result.final_loss,
            result.smooth_lower,
            result.hard_interval,
            result.smoothing_beta,
        ),
    )
    assert len(axes[0].lines[0].get_xdata()) == problem.cfg["H"] + 1
    finish(result)
    plt.close(fig)
    fig, axis, movie = animate_reach_avoid(
        result, problem.env, dt=problem.cfg["dt"], title="Either–Or"
    )
    movie._draw_next_frame(1, blit=False)
    assert len(axis.lines[1].get_xdata()) == 2
    movie._draw_was_started = True
    plt.close(fig)


def _certified_and_clear(name, result):
    problem = _problem(name)
    points = result.rollout.aux["mean_trace"][0]
    assert result.threshold_met
    for obstacle in problem.env.by_role("obstacle"):
        assert not _inside(points, obstacle).any(), obstacle.name
    return problem, points


@pytest.mark.slow
def test_narrow_passage_certifies_a_route_clear_of_the_narrow_corridor():
    result = run_reach_avoid(
        "configs/scenarios/narrow_passage.yaml", save=False
    )
    problem, points = _certified_and_clear("narrow_passage", result)
    corridor = RectangleRegion("corridor", "obstacle", 5.5, 8.0, 3.5, 3.8)
    assert not _inside(points, corridor).any()
    goals = problem.env.by_role("goal")
    assert any(_inside(points, goal).any() for goal in goals)


@pytest.mark.slow
def test_either_or_dwells_in_one_target_then_reaches_the_goal():
    result = run_reach_avoid("configs/scenarios/either_or.yaml", save=False)
    problem, points = _certified_and_clear("either_or", result)

    def longest_stay(region):
        best = run = 0
        for is_inside in _inside(points, region).tolist():
            run = run + 1 if is_inside else 0
            best = max(best, run)
        return best

    # always[0, 5] spans six samples.
    assert max(map(longest_stay, problem.env.by_role("target"))) >= 6
    assert _inside(points, problem.env.region("goal")).any()
