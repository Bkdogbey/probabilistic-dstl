"""Altitude safety: Always_[1,H](Z >= 50) for a 1-D stochastic single integrator.

    u -> SingleIntegrator(state_dim=1) -> GaussianBelief -> GreaterThan -> Always -> Planner
"""

import ast
import inspect

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch
from PIL import Image

from planning.runners import (
    build_dynamics,
    build_initial_belief,
    load_scenario_config,
    run_altitude_safety,
)
from models.beliefs import GaussianBelief
from models.dynamics import SingleIntegrator
from models.rollouts import gaussian_rollout
from pdstl.operators import Always, GreaterThan
from planning.planner import Planner
from visualization.animation import animate_altitude_optimization
from visualization.planning import plot_altitude_safety

CONFIG = "configs/scenarios/altitude_safety.yaml"


@pytest.fixture(scope="module")
def problem():
    cfg, planner_cfg = load_scenario_config(CONFIG)
    dyn = build_dynamics(cfg, "cpu")
    rollout = gaussian_rollout(dyn, *build_initial_belief(cfg, "cpu"))
    spec = Always(GreaterThan(cfg["threshold"], dim=0), interval=[1, cfg["H"]])
    planner = Planner(dyn, None, cfg["H"], config=planner_cfg)
    return cfg, planner_cfg, dyn, rollout, spec, planner


@pytest.fixture(scope="module")
def result():
    return run_altitude_safety(show=False, save=False)


def test_scenario_is_a_one_dimensional_gaussian_belief_rollout(problem):
    cfg, _, dyn, rollout, _, _ = problem
    predicted = rollout(torch.zeros(cfg["H"], 1))

    assert isinstance(dyn, SingleIntegrator)
    assert dyn.A.shape == dyn.B.shape == dyn.Q.shape == (1, 1)
    assert len(predicted.belief_trajectory) == cfg["H"] + 1
    assert all(isinstance(b, GaussianBelief) for b in predicted.belief_trajectory)


def test_objective_has_no_shaping(problem):
    _, planner_cfg, *_ = problem
    assert planner_cfg["w_dist"] == planner_cfg["w_obs"] == planner_cfg["w_visit"] == 0
    assert planner_cfg["w_phi"] > 0 and planner_cfg["scale"] <= 0


def test_downward_initial_plan_scores_poorly(result):
    assert result["interval_initial"][0] < 0.5


def test_optimized_lower_score_improves_substantially(result):
    lower_initial, lower_final = result["interval_initial"][0], result["interval_final"][0]
    assert lower_final > lower_initial + 0.5
    assert lower_final > 0.99


def test_returned_controls_replay_to_the_stored_interval(result):
    assert result["interval_final"] == pytest.approx(result["stored_interval"], abs=1e-5)


def test_optimized_probability_stays_high_over_the_window(result):
    assert result["atomic_final"][1:, 0].min() > 0.99


def test_controls_are_bounded_one_dimensional(result, problem):
    cfg, _, dyn, *_ = problem
    assert result["controls"].shape == (cfg["H"], 1)
    assert torch.isfinite(result["controls"]).all()
    assert (result["controls"].abs() <= dyn.u_max).all()


def test_gradient_flows_from_controls_to_the_objective(problem):
    cfg, _, dyn, rollout, spec, planner = problem
    u_init = torch.tensor(cfg["init_control"]).repeat(cfg["H"], 1)
    v = planner._control_parameters(u_init).clone().requires_grad_(True)

    predicted = rollout(v)
    smooth, _ = planner._scores(spec, predicted.belief_trajectory)
    planner._objective(predicted.nominal_trace, dyn.bound_control(v), smooth).backward()

    assert torch.isfinite(v.grad).all()
    assert v.grad.abs().sum() > 0


def test_runner_has_no_optimisation_loop():
    tree = ast.parse(inspect.getsource(run_altitude_safety))
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))


def test_every_optimizer_iterate_is_recorded(result):
    frames = result["frames"]
    assert len(frames) == result["iterations"]
    assert [f["objective"] for f in frames] == result["history"]
    assert frames[0]["interval"] == pytest.approx(result["interval_initial"], abs=1e-6)
    returned = min(frames, key=lambda f: f["objective"])
    assert returned["interval"] == pytest.approx(result["stored_interval"], abs=1e-6)


def test_initial_controls_are_the_configured_descent(result, problem):
    cfg, *_ = problem
    expected = torch.full((cfg["H"], 1), cfg["init_control"][0])
    torch.testing.assert_close(result["controls_initial"], expected, atol=1e-5, rtol=0)


def test_plot_has_belief_probability_and_control_panels(result, problem):
    cfg, _, dyn, *_ = problem
    fig, (ax_state, ax_prob, ax_u) = plot_altitude_safety(
        result, dt=cfg["dt"], threshold=cfg["threshold"], u_max=dyn.u_max, show=False
    )
    state_labels = ax_state.get_legend_handles_labels()[1]
    assert "Optimized predicted mean" in state_labels
    assert f"{cfg['threshold']:g} m threshold" in state_labels
    assert len(ax_prob.lines) == 2
    bounds = sorted(line.get_ydata()[0] for line in ax_u.lines[:2])
    assert bounds == [-dyn.u_max, dyn.u_max]
    assert ax_u.get_legend_handles_labels()[1] == ["Initial controls", "Optimized controls"]
    fig.canvas.draw()
    plt.close(fig)


def test_animation_runs_from_the_initial_guess_to_the_returned_plan(result, problem, tmp_path):
    cfg, _, dyn, *_ = problem
    fps = 2
    returned = min(range(len(result["frames"])), key=lambda i: result["history"][i])
    path = tmp_path / "optimization.gif"

    animate_altitude_optimization(
        result, dt=cfg["dt"], threshold=cfg["threshold"], u_max=dyn.u_max,
        filename=str(path), fps=fps,
    )

    with Image.open(path) as gif:
        assert gif.n_frames == returned + 1  # identical hold frames merge into the last
        gif.seek(gif.n_frames - 1)
        assert gif.info["duration"] > 1000  # the returned plan is held for about a second
