"""Single-shot reach-and-avoid through a double-slit barrier, environment-driven:

    scenario -> SingleIntegrator, Environment, b_0
    -> gaussian_rollout: u -> b_0:H(u) -> InsideRectangle / OutsideRectangle events
    -> GaussianBelief.probability_bounds -> pdSTL -> Planner.optimize_window

The planner optimises pdSTL robustness of the predicted belief trajectory with
every shaping weight at zero; the mean path is only checked post hoc.
"""

import ast
import inspect

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pytest
import torch

import planning.runners as runners
from planning.runners import (
    build_dynamics,
    build_environment,
    build_initial_belief,
    load_scenario_config,
    run_reach_avoid,
)
from models.beliefs import GaussianBelief
from models.dynamics import SingleIntegrator
from pdstl.predicates import InsideRectangle, OutsideRectangle
from planning.environment import Environment
from planning.planner import Planner
from visualization.planning import plot_event_probabilities, plot_reach_avoid, visualize_reach_avoid

CONFIG = "configs/scenarios/reach_avoid.yaml"


@pytest.fixture(scope="module")
def problem():
    """The same objects run_reach_avoid builds, for replay and gradient checks."""
    cfg, planner_cfg = load_scenario_config(CONFIG)
    dyn = build_dynamics(cfg, "cpu")
    env = build_environment(cfg, device="cpu")
    x0_mean, x0_cov = build_initial_belief(cfg, "cpu")
    rollout = runners.gaussian_rollout(dyn, x0_mean, x0_cov)
    spec = env.get_specification(cfg["H"])
    planner = Planner(dyn, env, cfg["H"], config=planner_cfg)
    return cfg, planner_cfg, dyn, env, rollout, spec, planner


@pytest.fixture(scope="module")
def result():
    return run_reach_avoid(show=False, save=False)


def _inside(points, x_range, y_range, strict):
    x, y = points[:, 0], points[:, 1]
    if strict:
        return (x > x_range[0]) & (x < x_range[1]) & (y > y_range[0]) & (y < y_range[1])
    return (x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (y <= y_range[1])


# --- Wiring ------------------------------------------------------------------


def test_runner_uses_the_scenario_model_environment_and_rollout(monkeypatch):
    calls = {"rollout": [], "spec": 0}
    real_rollout = runners.gaussian_rollout
    real_spec = Environment.get_specification

    def rollout_spy(dynamics, mean0, cov0):
        calls["rollout"].append(dynamics)
        return real_rollout(dynamics, mean0, cov0)

    def spec_spy(self, *args, **kwargs):
        calls["spec"] += 1
        return real_spec(self, *args, **kwargs)

    monkeypatch.setattr(runners, "gaussian_rollout", rollout_spy)
    monkeypatch.setattr(Environment, "get_specification", spec_spy)
    run_reach_avoid(show=False, save=False)

    assert len(calls["rollout"]) == 1
    assert isinstance(calls["rollout"][0], SingleIntegrator)
    assert calls["spec"] == 1


def test_spec_is_built_from_spatial_events(problem):
    cfg, *_, spec, _ = problem
    events = [m for m in spec.modules() if isinstance(m, (InsideRectangle, OutsideRectangle))]
    outside = [e for e in events if isinstance(e, OutsideRectangle)]
    inside = [e for e in events if isinstance(e, InsideRectangle)]

    assert len(outside) == len(cfg["obstacles"])   # one avoid event per barrier block
    assert len(inside) == 2                        # goal and workspace
    assert len(events) == len(outside) + len(inside)


def test_runner_contains_no_optimisation_or_probability_code():
    tree = ast.parse(inspect.getsource(run_reach_avoid))
    assert not any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(tree))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("backward", "optim", "Adam", "normal_cdf", "cdf", "erf", "step"):
        assert forbidden not in names

    module = ast.parse(inspect.getsource(runners))
    imported = {n.module for n in ast.walk(module) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(module) if isinstance(n, ast.Import) for a in n.names}
    assert not any("optim" in m for m in imported)


def test_planner_optimises_the_objective_the_scenario_configures(problem):
    """Whatever weights the YAML sets are the ones optimised -- none are assumed here.

    Works with the shaping heuristics on or off; it rebuilds the objective from the same
    configuration the planner read, rather than pinning a particular scenario's choices.
    """
    cfg, planner_cfg, dyn, _, rollout, spec, planner = problem
    assert planner_cfg["w_phi"] > 0, "the pdSTL term must carry some weight"

    v = planner._control_parameters(torch.full((cfg["H"], 2), 0.3))
    predicted = rollout(v)
    controls = dyn.bound_control(v)
    smooth_lower = spec(predicted.belief_trajectory, beta=planner._beta(0))[0, 0, 0]

    objective = planner._objective(predicted.nominal_trace, controls, smooth_lower)

    expected = -planner_cfg["w_phi"] * smooth_lower + planner._control_cost(controls)
    for key, term in (
        ("w_dist", planner._goal_dist_loss),
        ("w_obs", planner._obs_repulsion_loss),
        ("w_visit", planner._visit_loss),
    ):
        if planner_cfg[key]:
            expected = expected + planner_cfg[key] * term(predicted.nominal_trace)

    assert objective.item() == pytest.approx(expected.item(), rel=1e-6)


# --- Planning outcome ----------------------------------------------------------


def test_outputs_are_finite_and_controls_bounded(result, problem):
    _, _, dyn, *_ = problem
    for key in ("mean_trace", "cov_trace", "goal_trace", "safe_trace", "controls"):
        assert torch.isfinite(result[key]).all(), key
    assert torch.isfinite(torch.tensor(result["history"])).all()
    assert (result["controls"].abs() <= dyn.u_max).all()


def test_pdstl_lower_score_improves_from_a_poor_start(result, problem):
    """Improvement is measured against the scenario's own target, not a pinned number.

    The Frechet conjunction caps what any geometry can reach, so the bar is stated relative
    to the configured `alpha` and tightens automatically if the scenario changes.
    """
    _, planner_cfg, *_ = problem
    lower_initial, lower_final = result["interval_initial"][0], result["interval_final"][0]
    alpha = planner_cfg["alpha"]

    assert lower_initial < 0.1, "the initial guess should start far from satisfying"
    assert lower_final > lower_initial
    assert lower_final <= result["interval_final"][1] + 1e-6
    if result["iterations"] < planner_cfg["max_iters"]:
        # It stopped early, so it met the configured target.
        assert result["stored_interval"][0] >= alpha - 1e-3
    else:
        assert lower_final >= 0.5 * alpha, "used the whole budget without getting close"


def test_plan_has_useful_goal_and_safety_probabilities(result):
    """Each conjunct must dominate their Frechet conjunction -- a structural fact, not a number."""
    lower = result["interval_final"][0]
    assert result["goal_interval"][0] >= lower - 1e-6
    assert result["min_safe_interval"][0] >= lower - 1e-6
    assert result["safe_trace"][1:, 0].min() == pytest.approx(result["min_safe_interval"][0])
    assert result["goal_trace"][1:, 0].max() == pytest.approx(result["goal_interval"][0])


def test_safety_trace_is_the_conjunction_over_every_block(result, problem):
    cfg, *_ = problem
    per_obstacle = result["obstacle_traces"]
    assert len(per_obstacle) == len(cfg["obstacles"])

    # Frechet: the joint lower bound never exceeds any single block's, and the joint
    # upper bound never exceeds the tightest single upper bound.
    for trace in per_obstacle:
        assert (result["safe_trace"][:, 0] <= trace[:, 0] + 1e-6).all()
        assert (result["safe_trace"][:, 1] <= trace[:, 1] + 1e-6).all()


def test_returned_controls_replay_to_the_stored_pdstl_interval(result, problem):
    *_, rollout, spec, planner = problem
    replay = planner.evaluate_controls(rollout, result["controls"], spec=spec)

    assert all(isinstance(b, GaussianBelief) for b in replay.rollout.belief_trajectory)
    assert list(replay.hard_interval) == pytest.approx(result["interval_final"], abs=1e-4)
    assert result["interval_final"] == pytest.approx(result["stored_interval"], abs=1e-4)


def test_gradients_flow_from_controls_through_beliefs_and_events_to_the_objective(problem):
    cfg, _, dyn, _, rollout, spec, planner = problem
    u_init = torch.tensor(cfg["init_control"]).repeat(cfg["H"], 1)
    v = planner._control_parameters(u_init).clone().requires_grad_(True)

    predicted = rollout(v)
    smooth, _ = planner._scores(spec, predicted.belief_trajectory, planner._beta(0))
    objective = planner._objective(predicted.nominal_trace, dyn.bound_control(v), smooth)
    objective.backward()

    assert torch.isfinite(objective)
    assert torch.isfinite(v.grad).all()
    assert v.grad.abs().sum() > 0


def test_post_hoc_mean_avoids_every_block_and_reaches_the_goal(result, problem):
    cfg, *_ = problem
    path = result["mean_trace"][0, 1:]

    for obstacle in cfg["obstacles"]:
        assert not _inside(path, obstacle["x"], obstacle["y"], strict=True).any()
    assert result["min_mean_clearance"] > 0
    assert _inside(path, cfg["workspace"]["x"], cfg["workspace"]["y"], strict=False).all()
    assert _inside(path, cfg["goal"]["x"], cfg["goal"]["y"], strict=False).any()

    # and the initial guess did not: it runs straight into the middle block
    initial = result["mean_initial"][0, 1:]
    hits = [_inside(initial, o["x"], o["y"], strict=True).any() for o in cfg["obstacles"]]
    assert any(hits)


def test_optimized_mean_crosses_the_barrier_only_through_a_slit(result, problem):
    """The blocks tile the barrier's y-extent apart from the slits, so clearing every
    block inside the barrier x-span is exactly 'the plan went through a gap'."""
    cfg, *_ = problem
    path = result["mean_trace"][0, 1:]
    span = cfg["obstacles"][0]["x"]
    crossing = path[_inside(path, span, cfg["workspace"]["y"], strict=False)]

    assert len(crossing) > 0, "the plan never reaches the barrier"
    for obstacle in cfg["obstacles"]:
        blocked = (crossing[:, 1] > obstacle["y"][0]) & (crossing[:, 1] < obstacle["y"][1])
        assert not blocked.any()


# --- Visualization -------------------------------------------------------------


def test_plots_draw_the_belief_sequence_and_event_probabilities(result, problem):
    cfg, _, _, env, *_ = problem
    fig, ax = plot_reach_avoid(
        result["mean_initial"], result["cov_initial"],
        result["mean_trace"], result["cov_trace"],
        env, cfg["ellipse_every"], show=False,
    )
    ellipses = [p for p in ax.patches if isinstance(p, patches.Ellipse)]
    rectangles = [p for p in ax.patches if isinstance(p, patches.Rectangle)]
    assert len(rectangles) == len(cfg["obstacles"]) + 2   # blocks, goal, workspace
    assert len(ellipses) == len(range(0, cfg["H"] + 1, cfg["ellipse_every"]))
    labels = ax.get_legend_handles_labels()[1]
    assert "Predicted belief mean" in labels
    assert not any("Initial" in label for label in labels)
    fig.canvas.draw()
    plt.close(fig)

    traces = {
        "P(goal)": result["goal_trace"],
        "P(safe)": result["safe_trace"],
        "P(workspace)": result["bounds_trace"],
    }
    fig, ax = plot_event_probabilities(cfg["dt"], traces, show=False)
    assert len(ax.lines) == 2 * len(traces)   # a lower and an upper curve per event
    assert ax.get_legend_handles_labels()[1] == list(traces)
    assert len(ax.collections) == len(traces)
    plt.close(fig)


def test_optimization_trace_records_the_actual_semantics_and_checkpoint(result, problem):
    *_, rollout, spec, planner = problem
    records = result["optimization_trace"]
    assert [r["iteration"] for r in records] == list(range(result["iterations"]))
    assert [r["objective"] for r in records] == result["history"]
    # The record stores scalars, not the nominal trace, so the shaping penalties cannot be
    # recomputed from it. They are sums of squares, so they can only raise the objective.
    shaping_on = any(planner.cfg[key] for key in ("w_dist", "w_obs", "w_visit"))
    for record in records:
        assert record["beta"] == pytest.approx(planner._beta(record["iteration"]))
        pdstl_and_control = (
            -planner.cfg["w_phi"] * record["smooth_lower"] + record["control_cost"]
        )
        if shaping_on:
            assert record["objective"] >= pdstl_and_control - 1e-5
        else:
            assert record["objective"] == pytest.approx(pdstl_and_control, abs=1e-5)
    selected = records[result["returned_iteration"]]
    assert selected["hard_interval"] == pytest.approx(result["hard_interval"], abs=1e-4)
    predicted = rollout(planner._control_parameters(result["controls"]))
    smooth = spec(predicted.belief_trajectory, beta=selected["beta"])[0, 0, 0].item()
    assert selected["smooth_lower"] == pytest.approx(smooth, abs=1e-4)


def test_default_presentation_saves_three_views_and_debug_is_opt_in(result, problem, tmp_path):
    cfg, _, _, env, *_ = problem
    views = visualize_reach_avoid(result, env, dt=cfg["dt"], show=False,
                                  save_path=str(tmp_path / "reach.png"))
    assert {p.name for p in tmp_path.iterdir()} == {
        "reach.png", "reach_probabilities.png", "reach_optimization.png"
    }
    assert len(views[1][1].collections) == 2
    curve = views[2][1].lines[0]
    assert list(curve.get_ydata()) == [r["smooth_lower"] for r in result["optimization_trace"]]
    debug_views = visualize_reach_avoid(result, env, dt=cfg["dt"], show=False,
                                        show_initial=True, show_workspace=True)
    assert "Initial predicted belief mean" in debug_views[0][1].get_legend_handles_labels()[1]
    assert len(debug_views[1][1].collections) == 3
    for fig, _ in (*views, *debug_views):
        fig.canvas.draw()
        plt.close(fig)
