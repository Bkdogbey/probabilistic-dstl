"""Canonical gate-corridor reach-avoid, end to end:

    YAML -> SingleIntegrator, Environment, b_0
    -> gaussian_rollout: u -> beliefs -> AxisInterval probabilities
    -> pdSTL (Frechet Boolean, min/max temporal) -> smooth lower -> Adam
    -> exact hard evaluation

Everything geometric is configuration-driven; these tests assert that, not coordinates.
"""

import matplotlib

matplotlib.use("Agg")
import ast
import copy
from pathlib import Path

import pytest
import torch

import planning.runners as runners
from pdstl.operators import Always, And, Eventually
from pdstl.predicates import InsideRectangle, OutsideRectangle
from planning.environment import DeadlineExpired, Environment
from planning.planner import OptimizationRecord, PlanResult
from planning.runners import (
    build_initial_belief,
    build_planner,
    build_scenario,
    initialize_toward_goal,
)
from utils import load_config

CONFIG = "configs/scenarios/reach_avoid.yaml"
ROOT = Path(__file__).resolve().parents[1]
FAST = {"max_iterations": 40, "convergence_patience": 15}


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def problem(config):
    environment = build_scenario(config, "cpu")
    planner = build_planner(config, environment)
    mean0, covariance0 = build_initial_belief(config["initial_belief"], "cpu")
    return config, environment, planner, mean0, covariance0


def _solve(config, *, overrides=None, optimizer=None):
    """Solve the scenario with a small iteration budget, optionally after editing the config."""
    cfg = copy.deepcopy(config)
    cfg["optimizer"] = {**cfg["optimizer"], **FAST, **(optimizer or {})}
    if overrides:
        cfg["environment"] = {**cfg["environment"], **overrides}
    environment = build_scenario(cfg, "cpu")
    planner = build_planner(cfg, environment)
    mean0, covariance0 = build_initial_belief(cfg["initial_belief"], "cpu")
    controls = initialize_toward_goal(
        mean0=mean0, goal=environment.goal, dynamics=planner.dynamics, horizon=cfg["horizon"]
    )
    return environment, planner, planner.solve(mean0, covariance0, initial_controls=controls)


# --- Environment: configuration drives the geometry ------------------------------


def test_specification_is_always_safe_and_eventually_goal(problem):
    _, environment, _, _, _ = problem
    spec = environment.specification(12)

    assert isinstance(spec, And)
    safe, reach = spec.subformula1, spec.subformula2
    assert isinstance(safe, Always) and safe.interval == [1, 12]
    assert isinstance(reach, Eventually) and reach.interval == [1, 12]
    assert isinstance(reach.subformula, InsideRectangle)

    # Always covers every obstacle-outside event conjoined with the workspace.
    leaves, stack = [], [safe.subformula]
    while stack:
        node = stack.pop()
        if isinstance(node, And):
            stack.extend([node.subformula1, node.subformula2])
        else:
            leaves.append(node)
    assert sum(isinstance(leaf, OutsideRectangle) for leaf in leaves) == 2
    assert sum(isinstance(leaf, InsideRectangle) for leaf in leaves) == 1  # the workspace


@pytest.mark.parametrize("count", [0, 1, 3, 5])
def test_any_number_of_obstacles_is_accepted(config, count):
    blocks = [
        {"name": f"block_{i}", "x_range": [float(i), i + 0.5], "y_range": [0.0, 1.0]}
        for i in range(count)
    ]
    environment = Environment.from_config({**config["environment"], "obstacles": blocks})
    assert len(environment.obstacles) == count
    assert len(environment.predicates()["obstacles"]) == count
    environment.specification(10)  # still a well-formed formula


def test_moving_an_obstacle_in_yaml_moves_its_predicate(config):
    moved = [{"name": "block", "x_range": [1.25, 2.75], "y_range": [3.5, 4.5]}]
    environment = Environment.from_config({**config["environment"], "obstacles": moved})
    (event,) = environment.predicates()["obstacles"]
    assert event.x_range == (1.25, 2.75)
    assert event.y_range == (3.5, 4.5)
    assert "block" in event.name


def test_no_scenario_coordinates_are_embedded_in_python():
    """Gate-corridor numbers must appear only in YAML."""
    scenario_numbers = {8.5, 9.5, 4.0, 6.0, 10.0}
    for path in ("src/planning/environment.py", "src/planning/planner.py",
                 "src/planning/runners.py"):
        tree = ast.parse((ROOT / path).read_text())
        literals = {
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, float)
        }
        assert not (literals & scenario_numbers), f"{path} hard-codes scenario geometry"


# --- Planner ---------------------------------------------------------------------


def test_solve_returns_the_minimal_plan_result(config):
    _, _, result = _solve(config)
    assert isinstance(result, PlanResult)
    assert set(result.__dataclass_fields__) == {
        "controls", "rollout", "hard_interval", "history", "best_iteration"
    }
    assert all(isinstance(record, OptimizationRecord) for record in result.history)
    assert result.hard_interval.shape == (2,)


def test_returned_controls_respect_the_configured_limit(config):
    _, planner, result = _solve(config)
    assert result.controls.shape == (config["horizon"], 2)
    assert float(result.controls.abs().max()) <= planner.dynamics.u_max + 1e-6


def test_optimization_improves_the_exact_hard_lower_score(config):
    _, _, result = _solve(config)
    first, best = result.history[0].hard_lower, max(r.hard_lower for r in result.history)
    assert best > first, f"hard lower did not improve: {first} -> {best}"
    assert 0.0 <= result.hard_lower <= result.hard_upper <= 1.0


def test_replaying_the_returned_controls_reproduces_the_hard_interval(config):
    environment, planner, result = _solve(config)
    mean0, covariance0 = build_initial_belief(config["initial_belief"], "cpu")
    from models.rollouts import gaussian_rollout

    replay = gaussian_rollout(planner.dynamics, mean0, covariance0)(
        planner.control_parameters(result.controls)
    )
    spec = environment.specification(config["horizon"])
    with torch.no_grad():
        interval = spec(replay.belief_trajectory, scale=-1)[0, 0]
    torch.testing.assert_close(interval, result.hard_interval, atol=1e-5, rtol=0)


def test_hard_evaluation_never_supplies_gradients(config):
    """The returned interval is a detached diagnostic, not part of the objective graph."""
    _, _, result = _solve(config)
    assert not result.hard_interval.requires_grad
    assert not result.controls.requires_grad


def test_mean_path_crosses_the_gate_and_enters_neither_block(config):
    environment, _, result = _solve(config, optimizer={"max_iterations": 250})
    path = result.rollout.aux["mean_trace"][0]

    for obstacle in environment.obstacles:
        (x_lo, x_hi), (y_lo, y_hi) = obstacle["x"], obstacle["y"]
        inside = (
            (path[:, 0] >= x_lo) & (path[:, 0] <= x_hi)
            & (path[:, 1] >= y_lo) & (path[:, 1] <= y_hi)
        )
        assert not bool(inside.any()), f"mean path entered {obstacle['name']}"

    # and it went through the gap rather than around the wall
    crossing = path[(path[:, 0] >= 4.0) & (path[:, 0] <= 6.0)]
    assert len(crossing) > 0, "the path never crossed the barrier"
    assert bool(((crossing[:, 1] > 4.0) & (crossing[:, 1] < 6.0)).all())


def test_final_belief_reaches_the_goal_with_a_meaningful_lower_score(config):
    environment, _, result = _solve(config, optimizer={"max_iterations": 250})
    final_mean = result.rollout.aux["mean_trace"][0, -1]
    (x_lo, x_hi), (y_lo, y_hi) = environment.goal["x"], environment.goal["y"]
    assert x_lo <= float(final_mean[0]) <= x_hi
    assert y_lo <= float(final_mean[1]) <= y_hi
    assert result.hard_lower > 0.2


def test_planner_needs_no_repulsion_or_goal_distance_shaping(config):
    """The canonical loss is pdSTL plus control regularisation; shaping stays off."""
    cfg = copy.deepcopy(config)
    assert cfg["optimizer"]["loss"]["terminal_goal_weight"] == 0.0
    _, planner, result = _solve(cfg)
    assert planner.config.terminal_goal_weight == 0.0
    assert max(r.hard_lower for r in result.history) > result.history[0].hard_lower


def test_changing_goal_and_obstacles_in_yaml_needs_no_python_change(config):
    """Same code, different YAML: a wider gate must score at least as well."""
    wide = [
        {"name": "lower", "x_range": [4.0, 6.0], "y_range": [0.0, 3.0]},
        {"name": "upper", "x_range": [4.0, 6.0], "y_range": [7.0, 10.0]},
    ]
    environment, _, result = _solve(config, overrides={"obstacles": wide})
    assert [o["name"] for o in environment.obstacles] == ["lower", "upper"]
    assert 0.0 <= result.hard_lower <= 1.0


# --- Temporal obligations --------------------------------------------------------


def test_goal_window_is_relative_without_a_configured_deadline(problem):
    _, environment, _, _, _ = problem
    assert environment.goal_deadline is None
    assert environment.goal_window(40, step=0) == [1, 40]
    assert environment.goal_window(40, step=9) == [1, 40], "no deadline means no countdown"


def test_configured_deadline_counts_down_and_expires():
    environment = Environment.from_config({
        "goal_deadline": 12,
        "bounds": {"x_range": [0.0, 1.0], "y_range": [0.0, 1.0]},
        "goal": {"name": "g", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]},
    })
    assert environment.goal_window(20, step=0) == [1, 12]
    assert environment.goal_window(20, step=5) == [0, 7]
    assert environment.goal_window(20, step=12) == [0, 0]
    with pytest.raises(DeadlineExpired):
        environment.goal_window(20, step=13)


# --- Runner ----------------------------------------------------------------------


def test_runner_returns_the_plan_result_unchanged(config, monkeypatch, tmp_path):
    captured = {}
    cfg = copy.deepcopy(config)
    cfg["optimizer"] = {**cfg["optimizer"], **FAST}
    monkeypatch.setattr(runners, "load_config", lambda path: cfg)

    from planning.planner import Planner

    original = Planner.solve

    def spy(self, *args, **kwargs):
        captured["result"] = original(self, *args, **kwargs)
        return captured["result"]

    monkeypatch.setattr(Planner, "solve", spy)
    returned = runners.run_reach_avoid(show=False, save=False)

    assert returned is captured["result"], "the runner must not rebuild the result"
    assert isinstance(returned, PlanResult)


def test_runner_plotting_and_saving_can_both_be_disabled(config, monkeypatch):
    cfg = copy.deepcopy(config)
    cfg["optimizer"] = {**cfg["optimizer"], **FAST}
    monkeypatch.setattr(runners, "load_config", lambda path: cfg)

    def forbidden(*args, **kwargs):
        raise AssertionError("visualization ran with show=False, save=False")

    import visualization.planning as planning_viz

    monkeypatch.setattr(planning_viz, "visualize_plan", forbidden)
    runners.run_reach_avoid(show=False, save=False)


def test_default_scenario_is_deterministic_under_a_fixed_seed(config):
    torch.manual_seed(0)
    _, _, first = _solve(config)
    torch.manual_seed(0)
    _, _, second = _solve(config)
    torch.testing.assert_close(first.controls, second.controls)
    torch.testing.assert_close(first.hard_interval, second.hard_interval)


def test_initialization_points_from_the_start_toward_the_goal(problem):
    config, environment, planner, mean0, _ = problem
    controls = initialize_toward_goal(
        mean0=mean0, goal=environment.goal, dynamics=planner.dynamics,
        horizon=config["horizon"],
    )
    assert controls.shape == (config["horizon"], 2)
    assert float(controls.abs().max()) <= planner.dynamics.u_max + 1e-6
    centre = torch.tensor([
        sum(environment.goal["x"]) / 2.0, sum(environment.goal["y"]) / 2.0
    ])
    direction = (centre - mean0) / torch.linalg.norm(centre - mean0)
    step = controls[0] / torch.linalg.norm(controls[0])
    torch.testing.assert_close(step, direction, atol=1e-5, rtol=0)
