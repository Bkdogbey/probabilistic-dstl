"""2-D reach-and-avoid planning through the cleaned pipeline:

    v -> u -> SingleIntegrator -> gaussian_rollout -> GaussianBelief_k
      -> probability_bounds(x/y atoms) -> Frechet And/Or -> Always/Eventually -> J

with rectangles built only from primitive atoms and every shaping heuristic off.
"""

import ast
import inspect
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pytest
import torch

import experiments.planning as experiments_planning
import planning.environment as environment
from experiments.planning import load_scenario_config, run_reach_avoid
from models.beliefs import GaussianBelief
from models.dynamics import SingleIntegrator
from pdstl.operators import (
    Always,
    And,
    Eventually,
    GreaterThan,
    LessThan,
    Maxish,
    Minish,
    Or,
)
from planning.environment import Environment
from planning.examples import controls_to_params
from planning.specifications import inside_rectangle, outside_rectangle, reach_avoid
from visualization.planning import plot_reach_avoid

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/scenarios/reach_avoid.yaml"
LEGACY_PREDICATES = (
    "RectangularGoalPredicate",
    "RectangularObstaclePredicate",
    "CircularObstaclePredicate",
    "MovingRectangularObstaclePredicate",
)


@pytest.fixture(scope="module")
def result():
    return run_reach_avoid(show=False, save=False)


@pytest.fixture(scope="module")
def cfg():
    return load_scenario_config(CONFIG)


def _atoms(formula):
    return [m for m in formula.modules() if isinstance(m, (GreaterThan, LessThan))]


def _in_rectangle(points, rect, strict):
    (x0, x1), (y0, y1) = rect["x_range"], rect["y_range"]
    x, y = points[:, 0], points[:, 1]
    if strict:
        return (x > x0) & (x < x1) & (y > y0) & (y < y1)
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


# --- Specification builders --------------------------------------------------


@pytest.mark.parametrize(
    "builder, expected",
    [
        (inside_rectangle, {(">=", 0, 4.0), ("<=", 0, 5.0), (">=", 1, -2.0), ("<=", 1, 2.0)}),
        (outside_rectangle, {("<=", 0, 4.0), (">=", 0, 5.0), ("<=", 1, -2.0), (">=", 1, 2.0)}),
    ],
)
def test_rectangle_builders_contain_only_primitive_atoms(builder, expected):
    formula = builder([4.0, 5.0], [-2.0, 2.0])
    connective = And if builder is inside_rectangle else Or

    for module in formula.modules():
        assert isinstance(module, (connective, GreaterThan, LessThan))
    atoms = _atoms(formula)
    assert len(atoms) == 4
    assert {(a.sense, a.dim, a.threshold) for a in atoms} == expected


def test_reach_avoid_is_always_safe_and_eventually_goal():
    goal = {"x_range": [4.0, 5.0], "y_range": [-2.0, 2.0]}
    obstacle = {"x_range": [1.5, 2.8], "y_range": [-0.25, 1.0]}
    spec = reach_avoid(goal, obstacle, 30)

    assert isinstance(spec, And)
    assert isinstance(spec.subformula1, Always)
    assert isinstance(spec.subformula2, Eventually)
    assert spec.subformula1.interval == [1, 30]
    assert spec.subformula2.interval == [1, 30]
    # Minish/Maxish are the temporal operators' own window reducers.
    allowed = (And, Or, Always, Eventually, Minish, Maxish, GreaterThan, LessThan)
    assert all(isinstance(m, allowed) for m in spec.modules())
    assert len(_atoms(spec)) == 8


def test_rectangle_builders_reject_unordered_ranges():
    with pytest.raises(ValueError):
        inside_rectangle([5.0, 4.0], [0.0, 1.0])
    with pytest.raises(ValueError):
        outside_rectangle([0.0, 1.0], [1.0, 1.0])


def test_specifications_module_only_builds_formulas():
    tree = ast.parse((ROOT / "src/planning/specifications.py").read_text(encoding="utf-8"))
    imports = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ] + [
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert imports == ["pdstl.operators"]

    forbidden = {"torch", "probability_bounds", "mean", "covariance", "cdf",
                 "normal_cdf", "value", "beliefs", "environment"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not (names | attrs) & forbidden


# --- Example wiring ----------------------------------------------------------


def test_example_uses_single_integrator_and_gaussian_rollout(monkeypatch):
    calls = []
    real = experiments_planning.gaussian_rollout

    def spy(dynamics, mean0, cov0):
        calls.append(dynamics)
        return real(dynamics, mean0, cov0)

    monkeypatch.setattr(experiments_planning, "gaussian_rollout", spy)
    out = run_reach_avoid(max_iterations=3, show=False, save=False)

    assert isinstance(out["dynamics"], SingleIntegrator)
    assert len(calls) == 1 and calls[0] is out["dynamics"]


def test_no_legacy_environment_predicate_is_used(monkeypatch, result):
    for module in result["spec"].modules():
        assert type(module).__name__ not in LEGACY_PREDICATES

    source = inspect.getsource(run_reach_avoid)
    for name in (*LEGACY_PREDICATES, "extract_trajectory_stats", "normal_cdf",
                 "get_specification"):
        assert name not in source

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy environment probability path was used")

    monkeypatch.setattr(Environment, "get_specification", forbidden)
    monkeypatch.setattr(environment, "extract_trajectory_stats", forbidden)
    monkeypatch.setattr(environment, "normal_cdf", forbidden)
    for name in LEGACY_PREDICATES:
        monkeypatch.setattr(getattr(environment, name), "robustness_trace", forbidden)
    run_reach_avoid(max_iterations=3, show=False, save=False)


def test_shaping_weights_are_zero(cfg, result):
    _, planner_cfg = cfg
    for source in (planner_cfg, result["planner_cfg"]):
        assert source["w_dist"] == source["w_obs"] == source["w_visit"] == 0
        assert source["w_phi"] > 0
        assert source["scale"] <= 0


def test_all_outputs_are_finite(result):
    for key in ("mean_trace", "cov_trace", "mean_initial", "cov_initial", "controls"):
        assert torch.isfinite(result[key]).all(), key
    for key in ("interval_initial", "interval_final", "stored_hard_interval",
                "history", "final_state", "goal_state"):
        assert torch.isfinite(torch.tensor(result[key])).all(), key
    assert torch.isfinite(torch.tensor(result["min_obstacle_clearance"]))


def test_controls_stay_within_u_max(result):
    assert (result["controls"].abs() <= result["u_max"]).all()


def test_returned_controls_replay_to_the_stored_hard_interval(result):
    """The replayed interval is recomputed from best.controls, independent of the
    candidate the planner stored."""
    assert result["interval_final"] == pytest.approx(result["stored_hard_interval"], abs=1e-4)

    cfg, _ = load_scenario_config(CONFIG)
    dyn = result["dynamics"]
    x0_mean = torch.tensor(cfg["x0_mean"])
    x0_cov = torch.eye(2) * cfg["x0_cov_scale"]
    replay = experiments_planning.gaussian_rollout(dyn, x0_mean, x0_cov)(
        controls_to_params(dyn, result["controls"])
    )
    assert all(isinstance(b, GaussianBelief) for b in replay.belief_trajectory)
    interval = result["spec"](replay.belief_trajectory, scale=-1)[0, 0]
    assert interval.tolist() == pytest.approx(result["stored_hard_interval"], abs=1e-4)


# --- Planning outcome --------------------------------------------------------


def test_initial_trajectory_has_poor_reach_avoid_score(result):
    assert result["interval_initial"][0] < 0.1
    # the initial guess reaches the goal but runs through the obstacle
    path = result["mean_initial"][0, 1:]
    assert _in_rectangle(path, result["goal"], strict=False).any()
    assert _in_rectangle(path, result["obstacle"], strict=True).any()


def test_optimized_hard_lower_score_improves_substantially(result):
    lower_initial, lower_final = result["interval_initial"][0], result["interval_final"][0]
    assert lower_final > 0.8
    assert lower_final > lower_initial + 0.5
    assert lower_final <= result["interval_final"][1] + 1e-6


def test_optimized_mean_reaches_the_goal(result):
    path = result["mean_trace"][0, 1:]
    assert _in_rectangle(path, result["goal"], strict=False).any()
    goal_state = torch.tensor([result["goal_state"]])
    assert _in_rectangle(goal_state, result["goal"], strict=False).all()


def test_optimized_mean_avoids_the_obstacle_interior(result):
    path = result["mean_trace"][0, 1:]
    assert not _in_rectangle(path, result["obstacle"], strict=True).any()
    assert result["min_obstacle_clearance"] > 0


def test_gradients_flow_from_pdstl_to_controls(cfg):
    scenario, _ = cfg
    dyn = SingleIntegrator(dt=scenario["dt"], u_max=scenario["u_max"], q_std=scenario["q_std"])
    x0_mean = torch.tensor(scenario["x0_mean"])
    x0_cov = torch.eye(2) * scenario["x0_cov_scale"]
    spec = reach_avoid(scenario["goal"], scenario["obstacle"], scenario["H"])

    u_init = torch.tensor(scenario["init_control"]).repeat(scenario["H"], 1)
    v = controls_to_params(dyn, u_init).clone().requires_grad_(True)
    predicted = experiments_planning.gaussian_rollout(dyn, x0_mean, x0_cov)(v)
    assert all(isinstance(b, GaussianBelief) for b in predicted.belief_trajectory)

    score = spec(predicted.belief_trajectory, scale=-1)[0, 0, 0]
    score.backward()

    assert torch.isfinite(score)
    assert v.grad is not None
    assert torch.isfinite(v.grad).all()
    assert v.grad.abs().sum() > 0


# --- Visualization -----------------------------------------------------------


def test_reach_avoid_plot_draws_regions_paths_and_ellipses(result, cfg):
    scenario, _ = cfg
    env = Environment()
    env.set_goal(**scenario["goal"])
    env.add_obstacle(**scenario["obstacle"])

    fig, ax = plot_reach_avoid(
        result["mean_initial"], result["cov_initial"],
        result["mean_trace"], result["cov_trace"],
        env, scenario["ellipse_steps"], show=False,
    )
    rectangles = [p for p in ax.patches if isinstance(p, patches.Rectangle)]
    ellipses = [p for p in ax.patches if isinstance(p, patches.Ellipse)]
    assert len(rectangles) >= 2
    assert len(ellipses) == 2 * len(scenario["ellipse_steps"])
    assert len(ax.lines) >= 3  # initial path, optimized path, start marker
    fig.canvas.draw()
    plt.close(fig)
