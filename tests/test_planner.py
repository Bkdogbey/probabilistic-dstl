"""The planner as an optimizer over an arbitrary belief rollout.

`Planner` knows dynamics, a horizon and an objective. It builds no beliefs of its own, holds
no scenario geometry, and has exactly one planning operation. These tests pin that boundary
and the loop's contract: the smooth lower score carries every gradient, the exact hard
interval carries none, and the returned plan is the checkpoint the hard score selected.
"""

import ast
import logging
from pathlib import Path

import pytest
import torch

from models.dynamics import SingleIntegrator
from models.rollouts import BeliefRollout, gaussian_rollout
from pdstl.base import create_probability_belief_trajectory
from pdstl.operators import Eventually, GreaterThan, Predicate
from planning.environment import Environment
from planning.planner import OptimizationRecord, PlanConfig, Planner, PlanResult

ROOT = Path(__file__).resolve().parents[1]


def _planner(state_dim=2, horizon=5, **optimizer):
    dynamics = SingleIntegrator(dt=0.2, u_max=1.0, q_std=0.05, state_dim=state_dim)
    config = {"max_iterations": 8, "learning_rate": 0.1, **optimizer}
    return Planner(dynamics, None, horizon, config=config)


def _reach_spec(threshold=0.5, horizon=5):
    return Eventually(GreaterThan(threshold, dim=0), interval=[0, horizon])


# --- Architectural boundary --------------------------------------------------------


def test_planner_source_builds_no_concrete_beliefs():
    """The planner must not name a belief implementation; rollouts supply them."""
    source = (ROOT / "src/planning/planner.py").read_text()
    for forbidden in ("GaussianBelief", "probability_bounds", "InsideRectangle", "normal_cdf"):
        assert forbidden not in source, f"planner.py references {forbidden}"


def test_planner_holds_no_scenario_geometry_or_execution_loop():
    source = (ROOT / "src/planning/planner.py").read_text()
    for forbidden in (
        "lane_change", "lane_merge", "moving_obstacle", "circle",
        "run_receding_horizon", "_run_mpc", "savefig", "plt.",
    ):
        assert forbidden not in source, f"planner.py still contains {forbidden}"


def test_pdstl_imports_no_model_implementation():
    for module in ("src/pdstl/operators.py", "src/pdstl/base.py", "src/pdstl/predicates.py"):
        tree = ast.parse((ROOT / module).read_text())
        imported = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        assert not any((name or "").startswith(("models", "planning")) for name in imported), module


def test_solve_is_the_only_public_planning_operation():
    public = {name for name in vars(Planner) if not name.startswith("_")}
    assert "solve" in public
    assert not {"optimize_window", "run_receding_horizon", "evaluate_controls"} & public


# --- Rollout independence -----------------------------------------------------------


def test_planner_optimises_a_belief_rollout_it_knows_nothing_about():
    """A rollout of precomputed probability bounds, with no dynamics model behind it."""
    planner = _planner(horizon=4)
    event = Predicate(name="p")

    def rollout(parameters):
        base = torch.sigmoid(parameters[:, 0]).unsqueeze(-1)
        bounds = torch.cat([base, base], dim=-1)          # [T, 2], exact
        return BeliefRollout(create_probability_belief_trajectory(event, bounds))

    specification = Eventually(event, interval=[0, 3])
    result = planner._optimize_controls(rollout, specification, None)
    _, history, _ = result
    assert history[-1].hard_lower >= history[0].hard_lower


def test_belief_only_rollout_keeps_gradients_to_controls():
    """No nominal_trace, no aux: the pdSTL term alone must reach the controls."""
    planner = _planner(horizon=4)
    event = Predicate(name="p")
    parameters = torch.zeros(4, 2, requires_grad=True)

    base = torch.sigmoid(parameters[:, 0]).unsqueeze(-1)
    trajectory = create_probability_belief_trajectory(event, torch.cat([base, base], dim=-1))
    rollout = BeliefRollout(trajectory)

    smooth_lower = Eventually(event, interval=[0, 3])(trajectory, scale=5.0)[0, 0, 0]
    loss = planner.objective(
        smooth_lower=smooth_lower, controls=planner.dynamics.bound_control(parameters),
        rollout=rollout,
    )
    loss.backward()
    assert parameters.grad is not None and parameters.grad.abs().sum() > 0


def test_objective_without_a_nominal_trace_still_works():
    """Terminal-goal shaping is optional and silently absent when there is no trace."""
    planner = _planner()
    planner.environment = Environment()
    planner.environment.set_goal([0.0, 1.0], [0.0, 1.0])
    planner.config.terminal_goal_weight = 1.0

    controls = torch.zeros(5, 2, requires_grad=True)
    loss = planner.objective(
        smooth_lower=torch.tensor(0.5, requires_grad=True),
        controls=controls,
        rollout=BeliefRollout(belief_trajectory=None),
    )
    assert torch.isfinite(loss)


# --- Objective ----------------------------------------------------------------------


def test_objective_is_pdstl_plus_control_regularisation_only():
    """No repulsion, no visit regions: the canonical loss has four terms and J_g defaults off."""
    source = (ROOT / "src/planning/planner.py").read_text()
    for removed in ("_obs_repulsion_loss", "_visit_loss", "w_visit", "w_obs"):
        assert removed not in source, f"planner.py still has {removed}"
    assert PlanConfig().terminal_goal_weight == 0.0
    assert PlanConfig().pdstl_weight == 1.0


def test_control_effort_and_smoothness_match_their_definitions():
    planner = _planner()
    controls = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    assert float(planner._control_effort(controls)) == pytest.approx(4.0)
    # |u_0|^2 + sum |u_k - u_{k-1}|^2 = 1 + |[0,1]|^2 + |[-1,0]|^2 = 1 + 1 + 1
    assert float(planner._control_smoothness(controls)) == pytest.approx(3.0)


def test_scenario_extra_loss_is_added_when_supplied():
    planner = _planner()
    trace = torch.zeros(1, 6, 2)
    base = planner.objective(
        smooth_lower=torch.tensor(0.5), controls=torch.zeros(5, 2),
        rollout=BeliefRollout(None, trace),
    )
    with_extra = planner.objective(
        smooth_lower=torch.tensor(0.5), controls=torch.zeros(5, 2),
        rollout=BeliefRollout(None, trace),
        extra_loss=lambda nominal, config: torch.tensor(2.5),
    )
    assert float(with_extra - base) == pytest.approx(2.5)


# --- The loop -----------------------------------------------------------------------


def test_smooth_lower_carries_gradients_and_hard_interval_is_detached():
    planner = _planner()
    result = planner.solve(
        torch.zeros(2), torch.eye(2) * 0.05, specification=_reach_spec()
    )
    assert not result.hard_interval.requires_grad
    assert not result.controls.requires_grad
    assert all(isinstance(record, OptimizationRecord) for record in result.history)


def test_beta_anneals_geometrically_and_is_none_when_smoothing_is_off():
    planner = _planner(max_iterations=11, smoothing={"beta_start": 2.0, "beta_end": 20.0})
    assert planner.beta_schedule(0) == pytest.approx(2.0)
    assert planner.beta_schedule(10) == pytest.approx(20.0)
    assert planner.beta_schedule(5) == pytest.approx(2.0 * (10.0 ** 0.5))

    off = _planner(smoothing={"enabled": False})
    assert off.beta_schedule(0) is None


def test_early_stop_uses_the_exact_hard_lower_score():
    """probability_target is compared against the hard score, never the surrogate."""
    planner = _planner(max_iterations=200, probability_target=0.0, convergence_patience=3)
    result = planner.solve(torch.zeros(2), torch.eye(2) * 0.05, specification=_reach_spec(-1e9))
    assert len(result.history) == 3, "a target already met must stop after `patience` iterations"
    assert all(record.hard_lower >= 0.0 for record in result.history)


def test_checkpoint_is_the_best_hard_lower_and_ties_go_to_the_later_iterate():
    planner = _planner(max_iterations=12)
    result = planner.solve(torch.zeros(2), torch.eye(2) * 0.05, specification=_reach_spec())

    best = max(record.hard_lower for record in result.history)
    assert result.history[result.best_iteration].hard_lower == pytest.approx(best)
    tied = [r.iteration for r in result.history if r.hard_lower == pytest.approx(best)]
    assert result.best_iteration == max(tied), "ties must resolve to the later iterate"


def test_replaying_the_returned_controls_reproduces_the_hard_interval():
    planner = _planner(max_iterations=12)
    specification = _reach_spec()
    result = planner.solve(torch.zeros(2), torch.eye(2) * 0.05, specification=specification)

    replay = gaussian_rollout(planner.dynamics, torch.zeros(2), torch.eye(2) * 0.05)(
        planner.control_parameters(result.controls)
    )
    with torch.no_grad():
        interval = specification(replay.belief_trajectory, scale=-1)[0, 0]
    torch.testing.assert_close(interval, result.hard_interval, atol=1e-5, rtol=0)


def test_returned_controls_respect_the_bound():
    planner = _planner(max_iterations=12)
    result = planner.solve(torch.zeros(2), torch.eye(2) * 0.05, specification=_reach_spec())
    assert float(result.controls.abs().max()) <= planner.dynamics.u_max + 1e-6


def test_planner_handles_one_dimensional_controls():
    planner = _planner(state_dim=1, horizon=3, max_iterations=6)
    result = planner.solve(
        torch.zeros(1), torch.eye(1) * 0.05,
        specification=Eventually(GreaterThan(0.2, dim=0), interval=[0, 3]),
    )
    assert result.controls.shape == (3, 1)


def test_initial_controls_default_to_zero_and_are_otherwise_honoured():
    planner = _planner(max_iterations=1, smoothing={"enabled": False})
    guess = torch.full((5, 2), 0.4)
    result = planner.solve(
        torch.zeros(2), torch.eye(2) * 0.05, initial_controls=guess,
        specification=_reach_spec(),
    )
    # One iteration, so the checkpoint is the starting point: the guess, round-tripped.
    torch.testing.assert_close(result.controls, guess, atol=1e-3, rtol=0)


def test_solve_does_not_mutate_the_planner_environment():
    environment = Environment()
    environment.set_goal([0.0, 1.0], [0.0, 1.0])
    environment.set_bounds([-5.0, 5.0], [-5.0, 5.0])
    planner = Planner(
        SingleIntegrator(dt=0.2, u_max=1.0, q_std=0.05), environment, 5,
        config={"max_iterations": 4},
    )
    planner.solve(torch.zeros(2), torch.eye(2) * 0.05, specification=_reach_spec())
    assert planner.environment is environment
    assert environment.goal["x"] == [0.0, 1.0]


def test_quiet_optimisation_emits_no_log_records(caplog):
    planner = _planner(max_iterations=4)
    with caplog.at_level(logging.INFO, logger="planning"):
        planner.solve(torch.zeros(2), torch.eye(2) * 0.05, specification=_reach_spec())
    assert caplog.records == []


# --- Result type ---------------------------------------------------------------------


def test_plan_result_is_minimal():
    planner = _planner(max_iterations=4)
    result = planner.solve(torch.zeros(2), torch.eye(2) * 0.05, specification=_reach_spec())

    assert isinstance(result, PlanResult)
    assert set(result.__dataclass_fields__) == {
        "controls", "rollout", "hard_interval", "history", "best_iteration"
    }
    # Traces live in the rollout; they are not copied onto the result.
    assert "mean_trace" in result.rollout.aux and not hasattr(result, "mean_trace")


def test_detach_diagnostics_preserves_beliefs_and_handles_optional_fields():
    planner = _planner(max_iterations=4)
    result = planner.solve(torch.zeros(2), torch.eye(2) * 0.05, specification=_reach_spec())
    for tensor in result.rollout.aux.values():
        assert not tensor.requires_grad
    assert result.rollout.belief_trajectory is not None
