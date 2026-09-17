"""The one receding-horizon controller, shared by reach-avoid and lane merge.

Replaces the three ad-hoc MPC loops that preceded it (`Planner._run_mpc`,
`Planner.run_receding_horizon`, `examples.run_mpc_check`). The properties those checked --
absolute deadlines that do not restart, warm starts that shift, bounds that hold in every
window -- are asserted here against the generic implementation.
"""

import copy

import matplotlib

matplotlib.use("Agg")
import pytest
import torch

from planning.controllers import (
    CONTROLLER_ALIASES,
    RecedingHorizonController,
    RecedingHorizonResult,
    normalize_controller_type,
    shift_controls,
)
from planning.environment import DeadlineExpired, Environment
from planning.planner import PlanResult
from planning.runners import (
    build_initial_belief,
    build_planner,
    build_scenario,
    initialize_toward_goal,
)
from utils import load_config

CONFIG = "configs/scenarios/reach_avoid_receding.yaml"
FAST = {"max_iterations": 15}


def _run(*, max_steps=4, apply_steps=1, warm_start=True, config_overrides=None):
    config = copy.deepcopy(load_config(CONFIG))
    config["optimizer"] = {**config["optimizer"], **FAST}
    config.update(config_overrides or {})
    environment = build_scenario(config, "cpu")
    planner = build_planner(config, environment)
    mean0, covariance0 = build_initial_belief(config["initial_belief"], "cpu")
    controller = RecedingHorizonController(
        planner=planner, dynamics=planner.dynamics, scenario=environment,
        config={"warm_start": warm_start},
    )
    initial = initialize_toward_goal(
        mean0=mean0, goal=environment.goal, dynamics=planner.dynamics,
        horizon=config["horizon"],
    )
    torch.manual_seed(0)
    result = controller.run(
        mean0, (mean0, covariance0), max_steps=max_steps, apply_steps=apply_steps,
        initial_controls=initial,
    )
    return config, environment, planner, result


# --- Controller selection ---------------------------------------------------------


def test_mpc_is_an_alias_for_receding_horizon():
    assert CONTROLLER_ALIASES["mpc"] == "receding_horizon"
    assert normalize_controller_type("mpc") == "receding_horizon"
    assert normalize_controller_type("receding_horizon") == "receding_horizon"
    assert normalize_controller_type("single_shot") == "single_shot"


# --- Warm starting ----------------------------------------------------------------


@pytest.mark.parametrize("applied", [1, 2, 3])
def test_shift_controls_drops_applied_steps_and_holds_the_last(applied):
    controls = torch.arange(12.0).reshape(6, 2)
    shifted = shift_controls(controls, applied)

    assert shifted.shape == controls.shape
    torch.testing.assert_close(shifted[: 6 - applied], controls[applied:])
    for row in shifted[6 - applied:]:
        torch.testing.assert_close(row, controls[-1])


def test_warm_start_can_be_disabled():
    _, _, _, warm = _run(warm_start=True)
    _, _, _, cold = _run(warm_start=False)
    assert len(warm.plans) == len(cold.plans)
    # Same seed, same geometry: only the initial guess differs, so the plans must differ.
    assert not torch.allclose(warm.plans[-1].controls, cold.plans[-1].controls)


# --- Every window is the same planner ----------------------------------------------


def test_every_window_goes_through_planner_solve():
    _, _, planner, result = _run(max_steps=3)
    assert all(isinstance(plan, PlanResult) for plan in result.plans)
    assert len(result.plans) == 3


def test_control_bounds_hold_in_every_window():
    _, _, planner, result = _run(max_steps=3)
    limit = planner.dynamics.u_max + 1e-6
    for index, plan in enumerate(result.plans):
        assert float(plan.controls.abs().max()) <= limit, f"window {index} exceeded u_max"
    assert float(result.applied_controls.abs().max()) <= limit


@pytest.mark.parametrize("apply_steps", [1, 2])
def test_apply_steps_controls_how_much_of_each_plan_is_executed(apply_steps):
    _, _, _, result = _run(max_steps=3, apply_steps=apply_steps)
    assert result.applied_controls.shape[0] == 3 * apply_steps
    assert result.states.shape[0] == 3 * apply_steps + 1


# --- Result shape -------------------------------------------------------------------


def test_result_is_minimal_and_holds_no_duplicated_per_plan_data():
    _, _, _, result = _run(max_steps=3)
    assert isinstance(result, RecedingHorizonResult)
    assert set(result.__dataclass_fields__) == {
        "states", "beliefs", "applied_controls", "plans", "stopped_reason"
    }
    # Per-window scores are derived from plans[k], never stored alongside them.
    assert result.hard_lowers == [plan.hard_lower for plan in result.plans]


def test_states_and_beliefs_line_up():
    _, _, _, result = _run(max_steps=3)
    assert len(result.beliefs) == result.states.shape[0]
    for state, (mean, covariance) in zip(result.states, result.beliefs):
        torch.testing.assert_close(state, mean)
        assert covariance.shape == (state.shape[0], state.shape[0])


def test_covariance_grows_while_no_estimator_is_present():
    """Documents the preserved assumption: the mean is observed, the covariance is not."""
    _, _, _, result = _run(max_steps=3)
    traces = [float(torch.diagonal(covariance).sum()) for _, covariance in result.beliefs]
    assert traces == sorted(traces), "covariance must grow without a measurement update"


# --- Absolute temporal obligations ---------------------------------------------------


def test_the_goal_deadline_counts_down_and_is_never_restarted():
    config, environment, _, _ = _run(max_steps=3)
    horizon = config["horizon"]
    uppers = [environment.goal_window(horizon, step)[1] for step in range(6)]

    assert uppers == sorted(uppers, reverse=True), "deadline restarted during replanning"
    assert uppers[0] > uppers[-1]
    assert all(upper >= 0 for upper in uppers)


def test_execution_stops_once_the_deadline_expires():
    environment = Environment.from_config({
        "goal_deadline": 2,
        "bounds": {"x_range": [0.0, 10.0], "y_range": [0.0, 10.0]},
        "goal": {"name": "g", "x_range": [8.0, 9.0], "y_range": [8.0, 9.0]},
    })
    config = copy.deepcopy(load_config(CONFIG))
    config["optimizer"] = {**config["optimizer"], **FAST}
    planner = build_planner(config, environment)
    mean0, covariance0 = build_initial_belief(config["initial_belief"], "cpu")

    result = RecedingHorizonController(
        planner=planner, dynamics=planner.dynamics, scenario=environment, config={},
    ).run(mean0, (mean0, covariance0), max_steps=8, apply_steps=1)

    assert result.stopped_reason == "deadline_expired"
    assert len(result.plans) == 3, "windows planned at steps 0, 1 and 2 only"
    with pytest.raises(DeadlineExpired):
        environment.goal_window(config["horizon"], step=3)


def test_safety_stays_an_invariant_over_the_local_window():
    """Always is window-relative; only the Eventually deadline counts down."""
    config, environment, _, _ = _run(max_steps=2)
    horizon = config["horizon"]
    for step in (0, 3, 7):
        spec = environment.specification(horizon, context={"step": step})
        assert spec.subformula1.interval == [1, horizon]


# --- No covariance steering or feedback gains ------------------------------------------


def test_only_feedforward_controls_are_optimized():
    _, _, planner, result = _run(max_steps=2)
    plan = result.plans[0]
    assert plan.controls.shape == (planner.horizon, planner.control_dim)
    for attribute in ("K", "gains", "feedback_gain", "policy"):
        assert not hasattr(plan, attribute), f"PlanResult gained {attribute}"
        assert not hasattr(planner, attribute), f"Planner gained {attribute}"
