"""Behavioral contracts for the canonical optimizer and receding-horizon loop."""

import logging

import pytest
import torch

from models.dynamics import SingleIntegrator
from models.rollouts import (
    BeliefRollout,
    create_probability_belief_trajectory,
    gaussian_rollout,
)
from pdstl.operators import Always, Eventually, Predicate
from pdstl.predicates import GreaterThan, LessThan
from planning.planner import MPCResult, PlanResult, Planner


def problem(**config):
    dyn = SingleIntegrator(state_dim=1)
    planner = Planner(
        dyn,
        5,
        {
            "max_iters": 80,
            "w_u": 0.01,
            "w_du": 0.01,
            "smoothing": {"beta_start": 10.0, "beta_end": 10.0},
            **config,
        },
    )
    rollout = gaussian_rollout(dyn, torch.zeros(1), torch.eye(1) * 0.1)
    spec = Eventually(GreaterThan(0.4), interval=[0, 5])
    return planner, rollout, spec


def test_quiet_optimization(caplog):
    planner, rollout, spec = problem(max_iters=2)
    with caplog.at_level(logging.INFO):
        result = planner.optimize_window(rollout, spec=spec)
    assert caplog.records == []
    assert isinstance(result, PlanResult)


def test_neutral_initialization_and_bounded_controls():
    planner, rollout, spec = problem()
    assert torch.count_nonzero(planner._init_controls(None)) == 0
    initial = planner.evaluate_controls(rollout, torch.zeros(5, 1), spec=spec)
    result = planner.optimize_window(rollout, spec=spec)
    assert result.controls.shape == (5, 1)
    assert (result.controls.abs() <= planner.dyn.u_max).all()
    assert result.smooth_lower > initial.smooth_lower
    assert result.hard_interval[0] > initial.hard_interval[0] + 0.3


def test_loss_is_only_smooth_score_effort_and_smoothness():
    planner, _, _ = problem(w_phi=3.0, w_u=2.0, w_du=4.0)
    controls = torch.tensor([[0.1], [0.3], [0.2], [-0.2], [0.0]])
    expected = -3 * 0.7 + 2 * controls.square().sum()
    expected += 4 * (
        (controls[1:] - controls[:-1]).square().sum()
        + controls[0].square().sum()
    )
    torch.testing.assert_close(
        planner._objective(controls, torch.tensor(0.7)), expected
    )


def test_one_iteration_returns_the_updated_controls_and_replays():
    planner, rollout, spec = problem(max_iters=1)
    result = planner.optimize_window(rollout, spec=spec)
    assert result.controls.abs().sum() > 0
    assert len(result.loss_history) == 1
    replay = planner.evaluate_controls(rollout, result.controls, spec=spec)
    assert replay.hard_interval == pytest.approx(
        result.hard_interval, abs=1e-6
    )
    assert replay.smooth_lower == pytest.approx(result.smooth_lower, abs=1e-6)
    torch.testing.assert_close(
        replay.rollout.aux["mean_trace"], result.rollout.aux["mean_trace"]
    )
    assert not result.rollout.aux["mean_trace"].requires_grad
    assert not result.rollout.belief_trajectory[1].value().requires_grad


def test_returns_final_smooth_iterate_even_when_hard_score_worsens():
    planner, rollout, spec = problem(max_iters=3)
    observed = []
    original = spec.probability_interval
    # Reporting values cannot influence optimization when stopping is disabled.
    spec.probability_interval = lambda trajectory: -original(trajectory)
    result = planner.optimize_window(
        rollout, spec=spec, on_iteration=lambda k, p: observed.append(p)
    )
    torch.testing.assert_close(result.controls, observed[-1].controls)
    assert observed[-1].hard_interval[0] < observed[0].hard_interval[0]


def test_observer_is_optional_and_does_not_change_optimization():
    planner, rollout, spec = problem(max_iters=4)
    seen = []
    expected = planner.optimize_window(rollout, spec=spec)
    actual = planner.optimize_window(
        rollout, spec=spec, on_iteration=lambda k, p: seen.append((k, p))
    )
    assert [k for k, _ in seen] == list(range(4))
    torch.testing.assert_close(actual.controls, expected.controls)
    assert actual.loss_history == expected.loss_history


def test_optional_hard_stopping_and_annealing():
    planner, rollout, spec = problem(
        max_iters=20, alpha=0.0, converge_patience=2
    )
    result = planner.optimize_window(rollout, spec=spec)
    assert len(result.loss_history) == 2
    assert result.smoothing_beta == pytest.approx(planner._beta(1))
    planner, rollout, spec = problem(
        max_iters=4,
        smoothing={"beta_start": 2.0, "beta_end": 20.0},
    )
    assert planner._beta(0) == 2.0
    assert planner._beta(3) == 20.0
    assert len(planner.optimize_window(rollout, spec=spec).loss_history) == 4


def test_probability_only_rollout_and_gradients():
    dyn = SingleIntegrator(state_dim=1)
    event = Predicate("reach")
    spec = Eventually(event, interval=[0, 5])
    planner = Planner(dyn, 5, {"max_iters": 100, "w_u": 0.0, "w_du": 0.0})

    def rollout(v):
        progress = torch.cat(
            (torch.zeros(1), dyn.bound_control(v)[:, 0].cumsum(0))
        )
        p = torch.sigmoid(4 * (progress - 1))
        return BeliefRollout(
            create_probability_belief_trajectory(
                event, torch.stack((0.9 * p, p), dim=-1)
            )
        )

    parameters = torch.zeros(5, 1, requires_grad=True)
    lower = spec.smooth_lower(rollout(parameters).belief_trajectory, 10.0)
    planner._objective(dyn.bound_control(parameters), lower).backward()
    assert (
        torch.isfinite(parameters.grad).all()
        and parameters.grad.abs().sum() > 0
    )
    initial = planner.evaluate_controls(rollout, torch.zeros(5, 1), spec=spec)
    result = planner.optimize_window(rollout, spec=spec)
    assert result.rollout.nominal_trace is None and result.rollout.aux is None
    assert result.hard_interval[0] > initial.hard_interval[0] + 0.5


def test_fixed_step_zero_bottleneck():
    planner, rollout, _ = problem()
    spec = Always(GreaterThan(0.4), interval=[0, 5])
    initial = planner.evaluate_controls(rollout, torch.zeros(5, 1), spec=spec)
    result = planner.optimize_window(rollout, spec=spec)
    assert result.hard_interval[0] <= initial.hard_interval[0] + 1e-6


@pytest.mark.parametrize("steps", [0, 1, 3])
def test_mpc_first_control_execution_local_spec_and_warm_start(steps):
    planner, _, spec = problem(max_iters=2)
    state = (torch.zeros(1), torch.eye(1) * 0.1)
    specifications, guesses = [], []
    optimize = planner.optimize_window

    def wrapped(rollout, **kwargs):
        guesses.append(kwargs["init_guess"])
        return optimize(rollout, **kwargs)

    planner.optimize_window = wrapped

    def local_spec(state, step):
        specifications.append((state[0].clone(), step))
        return spec

    result = planner.run_receding_horizon(
        state,
        make_rollout=lambda s, k: gaussian_rollout(planner.dyn, *s),
        make_spec=local_spec,
        execute=lambda s, u, k: planner.dyn.step(*s, u),
        is_done=lambda s, k: False,
        max_steps=steps,
    )
    assert isinstance(result, MPCResult)
    assert result.applied_controls.shape == (steps, 1)
    assert len(result.states) == steps + 1
    assert len(result.window_plans) == steps
    assert result.stopped_reason == "max_steps"
    assert [k for _, k in specifications] == list(range(steps))
    for k, plan in enumerate(result.window_plans):
        torch.testing.assert_close(
            result.applied_controls[k], plan.controls[0]
        )
        expected = planner.dyn.step(*result.states[k], plan.controls[0])
        torch.testing.assert_close(result.states[k + 1], expected)
        torch.testing.assert_close(specifications[k][0], result.states[k][0])
        if k:
            previous = result.window_plans[k - 1].controls
            torch.testing.assert_close(
                guesses[k], torch.cat((previous[1:], previous[-1:]))
            )


@pytest.mark.parametrize("done_at", [0, 1])
def test_mpc_checks_completion_before_planning_and_after_execution(done_at):
    planner, _, spec = problem(max_iters=1)
    state = (torch.zeros(1), torch.eye(1) * 0.1)
    result = planner.run_receding_horizon(
        state,
        make_rollout=lambda s, k: gaussian_rollout(planner.dyn, *s),
        make_spec=lambda s, k: spec,
        execute=lambda s, u, k: planner.dyn.step(*s, u),
        is_done=lambda s, k: k >= done_at,
        max_steps=3,
    )
    assert len(result.window_plans) == done_at
    assert result.stopped_reason == "goal_reached"


def test_invalid_initial_guess_and_obsolete_settings_fail_clearly():
    planner, rollout, spec = problem()
    for guess in (
        torch.zeros(4, 1),
        torch.full((5, 1), float("nan")),
        torch.full((5, 1), 2.0),
    ):
        with pytest.raises(ValueError):
            planner.optimize_window(rollout, spec=spec, init_guess=guess)
    with pytest.raises(ValueError, match="unknown planner settings"):
        Planner(SingleIntegrator(), 3, {"w_dist": 1.0})


def test_bounded_warm_start_at_exact_saturation():
    planner, rollout, spec = problem(max_iters=1)
    result = planner.optimize_window(
        rollout, spec=spec, init_guess=torch.ones(5, 1)
    )
    assert torch.isfinite(result.controls).all()
    assert (result.controls.abs() <= 1).all()


def test_optimization_moves_away_from_saturated_initial_controls():
    planner, rollout, _ = problem(max_iters=60, w_u=0.0, w_du=0.0)
    spec = Eventually(LessThan(-0.2), interval=[0, 5])
    result = planner.optimize_window(
        rollout, spec=spec, init_guess=torch.ones(5, 1)
    )
    assert result.controls.min() < 0.9


def test_callback_loss_and_scores_describe_its_controls():
    planner, rollout, spec = problem(
        max_iters=3, smoothing={"beta_start": 2.0, "beta_end": 20.0}
    )
    records = []
    result = planner.optimize_window(
        rollout,
        spec=spec,
        on_iteration=lambda k, record: records.append(record),
    )
    assert len(records) == 3
    for record in records:
        assert not hasattr(record, "loss_history")
        replay = planner.evaluate_controls(rollout, record.controls, spec=spec)
        assert record.hard_interval == pytest.approx(replay.hard_interval)
        parameters = planner._control_parameters(
            record.controls, margin=torch.finfo(record.controls.dtype).eps
        )
        predicted = rollout(parameters).belief_trajectory
        smooth = spec.smooth_lower(predicted, record.beta).item()
        expected_loss = planner._objective(record.controls, smooth).item()
        assert record.smooth_lower == pytest.approx(smooth)
        assert record.loss == pytest.approx(expected_loss)
    assert result.loss_history == pytest.approx([r.loss for r in records])
    assert result.final_loss == pytest.approx(result.loss_history[-1])


def test_sampled_callbacks_include_first_last_and_early_stop():
    planner, rollout, spec = problem(max_iters=5)
    records = []
    result = planner.optimize_window(
        rollout,
        spec=spec,
        on_iteration=lambda iteration, record: records.append(
            (iteration, record)
        ),
        callback_every=2,
    )
    assert [iteration for iteration, _ in records] == [0, 1, 3, 4]
    for iteration, record in records:
        assert record.loss == pytest.approx(result.loss_history[iteration])
    planner, rollout, spec = problem(
        max_iters=20, alpha=0.0, converge_patience=2
    )
    records = []
    planner.optimize_window(
        rollout,
        spec=spec,
        on_iteration=lambda iteration, record: records.append(iteration),
        callback_every=10,
    )
    assert records == [0, 1]


def test_receding_horizon_reports_optimization_window_numbers():
    planner, _, spec = problem(max_iters=3)
    state = (torch.zeros(1), torch.eye(1) * 0.1)
    seen = []
    planner.run_receding_horizon(
        state,
        make_rollout=lambda current, step: gaussian_rollout(
            planner.dyn, *current
        ),
        make_spec=lambda current, step: spec,
        execute=lambda current, control, step: planner.dyn.step(
            *current, control
        ),
        is_done=lambda current, step: False,
        max_steps=2,
        on_iteration=lambda window, iteration, record: seen.append(
            (window, iteration)
        ),
        callback_every=2,
    )
    assert seen == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def test_result_round_trip(tmp_path):
    planner, rollout, spec = problem(max_iters=1)
    result = planner.optimize_window(rollout, spec=spec)
    path = tmp_path / "plan.pt"
    torch.save(result, path)
    loaded = torch.load(path, weights_only=False)
    assert isinstance(loaded, PlanResult)
    torch.testing.assert_close(loaded.controls, result.controls)
    assert loaded.hard_interval == result.hard_interval


def test_final_smooth_score_uses_beta_end_even_after_one_update():
    planner, rollout, spec = problem(
        max_iters=1, smoothing={"beta_start": 2.0, "beta_end": 20.0}
    )
    result = planner.optimize_window(rollout, spec=spec)
    expected = spec.smooth_lower(result.rollout.belief_trajectory, 20.0).item()
    assert result.smooth_lower == pytest.approx(expected)
