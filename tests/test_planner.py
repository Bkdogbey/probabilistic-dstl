"""Optimizer contracts, the receding-horizon loop, and altitude safety."""

import logging
from types import SimpleNamespace

import matplotlib
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
from planning.planner import MPCResult, Planner, PlanResult
from planning.runners import run_altitude_safety, setup_problem
from utils import load_config
from visualization.planning import plot_altitude_safety

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def scripted_hard_lower(scores):
    """Give each distinct candidate trajectory the next scripted hard lower bound.

    Memoizing by trajectory rather than by call order keeps the replay of a
    selected candidate consistent with the score it was selected on.
    """
    assigned, remaining = {}, iter(scores)

    def probability_interval(trajectory):
        key = tuple(
            round(value, 4)
            for value in trajectory.trace.mean.flatten().tolist()
        )
        if key not in assigned:
            assigned[key] = next(remaining)
        return torch.tensor([assigned[key], 1.0])

    return probability_interval


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
    expected += 4 * (controls[1:] - controls[:-1]).square().sum()
    torch.testing.assert_close(
        planner._objective(controls, torch.tensor(0.7)), expected
    )


def test_one_iteration_returns_the_updated_controls_and_replays():
    planner, rollout, spec = problem(max_iters=1)
    result = planner.optimize_window(rollout, spec=spec)
    assert result.controls.abs().sum() > 0
    assert len(result.loss_history) == 1
    assert len(result.smooth_history) == 2
    assert len(result.hard_lower_history) == 2
    assert result.selected_iteration == 0
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


def test_best_hard_candidate_wins_and_selection_leaves_gradients_alone():
    scores = (0.30, 0.82, 0.55, 0.40)
    planner, rollout, spec = problem(max_iters=3, alpha=0.95)
    spec.probability_interval = scripted_hard_lower(scores)
    observed = []
    result = planner.optimize_window(
        rollout, spec=spec, on_iteration=lambda k, p: observed.append(p)
    )
    # The best exact lower bound wins: the first update, not the final
    # smooth iterate.
    assert result.selected_iteration == 0
    assert not result.threshold_met
    assert result.hard_interval[0] == pytest.approx(0.82)
    assert result.hard_lower_history == pytest.approx(list(scores))
    torch.testing.assert_close(result.controls, observed[0].controls)

    # The same optimization without a threshold: identical iterates and losses,
    # so hard selection never reached the gradient path.
    baseline_planner, baseline_rollout, baseline_spec = problem(max_iters=3)
    baseline = baseline_planner.optimize_window(
        baseline_rollout, spec=baseline_spec
    )
    assert baseline.loss_history == pytest.approx(result.loss_history)
    torch.testing.assert_close(baseline.controls, observed[-1].controls)


def test_later_smooth_updates_cannot_discard_a_feasible_candidate():
    planner, rollout, spec = problem(max_iters=3, alpha=0.7)
    spec.probability_interval = scripted_hard_lower((0.4, 0.8, 0.6, 0.5))
    observed = []
    result = planner.optimize_window(
        rollout, spec=spec, on_iteration=lambda k, p: observed.append(p)
    )
    torch.testing.assert_close(result.controls, observed[0].controls)
    assert result.hard_interval[0] == pytest.approx(0.8)
    assert result.threshold_met


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


def test_alpha_neither_stops_nor_selects():
    planner, rollout, spec = problem(max_iters=4, alpha=0.95)
    spec.probability_interval = scripted_hard_lower(
        (0.1, 0.96, 0.97, 0.98, 0.99)
    )
    result = planner.optimize_window(rollout, spec=spec)
    assert len(result.loss_history) == 4
    assert result.hard_interval[0] == pytest.approx(0.99)
    assert result.threshold_met


def test_beta_anneals_geometrically():
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
    # Step 0 fixes the exact lower bound at first, so the certificate is flat
    # while the gradient leaves saturation; give it the whole budget.
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
        assert record.control_cost == pytest.approx(
            planner._control_cost(record.controls).item()
        )
    assert result.loss_history == pytest.approx([r.loss for r in records])
    assert result.final_loss == pytest.approx(result.loss_history[-1])


def test_sampled_callbacks_include_first_and_last():
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
    assert loaded.smooth_history == result.smooth_history


def test_selection_maximizes_the_exact_lower_bound_not_cost_at_alpha():
    planner, _, _ = problem(alpha=0.8)

    def candidate(lower, cost):
        return SimpleNamespace(hard_interval=(lower, 1.0), control_cost=cost)

    candidates = [
        (-1, candidate(0.79, 0.1)),
        (0, candidate(0.85, 2.0)),
        (1, candidate(0.82, 1.0)),
    ]
    iteration, selected = planner._select_candidate(candidates)
    assert iteration == 0
    assert selected.hard_interval[0] == 0.85


def test_selection_breaks_ties_within_tolerance_by_cost():
    planner, _, _ = problem(alpha=0.9, candidate_tolerance=1e-3)

    def candidate(lower, cost):
        return SimpleNamespace(hard_interval=(lower, 1.0), control_cost=cost)

    candidates = [
        (-1, candidate(0.70, 2.0)),
        (0, candidate(0.7005, 1.0)),
        (1, candidate(0.72, 3.0)),
        (2, candidate(0.7195, 0.5)),
    ]
    iteration, selected = planner._select_candidate(candidates)
    assert iteration == 2
    assert selected.control_cost == 0.5


def test_selected_smooth_score_is_reported_at_beta_end():
    """An early winner is scored at the final smoothing, not its own beta."""
    planner, rollout, spec = problem(
        max_iters=4,
        alpha=0.7,
        smoothing={"beta_start": 2.0, "beta_end": 20.0},
    )
    spec.probability_interval = scripted_hard_lower(
        (0.4, 0.8, 0.5, 0.45, 0.42)
    )
    result = planner.optimize_window(rollout, spec=spec)
    assert result.selected_iteration == 0
    assert planner._beta(0) == pytest.approx(2.0)
    assert result.smoothing_beta == pytest.approx(20.0)
    beliefs = result.rollout.belief_trajectory
    assert result.smooth_lower == pytest.approx(
        spec.smooth_lower(beliefs, 20.0).item()
    )
    assert result.smooth_lower != pytest.approx(
        spec.smooth_lower(beliefs, 2.0).item()
    )


@pytest.fixture(scope="module")
def altitude_problem():
    s = setup_problem(
        load_config("configs/scenarios/altitude_safety.yaml"), device="cpu"
    )
    spec = Always(GreaterThan(s.cfg["threshold"]), interval=[1, s.cfg["H"]])
    initial = s.planner.evaluate_controls(s.rollout, s.init_guess, spec=spec)
    return s, spec, initial


@pytest.fixture(scope="module")
def altitude_result():
    return run_altitude_safety(show=False, save=False)


def test_altitude_improves_and_replays(altitude_problem, altitude_result):
    s, spec, initial = altitude_problem
    assert altitude_result.controls.shape == (s.cfg["H"], 1)
    assert initial.hard_interval[0] < 0.5
    assert altitude_result.hard_interval[0] > 0.95
    assert altitude_result.smooth_lower > initial.smooth_lower
    assert altitude_result.controls.abs().max() <= s.dyn.u_max
    replay = s.planner.evaluate_controls(
        s.rollout, altitude_result.controls, spec=spec
    )
    assert replay.hard_interval == pytest.approx(
        altitude_result.hard_interval, abs=1e-6
    )


def test_altitude_three_panels(altitude_problem, altitude_result, tmp_path):
    s, _, initial = altitude_problem
    fig, axes = plot_altitude_safety(
        altitude_result,
        initial=initial,
        dt=s.cfg["dt"],
        threshold=s.cfg["threshold"],
        u_max=s.dyn.u_max,
        show=False,
        save_path=str(tmp_path / "altitude.png"),
    )
    assert len(axes) == 3
    assert (tmp_path / "altitude.png").exists()
    assert (tmp_path / "altitude.pdf").exists()
    for ax in axes:
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert len(labels) == len(set(labels))
    plt.close(fig)
