"""Planner contract: silent when asked, independent of the belief type."""

import logging
from pathlib import Path

import pytest
import torch

from models.dynamics import SingleIntegrator
from models.rollouts import BeliefRollout, gaussian_rollout
from pdstl.base import create_probability_belief_trajectory
from pdstl.operators import Eventually, GreaterThan, Predicate
from planning.environment import Environment
from planning.planner import Planner

ROOT = Path(__file__).resolve().parents[1]


def _planner(**config):
    base = {
        "max_iters": 60, "converge_patience": 1, "alpha": 0.0, "scale": -1,
        "w_dist": 0.0, "w_obs": 0.0, "w_visit": 0.0,
    }
    return Planner(SingleIntegrator(), None, 3, config={**base, **config})


def test_quiet_optimisation_emits_no_log_records(caplog):
    planner = _planner()
    spec = Eventually(GreaterThan(0.1), interval=[0, 3])

    with caplog.at_level(logging.INFO, logger="planning"):
        planner.optimize_window(
            gaussian_rollout(planner.dyn, torch.zeros(2), torch.eye(2) * 0.01),
            spec=spec, init_guess=torch.zeros(3, 2),
        )

    assert caplog.records == []


def _probability_rollout(dynamics, event):
    """A non-Gaussian upstream model: p_k = sigmoid(4 (progress_k - 1)), bounds [0.9 p, p]."""

    def rollout(v):
        progress = torch.cat([torch.zeros(1), torch.cumsum(dynamics.bound_control(v)[:, 0], 0) * 0.5])
        p = torch.sigmoid(4.0 * (progress - 1.0))
        bounds = torch.stack([0.9 * p, p], dim=-1)
        nominal = progress.reshape(1, -1, 1)
        return BeliefRollout(create_probability_belief_trajectory(event, bounds), nominal, {})

    return rollout


def test_planner_optimises_a_belief_rollout_it_knows_nothing_about():
    planner = Planner(SingleIntegrator(), None, 5, config={
        "max_iters": 150, "scale": -1, "w_u": 0.0, "w_du": 0.0,
        "w_dist": 0.0, "w_obs": 0.0, "w_visit": 0.0,
    })
    event = Predicate("reach")
    spec = Eventually(event, interval=[0, 5])
    rollout = _probability_rollout(planner.dyn, event)
    initial = spec(rollout(torch.zeros(5, 2)).belief_trajectory)[0, 0, 0].item()

    best, _ = planner.optimize_window(rollout, spec=spec, init_guess=torch.zeros(5, 2))

    assert best.hard_score > initial + 0.5


def test_planner_source_builds_no_concrete_beliefs():
    source = (ROOT / "src/planning/planner.py").read_text()
    for name in ("create_gaussian_belief_trajectory", "create_enclosure_belief_trajectory", "GaussianBelief("):
        assert name not in source


def test_legacy_environment_solve_modes_still_run():
    environment = Environment()
    environment.set_goal([0.5, 1.5], [-0.5, 0.5])
    x0_mean, x0_cov = torch.zeros(2), torch.eye(2) * 0.1

    for extra, mode in (({}, "single_shot"), ({"T_SIM": 2}, "mpc_fixed"), ({"MAX_STEPS": 2}, "mpc_goal")):
        planner = Planner(SingleIntegrator(), environment, 3, config={"max_iters": 2, **extra})
        result = planner.solve(x0_mean, x0_cov, verbose=False)
        assert result["mode"] == mode
        assert torch.isfinite(result["mean_trace"]).all()


def _smooth_problem(**config):
    planner = _planner(scale=5.0, lr=0.05, alpha=2.0, **config)
    spec = Eventually(GreaterThan(0.5), interval=[0, 3])
    rollout = gaussian_rollout(planner.dyn, torch.zeros(2), torch.eye(2) * 0.01)
    return planner, spec, rollout


def test_best_candidate_replays_to_its_own_hard_interval_and_objective():
    planner, spec, rollout = _smooth_problem()

    best, history = planner.optimize_window(rollout, spec=spec, init_guess=torch.zeros(3, 2))

    replay = rollout(torch.atanh(best.controls / planner.dyn.u_max))
    smooth, hard = planner._scores(spec, replay.belief_trajectory)
    objective = planner._objective(replay.nominal_trace, best.controls, smooth)
    torch.testing.assert_close(hard, torch.tensor(best.hard_interval), atol=1e-5, rtol=0)
    assert objective.item() == pytest.approx(best.objective, abs=1e-5)
    assert best.objective == min(history)


def test_smooth_score_carries_gradients_and_hard_score_is_the_exact_interval():
    planner, spec, rollout = _smooth_problem()
    v = torch.zeros(3, 2, requires_grad=True)
    predicted = rollout(v)

    smooth, hard = planner._scores(spec, predicted.belief_trajectory)
    smooth.backward()

    assert not hard.requires_grad
    torch.testing.assert_close(hard, spec(predicted.belief_trajectory, scale=-1)[0, 0].detach())
    assert smooth.item() != pytest.approx(hard[0].item())
    assert v.grad.abs().sum() > 0


def test_alpha_stopping_uses_the_hard_score():
    planner, spec, rollout = _smooth_problem()
    smooth, hard = planner._scores(spec, rollout(torch.zeros(3, 2)).belief_trajectory)
    assert hard[0] < smooth  # the soft max overestimates

    planner.cfg.update(
        lr=0.0, alpha=(hard[0].item() + smooth.item()) / 2,
        converge_patience=1, min_iters=20, max_iters=20,
    )
    _, history = planner.optimize_window(rollout, spec=spec, init_guess=torch.zeros(3, 2))

    assert len(history) == 20
