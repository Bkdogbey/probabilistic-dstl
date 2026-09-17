"""Planner contract: silent when asked, independent of the belief type."""

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
from planning.planner import Planner

ROOT = Path(__file__).resolve().parents[1]


def _planner(**config):
    base = {
        "max_iters": 60, "converge_patience": 1, "alpha": 0.0, "smoothing": {"enabled": False},
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


def _probability_rollout(dynamics, event, *, diagnostics=True):
    """A non-Gaussian upstream model: p_k = sigmoid(4 (progress_k - 1)), bounds [0.9 p, p]."""

    def rollout(v):
        progress = torch.cat([torch.zeros(1), torch.cumsum(dynamics.bound_control(v)[:, 0], 0) * 0.5])
        p = torch.sigmoid(4.0 * (progress - 1.0))
        bounds = torch.stack([0.9 * p, p], dim=-1)
        trajectory = create_probability_belief_trajectory(event, bounds)
        if not diagnostics:
            return BeliefRollout(trajectory)
        nominal = progress.reshape(1, -1, 1)
        return BeliefRollout(trajectory, nominal, {})

    return rollout


def test_planner_optimises_a_belief_rollout_it_knows_nothing_about():
    planner = Planner(SingleIntegrator(), None, 5, config={
        "max_iters": 150, "smoothing": {"enabled": False}, "w_u": 0.0, "w_du": 0.0,
        "w_dist": 0.0, "w_obs": 0.0, "w_visit": 0.0,
    })
    event = Predicate("reach")
    spec = Eventually(event, interval=[0, 5])
    rollout = _probability_rollout(planner.dyn, event)
    initial = spec(rollout(torch.zeros(5, 2)).belief_trajectory)[0, 0, 0].item()

    best, _ = planner.optimize_window(rollout, spec=spec, init_guess=torch.zeros(5, 2))

    assert best.hard_lower > initial + 0.5


def test_planner_source_builds_no_concrete_beliefs():
    source = (ROOT / "src/planning/planner.py").read_text(encoding="utf-8")
    for name in ("create_gaussian_belief_trajectory", "GaussianBelief("):
        assert name not in source
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
            assert all(
                "Belief" not in alias.name or alias.name == "BeliefRollout"
                for alias in node.names
            )
        else:
            continue
        assert all(
            not module.startswith("models") or module == "models.rollouts"
            for module in modules
        )


def test_pdstl_imports_no_model_implementation():
    for path in (ROOT / "src/pdstl").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(alias.name for alias in node.names)]
            else:
                continue
            assert not any(
                name.split(".")[0] in {"models", "planning", "baselines"}
                or "Gaussian" in name
                for name in names
            ), path


def test_planner_optimises_without_diagnostics_with_control_regularisation():
    planner = _planner(max_iters=100, alpha=2.0)
    event = Predicate("reach")
    spec = Eventually(event, interval=[0, 3])
    rollout = _probability_rollout(planner.dyn, event, diagnostics=False)
    initial = spec(rollout(torch.zeros(3, 2)).belief_trajectory)[0, 0, 0].item()

    best, history = planner.optimize_window(
        rollout, spec=spec, init_guess=torch.zeros(3, 2)
    )

    assert planner.cfg["w_u"] > 0 and planner.cfg["w_du"] > 0
    assert best.rollout.nominal_trace is None
    assert best.rollout.aux is None
    assert best.hard_lower > initial + 0.2
    replay = rollout(torch.atanh(best.controls / planner.dyn.u_max))
    score, exact = planner._scores(spec, replay.belief_trajectory)
    objective = planner._objective(None, best.controls, score)
    assert objective.item() == pytest.approx(best.objective, abs=1e-5)
    torch.testing.assert_close(exact, torch.tensor(best.hard_interval))


@pytest.mark.parametrize("beta", [None, 5.0])
def test_belief_only_rollout_keeps_gradients_to_controls(beta):
    planner = _planner()
    event = Predicate("reach")
    rollout = _probability_rollout(planner.dyn, event, diagnostics=False)
    v = torch.zeros(3, 2, requires_grad=True)
    predicted = rollout(v)
    smooth, _ = planner._scores(
        Eventually(event, interval=[0, 3]), predicted.belief_trajectory, beta
    )
    objective = planner._objective(None, planner.dyn.bound_control(v), smooth)
    objective.backward()

    assert v.grad is not None
    assert torch.isfinite(v.grad).all()
    assert v.grad.abs().sum() > 0


@pytest.mark.parametrize("heuristic", ["w_dist", "w_obs", "w_visit"])
def test_enabled_shaping_requires_nominal_trace(heuristic):
    planner = _planner(**{heuristic: 1.0})
    event = Predicate("reach")
    rollout = _probability_rollout(planner.dyn, event, diagnostics=False)

    with pytest.raises(ValueError, match=rf"{heuristic}.*nominal_trace"):
        planner.optimize_window(
            rollout, spec=Eventually(event, interval=[0, 3]),
            init_guess=torch.zeros(3, 2),
        )


@pytest.mark.parametrize("with_nominal", [False, True])
@pytest.mark.parametrize("aux_kind", ["absent", "empty", "populated"])
def test_detach_diagnostics_preserves_beliefs_and_handles_optional_fields(
    with_nominal, aux_kind
):
    event = Predicate("reach")
    v = torch.zeros(3, 2, requires_grad=True)
    original = _probability_rollout(SingleIntegrator(), event)(v)
    nominal = original.nominal_trace if with_nominal else None
    aux = None if aux_kind == "absent" else {}
    if aux_kind == "populated":
        aux["progress"] = original.nominal_trace
    predicted = BeliefRollout(original.belief_trajectory, nominal, aux)

    detached = predicted.detach_diagnostics()

    assert detached.belief_trajectory is predicted.belief_trajectory
    if nominal is None:
        assert detached.nominal_trace is None
    else:
        torch.testing.assert_close(detached.nominal_trace, nominal)
        assert not detached.nominal_trace.requires_grad
        assert nominal.requires_grad
    if aux is None:
        assert detached.aux is None
    else:
        assert detached.aux is not aux
        assert detached.aux.keys() == aux.keys()
        for name, trace in aux.items():
            torch.testing.assert_close(detached.aux[name], trace)
            assert not detached.aux[name].requires_grad
            assert trace.requires_grad
    event(detached.belief_trajectory).sum().backward()
    assert torch.isfinite(v.grad).all()
    assert v.grad.abs().sum() > 0


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
    smoothing = {"enabled": True, "beta_start": 2.0, "beta_end": 20.0}
    planner = _planner(smoothing=smoothing, lr=0.05, alpha=2.0, **config)
    spec = Eventually(GreaterThan(0.5), interval=[0, 3])
    rollout = gaussian_rollout(planner.dyn, torch.zeros(2), torch.eye(2) * 0.01)
    return planner, spec, rollout


def test_planner_handles_one_dimensional_controls():
    dyn = SingleIntegrator(state_dim=1)
    planner = Planner(dyn, None, 3, config={
        "max_iters": 5, "alpha": 2.0, "w_dist": 0.0, "w_obs": 0.0, "w_visit": 0.0,
    })
    rollout = gaussian_rollout(dyn, torch.zeros(1), torch.eye(1) * 0.01)
    spec = Eventually(GreaterThan(0.1), interval=[0, 3])

    best, _ = planner.optimize_window(rollout, spec=spec, init_guess=torch.zeros(3, 1))

    assert best.controls.shape == (3, 1)
    assert planner._init_controls(None).shape == (3, 1)
    assert planner._empty_u_trace().shape == (1, 0, 1)


def test_on_iteration_observes_every_iterate_without_changing_the_result():
    planner, spec, rollout = _smooth_problem()
    seen = []

    best, history = planner.optimize_window(
        rollout, spec=spec, init_guess=torch.zeros(3, 2),
        on_iteration=lambda k, candidate: seen.append((k, candidate.objective)),
    )
    plain, plain_history = planner.optimize_window(
        rollout, spec=spec, init_guess=torch.zeros(3, 2)
    )

    assert [k for k, _ in seen] == list(range(len(history)))
    assert [objective for _, objective in seen] == history
    assert history == plain_history
    assert best.hard_interval == plain.hard_interval


def test_evaluate_controls_scores_returned_controls_like_the_optimiser():
    planner, spec, rollout = _smooth_problem()
    best, _ = planner.optimize_window(rollout, spec=spec, init_guess=torch.zeros(3, 2))

    replay = planner.evaluate_controls(rollout, best.controls, spec=spec)

    torch.testing.assert_close(
        torch.tensor(replay.hard_interval), torch.tensor(best.hard_interval), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(replay.controls, best.controls, atol=1e-5, rtol=0)


def test_checkpoint_selection_uses_hard_lower_and_control_cost():
    planner, spec, rollout = _smooth_problem()
    seen = []

    best, _ = planner.optimize_window(
        rollout, spec=spec, init_guess=torch.zeros(3, 2),
        on_iteration=lambda k, candidate: seen.append(candidate),
    )

    top = max(c.hard_lower for c in seen)
    assert best.hard_lower == top
    assert best.control_cost == min(c.control_cost for c in seen if c.hard_lower == top)
    assert seen[best.iteration] == best


def test_smooth_lower_carries_gradients_and_hard_interval_is_detached():
    planner, spec, rollout = _smooth_problem()
    v = torch.zeros(3, 2, requires_grad=True)
    predicted = rollout(v)

    score, exact = planner._scores(spec, predicted.belief_trajectory, beta=5.0)
    score.backward()

    assert not exact.requires_grad
    torch.testing.assert_close(exact, spec(predicted.belief_trajectory, scale=-1)[0, 0].detach())
    assert score.item() != pytest.approx(exact[0].item())
    assert torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0


def test_beta_anneals_geometrically_and_is_none_when_smoothing_is_off():
    planner, _, _ = _smooth_problem(max_iters=11)
    betas = [planner._beta(k) for k in range(11)]

    assert betas[0] == pytest.approx(2.0) and betas[-1] == pytest.approx(20.0)
    assert all(b2 / b1 == pytest.approx(10 ** 0.1) for b1, b2 in zip(betas, betas[1:]))
    assert _planner()._beta(0) is None


def test_alpha_stopping_uses_hard_lower():
    planner, spec, rollout = _smooth_problem()
    smooth, exact = planner._scores(spec, rollout(torch.zeros(3, 2)).belief_trajectory, planner._beta(0))
    assert smooth < exact[0]  # the normalized smooth max underestimates the max

    planner.cfg.update(
        lr=0.0, alpha=(exact[0].item() + smooth.item()) / 2,
        converge_patience=1, min_iters=20, max_iters=20,
    )
    _, history = planner.optimize_window(rollout, spec=spec, init_guess=torch.zeros(3, 2))

    assert len(history) == 1
