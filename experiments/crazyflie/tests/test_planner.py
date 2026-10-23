"""Planner correctness; never a research outcome (which plan is better)."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import minsnap
import numpy as np
import planner
import pytest
import run
import torch
from geometry import Task
from tracking_model import TrackingModel
from trajectory import (
    file_hash,
    load_plan,
    read_plan,
    reference_plan,
    save_plan,
)


@pytest.fixture
def small(cfg):
    """A short budget: planning correctness, not plan quality."""
    return replace(
        cfg,
        planner={
            **cfg.planner,
            "max_iters": 6,
            "sweep_iters": 2,
            "lr_grid": [0.003, 0.01],
        },
    )


@pytest.fixture
def model():
    return TrackingModel(
        0.2,
        [0.8, 0.8],
        [0.01, -0.01],
        [[2e-5, 5e-6], [5e-6, 2e-5]],
        [0.0, 0.0],
        1e-4 * np.eye(2),
        "model_x",
        "abc",
    )


@pytest.fixture
def problem(small, model):
    return planner.Problem(small, Task(small), model, reference_plan(small))


@pytest.mark.parametrize("condition", planner.CONDITIONS)
def test_stored_score_is_recomputed_from_the_saved_rows(
    small, model, problem, tmp_path, condition
):
    task = Task(small)
    record = planner.plan(
        small, task, model, reference_plan(small), condition, "low"
    )
    path = save_plan(record, tmp_path / f"{condition}.json")
    saved = read_plan(path)
    rows = np.asarray(saved["coefficients"])
    fresh = problem.exact_score(condition, problem.rows_mean(rows))
    assert saved["exact_score"] == fresh
    # Three separate fields: feasibility, score and threshold.
    assert isinstance(saved["feasible"], bool)
    assert saved["threshold_met"] == (fresh >= saved["threshold"])
    assert saved["feasible"] == (saved["problems"] == [])
    # Provenance.
    assert saved["model_id"] == "model_x" and saved["model_hash"] == "abc"
    assert saved["config_hash"] == small.plan_hash
    assert saved["execution_hash"] == small.execution_hash
    assert saved["reference_hash"] == file_hash(small.reference)
    assert len(saved["prediction"]["mean"]) == small.steps + 1


@pytest.mark.parametrize("condition", planner.CONDITIONS)
def test_score_gradient_reaches_the_waypoints(problem, condition):
    def score(free):
        mean = problem.grid @ problem.waypoints(free) + problem.m
        return problem.smooth_score(condition, mean, 10.0)

    free = problem.init.clone().requires_grad_()
    assert torch.autograd.gradcheck(score, (free,), atol=1e-6)
    (grad,) = torch.autograd.grad(score(free), free)
    assert grad.abs().sum() > 0


def candidate(iteration, exact, feasible=True, smooth=0.0, penalty=0.0):
    return {
        "iteration": iteration,
        "exact": exact,
        "feasible": feasible,
        "smooth": smooth,
        "penalty": penalty,
    }


def test_an_earlier_feasible_candidate_beats_a_better_infeasible_one():
    chosen = planner.select(
        [candidate(-1, 0.5), candidate(0, 0.7), candidate(1, 0.9, False)]
    )
    assert chosen["iteration"] == 0


def test_ties_use_smooth_score_then_penalty_then_the_earlier_iterate():
    def pick(*candidates):
        return planner.select(list(candidates))["iteration"]

    assert (
        pick(candidate(0, 1.0, smooth=0.2), candidate(1, 1.0, smooth=0.3)) == 1
    )
    assert (
        pick(candidate(0, 1.0, penalty=2.0), candidate(1, 1.0, penalty=1.0))
        == 1
    )
    assert pick(candidate(-1, 0.0), candidate(0, 0.0), candidate(1, 0.0)) == -1


def test_no_feasible_candidate_returns_the_best_one_marked_infeasible():
    chosen = planner.select(
        [candidate(-1, 0.2, False), candidate(0, 0.4, False)]
    )
    assert chosen["iteration"] == 0 and not chosen["feasible"]


@pytest.mark.parametrize("condition", planner.CONDITIONS)
def test_the_reference_is_kept_when_no_iterate_beats_it(problem, condition):
    best = planner.optimize(
        problem, condition, lr=0.0, iterations=3, w_snap=0.0
    )
    assert best["iteration"] == -1
    assert np.array_equal(best["waypoints"][1:], problem.init.numpy())


def test_both_conditions_get_the_same_budget(small, model):
    task, reference = Task(small), reference_plan(small)
    budgets = []
    for condition in planner.CONDITIONS:
        objective = planner.plan(
            small, task, model, reference, condition, "low"
        )["objective"]
        budgets.append(
            (
                objective["iterations"],
                objective["sweep_iters"],
                [s["lr"] for s in objective["lr_sweep"]],
                objective["w_lim"],
                objective["snap_fraction"],
            )
        )
    assert budgets[0] == budgets[1]


def write_variant(cfg, tmp_path, **changes):
    plan = {**json.loads(Path(cfg.reference).read_text()), **changes}
    path = tmp_path / "variant.json"
    path.write_text(json.dumps(plan))
    return path


def test_infeasible_plans_never_load(cfg, tmp_path):
    path = write_variant(cfg, tmp_path, feasible=False, problems=["x"])
    with pytest.raises(ValueError, match="not feasible"):
        load_plan(path, cfg)


def test_below_threshold_plans_load_only_when_allowed(cfg, tmp_path):
    path = write_variant(cfg, tmp_path, exact_score=0.5, threshold_met=False)
    with pytest.raises(ValueError, match="allow-below-threshold"):
        load_plan(path, cfg)
    assert load_plan(path, cfg, allow_below_threshold=True)["feasible"]
    args = SimpleNamespace(
        plan=str(path), fan="off", allow_below_threshold=False
    )
    with pytest.raises(SystemExit, match="allow-below-threshold"):
        run.select_plan(cfg, args)
    args.allow_below_threshold = True
    assert run.select_plan(cfg, args)[0] == path


def test_planning_cannot_overwrite_the_frozen_reference(cfg):
    before = Path(cfg.reference).read_bytes()
    with pytest.raises(SystemExit, match="reference stays frozen"):
        run.cmd_plan(cfg, SimpleNamespace(output=cfg.reference))
    assert Path(cfg.reference).read_bytes() == before


def test_prediction_trails_the_curve_by_the_model_delay(small, model):
    model.delay = 0.5
    reference = reference_plan(small)
    problem = planner.Problem(small, Task(small), model, reference)
    rows = np.asarray(reference["coefficients"])
    behind = minsnap.evaluate(rows, problem.times - 0.5)[:, :2]
    mean = problem.rows_mean(rows) - problem.m
    assert np.allclose(mean.numpy(), behind)
    free = torch.as_tensor(np.asarray(reference["waypoints"])[1:])
    grid_mean = problem.grid @ problem.waypoints(free)
    assert np.allclose(grid_mean.numpy(), behind, atol=1e-5)
