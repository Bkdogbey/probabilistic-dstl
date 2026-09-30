"""The hinge comparison entry points: isolated runs and saved outputs."""

import json

import matplotlib

matplotlib.use("Agg")
import torch

from experiments import hinge_comparison, reach_avoid
from pdstl.operators import Always, Eventually
from utils import load_config

MC = {"samples": 50, "seed": 0}


def test_comparison_isolates_runs_and_saves_outputs(tmp_path):
    rows, folder = hinge_comparison.run(
        ("obstacle",),
        max_iters=3,
        monte_carlo_settings=MC,
        output_root=tmp_path,
    )
    assert len(rows) == 6
    for start in hinge_comparison.INITIALIZATIONS:
        same_start = [row for row in rows if row["initialization"] == start]
        assert len({row["initial_hard_lower"] for row in same_start}) == 1
    runs = json.loads((folder / "runs.json").read_text())
    assert [run["planner_config"]["w_margin"] for run in runs[:3]] == [
        0.0,
        0.1,
        0.1,
    ]
    assert all(run["planner_config"]["margin_beta"] is None for run in runs)
    assert all(len(run["hard_lower_history"]) == 4 for run in runs)
    for name in ("results.csv", "settings.json", "hard_lower_history.png"):
        assert (folder / name).is_file()

    # pdSTL-only runs reproduce the existing planner unchanged.
    cfg = load_config(hinge_comparison.SCENARIOS["obstacle"])
    cfg["planner"] = {**cfg["planner"], "max_iters": 3}
    problem = reach_avoid.build(cfg, device="cpu")
    for guess, run in ((problem.init_guess, runs[0]), (None, runs[3])):
        direct = problem.planner.optimize_window(
            problem.rollout, spec=problem.spec, init_guess=guess
        )
        assert direct.hard_lower_history == run["hard_lower_history"]
        assert torch.equal(
            direct.controls, torch.tensor(run["selected_controls"])
        )


def test_conjuncts_split_goal_with_its_window():
    problem = reach_avoid.build(
        load_config(hinge_comparison.SCENARIOS["either_or"]), device="cpu"
    )
    safety, goal, visit = hinge_comparison._conjuncts(problem.spec)
    assert isinstance(safety, Always)
    assert isinstance(goal, Eventually) and isinstance(visit, Eventually)
    assert goal is problem.spec.subformula1.subformula2


def test_diagnostics_leave_optimization_unchanged(tmp_path):
    context = hinge_comparison._Context("cpu", 11, MC)
    overrides = {**hinge_comparison.VARIANTS["hinge_mean"], "w_margin": 1.0}
    plain = hinge_comparison._execute(
        context, "compare", "obstacle", "cold", "hinge_mean", overrides
    )
    observed = hinge_comparison._execute(
        context, "screen", "obstacle", "cold", "hinge_mean", overrides
    )
    for key in ("hard_lower_history", "loss_history", "selected_controls"):
        assert observed[key] == plain[key]
    assert not plain["diagnostics"]
    iterations = [p["iteration"] for p in observed["diagnostics"]]
    assert iterations == [0, 9, 10]
    point = observed["diagnostics"][-1]
    assert point["auxiliary_rho_smooth"] is None
    assert point["active_predicates"][0]["predicate"] == "goal.x"
    assert set(point["gradient_norms"]) == {"pdstl", "hinge", "control"}
    assert -1 <= point["alignment"]["hinge_vs_control"] <= 1
