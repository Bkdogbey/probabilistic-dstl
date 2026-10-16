"""2D inspection: map, specs R1-R3 and their parts on noise-free paths."""

import pytest
import torch

from experiments import inspection_2d as exp
from utils import load_config

START, SLOT, P1 = (1.0, 1.0), (4.0, 1.5), (6.0, 1.5)
C, D = (8.1, 4.6), (8.6, 8.6)


@pytest.fixture(scope="module")
def specs():
    cfg = load_config(exp.CONFIG)
    return exp.build_specs(exp.build_environment(cfg), cfg), cfg["H"]


def _path(legs, H):
    """Hold each point for its count of samples; stay at the last one."""
    points = [point for point, count in legs for _ in range(count)]
    return points + [points[-1]] * (H + 1 - len(points))


def _truth(specs, legs):
    specs, H = specs
    path = _path(legs, H)
    return {
        name: (
            bool(exp.satisfied(formula, path)),
            {k: bool(exp.satisfied(p, path)) for k, p in parts.items()},
        )
        for name, (formula, parts) in specs.items()
    }


GOOD = [(START, 1), (SLOT, 1), (P1, 11), (C, 1), (D, 1)]
CASES = {
    "good": (GOOD, (True, True, True)),
    "C before inspection": (
        [(START, 1), (SLOT, 1), (C, 1), *GOOD[2:]],
        (True, True, False),
    ),
    "nine dwell steps": (
        [(START, 1), (SLOT, 1), (P1, 10), (C, 1), (D, 1)],
        (True, False, False),
    ),
    "D before C": (
        [(START, 1), (SLOT, 1), (P1, 11), (D, 1), (C, 1), (D, 1)],
        (True, True, False),
    ),
    "never leaves": ([(START, 1)], (False, False, False)),
}


@pytest.mark.parametrize("case", CASES)
def test_specs_and_parts_agree(specs, case):
    legs, expected = CASES[case]
    truth = _truth(specs, legs)
    assert tuple(truth[name][0] for name in ("R1", "R2", "R3")) == expected
    for holds, parts in truth.values():
        assert holds == all(parts.values())


def test_staying_home_is_an_inspection_failure_not_ordering(specs):
    parts = _truth(specs, [(START, 1)])["R3"][1]
    assert parts == {
        "safety": True,
        "ordering": True,
        "inspection": False,
        "deadline": False,
    }


def test_unsafe_start_raises():
    cfg = load_config(exp.CONFIG)
    cfg["x0_mean"] = [4.0, 3.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="not safe"):
        exp.build_environment(cfg)


def test_slot_route_seed_holds_in_p1_and_satisfies_r3():
    cfg = load_config(exp.CONFIG)
    env = exp.build_environment(cfg)
    r3 = exp.build_specs(env, cfg)["R3"][0]
    length, _, _ = exp.prescribed_route(env, cfg, ["P1"])
    slot, _, _ = exp.prescribed_route(env, cfg, cfg["routes"]["slot"]["stops"])
    assert 5.0 < length < slot
    s = exp.problem(cfg, **cfg["nominal"])
    reference, controls = exp.route_seed(env, cfg, s.dyn, "slot")
    assert reference.shape == (cfg["H"] + 1, 2)
    assert controls.abs().max() <= s.dyn.u_max
    s.spec = r3
    mean = exp._mean(s, controls).rollout.aux["mean_trace"][0]
    assert exp.longest_stay(env, mean, "P1") > cfg["dwell"]
    assert exp.satisfied(r3, mean)


def test_failure_class_is_the_first_failing_part():
    flags = {
        "safety": torch.tensor([True, True, False, True]),
        "ordering": torch.tensor([True, False, False, True]),
        "deadline": torch.tensor([True, False, True, True]),
    }
    classes = exp.failure_classes(flags)
    assert classes.tolist() == ["", "ordering", "safety", ""]


def test_slot_traversal_is_read_from_the_mean_path():
    cfg = load_config(exp.CONFIG)
    env = exp.build_environment(cfg)
    assert exp.slot_traversed(env, _path(GOOD, cfg["H"]))
    assert not exp.slot_traversed(env, _path([(START, 1), (C, 1)], 3))


@pytest.mark.parametrize("noise", ["low", "medium", "high"])
def test_refinement_initial_score_matches_stlcg_plan(noise):
    cfg = load_config(exp.CONFIG)
    for method in cfg["methods"].values():
        method["max_iters"] = 1
    env = exp.build_environment(cfg)
    formula = exp.build_specs(env, cfg)["R2"][0]
    group, plans, _ = exp.stlcg_plan(cfg, env, ("R2", formula))
    base = next(r for r in group if r["selected"])
    controls = plans[exp._key(base)]
    s = exp.problem(cfg, q_std=exp.noise_level(cfg, noise))
    expected = s.planner.evaluate_controls(s.rollout, controls, spec=formula)
    row, _, log = exp.run_start(
        cfg,
        env,
        ("R2", formula),
        "pdstl",
        base["start"],
        noise,
        init_guess=controls,
        stlcg_time=base["stlcg_time"],
    )
    assert row["solver_status"] == "completed"
    assert row["initial_score"] == pytest.approx(expected.hard_interval[0])
    torch.testing.assert_close(log[0][2], controls, rtol=0, atol=0)
    assert row["planning_time"] == row["stlcg_time"] + row["refine_time"]


def test_refinement_preserves_near_saturated_controls():
    cfg = load_config(exp.CONFIG)
    s = exp.problem(cfg, q_std=cfg["noise"]["medium"])
    controls = torch.full((cfg["H"], 2), 0.99999 * cfg["u_max"])
    restored = s.dyn.bound_control(s.planner._init_controls(controls))
    torch.testing.assert_close(restored, controls, rtol=0, atol=1e-7)


def test_stlcg_selects_by_final_mean_robustness(monkeypatch):
    cfg = load_config(exp.CONFIG)

    def run_start(cfg, env, spec, method, start, **kwargs):
        assert method == "stlcg"
        score = {"slot": -0.1, "top": 0.2}[start]
        controls = torch.full((cfg["H"], 2), score)
        row = {
            "spec": "R2",
            "noise": "none",
            "method": method,
            "start": start,
            "solver_status": "completed",
            "own_score": score,
            "final_score": 1 - score,
            "planning_time": 2.0,
        }
        return row, controls, [(-1, 0.0, controls), (0, 2.0, controls)]

    monkeypatch.setattr(exp, "run_start", run_start)
    rows, _, logs = exp.stlcg_plan(cfg, None, ("R2", None))
    best = next(r for r in rows if r["selected"])
    assert best["start"] == "top"
    assert best["planning_time"] == best["stlcg_time"] == 4.0
    assert logs[exp._key(best)][-1][1] == 4.0


@pytest.mark.slow
def test_every_step_runs_end_to_end(tmp_path, monkeypatch):
    cfg = load_config(exp.CONFIG)
    for method in cfg["methods"].values():
        method["max_iters"] = 2
    cfg.update(lr_grid=[0.001], log_every=1)
    cfg["monte_carlo"]["samples"] = 100
    monkeypatch.setattr(exp, "OUTPUTS", tmp_path)
    for step in ("pilot", "runs", "monte-carlo", "paper"):
        exp.STEPS[step](cfg)
    runs = exp._read("runs.csv")
    assert len(runs) == 11
    assert sum(r["method"] == "stlcg" for r in runs) == 6
    assert sum(r["method"] == "pdstl" for r in runs) == 5
    assert sum(r["selected"] == "True" for r in runs) == 8
    checked = exp._read("monte_carlo.csv")
    assert len(checked) == 10
    scores = {
        (r["spec"], r["noise"], r["method"]): float(r["hard_score"])
        for r in checked
    }
    convergence = exp._read("convergence.csv")
    selected = exp.selected_runs()
    for (spec, noise, method), run in selected.items():
        if method != "pdstl":
            continue
        base = selected[(spec, noise, "stlcg")]
        assert run["start"] == base["start"]
        assert float(run["initial_score"]) == pytest.approx(
            scores[(spec, noise, "stlcg")]
        )
        assert float(run["planning_time"]) == pytest.approx(
            float(run["stlcg_time"]) + float(run["refine_time"])
        )
        initial = next(r for r in convergence if exp._key(r) == exp._key(run))
        assert float(initial["time_s"]) == float(base["planning_time"])
        assert float(initial["score"]) == pytest.approx(
            float(run["initial_score"])
        )
    assert (tmp_path / "figure1.pdf").exists()
    assert "Optimizer wall time" in (tmp_path / "table1.tex").read_text()
