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


@pytest.mark.parametrize("winner", ["slot", "top"])
def test_pdstl_selects_the_higher_scoring_refinement(
    tmp_path, monkeypatch, winner
):
    cfg = load_config(exp.CONFIG)
    monkeypatch.setattr(exp, "OUTPUTS", tmp_path)
    monkeypatch.setattr(exp, "CASES", [("R2", "medium")])
    monkeypatch.setattr(exp, "build_environment", lambda cfg: None)
    monkeypatch.setattr(
        exp, "build_specs", lambda env, cfg: {"R2": (None, {})}
    )
    monkeypatch.setattr(exp, "_score_log", lambda *args, **kwargs: [])
    calls = []

    def run_start(cfg, env, spec, method, start, noise="none", **kwargs):
        controls = torch.full((cfg["H"], 2), 0.1 if start == "slot" else 0.2)
        if method == "stlcg":
            score = controls[0, 0].item()
            seconds = 2.0 if start == "slot" else 3.0
        else:
            torch.testing.assert_close(kwargs["init_guess"], controls)
            assert kwargs["stlcg_time"] == 5.0
            score = 0.9 if start == winner else 0.4
            seconds = 7.0 if start == "slot" else 11.0
        calls.append((method, start))
        row = {
            "spec": spec[0],
            "noise": noise,
            "method": method,
            "start": start,
            "solver_status": "completed",
            "own_score": score,
            "final_score": score,
            "planning_time": seconds,
            "stlcg_time": kwargs.get("stlcg_time", seconds),
            "refine_time": seconds if method == "pdstl" else 0.0,
        }
        return row, controls, [(-1, 0.0, controls), (0, seconds, controls)]

    monkeypatch.setattr(exp, "run_start", run_start)
    exp.runs(cfg)
    rows = exp._read("runs.csv")
    assert calls == [
        ("stlcg", "slot"),
        ("stlcg", "top"),
        ("pdstl", "slot"),
        ("pdstl", "top"),
    ]
    selected = exp.selected_runs()
    assert selected[("R2", "medium", "stlcg")]["start"] == "top"
    best = selected[("R2", "medium", "pdstl")]
    assert best["start"] == winner
    assert float(best["final_score"]) == max(
        float(r["final_score"]) for r in rows if r["method"] == "pdstl"
    )
    assert float(best["stlcg_time"]) == 5.0
    assert float(best["refine_time"]) == 18.0
    assert float(best["planning_time"]) == 23.0


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
    assert len(runs) == 16
    assert sum(r["method"] == "stlcg" for r in runs) == 6
    assert sum(r["method"] == "pdstl" for r in runs) == 10
    assert sum(r["selected"] == "True" for r in runs) == 8
    checked = exp._read("monte_carlo.csv")
    assert len(checked) == 10
    convergence = exp._read("convergence.csv")
    selected = exp.selected_runs()
    for (spec, noise, method), run in selected.items():
        if method != "pdstl":
            continue
        base = selected[(spec, noise, "stlcg")]
        candidates = [
            r
            for r in runs
            if r["spec"] == spec
            and r["noise"] == noise
            and r["method"] == method
        ]
        assert len(candidates) == 2
        assert float(run["final_score"]) == max(
            float(r["final_score"]) for r in candidates
        )
        source = next(
            r
            for r in reversed(convergence)
            if r["spec"] == spec
            and r["noise"] == noise
            and r["method"] == "stlcg"
            and r["start"] == run["start"]
        )
        assert float(run["initial_score"]) == pytest.approx(
            float(source["score"])
        )
        assert float(run["planning_time"]) == pytest.approx(
            float(run["stlcg_time"]) + float(run["refine_time"])
        )
        initial = next(r for r in convergence if exp._key(r) == exp._key(run))
        assert float(initial["time_s"]) >= float(base["planning_time"])
        assert float(initial["score"]) == pytest.approx(
            float(run["initial_score"])
        )
        final = next(
            r for r in reversed(convergence) if exp._key(r) == exp._key(run)
        )
        assert float(final["time_s"]) == float(run["planning_time"])
    assert (tmp_path / "figure1.pdf").exists()
    assert "Optimizer wall time" in (tmp_path / "table1.tex").read_text()
    assert "both refinements" in (tmp_path / "table1.tex").read_text()
    assert "\\% (slot)" in (tmp_path / "table1.tex").read_text()
    for row in exp._read("table1.csv"):
        assert row["STLCG route"] in ("slot", "top")
        assert row["pdSTL route"] in ("slot", "top")


def test_instance_zero_and_deterministic_geometry(monkeypatch):
    cfg = load_config(exp.CONFIG)
    original = exp.deepcopy(cfg)
    # Geometry tests avoid repeating the full optimizer gate forty times.
    monkeypatch.setattr(exp, "nominal_instance", lambda cfg: ([], {}, []))
    first = list(exp.generate_instances(cfg))
    second = list(exp.generate_instances(cfg))
    assert first[0][1] == cfg == original
    assert first[0][2] == 0
    assert [(i, c, r) for i, c, r, _ in first] == [
        (i, c, r) for i, c, r, _ in second
    ]
    for index, candidate, _, _ in first:
        lower, upper = candidate["obstacles"]
        centre = (lower["y_range"][1] + upper["y_range"][0]) / 2
        width = upper["y_range"][0] - lower["y_range"][1]
        assert 0.5 <= width <= 0.7
        assert 1.4 <= centre <= 1.6
        assert candidate["routes"]["slot"]["stops"][0] == [4.0, centre]
        assert candidate["routes"]["slot"]["narrow"] == [4.0, centre]
        for rack, base in zip(candidate["obstacles"], cfg["obstacles"]):
            assert rack["x_range"] == base["x_range"]
        for name, box in candidate["regions"].items():
            for axis in ("x_range", "y_range"):
                a, b = box[axis]
                assert (
                    cfg["bounds"][axis][0] <= a < b <= cfg["bounds"][axis][1]
                )
                assert b - a == pytest.approx(
                    cfg["regions"][name][axis][1]
                    - cfg["regions"][name][axis][0]
                )
                assert (
                    abs(
                        sum(box[axis]) / 2
                        - sum(cfg["regions"][name][axis]) / 2
                    )
                    <= 0.2
                )
            for rack in candidate["obstacles"]:
                assert any(
                    box[axis][1] < rack[axis][0]
                    or rack[axis][1] < box[axis][0]
                    for axis in ("x_range", "y_range")
                )
        for axis in (0, 1):
            assert (
                abs(candidate["x0_mean"][axis] - cfg["x0_mean"][axis]) <= 0.2
            )
        if index:
            assert candidate != cfg


@pytest.mark.parametrize("failure", ["bounds", "rack"])
def test_instance_geometry_rejects_invalid_regions(failure):
    cfg = load_config(exp.CONFIG)
    if failure == "bounds":
        cfg["regions"]["P1"]["x_range"] = [-0.1, 0.9]
    else:
        cfg["regions"]["P1"] = {"x_range": [3.6, 4.4], "y_range": [2.0, 3.0]}
    assert not exp.valid_geometry(cfg)


@pytest.mark.parametrize("flags", [(True, True), (False, True), (True, False)])
def test_nominal_instance_requires_both_r3_mean_paths(monkeypatch, flags):
    cfg = load_config(exp.CONFIG)
    group = [
        {"solver_status": "completed", "mean_satisfies_spec": flag}
        for flag in flags
    ]

    def nominal(cfg, env, spec):
        assert spec[0] == "R3"
        return group, {}, {}

    monkeypatch.setattr(exp, "stlcg_plan", nominal)
    result = exp.nominal_instance(cfg)
    assert (result is not None) == all(flags)


def test_instance_rejections_replay_without_nominal_checks(monkeypatch):
    cfg = load_config(exp.CONFIG)
    cfg["instances"]["count"] = 3
    calls = []

    def gate(candidate):
        calls.append(candidate)
        # Accept the base; reject the first random draw, then accept all.
        return None if len(calls) == 2 else ([], {}, [])

    monkeypatch.setattr(exp, "nominal_instance", gate)
    generated = list(exp.generate_instances(cfg))
    assert [r for _, _, r, _ in generated] == [0, 1, 0]
    saved = [exp.instance_parameters(i, c, r) for i, c, r, _ in generated]
    monkeypatch.setattr(
        exp, "nominal_instance", lambda cfg: pytest.fail("rechecked saved map")
    )
    replayed = list(exp.generate_instances(cfg, saved))
    assert [(i, c, r) for i, c, r, _ in replayed] == [
        (i, c, r) for i, c, r, _ in generated
    ]


def test_instance_rejection_limit(monkeypatch):
    cfg = load_config(exp.CONFIG)
    cfg["instances"].update(count=2, max_tries=2)
    calls = []

    def gate(candidate):
        calls.append(candidate)
        return ([], {}, []) if len(calls) == 1 else None

    monkeypatch.setattr(exp, "nominal_instance", gate)
    with pytest.raises(
        RuntimeError, match="instance 1: no valid map in 2 attempts"
    ):
        list(exp.generate_instances(cfg))
    assert len(calls) == 3


@pytest.mark.slow
def test_instances_pairing_and_resume(tmp_path, monkeypatch):
    cfg = load_config(exp.CONFIG)
    cfg["instances"]["count"] = 2
    cfg["instances"]["monte_carlo"]["samples"] = 40
    for method in cfg["methods"].values():
        method["max_iters"] = 2
    monkeypatch.setattr(exp, "OUTPUTS", tmp_path)
    # Exercise the pipeline with short optimizations; gate logic is tested above.
    gates, cases, seeds = [], [], []

    def short_nominal(candidate):
        gates.append(candidate)
        env = exp.build_environment(candidate)
        formula = exp.build_specs(env, candidate)["R3"][0]
        return exp.stlcg_plan(candidate, env, ("R3", formula))

    case, mc = exp.instance_case, exp.monte_carlo

    def record_case(candidate, index, spec, nominal=None):
        cases.append((index, spec))
        return case(candidate, index, spec, nominal)

    def record_mc(s, result, **kwargs):
        seeds.append(kwargs)
        return mc(s, result, **kwargs)

    monkeypatch.setattr(exp, "nominal_instance", short_nominal)
    monkeypatch.setattr(exp, "instance_case", record_case)
    monkeypatch.setattr(exp, "monte_carlo", record_mc)
    exp.instances(cfg)
    rows = exp._read("instance_runs.csv")
    assert len(rows) == 12
    assert all(
        set(p) == set(exp.METHODS) for p in exp.instance_pairs(rows).values()
    )
    assert len(exp._read("instances.csv")) == len(gates) == 2
    assert seeds == [
        {"samples": 40, "seed": 3000 + i} for i in range(2) for _ in range(6)
    ]
    cases.clear()
    exp.instances(cfg)
    assert cases == [] and len(gates) == 2
    assert exp._read("instance_runs.csv") == rows
    exp._write([r for r in rows if r["instance"] == "0"], "instance_runs.csv")
    exp.instances(cfg)
    assert cases == [(1, spec) for spec in ("R1", "R2", "R3")]
    assert len(gates) == 2
    restored = exp._read("instance_runs.csv")
    assert len(restored) == 12
    assert [r for r in restored if r["instance"] == "0"] == rows[:6]
    cases.clear()
    exp._write(restored[:-1], "instance_runs.csv")
    exp.instances(cfg)
    assert cases == [(1, "R3")]
    assert len(exp._read("instance_runs.csv")) == 12
    summaries = exp.instance_paper(cfg)
    assert all(
        r["pdstl_higher"] + r["tie"] + r["pdstl_lower"] == 2 for r in summaries
    )
    assert (tmp_path / "instances.pdf").exists()
    assert (tmp_path / "table_instances.tex").exists()


def test_instance_summary_uses_paired_differences_and_overlapping_intervals(
    tmp_path, monkeypatch
):
    cfg = load_config(exp.CONFIG)
    cfg["instances"]["count"] = 3
    monkeypatch.setattr(exp, "OUTPUTS", tmp_path)
    rates = [(0.8, 0.95), (0.9, 0.89), (0.95, 0.8)]
    rows = []
    for i, (st, pd) in enumerate(rates):
        for spec in ("R1", "R2", "R3"):
            for method, rate in (("stlcg", st), ("pdstl", pd)):
                rows.append(
                    {
                        "instance": i,
                        "spec": spec,
                        "method": method,
                        "success_rate": rate,
                        "ci_low": rate - 0.02,
                        "ci_high": rate + 0.02,
                        "route": "slot"
                        if method == "pdstl" or i == 0
                        else "top",
                        "planning_time": (10 if method == "stlcg" else 20) + i,
                    }
                )
    exp._write(rows, "instance_runs.csv")
    summaries = exp.instance_paper(cfg)
    for row in summaries:
        assert row["paired_difference_mean"] == pytest.approx(-0.01 / 3)
        assert (row["pdstl_higher"], row["tie"], row["pdstl_lower"]) == (
            1,
            1,
            1,
        )
        assert row["stlcg_success_mean"] == pytest.approx(
            (0.8 + 0.9 + 0.95) / 3
        )
        assert row["pdstl_success_min"] == 0.8
        assert row["pdstl_success_max"] == 0.95
        assert row["stlcg_slot_share"] == pytest.approx(1 / 3)
        assert row["pdstl_slot_share"] == 1.0
        assert (row["stlcg_time_s"], row["pdstl_time_s"]) == (11, 21)
    tex = (tmp_path / "table_instances.tex").read_text()
    assert "both refinements" in tex
    assert "\\%" in tex and "\\\\%" not in tex
    exp._write(rows[:-1], "instance_runs.csv")
    with pytest.raises(ValueError, match="complete paired results"):
        exp.instance_paper(cfg)
