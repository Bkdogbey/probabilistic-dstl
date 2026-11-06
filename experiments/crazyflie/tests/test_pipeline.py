"""Reference, plan records, import/export and mission scoring."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import analysis
import flight
import minsnap
import numpy as np
import pytest
import run
from geometry import Task
from logger import TrialLog
from trajectory import (
    check_plan,
    file_hash,
    import_plan,
    load_plan,
    read_plan,
    reference_plan,
)


def test_reference_is_the_arc_feasible_and_ends_in_the_goal(cfg):
    plan = reference_plan(cfg)
    rows = np.asarray(plan["coefficients"])
    assert np.allclose(plan["start"], [0.69, -1.97]) and cfg.altitude == 0.3
    assert minsnap.total_duration(rows) == 13.0 == cfg.duration
    assert cfg.steps == 65 and len(rows) <= minsnap.MAX_PIECES
    assert len(cfg.obstacles) == 3
    task = Task(cfg)
    assert check_plan(cfg, task, dict(plan))["problems"] == []
    curve = analysis.curve_xy(rows)
    assert np.allclose(curve[-1], [0.69, -0.22], atol=1e-5)  # at the goal
    assert all(task.requirements(curve).values())
    assert plan["source"].endswith("plans/reference.csv")


def test_reference_ignores_optimizer_tuning_but_not_the_arena(cfg):
    tuned = replace(cfg, planner={**cfg.planner, "max_iters": 3})
    assert (
        load_plan(cfg.reference, tuned)["coefficients"]
        == (json.loads(Path(cfg.reference).read_text())["coefficients"])
    )
    moved = replace(cfg, workspace=replace(cfg.workspace, x=(0.0, 1.2)))
    with pytest.raises(ValueError, match="another arena"):
        load_plan(cfg.reference, moved)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"waypoints": [[0.69, -1.97]] * 14}, "miss the waypoints"),
        ({"dt": 0.35}, "whole number"),
        ({"durations": [2.0] * 8}, "plan needs"),
    ],
)
def test_read_plan_rejects_inconsistent_records(
    cfg, tmp_path, change, message
):
    plan = {**json.loads(Path(cfg.reference).read_text()), **change}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match=message):
        read_plan(path)


def test_import_rebuilds_the_reference_and_export_round_trips(cfg, tmp_path):
    reference = reference_plan(cfg)
    waypoints = tmp_path / "waypoints.csv"
    np.savetxt(waypoints, reference["waypoints"], delimiter=",", fmt="%.17g")
    output = tmp_path / "imported.json"
    path = import_plan(waypoints, [1.0], cfg, output)
    imported = load_plan(path, cfg)
    assert imported["coefficients"] == reference["coefficients"]
    assert imported["source_hash"] == file_hash(waypoints)
    with pytest.raises(ValueError, match="never replaced"):
        import_plan(waypoints, [1.0], cfg, output)

    csv = tmp_path / "curve.csv"
    run.cmd_export(cfg, SimpleNamespace(plan=str(path), csv=str(csv)))
    assert np.array_equal(
        minsnap.read_csv(csv), np.asarray(reference["coefficients"])
    )


def test_explicit_selection_never_chooses_newest(cfg, tmp_path):
    args = SimpleNamespace(plan=None, fan="off", allow_below_threshold=False)
    selected, _ = run.select_plan(cfg, args)
    assert str(selected) == cfg.reference
    plan = json.loads(Path(cfg.reference).read_text())
    plan["fan"] = "low"
    other = tmp_path / "low.json"
    other.write_text(json.dumps(plan))
    args.plan = str(other)
    with pytest.raises(SystemExit, match="for fan"):
        run.select_plan(cfg, args)


def test_recorded_plan_is_portable_and_matched_by_its_curve(
    cfg, tmp_path, monkeypatch
):
    path = tmp_path / "reference.json"
    path.write_bytes(Path(cfg.reference).read_bytes())
    monkeypatch.setattr(analysis, "PLANS", tmp_path)
    meta = {
        "plan_id": "reference",
        "plan_path": "/missing/reference.json",
        "coefficients": reference_plan(cfg)["coefficients"],
    }
    assert (
        analysis._plan(meta)["coefficients"]
        == reference_plan(cfg)["coefficients"]
    )
    moved = np.asarray(meta["coefficients"]) + 0.01
    assert analysis._plan({**meta, "coefficients": moved.tolist()}) is None


def recorded(cfg, path, outcome="completed"):
    """A trial whose state follows `path` [M, 2] at log_hz from t_play = 0."""
    t = np.arange(len(path)) / cfg.log_hz
    zeros = np.zeros(len(path))
    meta = {
        "outcome": outcome,
        "mission_outcome": outcome,
        "mission_t0": 0.0,
        "t_play": 0.0,
        "dt": cfg.dt,
        "duration": cfg.duration,
        "coefficients": reference_plan(cfg)["coefficients"],
        "altitude": cfg.altitude,
        "altitude_tolerance": cfg.altitude_tolerance,
        "altitude_average_s": flight.ALTITUDE_AVG_S,
        "log_hz": cfg.log_hz,
    }
    if outcome == "completed":
        meta["mission_t_end"] = cfg.duration + flight.END_HOLD_S
    state = {
        "t_host": t,
        "device_ms": 1000 * t,
        "x": path[:, 0],
        "y": path[:, 1],
        "z": zeros + cfg.altitude,
        "vx": zeros,
        "vy": zeros,
    }
    return {"meta": meta, "state": state, "target": None}


def reference_path(cfg):
    rows = reference_plan(cfg)["coefficients"]
    t = np.arange(0, cfg.duration + 1, 1 / cfg.log_hz)
    return minsnap.evaluate(rows, t)[:, :2]


def test_evaluate_scores_each_requirement(cfg):
    task = Task(cfg)
    result = analysis.evaluate(recorded(cfg, reference_path(cfg)), task)
    assert result["success"] and result["min_clearance"] > 0

    bad = np.linspace([0.5, -1.75], [2.5, -1.75], 480)  # leaves, no goal
    result = analysis.evaluate(recorded(cfg, bad), task)
    assert not result["success"] and not result["requirements"]["workspace"]
    assert not result["requirements"]["goal"] and result["min_clearance"] < 0

    excluded = recorded(cfg, reference_path(cfg), outcome="aborted")
    excluded["meta"].update(abort_kind="DataFault", abort_reason="state stale")
    assert analysis.evaluate(excluded, task) == {"excluded": "state stale"}


def test_first_counted_keeps_flight_order_and_earlier_exclusions():
    results = [
        {"success": True},
        {"excluded": "no connection"},
        {"success": False},
        {"success": True},
    ]
    assert analysis.first_counted(results, 2) == results[:3]
    assert analysis.first_counted(results, 9) == results


def test_abort_boundary_excludes_landing_from_mission_metrics(cfg):
    trial = recorded(cfg, reference_path(cfg), outcome="aborted")
    trial["meta"].update(abort_kind="TaskViolation", mission_t_end=1.0)
    after = trial["state"]["t_host"] > 1.0
    trial["state"]["x"][after] = 100.0
    trial["state"]["z"][after] = 0.0
    result = analysis.evaluate(trial, Task(cfg))
    assert not result["success"] and not result["requirements"]["goal"]
    assert result["min_clearance"] > 0 and result["height_held"]


def test_height_is_reported_beside_success_not_in_it(cfg):
    trial = recorded(cfg, reference_path(cfg))
    trial["state"]["z"][40] = cfg.altitude + 0.35
    assert analysis.evaluate(trial, Task(cfg))["height_held"]
    trial["state"]["z"][40:60] = 0.0  # a dip to the floor, then regained
    result = analysis.evaluate(trial, Task(cfg))
    assert not result["height_held"] and result["fell"]
    assert result["success"]


@pytest.mark.parametrize("setting", ["off", "low", "medium", "high"])
def test_fan_setting_is_recorded(cfg, tmp_path, setting):
    measured = replace(cfg, fan=setting)
    assert measured.ready_for_data(setting) is None
    log = TrialLog.for_flight(
        measured,
        read_plan(cfg.reference),
        Path(cfg.reference),
        purpose="identification",
        fan=setting,
        root=tmp_path,
    )
    assert f"fan-{setting}" in log.id and log.meta["fan"] == setting
    assert log.meta["coefficients"] == reference_plan(cfg)["coefficients"]


def test_unmeasured_config_marks_flights_as_not_fit_data(cfg):
    assert replace(cfg, measured=False).ready_for_data("off")


def test_old_numeric_baseline_reads_as_off():
    import config

    assert config.fan_setting(0) == config.fan_setting(False) == "off"
    with pytest.raises(ValueError):
        config.fan_setting(6)
