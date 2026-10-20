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


def test_reference_is_feasible_reaches_the_goal_and_fits_memory(cfg):
    plan = reference_plan(cfg)
    rows = np.asarray(plan["coefficients"])
    assert plan["start"] == [1.0, -2.0] and cfg.start == (1.0, -2.0)
    assert minsnap.total_duration(rows) == 18.0 == cfg.duration
    assert cfg.steps == 90 and len(rows) <= minsnap.MAX_PIECES
    task = Task(cfg)
    assert check_plan(cfg, task, dict(plan))["problems"] == []
    curve = analysis.curve_xy(rows)
    assert all(task.requirements(curve, curve[-1]).values())
    assert plan["source_hash"] and "velocity reference" in plan["method"]


def test_reference_ignores_optimizer_tuning_but_not_the_arena(cfg):
    tuned = replace(cfg, planner={**cfg.planner, "max_iters": 3})
    assert (
        load_plan(cfg.reference, tuned)["coefficients"]
        == (json.loads(Path(cfg.reference).read_text())["coefficients"])
    )
    moved = replace(
        cfg,
        obstacles=(
            replace(cfg.obstacles[0], x=(0.0, 0.1)),
            *cfg.obstacles[1:],
        ),
    )
    with pytest.raises(ValueError, match="another arena"):
        load_plan(cfg.reference, moved)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"waypoints": [[1.0, -2.0]] * 10}, "miss the waypoints"),
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
    path = import_plan(waypoints, [2.0], cfg, output)
    imported = load_plan(path, cfg)
    assert imported["coefficients"] == reference["coefficients"]
    assert imported["source_hash"] == file_hash(waypoints)
    with pytest.raises(ValueError, match="never replaced"):
        import_plan(waypoints, [2.0], cfg, output)

    csv = tmp_path / "curve.csv"
    run.cmd_export(cfg, SimpleNamespace(plan=str(path), csv=str(csv)))
    assert np.array_equal(
        minsnap.read_csv(csv), np.asarray(reference["coefficients"])
    )


def test_explicit_selection_never_chooses_newest(cfg, tmp_path):
    args = SimpleNamespace(plan=None, fan=0, allow_below_threshold=False)
    selected, _ = run.select_plan(cfg, args)
    assert str(selected) == cfg.reference
    plan = json.loads(Path(cfg.reference).read_text())
    plan["fan"] = 6
    other = tmp_path / "fan6.json"
    other.write_text(json.dumps(plan))
    args.plan = str(other)
    with pytest.raises(SystemExit, match="for fan"):
        run.select_plan(cfg, args)


def test_recorded_plan_is_portable_and_hash_checked(
    cfg, tmp_path, monkeypatch
):
    path = tmp_path / "reference.json"
    path.write_bytes(Path(cfg.reference).read_bytes())
    monkeypatch.setattr(analysis, "PLANS", tmp_path)
    meta = {
        "plan_id": "reference",
        "plan_path": "/missing/reference.json",
        "plan_hash": file_hash(path),
    }
    assert (
        analysis._plan(meta)["coefficients"]
        == reference_plan(cfg)["coefficients"]
    )
    assert analysis._plan({**meta, "plan_hash": "different"}) is None


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


def reference_path(cfg, seconds=20.0):
    rows = reference_plan(cfg)["coefficients"]
    return minsnap.evaluate(rows, np.arange(0, seconds, 1 / cfg.log_hz))[:, :2]


def test_evaluate_scores_each_requirement(cfg):
    task = Task(cfg)
    result = analysis.evaluate(recorded(cfg, reference_path(cfg)), task)
    assert result["success"] and result["min_clearance"] > 0
    goal = np.array(task.goal.centre)
    assert result["endpoint_error"] == pytest.approx(
        np.linalg.norm(reference_path(cfg)[-1] - goal), abs=1e-6
    )
    assert result["duration"] == pytest.approx(
        cfg.duration + flight.END_HOLD_S
    )

    bad = np.linspace([0.65, -2.0], [0.65, -0.2], 400)  # through obs_1
    result = analysis.evaluate(recorded(cfg, bad), task)
    assert not result["success"] and not result["requirements"]["obs_1"]
    assert not result["requirements"]["goal"] and result["min_clearance"] < 0

    excluded = recorded(cfg, reference_path(cfg), outcome="aborted")
    excluded["meta"].update(abort_kind="DataFault", abort_reason="state stale")
    assert analysis.evaluate(excluded, task) == {"excluded": "state stale"}


def test_abort_boundary_excludes_landing_from_mission_metrics(cfg):
    trial = recorded(cfg, reference_path(cfg), outcome="aborted")
    trial["meta"].update(abort_kind="TaskViolation", mission_t_end=1.0)
    after = trial["state"]["t_host"] > 1.0
    trial["state"]["x"][after] = 100.0
    trial["state"]["z"][after] = 0.0
    result = analysis.evaluate(trial, Task(cfg))
    assert not result["success"] and np.isnan(result["endpoint_error"])
    assert result["min_clearance"] > 0 and result["requirements"]["altitude"]
    assert result["duration"] == 1.0


def test_altitude_scoring_uses_the_same_average_as_execution(cfg):
    trial = recorded(cfg, reference_path(cfg))
    trial["state"]["z"][40] = 0.6
    assert analysis.evaluate(trial, Task(cfg))["success"]
    trial["state"]["z"][40:60] = 0.6
    assert not analysis.evaluate(trial, Task(cfg))["requirements"]["altitude"]


def test_held_out_coverage_uses_the_plan_prediction(cfg):
    offset = np.array([0.01, -0.01])
    trial = recorded(cfg, reference_path(cfg) + offset)
    rows = reference_plan(cfg)["coefficients"]
    r = minsnap.evaluate(rows, minsnap.grid_times(rows, cfg.dt))[:, :2]
    plan = {
        "prediction": {
            "mean": (r + offset).tolist(),
            "cov": np.tile(1e-4 * np.eye(2), (len(r), 1, 1)).tolist(),
        }
    }
    coverage = analysis.coverage([trial], plan)
    assert all(value == 1.0 for value in coverage.values())
    assert analysis.coverage([trial], {"prediction": None}) is None


@pytest.mark.parametrize("setting", [0, 6, 12, 18])
def test_numeric_fan_setting_is_recorded(cfg, tmp_path, setting):
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
    assert f"fan{setting:02d}" in log.id and log.meta["fan"] == setting
    assert log.meta["coefficients"] == reference_plan(cfg)["coefficients"]


def test_unmeasured_config_marks_flights_as_not_fit_data(cfg):
    assert replace(cfg, measured=False).ready_for_data(0)
