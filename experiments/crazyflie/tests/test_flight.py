"""Flight flow, cleanup, logging, analysis and the run.py pipeline."""

import csv
import json
import sys
import threading
import time

import pytest
import yaml

import analysis
import config
import flight
import planners
import plots
import run
from flight_log import FlightLog

AWAY = (0.5, -0.1, 0.0)  # placed away from the start
LANDED = ["notify_setpoint_stop", "land", "hl_stop", "disarm", "disconnect"]


def _fly(cfg, drone, stop=None):
    t, xyz, info = planners.s_path(cfg)
    log = FlightLog(cfg, t, xyz, plan_info=info, dry_run=True)
    summary = flight.run_flight(
        cfg, t, xyz, log, stop=stop, connect_fn=lambda c: drone
    )
    return summary, log


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _phases(path):
    seen = []
    for row in _rows(path):
        if not seen or seen[-1] != row["phase"]:
            seen.append(row["phase"])
    return seen


# --- flight flow and logs ------------------------------------------------------


def test_nominal_dry_run_logs_everything(cfg):
    drone = flight.SimDrone(home=AWAY)
    summary, log = _fly(cfg, drone)

    assert summary["outcome"] == "completed"
    assert summary["last_phase"] == "landed"
    assert summary["error"] is None
    assert summary["method"] == "s_path"
    for name in (
        "config.yaml",
        "trajectory.csv",
        "plan.json",
        "commands.csv",
        "telemetry.csv",
        "summary.json",
    ):
        assert (log.dir / name).exists()
    assert json.loads((log.dir / "summary.json").read_text()) == summary
    assert yaml.safe_load((log.dir / "config.yaml").read_text()) == cfg
    assert _phases(log.dir / "commands.csv") == [
        "takeoff",
        "to_start",
        "trajectory",
        "land",
    ]
    assert "positioning" in _phases(log.dir / "telemetry.csv")
    assert summary["telemetry_rows"] == len(_rows(log.dir / "telemetry.csv"))
    assert summary["tracking"]["rmse_m"] < 0.2
    assert drone.calls[-5:] == LANDED
    assert drone.calls.index("arm") < drone.calls.index("takeoff")
    # Away from the start: climb to transit_z, cross, then descend.
    assert drone.calls.count("go_to") == 2
    commands = _rows(log.dir / "commands.csv")
    assert float(commands[0]["z_cmd"]) == cfg["flight"]["transit_z"]


def test_placed_at_start_takes_off_directly(cfg):
    drone = flight.SimDrone(home=(*cfg["planning"]["start"], 0.0))
    summary, log = _fly(cfg, drone)
    assert summary["outcome"] == "completed"
    assert drone.calls.count("go_to") == 1
    commands = _rows(log.dir / "commands.csv")
    assert float(commands[0]["z_cmd"]) == cfg["planning"]["height"]


def test_flown_ring_passage_is_reported(cfg):
    cfg["environment"]["rings"]["enabled"] = True
    summary, _ = _fly(cfg, flight.SimDrone(home=AWAY))
    assert summary["outcome"] == "completed"
    assert [r["name"] for r in summary["rings"]] == [
        "ring_1",
        "ring_2",
        "ring_3",
    ]
    assert all(r["passed"] for r in summary["rings"])


def test_run_ids_are_unique(cfg):
    t, xyz, _ = planners.s_path(cfg)
    a = FlightLog(cfg, t, xyz)
    b = FlightLog(cfg, t, xyz)
    assert a.run_id != b.run_id and a.dir != b.dir
    assert "_ring_course_s_path_" in a.run_id
    a.finish("error", "test")
    b.finish("error", "test")


# --- interruption and cleanup --------------------------------------------------


def test_error_mid_trajectory_lands_and_keeps_partial_logs(cfg):
    drone = flight.SimDrone(home=AWAY, fail_after=10)
    summary, log = _fly(cfg, drone)
    assert summary["outcome"] == "error"
    assert summary["last_phase"] == "trajectory"
    assert "simulated failure" in summary["error"]
    assert summary["traceback"]
    assert drone.calls[-5:] == LANDED
    commands = _rows(log.dir / "commands.csv")
    assert sum(r["phase"] == "trajectory" for r in commands) == 10
    assert commands[-1]["phase"] == "cleanup_land"
    assert len(_rows(log.dir / "telemetry.csv")) > 0


def test_stop_request_interrupts_and_lands(cfg):
    cfg["planning"]["s_path"]["speed"] = 0.5  # time to interrupt mid-path
    drone = flight.SimDrone(home=AWAY)
    stop = threading.Event()
    t, xyz, _ = planners.s_path(cfg)
    log = FlightLog(cfg, t, xyz)

    def trigger():
        while log.phase != "trajectory":
            time.sleep(0.01)
        time.sleep(0.3)
        stop.set()

    threading.Thread(target=trigger, daemon=True).start()
    summary = flight.run_flight(
        cfg, t, xyz, log, stop=stop, connect_fn=lambda c: drone
    )
    assert summary["outcome"] == "interrupted"
    assert summary["last_phase"] == "trajectory"
    assert drone.calls[-5:] == LANDED


def test_unstable_positioning_never_takes_off(cfg):
    drone = flight.SimDrone(noise=0.1)
    summary, _ = _fly(cfg, drone)
    assert summary["outcome"] == "error"
    assert summary["last_phase"] == "positioning"
    assert "not stable" in summary["error"]
    assert "arm" not in drone.calls and "takeoff" not in drone.calls
    assert drone.calls[-2:] == ["disarm", "disconnect"]


def test_failed_landing_still_stops_and_disconnects(cfg):
    drone = flight.SimDrone(home=AWAY, fail_on={"land"}, fail_after=5)
    summary, _ = _fly(cfg, drone)
    assert summary["outcome"] == "error"
    assert drone.calls[-4:] == [
        "land",
        "stop_setpoint",
        "disarm",
        "disconnect",
    ]


def test_geofence_violation_aborts(cfg):
    cfg["bounds"]["margin"] = 0.0
    cfg["bounds"]["z"] = [0.0, 0.45]  # transit climb to 0.65 m overshoots
    drone = flight.SimDrone(home=AWAY)
    summary, _ = _fly(cfg, drone)
    assert summary["outcome"] == "error"
    assert "geofence" in summary["error"]
    assert "disconnect" in drone.calls


def test_missing_hardware_stack_fails_cleanly(cfg):
    try:
        import irobot  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("irobot is installed; this checks the offline path")
    t, xyz, _ = planners.s_path(cfg)
    log = FlightLog(cfg, t, xyz)
    summary = flight.run_flight(cfg, t, xyz, log)
    assert summary["outcome"] == "error"
    assert summary["last_phase"] == "connect"
    assert (log.dir / "summary.json").exists()


def test_offline_modules_do_not_import_hardware():
    assert "cflib" not in sys.modules
    assert "ros_sugar" not in sys.modules


# --- analysis ----------------------------------------------------------------


def test_analysis_matches_logged_summary(cfg):
    cfg["environment"]["rings"]["enabled"] = True
    summary, log = _fly(cfg, flight.SimDrone(home=AWAY))
    loaded = analysis.load_run(log.dir)
    assert loaded["summary"] == summary
    assert loaded["plan_info"]["method"] == "s_path"
    assert analysis.metrics(loaded) == {
        "tracking": summary["tracking"],
        "rings": summary["rings"],
    }
    tau, flown, reference = analysis.trajectory_segment(loaded)
    assert tau[0] >= 0 and flown.shape == reference.shape


def test_partial_run_can_be_analysed_and_plotted(cfg, tmp_path):
    _, log = _fly(cfg, flight.SimDrone(home=AWAY, fail_after=10))
    loaded = analysis.load_run(log.dir)
    assert loaded["summary"]["outcome"] == "error"
    plots.plot_run(loaded, tmp_path / "flight")
    assert (tmp_path / "flight.png").exists()
    row = analysis.summary_row(loaded)
    assert row["outcome"] == "error" and row["rmse_m"] is not None
    assert row["method"] == "s_path"


# --- run.py pipeline -----------------------------------------------------------


def test_pipeline_plan_fly_analyze(cfg, cfg_file):
    path = cfg_file()
    logs = config.resolve(cfg, "dir")
    assert run.main(["--config", path, "plan", "--no-show"]) == 0
    plan = config.resolve(cfg, "plans_dir") / "ring_course" / "s_path"
    assert (plan / "plan.json").exists() and (plan / "plan.png").exists()

    assert run.main(["--config", path, "fly", "--dry-run"]) == 0
    assert run.main(["--config", path, "fly", "--dry-run"]) == 0
    (run_dir,) = [d for d in analysis.run_dirs(logs)][-1:]
    for name in ("plan.json", "plan.png", "flight.png", "summary.json"):
        assert (run_dir / name).exists()
    plan_info = json.loads((plan / "plan.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["plan_signature"] == plan_info["signature"]

    assert run.main(["--config", path, "analyze", "--all", "--no-show"]) == 0
    rows = _rows(logs / "runs.csv")
    assert [r["outcome"] for r in rows] == ["completed", "completed"]
    assert (logs / "repeats.png").exists()


def test_pipeline_pdstl_dry_run(cfg, cfg_file):
    cfg["environment"]["rings"]["enabled"] = True
    path = cfg_file(cfg)
    assert (
        run.main(["--config", path, "fly", "--method", "pdstl", "--dry-run"])
        == 0
    )
    (run_dir,) = analysis.run_dirs(config.resolve(cfg, "dir"))
    loaded = analysis.load_run(run_dir)
    assert "_pdstl_" in run_dir.name
    assert loaded["summary"]["method"] == "pdstl"
    assert "p_lower" in loaded["plan_info"]
    assert all(r["passed"] for r in loaded["summary"]["rings"])


def test_declined_flight_does_not_fly(cfg, cfg_file, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    monkeypatch.setattr(run, "launch", lambda *a: pytest.fail("flew"))
    assert run.main(["--config", cfg_file(), "fly"]) == 1
    assert analysis.run_dirs(config.resolve(cfg, "dir")) == []


def test_analyze_without_runs(cfg, cfg_file):
    assert run.main(["--config", cfg_file(), "analyze", "--no-show"]) == 1
