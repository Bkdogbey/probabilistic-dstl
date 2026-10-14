"""Offline checks: commands, executor, aborts, logs, model fit, evaluation."""

import threading
from dataclasses import replace

import analysis
import numpy as np
import pytest
import tracking_model
from flight import (
    STATE_VARS,
    DataFault,
    SimDrone,
    StateFeed,
    at_start,
    run_flight,
    world_to_body,
)
from logger import TrialLog, aligned_states, load_trial
from planner import (
    Task,
    check_feasible,
    make_plan,
    nominal_path,
    plan_mean,
    plan_pdstl,
    reference_velocities,
)
from tracking_model import TrackingModel, response_matrices


def test_world_to_body():
    assert np.allclose(world_to_body((0.3, 0.1), 0.0), (0.3, 0.1))
    # Facing +y: world +x is the body's right (-y).
    assert np.allclose(world_to_body((0.3, 0.0), 90.0), (0.0, -0.3))


def test_plan_hash_ignores_flight_tuning_but_not_the_path(cfg):
    tuned = replace(
        cfg, flight=replace(cfg.flight, start_speed=0.2), measured=True
    )
    assert tuned.hash != cfg.hash and tuned.plan_hash == cfg.plan_hash
    moved = replace(cfg, reference={**cfg.reference, "speed": 0.1})
    assert moved.plan_hash != cfg.plan_hash


def test_reference_reaches_goal_within_limits(cfg):
    task = Task(cfg)
    v = reference_velocities(cfg)
    assert v.shape == (cfg.steps, 2) and np.abs(v).max() <= cfg.u_max
    path = nominal_path(cfg, cfg.start, v)
    assert len(path) == cfg.steps * cfg.repeats + 1
    assert task.requirements(path, path[-1]) == {
        "workspace": True,
        "obs_1": True,
        "obs_2": True,
        "obs_3": True,
        "goal": True,
    }
    assert check_feasible(cfg, task, cfg.start, v) == []
    through_obstacle = np.tile([0.0, 0.1], (cfg.steps, 1))
    assert check_feasible(cfg, task, (0.65, -2.0), through_obstacle)


def fly_sim(cfg, tmp_path, drone=None, stop=None, steps=5, connect_fn=None):
    plan = {"start": list(cfg.start), "velocities": [[0.1, 0.05]] * steps}
    log = TrialLog(
        {
            "condition": "deterministic",
            "purpose": "pilot",
            "dt": cfg.dt,
            "repeats": cfg.repeats,
        },
        root=tmp_path,
    )
    drone = drone or SimDrone((*cfg.start, 0.0))
    meta = run_flight(cfg, plan, log, stop, connect_fn or (lambda _: drone))
    return meta, load_trial(log.dir), drone


def test_dry_flight_logs_aligned_commands_and_state(cfg, tmp_path):
    meta, trial, drone = fly_sim(cfg, tmp_path)
    assert meta["outcome"] == "completed"
    cmd = trial["commands"]
    hold = round(cfg.flight.hold_end * cfg.stream_hz)
    assert len(cmd["k"]) == 5 * cfg.repeats + hold
    assert np.all(np.diff(cmd["t_sent"]) > 0) and cmd["lateness"].max() < 0.05
    states, controls = aligned_states(trial)
    assert states.shape == (6, 4) and np.allclose(controls, [0.1, 0.05])
    assert np.allclose(
        states[0, :2], cfg.start, atol=cfg.flight.start_tolerance
    )
    assert drone.calls[-3:] == ["hl_stop", "disarm", "disconnect"]


def test_stale_state_aborts_lands_and_keeps_logs(cfg, tmp_path):
    meta, trial, drone = fly_sim(
        cfg, tmp_path, SimDrone((*cfg.start, 0.0), stall_after=8)
    )
    assert (meta["outcome"], meta["abort_kind"]) == ("aborted", "DataFault")
    assert meta["abort_phase"] == "mission" and "mission_t0" in meta
    assert "land" in drone.calls and drone.calls[-1] == "disconnect"
    assert len(trial["commands"]["k"]) >= 8 and len(trial["state"]["x"]) > 0


def test_stop_request_interrupts_and_lands(cfg, tmp_path):
    stop = threading.Event()

    class Stopping(SimDrone):
        def send_hover_setpoint(self, *args):
            super().send_hover_setpoint(*args)
            if self.setpoints == 5:
                stop.set()

    meta, _, drone = fly_sim(cfg, tmp_path, Stopping((*cfg.start, 0.0)), stop)
    assert meta["outcome"] == "interrupted" and "land" in drone.calls


def test_connect_failure_still_records_the_attempt(cfg, tmp_path):
    def fail(_):
        raise DataFault("could not connect")

    meta, trial, _ = fly_sim(cfg, tmp_path, connect_fn=fail)
    assert (
        meta["outcome"] == "aborted"
        and trial["meta"]["abort_phase"] == "connect"
    )


def gate_verdicts(cfg, position, velocity, seconds=8, seed=0):
    """at_start on noisy hover data: 0.02 m position and 0.15 m/s velocity noise."""
    rng, now = np.random.default_rng(seed), [0.0]

    class Log:
        def state(self, row):
            pass

    feed = StateFeed(Log(), lambda: now[0])
    out = []
    for k in range(seconds * cfg.stream_hz):
        now[0] = k / cfg.stream_hz
        x, y = np.add(position, rng.normal(0, 0.02, 2))
        vx, vy = np.add(velocity, rng.normal(0, 0.15, 2))
        values = [x, y, cfg.altitude, vx, vy, 0.0]
        feed.on_state(k * 50, dict(zip(STATE_VARS, values)))
        if now[0] > 2:
            out.append(at_start(feed, np.array(cfg.start), cfg))
    return np.array(out)


def test_start_gate_passes_noisy_hover_but_rejects_offset_and_motion(cfg):
    assert gate_verdicts(cfg, cfg.start, (0, 0)).mean() > 0.9
    assert not gate_verdicts(cfg, np.add(cfg.start, (0.12, 0)), (0, 0)).any()
    assert not gate_verdicts(cfg, cfg.start, (0.3, 0)).any()


# Tracking model


TAU, GAIN = [0.4, 0.55], [0.95, 1.05]
C = np.array([0.002, -0.001, 0.01, -0.005])
Q = np.diag([1e-5, 1e-5, 4e-4, 4e-4])


def synthetic_flights(cfg, count, rng, steps=60):
    A, B = response_matrices(cfg.dt, TAU, GAIN)
    flights = []
    for i in range(count):
        s = [
            rng.multivariate_normal(
                [0.5, -1.0, 0, 0], np.diag([4e-4, 4e-4, 1e-4, 1e-4])
            )
        ]
        u = rng.uniform(-0.3, 0.3, (steps // 5, 2)).repeat(5, axis=0)
        for k in range(steps):
            s.append(
                A @ s[-1]
                + B @ u[k]
                + C
                + rng.multivariate_normal(np.zeros(4), Q)
            )
        meta = {
            "purpose": "identification",
            "outcome": "completed",
            "fan": "off",
            "trial_id": f"t{i:02d}",
        }
        flights.append({"meta": meta, "data": (np.array(s), u)})
    return flights


def test_fit_recovers_model_and_prediction_intervals(cfg, monkeypatch):
    monkeypatch.setattr(tracking_model, "aligned_states", lambda t: t["data"])
    cfg = replace(cfg, model={**cfg.model, "bootstrap": 50})
    report = tracking_model.fit(
        cfg, synthetic_flights(cfg, 60, np.random.default_rng(0))
    )
    assert np.allclose(report["tau"], TAU, rtol=0.1)
    assert np.allclose(report["gain"], GAIN, rtol=0.05)
    fan = report["fans"]["off"]
    assert len(fan["held_out_trials"]) == 20 and not set(
        fan["held_out_trials"]
    ) & set(fan["fit_trials"])
    assert np.allclose(fan["c"], C, atol=3e-3)
    assert np.allclose(np.diag(fan["Q"]), np.diag(Q), rtol=0.3)
    coverage = report["validation"]["off"]["coverage"]
    for (name, (estimate, *_)), nominal in zip(
        coverage.items(), (0.393, 0.865, 0.95)
    ):
        assert abs(estimate - nominal) < 0.1, name


# Planning and evaluation


def test_model_plans_have_matched_provenance(cfg):
    cfg = replace(cfg, planner={**cfg.planner, "max_iters": 10})
    model = TrackingModel(
        cfg.dt,
        [0.4, 0.4],
        [1.0, 1.0],
        np.zeros(4),
        1e-5 * np.eye(4),
        ([*cfg.start, 0, 0], 1e-4 * np.eye(4)),
        cfg.u_max,
        "model_x",
        "abc",
    )
    task, init = Task(cfg), reference_velocities(cfg)
    for name, optimize in (("pdstl", plan_pdstl), ("mean_model", plan_mean)):
        controls, score = optimize(cfg, task, model, init)
        assert controls.shape == (cfg.steps, 2) and "accepted" in score
        plan = make_plan(
            cfg,
            name,
            "fan-off",
            cfg.start,
            controls,
            task,
            "off",
            model,
            score,
        )
        assert (
            plan["model_hash"] == "abc"
            and plan["config_hash"] == cfg.plan_hash
        )
        assert len(plan["prediction"]["mean"]) == cfg.steps + 1
        assert len(plan["prediction"]["safe_probability"]) == cfg.steps + 1


def recorded(cfg, path, outcome="completed"):
    """A trial whose state follows `path` [M, 2] at stream_hz."""
    t = np.arange(len(path)) / cfg.stream_hz
    zeros = np.zeros(len(path))
    mission = np.arange(cfg.steps * cfg.repeats)
    return {
        "meta": {
            "outcome": outcome,
            "mission_t0": 0.0,
            "steps": cfg.steps,
            "dt": cfg.dt,
            "stream_hz": cfg.stream_hz,
            "altitude": cfg.altitude,
            "altitude_tolerance": cfg.altitude_tolerance,
        },
        "state": {
            "t_host": t,
            "device_ms": 1000 * t,
            "x": path[:, 0],
            "y": path[:, 1],
            "z": zeros + cfg.altitude,
            "vx": zeros,
            "vy": zeros,
        },
        "commands": {
            "step": mission // cfg.repeats,
            "t_sent": mission / cfg.stream_hz,
            "vx": 0.1 + 0 * mission,
            "vy": 0 * mission,
        },
    }


def test_evaluate_scores_each_requirement(cfg):
    task = Task(cfg)
    good = nominal_path(cfg, cfg.start, reference_velocities(cfg))
    result = analysis.evaluate(recorded(cfg, good), task)
    assert result["success"] and result["min_clearance"] > 0
    assert result["endpoint_error"] == pytest.approx(0, abs=1e-6)
    assert result["duration"] == pytest.approx(cfg.duration)

    bad = nominal_path(cfg, (0.65, -2.0), np.tile([0.0, 0.1], (cfg.steps, 1)))
    result = analysis.evaluate(recorded(cfg, bad), task)
    assert not result["success"] and not result["requirements"]["obs_1"]
    assert not result["requirements"]["goal"] and result["min_clearance"] < 0

    excluded = recorded(cfg, good, outcome="aborted")
    excluded["meta"].update(abort_kind="DataFault", abort_reason="state stale")
    assert analysis.evaluate(excluded, task) == {"excluded": "state stale"}
