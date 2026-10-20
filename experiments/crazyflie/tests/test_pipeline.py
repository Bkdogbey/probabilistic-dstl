"""Offline checks: commands, executor, aborts, logs, model fit, evaluation."""

import json
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import analysis
import config
import flight
import numpy as np
import pytest
import run
import tracking_model
import yaml
from flight import (
    STATE_VARS,
    DataFault,
    StateFeed,
    at_start,
    run_flight,
    world_to_body,
)
from geometry import Task, check_feasible, nominal_path
from logger import TrialLog, aligned_states, load_trial
from planner import (
    make_plan,
    plan_mean,
    plan_pdstl,
)
from simulator import SimDrone, SimLogConfig
from tracking_model import TrackingModel, response_matrices
from trajectory import import_plan, load_plan, reference_velocities


def test_world_to_body():
    assert np.allclose(world_to_body((0.3, 0.1), 0.0), (0.3, 0.1))
    # Facing +y: world +x is the body's right (-y).
    assert np.allclose(world_to_body((0.3, 0.0), 90.0), (0.0, -0.3))


def test_frozen_reference_is_unchanged_and_ignores_optimizer_tuning(cfg):
    original = json.loads(Path(cfg.reference).read_text())
    saved = load_plan(cfg.reference, cfg)
    assert saved["velocities"] == original["velocities"]
    assert (saved["start"], saved["dt"], saved["altitude"]) == (
        original["start"],
        original["dt"],
        original["altitude"],
    )
    tuned = replace(
        cfg,
        planner={**cfg.planner, "max_iters": 3},
        mean_model={**cfg.mean_model, "beta": 1.0},
    )
    assert (
        load_plan(cfg.reference, tuned)["velocities"] == original["velocities"]
    )
    moved = replace(
        cfg,
        obstacles=(
            replace(cfg.obstacles[0], x=(0.0, 0.1)),
            *cfg.obstacles[1:],
        ),
    )
    with pytest.raises(ValueError, match="offline import"):
        load_plan(cfg.reference, moved)


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


def test_dry_flight_logs_aligned_commands_then_lands_in_place(cfg, tmp_path):
    meta, trial, drone = fly_sim(cfg, tmp_path)
    assert meta["outcome"] == "completed"
    assert meta["landing"] == "landed" and "recovery" not in meta
    cmd = trial["commands"]
    assert len(cmd["k"]) == 5 * cfg.repeats and (cmd["step"] >= 0).all()
    assert np.all(np.diff(cmd["t_sent"]) > 0) and cmd["lateness"].max() < 0.05
    states, controls = aligned_states(trial)
    assert states.shape == (6, 4) and np.allclose(controls, [0.1, 0.05])
    assert np.allclose(
        states[0, :2], cfg.start, atol=cfg.flight.start_tolerance
    )
    # Only the initial positioning move; after the mission it lands in place.
    assert drone.calls.count("go_to") == 1
    after = trial["state"]["t_host"] > meta["mission_t0"] + 5 * cfg.dt
    assert trial["state"]["z"][after].max() <= cfg.altitude + 0.01
    assert "start_error" not in meta
    assert drone.calls[-3:] == ["hl_stop", "disarm", "disconnect"]


def test_failed_start_move_lands_and_disconnects(cfg, tmp_path):
    class Failing(SimDrone):
        def go_to(self, *args):
            raise RuntimeError("simulated radio failure")

    meta, _, drone = fly_sim(cfg, tmp_path, Failing((*cfg.start, 0.0)))
    assert meta["outcome"] == "error"
    assert meta["mission_outcome"] == "not_started"
    assert "land" in drone.calls and drone.calls[-1] == "disconnect"


def test_off_start_placement_is_rejected_before_arming(cfg, tmp_path):
    meta, trial, drone = fly_sim(
        cfg, tmp_path, SimDrone((cfg.start[0] + 0.3, cfg.start[1], 0.0))
    )
    assert meta["outcome"] == "aborted"
    assert meta["mission_outcome"] == "not_started"
    assert "place the drone" in meta["abort_reason"]
    assert not len(trial["commands"]["k"])
    assert not any(
        c in drone.calls for c in ("arm", "takeoff", "go_to", "land")
    )
    assert drone.calls[-1] == "disconnect"


def test_stale_state_aborts_lands_and_keeps_logs(cfg, tmp_path):
    meta, trial, drone = fly_sim(
        cfg, tmp_path, SimDrone((*cfg.start, 0.0), stall_after=8)
    )
    assert (meta["outcome"], meta["abort_kind"]) == ("aborted", "DataFault")
    assert meta["abort_phase"] == "mission" and "mission_t0" in meta
    assert meta["landing"] == "landed"
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


def test_one_altitude_spike_is_ignored_but_a_sustained_error_is_not(cfg):
    from flight import mean_altitude

    class Log:
        def state(self, row):
            pass

    now = [0.0]
    feed = StateFeed(Log(), lambda: now[0])

    def run(z_at):
        for k in range(60):
            now[0] = k / cfg.stream_hz
            z = z_at(k)
            feed.on_state(k * 50, dict(zip(STATE_VARS, [1, -2, z, 0, 0, 0])))
        return mean_altitude(feed, cfg, feed.snapshot(1.0))

    tol = cfg.altitude_tolerance
    spike = run(lambda k: 0.6 if k == 59 else cfg.altitude)
    assert abs(spike - cfg.altitude) < tol  # one bad reading: no abort
    high = run(lambda k: cfg.altitude + 2 * tol)
    assert abs(high - cfg.altitude) > tol  # sustained: abort


def test_positioning_waits_for_the_first_fresh_state(cfg):
    """No packet yet must wait for fresh samples, not crash on an empty window."""
    from flight import wait_for_positioning

    class Log:
        def state(self, row):
            pass

    feed = StateFeed(Log())
    packet = dict(zip(STATE_VARS, [1.0, -2.0, 0.0, 0.0, 0.0, 0.0]))

    def packets():  # the first one arrives only after 0.5 s
        for _ in range(40):
            feed.on_state(0, packet)
            time.sleep(0.05)

    threading.Timer(0.5, packets).start()
    home = wait_for_positioning(SimDrone(), feed, cfg, threading.Event())
    assert np.allclose(home[:2], [1.0, -2.0])


def test_start_gate_passes_noisy_hover_but_rejects_offset_and_motion(cfg):
    idle = gate_verdicts(cfg, cfg.start, (0, 0))
    # Noisy recent velocity can delay individual ticks; a stationary drone
    # should still pass within each second instead of timing out.
    for k in range(0, len(idle), cfg.stream_hz):
        assert idle[k : k + cfg.stream_hz].any()
    assert not gate_verdicts(cfg, np.add(cfg.start, (0.12, 0)), (0, 0)).any()
    assert not gate_verdicts(cfg, cfg.start, (0.3, 0)).any()


@pytest.mark.parametrize(
    "recent_x_offset, recent_vx, expected",
    [
        (0.0, 0.0, True),
        (0.08, 0.0, False),
        (0.0, 0.15, False),
        (0.0, -0.15, False),
    ],
)
def test_start_gate_rejects_recent_motion_hidden_by_the_long_mean(
    cfg, recent_x_offset, recent_vx, expected
):
    now = [0.0]
    feed = StateFeed(SimpleNamespace(state=lambda row: None), lambda: now[0])
    for k in range(21):
        now[0] = k / cfg.stream_hz
        x = cfg.start[0] + (recent_x_offset if k >= 15 else 0.0)
        vx = recent_vx if k >= 15 else 0.0
        feed.on_state(
            k * 50,
            dict(
                zip(STATE_VARS, [x, cfg.start[1], cfg.altitude, vx, 0.0, 0.0])
            ),
        )
    assert bool(at_start(feed, cfg.start, cfg)) is expected


@pytest.mark.parametrize(
    "heights, expected",
    [
        ([0.05] * 21, False),  # formerly passed with the wide mission band
        ([0.25] * 21, True),
        ([0.32] * 16 + [0.04] * 5, False),  # full mean hides recent descent
        ([0.22] * 16 + [0.35] * 5, False),  # full mean hides recent climb
    ],
)
def test_start_gate_requires_both_height_windows_at_the_target(
    cfg, heights, expected
):
    now = [0.0]
    feed = StateFeed(SimpleNamespace(state=lambda row: None), lambda: now[0])
    for k, z in enumerate(heights):
        now[0] = k / cfg.stream_hz
        feed.on_state(
            k * 50, dict(zip(STATE_VARS, [*cfg.start, z, 0.0, 0.0, 0.0]))
        )
    assert bool(at_start(feed, cfg.start, cfg)) is expected


@pytest.mark.parametrize(
    "check_delay, send_delay, commands_sent, reason",
    [
        (0.0, 0.2, 1, "command 0 send returned 0.200 s late"),
        (0.1, 0.0, 0, "command 0 late by 0.100 s before send"),
    ],
)
def test_timing_stall_is_reported_on_the_affected_command(
    cfg, tmp_path, monkeypatch, check_delay, send_delay, commands_sent, reason
):
    now = [2.0]
    monkeypatch.setattr(
        flight, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    state = np.array([*cfg.start, cfg.altitude, 0.0, 0.0, 0.0])
    feed = SimpleNamespace(
        snapshot=lambda stale: state,
        window=lambda seconds, stale: np.tile(state, (5, 1)),
    )

    def check(*args):
        now[0] += check_delay
        return state

    sent = []

    def send(*args):
        sent.append(args)
        now[0] += send_delay

    monkeypatch.setattr(flight, "_check", check)
    drone = SimpleNamespace(
        cf=SimpleNamespace(commander=SimpleNamespace(send_hover_setpoint=send))
    )
    log = TrialLog(
        {"condition": "reference", "purpose": "pilot"}, root=tmp_path
    )
    # A one-command mission also exercises a blocked final send: it must
    # never be marked complete merely because there is no next iteration.
    with pytest.raises(DataFault, match=reason):
        flight.send(drone, feed, cfg, log, threading.Event(), [[0.1, 0.05]])
    assert len(sent) == commands_sent
    assert "mission_t_end" not in log.meta
    log.finish("aborted")
    cmd = load_trial(log.dir)["commands"]
    assert len(cmd["k"]) == commands_sent
    if commands_sent:
        assert cmd["lateness"][0] == pytest.approx(0.2)
        assert cmd["t_sent"][0] == pytest.approx(2.2)


@pytest.mark.parametrize(
    "delayed_k, delay, failure_k",
    [
        (3, 0.08, None),  # refresh recovers before the next step deadline
        (5, 0.07, None),
        (4, 0.08, 4),  # new trajectory step is too late
        (3, 0.12, 4),  # refresh also delays the next step beyond its deadline
        (3, 0.21, 3),  # refresh exceeds one trajectory interval
        (11, 0.08, 11),  # final refresh cannot extend the mission
    ],
)
def test_refresh_jitter_recovers_without_moving_trajectory_deadlines(
    cfg, tmp_path, monkeypatch, delayed_k, delay, failure_k
):
    now, sent = [2.0], []
    monkeypatch.setattr(
        flight, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    state = np.array([*cfg.start, cfg.altitude, 0.0, 0.0, 0.0])
    feed = SimpleNamespace(
        snapshot=lambda stale: state,
        window=lambda seconds, stale: np.tile(state, (5, 1)),
    )

    class Stop:
        def wait(self, seconds):
            now[0] += seconds
            return False

    def send(*args):
        k = len(sent)
        sent.append(args[:2])
        if k == delayed_k:
            now[0] += delay

    drone = SimpleNamespace(
        is_connected=True,
        cf=SimpleNamespace(
            commander=SimpleNamespace(send_hover_setpoint=send)
        ),
    )
    log = TrialLog(
        {"condition": "reference", "purpose": "pilot"}, root=tmp_path
    )
    velocities = np.repeat([[0.01, 0], [0, 0.01], [0, 0]], cfg.repeats, axis=0)
    if failure_k is None:
        flight.send(drone, feed, cfg, log, Stop(), velocities)
        assert log.meta["mission_t_end"] - log.meta[
            "mission_t0"
        ] == pytest.approx(len(velocities) / cfg.stream_hz)
        assert len(sent) == len(velocities)
        outcome = "completed"
    else:
        with pytest.raises(DataFault, match=f"command {failure_k} "):
            flight.send(drone, feed, cfg, log, Stop(), velocities)
        assert "mission_t_end" not in log.meta
        outcome = "aborted"
    assert np.allclose(sent, velocities[: len(sent)])
    log.finish(outcome)
    cmd = load_trial(log.dir)["commands"]
    assert cmd["lateness"][delayed_k] == pytest.approx(delay)
    if failure_k is None:
        new_step = cmd["k"] % cfg.repeats == 0
        assert cmd["lateness"][new_step].max() <= cfg.flight.max_lateness_s


def test_blocked_send_aborts_then_lands_and_disconnects(cfg, tmp_path):
    class Blocked(SimDrone):
        def send_hover_setpoint(self, *args):
            super().send_hover_setpoint(*args)
            time.sleep(0.15)

    meta, trial, drone = fly_sim(cfg, tmp_path, Blocked((*cfg.start, 0.0)))
    assert meta["outcome"] == meta["mission_outcome"] == "aborted"
    assert meta["abort_kind"] == "DataFault"
    assert "command 0 send returned" in meta["abort_reason"]
    assert len(trial["commands"]["k"]) == 1
    assert trial["commands"]["lateness"][0] >= 0.15
    assert meta["landing"] == "landed"
    assert "land" in drone.calls and drone.calls[-1] == "disconnect"


# Tracking model


TAU, GAIN, DRIFT = np.array([0.4, 0.55]), [0.95, 1.05], np.array([-0.02, 0.01])
Q_V = 4e-4  # per-step velocity process variance
MEASUREMENT_SD = [
    0.01,
    0.01,
    0.15,
    0.15,
]  # Synthetic measurements with larger velocity noise than position noise.


def synthetic_flights(cfg, count, rng, steps=60):
    """Synthetic flights with known response parameters and measurement noise."""
    A, B = response_matrices(cfg.dt, TAU, GAIN)
    a = np.exp(-cfg.dt / TAU)
    lag = TAU * (1 - a)
    c = np.concatenate([(cfg.dt - lag) * DRIFT, (1 - a) * DRIFT])
    flights = []
    for i in range(count):
        s = [np.array([*rng.normal(cfg.start, 0.02), 0.0, 0.0])]
        u = rng.uniform(-0.3, 0.3, (steps // 5, 2)).repeat(5, axis=0)
        for k in range(steps):
            noise = rng.normal(0, np.sqrt([1e-6, 1e-6, Q_V, Q_V]))
            s.append(A @ s[-1] + B @ u[k] + c + noise)
        measured = np.array(s) + rng.normal(0, MEASUREMENT_SD, (steps + 1, 4))
        meta = {
            "purpose": "identification",
            "outcome": "completed",
            "fan": 0,
            "trial_id": f"t{i:02d}",
        }
        flights.append({"meta": meta, "data": (measured, u)})
    return flights


def test_fit_recovers_model_from_noisy_velocity_and_intervals_hold(
    cfg, monkeypatch
):
    monkeypatch.setattr(tracking_model, "aligned_states", lambda t: t["data"])
    cfg = replace(cfg, model={**cfg.model, "bootstrap": 50})
    report = tracking_model.fit(
        cfg, synthetic_flights(cfg, 60, np.random.default_rng(0))
    )
    assert np.allclose(report["tau"], TAU, rtol=0.2)
    assert np.allclose(report["gain"], GAIN, rtol=0.1)
    fan = report["fans"]["0"]
    assert len(fan["held_out_trials"]) == 20 and not set(
        fan["held_out_trials"]
    ) & set(fan["fit_trials"])
    assert np.allclose(fan["drift"], DRIFT, atol=0.01)
    assert np.allclose(np.diag(fan["Q"])[2:], Q_V, rtol=0.5)
    assert np.allclose(fan["initial_mean"][:2], cfg.start, atol=0.02)
    coverage = report["validation"]["0"]["coverage"]
    for (name, (estimate, *_)), nominal in zip(
        coverage.items(), (0.393, 0.865, 0.95)
    ):
        assert abs(estimate - nominal) < 0.12, name


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
        cfg.a_max,
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
            "fan-0",
            cfg.start,
            controls,
            task,
            0,
            model,
            score,
        )
        assert (
            plan["model_hash"] == "abc"
            and plan["config_hash"] == cfg.plan_hash
        )
        assert len(plan["prediction"]["mean"]) == cfg.steps + 1
        assert len(plan["prediction"]["safe_probability"]) == cfg.steps + 1


def test_planning_cannot_overwrite_the_frozen_reference(cfg):
    before = Path(cfg.reference).read_bytes()
    with pytest.raises(SystemExit, match="reference stays frozen"):
        run.cmd_plan(cfg, SimpleNamespace(output=cfg.reference))
    assert Path(cfg.reference).read_bytes() == before


def test_recorded_plan_is_portable_and_hash_checked(
    cfg, tmp_path, monkeypatch
):
    from trajectory import file_hash

    path = tmp_path / "reference.json"
    path.write_bytes(Path(cfg.reference).read_bytes())
    monkeypatch.setattr(analysis, "PLANS", tmp_path)
    meta = {
        "plan_id": "reference",
        "plan_path": "/missing/reference.json",
        "plan_hash": file_hash(path),
    }
    assert (
        analysis._plan(meta)["velocities"]
        == json.loads(path.read_text())["velocities"]
    )
    assert analysis._plan({**meta, "plan_hash": "different"}) is None


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


def test_real_flight_forwards_irobot_packets_without_a_second_state_block(
    cfg, monkeypatch
):
    calls = []

    class Base:
        def __init__(self, hw_config):
            self.config, self.is_connected = hw_config, False
            self.cf = SimpleNamespace(
                param=SimpleNamespace(
                    is_updated=True,
                )
            )
            self._connected(hw_config.uri)

        def _connected(self, uri):
            pytest.fail("irobot's auto-arming callback must not run")

        def _setup_lighthouse_logging(self):
            assert not self.is_connected
            calls.append("state logging")

        def _state_callback(self, timestamp, data, log_config):
            self.current_x = data["stateEstimate.x"]

    monkeypatch.setitem(
        sys.modules,
        "irobot",
        SimpleNamespace(
            CrazyflieBase=Base,
            CrazyflieConfig=lambda **kw: SimpleNamespace(**kw),
        ),
    )
    rows, blocks = [], []
    log = SimpleNamespace(
        state=rows.append,
        battery=lambda v: None,
        event=lambda e: None,
        meta={},
    )
    feed = StateFeed(log, clock=lambda: 2.0)
    drone = flight.connect(cfg, feed)
    assert calls == ["state logging"] and drone.is_connected
    packet = dict(zip(STATE_VARS, [1.0, -2.0, 0.25, 0, 0, 90]))
    drone._state_callback(1234, packet, None)
    assert rows == [[2.0, 1234, 1.0, -2.0, 0.25, 0, 0, 90]]
    assert drone.current_x == 1.0 and drone.config.log_rate_ms == 50
    cf = SimpleNamespace(log=SimpleNamespace(add_config=blocks.append))
    flight.start_logging(
        cf,
        cfg,
        feed,
        log,
        SimLogConfig,
        state_logged=drone.state_logged,
    )
    assert [b.name for b in blocks] == ["battery"]


@pytest.mark.parametrize("outcome", ["ready", "timeout", "interrupted"])
def test_connection_finishes_parameter_download_before_flight(
    cfg, monkeypatch, outcome
):
    now, calls = [0.0], []

    class Parameters:
        @property
        def is_updated(self):
            return outcome == "ready" and now[0] >= 0.1

    class Base:
        def __init__(self, config):
            self.is_connected = True
            self.cf = SimpleNamespace(param=Parameters())

        def disarm(self):
            calls.append("disarm")

        def disconnect(self):
            calls.append("disconnect")
            self.is_connected = False

    def sleep(seconds):
        if outcome == "interrupted":
            raise KeyboardInterrupt
        now[0] += seconds

    monkeypatch.setitem(
        sys.modules,
        "irobot",
        SimpleNamespace(CrazyflieBase=Base, CrazyflieConfig=lambda **kw: kw),
    )
    monkeypatch.setattr(
        flight, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=sleep)
    )
    cfg = replace(cfg, flight=replace(cfg.flight, connect_timeout=0.2))
    log = SimpleNamespace(meta={})
    feed = StateFeed(log)
    if outcome == "ready":
        flight.connect(cfg, feed)
        assert now[0] == pytest.approx(0.1)
        assert calls == []  # arming is deferred until after preflight
    else:
        exception = (
            KeyboardInterrupt if outcome == "interrupted" else DataFault
        )
        with pytest.raises(exception):
            flight.connect(cfg, feed)
        assert calls == ["disconnect"]
        assert not log.meta


def test_ros_component_executes_once_without_a_mission_thread(
    cfg, tmp_path, monkeypatch
):
    calls = []

    class Component:
        def __init__(self, **kwargs):
            pass

        def custom_on_shutdown(self):
            pass

    class Launcher:
        def add_pkg(self, *, components, activate_all_components_on_start):
            self.component = components[0]

        def bringup(self):
            self.component._execute_once()
            self.component._execute_once()  # never fly a second attempt
            self.component.custom_on_shutdown()
            assert self.component.stop.is_set()

    monkeypatch.setitem(
        sys.modules, "ros_sugar", SimpleNamespace(Launcher=Launcher)
    )
    monkeypatch.setitem(
        sys.modules, "ros_sugar.core", SimpleNamespace(BaseComponent=Component)
    )
    monkeypatch.setattr(
        flight,
        "run_flight",
        lambda *args: calls.append(args) or {"outcome": "completed"},
    )

    def no_worker(*args, **kwargs):
        pytest.fail("flight must execute on ROS Sugar's executor")

    monkeypatch.setattr(flight.threading, "Thread", no_worker)
    assert flight.launch(cfg, {}, None)["outcome"] == "completed"
    assert len(calls) == 1


def test_landing_error_is_separate_from_completed_mission(cfg, tmp_path):
    class Failing(SimDrone):
        def land(self, *args):
            raise RuntimeError("landing command failed")

    meta, trial, drone = fly_sim(cfg, tmp_path, Failing((*cfg.start, 0.0)))
    assert meta["outcome"] == meta["mission_outcome"] == "completed"
    assert (
        meta["landing"] == "failed"
        and "landing command failed" in meta["landing_error"]
    )
    assert meta["mission_t_end"] >= trial["commands"]["t_sent"][-1]
    assert drone.calls[-1] == "disconnect" and "stop_setpoint" in drone.calls


def test_explicit_selection_never_chooses_newest_or_rebuilds(cfg, tmp_path):
    args = SimpleNamespace(plan=None, fan=0)
    selected, plan = run.select_plan(cfg, args)
    assert str(selected) == cfg.reference
    legacy = json.loads(Path(cfg.reference).read_text())
    source = tmp_path / "supplied.json"
    source.write_text(json.dumps(legacy))
    imported = import_plan(source, cfg, tmp_path / "frozen.json")
    args.plan = str(imported)
    selected, plan = run.select_plan(cfg, args)
    assert selected == imported and plan["velocities"] == legacy["velocities"]
    plan["fan"] = 6
    imported.write_text(json.dumps(plan))
    with pytest.raises(SystemExit, match="for fan"):
        run.select_plan(cfg, args)


def test_abort_boundary_excludes_landing_from_mission_metrics(cfg):
    path = nominal_path(cfg, cfg.start, reference_velocities(cfg))
    trial = recorded(cfg, path, outcome="aborted")
    trial["meta"].update(abort_kind="TaskViolation", mission_t_end=1.0)
    after = trial["state"]["t_host"] > 1.0
    trial["state"]["x"][after] = 100.0
    trial["state"]["z"][after] = 0.0
    result = analysis.evaluate(trial, Task(cfg))
    assert not result["success"] and np.isnan(result["endpoint_error"])
    assert result["min_clearance"] > 0 and result["requirements"]["altitude"]
    assert result["duration"] == 1.0


def test_altitude_scoring_uses_the_same_average_as_execution(cfg):
    path = nominal_path(cfg, cfg.start, reference_velocities(cfg))
    trial = recorded(cfg, path)
    trial["meta"]["altitude_average_s"] = flight.ALTITUDE_AVG_S
    trial["state"]["z"][40] = 0.6
    assert analysis.evaluate(trial, Task(cfg))["success"]
    trial["state"]["z"][40:60] = 0.6
    assert not analysis.evaluate(trial, Task(cfg))["requirements"]["altitude"]


def test_import_can_initialize_a_missing_reference_without_changing_commands(
    cfg, tmp_path
):
    source = tmp_path / "supplied.json"
    original = json.loads(Path(cfg.reference).read_text())
    source.write_text(json.dumps(original))
    before = source.read_bytes()
    output = tmp_path / "frozen.json"
    raw = asdict(cfg)
    raw["reference"] = str(output)
    raw["goal"].pop("name")
    settings = tmp_path / "config.yml"
    settings.write_text(yaml.safe_dump(raw))
    loaded = config.load(settings, trajectory_source=source)
    imported = import_plan(source, loaded)
    assert imported == output and source.read_bytes() == before
    assert load_plan(output, loaded)["velocities"] == original["velocities"]
    assert config.load(settings).duration == 18.0


def test_unmeasured_identification_is_recorded_but_not_fitted(cfg):
    trials = synthetic_flights(cfg, 6, np.random.default_rng(2))
    for trial in trials:
        trial["meta"]["data_readiness"] = "config.yml has measured: false"
    assert tracking_model.split(trials, 3) == ({}, {})


@pytest.mark.parametrize("setting", [0, 6, 12, 18])
def test_numeric_fan_setting_is_recorded_and_loaded_from_model_json(
    cfg, tmp_path, setting
):
    from trajectory import read_plan

    measured = replace(
        cfg, measured=True, fan=setting, fan_position=None, fan_direction=None
    )
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
    assert "fan_setting" not in log.meta
    report = {
        "id": "numeric-fan-model",
        "hash": "test",
        "dt": cfg.dt,
        "tau": [0.4, 0.4],
        "gain": [1, 1],
        "fans": {
            str(setting): {
                "c": [setting / 1000] * 4,
                "Q": (1e-4 * np.eye(4)).tolist(),
                "initial_mean": [*cfg.start, 0, 0],
                "initial_cov": (1e-4 * np.eye(4)).tolist(),
            }
        },
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(report))
    loaded = tracking_model.load_model(path, setting, cfg)
    assert np.allclose(loaded.c.numpy(), setting / 1000)
