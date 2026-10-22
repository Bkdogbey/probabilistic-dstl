"""Executor checks: connection, upload, start gate, mission monitor, aborts."""

import sys
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import flight
import numpy as np
import pytest
from flight import STATE_VARS, DataFault, StateFeed, at_start, run_flight
from logger import TrialLog, aligned_states, device_times, load_trial
from simulator import SimDrone, SimLogConfig


def fly_sim(cfg, tmp_path, short_plan, drone=None, stop=None, connect_fn=None):
    path, plan = short_plan
    cfg = cfg.for_plan(plan)
    log = TrialLog.for_flight(
        cfg, plan, path, purpose="pilot", fan=0, root=tmp_path / "logs"
    )
    drone = drone or SimDrone((*cfg.start, 0.0))
    meta = run_flight(cfg, plan, log, stop, connect_fn or (lambda _: drone))
    return meta, load_trial(log.dir), drone


def test_dry_flight_uploads_before_arming_flies_once_and_lands_in_place(
    cfg, tmp_path, short_plan
):
    meta, trial, drone = fly_sim(cfg, tmp_path, short_plan)
    assert meta["outcome"] == "completed" and meta["landing"] == "landed"
    calls = drone.calls
    assert (
        calls.index("param stabilizer.controller=1")
        < calls.index("write_data_sync")
        < calls.index("define_trajectory")
        < calls.index("arm")
    )
    assert drone.defined == (flight.TRAJ_ID, 0, 2)
    assert meta["pieces"] == 2 and meta["upload_bytes"] == 264
    assert drone.starts == [
        {
            "trajectory_id": flight.TRAJ_ID,
            "time_scale": 1.0,
            "relative_position": False,
            "relative_yaw": False,
        }
    ]
    mission = meta["mission_t_end"] - meta["mission_t0"]
    assert mission == pytest.approx(1.2 + flight.END_HOLD_S, abs=0.1)
    assert (trial["dir"] / "target.csv").exists() and len(
        trial["target"]["tx"]
    )

    # The fitted playback start is the simulator's, within one 10 ms tick.
    state = trial["state"]
    offset = (
        device_times(state, trial["target"])[0][0]
        - state["device_ms"][0] / 1000
    )
    true_start = (123_456 + 10 * drone.start_tick) / 1000 + offset
    assert meta["alignment"] == "ctrltarget" and meta["identifiable"]
    assert abs(meta["t_play"] - true_start) <= 0.011
    assert meta["alignment_max"] < 0.001
    _, _, e = aligned_states(trial)
    assert len(e) == 7 and np.abs(e).max() < 0.05

    # Only the move to the start; afterwards it lands where it is.
    assert calls.count("go_to") == 1
    assert calls[-3:] == ["hl_stop", "disarm", "disconnect"]


def test_failed_upload_aborts_before_arming(cfg, tmp_path, short_plan):
    meta, _, drone = fly_sim(
        cfg,
        tmp_path,
        short_plan,
        SimDrone((*cfg.start, 0.0), fail_upload=True),
    )
    assert (meta["outcome"], meta["abort_kind"]) == ("aborted", "DataFault")
    assert meta["abort_phase"] == "upload"
    assert meta["mission_outcome"] == "not_started"
    assert "arm" not in drone.calls and "takeoff" not in drone.calls
    assert drone.calls[-1] == "disconnect"


def test_too_many_pieces_are_refused_before_connecting(
    cfg, tmp_path, short_plan
):
    path, plan = short_plan
    plan = {**plan, "coefficients": plan["coefficients"] * 16}  # 32 pieces

    def connect(_):
        pytest.fail("must not connect")

    meta, _, _ = fly_sim(cfg, tmp_path, (path, plan), connect_fn=connect)
    assert (meta["outcome"], meta["abort_kind"]) == ("aborted", "DataFault")
    assert (
        meta["abort_phase"] == "connect"
        and "32 pieces" in meta["abort_reason"]
    )


def test_stale_state_mid_trajectory_aborts_and_lands(
    cfg, tmp_path, short_plan
):
    meta, _, drone = fly_sim(
        cfg, tmp_path, short_plan, SimDrone((*cfg.start, 0.0), stall_after=0.3)
    )
    assert (meta["outcome"], meta["abort_kind"]) == ("aborted", "DataFault")
    assert meta["abort_phase"] == "mission" and meta["landing"] == "landed"
    assert "land" in drone.calls and drone.calls[-1] == "disconnect"


def test_sustained_altitude_error_is_a_task_violation(
    cfg, tmp_path, short_plan
):
    class Sinking(SimDrone):
        def _step(self):
            super()._step()
            if self.playback_tick is not None:
                self.pos[2] -= 0.3

    meta, _, drone = fly_sim(
        cfg, tmp_path, short_plan, Sinking((*cfg.start, 0.0))
    )
    assert (meta["outcome"], meta["abort_kind"]) == (
        "aborted",
        "TaskViolation",
    )
    assert "altitude" in meta["abort_reason"] and meta["landing"] == "landed"
    assert "land" in drone.calls


def test_stop_request_interrupts_and_lands(cfg, tmp_path, short_plan):
    stop = threading.Event()

    class Stopping(SimDrone):
        def start_trajectory(self, *args, **kwargs):
            super().start_trajectory(*args, **kwargs)
            stop.set()

    meta, _, drone = fly_sim(
        cfg, tmp_path, short_plan, Stopping((*cfg.start, 0.0)), stop
    )
    assert meta["outcome"] == "interrupted" and "land" in drone.calls


def test_failed_start_move_lands_and_disconnects(cfg, tmp_path, short_plan):
    class Failing(SimDrone):
        def go_to(self, *args):
            raise RuntimeError("simulated radio failure")

    meta, _, drone = fly_sim(
        cfg, tmp_path, short_plan, Failing((*cfg.start, 0.0))
    )
    assert meta["outcome"] == "error"
    assert meta["mission_outcome"] == "not_started"
    assert "land" in drone.calls and drone.calls[-1] == "disconnect"


def test_drone_away_from_the_start_flies_there_before_the_mission(
    cfg, tmp_path, short_plan
):
    home = (cfg.start[0] - 0.5, cfg.start[1] + 0.3, 0.0)
    meta, _, drone = fly_sim(cfg, tmp_path, short_plan, SimDrone(home))
    assert meta["outcome"] == "completed"
    assert np.allclose(meta["home"][:2], home[:2], atol=0.01)
    calls = drone.calls
    assert calls.index("takeoff") < calls.index("go_to")
    assert calls.index("go_to") < calls.index("start_trajectory")
    assert np.allclose(meta["start_state"][:2], cfg.start, atol=0.05)


def test_connect_failure_still_records_the_attempt(cfg, tmp_path, short_plan):
    def fail(_):
        raise DataFault("could not connect")

    meta, trial, _ = fly_sim(cfg, tmp_path, short_plan, connect_fn=fail)
    assert meta["outcome"] == "aborted"
    assert trial["meta"]["abort_phase"] == "connect"


def test_landing_error_is_separate_from_completed_mission(
    cfg, tmp_path, short_plan
):
    class Failing(SimDrone):
        def land(self, *args):
            raise RuntimeError("landing command failed")

    meta, _, drone = fly_sim(
        cfg, tmp_path, short_plan, Failing((*cfg.start, 0.0))
    )
    assert meta["outcome"] == meta["mission_outcome"] == "completed"
    assert meta["landing"] == "failed"
    assert "landing command failed" in meta["landing_error"]
    assert meta["mission_t_end"] > meta["mission_t0"]
    assert drone.calls[-1] == "disconnect" and "stop_setpoint" in drone.calls


# State feed, start gate and altitude


def quiet_feed(now):
    return StateFeed(SimpleNamespace(state=lambda row: None), lambda: now[0])


def gate_verdicts(cfg, position, velocity, seconds=8, seed=0):
    """at_start on noisy hover data: 0.02 m position and 0.15 m/s velocity noise."""
    rng, now = np.random.default_rng(seed), [0.0]
    feed = quiet_feed(now)
    out = []
    for k in range(seconds * cfg.log_hz):
        now[0] = k / cfg.log_hz
        x, y = np.add(position, rng.normal(0, 0.02, 2))
        vx, vy = np.add(velocity, rng.normal(0, 0.15, 2))
        values = [x, y, cfg.altitude, vx, vy, 0.0]
        feed.on_state(k * 50, dict(zip(STATE_VARS, values)))
        if now[0] > 2:
            out.append(at_start(feed, np.array(cfg.start), cfg))
    return np.array(out)


def test_one_altitude_spike_is_ignored_but_a_sustained_error_is_not(cfg):
    now = [0.0]
    feed = quiet_feed(now)

    def run(z_at):
        for k in range(60):
            now[0] = k / cfg.log_hz
            z = z_at(k)
            feed.on_state(k * 50, dict(zip(STATE_VARS, [1, -2, z, 0, 0, 0])))
        return flight.mean_altitude(feed, cfg, feed.snapshot(1.0))

    tol = cfg.altitude_tolerance
    spike = run(lambda k: cfg.altitude + (0.35 if k == 59 else 0.0))
    assert abs(spike - cfg.altitude) < tol  # one bad reading: no abort
    high = run(lambda k: cfg.altitude + 2 * tol)
    assert abs(high - cfg.altitude) > tol  # sustained: abort


def test_positioning_waits_for_the_first_fresh_state(cfg):
    """No packet yet must wait for fresh samples, not crash on an empty window."""
    feed = StateFeed(SimpleNamespace(state=lambda row: None))
    packet = dict(zip(STATE_VARS, [1.0, -2.0, 0.0, 0.0, 0.0, 0.0]))

    def packets():  # the first one arrives only after 0.5 s
        for _ in range(40):
            feed.on_state(0, packet)
            time.sleep(0.05)

    threading.Timer(0.5, packets).start()
    drone = SimDrone()
    home = flight.wait_for_positioning(drone, feed, cfg, threading.Event())
    drone.disconnect()
    assert np.allclose(home[:2], [1.0, -2.0])


def test_start_gate_passes_noisy_hover_but_rejects_offset_and_motion(cfg):
    idle = gate_verdicts(cfg, cfg.start, (0, 0))
    # Noisy recent velocity can delay individual ticks; a stationary drone
    # should still pass within each second instead of timing out.
    for k in range(0, len(idle), cfg.log_hz):
        assert idle[k : k + cfg.log_hz].any()
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
    feed = quiet_feed(now)
    for k in range(21):
        now[0] = k / cfg.log_hz
        x = cfg.start[0] + (recent_x_offset if k >= 15 else 0.0)
        vx = recent_vx if k >= 15 else 0.0
        values = [x, cfg.start[1], cfg.altitude, vx, 0.0, 0.0]
        feed.on_state(k * 50, dict(zip(STATE_VARS, values)))
    assert bool(at_start(feed, cfg.start, cfg)) is expected


@pytest.mark.parametrize(
    "offsets, expected",
    [
        ([-0.20] * 21, False),
        ([0.0] * 21, True),
        ([0.07] * 16 + [-0.21] * 5, False),  # full mean hides recent descent
        ([-0.03] * 16 + [0.15] * 5, False),  # full mean hides recent climb
        (list(np.linspace(-0.15, 0.15, 21)), False),  # climbing through
        (
            list(np.linspace(-0.08, 0.08, 21)),
            False,
        ),  # climbing inside the band
        ([0.08] * 21, True),  # level, high in the band
    ],
)
def test_start_gate_requires_both_height_windows_at_the_target(
    cfg, offsets, expected
):
    now = [0.0]
    feed = quiet_feed(now)
    for k, offset in enumerate(offsets):
        z = cfg.altitude + offset
        now[0] = k / cfg.log_hz
        feed.on_state(
            k * 50, dict(zip(STATE_VARS, [*cfg.start, z, 0.0, 0.0, 0.0]))
        )
    assert bool(at_start(feed, cfg.start, cfg)) is expected


# Connection and ROS


def test_real_flight_forwards_irobot_packets_and_logs_the_setpoint(
    cfg, monkeypatch
):
    calls = []

    class Base:
        def __init__(self, hw_config):
            self.config, self.is_connected = hw_config, False
            self.cf = SimpleNamespace(param=SimpleNamespace(is_updated=True))
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
    rows, targets, blocks = [], [], []
    log = SimpleNamespace(
        state=rows.append,
        target=targets.append,
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
        cf, cfg, feed, log, SimLogConfig, state_logged=drone.state_logged
    )
    assert [b.name for b in blocks] == ["battery", "target"]
    target = blocks[1]
    assert target.variables == flight.TARGET_VARS
    target.data_received_cb.call(
        1300, dict(zip(flight.TARGET_VARS, [1.0, -2.0, 0.25])), target
    )
    assert targets == [[2.0, 1300, 1.0, -2.0, 0.25]]


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
    cfg, monkeypatch
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
