"""One flight attempt: upload, takeoff, start gate, onboard curve, landing.

`drone` is an irobot `CrazyflieBase` (radio link, arming) or a `SimDrone`.
The plan's polynomial pieces are uploaded on the ground, before arming.
The high-level commander takes off, reaches the start, flies the uploaded
curve (`start_trajectory`) and lands in place. The host only monitors the
mission; every safety decision uses a fresh state packet.

Abort kinds: `TaskViolation` (left the workspace or altitude band: a task
failure) and `DataFault` (stale state, lost link, failed upload: not a task
outcome). Either one lands through the high-level commander.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

import numpy as np
from minsnap import MAX_PIECES, PIECE_BYTES

logger = logging.getLogger(__name__)

ALTITUDE_AVG_S = 0.25  # s of state averaged for the altitude limit
MONITOR_S = 0.05  # mission check period
# Monitoring continues this long after the curve ends, while the drone holds
# its final setpoint, so the radio delay before playback starts cannot drop
# the last plan step from the mission data.
END_HOLD_S = 0.25
TRAJ_ID = 1

STATE_VARS = [
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.vx",
    "stateEstimate.vy",
    "stateEstimate.yaw",
]
TARGET_VARS = ["ctrltarget.x", "ctrltarget.y", "ctrltarget.z"]


class TaskViolation(RuntimeError):
    """The drone left the geofence or the altitude band."""


class DataFault(RuntimeError):
    """Stale state, lost link, failed upload or a failed preflight check."""


class Interrupted(Exception):
    """Stop requested (Ctrl-C or ROS shutdown)."""


class StateFeed:
    """Recent state packets, forwarded to the log; [x, y, z, vx, vy, yaw]."""

    def __init__(self, log, clock=time.monotonic):
        self.log, self.clock = log, clock
        self._recent = deque(maxlen=400)  # (host time, values)
        self._lock = threading.Lock()

    def on_state(self, device_ms, data):
        t = self.clock()
        values = [data[name] for name in STATE_VARS]
        with self._lock:
            self._recent.append((t, np.array(values)))
        self.log.state([t, device_ms, *values])

    def snapshot(self, stale_s):
        with self._lock:
            latest = self._recent[-1] if self._recent else None
        if latest is None:
            raise DataFault("no state received")
        age = self.clock() - latest[0]
        if age > stale_s:
            raise DataFault(f"state {age:.3f} s old (limit {stale_s} s)")
        return latest[1]

    def window(self, seconds, stale_s):
        """States of the last `seconds` [n, 6]; None if there are none yet."""
        self.snapshot(stale_s)  # raises if stale
        now = self.clock()
        with self._lock:
            states = [v for t, v in self._recent if now - t <= seconds]
            oldest = self._recent[0][0]
        full = bool(states) and now - oldest >= seconds
        return np.array(states) if full else None


def start_logging(cf, cfg, feed, log, make_config=None, *, state_logged=False):
    """Battery and setpoint logging; the simulator also needs a state block."""
    if make_config is None:
        from cflib.crazyflie.log import LogConfig as make_config
    battery = make_config(name="battery", period_in_ms=1000)
    battery.add_variable("pm.vbat", "float")
    battery.data_received_cb.add_callback(
        lambda ts, data, _: log.battery(data["pm.vbat"])
    )
    # The setpoint the onboard controller is tracking, for time alignment.
    target = make_config(name="target", period_in_ms=1000 // cfg.log_hz)
    for name in TARGET_VARS:
        target.add_variable(name, "float")
    target.data_received_cb.add_callback(
        lambda ts, data, _: log.target(
            [feed.clock(), ts, *(data[name] for name in TARGET_VARS)]
        )
    )
    blocks = [battery, target]
    if not state_logged:
        state = make_config(name="state", period_in_ms=1000 // cfg.log_hz)
        for name in STATE_VARS:
            state.add_variable(name, "float")
        state.data_received_cb.add_callback(
            lambda ts, data, _: feed.on_state(ts, data)
        )
        blocks.insert(0, state)
    for conf in blocks:
        cf.log.add_config(conf)
        conf.error_cb.add_callback(
            lambda conf, msg: log.event(f"log error {msg}")
        )
    for conf in blocks:
        conf.start()
    return blocks


def connect(cfg, feed):
    """Use irobot's connection and timestamped state callback directly."""
    from irobot import CrazyflieBase, CrazyflieConfig

    class RecordedCrazyflie(CrazyflieBase):
        state_logged = True

        def _connected(self, link_uri):
            # irobot's default callback arms immediately and sleeps on the
            # receive thread. Here arming belongs to fly(), after preflight.
            self._setup_lighthouse_logging()
            self.is_connected = True
            logger.info("Connected to %s", link_uri)

        def _state_callback(self, timestamp, data, log_config):
            super()._state_callback(timestamp, data, log_config)
            feed.on_state(timestamp, data)

    drone = RecordedCrazyflie(
        CrazyflieConfig(
            uri=cfg.radio_uri,
            connect_timeout=cfg.flight.connect_timeout,
            log_rate_ms=1000 // cfg.log_hz,
        )
    )
    if not drone.is_connected:
        drone.disconnect()
        raise DataFault(f"could not connect to {cfg.radio_uri}")
    # `connected` precedes parameter download completion. Let that traffic
    # finish before preflight and the trajectory upload.
    try:
        deadline = time.monotonic() + cfg.flight.connect_timeout
        while not drone.cf.param.is_updated:
            if not drone.is_connected or time.monotonic() >= deadline:
                raise DataFault(
                    "parameter download did not finish before preflight"
                )
            time.sleep(0.05)
    except (Exception, KeyboardInterrupt):
        drone.disconnect()
        raise
    return drone


def _wait(stop, seconds):
    if stop.wait(max(seconds, 0.0)):
        raise Interrupted("stop requested")


def _check(drone, feed, cfg):
    """Fresh state, or abort on a lost link or a geofence violation."""
    if not drone.is_connected:
        raise DataFault("radio link lost")
    s = feed.snapshot(cfg.flight.stale_s)
    m, ws = cfg.flight.geofence_margin, cfg.workspace
    if not (
        ws.x[0] - m <= s[0] <= ws.x[1] + m
        and ws.y[0] - m <= s[1] <= ws.y[1] + m
    ):
        raise TaskViolation(f"geofence at ({s[0]:.2f}, {s[1]:.2f})")
    return s


def _monitor(drone, feed, cfg, stop, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        _check(drone, feed, cfg)
        _wait(stop, min(0.05, end - time.monotonic()))


def _go_to(drone, feed, cfg, stop, target, speed):
    """Send a high-level move and monitor it for its planned duration."""
    distance = np.linalg.norm(
        np.asarray(target) - _check(drone, feed, cfg)[:3]
    )
    duration = max(distance / speed, 1.0)
    drone.cf.high_level_commander.go_to(*target, 0.0, duration)
    _monitor(drone, feed, cfg, stop, duration)


def wait_for_positioning(drone, feed, cfg, stop, samples=10):
    """Reset the Kalman filter; return the mean of a stable 1 s window.

    Only x and y must be steady: z reads +-0.05 m on the ground.
    """
    drone.cf.param.set_value("kalman.resetEstimation", "1")
    _wait(stop, 0.1)
    drone.cf.param.set_value("kalman.resetEstimation", "0")
    deadline = time.monotonic() + cfg.flight.position_timeout
    window = []
    while True:
        _wait(stop, 0.1)
        try:
            window = (window + [feed.snapshot(cfg.flight.stale_s)[:3]])[
                -samples:
            ]
        except DataFault:
            window = []
        # `and` short-circuits: an empty window is never reduced.
        if (
            len(window) == samples
            and np.ptp(window, axis=0)[:2].max() <= cfg.flight.position_spread
        ):
            return np.mean(window, axis=0)
        if time.monotonic() > deadline:
            raise DataFault("position estimate did not settle")


def at_start(feed, start, cfg):
    """Is the last `start_window` s, averaged, at the start and at rest?

    The velocity estimate is noisy (0.2-0.3 m/s at rest), so a single sample
    never passes; the window mean does.
    """
    f = cfg.flight
    height_tolerance = min(f.start_altitude_tolerance, cfg.altitude_tolerance)
    # An older average must not hide current motion at the handoff.
    for seconds in (f.start_window, ALTITUDE_AVG_S):
        window = feed.window(seconds, f.stale_s)
        if window is None:
            return False
        mean = window.mean(axis=0)
        if not (
            np.linalg.norm(mean[:2] - start[:2]) <= f.start_tolerance
            and np.linalg.norm(mean[3:5]) <= f.start_speed
            and abs(mean[2] - cfg.altitude) <= height_tolerance
        ):
            return False
    return True


def upload(drone, plan, log):
    """Write the plan's pieces to trajectory memory and define them.

    Runs on the ground before arming. cflib imports stay lazy so tests run
    without it; the simulator supplies its own Poly4D and memory type.
    """
    Poly4D = getattr(drone, "Poly4D", None)
    memory_type = getattr(drone, "TYPE_TRAJ", None)
    if Poly4D is None:
        from cflib.crazyflie.mem import MemoryElement
        from cflib.crazyflie.mem.trajectory_memory import Poly4D

        memory_type = MemoryElement.TYPE_TRAJ
    rows = np.asarray(plan["coefficients"], dtype=float)
    mem = drone.cf.mem.get_mems(memory_type)[0]
    capacity = mem.size // PIECE_BYTES
    if len(rows) > capacity:
        raise DataFault(
            f"{len(rows)} pieces, trajectory memory holds {capacity}"
        )
    mem.trajectory = [
        Poly4D(
            float(row[0]),
            Poly4D.Poly(row[1:9].tolist()),
            Poly4D.Poly(row[9:17].tolist()),
            Poly4D.Poly(row[17:25].tolist()),
            Poly4D.Poly(row[25:33].tolist()),
        )
        for row in rows
    ]
    started = time.monotonic()
    if not mem.write_data_sync():
        raise DataFault("trajectory upload failed")
    drone.cf.high_level_commander.define_trajectory(TRAJ_ID, 0, len(rows))
    log.meta.update(
        pieces=len(rows),
        upload_bytes=PIECE_BYTES * len(rows),
        upload_s=time.monotonic() - started,
    )


def fly(drone, feed, plan, cfg, log, stop, status):
    """Nominal attempt; sets status['airborne'] once takeoff is sent."""
    f, hl = cfg.flight, drone.cf.high_level_commander

    log.set_phase("positioning")
    home = wait_for_positioning(drone, feed, cfg, stop)
    start = np.array([*plan["start"], cfg.altitude])
    if np.linalg.norm(home[:2] - start[:2]) > f.placement_tolerance:
        raise DataFault(
            f"place the drone within {f.placement_tolerance:.2f} m "
            f"of start {plan['start']} before takeoff"
        )

    log.set_phase("upload")
    upload(drone, plan, log)

    log.set_phase("takeoff")
    drone.arm()
    _wait(stop, 1.0)
    duration = cfg.altitude / f.takeoff_speed
    status["airborne"] = True
    hl.takeoff(cfg.altitude, duration)
    _monitor(drone, feed, cfg, stop, duration)

    log.set_phase("to_start")
    _go_to(drone, feed, cfg, stop, start, f.transit_speed)

    log.set_phase("start_gate")
    # The averaged start gate establishes settling; no separate hover delay.
    deadline = time.monotonic() + f.start_timeout
    while not at_start(feed, start, cfg):
        _check(drone, feed, cfg)
        if time.monotonic() > deadline:
            raise DataFault("start position/velocity not reached")
        _wait(stop, 0.05)

    log.set_phase("mission")
    mission(drone, feed, cfg, log, stop, sum(plan["durations"]))

    # Landing and disconnect belong to run_flight's single cleanup path.


def mean_altitude(feed, cfg, s):
    """Altitude averaged over the last ALTITUDE_AVG_S; one reading can spike."""
    window = feed.window(ALTITUDE_AVG_S, cfg.flight.stale_s)
    return float(np.mean(window[:, 2])) if window is not None else float(s[2])


def mission(drone, feed, cfg, log, stop, duration):
    """Start the uploaded curve, then monitor it until it ends.

    The host times around `start_trajectory` only schedule monitoring:
    cflib sends a packet and returns, it does not confirm playback. The
    analysis start time is fitted from the logged setpoints.
    """
    state = feed.snapshot(cfg.flight.stale_s)
    log.meta["t_cmd_before"] = time.monotonic()
    drone.cf.high_level_commander.start_trajectory(
        TRAJ_ID,
        time_scale=1.0,
        relative_position=False,
        relative_yaw=False,
    )
    t0 = log.meta["t_cmd_after"] = time.monotonic()
    log.mission_start(t0, state)
    end = t0 + duration + END_HOLD_S
    while time.monotonic() < end:
        s = _check(drone, feed, cfg)
        z = mean_altitude(feed, cfg, s)
        if abs(z - cfg.altitude) > cfg.altitude_tolerance:
            raise TaskViolation(f"altitude {z:.2f} m")
        _wait(stop, min(MONITOR_S, end - time.monotonic()))
    log.mission_end(time.monotonic())


def land(drone, feed, cfg):
    """Hand back to the high-level commander and land from the fresh height."""
    cf = drone.cf
    cf.commander.send_notify_setpoint_stop()
    try:
        z = feed.snapshot(cfg.flight.stale_s)[2]
    except DataFault:
        z = cfg.altitude + cfg.altitude_tolerance  # mission height upper bound
    duration = max(z, 0.1) / cfg.flight.land_speed
    cf.high_level_commander.land(0.0, duration)
    end = time.monotonic() + duration + 0.5
    while time.monotonic() < end:
        try:
            time.sleep(max(0.0, end - time.monotonic()))
        except KeyboardInterrupt:
            continue  # repeated Ctrl-C must not interrupt landing
    cf.high_level_commander.stop()


def safe_shutdown(drone, feed, cfg, airborne, blocks, log):
    """One cleanup path; recovery errors never replace the mission result."""
    if drone is None:
        return
    if airborne:
        try:
            land(drone, feed, cfg)
            log.meta["landing"] = "landed"
        except Exception as exc:
            log.meta["landing"] = "failed"
            log.meta["landing_error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("Landing failed; stopping motors")
            try:
                drone.cf.commander.send_stop_setpoint()
            except Exception:
                logger.exception("Stop setpoint failed")
    for step in [conf.stop for conf in blocks] + [
        drone.disarm,
        drone.disconnect,
    ]:
        try:
            step()
        except Exception as exc:
            log.meta.setdefault("cleanup_errors", []).append(str(exc))
            logger.exception("Cleanup step failed")


def run_flight(cfg, plan, log, stop=None, connect_fn=None):
    """One attempt end to end; always cleans up and writes the log."""
    stop = stop or threading.Event()
    drone, blocks, status = None, [], {"airborne": False}
    feed = StateFeed(log)
    outcome, reason, kind = "error", None, "error"
    try:
        if len(plan["coefficients"]) > MAX_PIECES:
            raise DataFault(
                f"{len(plan['coefficients'])} pieces, "
                f"trajectory memory holds {MAX_PIECES}"
            )
        drone = connect_fn(cfg) if connect_fn else connect(cfg, feed)
        blocks = start_logging(
            drone.cf,
            cfg,
            feed,
            log,
            getattr(drone, "LogConfig", None),
            state_logged=getattr(drone, "state_logged", False),
        )
        fly(drone, feed, plan, cfg, log, stop, status)
        outcome, kind = "completed", None
    except (KeyboardInterrupt, Interrupted):
        outcome, reason, kind = "interrupted", "stop requested", "interrupted"
    except (TaskViolation, DataFault) as exc:
        outcome, reason, kind = "aborted", str(exc), type(exc).__name__
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.exception("Flight failed during %s", log.phase)
    finally:
        phase = log.phase
        if "mission_t0" in log.meta:
            log.meta.setdefault("mission_t_end", time.monotonic())
        log.meta["mission_outcome"] = (
            outcome if "mission_t0" in log.meta else "not_started"
        )
        if status["airborne"]:
            log.set_phase("land" if outcome == "completed" else "cleanup_land")
        safe_shutdown(drone, feed, cfg, status["airborne"], blocks, log)
    if reason:
        logger.warning("Flight %s during %s: %s", outcome, phase, reason)
    return log.finish(outcome, reason, kind, phase)


def launch(cfg, plan, log, connect_fn=None):
    """Execute one mission in ROS Sugar, then idle until Ctrl-C."""
    from ros_sugar import Launcher
    from ros_sugar.core import BaseComponent

    class FlightComponent(BaseComponent):
        """ROS Sugar's executor owns execution; no extra mission thread."""

        def __init__(self, **kwargs):
            self.stop, self.finished = threading.Event(), threading.Event()
            self.started, self.result = False, None
            super().__init__(**kwargs)

        def _execute_once(self):
            if self.started:
                return
            self.started = True
            try:
                self.result = run_flight(cfg, plan, log, self.stop, connect_fn)
                print(
                    f"Flight {self.result['outcome']}; cleanup finished "
                    f"(landing: {self.result.get('landing', 'not airborne')}). "
                    "Ctrl-C to exit."
                )
            finally:
                self.finished.set()

        def _execution_step(self):
            pass

        def custom_on_deactivate(self):
            self.stop.set()
            return super().custom_on_deactivate()

        def custom_on_shutdown(self):
            self.stop.set()
            return super().custom_on_shutdown()

    component = FlightComponent(component_name="crazyflie_flight")
    launcher = Launcher()
    launcher.add_pkg(
        components=[component], activate_all_components_on_start=True
    )
    try:
        launcher.bringup()
    except KeyboardInterrupt:
        pass
    finally:
        # Never exit while airborne: an unfinished mission lands first.
        component.stop.set()
        while component.started and not component.finished.is_set():
            try:
                component.finished.wait(timeout=0.5)
            except KeyboardInterrupt:
                print(
                    "Landing and cleaning up; waiting for the flight to end."
                )
    return component.result or log.finish(
        "error", "component did not run", "error"
    )
