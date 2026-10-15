"""One flight attempt: preflight, start gate, timed velocity commands, landing.

`drone` is an irobot `CrazyflieBase` (radio link, arming) or a `SimDrone`.
The high-level commander takes off, reaches the start and lands; the
mission streams world-frame velocities as body-frame hover setpoints at
`stream_hz`. Every safety decision uses a fresh state packet.

Place the drone near the saved start before launching. Take off, reach the
start, replay the commands, then land in place. Scoring ends before landing.

Abort kinds: `TaskViolation` (left the workspace or altitude band: a task
failure) and `DataFault` (stale state, lost link, late command: not a task
outcome). Either one lands through the high-level commander.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

ALTITUDE_AVG_S = 0.25  # s of state averaged for the altitude limit

STATE_VARS = [
    "stateEstimate.x",
    "stateEstimate.y",
    "stateEstimate.z",
    "stateEstimate.vx",
    "stateEstimate.vy",
    "stateEstimate.yaw",
]


class TaskViolation(RuntimeError):
    """The drone left the geofence or the altitude band."""


class DataFault(RuntimeError):
    """Stale state, lost link, late command or a failed preflight check."""


class Interrupted(Exception):
    """Stop requested (Ctrl-C or ROS shutdown)."""


def world_to_body(v, yaw_deg):
    """Rotate a world-frame XY velocity into the body frame."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return c * v[0] + s * v[1], -s * v[0] + c * v[1]


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
    """Battery logging; the simulator also supplies irobot's state block."""
    if make_config is None:
        from cflib.crazyflie.log import LogConfig as make_config
    battery = make_config(name="battery", period_in_ms=1000)
    battery.add_variable("pm.vbat", "float")
    battery.data_received_cb.add_callback(
        lambda ts, data, _: log.battery(data["pm.vbat"])
    )
    blocks = [battery]
    if not state_logged:
        state = make_config(name="state", period_in_ms=1000 // cfg.stream_hz)
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
            log_rate_ms=1000 // cfg.stream_hz,
        )
    )
    if not drone.is_connected:
        drone.disconnect()
        raise DataFault(f"could not connect to {cfg.radio_uri}")
    # `connected` precedes parameter download completion. Let that traffic
    # finish before preflight and before streaming time-critical commands.
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
    velocities = np.repeat(np.asarray(plan["velocities"]), cfg.repeats, axis=0)
    send(drone, feed, cfg, log, stop, velocities)

    # Landing and disconnect belong to run_flight's single cleanup path.


def mean_altitude(feed, cfg, s):
    """Altitude averaged over the last ALTITUDE_AVG_S; one reading can spike."""
    window = feed.window(ALTITUDE_AVG_S, cfg.flight.stale_s)
    return float(np.mean(window[:, 2])) if window is not None else float(s[2])


def send(drone, feed, cfg, log, stop, velocities):
    """Replay world velocities on an absolute schedule, rotating into body axes.

    New trajectory steps have a strict deadline. Refresh delays may recover
    within one step, without shifting the schedule or extending the mission.
    """
    f, period = cfg.flight, 1.0 / cfg.stream_hz
    t0 = time.monotonic()
    log.mission_start(t0, feed.snapshot(f.stale_s))
    for k, v in enumerate(velocities):
        due = t0 + k * period
        limit = (
            f.max_lateness_s
            if k % cfg.repeats == 0
            else min(cfg.dt, f.stale_s)
        )
        limit = min(limit, (len(velocities) - k) * period)
        s = _check(drone, feed, cfg)
        z = mean_altitude(feed, cfg, s)
        if abs(z - cfg.altitude) > cfg.altitude_tolerance:
            raise TaskViolation(f"altitude {z:.2f} m")
        body = world_to_body(v, s[5])
        lateness = time.monotonic() - due
        if lateness > limit:
            raise DataFault(
                f"command {k} late by {lateness:.3f} s before send"
            )
        drone.cf.commander.send_hover_setpoint(*body, 0.0, cfg.altitude)
        sent = time.monotonic()
        lateness = sent - due
        log.command(
            [
                k,
                k // cfg.repeats,
                k * period,
                sent,
                lateness,
                *v,
                s[5],
                *body,
                cfg.altitude,
            ]
        )
        if lateness > limit:
            raise DataFault(f"command {k} send returned {lateness:.3f} s late")
        _wait(stop, t0 + (k + 1) * period - time.monotonic())
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
