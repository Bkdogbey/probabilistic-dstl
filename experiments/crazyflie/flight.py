"""One flight attempt: preflight, start gate, timed velocity commands, landing.

`drone` is an irobot `CrazyflieBase` (radio link, arming) or a `SimDrone`.
The high-level commander takes off, reaches the start and lands; the
mission streams world-frame velocities as body-frame hover setpoints at
`stream_hz`. Every safety decision uses a fresh state packet.

Abort kinds: `TaskViolation` (left the workspace or altitude band: a task
failure) and `DataFault` (stale state, lost link, late command: not a task
outcome). Either one lands through the high-level commander.
"""

from __future__ import annotations

import logging
import math
import os
import signal
import threading
import time
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

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
        """States of the last `seconds` [n, 6]; None until they are all there."""
        self.snapshot(stale_s)  # raises if stale
        now = self.clock()
        with self._lock:
            states = [v for t, v in self._recent if now - t <= seconds]
            oldest = self._recent[0][0]
        return np.array(states) if now - oldest >= seconds else None


def start_logging(cf, cfg, feed, log, make_config=None):
    """One state block at stream_hz (6 floats, 24 bytes) and a 1 Hz battery block."""
    if make_config is None:
        from cflib.crazyflie.log import LogConfig as make_config
    state = make_config(name="state", period_in_ms=1000 // cfg.stream_hz)
    for name in STATE_VARS:
        state.add_variable(name, "float")
    battery = make_config(name="battery", period_in_ms=1000)
    battery.add_variable("pm.vbat", "float")
    blocks = [state, battery]
    for conf in blocks:
        cf.log.add_config(conf)
        conf.error_cb.add_callback(
            lambda conf, msg: log.event(f"log error {msg}")
        )
    state.data_received_cb.add_callback(
        lambda ts, data, _: feed.on_state(ts, data)
    )
    battery.data_received_cb.add_callback(
        lambda ts, data, _: log.battery(data["pm.vbat"])
    )
    for conf in blocks:
        conf.start()
    return blocks


def connect(cfg):
    """Open the radio with irobot; stay disarmed until positioning is stable."""
    from irobot import CrazyflieBase, CrazyflieConfig

    drone = CrazyflieBase(
        CrazyflieConfig(
            uri=cfg.radio_uri, connect_timeout=cfg.flight.connect_timeout
        )
    )
    if not drone.is_connected:
        drone.disconnect()
        raise DataFault(f"could not connect to {cfg.radio_uri}")
    drone.disarm()  # irobot arms on connect
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
        steady = np.ptp(window, axis=0)[:2].max() <= cfg.flight.position_spread
        if len(window) == samples and steady:
            return np.mean(window, axis=0)
        if time.monotonic() > deadline:
            raise DataFault("position estimate did not settle")


def at_start(feed, start, cfg):
    """Is the last `start_window` s, averaged, at the start and at rest?

    The velocity estimate is noisy (0.2-0.3 m/s at rest), so a single sample
    never passes; the window mean does.
    """
    f = cfg.flight
    window = feed.window(f.start_window, f.stale_s)
    if window is None:
        return False
    mean = window.mean(axis=0)
    return (
        np.linalg.norm(mean[:2] - start[:2]) <= f.start_tolerance
        and np.linalg.norm(mean[3:5]) <= f.start_speed
    )


def fly(drone, feed, plan, cfg, log, stop, status):
    """Nominal attempt; sets status['airborne'] once takeoff is sent."""
    f, hl = cfg.flight, drone.cf.high_level_commander

    log.set_phase("positioning")
    home = wait_for_positioning(drone, feed, cfg, stop)
    start = np.array([*plan["start"], cfg.altitude])
    # Placed at the start: climb straight to the altitude. Otherwise cross
    # the arena at transit_z, above the obstacles.
    direct = np.linalg.norm(home[:2] - start[:2]) <= f.home_tolerance
    height = cfg.altitude if direct else f.transit_z

    log.set_phase("takeoff")
    drone.arm()
    _wait(stop, 1.0)
    duration = height / f.takeoff_speed
    hl.takeoff(height, duration)
    status["airborne"] = True
    _monitor(drone, feed, cfg, stop, duration + f.settle)

    log.set_phase("to_start")
    for leg in [start] if direct else [[*start[:2], height], start]:
        distance = np.linalg.norm(
            np.asarray(leg) - _check(drone, feed, cfg)[:3]
        )
        duration = max(distance / f.transit_speed, 1.0)
        hl.go_to(*leg, 0.0, duration)
        _monitor(drone, feed, cfg, stop, duration)

    log.set_phase("start_gate")
    deadline = time.monotonic() + f.start_timeout
    while not at_start(feed, start, cfg):
        _check(drone, feed, cfg)
        if time.monotonic() > deadline:
            raise DataFault("start position/velocity not reached")
        _wait(stop, 0.05)
    _monitor(drone, feed, cfg, stop, f.settle)
    if not at_start(feed, start, cfg):
        raise DataFault("drifted from the start while settling")

    log.set_phase("mission")
    stream(drone, feed, plan, cfg, log, stop)

    log.set_phase("land")
    land(drone, feed, cfg)
    status["airborne"] = False
    log.set_phase("landed")


def stream(drone, feed, plan, cfg, log, stop):
    """Send command k at mission_t0 + k / stream_hz, then hold still.

    Each planned velocity is repeated `cfg.repeats` times. A command that
    would be later than max_lateness_s aborts instead of being sent.
    Logged per command: plan step (-1 while holding), reference and sent
    time, lateness, world and body velocity, and the yaw used.
    """
    f, period = cfg.flight, 1.0 / cfg.stream_hz
    velocities = np.repeat(np.asarray(plan["velocities"]), cfg.repeats, axis=0)
    hold = round(f.hold_end * cfg.stream_hz)
    t0 = time.monotonic()
    log.mission_start(t0, feed.snapshot(f.stale_s))
    for k in range(len(velocities) + hold):
        lateness = time.monotonic() - (t0 + k * period)
        if lateness > f.max_lateness_s:
            raise DataFault(f"command {k} late by {lateness:.3f} s")
        s = _check(drone, feed, cfg)
        if abs(s[2] - cfg.altitude) > cfg.altitude_tolerance:
            raise TaskViolation(f"altitude {s[2]:.2f} m")
        mission = k < len(velocities)
        v = velocities[k] if mission else (0.0, 0.0)
        body = world_to_body(v, s[5])
        drone.cf.commander.send_hover_setpoint(*body, 0.0, cfg.altitude)
        log.command(
            [
                k,
                k // cfg.repeats if mission else -1,
                k * period,
                time.monotonic(),
                lateness,
                *v,
                s[5],
                *body,
                cfg.altitude,
            ]
        )
        _wait(stop, t0 + (k + 1) * period - time.monotonic())


def land(drone, feed, cfg):
    """Hand back to the high-level commander and land from the fresh height."""
    cf = drone.cf
    cf.commander.send_notify_setpoint_stop()
    try:
        z = feed.snapshot(cfg.flight.stale_s)[2]
    except DataFault:
        z = cfg.flight.transit_z  # the highest planned height: never too fast
    duration = max(z, 0.1) / cfg.flight.land_speed
    cf.high_level_commander.land(0.0, duration)
    time.sleep(duration + 0.5)
    cf.high_level_commander.stop()


def safe_shutdown(drone, feed, cfg, airborne, blocks):
    """Land if airborne, stop logging, disarm, disconnect; never raises."""
    if drone is None:
        return
    if airborne:
        try:
            land(drone, feed, cfg)
        except Exception:
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
        except Exception:
            logger.exception("Cleanup step failed")


def run_flight(cfg, plan, log, stop=None, connect_fn=connect):
    """One attempt end to end; always cleans up and writes the log."""
    stop = stop or threading.Event()
    drone, blocks, status = None, [], {"airborne": False}
    feed = StateFeed(log)
    outcome, reason, kind = "error", None, "error"
    try:
        drone = connect_fn(cfg)
        blocks = start_logging(
            drone.cf, cfg, feed, log, getattr(drone, "LogConfig", None)
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
        if status["airborne"]:
            log.set_phase("cleanup_land")
        safe_shutdown(drone, feed, cfg, status["airborne"], blocks)
    if reason:
        logger.warning("Flight %s during %s: %s", outcome, phase, reason)
    return log.finish(outcome, reason, kind, phase)


def launch(cfg, plan, log, connect_fn=connect):
    """Fly inside a ROS Sugar component; returns the trial metadata."""
    from ros_sugar import Launcher
    from ros_sugar.core import BaseComponent

    class FlightComponent(BaseComponent):
        """Runs the mission on its own thread.

        `_execute_once` runs in a timer callback; blocking it would also
        block the lifecycle callbacks that deliver shutdown (which land).
        """

        def __init__(self, **kwargs):
            self.stop, self.mission, self.result = (
                threading.Event(),
                None,
                None,
            )
            super().__init__(**kwargs)

        def _execute_once(self):
            self.mission = threading.Thread(target=self._fly, name="mission")
            self.mission.start()

        def _fly(self):
            try:
                self.result = run_flight(cfg, plan, log, self.stop, connect_fn)
            finally:
                if not self.stop.is_set():  # finished: end Launcher.bringup()
                    os.kill(os.getpid(), signal.SIGINT)

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
        while component.mission is not None and component.mission.is_alive():
            try:
                component.mission.join(timeout=0.5)
            except KeyboardInterrupt:
                print(
                    "Landing and cleaning up; waiting for the flight to end."
                )
    return component.result or log.finish(
        "error", "component did not run", "error"
    )


# Simulation for tests and --dry-run


class _Caller(list):
    add_callback = list.append

    def call(self, *args):
        for callback in self:
            callback(*args)


class SimLogConfig:
    def __init__(self, name, period_in_ms):
        self.name, self.period_in_ms, self.variables = name, period_in_ms, []
        self.data_received_cb, self.error_cb = _Caller(), _Caller()
        self.started = False

    def add_variable(self, name, fetch_as=None):
        self.variables.append(name)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False


class SimDrone:
    """Stand-in for `CrazyflieBase` and its `cf`, simulated at 100 Hz.

    High-level moves are linear in time; hover setpoints drive a
    first-order velocity lag (time constant `tau`) at a fixed `yaw`.
    `fail_after` makes the n-th hover setpoint raise; `stall_after` stops
    state packets after n hover setpoints.
    """

    LogConfig = SimLogConfig

    def __init__(
        self,
        home=(1.0, -2.0, 0.0),
        yaw=0.0,
        tau=0.3,
        fail_after=None,
        stall_after=None,
    ):
        self.cf = self.commander = self.high_level_commander = self.param = (
            self.log
        ) = self
        self.calls, self.configs, self.is_connected = [], [], True
        self.pos, self.vel = np.array(home, float), np.zeros(3)
        self.yaw, self.tau = yaw, tau
        self.fail_after, self.stall_after, self.setpoints = (
            fail_after,
            stall_after,
            0,
        )
        self._move = (
            None  # (from, to, t_start, duration) or ("hover", v_body, z)
        )
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self, period=0.01):
        start = last = time.monotonic()
        tick = 0
        while self.is_connected:
            time.sleep(period)
            tick += 1
            now = time.monotonic()
            with self._lock:
                self._step(now - last)
                values = dict(
                    zip(STATE_VARS, [*self.pos, *self.vel[:2], self.yaw])
                )
                values["pm.vbat"] = 3.9
            last = now
            device_ms = 123_456 + round(1000 * (now - start))
            stalled = (
                self.stall_after is not None
                and self.setpoints >= self.stall_after
            )
            for conf in list(self.configs):
                due = tick % (conf.period_in_ms // 10) == 0
                if (
                    conf.started
                    and due
                    and not (stalled and conf.name == "state")
                ):
                    data = {v: values[v] for v in conf.variables}
                    conf.data_received_cb.call(device_ms, data, conf)

    def _step(self, dt):
        if self._move is None:
            return
        if isinstance(self._move[0], str):  # ("hover", v_body, z)
            _, (vx, vy), z = self._move
            target = np.array([*world_to_body((vx, vy), -self.yaw), 0.0])
            self.vel += (target - self.vel) * (1 - math.exp(-dt / self.tau))
            self.pos += self.vel * dt
            self.pos[2] += (z - self.pos[2]) * min(dt / 0.1, 1.0)
            return
        start, goal, t_start, duration = self._move
        a = min((time.monotonic() - t_start) / max(duration, 1e-9), 1.0)
        new = start + a * (goal - start)
        self.vel = (new - self.pos) / dt
        self.pos = new

    def _goto(self, goal, duration):
        with self._lock:
            self._move = (
                self.pos.copy(),
                np.asarray(goal, float),
                time.monotonic(),
                duration,
            )

    def _record(self, name):
        self.calls.append(name)

    # CrazyflieBase
    def arm(self):
        self._record("arm")

    def disarm(self):
        self._record("disarm")

    def disconnect(self):
        self._record("disconnect")
        self.is_connected = False

    # cf.param, cf.log
    def set_value(self, name, value):
        self._record(f"param {name}={value}")

    def add_config(self, conf):
        self.configs.append(conf)

    # cf.high_level_commander
    def takeoff(self, height, duration):
        self._record("takeoff")
        self._goto([*self.pos[:2], height], duration)

    def go_to(self, x, y, z, yaw, duration):
        self._record("go_to")
        self._goto([x, y, z], duration)

    def land(self, height, duration):
        self._record("land")
        self._goto([*self.pos[:2], height], duration)

    def stop(self):
        self._record("hl_stop")

    # cf.commander
    def send_hover_setpoint(self, vx, vy, yawrate, z):
        self.setpoints += 1
        if self.fail_after is not None and self.setpoints > self.fail_after:
            raise RuntimeError("simulated radio failure")
        with self._lock:
            self._move = ("hover", (vx, vy), z)

    def send_notify_setpoint_stop(self):
        self._record("notify_setpoint_stop")
        with self._lock:
            self.vel[:] = 0.0
            self._move = None

    def send_stop_setpoint(self):
        self._record("stop_setpoint")
