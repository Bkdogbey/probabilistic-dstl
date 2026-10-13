"""Crazyflie flight execution: connect, position check, fly, land, clean up.

``drone`` is an irobot ``CrazyflieBase`` (connection and arming) or a
``SimDrone`` for dry runs. State comes only from the experiment's own
timestamped log blocks (``telemetry.py``): every safety decision uses a
fresh ``StateFeed.snapshot`` and aborts on stale data. Takeoff, transit to
the start and landing use the cflib high-level commander; the timed
trajectory is streamed as low-level position setpoints, as in irobot's
``CrazyflieController._move_to``, and aborts if a command would be sent
later than ``flight.max_lateness_s``.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import traceback
from collections import deque

import numpy as np

import telemetry
import trajectory

logger = logging.getLogger(__name__)


class Interrupted(Exception):
    """Raised when a stop is requested (Ctrl-C or ROS shutdown)."""


def connect(cfg):
    """Open the radio link with irobot and wait for the first telemetry."""
    from irobot.src.robots.crazyflie import CrazyflieBase, CrazyflieConfig

    timeout = cfg["flight"]["connect_timeout_s"]
    drone = CrazyflieBase(
        CrazyflieConfig(uri=cfg["radio"]["uri"], connect_timeout=timeout)
    )
    try:
        if not drone.is_connected:
            raise RuntimeError(f"could not connect to {cfg['radio']['uri']}")
        deadline = time.monotonic() + timeout
        while not drone.state_ready:
            if time.monotonic() > deadline:
                raise RuntimeError("no state estimate received after connect")
            time.sleep(0.1)
        # irobot arms on connect; stay disarmed until positioning is stable.
        drone.disarm()
    except BaseException:
        drone.disconnect()
        raise
    return drone


def _wait(stop, seconds):
    if stop.wait(max(seconds, 0.0)):
        raise Interrupted("stop requested")


def _position(feed, cfg):
    """Fresh position for a decision; raises ``telemetry.StaleState``."""
    return feed.snapshot(cfg["telemetry"]["stale_s"])["xyz"]


def _inside(pos, bounds, margin):
    return all(
        bounds[a][0] - margin <= pos[i] <= bounds[a][1] + margin
        for i, a in enumerate("xyz")
    )


def _check(drone, feed, cfg):
    """Abort (raise) on a lost link, stale state or a geofence violation."""
    if not drone.is_connected:
        raise RuntimeError("radio link lost")
    pos = _position(feed, cfg)
    if not _inside(pos, cfg["bounds"], cfg["bounds"]["margin"]):
        raise RuntimeError(f"geofence violated at {np.round(pos, 3).tolist()}")


def _monitor(drone, feed, cfg, stop, seconds):
    """Wait ``seconds`` while checking the link, freshness and geofence."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        _check(drone, feed, cfg)
        _wait(stop, min(0.05, end - time.monotonic()))


def wait_for_telemetry(feed, cfg, stop):
    """Wait until the first fresh snapshot arrives."""
    deadline = time.monotonic() + cfg["flight"]["connect_timeout_s"]
    while True:
        try:
            return _position(feed, cfg)
        except telemetry.StaleState:
            if time.monotonic() > deadline:
                raise
        _wait(stop, 0.05)


def wait_for_positioning(drone, feed, cfg, stop):
    """Reset the Kalman filter and wait for a stable position estimate.

    Ported from irobot's ``examples/waypoint_flight.py``
    (``reset_estimator`` / ``wait_for_stable_estimate``).
    """
    p = cfg["positioning"]
    drone.cf.param.set_value("kalman.resetEstimation", "1")
    _wait(stop, 0.1)
    drone.cf.param.set_value("kalman.resetEstimation", "0")
    _wait(stop, p["reset_wait_s"])

    window = deque(maxlen=p["window"])
    spread = np.full(3, np.inf)
    deadline = time.monotonic() + p["timeout_s"]
    while True:
        window.append(_position(feed, cfg))
        if len(window) == window.maxlen:
            samples = np.asarray(window)
            spread = samples.max(axis=0) - samples.min(axis=0)
            if spread.max() <= p["spread_limit"]:
                return samples.mean(axis=0)
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"position estimate not stable after {p['timeout_s']} s "
                f"(last spread {np.round(spread, 3).tolist()} m)"
            )
        _wait(stop, 0.1)


def _land(drone, feed, cfg, log):
    """Hand control back to the high-level commander and land.

    The landing duration uses the fresh height; with stale telemetry it
    assumes the highest planned altitude, so the descent is never faster
    than ``land_speed``.
    """
    cf = drone.cf
    cf.commander.send_notify_setpoint_stop()
    try:
        pos = _position(feed, cfg)
    except telemetry.StaleState:
        log.event("stale", "landing without fresh height")
        pos = np.array([np.nan, np.nan, cfg["flight"]["transit_z"]])
    duration = max(pos[2], 0.1) / cfg["flight"]["land_speed"]
    cf.high_level_commander.land(0.0, duration)
    log.command((pos[0], pos[1], 0.0))
    time.sleep(duration + 0.5)
    cf.high_level_commander.stop()


def fly(drone, feed, t, xyz, cfg, log, stop, status):
    """Nominal flight. Sets ``status['airborne']`` once takeoff is sent."""
    f = cfg["flight"]
    hl = drone.cf.high_level_commander

    log.set_phase("positioning")
    home = wait_for_positioning(drone, feed, cfg, stop)
    floor = (home[0], home[1], cfg["bounds"]["z"][0])  # ignore ground z
    if not _inside(floor, cfg["bounds"], cfg["bounds"]["margin"]):
        raise RuntimeError(f"home {np.round(home, 3).tolist()} outside bounds")
    logger.info("Position stable at %s", np.round(home, 3).tolist())

    # As on RA-L: take off straight to the start height when placed at the
    # start, otherwise climb to transit_z and cross over the arena.
    start = xyz[0]
    direct = np.linalg.norm(home[:2] - start[:2]) <= f["home_tolerance"]
    height = float(start[2]) if direct else f["transit_z"]

    log.set_phase("takeoff")
    drone.arm()
    _wait(stop, 1.0)
    duration = height / f["takeoff_speed"]
    hl.takeoff(height, duration)
    status["airborne"] = True
    log.command((home[0], home[1], height))
    _monitor(drone, feed, cfg, stop, duration + f["settle_s"])

    log.set_phase("to_start")
    legs = [start] if direct else [(start[0], start[1], height), start]
    for leg in legs:
        leg = np.asarray(leg, dtype=float)
        distance = np.linalg.norm(leg - _position(feed, cfg))
        duration = max(distance / f["transit_speed"], 1.0)
        hl.go_to(*leg, 0.0, duration)
        log.command(leg)
        _monitor(drone, feed, cfg, stop, duration)
    deadline = time.monotonic() + f["start_timeout_s"]
    while np.linalg.norm(_position(feed, cfg) - start) > f["start_tolerance"]:
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"start not reached within {f['start_tolerance']} m "
                f"(at {np.round(_position(feed, cfg), 3).tolist()})"
            )
        _monitor(drone, feed, cfg, stop, 0.1)
    _monitor(drone, feed, cfg, stop, f["settle_s"])

    log.set_phase("trajectory")
    stream(drone, feed, t, xyz, cfg, log, stop)

    log.set_phase("land")
    _land(drone, feed, cfg, log)
    status["airborne"] = False
    log.set_phase("landed")


def stream(drone, feed, t, xyz, cfg, log, stop):
    """Send the trajectory on its schedule ``t_begin + k * period``.

    Every command logs its reference time and lateness. A command later
    than ``max_lateness_s`` aborts instead of being sent, so a scheduling
    stall never turns into a burst of stale setpoints.
    """
    f = cfg["flight"]
    period = 1.0 / f["stream_hz"]
    end = t[-1] + f["hold_end_s"]
    t_begin = time.monotonic()
    for k in range(math.floor(end / period) + 1):
        lateness = time.monotonic() - (t_begin + k * period)
        if lateness > f["max_lateness_s"]:
            raise RuntimeError(
                f"executor late by {lateness:.3f} s at k={k} "
                f"(limit {f['max_lateness_s']} s)"
            )
        setpoint = trajectory.at(t, xyz, k * period)
        drone.cf.commander.send_position_setpoint(*setpoint, 0.0)
        log.command(setpoint, t_ref=k * period, k=k, lateness=lateness)
        _check(drone, feed, cfg)
        _wait(stop, t_begin + (k + 1) * period - time.monotonic())


def safe_shutdown(drone, feed, cfg, log, airborne, blocks):
    """Land if airborne, stop telemetry, disarm, disconnect; never raises."""
    if drone is None:
        return
    if airborne:
        try:
            _land(drone, feed, cfg, log)
        except Exception:
            logger.exception("Landing failed; stopping motors")
            try:
                drone.cf.commander.send_stop_setpoint()
            except Exception:
                logger.exception("Stop setpoint failed")
    telemetry.stop(blocks)
    for step in (drone.disarm, drone.disconnect):
        try:
            step()
        except Exception:
            logger.exception("Cleanup step %s failed", step.__name__)


def run_flight(cfg, t, xyz, log, stop=None, connect_fn=connect):
    """Run one attempt end to end; always cleans up and writes the summary."""
    stop = stop or threading.Event()
    drone, blocks = None, []
    status = {"airborne": False}
    outcome, error, tb = "error", None, None
    feed = telemetry.StateFeed(
        cfg["telemetry"]["period_ms"],
        sink=log.packet,
        on_error=log.block_error,
    )
    try:
        log.set_phase("connect")
        drone = connect_fn(cfg)
        blocks = telemetry.start(
            drone.cf, cfg, feed, getattr(drone, "LogConfig", None)
        )
        wait_for_telemetry(feed, cfg, stop)
        fly(drone, feed, t, xyz, cfg, log, stop, status)
        outcome = "completed"
    except (KeyboardInterrupt, Interrupted):
        outcome, error = "interrupted", f"interrupted during {log.phase}"
        logger.warning("Flight %s", error)
    except Exception as exc:
        error, tb = f"{type(exc).__name__}: {exc}", traceback.format_exc()
        logger.error("Flight failed during %s: %s", log.phase, error)
    finally:
        phase = log.phase
        if status["airborne"]:
            log.set_phase("cleanup_land")
        safe_shutdown(drone, feed, cfg, log, status["airborne"], blocks)
        summary = log.finish(outcome, error, phase=phase, traceback=tb)
    return summary


class _Caller:
    """cflib ``Caller`` stand-in."""

    def __init__(self):
        self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def call(self, *args):
        for callback in self.callbacks:
            callback(*args)


class SimLogConfig:
    """The part of cflib's ``LogConfig`` that ``telemetry.start`` uses."""

    def __init__(self, name, period_in_ms):
        self.name, self.period_in_ms = name, period_in_ms
        self.variables = []
        self.data_received_cb, self.error_cb = _Caller(), _Caller()
        self.started = False

    def add_variable(self, name, fetch_as=None):
        self.variables.append(name)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False


class SimDrone:
    """Stand-in for an irobot ``CrazyflieBase`` and its ``cf``.

    The position follows the commanded target with a first-order lag. An
    emitter thread sends every started log block once per period, with a
    24-bit device-ms counter (starting at ``device_origin_ms``), noisy
    position, finite-difference velocity and the active target. Commands
    are recorded in ``calls``.

    Fault knobs: ``fail_on`` names a command that raises; ``fail_after``
    makes the n-th streamed setpoint raise; ``stall_after`` stops all
    packets (or only ``stall_block``) once n setpoints were streamed;
    ``drop_every`` skips every n-th tick (the device clock still advances);
    ``callback_delay_s`` delays each callback.
    """

    LogConfig = SimLogConfig
    VBAT = 3.9

    def __init__(
        self,
        home=(0.5, -0.2, 0.0),
        tau=0.05,
        noise=0.0,
        fail_on=(),
        fail_after=None,
        stall_after=None,
        stall_block=None,
        drop_every=None,
        callback_delay_s=0.0,
        device_origin_ms=123_456,
        seed=0,
    ):
        self.cf = self.commander = self.high_level_commander = self.param = (
            self.log
        ) = self
        self.calls = []
        self.is_connected = True
        self.state_ready = True
        self.current_vx = self.current_vy = self.current_yaw = 0.0
        self._pos = np.asarray(home, dtype=float)
        self._seg = (self._pos.copy(), self._pos.copy(), 0.0, 0.0)
        self._last = time.monotonic()
        self._tau, self._noise = tau, noise
        self._rng = np.random.default_rng(seed)
        self._fail_on, self._fail_after = set(fail_on), fail_after
        self._stall_after, self._stall_block = stall_after, stall_block
        self._drop_every, self._delay = drop_every, callback_delay_s
        self._origin = device_origin_ms
        self._setpoints = 0
        self._configs = []
        self._lock = threading.Lock()
        threading.Thread(target=self._emit_loop, daemon=True).start()

    def _record(self, name):
        self.calls.append(name)
        if name in self._fail_on:
            raise RuntimeError(f"simulated {name} failure")

    def _target(self, now):
        p0, p1, t0, dur = self._seg
        a = 1.0 if dur <= 0 else min(max((now - t0) / dur, 0.0), 1.0)
        return p0 + a * (p1 - p0)

    def _move(self, goal, duration):
        with self._lock:
            now = time.monotonic()
            self._seg = (
                self._target(now),
                np.asarray(goal, float),
                now,
                duration,
            )

    def _true_position(self):
        with self._lock:
            now = time.monotonic()
            gain = 1.0 - math.exp(-(now - self._last) / self._tau)
            self._pos = self._pos + gain * (self._target(now) - self._pos)
            self._last = now
            return self._pos.copy(), self._target(now)

    def _read(self):
        return self._true_position()[0] + self._rng.normal(0.0, self._noise, 3)

    current_x = property(lambda self: self._read()[0])
    current_y = property(lambda self: self._read()[1])
    current_z = property(lambda self: self._read()[2])

    # --- log blocks -----------------------------------------------------------

    def add_config(self, conf):
        """``cf.log.add_config``."""
        self._configs.append(conf)

    def _emit_loop(self):
        tick, previous = 0, None
        while self.is_connected:
            period_ms = self._configs[0].period_in_ms if self._configs else 50
            time.sleep(period_ms / 1000)
            tick += 1
            pos, target = self._true_position()
            velocity = (
                np.zeros(3)
                if previous is None
                else (pos - previous) / (period_ms / 1000)
            )
            previous = pos
            if self._drop_every and tick % self._drop_every == 0:
                continue
            device_ms = (self._origin + tick * period_ms) % (1 << 24)
            seen = self._rng.normal(0.0, self._noise, 3) + pos
            values = {
                "stateEstimate.x": seen[0],
                "stateEstimate.y": seen[1],
                "stateEstimate.z": seen[2],
                "stateEstimate.vx": velocity[0],
                "stateEstimate.vy": velocity[1],
                "stateEstimate.vz": velocity[2],
                "ctrltarget.x": target[0],
                "ctrltarget.y": target[1],
                "ctrltarget.z": target[2],
                "pm.vbat": self.VBAT,
            }
            for conf in list(self._configs):
                stalled = (
                    self._stall_after is not None
                    and self._setpoints >= self._stall_after
                )
                if stalled and self._stall_block in (None, conf.name):
                    continue
                if not conf.started:
                    continue
                if self._delay:
                    time.sleep(self._delay)
                data = {v: values[v] for v in conf.variables}
                conf.data_received_cb.call(device_ms, data, conf)

    # --- irobot CrazyflieBase ---------------------------------------------------

    def arm(self):
        self._record("arm")

    def disarm(self):
        self._record("disarm")

    def disconnect(self):
        self._record("disconnect")
        self.is_connected = False

    # --- cf.param -------------------------------------------------------------

    def set_value(self, name, value):
        self._record(f"param {name}={value}")

    # --- cf.high_level_commander --------------------------------------------

    def takeoff(self, height, duration):
        self._record("takeoff")
        self._move((*self._target(time.monotonic())[:2], height), duration)

    def go_to(self, x, y, z, yaw, duration):
        self._record("go_to")
        self._move((x, y, z), duration)

    def land(self, height, duration):
        self._record("land")
        self._move((*self._target(time.monotonic())[:2], height), duration)

    def stop(self):
        self._record("hl_stop")

    # --- cf.commander ---------------------------------------------------------

    def send_position_setpoint(self, x, y, z, yaw):
        self._setpoints += 1
        if self._fail_after is not None and self._setpoints > self._fail_after:
            raise RuntimeError("simulated failure during trajectory")
        self._move((x, y, z), 0.0)

    def send_notify_setpoint_stop(self):
        self._record("notify_setpoint_stop")

    def send_stop_setpoint(self):
        self._record("stop_setpoint")
