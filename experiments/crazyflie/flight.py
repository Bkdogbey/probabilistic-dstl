"""Crazyflie flight execution: connect, position check, fly, land, clean up.

``drone`` is an irobot ``CrazyflieBase`` (connection, arming and the 20 Hz
``stateEstimate`` stream in ``current_*``) or a ``SimDrone`` for dry runs.
Takeoff, transit to the start and landing use the cflib high-level
commander; the timed trajectory is streamed as low-level position
setpoints, as in irobot's ``CrazyflieController._move_to``.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import traceback
from collections import deque

import numpy as np

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


def _position(drone):
    return np.array([drone.current_x, drone.current_y, drone.current_z])


def _inside(pos, bounds, margin):
    return all(
        bounds[a][0] - margin <= pos[i] <= bounds[a][1] + margin
        for i, a in enumerate("xyz")
    )


def _check(drone, cfg):
    """Abort (raise) on a lost link or an estimate outside the geofence."""
    if not drone.is_connected:
        raise RuntimeError("radio link lost")
    pos = _position(drone)
    if not _inside(pos, cfg["bounds"], cfg["bounds"]["margin"]):
        raise RuntimeError(f"geofence violated at {np.round(pos, 3).tolist()}")


def _monitor(drone, cfg, stop, seconds):
    """Wait ``seconds`` while checking the link and geofence."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        _check(drone, cfg)
        _wait(stop, min(0.05, end - time.monotonic()))


def wait_for_positioning(drone, cfg, stop):
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
        window.append(_position(drone))
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


def _land(drone, cfg, log):
    """Hand control back to the high-level commander and land."""
    cf = drone.cf
    cf.commander.send_notify_setpoint_stop()
    pos = _position(drone)
    duration = max(pos[2], 0.1) / cfg["flight"]["land_speed"]
    cf.high_level_commander.land(0.0, duration)
    log.command((pos[0], pos[1], 0.0))
    time.sleep(duration + 0.5)
    cf.high_level_commander.stop()


def fly(drone, t, xyz, cfg, log, stop, status):
    """Nominal flight. Sets ``status['airborne']`` once takeoff is sent."""
    f = cfg["flight"]
    hl = drone.cf.high_level_commander

    log.set_phase("positioning")
    home = wait_for_positioning(drone, cfg, stop)
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
    _monitor(drone, cfg, stop, duration + f["settle_s"])

    log.set_phase("to_start")
    legs = [start] if direct else [(start[0], start[1], height), start]
    for leg in legs:
        leg = np.asarray(leg, dtype=float)
        distance = np.linalg.norm(leg - _position(drone))
        duration = max(distance / f["transit_speed"], 1.0)
        hl.go_to(*leg, 0.0, duration)
        log.command(leg)
        _monitor(drone, cfg, stop, duration)
    deadline = time.monotonic() + f["start_timeout_s"]
    while np.linalg.norm(_position(drone) - start) > f["start_tolerance"]:
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"start not reached within {f['start_tolerance']} m "
                f"(at {np.round(_position(drone), 3).tolist()})"
            )
        _monitor(drone, cfg, stop, 0.1)
    _monitor(drone, cfg, stop, f["settle_s"])

    log.set_phase("trajectory")
    period = 1.0 / f["stream_hz"]
    end = t[-1] + f["hold_end_s"]
    t_begin = time.monotonic()
    for k in range(math.floor(end / period) + 1):
        setpoint = trajectory.at(t, xyz, k * period)
        drone.cf.commander.send_position_setpoint(*setpoint, 0.0)
        log.command(setpoint)
        _check(drone, cfg)
        _wait(stop, t_begin + (k + 1) * period - time.monotonic())

    log.set_phase("land")
    _land(drone, cfg, log)
    status["airborne"] = False
    log.set_phase("landed")


def safe_shutdown(drone, cfg, log, airborne):
    """Land if airborne, then disarm and disconnect; never raises."""
    if drone is None:
        return
    if airborne:
        try:
            _land(drone, cfg, log)
        except Exception:
            logger.exception("Landing failed; stopping motors")
            try:
                drone.cf.commander.send_stop_setpoint()
            except Exception:
                logger.exception("Stop setpoint failed")
    for step in (drone.disarm, drone.disconnect):
        try:
            step()
        except Exception:
            logger.exception("Cleanup step %s failed", step.__name__)


def run_flight(cfg, t, xyz, log, stop=None, connect_fn=connect):
    """Run one attempt end to end; always cleans up and writes the summary."""
    stop = stop or threading.Event()
    drone = None
    status = {"airborne": False}
    outcome, error, tb = "error", None, None
    try:
        log.set_phase("connect")
        drone = connect_fn(cfg)
        log.start_telemetry(drone)
        fly(drone, t, xyz, cfg, log, stop, status)
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
        safe_shutdown(drone, cfg, log, status["airborne"])
        summary = log.finish(outcome, error, phase=phase, traceback=tb)
    return summary


class SimDrone:
    """Minimal stand-in for an irobot ``CrazyflieBase`` and its ``cf``.

    The position follows the commanded target with a first-order lag. All
    commands are recorded in ``calls``. ``fail_on`` names a command that
    raises; ``fail_after`` makes the n-th streamed setpoint raise.
    """

    def __init__(
        self,
        home=(0.5, -0.2, 0.0),
        tau=0.05,
        noise=0.0,
        fail_on=(),
        fail_after=None,
        seed=0,
    ):
        self.cf = self.commander = self.high_level_commander = self.param = (
            self
        )
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
        self._setpoints = 0
        self._lock = threading.Lock()

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

    def _read(self):
        with self._lock:
            now = time.monotonic()
            gain = 1.0 - math.exp(-(now - self._last) / self._tau)
            self._pos = self._pos + gain * (self._target(now) - self._pos)
            self._last = now
            return self._pos + self._rng.normal(0.0, self._noise, 3)

    current_x = property(lambda self: self._read()[0])
    current_y = property(lambda self: self._read()[1])
    current_z = property(lambda self: self._read()[2])

    # irobot CrazyflieBase
    def arm(self):
        self._record("arm")

    def disarm(self):
        self._record("disarm")

    def disconnect(self):
        self._record("disconnect")
        self.is_connected = False

    # cf.param
    def set_value(self, name, value):
        self._record(f"param {name}={value}")

    # cf.high_level_commander
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

    # cf.commander
    def send_position_setpoint(self, x, y, z, yaw):
        self._setpoints += 1
        if self._fail_after is not None and self._setpoints > self._fail_after:
            raise RuntimeError("simulated failure during trajectory")
        self._move((x, y, z), 0.0)

    def send_notify_setpoint_stop(self):
        self._record("notify_setpoint_stop")

    def send_stop_setpoint(self):
        self._record("stop_setpoint")
