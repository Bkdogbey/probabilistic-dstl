"""Velocity tracking of a timed reference; no ROS and no cflib.

`track` runs against two callables, so the same loop flies the real drone
(flight.py) and a simulated plant (tests).
"""

from __future__ import annotations

import time

import numpy as np


class Abort(RuntimeError):
    """Stop tracking and land: stale state, fence breach or lagging."""


def settle(drone, fl, clock=time.monotonic, sleep=time.sleep):
    """Reset the Kalman estimator and wait until the position is steady.

    The estimate can jump for several seconds after a reset while the
    Lighthouse solves. A 1 s window must stay within fl.settle_spread; the
    check repeats until fl.settle_timeout. Returns the mean position.
    """
    for value in ("1", "0"):
        drone.cf.param.set_value("kalman.resetEstimation", value)
        sleep(0.1)
    deadline = clock() + fl.settle_timeout
    while True:
        samples = []
        for _ in range(20):
            sleep(0.05)
            samples.append(drone.state()[0])
        spread = np.ptp(samples, axis=0)
        fresh = drone.state()[1] <= fl.stale_s
        print(f"  settling: pos={np.round(np.mean(samples, axis=0), 2)} "
              f"spread={np.round(spread, 3)}" + ("" if fresh else " (no data)"))
        if fresh and spread.max() <= fl.settle_spread:
            return np.mean(samples, axis=0)
        if clock() > deadline:
            raise Abort("position never settled: check that the Lighthouse "
                        "sees two base stations (check_positioning.py)")


def velocity_command(vel_ref, err, gain, v_max):
    """Feedforward plus P on position error, clamped to v_max."""
    v = np.asarray(vel_ref, float) + gain * np.asarray(err, float)
    speed = np.linalg.norm(v)
    return v * (v_max / speed) if speed > v_max else v


def inside_workspace(pos, ws, margin=0.0):
    x, y, z = pos
    return bool(
        ws.x[0] - margin <= x <= ws.x[1] + margin
        and ws.y[0] - margin <= y <= ws.y[1] + margin
        and z <= ws.z_max + margin
    )


def track(ref, get_state, send, cfg, on_tick=None,
          clock=time.monotonic, sleep=time.sleep):
    """Follow `ref` until its end point is reached; raise Abort on a fault.

    get_state() -> (xyz, age_s). send(v) sends a world velocity [vx, vy, vz].
    on_tick(t, pos, ref_pos, v) is called once per control step.
    """
    ctl, fl = cfg.control, cfg.flight
    dt = 1.0 / ctl.rate
    t0 = next_tick = clock()
    while True:
        t = clock() - t0
        pos, age = get_state()
        if age > fl.stale_s:
            raise Abort(f"state is {age:.2f} s old")
        if not inside_workspace(pos, cfg.workspace, fl.fence_margin):
            raise Abort(f"outside the workspace at {np.round(pos, 2)}")
        r, r_dot = ref.at(t)
        err = r - pos
        if np.linalg.norm(err) > ctl.max_error:
            raise Abort(f"{np.linalg.norm(err):.2f} m from the reference")
        if t >= ref.duration:
            if np.linalg.norm(ref.end - pos) < ctl.tol:
                return
            if t > ref.duration + ctl.end_timeout:
                raise Abort("end point not reached")
        v = velocity_command(r_dot, err, ctl.gain, ctl.v_max)
        send(v)
        if on_tick:
            on_tick(t, pos, r, v)
        next_tick += dt
        sleep(max(0.0, next_tick - clock()))
