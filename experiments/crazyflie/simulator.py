"""Hardware-free velocity replay for dry runs and executor tests."""

import math
import threading
import time

import numpy as np
from flight import STATE_VARS, world_to_body


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
                values["pm.vbat"] = 4.1
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
