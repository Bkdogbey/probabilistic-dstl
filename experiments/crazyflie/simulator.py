"""Hardware-free stand-in for the drone: trajectory upload and playback.

During playback the position is the uploaded curve plus a tracking error
that follows the same AR(1) model the fit assumes. Data generated this way
checks plumbing and parameter recovery only. It says nothing about whether
the model suits the hardware, so synthetic results are kept apart from
held-out hardware validation.
"""

import threading
import time
from dataclasses import dataclass

import minsnap
import numpy as np
from flight import STATE_VARS

PERIOD = 0.01  # s per simulation tick
MODEL_DT = 0.2  # s per step of the tracking-error model
TICKS = round(MODEL_DT / PERIOD)


@dataclass(frozen=True)
class ErrorModel:
    """e[k+1] = R e[k] + (I - R) b + w, w ~ N(0, Q), at MODEL_DT; e0 ~ N(m0, P0)."""

    rho: tuple = (0.8, 0.8)
    b: tuple = (0.01, -0.01)
    Q: tuple = ((2e-5, 5e-6), (5e-6, 2e-5))
    m0: tuple = (0.0, 0.0)
    P0: tuple = ((1e-4, 0.0), (0.0, 1e-4))


DEFAULT_ERROR = ErrorModel()


def per_tick(rho, Q, n):
    """Per-tick (r, Q_h) whose every n-th sample is exactly the model.

    For diagonal R_h, Q = sum_{j<n} R_h^j Q_h R_h^j, so r_i = rho_i^(1/n)
    and Q_h[i,l] = Q[i,l] (1 - r_i r_l) / (1 - rho_i rho_l), with the
    limit Q / n when rho_i rho_l = 1. The bias b is unchanged.
    """
    rho, Q = np.asarray(rho, dtype=float), np.asarray(Q, dtype=float)
    r = rho ** (1.0 / n)
    unit = np.isclose(np.outer(rho, rho), 1.0)
    denominator = np.where(unit, 1.0, 1 - np.outer(rho, rho))
    ratio = np.where(unit, 1.0 / n, (1 - np.outer(r, r)) / denominator)
    return r, Q * ratio


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


class SimPoly4D:
    """cflib's Poly4D, without packing."""

    class Poly:
        def __init__(self, values):
            self.values = values

    def __init__(self, duration, x, y, z, yaw):
        self.duration, self.x, self.y, self.z, self.yaw = (
            duration, x, y, z, yaw,
        )  # fmt: skip


class SimMemory:
    """The 4096-byte trajectory memory."""

    size = 4096

    def __init__(self, drone):
        self.drone, self.trajectory = drone, []

    def write_data_sync(self):
        self.drone.calls.append("write_data_sync")
        return not self.drone.fail_upload


class SimDrone:
    """Stand-in for `CrazyflieBase` and its `cf`, simulated at 100 Hz.

    High-level moves are linear in time. `start_trajectory` plays the
    uploaded pieces on the tick clock; the position is the setpoint plus
    the AR(1) error. `stall_after` stops state packets after that many
    seconds of playback; `fail_upload` makes the upload fail.
    """

    LogConfig = SimLogConfig
    Poly4D = SimPoly4D
    TYPE_TRAJ = 18

    def __init__(
        self,
        home=(1.0, -2.0, 0.0),
        error=DEFAULT_ERROR,
        seed=0,
        fail_upload=False,
        stall_after=None,
    ):
        self.cf = self.commander = self.high_level_commander = self
        self.param = self.log = self.mem = self
        self.calls, self.configs, self.starts = [], [], []
        self.is_connected = True
        self.fail_upload, self.stall_after = fail_upload, stall_after
        self.memory = SimMemory(self)
        self.rows = None
        self.pos, self.vel = np.array(home, float), np.zeros(3)
        self.setpoint = self.pos.copy()
        self.error, self.rng = error, np.random.default_rng(seed)
        self.r, Q_h = per_tick(error.rho, error.Q, TICKS)
        self.noise = np.linalg.cholesky(Q_h + 1e-15 * np.eye(2))
        self.e = np.zeros(2)
        self.tick, self.playback_tick = 0, None
        self._move = None  # (from, to, start tick, duration)
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        start = time.monotonic()
        while self.is_connected:
            self.tick += 1
            # Absolute schedule: device time keeps pace with the host clock.
            time.sleep(max(0.0, start + self.tick * PERIOD - time.monotonic()))
            with self._lock:
                self._step()
                values = dict(zip(STATE_VARS, [*self.pos, *self.vel[:2], 0.0]))
                for name, value in zip("xyz", self.setpoint):
                    values[f"ctrltarget.{name}"] = value
                values["pm.vbat"] = 4.1
            device_ms = 123_456 + 10 * self.tick
            for conf in list(self.configs):
                due = self.tick % (conf.period_in_ms // 10) == 0
                if conf.started and due and not self._stalled(conf):
                    data = {v: values[v] for v in conf.variables}
                    conf.data_received_cb.call(device_ms, data, conf)

    def _stalled(self, conf):
        if self.stall_after is None or self.playback_tick is None:
            return False
        playing = (self.tick - self.playback_tick) * PERIOD
        return conf.name == "state" and playing >= self.stall_after

    def _step(self):
        old = self.pos.copy()
        if self.playback_tick is not None:
            t = (self.tick - self.playback_tick) * PERIOD
            if t > 0:  # advance the error from t - PERIOD to t
                noise = self.noise @ self.rng.standard_normal(2)
                self.e = self.r * self.e + (1 - self.r) * self.error.b + noise
            self.setpoint = minsnap.evaluate(self.rows, t)[0]
            self.pos = self.setpoint + [*self.e, 0.0]
        elif self._move is not None:
            start, goal, tick0, duration = self._move
            a = min((self.tick - tick0) * PERIOD / max(duration, 1e-9), 1.0)
            self.setpoint = start + a * (goal - start)
            self.pos = self.setpoint.copy()
        self.vel = (self.pos - old) / PERIOD

    def _goto(self, goal, duration):
        with self._lock:
            self.playback_tick = None
            self._move = (
                self.pos.copy(), np.asarray(goal, float), self.tick, duration,
            )  # fmt: skip

    # CrazyflieBase
    def arm(self):
        self.calls.append("arm")

    def disarm(self):
        self.calls.append("disarm")

    def disconnect(self):
        self.calls.append("disconnect")
        self.is_connected = False

    # cf.param, cf.log, cf.mem
    def set_value(self, name, value):
        self.calls.append(f"param {name}={value}")

    def add_config(self, conf):
        self.configs.append(conf)

    def get_mems(self, memory_type):
        assert memory_type == self.TYPE_TRAJ
        return [self.memory]

    # cf.high_level_commander
    def define_trajectory(self, trajectory_id, offset, n_pieces, type=0):
        self.calls.append("define_trajectory")
        self.defined = (trajectory_id, offset, n_pieces)
        pieces = self.memory.trajectory[offset : offset + n_pieces]
        self.rows = np.array(
            [
                [p.duration, *p.x.values, *p.y.values, *p.z.values,
                 *p.yaw.values]
                for p in pieces
            ]
        )  # fmt: skip

    def start_trajectory(
        self,
        trajectory_id,
        time_scale=1.0,
        relative_position=False,
        relative_yaw=False,
        reversed=False,
    ):
        self.calls.append("start_trajectory")
        self.starts.append(
            {
                "trajectory_id": trajectory_id,
                "time_scale": time_scale,
                "relative_position": relative_position,
                "relative_yaw": relative_yaw,
            }
        )
        with self._lock:
            self.e = self.rng.multivariate_normal(self.error.m0, self.error.P0)
            self.playback_tick = self.start_tick = self.tick
            self._move = None

    def takeoff(self, height, duration):
        self.calls.append("takeoff")
        self._goto([*self.pos[:2], height], duration)

    def go_to(self, x, y, z, yaw, duration):
        self.calls.append("go_to")
        self._goto([x, y, z], duration)

    def land(self, height, duration):
        self.calls.append("land")
        self._goto([*self.pos[:2], height], duration)

    def stop(self):
        self.calls.append("hl_stop")

    # cf.commander
    def send_notify_setpoint_stop(self):
        self.calls.append("notify_setpoint_stop")

    def send_stop_setpoint(self):
        self.calls.append("stop_setpoint")
