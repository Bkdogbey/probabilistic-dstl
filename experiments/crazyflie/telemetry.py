"""Experiment-owned Crazyflie telemetry: two timestamped log blocks.

Each callback is one device packet: ``(block, device_ms, values)``. The
``StateFeed`` stamps the host receive time, keeps the newest packet per
block and forwards the row to the flight log queue. Callbacks never touch
the disk.

A position used for a safety decision is a ``snapshot``: x/y from the
newest ``state`` packet and z from the newest ``alt`` packet, both fresher
than ``stale_s`` and sampled close together on the device clock.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

# 6 floats = 24 bytes, within cflib's 26-byte LogConfig limit.
BLOCKS = {
    "state": [
        "stateEstimate.x",
        "stateEstimate.y",
        "stateEstimate.vx",
        "stateEstimate.vy",
        "ctrltarget.x",
        "ctrltarget.y",
    ],
    "alt": ["stateEstimate.z", "stateEstimate.vz", "ctrltarget.z", "pm.vbat"],
}
COLUMNS = {
    "state": ["x", "y", "vx", "vy", "target_x", "target_y"],
    "alt": ["z", "vz", "target_z", "vbat"],
}
CLOCK_WRAP = 1 << 24  # the device timestamp is a 24-bit ms counter


class StaleState(RuntimeError):
    """No sufficiently fresh, coherent telemetry for a safety decision."""


def device_diff_ms(a, b):
    """``a - b`` on the wrapping 24-bit device clock."""
    d = (int(a) - int(b)) % CLOCK_WRAP
    return d - CLOCK_WRAP if d >= CLOCK_WRAP // 2 else d


def unwrap_ms(device_ms):
    """Monotonic int64 device times from a wrapped 24-bit counter."""
    ms = np.asarray(device_ms, dtype=np.int64)
    if ms.size < 2:
        return ms
    steps = np.diff(ms)
    steps[steps < -CLOCK_WRAP // 2] += CLOCK_WRAP
    return np.concatenate([ms[:1], ms[0] + np.cumsum(steps)])


class StateFeed:
    """Newest packet per block, plus a non-blocking forward to the log.

    Args:
        period_ms: Log period of both blocks.
        sink: ``sink(block, device_ms, t_recv, values)``; must not block.
        on_error: ``on_error(block, message)`` for log-block errors.
        clock: Host clock (seconds); injectable for tests.
    """

    def __init__(self, period_ms, sink=None, on_error=None, clock=None):
        self.period_ms = period_ms
        self._sink = sink
        self._on_error = on_error
        self._clock = clock or time.monotonic
        self._latest = {}
        self._lock = threading.Lock()

    def on_packet(self, block, device_ms, values):
        t_recv = self._clock()
        with self._lock:
            self._latest[block] = (device_ms, t_recv, values)
        if self._sink is not None:
            self._sink(block, device_ms, t_recv, values)

    def on_error(self, block, message):
        logger.error("Log block %s error: %s", block, message)
        if self._on_error is not None:
            self._on_error(block, message)

    def snapshot(self, stale_s):
        """Fresh, coherent position for a safety decision.

        Returns:
            ``{"xyz", "vxy", "target", "device_ms", "age"}``.

        Raises:
            StaleState: A block is missing or older than ``stale_s``, or the
                two blocks were sampled more than 1.5 periods apart (one
                period plus jitter between the two block timers).
        """
        now = self._clock()
        with self._lock:
            state = self._latest.get("state")
            alt = self._latest.get("alt")
        if state is None or alt is None:
            raise StaleState("no telemetry received yet")
        age = now - min(state[1], alt[1])
        if age > stale_s:
            raise StaleState(f"telemetry {age:.3f} s old (limit {stale_s} s)")
        skew = abs(device_diff_ms(state[0], alt[0]))
        if skew > 1.5 * self.period_ms:
            raise StaleState(f"state/alt packets {skew} ms apart")
        x, y, vx, vy, tx, ty = state[2]
        z, _, tz, _ = alt[2]
        return {
            "xyz": np.array([x, y, z]),
            "vxy": np.array([vx, vy]),
            "target": np.array([tx, ty, tz]),
            "device_ms": state[0],
            "age": age,
        }


def _cflib_config(name, period_ms):
    from cflib.crazyflie.log import LogConfig

    return LogConfig(name=name, period_in_ms=period_ms)


def start(cf, cfg, feed, make_config=None):
    """Register and start both log blocks on ``cf``.

    Args:
        cf: cflib ``Crazyflie`` (``drone.cf``) or a ``SimDrone``.
        cfg: Full config; reads ``telemetry.period_ms``.
        feed: The ``StateFeed`` receiving every packet.
        make_config: ``(name, period_ms) -> LogConfig``; cflib's when
            ``None``.

    Returns:
        The started log configs, for ``stop``.
    """
    make_config = make_config or _cflib_config
    period_ms = cfg["telemetry"]["period_ms"]
    handles = []
    for block, variables in BLOCKS.items():
        conf = make_config(block, period_ms)
        for name in variables:
            conf.add_variable(name, "float")
        cf.log.add_config(conf)
        conf.data_received_cb.add_callback(
            lambda ts, data, _, block=block, variables=variables: (
                feed.on_packet(block, ts, tuple(data[v] for v in variables))
            )
        )
        conf.error_cb.add_callback(
            lambda _, msg, block=block: feed.on_error(block, msg)
        )
        conf.start()
        handles.append(conf)
    return handles


def stop(handles):
    """Stop the log blocks; never raises."""
    for conf in handles:
        try:
            conf.stop()
        except Exception:
            logger.exception("Could not stop log block %s", conf.name)
