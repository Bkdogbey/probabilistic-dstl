"""Preflight start-position calibration.

Hardware/simulation drift means the drone's actual tracked position rarely
matches the plan's assumed start exactly. This module hovers at the planned
start, measures the real position, and either flags too-large an offset (so
the flight aborts before doing anything risky) or hands back a translation
to shift the whole plan into the drone's actual frame.

Pure Python — no cflib/hardware imports — so it's testable standalone.
"""

from __future__ import annotations

import math
import time
from typing import Callable

PositionGetter = Callable[[], tuple[float, float]]


def hover_and_measure(
    position_getter: PositionGetter,
    duration_s: float = 2.0,
    rate_hz: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[float, float]:
    """Sample `position_getter` for `duration_s` and return the mean (x, y).

    Call this once the drone is holding at the planned start position.
    `position_getter` is typically `lambda: (cf.current_x, cf.current_y)`.
    """
    n = max(1, int(duration_s * rate_hz))
    xs, ys = [], []
    for _ in range(n):
        x, y = position_getter()
        xs.append(x)
        ys.append(y)
        sleep(1.0 / rate_hz)
    return sum(xs) / len(xs), sum(ys) / len(ys)


def compute_offset(
    measured_xy: tuple[float, float], assumed_xy: tuple[float, float]
) -> tuple[float, float]:
    """(dx, dy) = measured - assumed — the shift to apply to the plan."""
    return measured_xy[0] - assumed_xy[0], measured_xy[1] - assumed_xy[1]


def offset_norm(offset: tuple[float, float]) -> float:
    return math.hypot(offset[0], offset[1])


def check_offset_or_abort(offset: tuple[float, float], tolerance: float) -> None:
    """Raise RuntimeError if the offset magnitude exceeds `tolerance` (m)."""
    norm = offset_norm(offset)
    if norm > tolerance:
        raise RuntimeError(
            f'Start offset {norm:.3f} m (dx={offset[0]:.3f}, dy={offset[1]:.3f}) '
            f'exceeds tolerance {tolerance:.3f} m — aborting flight.'
        )


def shift_waypoints(
    waypoints: list[tuple[float, float, float]], offset: tuple[float, float]
) -> list[tuple[float, float, float]]:
    """Translate every (x, y, z) waypoint by (dx, dy), leaving z untouched."""
    dx, dy = offset
    return [(x + dx, y + dy, z) for x, y, z in waypoints]
