"""Timed XYZ trajectories: validation, sampling and CSV I/O.

A trajectory is a pair ``(t, xyz)`` with ``t`` of shape ``[N]`` (seconds,
starting at 0, strictly increasing) and ``xyz`` of shape ``[N, 3]``
(metres). Planning produces it and flight consumes it; neither side needs
to know about the other.
"""

from __future__ import annotations

import numpy as np

import environment


def check(t, xyz, bounds, max_speed, layout=None):
    """Raise ValueError unless the trajectory is well formed and flyable.

    Args:
        t: Times, shape ``[N]``.
        xyz: Positions, shape ``[N, 3]``.
        bounds: The ``bounds`` config section; every point must lie inside.
        max_speed: Largest allowed speed between consecutive points (m/s).
        layout: Optional ``environment`` config section; its enabled
            obstacles and rings are checked on a densified path.
    """
    t = np.asarray(t, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    if t.ndim != 1 or xyz.shape != (len(t), 3) or len(t) < 2:
        raise ValueError(
            f"expected t[N] and xyz[N,3], got {t.shape}, {xyz.shape}"
        )
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(xyz)):
        raise ValueError("trajectory contains non-finite values")
    if abs(t[0]) > 1e-9:
        raise ValueError(f"trajectory must start at t=0, got {t[0]}")
    if np.any(np.diff(t) <= 0):
        raise ValueError("trajectory times must be strictly increasing")
    for i, axis in enumerate("xyz"):
        lo, hi = bounds[axis]
        if xyz[:, i].min() < lo or xyz[:, i].max() > hi:
            raise ValueError(
                f"trajectory leaves {axis} bounds [{lo}, {hi}]: "
                f"[{xyz[:, i].min():.3f}, {xyz[:, i].max():.3f}]"
            )
    if speeds(t, xyz).max() > max_speed:
        raise ValueError(
            f"trajectory speed {speeds(t, xyz).max():.3f} m/s exceeds "
            f"{max_speed} m/s"
        )
    if layout:
        environment.check(densify(xyz), layout)


def densify(xyz, per_segment=10):
    """Points along each straight segment, so checks do not skip gaps."""
    a = np.linspace(0.0, 1.0, per_segment, endpoint=False)[:, None, None]
    seg = xyz[:-1] + a * (xyz[1:] - xyz[:-1])
    return np.vstack([seg.transpose(1, 0, 2).reshape(-1, 3), xyz[-1:]])


def speeds(t, xyz):
    """Segment speeds (m/s), shape ``[N-1]``."""
    return np.linalg.norm(np.diff(xyz, axis=0), axis=1) / np.diff(t)


def at(t, xyz, time):
    """Linearly interpolated setpoint at ``time``, clamped to the ends."""
    return np.array([np.interp(time, t, xyz[:, i]) for i in range(3)])


def save_csv(path, t, xyz):
    """Write ``t,x,y,z`` rows with a header."""
    data = np.column_stack([t, xyz])
    np.savetxt(path, data, delimiter=",", header="t,x,y,z", comments="")


def load_csv(path):
    """Read a ``t,x,y,z`` CSV; returns ``(t, xyz)``."""
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return data[:, 0], data[:, 1:4]
