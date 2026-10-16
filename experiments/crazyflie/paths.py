"""The figure-8 path and the timed reference the drone follows; no ROS."""

from __future__ import annotations

import numpy as np


def figure8(cfg, n=400):
    """Vertical Lemniscate of Gerono [n, 3]; starts and ends at the bottom tip.

    Raises if any point leaves the workspace.
    """
    p, ws = cfg.path, cfg.workspace
    phi = np.linspace(np.pi, 3 * np.pi, n)
    x = p.centre[0] + p.width / 2 * np.sin(2 * phi)
    y = p.centre[1] + p.height / 2 * np.cos(phi)
    inside = (
        ws.x[0] <= x.min() and x.max() <= ws.x[1]
        and ws.y[0] <= y.min() and y.max() <= ws.y[1]
        and cfg.altitude <= ws.z_max
    )
    if not inside:
        raise ValueError("the path leaves the workspace; adjust config.yml")
    return np.column_stack([x, y, np.full(n, cfg.altitude)])


def transit(cfg, here, there):
    """Straight line [2, 3] from `here` to `there` at the flight altitude.

    Raises if the line passes through an obstacle grown by the airframe
    radius, since the physical boxes are in the way.
    """
    a = np.array([here[0], here[1], cfg.altitude])
    b = np.array([there[0], there[1], cfg.altitude])
    xy = a[:2] + np.linspace(0.0, 1.0, 200)[:, None] * (b - a)[:2]
    r = cfg.airframe_radius
    for box in cfg.obstacles:
        hit = ((box.x[0] - r <= xy[:, 0]) & (xy[:, 0] <= box.x[1] + r)
               & (box.y[0] - r <= xy[:, 1]) & (xy[:, 1] <= box.y[1] + r))
        if hit.any():
            raise ValueError(f"the way to the start crosses obstacle "
                             f"{box.name!r}; move the drone")
    return np.array([a, b])


class Reference:
    """Position r(t) and velocity along the path at constant speed."""

    def __init__(self, points, speed):
        self.points = points
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(steps)])
        self.length = float(self.s[-1])
        self.speed = speed
        self.duration = self.length / speed
        self.end = points[-1]

    def at(self, t):
        """(position, velocity) at time t; velocity is zero past the end."""
        s = self.speed * t
        if s >= self.length:
            return self.end, np.zeros(3)
        i = min(np.searchsorted(self.s, s, side="right") - 1, len(self.s) - 2)
        seg = self.points[i + 1] - self.points[i]
        frac = (s - self.s[i]) / (self.s[i + 1] - self.s[i])
        return self.points[i] + frac * seg, seg / np.linalg.norm(seg) * self.speed
