"""The task around the figure-8: workspace, obstacles and goal.

The specification comes from the pdSTL core (planning.environment), so the
planner and this experiment score paths against the same task. Obstacles
grow and the workspace shrinks by the airframe radius, so every check is on
the drone's centre. This module only measures; it never blocks a flight.
"""

from __future__ import annotations

import sys
from math import ceil

import numpy as np
import torch

from config import HERE

sys.path.insert(0, str(HERE.parents[1] / "src"))  # the pdSTL core

from baselines.det_stl import from_pdstl  # noqa: E402
from planning.environment import Environment  # noqa: E402


def _grow(lo_hi, r):
    return (lo_hi[0] - r, lo_hi[1] + r)


def environment(cfg):
    r = cfg.airframe_radius
    env = Environment()
    env.set_bounds(_grow(cfg.workspace.x, -r), _grow(cfg.workspace.y, -r))
    for box in cfg.obstacles:
        env.add_obstacle(_grow(box.x, r), _grow(box.y, r), name=box.name)
    env.set_goal(cfg.goal.x, cfg.goal.y, name="goal")
    return env


def sample(t, xy, dt):
    """Positions every dt from timestamps t [n] and positions xy [n, 2]."""
    steps = np.arange(ceil(t[-1] / dt - 1e-9) + 1) * dt
    idx = np.minimum(np.searchsorted(t, steps - 1e-9), len(t) - 1)
    return xy[idx]


def _box_distance(xy, x, y):
    """Signed distance from points [n, 2] to a box: > 0 outside."""
    lo, hi = np.array([x[0], y[0]]), np.array([x[1], y[1]])
    d = np.maximum(lo - xy, xy - hi)
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    return np.where(d.max(1) > 0, outside, d.max(1))


def score(cfg, xy):
    """Robustness of the task and the closest approach, for xy every dt.

    robustness > 0: the path satisfies the task (STL, with these boxes).
    clearance: closest distance [m] from the airframe to a wall or obstacle.
    """
    env = environment(cfg)
    xy = np.asarray(xy, float)
    spec = from_pdstl(env.get_specification(len(xy) - 1))
    traj = torch.tensor(xy[None], dtype=torch.float64)
    ws = env.single_region("workspace")
    gaps = [-_box_distance(xy, ws.x, ws.y)]
    gaps += [_box_distance(xy, b.x, b.y) for b in env.by_role("obstacle")]
    return {
        "robustness": float(spec.robustness(traj)),
        "clearance": float(np.min(gaps)),
    }
