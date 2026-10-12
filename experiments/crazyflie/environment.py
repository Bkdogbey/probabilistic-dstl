"""Physical layout of the arena: flight bounds, obstacles and rings.

The layout lives in ``config.yaml`` (``bounds`` and ``environment``). This
module answers geometric questions about it so planning, preflight checks,
plotting and analysis all use the same definitions.
"""

from __future__ import annotations

import numpy as np


def inside_bounds(xyz, bounds, margin=0.0):
    """Per-point mask: inside the flight volume grown by ``margin``.

    Args:
        xyz: Positions, shape ``[N, 3]`` or ``[3]``.
        bounds: The ``bounds`` config section.
        margin: Tolerance added on every side.

    Returns:
        Boolean array of shape ``[N]`` (or a bool for a single point).
    """
    xyz = np.asarray(xyz, dtype=float)
    ok = np.ones(xyz.shape[:-1], dtype=bool)
    for i, axis in enumerate("xyz"):
        lo, hi = bounds[axis]
        ok &= (xyz[..., i] >= lo - margin) & (xyz[..., i] <= hi + margin)
    return ok


def obstacle_hits(xyz, obstacles):
    """Name of the first obstacle each point is too close to.

    Args:
        xyz: Positions, shape ``[N, 3]``.
        obstacles: The ``environment.obstacles`` config section. Boxes span
            the floor to ``height``; ``clearance`` inflates them.

    Returns:
        List of length ``N`` with an obstacle name or ``None``.
    """
    xyz = np.asarray(xyz, dtype=float)
    c = obstacles["clearance"]
    hits = [None] * len(xyz)
    for box in obstacles["boxes"]:
        mask = (
            (xyz[:, 0] > box["x"][0] - c)
            & (xyz[:, 0] < box["x"][1] + c)
            & (xyz[:, 1] > box["y"][0] - c)
            & (xyz[:, 1] < box["y"][1] + c)
            & (xyz[:, 2] < box["height"] + c)
        )
        for k in np.flatnonzero(mask):
            hits[k] = hits[k] or box["name"]
    return hits


def ring_distances(xyz, regions):
    """Closest approach of a path to each ring centre (metres)."""
    xyz = np.asarray(xyz, dtype=float)
    return [
        float(np.linalg.norm(xyz - ring["center"], axis=1).min())
        for ring in regions
    ]


def check(xyz, environment):
    """Raise ValueError if a path violates the enabled layout constraints.

    Enabled obstacles must be avoided by ``clearance``; enabled rings must
    be passed through (within ``radius`` of the centre) in listed order.
    ``xyz`` should be dense enough that no constraint falls between points.
    """
    xyz = np.asarray(xyz, dtype=float)
    obstacles = environment["obstacles"]
    if obstacles["enabled"]:
        hits = obstacle_hits(xyz, obstacles)
        for k, name in enumerate(hits):
            if name:
                raise ValueError(
                    f"trajectory within {obstacles['clearance']} m of {name} "
                    f"at {np.round(xyz[k], 3).tolist()}"
                )
    rings = environment["rings"]
    if not rings["enabled"]:
        return
    first = []
    for ring in rings["regions"]:
        dist = np.linalg.norm(xyz - ring["center"], axis=1)
        inside = np.flatnonzero(dist <= ring["radius"])
        if not len(inside):
            raise ValueError(
                f"trajectory misses {ring['name']} "
                f"(closest {dist.min():.3f} m > {ring['radius']} m)"
            )
        first.append(inside[0])
    if first != sorted(first):
        raise ValueError("trajectory passes the rings out of order")
