"""Reference generation: config -> plan ``(t, xyz, info)``.

Both planners share ``planning.start``, ``planning.goal``,
``planning.height`` and the arena (``bounds``, ``environment``), so a
deterministic S-path and a pdSTL plan of the same config fly the same task.

- ``s_path``: deterministic constant-speed S from start to goal.
- ``pdstl``: the pdSTL planner in ``src/`` (2D, constant height). The task
  is G[1,H] (inside bounds and outside enabled obstacles) & F goal &
  F_[window_k] ring_k for each enabled ring, solved from a straight-line
  warm start through the rings.

``info`` is JSON-serialisable metadata (method, probability bounds,
covariance, ...). ``save_plan``/``load_plan`` store a plan with the
signature of the config sections it depends on, so a flight can tell
whether its plan is still in sync with the config.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import trajectory

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# A 2D Gaussian lies inside its 95 % ellipse at this many standard
# deviations (sqrt of the chi-square 0.95 quantile with 2 dof).
CHI2_95_2D = 2.4477


# --- S-path ------------------------------------------------------------------


def s_path(cfg):
    """Constant-height S-path from start to goal at constant speed.

    The path follows the start->goal line with a lateral offset
    ``amplitude * sin(2 pi s)`` (positive to the right of travel) for s in
    [0, 1], reparametrised by arc length and sampled every ``dt``.

    Args:
        cfg: Full config; reads ``planning`` and ``planning.s_path``.

    Returns:
        ``(t, xyz, info)``; the last sample is exactly the goal.
    """
    p, sp = cfg["planning"], cfg["planning"]["s_path"]
    start, goal = np.asarray(p["start"], float), np.asarray(p["goal"], float)
    direction = (goal - start) / np.linalg.norm(goal - start)
    right = np.array([direction[1], -direction[0]])
    s = np.linspace(0.0, 1.0, 2001)[:, None]
    dense = start + s * (goal - start)
    dense += sp["amplitude"] * np.sin(2 * np.pi * s) * right

    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    )
    duration = arc[-1] / sp["speed"]
    t = np.arange(0.0, duration, sp["dt"])
    if duration - t[-1] < 1e-9:
        t = t[:-1]
    t = np.append(t, duration)
    target = t * sp["speed"]
    xy = np.column_stack(
        [np.interp(target, arc, dense[:, i]) for i in range(2)]
    )
    xyz = np.column_stack([xy, np.full(len(t), float(p["height"]))])
    return t, xyz, {"method": "s_path"}


# --- pdSTL -------------------------------------------------------------------


def _rect(center, half_width):
    (x, y), h = center, half_width
    return {"x_range": [x - h, x + h], "y_range": [y - h, y + h]}


def _ring_footprint(ring, height):
    """Half-width of the square inside a ring's disc at the plan height."""
    dz = height - ring["center"][2]
    if abs(dz) >= ring["radius"]:
        raise ValueError(
            f"{ring['name']} (z={ring['center'][2]}) cannot be passed at "
            f"planning height {height}"
        )
    return np.sqrt(ring["radius"] ** 2 - dz**2) / np.sqrt(2)


def _ring_windows(start, route, ring_count, horizon, arrival=0.8):
    """Ordered, disjoint step windows, one per ring.

    The warm start reaches the route points at constant speed by
    ``arrival * horizon`` (as ``straight_line_guess`` does); window k spans
    the midpoints between the warm-start arrival steps at rings k-1, k, k+1.
    """
    points = np.vstack([start, route])
    ends = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )
    steps = ends[1 : ring_count + 1] / ends[-1] * arrival * horizon
    cuts = [1, *((steps[:-1] + steps[1:]) / 2), horizon]
    return [
        [int(round(cuts[k])), int(round(cuts[k + 1]))]
        for k in range(ring_count)
    ]


def pdstl_scenario(cfg):
    """Translate the config into a ``src/`` reach-avoid scenario.

    Enabled obstacles become avoid regions (inflated by ``clearance``,
    only if the plan height is below their top plus clearance). Enabled
    rings become ordered visit regions: the square inscribed in each ring's
    disc at the plan height, each within its own step window. The goal is a
    square of ``goal_half_width`` around ``planning.goal``.

    Returns:
        ``(scenario, route)``: the scenario dict for
        ``planning.environment.build_reach_avoid_environment`` /
        ``planning.runners.setup_problem`` and the warm-start waypoints.
    """
    p, pd = cfg["planning"], cfg["planning"]["pdstl"]
    env = cfg["environment"]
    height = float(p["height"])
    start = [float(v) for v in p["start"]]
    double = pd["dynamics"] == "double_integrator"

    obstacles = []
    if env["obstacles"]["enabled"]:
        c = env["obstacles"]["clearance"]
        for box in env["obstacles"]["boxes"]:
            if height < box["height"] + c:
                obstacles.append(
                    {
                        "name": box["name"],
                        "x_range": [box["x"][0] - c, box["x"][1] + c],
                        "y_range": [box["y"][0] - c, box["y"][1] + c],
                    }
                )
    rings = env["rings"]["regions"] if env["rings"]["enabled"] else []
    route = [list(r["center"][:2]) for r in rings] + [list(p["goal"])]
    windows = _ring_windows(start, route, len(rings), pd["H"])
    visits = [
        {
            "name": ring["name"],
            **_rect(ring["center"][:2], _ring_footprint(ring, height)),
            "interval": window,
        }
        for ring, window in zip(rings, windows)
    ]
    scenario = {
        "name": cfg["name"],
        "dynamics": pd["dynamics"],
        "H": pd["H"],
        "dt": pd["dt"],
        "u_max": pd["u_max"],
        "q_std": pd["q_std"],
        "x0_mean": start + ([0.0, 0.0] if double else []),
        "x0_cov_scale": [pd["x0_var"]] * 2 + ([0.0, 0.0] if double else []),
        "bounds": {
            "x_range": cfg["bounds"]["x"],
            "y_range": cfg["bounds"]["y"],
        },
        "obstacles": obstacles,
        "goal": _rect(p["goal"], pd["goal_half_width"]),
        "visit_regions": visits,
        "planner": dict(pd["planner"]),
    }
    return scenario, route


def _import_src():
    """Make the ``src/`` packages importable when not pip-installed."""
    src = str(REPO / "src")
    if src not in sys.path:
        sys.path.append(src)


def pdstl(cfg):
    """Plan with pdSTL; returns ``(t, xyz, info)``.

    The flown path is the planned mean ``mean_trace[0, :, :2]`` at
    ``t_k = k * dt`` and the plan height. ``info`` keeps the exact
    probability interval, the alpha status, a Monte Carlo check and the
    per-step xy covariance.
    """
    _import_src()
    import torch
    from experiments.reach_avoid import straight_line_guess
    from planning.environment import (
        build_reach_avoid_environment,
        shortest_route,
    )
    from planning.runners import monte_carlo, setup_problem, solve

    pd = cfg["planning"]["pdstl"]
    scenario, route = pdstl_scenario(cfg)
    torch.manual_seed(pd["seed"])
    problem = setup_problem(scenario, device="cpu")
    problem.env = build_reach_avoid_environment(scenario)
    problem.spec = problem.env.get_specification(scenario["H"])
    if scenario["obstacles"] and not scenario["visit_regions"]:
        route = shortest_route(problem.env, scenario["x0_mean"][:2])
    problem.init_guess = straight_line_guess(scenario, problem.dyn, route)
    result = solve(problem, label=f"{cfg['name']} pdSTL")

    mean = result.rollout.aux["mean_trace"][0, :, :2].detach().cpu().numpy()
    cov = result.rollout.aux["cov_trace"][0, :, :2, :2].detach().cpu().numpy()
    t = np.arange(len(mean)) * scenario["dt"]
    xyz = np.column_stack(
        [mean, np.full(len(t), float(cfg["planning"]["height"]))]
    )
    info = {
        "method": "pdstl",
        "p_lower": float(result.hard_interval[0]),
        "p_upper": float(result.hard_interval[1]),
        "p_lower_initial": float(result.initial_hard_lower),
        "alpha": result.alpha,
        "alpha_met": result.threshold_met,
        "planning_time_s": float(result.planning_time),
        "ring_windows": [v["interval"] for v in scenario["visit_regions"]],
        "cov_xy": cov.round(8).tolist(),
    }
    if pd["monte_carlo_samples"]:
        rate, lower, upper = monte_carlo(
            problem, result, samples=pd["monte_carlo_samples"]
        )
        info["monte_carlo"] = {
            "rate": rate,
            "lower": lower,
            "upper": upper,
            "samples": pd["monte_carlo_samples"],
        }
    return t, xyz, info


# --- dispatch and storage ------------------------------------------------------

PLANNERS = {"s_path": s_path, "pdstl": pdstl}


def plan(cfg):
    """Run the planner named by ``planning.method``; returns ``(t, xyz, info)``."""
    method = cfg["planning"]["method"]
    if method not in PLANNERS:
        raise ValueError(
            f"unknown planning.method {method!r}; choose from {list(PLANNERS)}"
        )
    return PLANNERS[method](cfg)


def signature(cfg):
    """Hash of every config value a plan depends on."""
    relevant = {
        "planning": cfg["planning"],
        "bounds": cfg["bounds"],
        "environment": cfg["environment"],
        "max_speed": cfg["flight"]["max_speed"],
    }
    blob = json.dumps(relevant, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def plan_dir(cfg, root):
    """Where the current config's plan lives: ``<root>/<name>/<method>``."""
    return Path(root) / cfg["name"] / cfg["planning"]["method"]


def save_plan(directory, cfg, t, xyz, info):
    """Write ``trajectory.csv`` and ``plan.json`` (info + signature)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    trajectory.save_csv(directory / "trajectory.csv", t, xyz)
    meta = {
        **info,
        "signature": signature(cfg),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "points": len(t),
        "duration_s": float(t[-1]),
    }
    (directory / "plan.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_plan(directory):
    """Read a saved plan; returns ``(t, xyz, info)`` or ``None`` if absent."""
    directory = Path(directory)
    if not (directory / "plan.json").exists():
        return None
    t, xyz = trajectory.load_csv(directory / "trajectory.csv")
    return t, xyz, json.loads((directory / "plan.json").read_text())
