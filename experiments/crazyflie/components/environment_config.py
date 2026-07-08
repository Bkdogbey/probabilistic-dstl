"""Single source of truth for arena geometry and how to build a planner from it.

Every other file in this experiment (offline planning, flight execution,
logging) imports geometry and pdSTL types from here instead of duplicating
constants or reaching into src/ directly. Update the arena numbers below in
this one place when the flight area is re-measured.
"""

from __future__ import annotations

import pathlib
import sys

# Use the pdSTL library from this repo's src/ (no vendored copies). Every
# other file gets Environment/SingleIntegrator/Planner/etc. through this
# module's re-exports below, so nothing else needs to touch sys.path.
# This file lives at experiments/crazyflie/components/environment_config.py,
# three levels below the repo root -- parents[3], not parents[2] (that was a
# latent bug: pointed at a nonexistent experiments/src, silently masked
# whenever nothing else registered a conflicting `planning`/`pdstl` package).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / 'src'))

import numpy as np
import torch

from pdstl.base import BeliefTrajectory as BeliefTrajectory
from planning.dynamics import SingleIntegrator as SingleIntegrator
from planning.environment import Environment as Environment
from planning.planner import Planner as Planner
from planning.planner import TorchGaussianBelief as TorchGaussianBelief

# ── Arena geometry (measured; arena extended 0.5 m past the raw x=1.0 wall so
# ── that start/end/obstacle_3 — all of which sit exactly at x=1.0 — don't sit
# ── flush on a hard planning boundary) ───────────────────────────────────────
FLIGHT_X_BOUNDS: list[float] = [-0.5, 1.5]
FLIGHT_Y_BOUNDS: list[float] = [-2.0, 0.5]
Z_HEIGHT: float = 0.2  # flight height [m], fixed altitude (2D planning)

# Measured obstacle boxes (converted from inches at 0.0254 m/in) — single
# list consumed by both the planner (generate_waypoints.py) and the safety
# flags in the flight logger — no more duplicating obstacle boxes across
# files. 'height' is metadata for the future-3D seam (z_profile()); the
# current 2D collision logic ignores it.
OBSTACLES: list[dict] = [
    {'name': 'obs_1', 'x': (0.5, 0.8175), 'y': (-1.2667, -1.0), 'height': 0.4064},
    {'name': 'obs_2', 'x': (0.25, 0.4405), 'y': (-0.6651, -0.5), 'height': 0.5143},
    {'name': 'obs_3', 'x': (0.7936, 1.0), 'y': (-0.6651, -0.5), 'height': 0.3302},
]

GOAL: dict = {'x': (0.85, 1.15), 'y': (-0.15, 0.15)}  # ±0.15 m box around END_XY

START_XY: tuple[float, float] = (1.0, -2.0)
END_XY: tuple[float, float] = (1.0, 0.0)  # safe-path target; goal box center
# Abort the flight if the measured hover position at start differs from
# START_XY by more than this (m) — see components/calibration.py.
START_TOLERANCE: float = 0.08

# ── Planning constants (measured from the experiment setup) ─────────────────
T: int = 10  # nominal planning horizon (number of control steps)
DT: float = 0.693  # mean inter-waypoint time [s]
U_MAX: float = 0.44  # max inter-waypoint speed [m/s]

Q_STD_PER_FAN: dict[int, float] = {
    2: 0.001,
    6: 0.006,
    12: 0.020,
    16: 0.050,
}

PLANNER_ALPHA: float = 0.90
PLANNER_CONFIG: dict = {
    'w_phi': 20.0,
    'w_dist': 5.0,
    'w_obs': 5.0,
    'alpha': PLANNER_ALPHA,
    'max_iters': 1000,
    'lr': 0.02,
}

X0_COV_DIAG: tuple[float, float] = (1e-2, 1e-2)  # initial belief uncertainty
# Tighter: a live Lighthouse reading is much more certain than the initial
# pre-flight belief above, used when replanning mid-flight from measured state.
MEASURED_COV_DIAG: tuple[float, float] = (1e-3, 1e-3)

# Waypoint indices (0-based, into the T+1 = 11-point plan) at which
# components/crazyflie.py replans from the actual measured position instead
# of blindly continuing the offline plan. Append a second index for two
# mid-flight replans.
REPLAN_CHECKPOINTS: list[int] = [5]

# Via-points (x, y) the deterministic safe path passes through between
# START_XY and END_XY, computed without any disturbance/wind — the risk in
# the experiment comes from *flying* this nominally-safe path under real fan
# noise, not from the path itself being unsafe. (0.25, -1.0) clears obs_1 on
# its left side; (0.617, -0.58) is the midpoint of the gap between obs_2 and
# obs_3, which share the same y-band and so can't both be given a wide berth
# by a single via-point.
SAFE_PATH_VIA_POINTS: list[tuple[float, float]] = [
    (0.25, -1.0),
    (0.617, -0.58),
]


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson monotone cubic Hermite slopes.

    Unlike a natural cubic spline, this doesn't overshoot past the via
    points' x-range between nodes — important here since overshoot would
    eat into the obstacle clearances the via points were chosen to give.
    """
    h = np.diff(x)
    delta = np.diff(y) / h
    n = len(x)
    m = np.zeros(n)
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    return m


def _pchip_eval(xq: np.ndarray, x: np.ndarray, y: np.ndarray, m: np.ndarray) -> np.ndarray:
    xq = np.atleast_1d(xq)
    out = np.zeros_like(xq, dtype=float)
    for j, xv in enumerate(xq):
        i = np.searchsorted(x, xv) - 1
        i = min(max(i, 0), len(x) - 2)
        h = x[i + 1] - x[i]
        t = (xv - x[i]) / h
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        out[j] = h00 * y[i] + h10 * h * m[i] + h01 * y[i + 1] + h11 * h * m[i + 1]
    return out


def nominal_safe_waypoints() -> list[tuple[float, float, float]]:
    """10-waypoint deterministic path from START_XY through SAFE_PATH_VIA_POINTS to END_XY.

    Computed to safely clear all three obstacles with no disturbance/wind
    modelled — the experiment's risk comes from flying this nominally-safe
    path under real fan noise, not from the path itself cutting close to
    anything. A monotone (PCHIP) cubic Hermite spline through the via
    points, self-contained in numpy (no new dependency). This is the
    baseline flown by the deterministic condition, and the warm start the
    pdSTL optimizer improves on. Defined once here (not duplicated in
    generate_waypoints.py/crazyflie.py) so the two never drift out of sync
    with each other or with START_XY/END_XY again.
    """
    nodes = [START_XY, *SAFE_PATH_VIA_POINTS, END_XY]
    x_nodes = np.array([p[0] for p in nodes], dtype=float)
    y_nodes = np.array([p[1] for p in nodes], dtype=float)

    slopes = _pchip_slopes(y_nodes, x_nodes)
    y_pos = np.linspace(START_XY[1], END_XY[1], 10)
    x_pos = _pchip_eval(y_pos, y_nodes, x_nodes, slopes)
    return [(float(x), float(y), Z_HEIGHT) for x, y in zip(x_pos, y_pos)]


def build_environment() -> Environment:
    """Build the pdSTL Environment (bounds, obstacles, goal) from the geometry above."""
    env = Environment()
    env.set_bounds(x_range=FLIGHT_X_BOUNDS, y_range=FLIGHT_Y_BOUNDS)
    for obs in OBSTACLES:
        env.add_obstacle(x_range=list(obs['x']), y_range=list(obs['y']))
    env.set_goal(x_range=list(GOAL['x']), y_range=list(GOAL['y']))
    return env


def build_planner(fan_speed: int, T_horizon: int | None = None) -> tuple[Planner, SingleIntegrator, Environment]:
    """Build a (Planner, dynamics, environment) sized to `T_horizon` steps.

    `T_horizon` defaults to the full nominal horizon `T`; pass a smaller
    value to size a planner for a mid-flight replan over the remaining steps.
    """
    horizon = T if T_horizon is None else T_horizon
    dynamics = SingleIntegrator(dt=DT, u_max=U_MAX, q_std=Q_STD_PER_FAN[fan_speed])
    env = build_environment()
    planner = Planner(dynamics, env, horizon, config=dict(PLANNER_CONFIG))
    return planner, dynamics, env


def x0_belief() -> tuple[torch.Tensor, torch.Tensor]:
    """Initial (mean, cov) belief for offline planning, from the assumed start."""
    mean = torch.tensor(START_XY, dtype=torch.float32)
    cov = torch.diag(torch.tensor(X0_COV_DIAG, dtype=torch.float32))
    return mean, cov


def measured_belief(xy: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean, cov) belief built from a live measured position, for mid-flight replanning."""
    mean = torch.tensor(xy, dtype=torch.float32)
    cov = torch.diag(torch.tensor(MEASURED_COV_DIAG, dtype=torch.float32))
    return mean, cov
