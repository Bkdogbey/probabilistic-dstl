"""Single source of truth for the Crazyflie experiment.

Everything you edit to run the experiment lives here: arena geometry, the
per-fan uncertainty model, planner hyperparameters, and flight parameters.
Every other file imports from this module instead of duplicating constants or
reaching into src/ directly.

This module is import-safe with NO hardware/ROS dependencies, so the offline
planning path (`run.py plan`) needs only torch/numpy/matplotlib. The ros_sugar
trial config (CrazyflieConfig) lives in components/crazyflie.py, next to the
flight component that uses it, so importing this file never pulls in ros_sugar.

Sections:
    1. Arena geometry          — bounds, obstacles, goal, start/end, safe path
    2. Per-fan uncertainty     — SIGMA0_PER_FAN, Q_STD  (drives planning + plots)
    3. Planner hyperparameters — PLANNER_CONFIG (every optimizer knob, labeled)
    4. Flight parameters       — velocities, heights, calibration
    5. Factories & helpers     — build_environment/build_planner, waypoint I/O
"""

from __future__ import annotations

import json
import pathlib
import sys

# Use the pdSTL library from this repo's src/ (no vendored copies). Every other
# file gets Environment/SingleIntegrator/Planner/etc. through this module's
# re-exports below, so nothing else needs to touch sys.path. This file lives at
# experiments/crazyflie/components/config.py, three levels below the repo root.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / 'src'))

import numpy as np
import torch

from pdstl.base import BeliefTrajectory as BeliefTrajectory
from planning.dynamics import SingleIntegrator as SingleIntegrator
from planning.environment import Environment as Environment
from planning.planner import Planner as Planner
from planning.planner import TorchGaussianBelief as TorchGaussianBelief

# Experiment root (where run.py, waypoints/ and plots/ live).
EXPERIMENT_DIR = pathlib.Path(__file__).resolve().parents[1]

VALID_FANS: tuple[int, ...] = (2, 6, 12, 16)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Arena geometry (measured; arena extended 0.5 m past the raw x=1.0 wall so
#    start/end/obstacle_3 — all at x=1.0 — don't sit flush on a hard boundary)
# ═════════════════════════════════════════════════════════════════════════════
FLIGHT_X_BOUNDS: list[float] = [-0.5, 1.5]
FLIGHT_Y_BOUNDS: list[float] = [-2.0, 0.5]
Z_HEIGHT: float = 0.2  # flight height [m], fixed altitude (2D planning)

# Measured obstacle boxes (converted from inches at 0.0254 m/in), consumed by
# both the planner and the safety flags in the flight logger. 'height' is
# metadata for the future-3D seam (z_profile()); the current 2D collision
# logic ignores it.
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

# Via-points (x, y) the deterministic safe path passes through between START_XY
# and END_XY, computed with no disturbance/wind — the risk in the experiment
# comes from *flying* this nominally-safe path under real fan noise, not from
# the path itself being unsafe. (0.25, -1.0) clears obs_1 on its left side;
# (0.617, -0.58) is the midpoint of the gap between obs_2 and obs_3.
SAFE_PATH_VIA_POINTS: list[tuple[float, float]] = [
    (0.25, -1.0),
    (0.617, -0.58),
]

# Waypoint count actually flown for the deterministic condition (denser than
# the T-step planning horizon, purely for a smoother flight path — the T-point
# warm-start fed to the optimizer is unaffected).
SAFE_PATH_FLIGHT_POINTS: int = 30


# ═════════════════════════════════════════════════════════════════════════════
# 2. Per-fan uncertainty model
#    Σ_t(fan) = SIGMA0_PER_FAN[fan] + t · Q_STD²   (open-loop growth)
# ═════════════════════════════════════════════════════════════════════════════
# Initial belief *variance* (m², per axis) for each fan setting — used directly
# as Σ0 (NOT squared). Fans 2/6/12 are the paper's Settings 1–3; fan 16 has no
# paper calibration and is a flagged extrapolation. This is the one knob that
# distinguishes the fan levels, so each fan's plot shows its own uncertainty.
SIGMA0_PER_FAN: dict[int, float] = {
    2: 0.001,
    6: 0.006,
    12: 0.020,
    16: 0.050,  # uncalibrated extrapolation — no paper source
}

# Shared per-step process-noise std (m). Modest vs Σ0 so the per-fan Σ0
# differences dominate the plots, while ellipses still grow along the path.
# Ideally calibrate from tracking residuals; a single value keeps it simple.
Q_STD: float = 0.01

# Planning horizon (number of control steps → T+1 waypoints), inter-waypoint
# time and max inter-waypoint speed (measured from the experiment setup).
T: int = 10
DT: float = 0.693   # mean inter-waypoint time [s]
U_MAX: float = 0.44  # max inter-waypoint speed [m/s]


# ═════════════════════════════════════════════════════════════════════════════
# 3. Planner hyperparameters — every knob in one place (was split between
#    configs/planning.yaml and an override dict). Each value below shadows the
#    repo-level YAML so the Crazyflie run is fully described here.
# ═════════════════════════════════════════════════════════════════════════════
PLANNER_ALPHA: float = 0.90  # P(sat) early-stop threshold (also recorded in outputs)

PLANNER_CONFIG: dict = {
    # ── Paper-faithful objective (J = w_phi·(−log(P_sat+ε)) + w_u·|u|²) ──
    'w_phi': 20.0,   # weight on the STL satisfaction objective
    'w_u': 0.1,      # λ on control effort |u|²
    'alpha': PLANNER_ALPHA,

    # ── Heuristic shaping (NOT in the paper — potential fields that speed up
    #    convergence; set to 0.0 for the clean paper objective) ──
    'w_dist': 5.0,      # pull final position toward goal centre
    'w_obs': 5.0,       # repel trajectory from obstacle centres
    'w_visit': 5.0,     # pull toward visit regions (none defined here → inert)
    'obs_margin': 0.75,  # safety margin (m) added to obstacle radius in repulsion

    # ── Numerical / optimizer internals ──
    'w_du': 0.1,             # smoothness (penalise input rate of change)
    'lr': 0.02,              # Adam learning rate
    'max_iters': 1000,       # max optimisation iterations
    'min_iters': 10,         # min iters before the loss-tolerance check activates
    'converge_patience': 50,  # consecutive iters at P≥alpha before early stop
    'loss_tol': 1.0e-4,      # loss convergence tolerance

    # ── Robustness smoothing ──
    # β log-sum-exp smoothing of min/max in the STL robustness. scale > 0 =
    # smooth (paper's differentiable form); scale <= 0 = exact min/max. Kept
    # exact by default to preserve current behaviour; raise to e.g. 10–50 to
    # enable smoothing.
    'scale': -1,
}


# ═════════════════════════════════════════════════════════════════════════════
# 4. Flight parameters (used only at flight time by components/crazyflie.py)
# ═════════════════════════════════════════════════════════════════════════════
FLIGHT_VELOCITY: float = U_MAX  # PositionHlCommander default velocity [m/s]
CALIBRATION_HOVER_SECONDS: float = 2.0
TAKEOFF_Z: float = Z_HEIGHT
RETURN_Z: float = 0.65
LAND_Z: float = 0.1


# ═════════════════════════════════════════════════════════════════════════════
# 5. Factories & helpers
# ═════════════════════════════════════════════════════════════════════════════
def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson monotone cubic Hermite slopes.

    Unlike a natural cubic spline, this doesn't overshoot past the via points'
    x-range between nodes — important here since overshoot would eat into the
    obstacle clearances the via points were chosen to give.
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


def nominal_safe_waypoints(n_points: int = T) -> list[tuple[float, float, float]]:
    """Deterministic path from START_XY through SAFE_PATH_VIA_POINTS to END_XY.

    A monotone (PCHIP) cubic Hermite spline through the via points, computed to
    clear all three obstacles with no disturbance modelled. This is the baseline
    flown by the deterministic condition, and (at n_points=T, the default) the
    warm start the pdSTL optimizer improves on.
    """
    nodes = [START_XY, *SAFE_PATH_VIA_POINTS, END_XY]
    x_nodes = np.array([p[0] for p in nodes], dtype=float)
    y_nodes = np.array([p[1] for p in nodes], dtype=float)

    slopes = _pchip_slopes(y_nodes, x_nodes)
    y_pos = np.linspace(START_XY[1], END_XY[1], n_points)
    x_pos = _pchip_eval(y_pos, y_nodes, x_nodes, slopes)
    return [(float(x), float(y), Z_HEIGHT) for x, y in zip(x_pos, y_pos)]


def reference_direct_path(n_points: int = 200) -> list[tuple[float, float]]:
    """The 'obstacles are not there' straight line START_XY -> END_XY.

    Plotting-only reference so via-points can be chosen by comparing against the
    obstacle layout — never flown, not used for planning.
    """
    x_nodes = np.array([START_XY[0], END_XY[0]], dtype=float)
    y_nodes = np.array([START_XY[1], END_XY[1]], dtype=float)
    slopes = _pchip_slopes(y_nodes, x_nodes)
    y_pos = np.linspace(START_XY[1], END_XY[1], n_points)
    x_pos = _pchip_eval(y_pos, y_nodes, x_nodes, slopes)
    return [(float(x), float(y)) for x, y in zip(x_pos, y_pos)]


def build_environment() -> Environment:
    """Build the pdSTL Environment (bounds, obstacles, goal) from the geometry above."""
    env = Environment()
    env.set_bounds(x_range=FLIGHT_X_BOUNDS, y_range=FLIGHT_Y_BOUNDS)
    for obs in OBSTACLES:
        env.add_obstacle(x_range=list(obs['x']), y_range=list(obs['y']))
    env.set_goal(x_range=list(GOAL['x']), y_range=list(GOAL['y']))
    return env


def build_planner(fan_speed: int) -> tuple[Planner, SingleIntegrator, Environment]:
    """Build a (Planner, dynamics, environment) for the given fan level.

    Fan level selects the *initial* belief covariance Σ0 (see x0_belief); the
    per-step process noise Q_STD is shared across fans (belief growth term).
    """
    dynamics = SingleIntegrator(dt=DT, u_max=U_MAX, q_std=Q_STD)
    env = build_environment()
    planner = Planner(dynamics, env, T, config=dict(PLANNER_CONFIG))
    return planner, dynamics, env


def x0_belief(fan_speed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Initial (mean, cov) belief for the given fan level.

    The covariance is SIGMA0_PER_FAN[fan_speed] on each axis — this is what
    makes each fan's optimisation and plot reflect its own uncertainty.
    """
    sigma0 = SIGMA0_PER_FAN[fan_speed]
    mean = torch.tensor(START_XY, dtype=torch.float32)
    cov = torch.diag(torch.tensor([sigma0, sigma0], dtype=torch.float32))
    return mean, cov


# ── Per-fan optimised waypoint files (waypoints/pdstl_fan<L>.json) ────────────
def _waypoints_path(fan_speed: int) -> pathlib.Path:
    return EXPERIMENT_DIR / 'waypoints' / f'pdstl_fan{fan_speed}.json'


def save_pdstl_waypoints(fan_speed: int, waypoints: list[tuple[float, float, float]],
                         meta: dict) -> pathlib.Path:
    """Write optimised waypoints for one fan level as JSON (with metadata)."""
    path = _waypoints_path(fan_speed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'fan': fan_speed, **meta,
               'waypoints': [[float(x), float(y), float(z)] for x, y, z in waypoints]}
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return path


def load_pdstl_waypoints(fan_speed: int) -> list[tuple[float, float, float]]:
    """Load the optimised waypoints for one fan level, checking the fan matches.

    Raises FileNotFoundError with a clear hint if the file hasn't been generated
    (run `python run.py plan --fan <L>` first).
    """
    path = _waypoints_path(fan_speed)
    if not path.exists():
        raise FileNotFoundError(
            f'No optimised waypoints for fan {fan_speed} at {path}. '
            f'Generate them first:  python run.py plan --fan {fan_speed}'
        )
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('fan') != fan_speed:
        raise ValueError(
            f'{path} was generated for fan {data.get("fan")}, not {fan_speed}.'
        )
    return [tuple(wp) for wp in data['waypoints']]
