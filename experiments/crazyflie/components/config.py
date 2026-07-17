"""Load config.yml and expose it as module-level constants, plus the
genuinely-logic helpers that are shared between the 2D baseline and 3D
gate scenarios (spline math, waypoint validation, waypoint JSON I/O,
obstacle-clearance distance, and the geometry signature used to derive
BASELINE_PATH_ID).

This file is a thin loader now -- arena/uncertainty/planner/flight/gate
INPUT VALUES live in config.yml (edit that file, not this one, to change
geometry, obstacle positions, planner hyperparameters, etc.). Nothing
scenario-specific is *constructed* here (that's still planning_2d.py /
planning_3d.py); gate-only geometry VALUES do live in config.yml (section
6 below), but the gate scenario's construction logic (Environment3D,
Planner3D, nominal_gate_waypoints, ...) is entirely in planning_3d.py,
unchanged.

This module is import-safe with NO torch/hardware/ROS dependencies (pure
Python + numpy + PyYAML) -- deliberately NOT using src/utils.py's
load_config() helper, since that module imports torch. Keeping this file
torch-free matters: flight_logger.py and run.py's top-level
`from components.config import VALID_FANS` must never pull in torch.

Sections:
    1. Arena geometry          — bounds, obstacles, goal, start/end, safe path
    2. Deterministic-path calc — required obstacle-clearance margin
    3. Per-fan uncertainty     — SIGMA0_PER_FAN, Q_STD (planning + plots)
    4. Planner hyperparameters — PLANNER_CONFIG (every optimizer knob, labeled)
    5. Trial selection         — TRIAL_FAN/TRIAL_CONDITION/TRIAL_SCENARIO, the
                                  run.py CLI's --fan/--condition/--scenario defaults
    6. Flight parameters       — velocities, heights, calibration, log paths
    7. Gate geometry           — 3D-only; consumed solely by planning_3d.py
    8. Shared helpers          — spline math, waypoint validation/I/O,
                                  obstacle clearance, geometry signature
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

# Use the pdSTL library from this repo's src/ (no vendored copies). This file
# lives at experiments/crazyflie/components/config.py, three levels below
# the repo root. planning_2d.py/planning_3d.py both import this module
# before their own planning.*/pdstl.* imports specifically so this insert
# always runs first -- see the comment at the top of their import blocks.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / 'src'))

import numpy as np
import yaml

# Experiment root (where run.py, config.yml, waypoints/ and plots/ live).
EXPERIMENT_DIR = pathlib.Path(__file__).resolve().parents[1]

VALID_FANS: tuple[int, ...] = (2, 6, 12, 16)


def _t2(v) -> tuple[float, float]:
    """2-tuple-of-float cast for YAML lists that must behave like the old
    Python tuple literals (e.g. unpacked with `*obs['x']`)."""
    return (float(v[0]), float(v[1]))


_CONFIG_PATH = EXPERIMENT_DIR / 'config.yml'
_cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding='utf-8'))


# ═════════════════════════════════════════════════════════════════════════════
# 1. Arena geometry — see config.yml for values + measurement rationale
# ═════════════════════════════════════════════════════════════════════════════
_arena = _cfg['arena']
FLIGHT_X_BOUNDS: list[float] = list(_arena['flight_x_bounds'])
FLIGHT_Y_BOUNDS: list[float] = list(_arena['flight_y_bounds'])
FLIGHT_Z_BOUNDS: list[float] = list(_arena['flight_z_bounds'])
Z_HEIGHT: float = float(_arena['z_height'])

OBSTACLES: list[dict] = [
    {'name': o['name'], 'x': _t2(o['x']), 'y': _t2(o['y']), 'height': float(o['height'])}
    for o in _arena['obstacles']
]
GOAL: dict = {
    'x': _t2(_arena['goal']['x']), 'y': _t2(_arena['goal']['y']), 'z': _t2(_arena['goal']['z']),
}

START_XY: tuple[float, float] = _t2(_arena['start_xy'])
END_XY: tuple[float, float] = _t2(_arena['end_xy'])
START_Z: float = Z_HEIGHT   # cruise altitude at the start of the gate trajectory (derived, not stored)
END_Z: float = Z_HEIGHT     # cruise altitude the goal box is centered on (derived, not stored)
START_TOLERANCE: float = float(_arena['start_tolerance'])

SAFE_PATH_VIA_POINTS: list[tuple[float, float]] = [_t2(p) for p in _arena['safe_path_via_points']]
SAFE_PATH_FLIGHT_POINTS: int = int(_arena['safe_path_flight_points'])


# ═════════════════════════════════════════════════════════════════════════════
# 2. Deterministic-path calculation input (2D baseline only — consumed by
#    planning_2d.py's _calculate_sine_amplitude()). The required per-obstacle
#    clearance margin for the closed-form sine-amplitude calculation; see
#    that function's docstring for how it's used.
# ═════════════════════════════════════════════════════════════════════════════
_det = _cfg['deterministic_path']
DETERMINISTIC_PATH_MARGIN: float = float(_det['margin'])


# ═════════════════════════════════════════════════════════════════════════════
# 3. Per-fan uncertainty model (paper Experiment 3)
#    Σ_t(fan) = SIGMA0_PER_FAN[fan] + t · Q_STD²
# ═════════════════════════════════════════════════════════════════════════════
_unc = _cfg['uncertainty']
SIGMA0_PER_FAN: dict[int, float] = {int(k): float(v) for k, v in _unc['sigma0_per_fan'].items()}
Q_STD: float = float(_unc['q_std'])


# ═════════════════════════════════════════════════════════════════════════════
# 4. Planner hyperparameters — every knob in one place, labeled
#    paper-faithful/heuristic/numerical. Shared by both planning_2d.py's
#    Planner and planning_3d.py's Planner3D.
# ═════════════════════════════════════════════════════════════════════════════
_planner = _cfg['planner']
T: int = int(_planner['T'])
DT: float = float(_planner['dt'])
U_MAX: float = float(_planner['u_max'])
PLANNER_ALPHA: float = float(_planner['alpha'])
PLANNER_CONFIG: dict = {**_planner['config'], 'alpha': PLANNER_ALPHA}


# ═════════════════════════════════════════════════════════════════════════════
# 5. Trial selection — which fan/condition/scenario `run.py` acts on when
#    --fan/--condition/--scenario aren't passed on the command line. Edit
#    config.yml's trial: section instead of typing the same flags every run;
#    CLI flags still exist and override these for one-off invocations.
# ═════════════════════════════════════════════════════════════════════════════
_trial = _cfg['trial']
TRIAL_FAN: int = int(_trial['fan'])
TRIAL_CONDITION: str = str(_trial['condition'])
TRIAL_SCENARIO: str = str(_trial['scenario'])


# ═════════════════════════════════════════════════════════════════════════════
# 6. Flight parameters (used only at flight time by components/crazyflie.py)
#    + log/calibration directories (shared by uncertainty_calibration.py,
#    analyze_logs.py, and components/flight_logger.py).
# ═════════════════════════════════════════════════════════════════════════════
_flight = _cfg['flight']
FLIGHT_VELOCITY: float = U_MAX  # PositionHlCommander default velocity [m/s]
CALIBRATION_HOVER_SECONDS: float = float(_flight['calibration_hover_seconds'])
TAKEOFF_Z: float = Z_HEIGHT
RETURN_Z: float = float(_flight['return_z'])
LAND_Z: float = float(_flight['land_z'])
Z_HOLD: float = float(_flight['z_hold'])

# Drone radio address, e.g. 'radio://0/80/2M/E7E7E7E780'. Set this once for
# your drone in config.yml; leave null there to use irobot's own
# CrazyflieConfig.uri default. Picked up automatically by
# components/crazyflie.py -- no per-run flag needed.
DRONE_URI: str | None = _flight.get('drone_uri')

LOGS_DIR: pathlib.Path = EXPERIMENT_DIR / 'components' / 'logs'
CALIBRATION_DIR: pathlib.Path = EXPERIMENT_DIR / 'calibration'


# ═════════════════════════════════════════════════════════════════════════════
# 7. Gate scenario geometry (3D-only; construction logic lives entirely in
#    planning_3d.py, which imports these names)
# ═════════════════════════════════════════════════════════════════════════════
_gate = _cfg['gate']
GATE_CENTER_XY: tuple[float, float] = _t2(_gate['center_xy'])
GATE_X: tuple[float, float] = _t2(_gate['x'])
GATE_Z: tuple[float, float] = _t2(_gate['z'])
GATE_Z_CENTER: float = float(_gate['z_center'])
GATE_Y_MARGIN: float = float(_gate['y_margin'])
GATE_Y: tuple[float, float] = (GATE_CENTER_XY[1] - GATE_Y_MARGIN, GATE_CENTER_XY[1] + GATE_Y_MARGIN)
POST_GATE_Z: float = float(_gate['post_gate_z'])
POST_GATE_Z_BAND: tuple[float, float] = _t2(_gate['post_gate_z_band'])
GATE_T: int = int(_gate['T'])
GATE_T_START: int = int(_gate['t_start'])
GATE_T_END: int = int(_gate['t_end'])
GATE_T_DESCEND_START: int = int(_gate['t_descend_start'])


# ═════════════════════════════════════════════════════════════════════════════
# 8. Shared helpers — spline math, waypoint validation, waypoint JSON I/O,
#    geometry signature. Used by both planning_2d.py and planning_3d.py.
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


def obstacle_clearance_2d(curve_xy: np.ndarray, obs: dict) -> float:
    """Min x,y-only distance from any point on curve_xy [N,2] to an obstacle's box (0 = inside).

    No altitude term -- callers with a 2D-only view of the arena (the sine-
    amplitude calculation in planning_2d.py, the before/after comparison
    plots in waypoint_planning.py) share this single implementation.
    """
    dx = np.maximum(np.maximum(obs['x'][0] - curve_xy[:, 0], 0.0), curve_xy[:, 0] - obs['x'][1])
    dy = np.maximum(np.maximum(obs['y'][0] - curve_xy[:, 1], 0.0), curve_xy[:, 1] - obs['y'][1])
    return float(np.min(np.sqrt(dx**2 + dy**2)))


def validate_waypoints_in_bounds(
    waypoints: list[tuple[float, float, float]], *, label: str = 'Waypoint',
) -> None:
    """Raise ValueError if any (x, y, z) waypoint falls outside FLIGHT_*_BOUNDS.

    Shared by the planner (checking optimizer output, before saving a plan)
    and the flight component (checking what's about to be commanded, before
    takeoff) so both paths use the same bounds check.
    """
    x_min, x_max = FLIGHT_X_BOUNDS
    y_min, y_max = FLIGHT_Y_BOUNDS
    z_min, z_max = FLIGHT_Z_BOUNDS
    outside = [
        (idx, x, y, z)
        for idx, (x, y, z) in enumerate(waypoints)
        if not (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max)
    ]
    if outside:
        details = ', '.join(f'#{idx}=({x:.3f}, {y:.3f}, {z:.3f})' for idx, x, y, z in outside)
        raise ValueError(
            f'{label}(s) outside flight area '
            f'x=[{x_min}, {x_max}], y=[{y_min}, {y_max}], z=[{z_min}, {z_max}]: {details}'
        )


def geometry_signature_2d() -> str:
    """sha256-based signature of everything that determines
    nominal_safe_waypoints()'s calculated sine amplitude: obstacle geometry,
    start/end points, and the required clearance margin (see
    planning_2d.py's _calculate_sine_amplitude). Deliberately torch-free
    (pure Python + numpy) so it can be computed here and imported by
    flight_logger.py (via BASELINE_PATH_ID below) without pulling in torch.
    Used, via BASELINE_PATH_ID, to auto-invalidate stale calibration/flight
    logs whenever a change to this file would actually change the
    deterministic path's shape -- replacing the old convention of
    hand-bumping a version string.
    """
    payload = {
        'obstacles': OBSTACLES, 'start_xy': START_XY, 'end_xy': END_XY,
        'margin': DETERMINISTIC_PATH_MARGIN,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:12]


BASELINE_PATH_ID: str = f'sine_calc_v1_{geometry_signature_2d()}'


# ── Per-fan optimised waypoint files (waypoints/pdstl[_<scenario>]_fan<L>.json) ─
def _waypoints_path(fan_speed: int, scenario: str = 'baseline') -> pathlib.Path:
    suffix = '' if scenario == 'baseline' else f'_{scenario}'
    return EXPERIMENT_DIR / 'waypoints' / f'pdstl{suffix}_fan{fan_speed}.json'


def save_pdstl_waypoints(fan_speed: int, waypoints: list[tuple[float, float, float]],
                         meta: dict, scenario: str = 'baseline') -> pathlib.Path:
    """Write optimised waypoints for one fan level/scenario as JSON (with metadata)."""
    path = _waypoints_path(fan_speed, scenario)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'fan': fan_speed, 'scenario': scenario, **meta,
               'waypoints': [[float(x), float(y), float(z)] for x, y, z in waypoints]}
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return path


def pdstl_plan_meta(fan_speed: int, scenario: str = 'baseline') -> dict:
    """Read the full waypoints JSON payload (rho_before, rho_after, ...) for one fan/scenario.

    Unlike load_pdstl_waypoints, this doesn't fail on a non-converged plan --
    it's used to *decide* whether a plan is flyable (see run.py's --condition
    pdstl guard) before committing to loading/flying it.
    """
    path = _waypoints_path(fan_speed, scenario)
    if not path.exists():
        scenario_flag = '' if scenario == 'baseline' else f' --scenario {scenario}'
        raise FileNotFoundError(
            f'No optimised waypoints for fan {fan_speed} (scenario={scenario}) at {path}. '
            f'Generate them first:  python run.py plan --fan {fan_speed}{scenario_flag}'
        )
    return json.loads(path.read_text(encoding='utf-8'))


def load_pdstl_waypoints(fan_speed: int, scenario: str = 'baseline') -> list[tuple[float, float, float]]:
    """Load the optimised waypoints for one fan level/scenario, checking fan/scenario match.

    Raises FileNotFoundError with a clear hint if the file hasn't been generated
    (run `python run.py plan --fan <L>` first) -- reuses pdstl_plan_meta for that
    check instead of re-reading the file. Files predating the 'scenario' field
    default to 'baseline' so they still load under the new signature.
    """
    data = pdstl_plan_meta(fan_speed, scenario)
    if data.get('scenario', 'baseline') != scenario:
        raise ValueError(
            f'Waypoints for fan {fan_speed} were generated for scenario '
            f'{data.get("scenario", "baseline")!r}, not {scenario!r}.'
        )
    if data.get('fan') != fan_speed:
        raise ValueError(
            f'Waypoints file was generated for fan {data.get("fan")}, not {fan_speed}.'
        )
    return [tuple(wp) for wp in data['waypoints']]
