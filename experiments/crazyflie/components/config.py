"""Loads config.yml into module-level constants, plus helpers shared by the
2D baseline and 3D gate scenarios (spline math, waypoint validation,
waypoint JSON I/O, obstacle-clearance distance).

Edit config.yml, not this file, to change geometry, obstacle positions, or
planner hyperparameters. Scenario construction logic lives in
planning_2d.py / planning_3d.py.

Import-safe with no torch/hardware/ROS dependencies (pure Python + numpy +
PyYAML) so flight_logger.py and run.py's top-level `from components.config
import VALID_FANS` never pull in torch.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

# planning_2d.py/planning_3d.py both import this module before their own
# planning.*/pdstl.* imports so this sys.path insert always runs first.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / 'src'))

import numpy as np
import yaml

# Experiment root (where run.py, config.yml, waypoints/ and plots/ live).
EXPERIMENT_DIR = pathlib.Path(__file__).resolve().parents[1]

VALID_FANS: tuple[int, ...] = (2, 6, 12, 16)
VALID_SCENARIOS: tuple[str, ...] = ('baseline', 'gate', 'figure8')


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

def _obstacle_z_range(o: dict) -> tuple[float, float]:
    """Resolve an obstacle's vertical extent to a canonical (z_min, z_max) tuple.

    An obstacle authors its z-extent in exactly one of two mutually exclusive
    ways: `height: h` (floor-mounted shorthand, z spans 0..h) or explicit
    `z: [z_min, z_max]` (a box floating/hanging at an arbitrary height).
    Downstream code (planning_3d.py) reads only the resulting 'z' tuple.
    """
    has_z = 'z' in o
    has_height = 'height' in o
    name = o.get('name', '?')
    if has_z == has_height:
        raise ValueError(
            f"Obstacle {name!r} must set exactly one of 'height' (floor-mounted) "
            f"or 'z' (explicit [z_min, z_max]); got "
            f"{'both' if has_z else 'neither'}."
        )
    z_min, z_max = _t2(o['z']) if has_z else (0.0, float(o['height']))
    if z_min < 0.0:
        raise ValueError(f"Obstacle {name!r} z_min={z_min} must be >= 0.")
    if z_min >= z_max:
        raise ValueError(f"Obstacle {name!r} needs z_min < z_max; got [{z_min}, {z_max}].")
    if z_max > FLIGHT_Z_BOUNDS[1]:
        raise ValueError(
            f"Obstacle {name!r} z_max={z_max} exceeds the flight ceiling "
            f"FLIGHT_Z_BOUNDS[1]={FLIGHT_Z_BOUNDS[1]} (likely a config typo)."
        )
    return z_min, z_max


OBSTACLES: list[dict] = [
    {
        'name': o['name'], 'x': _t2(o['x']), 'y': _t2(o['y']),
        'z': _obstacle_z_range(o),
        # Original floor-mounted shorthand preserved when supplied (footprint-vs-height
        # consumers may still read it); planning reads the canonical 'z' above.
        **({'height': float(o['height'])} if 'height' in o else {}),
    }
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
# 2. Deterministic path (2D baseline only) — via-points the sine curve in
#    planning_2d.py's nominal_safe_waypoints() passes through, between
#    START_XY and END_XY.
# ═════════════════════════════════════════════════════════════════════════════
_det = _cfg['deterministic_path']
DETERMINISTIC_VIA_POINTS: list[tuple[float, float]] = [_t2(p) for p in _det['via_points']]


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
#    + log directory (shared by analyze_logs.py and components/flight_logger.py).
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
# 8. Shared helpers — spline math, waypoint validation, waypoint JSON I/O.
#    Used by both planning_2d.py and planning_3d.py.
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
    """Min x,y-only distance from any point on curve_xy [N,2] to an obstacle's box (0 = inside)."""
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


def validate_waypoint_velocities(
    waypoints: list[tuple[float, float, float]], *,
    start: tuple[float, float, float], dt: float = DT, u_max: float = U_MAX,
    label: str = 'Leg',
) -> None:
    """Raise ValueError if any inter-waypoint leg's implied velocity
    (segment distance / dt) exceeds u_max -- called before takeoff so a bad
    plan/nominal path is caught on the ground, not mid-flight. Shared by all
    three scenarios' pre-flight checks in components/crazyflie.py.
    """
    pts = [start, *waypoints]
    bad = []
    for i in range(len(pts) - 1):
        d = math.dist(pts[i], pts[i + 1])
        v = d / dt
        if v > u_max + 1e-9:
            bad.append((i, v))
    if bad:
        details = ', '.join(f'#{i}->#{i + 1}={v:.3f} m/s' for i, v in bad)
        raise ValueError(
            f'{label}(s) exceed u_max={u_max} m/s given dt={dt}s: {details}'
        )


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


# ═════════════════════════════════════════════════════════════════════════════
# 9. Figure8 scenario geometry (3D-only; construction logic lives entirely in
#    planning_3d.py's figure8 section, which imports these names)
# ═════════════════════════════════════════════════════════════════════════════
_fig8 = _cfg['figure8']
FIG8_X_BOUNDS: list[float] = list(_fig8['workspace']['x'])
FIG8_Y_BOUNDS: list[float] = list(_fig8['workspace']['y'])
FIG8_Z_BOUNDS: list[float] = list(_fig8['workspace']['z'])

_fig8_path = _fig8['path']
FIG8_CENTER_X: float = float(_fig8_path['center_x'])
FIG8_CENTER_Y: float = float(_fig8_path['center_y'])
FIG8_HALF_WIDTH: float = float(_fig8_path['half_width'])
FIG8_Z_BASE: float = float(_fig8_path['z_base'])
FIG8_Z_AMPLITUDE: float = float(_fig8_path['z_amplitude'])

_fig8_mid = _fig8['midpoint_altitude']
FIG8_MIDPOINT_Z: tuple[float, float] = _t2(_fig8_mid['z'])
FIG8_MIDPOINT_T_START: int = int(_fig8_mid['t_start'])
FIG8_MIDPOINT_T_END: int = int(_fig8_mid['t_end'])

FIG8_RETURN_TOLERANCE: float = float(_fig8['return_tolerance'])

FIG8_T: int = int(_fig8['T'])
FIG8_FLIGHT_POINTS: int = int(_fig8['flight_points'])
FIG8_PLOT_POINTS: int = int(_fig8['plot_points'])
FIG8_NOMINAL_MIN_CLEARANCE: float = float(_fig8['nominal_min_clearance'])

_fig8_planner_extra = _fig8['planner_extra']
FIG8_W_REF_XY: float = float(_fig8_planner_extra['w_ref_xy'])
FIG8_W_REF_Z: float = float(_fig8_planner_extra['w_ref_z'])
FIG8_W_TERMINAL: float = float(_fig8_planner_extra['w_terminal'])

FIG8_OBSTACLES: list[dict] = []
for _o in _fig8['obstacles']:
    _xr, _yr = _t2(_o['x']), _t2(_o['y'])
    if _xr[0] >= _xr[1] or _yr[0] >= _yr[1]:
        raise ValueError(
            f"figure8 obstacle {_o['name']!r} needs ascending x/y ranges; "
            f"got x={_xr}, y={_yr}."
        )
    FIG8_OBSTACLES.append({
        'name': _o['name'], 'x': _xr, 'y': _yr,
        'z': _obstacle_z_range(_o),
        **({'height': float(_o['height'])} if 'height' in _o else {}),
    })
