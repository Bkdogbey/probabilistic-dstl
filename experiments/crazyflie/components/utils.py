"""Shared geometry, plotting, and safety helpers."""

from __future__ import annotations

import math
import json
import hashlib
import pathlib
import sys
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml

from .profile import flight_profile_signature as _flight_profile_signature

# On this machine, a legacy matplotlib-3.5.1-nspkg.pth in the system Python's
# dist-packages pre-loads its own (mplot3d-incompatible) mpl_toolkits into
# sys.modules at interpreter startup, before any project code runs. Evicting
# it here -- the first module everything else in this package imports --
# lets the later `import matplotlib` in analysis.py/planner.py resolve
# mpl_toolkits from the correct (pip-installed) matplotlib instead. A no-op
# on any environment that doesn't have that stale .pth file.
sys.modules.pop('mpl_toolkits', None)

EXPERIMENT_DIR = pathlib.Path(__file__).resolve().parents[1]
_CONFIG_PATH = pathlib.Path(__file__).with_name('config.yml')
_cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding='utf-8'))

VALID_FANS = (2, 6, 12, 16)
VALID_SCENARIOS = ('baseline', 'figure8')
VALID_CONDITIONS = ('pdstl', 'deterministic')
PLAN_SCHEMA_VERSION = 4

Waypoint = tuple[float, float, float]


def _pair(value) -> tuple[float, float]:
    return float(value[0]), float(value[1])


_arena = _cfg['arena']
FLIGHT_X_BOUNDS = list(_arena['flight_x_bounds'])
FLIGHT_Y_BOUNDS = list(_arena['flight_y_bounds'])
FLIGHT_Z_BOUNDS = list(_arena['flight_z_bounds'])
Z_HEIGHT = float(_arena['z_height'])
START_XY = _pair(_arena['start_xy'])
END_XY = _pair(_arena['end_xy'])
START_TOLERANCE = float(_arena['start_tolerance'])
SAFE_PATH_VIA_POINTS = [_pair(point) for point in _arena['safe_path_via_points']]
SAFE_PATH_FLIGHT_POINTS = int(_arena['safe_path_flight_points'])


def _obstacle(source: dict) -> dict:
    has_height, has_z = 'height' in source, 'z' in source
    if has_height == has_z:
        raise ValueError(f"Obstacle {source.get('name', '?')!r} needs exactly one of height or z.")
    z_range = (0.0, float(source['height'])) if has_height else _pair(source['z'])
    if not 0.0 <= z_range[0] < z_range[1] <= FLIGHT_Z_BOUNDS[1]:
        raise ValueError(f"Invalid obstacle z range for {source.get('name', '?')!r}: {z_range}.")
    obstacle = {
        'name': source['name'], 'x': _pair(source['x']), 'y': _pair(source['y']), 'z': z_range,
    }
    if has_height:
        obstacle['height'] = float(source['height'])
    return obstacle


OBSTACLES = [_obstacle(source) for source in _arena['obstacles']]
GOAL = {axis: _pair(_arena['goal'][axis]) for axis in ('x', 'y', 'z')}
DETERMINISTIC_VIA_POINTS = [_pair(point) for point in _cfg['deterministic_path']['via_points']]

_uncertainty = _cfg['uncertainty']
UNCERTAINTY_MODEL = str(_uncertainty['model'])
INITIAL_VARIANCE = float(_uncertainty['initial_variance'])


def _fan_xyz_values(key: str) -> MappingProxyType:
    values = {
        int(fan): tuple(float(component) for component in vector)
        for fan, vector in _uncertainty[key].items()
    }
    if set(values) != set(VALID_FANS):
        raise ValueError(f'uncertainty.{key} must define {VALID_FANS}.')
    if any(len(vector) != 3 for vector in values.values()):
        raise ValueError(f'uncertainty.{key} values must be XYZ triples.')
    if not all(math.isfinite(value) for vector in values.values() for value in vector):
        raise ValueError(f'uncertainty.{key} must contain only finite values.')
    return MappingProxyType(values)


STATIONARY_RESIDUAL_VARIANCE_PER_FAN = _fan_xyz_values(
    'stationary_residual_variance_per_fan'
)
RESIDUAL_MEAN_PER_FAN = _fan_xyz_values('residual_mean_per_fan')
RESPONSE_ENABLED_AXES = tuple(bool(value) for value in _uncertainty['response_enabled_axes'])
RESPONSE_TIME_CONSTANT = tuple(
    float(value) for value in _uncertainty['response_time_constant']
)

if UNCERTAINTY_MODEL != 'first_order_response_stationary_residual_per_fan':
    raise ValueError(f'Unsupported Crazyflie uncertainty model {UNCERTAINTY_MODEL!r}.')
if not math.isfinite(INITIAL_VARIANCE) or INITIAL_VARIANCE != 0.0:
    raise ValueError('uncertainty.initial_variance must be exactly 0.0.')
if len(RESPONSE_ENABLED_AXES) != 3 or len(RESPONSE_TIME_CONSTANT) != 3:
    raise ValueError('Response enabled flags and time constants must be XYZ triples.')
if not all(math.isfinite(value) and value >= 0.0 for value in RESPONSE_TIME_CONSTANT):
    raise ValueError('Response time constants must be finite and nonnegative.')
if any(enabled != (tau > 0.0) for enabled, tau in zip(
    RESPONSE_ENABLED_AXES, RESPONSE_TIME_CONSTANT,
)):
    raise ValueError(
        'Enabled response axes require a positive time constant; disabled axes require 0.0.'
    )
if any(
    value < 0.0
    for vector in STATIONARY_RESIDUAL_VARIANCE_PER_FAN.values()
    for value in vector
):
    raise ValueError('Stationary residual variances must be nonnegative.')

_planner = _cfg['planner']
T = int(_planner['T'])
DT = float(_planner['dt'])
U_MAX = float(_planner['u_max'])
PLANNER_ALPHA = float(_planner['alpha'])
PLANNER_CONFIG = {**_planner['config'], 'alpha': PLANNER_ALPHA}

_flight = _cfg['flight']
TAKEOFF_VELOCITY = float(_flight['takeoff_velocity'])
LANDING_VELOCITY = float(_flight['landing_velocity'])
LANDING_SETTLE_SECONDS = float(_flight['landing_settle_seconds'])
LOG_SAMPLE_HZ = int(_flight['log_sample_hz'])
ARMING_WAIT_SECONDS = float(_flight['arming_wait_seconds'])
START_POSITION_TOLERANCE = float(_flight['start_position_tolerance'])
START_POSITION_TIMEOUT = float(_flight['start_position_timeout'])
START_SETTLE_SECONDS = float(_flight['start_settle_seconds'])
RETURN_Z = float(_flight['return_z'])
LAND_Z = float(_flight['land_z'])
DRONE_URI = _flight.get('drone_uri')
if not all(math.isfinite(value) and value > 0.0 for value in (
    START_POSITION_TOLERANCE, START_POSITION_TIMEOUT, START_SETTLE_SECONDS,
)):
    raise ValueError('Flight start-position settings must be finite and positive.')

_estimator = _flight['estimator']
ESTIMATOR_SAMPLE_PERIOD_MS = int(_estimator['sample_period_ms'])
ESTIMATOR_SAMPLE_COUNT = int(_estimator['sample_count'])
ESTIMATOR_SPREAD_LIMIT = float(_estimator['spread_limit'])
ESTIMATOR_SETTLE_TIMEOUT = float(_estimator['settle_timeout'])
ESTIMATOR_RESET_WAIT_SECONDS = float(_estimator['reset_wait_seconds'])
ESTIMATOR_MIN_BASE_STATIONS = int(_estimator['min_base_stations'])

LOGS_2D_DIR = EXPERIMENT_DIR / 'logs' / '2d'
LOGS_3D_DIR = EXPERIMENT_DIR / 'logs' / '3d'

_calibration = _cfg['calibration']
CALIBRATION_CAMPAIGN = str(_calibration['active_campaign'])
CALIBRATION_PILOT_RUNS = int(_calibration['pilot_runs'])
CALIBRATION_FINAL_RUNS = int(_calibration['final_runs_per_fan'])

if CALIBRATION_CAMPAIGN not in ('pilot', 'final'):
    raise ValueError("calibration.active_campaign must be 'pilot' or 'final'.")


def flight_profile_signature(scenario: str) -> str:
    """Hash the physical path, controller, estimator, logging, and obstacles."""
    return _flight_profile_signature(_cfg, scenario)


def logs_directory(scenario: str) -> pathlib.Path:
    return LOGS_3D_DIR if scenario == 'figure8' else LOGS_2D_DIR


_figure8 = _cfg['figure8']
FIG8_X_BOUNDS = list(_figure8['workspace']['x'])
FIG8_Y_BOUNDS = list(_figure8['workspace']['y'])
FIG8_Z_BOUNDS = list(_figure8['workspace']['z'])
FIG8_CHANCE_X_BOUNDS = list(_figure8['chance_workspace']['x'])
FIG8_CHANCE_Y_BOUNDS = list(_figure8['chance_workspace']['y'])
FIG8_CHANCE_Z_BOUNDS = list(_figure8['chance_workspace']['z'])
FIG8_CENTER_X = float(_figure8['path']['center_x'])
FIG8_CENTER_Y = float(_figure8['path']['center_y'])
FIG8_HALF_WIDTH = float(_figure8['path']['half_width'])
FIG8_VERTICAL_HALF = float(_figure8['path']['vertical_half'])
FIG8_BASE_HEIGHT = float(_figure8['path']['base_height'])
FIG8_CROSSING_HEIGHT = float(_figure8['path']['crossing_height'])
FIG8_TOP_HEIGHT = float(_figure8['path']['top_height'])
FIG8_DETERMINISTIC_CRUISE_VELOCITY = float(_figure8['deterministic_cruise_velocity'])
FIG8_RETURN_TOLERANCE = float(_figure8['return_tolerance'])
FIG8_T = int(_figure8['T'])
FIG8_FLIGHT_POINTS = int(_figure8['flight_points'])
FIG8_PLOT_POINTS = int(_figure8['plot_points'])
FIG8_W_PHI = float(_figure8['planner_extra']['w_phi'])
FIG8_W_REF_XY = float(_figure8['planner_extra']['w_ref_xy'])
FIG8_W_REF_Z = float(_figure8['planner_extra']['w_ref_z'])
FIG8_W_TERMINAL = float(_figure8['planner_extra']['w_terminal'])
FIG8_W_CORRIDOR = float(_figure8['planner_extra']['w_corridor'])
FIG8_CORRIDOR_RADIUS = float(_figure8['planner_extra']['corridor_radius'])
FIG8_MAX_CONTROL_DELTA = float(_figure8['planner_extra']['max_control_delta'])
FIG8_OBSTACLES = [_obstacle(source) for source in _figure8['obstacles']]

if not 0.0 < FIG8_DETERMINISTIC_CRUISE_VELOCITY <= U_MAX:
    raise ValueError(
        'figure8.deterministic_cruise_velocity must be positive and no greater '
        f'than planner.u_max ({U_MAX} m/s).'
    )
if not FIG8_BASE_HEIGHT < FIG8_CROSSING_HEIGHT < FIG8_TOP_HEIGHT:
    raise ValueError(
        'Figure-eight heights must satisfy base_height < crossing_height < top_height.'
    )


def plan_config_signature(fan: int, scenario: str) -> str:
    if fan not in VALID_FANS or scenario not in VALID_SCENARIOS:
        raise ValueError(f'Invalid plan target: fan={fan}, scenario={scenario!r}.')
    scenario_config = {
        'baseline': {'arena': _cfg['arena'], 'path': _cfg['deterministic_path']},
        'figure8': {'figure8': _cfg['figure8']},
    }[scenario]
    payload = {
        'schema_version': PLAN_SCHEMA_VERSION,
        'fan': fan,
        'scenario': scenario,
        'uncertainty': {
            'model': _uncertainty['model'],
            'initial_variance': float(_uncertainty['initial_variance']),
            'response_enabled_axes': list(_uncertainty['response_enabled_axes']),
            'response_time_constant': list(_uncertainty['response_time_constant']),
            'stationary_residual_variance': list(
                _uncertainty['stationary_residual_variance_per_fan'][fan]
            ),
            'residual_mean': list(_uncertainty['residual_mean_per_fan'][fan]),
            'source_report': _uncertainty.get('source_report'),
        },
        'planner': _planner,
        'scenario_config': scenario_config,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def uncertainty_metadata(fan: int) -> dict[str, Any]:
    """Return the signed uncertainty inputs stored with a generated plan."""
    if fan not in VALID_FANS:
        raise ValueError(f'Invalid fan setting {fan}.')
    return {
        'uncertainty_model': UNCERTAINTY_MODEL,
        'initial_variance': INITIAL_VARIANCE,
        'response_enabled_axes': list(RESPONSE_ENABLED_AXES),
        'response_time_constant': list(RESPONSE_TIME_CONSTANT),
        'stationary_residual_variance': list(
            STATIONARY_RESIDUAL_VARIANCE_PER_FAN[fan]
        ),
        'residual_mean': list(RESIDUAL_MEAN_PER_FAN[fan]),
        'source_report': _uncertainty.get('source_report'),
    }


def validate_waypoints(
    waypoints: list[tuple[float, float, float]],
    *,
    scenario: str = 'baseline',
    label: str = 'Waypoint',
    tolerance: float = 0.01,
) -> None:
    """Reject waypoints outside the scenario workspace.

    `tolerance` absorbs sub-centimetre optimiser/float noise at the
    boundary -- the figure8 curve's own nominal path touches the workspace
    box exactly (its lowest point sits at both `workspace.y` and
    `workspace.z`'s lower bound), so a strict inequality would spuriously
    reject a numerically-clean plan.
    """
    if scenario == 'figure8':
        bounds = (FIG8_X_BOUNDS, FIG8_Y_BOUNDS, FIG8_Z_BOUNDS)
    elif scenario == 'baseline':
        bounds = (FLIGHT_X_BOUNDS, FLIGHT_Y_BOUNDS, FLIGHT_Z_BOUNDS)
    else:
        raise ValueError(f'Unknown scenario {scenario!r}.')
    outside = [
        (index, x, y, z)
        for index, (x, y, z) in enumerate(waypoints)
        if not all(
            low - tolerance <= value <= high + tolerance
            for value, (low, high) in zip((x, y, z), bounds)
        )
    ]
    if outside:
        details = ', '.join(
            f'#{index}=({x:.3f}, {y:.3f}, {z:.3f})' for index, x, y, z in outside
        )
        raise ValueError(f'{label}(s) outside {scenario} workspace: {details}')


def validate_waypoint_velocities(
    waypoints: list[tuple[float, float, float]],
    *,
    start: tuple[float, float, float],
    dt: float = DT,
    u_max: float = U_MAX,
    label: str = 'Leg',
) -> None:
    points = [start, *waypoints]
    invalid = [
        (index, math.dist(points[index], points[index + 1]) / dt)
        for index in range(len(points) - 1)
        if math.dist(points[index], points[index + 1]) / dt > u_max + 1e-9
    ]
    if invalid:
        details = ', '.join(f'#{index}->#{index + 1}={speed:.3f} m/s' for index, speed in invalid)
        raise ValueError(f'{label}(s) exceed u_max={u_max} m/s given dt={dt}s: {details}')


def needs_safe_transit(current: Waypoint, start: Waypoint) -> bool:
    """Whether reaching the mission start requires the obstacle-clearance altitude."""
    return math.dist(current[:2], start[:2]) > START_TOLERANCE


def build_return_legs(
    waypoints: list[Waypoint], start: Waypoint,
) -> list[tuple[Waypoint, float]]:
    """Build reverse legs along the already-validated flyable mission path."""
    previous = waypoints[-1]
    if math.dist(previous[:2], start[:2]) <= START_TOLERANCE:
        return []
    legs = []
    for waypoint in reversed([start, *waypoints[:-1]]):
        if waypoint == previous:
            continue
        velocity = math.dist(previous, waypoint) / DT
        if velocity > U_MAX + 1e-6:
            raise ValueError(f'Return leg requires {velocity:.3f} m/s; limit is {U_MAX} m/s.')
        legs.append((waypoint, velocity))
        previous = waypoint
    return legs


def draw_environment_2d(axis, environment) -> None:
    import matplotlib.patches as patches

    x_bounds, y_bounds = FLIGHT_X_BOUNDS, FLIGHT_Y_BOUNDS
    axis.set(xlim=(x_bounds[0] - 0.1, x_bounds[1] + 0.1),
             ylim=(y_bounds[0] - 0.1, y_bounds[1] + 0.1),
             xlabel='x [m]', ylabel='y [m]')
    axis.set_aspect('equal')
    axis.grid(True, alpha=0.3)
    axis.add_patch(patches.Rectangle(
        (x_bounds[0], y_bounds[0]), x_bounds[1] - x_bounds[0], y_bounds[1] - y_bounds[0],
        facecolor='none', edgecolor='black', linestyle='dashed', alpha=0.4,
    ))
    regions = [(item, 'red') for item in environment.obstacles]
    if environment.goal:
        regions.append((environment.goal, 'green'))
    for region, color in regions:
        x_range, y_range = region['x'], region['y']
        axis.add_patch(patches.Rectangle(
            (x_range[0], y_range[0]), x_range[1] - x_range[0], y_range[1] - y_range[0],
            facecolor=color, edgecolor=color, alpha=0.3,
        ))


def _draw_box(axis, region: dict, color: str) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    x0, x1 = region['x']
    y0, y1 = region['y']
    z0, z1 = region.get('z', FLIGHT_Z_BOUNDS)
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    faces = [corners[index] for index in (
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4],
    )]
    axis.add_collection3d(Poly3DCollection(faces, facecolor=color, edgecolor=color, alpha=0.3))


def draw_environment_3d(axis, environment, *, fit_points: np.ndarray | None = None) -> None:
    """Draw obstacles/goal/bounds in 3D, framed tightly around the region
    actually in play (those regions plus any given trajectory points) rather
    than the full configured workspace, which is mostly empty margin.
    """
    from visualization.style import PALETTE

    regions = list(environment.obstacles)
    if environment.goal:
        regions.append(environment.goal)
    regions.extend(getattr(environment, 'time_windowed_bounds', []))

    xs = [v for r in regions for v in r['x']]
    ys = [v for r in regions for v in r['y']]
    zs = [v for r in regions for v in r.get('z', FLIGHT_Z_BOUNDS)]
    if fit_points is not None and len(fit_points):
        xs += [float(fit_points[:, 0].min()), float(fit_points[:, 0].max())]
        ys += [float(fit_points[:, 1].min()), float(fit_points[:, 1].max())]
        zs += [float(fit_points[:, 2].min()), float(fit_points[:, 2].max())]
    if not xs:
        bounds = environment.bounds or {}
        xs, ys = bounds.get('x', FLIGHT_X_BOUNDS), bounds.get('y', FLIGHT_Y_BOUNDS)
        zs = bounds.get('z', FLIGHT_Z_BOUNDS)

    margin = 0.1
    x_lim = (min(xs) - margin, max(xs) + margin)
    y_lim = (min(ys) - margin, max(ys) + margin)
    z_lim = (min(zs) - margin, max(zs) + margin)
    axis.set(xlim=x_lim, ylim=y_lim, zlim=z_lim,
             xlabel='x [m]', ylabel='y [m]', zlabel='z [m]')
    # Shape the 3D box to the data's own proportions -- mplot3d's default is
    # a cube regardless of axis ranges, which pads a wide/flat scene (e.g.
    # y spans 2x x, z a fraction of both) with a lot of visibly empty volume.
    axis.set_box_aspect((
        x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0],
    ))

    # mplot3d's default pane fill reads as a heavy gray box; flatten it so
    # the trajectory, not the chrome, draws the eye.
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        pane.set_edgecolor((0.0, 0.0, 0.0, 0.2))
    axis.grid(True, alpha=0.15)

    for obstacle in environment.obstacles:
        _draw_box(axis, obstacle, PALETTE['obs_static']['stroke'])
    if environment.goal:
        _draw_box(axis, environment.goal, PALETTE['goal']['stroke'])


def shutdown_hardware(position_commander: Any, hardware: Any, *, airborne: bool,
                      landing_velocity: float) -> None:
    try:
        if airborne and position_commander is not None:
            position_commander.land(velocity=landing_velocity)
    finally:
        try:
            hardware.disarm()
        finally:
            hardware.disconnect()


@dataclass(frozen=True)
class StoredPlan:
    fan: int
    scenario: str
    waypoints: tuple[tuple[float, float, float], ...]
    metadata: dict[str, Any]


class PlanRepository:
    def __init__(self, directory: pathlib.Path | None = None) -> None:
        self.directory = directory or EXPERIMENT_DIR / 'waypoints'

    def path(self, fan: int, scenario: str = 'baseline') -> pathlib.Path:
        suffix = '' if scenario == 'baseline' else f'_{scenario}'
        return self.directory / f'pdstl{suffix}_fan{fan}.json'

    def save(self, fan: int, scenario: str, waypoints, metadata: dict[str, Any]) -> pathlib.Path:
        path = self.path(fan, scenario)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **metadata,
            'schema_version': PLAN_SCHEMA_VERSION,
            'config_signature': plan_config_signature(fan, scenario),
            'fan': fan,
            'scenario': scenario,
            'waypoints': [list(map(float, waypoint)) for waypoint in waypoints],
        }
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        return path

    def load(self, fan: int, scenario: str = 'baseline') -> StoredPlan:
        path = self.path(fan, scenario)
        if not path.exists():
            raise FileNotFoundError(f'No plan at {path}; run ACTION = "plan" first.')
        data = json.loads(path.read_text(encoding='utf-8'))
        stored_scenario = data.get('scenario', 'baseline')
        if data.get('fan') != fan or stored_scenario != scenario:
            raise ValueError('Plan fan or scenario does not match the selected run.')
        waypoints = tuple(tuple(map(float, point)) for point in data['waypoints'])
        return StoredPlan(fan, scenario, waypoints, {k: v for k, v in data.items() if k != 'waypoints'})

    def require_flyable(self, fan: int, scenario: str = 'baseline') -> StoredPlan:
        plan = self.load(fan, scenario)
        metadata = plan.metadata
        if metadata.get('schema_version') != PLAN_SCHEMA_VERSION:
            raise RuntimeError('Legacy plan; regenerate it before flight.')
        if metadata.get('config_signature') != plan_config_signature(fan, scenario):
            raise RuntimeError('Stale plan; regenerate it before flight.')
        expected_uncertainty = uncertainty_metadata(fan)
        if any(metadata.get(key) != value for key, value in expected_uncertainty.items()):
            raise RuntimeError('Plan uncertainty does not match config.yml.')
        if not metadata.get('source_report'):
            raise RuntimeError(
                'Plan uncertainty is provisional; accept a final covariance report '
                'and set uncertainty.source_report before flight.'
            )
        rho_after = float(metadata.get('rho_after', 0.0))
        if not math.isfinite(rho_after) or rho_after < PLANNER_ALPHA:
            raise RuntimeError(
                f'Plan satisfaction {rho_after:.4f} is below alpha={PLANNER_ALPHA:.4f}.'
            )
        if scenario == 'figure8':
            required_geometry = {
                'return_error', 'return_tolerance', 'corridor_radius',
                'max_reference_deviation', 'max_control_delta',
                'max_control_delta_limit',
            }
            missing = sorted(required_geometry - metadata.keys())
            if missing:
                raise RuntimeError(
                    f'Figure-eight plan lacks geometry validation metadata: {missing}.'
                )
            return_error = math.dist(plan.waypoints[0], plan.waypoints[-1])
            if return_error > FIG8_RETURN_TOLERANCE:
                raise RuntimeError(
                    f'Figure-eight plan return error {return_error:.3f} m exceeds '
                    f'{FIG8_RETURN_TOLERANCE:.3f} m.'
                )
            if float(metadata['max_reference_deviation']) > FIG8_CORRIDOR_RADIUS:
                raise RuntimeError('Figure-eight plan leaves its reference corridor.')
            if float(metadata['max_control_delta']) > FIG8_MAX_CONTROL_DELTA:
                raise RuntimeError('Figure-eight plan exceeds its smoothness limit.')
        return plan
