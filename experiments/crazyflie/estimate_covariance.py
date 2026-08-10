#!/usr/bin/env python3
"""Fit response-aware figure-eight tracking uncertainty from real flights.

Pilot reports are diagnostic only. Final reports require the configured number
of valid runs for every fan and expose conservative residual-covariance bounds
only when both the mean-response and residual-stationarity diagnostics pass.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import yaml

try:  # Package import during tests and ``python -m``.
    from .components.profile import flight_profile_payload, flight_profile_signature
except ImportError:  # Direct ``python experiments/crazyflie/estimate_covariance.py``.
    from components.profile import flight_profile_payload, flight_profile_signature


EXPERIMENT_DIR = pathlib.Path(__file__).resolve().parent
LOGS_DIR = EXPERIMENT_DIR / 'logs' / '3d'
_CONFIG_PATH = EXPERIMENT_DIR / 'components' / 'config.yml'
_CONFIG = yaml.safe_load(_CONFIG_PATH.read_text(encoding='utf-8'))

MODEL_NAME = 'first_order_response_stationary_residual_per_fan'
VALID_FANS = tuple(
    int(fan) for fan in _CONFIG['uncertainty']['stationary_residual_variance_per_fan']
)
FIG8_FLIGHT_POINTS = int(_CONFIG['figure8']['flight_points'])
LOG_SAMPLE_HZ = int(_CONFIG['flight']['log_sample_hz'])
PROFILE_SIGNATURE = flight_profile_signature(_CONFIG, 'figure8')
PROFILE_PAYLOAD = flight_profile_payload(_CONFIG, 'figure8')

_UNCERTAINTY = _CONFIG['uncertainty']
RESPONSE_ENABLED_AXES = tuple(bool(value) for value in _UNCERTAINTY['response_enabled_axes'])

_CALIBRATION = _CONFIG['calibration']
ACTIVE_CAMPAIGN = str(_CALIBRATION['active_campaign'])
PILOT_RUNS = int(_CALIBRATION['pilot_runs'])
FINAL_RUNS_PER_FAN = int(_CALIBRATION['final_runs_per_fan'])
BOOTSTRAP_SAMPLES = int(_CALIBRATION['bootstrap_samples'])
BOOTSTRAP_SEED = int(_CALIBRATION['bootstrap_seed'])
STATIONARITY_BIN_SIZE = int(_CALIBRATION['stationarity_bin_size'])
MAX_STATIONARITY_RATIO = float(_CALIBRATION['max_stationarity_ratio'])
RESPONSE_TIME_CONSTANT_BOUNDS = tuple(
    float(value) for value in _CALIBRATION['response_time_constant_bounds']
)
MIN_RESPONSE_R_SQUARED = float(_CALIBRATION['minimum_response_r_squared'])
MIN_BOOTSTRAP_VALID_FRACTION = float(
    _CALIBRATION['minimum_valid_bootstrap_fraction']
)
PLAUSIBLE_POSITION_MARGIN = float(_CALIBRATION['plausible_position_margin'])

AXES = ('x', 'y', 'z')

_SCENARIO = 'figure8'
_CONDITION = 'deterministic'
_TERMINAL_HOLD_TOLERANCE = 1.0 / LOG_SAMPLE_HZ


@dataclass(frozen=True)
class AlignedRun:
    """Command and interpolated actual positions at the same arrival times."""

    commanded_t: np.ndarray
    commanded_xyz: np.ndarray
    actual_xyz: np.ndarray

    @property
    def tracking_error(self) -> np.ndarray:
        return self.actual_xyz - self.commanded_xyz


def log_prefix(condition: str, scenario: str, fan: int) -> str:
    suffix = '' if scenario == 'baseline' else f'_{scenario}'
    return f'{condition}{suffix}_fan{fan:02d}_run'


def _read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(newline='') as fh:
        return list(csv.DictReader(fh))


def discover_runs(fan: int) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """Discover every attempted run, including incomplete one-sided pairs."""
    prefix = log_prefix(_CONDITION, _SCENARIO, fan)
    attempts: dict[str, dict[str, pathlib.Path]] = {}
    for kind in ('commanded', 'actual'):
        suffix = f'_{kind}.csv'
        for path in LOGS_DIR.glob(f'{prefix}*{suffix}'):
            stem = path.name.removesuffix(suffix)
            attempts.setdefault(stem, {})[kind] = path

    return [
        (
            paths.get('commanded', LOGS_DIR / f'{stem}_commanded.csv'),
            paths.get('actual', LOGS_DIR / f'{stem}_actual.csv'),
        )
        for stem, paths in sorted(attempts.items())
    ]


def _uniform_field(rows: list[dict], field: str, expected: str, label: str) -> str | None:
    values = {row.get(field) for row in rows}
    if values != {expected}:
        shown = sorted(repr(value) for value in values)
        return f'{label} {field} mismatch: expected {expected!r}, found {shown}'
    return None


def _validate_provenance(rows: list[dict], label: str) -> str | None:
    for field, expected in (
        ('condition', _CONDITION),
        ('scenario', _SCENARIO),
        ('campaign', ACTIVE_CAMPAIGN),
        ('profile_signature', PROFILE_SIGNATURE),
    ):
        if reason := _uniform_field(rows, field, expected, label):
            return reason
    return None


def _plausible_actual_positions(actual_xyz: np.ndarray) -> bool:
    workspace = _CONFIG['figure8']['workspace']
    bounds = np.asarray([workspace[axis] for axis in AXES], dtype=float)
    lower = bounds[:, 0] - PLAUSIBLE_POSITION_MARGIN
    upper = bounds[:, 1] + PLAUSIBLE_POSITION_MARGIN
    return bool(np.all((actual_xyz >= lower) & (actual_xyz <= upper)))


def load_aligned_run(
    commanded_path: pathlib.Path,
    actual_path: pathlib.Path,
) -> tuple[AlignedRun | None, str | None]:
    """Return aligned command/actual samples or a precise exclusion reason."""
    if not commanded_path.exists():
        return None, 'missing commanded file'
    if not actual_path.exists():
        return None, 'missing actual file'
    if '_CRASH' in actual_path.name or '_CRASH' in commanded_path.name:
        return None, 'run marked CRASH'
    if '_VIOLATION' in actual_path.name or '_VIOLATION' in commanded_path.name:
        return None, 'run marked VIOLATION'

    commanded_rows = _read_csv(commanded_path)
    actual_rows = _read_csv(actual_path)
    if not commanded_rows:
        return None, 'empty commanded file'
    if len(actual_rows) < 2:
        return None, 'actual log has too few samples to interpolate'
    if reason := _validate_provenance(commanded_rows, 'commanded'):
        return None, reason
    if reason := _validate_provenance(actual_rows, 'actual'):
        return None, reason
    if any(str(row.get('safe')) != '1' for row in commanded_rows):
        return None, 'commanded trajectory contains an unsafe waypoint'
    if any(str(row.get('safe')) != '1' for row in actual_rows):
        return None, 'actual trajectory contains an unsafe sample'

    mission_rows = commanded_rows[1:]  # Drop the pre-mission hover-at-start row.
    if len(mission_rows) != FIG8_FLIGHT_POINTS:
        return None, (
            f'incomplete trajectory: {len(mission_rows)} mission waypoints logged, '
            f'expected {FIG8_FLIGHT_POINTS}'
        )

    try:
        commanded_t = np.asarray([float(row['t']) for row in mission_rows])
        commanded_xyz = np.asarray([
            [float(row[axis]) for axis in AXES] for row in mission_rows
        ])
        actual_t = np.asarray([float(row['t']) for row in actual_rows])
        actual_xyz = np.asarray([
            [float(row[axis]) for axis in AXES] for row in actual_rows
        ])
    except (KeyError, ValueError):
        return None, 'invalid numeric log data'

    if not all(np.all(np.isfinite(values)) for values in (
        commanded_t, commanded_xyz, actual_t, actual_xyz,
    )):
        return None, 'non-finite log data'
    if np.any(np.diff(commanded_t) <= 0):
        return None, 'commanded timestamps are not strictly increasing'
    if not _plausible_actual_positions(actual_xyz):
        return None, 'actual position outside plausible workspace envelope'

    order = np.argsort(actual_t)
    actual_t = actual_t[order]
    actual_xyz = actual_xyz[order]
    if np.any(np.diff(actual_t) <= 0):
        return None, 'actual timestamps are not strictly increasing'

    t_min, t_max = actual_t[0], actual_t[-1]
    aligned_actual = np.empty_like(commanded_xyz)
    for k, t in enumerate(commanded_t):
        terminal_hold = (
            k == FIG8_FLIGHT_POINTS - 1
            and t_max < t <= t_max + _TERMINAL_HOLD_TOLERANCE
        )
        if not (t_min <= t <= t_max) and not terminal_hold:
            return None, (
                f'waypoint {k} arrival time {t:.3f}s outside actual log range '
                f'[{t_min:.3f}, {t_max:.3f}]s -- tracking loss or truncated log'
            )
        aligned_actual[k] = actual_xyz[-1] if terminal_hold else np.asarray([
            np.interp(t, actual_t, actual_xyz[:, axis]) for axis in range(3)
        ])
    return AlignedRun(commanded_t, commanded_xyz, aligned_actual), None


def load_run(
    commanded_path: pathlib.Path,
    actual_path: pathlib.Path,
) -> tuple[np.ndarray | None, str | None]:
    """Compatibility wrapper returning raw per-waypoint tracking errors."""
    run, reason = load_aligned_run(commanded_path, actual_path)
    return (None if run is None else run.tracking_error), reason


def align_run_data(
    fan: int,
) -> tuple[list[AlignedRun], list[str], list[tuple[str, str]]]:
    """Load all valid runs and account for every excluded attempt."""
    included: list[str] = []
    excluded: list[tuple[str, str]] = []
    runs: list[AlignedRun] = []
    for commanded_path, actual_path in discover_runs(fan):
        run_path = actual_path if actual_path.exists() else commanded_path
        run_id = run_path.stem.removesuffix('_actual').removesuffix('_commanded')
        run, reason = load_aligned_run(commanded_path, actual_path)
        if run is None:
            excluded.append((run_id, reason or 'unknown exclusion'))
        else:
            runs.append(run)
            included.append(run_id)
    return runs, included, excluded


def align_runs(fan: int) -> tuple[np.ndarray, list[str], list[tuple[str, str]]]:
    """Compatibility view of aligned runs as raw tracking-error tensors."""
    runs, included, excluded = align_run_data(fan)
    errors = (
        np.stack([run.tracking_error for run in runs])
        if runs else np.empty((0, FIG8_FLIGHT_POINTS, 3))
    )
    return errors, included, excluded


def predict_response(
    commanded_t: np.ndarray,
    commanded_xyz: np.ndarray,
    response_time_constant: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Predict mean actual position under first-order velocity response.

    For tau > 0, ``dv/dt = (u-v)/tau`` and ``dp/dt = v`` are solved exactly
    under zero-order-held commanded velocity. Tau = 0 selects the commanded
    position directly for that axis.
    """
    times = np.asarray(commanded_t, dtype=float)
    commands = np.asarray(commanded_xyz, dtype=float)
    tau = np.asarray(response_time_constant, dtype=float)
    if times.ndim != 1 or commands.shape != (len(times), 3):
        raise ValueError('commanded data must have shapes [K] and [K, 3].')
    if tau.shape != (3,) or not np.all(np.isfinite(tau)) or np.any(tau < 0):
        raise ValueError('response_time_constant must be a finite nonnegative XYZ vector.')
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('commanded timestamps must be strictly increasing.')

    predicted = np.empty_like(commands)
    predicted[0] = commands[0]
    velocity = np.zeros(3, dtype=float)
    enabled = tau > 0
    for step in range(1, len(times)):
        dt = times[step] - times[step - 1]
        control = (commands[step] - commands[step - 1]) / dt
        predicted[step, ~enabled] = commands[step, ~enabled]
        velocity[~enabled] = control[~enabled]
        if np.any(enabled):
            decay = np.exp(-dt / tau[enabled])
            displacement = (
                tau[enabled] * (1.0 - decay) * velocity[enabled]
                + (dt - tau[enabled] * (1.0 - decay)) * control[enabled]
            )
            predicted[step, enabled] = predicted[step - 1, enabled] + displacement
            velocity[enabled] = (
                decay * velocity[enabled] + (1.0 - decay) * control[enabled]
            )
    return predicted


def _bounded_log_minimum(function, lower: float, upper: float) -> float:
    """Dependency-free bounded scalar minimization in log time-constant space."""
    left, right = math.log(lower), math.log(upper)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1, f2 = function(x1), function(x2)
    for _ in range(36):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = function(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = function(x2)
    return math.exp((left + right) / 2.0)


def _fan_centered_sse(values_by_fan: dict[int, list[np.ndarray]], axis: int) -> float:
    total = 0.0
    for values in values_by_fan.values():
        flattened = np.concatenate([value[:, axis] for value in values])
        total += float(np.sum((flattened - np.mean(flattened)) ** 2))
    return total


def fit_response_model(
    runs_by_fan: dict[int, list[AlignedRun]],
) -> dict[str, Any]:
    """Fit one shared per-axis response and fan-specific residual offsets."""
    available = {fan: runs for fan, runs in runs_by_fan.items() if runs}
    if not available:
        raise ValueError('At least one aligned run is required to fit the response model.')
    lower, upper = RESPONSE_TIME_CONSTANT_BOUNDS
    if not 0.0 < lower < upper:
        raise ValueError('Response time-constant bounds must satisfy 0 < lower < upper.')

    fitted_tau = np.zeros(3, dtype=float)
    for axis, enabled in enumerate(RESPONSE_ENABLED_AXES):
        if not enabled:
            continue

        def objective(log_tau: float) -> float:
            tau = np.zeros(3, dtype=float)
            tau[axis] = math.exp(log_tau)
            residuals = {
                fan: [
                    run.actual_xyz - predict_response(run.commanded_t, run.commanded_xyz, tau)
                    for run in runs
                ]
                for fan, runs in available.items()
            }
            return _fan_centered_sse(residuals, axis)

        fitted_tau[axis] = _bounded_log_minimum(objective, lower, upper)

    raw_errors = {
        fan: [run.tracking_error for run in runs] for fan, runs in available.items()
    }
    residuals = {
        fan: [
            run.actual_xyz
            - predict_response(run.commanded_t, run.commanded_xyz, fitted_tau)
            for run in runs
        ]
        for fan, runs in available.items()
    }
    raw_sse = np.asarray([_fan_centered_sse(raw_errors, axis) for axis in range(3)])
    residual_sse = np.asarray([_fan_centered_sse(residuals, axis) for axis in range(3)])
    r_squared = np.ones(3, dtype=float)
    for axis in range(3):
        if raw_sse[axis] <= np.finfo(float).eps:
            r_squared[axis] = 1.0 if residual_sse[axis] <= np.finfo(float).eps else -math.inf
        else:
            r_squared[axis] = 1.0 - residual_sse[axis] / raw_sse[axis]

    acceptable = all(
        not enabled or (
            math.isfinite(fitted_tau[axis])
            and lower <= fitted_tau[axis] <= upper
            and math.isfinite(r_squared[axis])
            and r_squared[axis] >= MIN_RESPONSE_R_SQUARED
        )
        for axis, enabled in enumerate(RESPONSE_ENABLED_AXES)
    )
    return {
        'enabled_axes': {
            axis: bool(enabled) for axis, enabled in zip(AXES, RESPONSE_ENABLED_AXES)
        },
        'time_constant': fitted_tau.tolist(),
        'r_squared': r_squared.tolist(),
        'minimum_r_squared': MIN_RESPONSE_R_SQUARED,
        'time_constant_bounds': list(RESPONSE_TIME_CONSTANT_BOUNDS),
        'acceptable': bool(acceptable),
    }


def response_residuals(
    runs: list[AlignedRun], response_model: dict[str, Any],
) -> np.ndarray:
    tau = np.asarray(response_model['time_constant'], dtype=float)
    return np.stack([
        run.actual_xyz - predict_response(run.commanded_t, run.commanded_xyz, tau)
        for run in runs
    ])


def compute_error_statistics(
    errors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return waypoint mean, centered covariance, second moment, and XYZ MSE."""
    if errors.ndim != 3 or errors.shape[0] < 2 or errors.shape[2] != 3:
        raise ValueError('errors must have shape [N>=2, K, 3].')
    mean_error = errors.mean(axis=0)
    centered = errors - mean_error[None, :, :]
    covariance = np.einsum('nki,nkj->kij', centered, centered) / (errors.shape[0] - 1)
    second_moment = np.einsum('nki,nkj->kij', errors, errors) / errors.shape[0]
    bias_inclusive_mse = np.mean(errors**2, axis=0)
    return mean_error, covariance, second_moment, bias_inclusive_mse


def compute_mean_and_covariance(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return waypoint mean error and centered sample covariance."""
    mean_error, covariance, _second_moment, _mse = compute_error_statistics(errors)
    return mean_error, covariance


def planner_residual_statistics(residuals: np.ndarray) -> dict[str, np.ndarray]:
    """Summarize residuals around the single mean used by the planner.

    Waypoint-centered covariance is retained as a repeatability diagnostic, but
    it is not sufficient for planning when the remaining mean changes with
    phase. The planner-facing variance therefore includes deviations of every
    residual sample from the one pooled fan-specific mean stored in config.
    """
    mean, centered_covariance, second_moment, mse = compute_error_statistics(residuals)
    pooled_mean = np.mean(residuals, axis=(0, 1))
    planner_variance_trace = np.mean(
        (residuals - pooled_mean[None, None, :]) ** 2, axis=0,
    )
    stationary_variance = np.mean(planner_variance_trace, axis=0)
    centered_stationary_variance = np.mean(
        np.diagonal(centered_covariance, axis1=1, axis2=2), axis=0,
    )
    phase_mean_variance = np.mean((mean - pooled_mean[None, :]) ** 2, axis=0)
    phase_fraction = np.divide(
        phase_mean_variance,
        stationary_variance,
        out=np.zeros(3, dtype=float),
        where=stationary_variance > np.finfo(float).eps,
    )
    return {
        'mean': mean,
        'centered_covariance': centered_covariance,
        'second_moment': second_moment,
        'bias_inclusive_mse': mse,
        'pooled_mean': pooled_mean,
        'planner_variance_trace': planner_variance_trace,
        'stationary_variance': stationary_variance,
        'centered_stationary_variance': centered_stationary_variance,
        'phase_mean_variance': phase_mean_variance,
        'phase_fraction': phase_fraction,
    }


def stationarity_diagnostics(covariance: np.ndarray) -> dict[str, Any]:
    """Measure whether per-axis residual variance is reasonably stationary."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 3 or covariance.shape[1:] != (3, 3):
        raise ValueError('covariance must have shape [K, 3, 3].')
    variance_trace = np.diagonal(covariance, axis1=1, axis2=2)
    pooled = np.mean(variance_trace, axis=0)
    binned = np.stack([
        np.mean(variance_trace[start:start + STATIONARITY_BIN_SIZE], axis=0)
        for start in range(0, len(variance_trace), STATIONARITY_BIN_SIZE)
    ])
    ratios = np.empty(3, dtype=float)
    for axis in range(3):
        if pooled[axis] <= np.finfo(float).eps:
            ratios[axis] = (
                1.0 if np.max(binned[:, axis]) <= np.finfo(float).eps else math.inf
            )
        else:
            ratios[axis] = np.max(binned[:, axis]) / pooled[axis]
    acceptable = bool(
        np.all(np.isfinite(pooled))
        and np.all(pooled >= 0.0)
        and np.all(np.isfinite(ratios))
        and np.all(ratios <= MAX_STATIONARITY_RATIO)
    )
    return {
        'bin_size': STATIONARITY_BIN_SIZE,
        'binned_variance': binned.tolist(),
        'max_bin_to_pooled_ratio': ratios.tolist(),
        'threshold': MAX_STATIONARITY_RATIO,
        'acceptable': acceptable,
    }


def _calibrate_loaded_fan(
    fan: int,
    runs: list[AlignedRun],
    included: list[str],
    excluded: list[tuple[str, str]],
    response_model: dict[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        'fan': fan,
        'n_runs': len(runs),
        'n_attempts': len(included) + len(excluded),
        'included_runs': included,
        'excluded_runs': excluded,
        'raw_mean_tracking_error': None,
        'raw_bias_inclusive_mse': None,
        'mean_residual': None,
        'centered_residual_covariance': None,
        'residual_second_moment': None,
        'bias_inclusive_residual_mse': None,
        'pooled_residual_mean': None,
        'planner_residual_variance': None,
        'stationary_residual_variance': None,
        'centered_stationary_residual_variance': None,
        'phase_dependent_residual_mean_variance': None,
        'phase_fraction_of_planner_variance': None,
        'residual_stationarity': None,
        '_runs': runs,
        '_residuals': np.empty((0, FIG8_FLIGHT_POINTS, 3)),
    }
    if len(runs) >= 2 and response_model is not None:
        raw_errors = np.stack([run.tracking_error for run in runs])
        residuals = response_residuals(runs, response_model)
        raw_mean, _raw_cov, _raw_second, raw_mse = compute_error_statistics(raw_errors)
        stats = planner_residual_statistics(residuals)
        planner_covariance = np.asarray([
            np.diag(variance) for variance in stats['planner_variance_trace']
        ])
        result.update({
            'raw_mean_tracking_error': raw_mean.tolist(),
            'raw_bias_inclusive_mse': raw_mse.tolist(),
            'mean_residual': stats['mean'].tolist(),
            'centered_residual_covariance': stats['centered_covariance'].tolist(),
            'residual_second_moment': stats['second_moment'].tolist(),
            'bias_inclusive_residual_mse': stats['bias_inclusive_mse'].tolist(),
            'pooled_residual_mean': stats['pooled_mean'].tolist(),
            'planner_residual_variance': stats['planner_variance_trace'].tolist(),
            'stationary_residual_variance': stats['stationary_variance'].tolist(),
            'centered_stationary_residual_variance': (
                stats['centered_stationary_variance'].tolist()
            ),
            'phase_dependent_residual_mean_variance': (
                stats['phase_mean_variance'].tolist()
            ),
            'phase_fraction_of_planner_variance': stats['phase_fraction'].tolist(),
            'residual_stationarity': stationarity_diagnostics(planner_covariance),
            '_residuals': residuals,
        })
    return result


def calibrate_campaign(
    fans: tuple[int, ...] | list[int],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any] | None]:
    loaded = {fan: align_run_data(fan) for fan in fans}
    runs_by_fan = {fan: data[0] for fan, data in loaded.items()}
    response_model = None
    if any(runs_by_fan.values()):
        response_model = fit_response_model(runs_by_fan)
    results = {
        fan: _calibrate_loaded_fan(fan, *loaded[fan], response_model)
        for fan in fans
    }
    return results, response_model


def calibrate_fan(fan: int) -> dict[str, Any]:
    """Calibrate one fan in isolation (primarily useful for diagnostics/tests)."""
    results, _response = calibrate_campaign([fan])
    return results[fan]


def _axis_intervals(samples: list[np.ndarray]) -> dict[str, dict[str, float]] | None:
    if not samples:
        return None
    lower, upper = np.percentile(np.asarray(samples), [2.5, 97.5], axis=0)
    return {
        axis: {'lower': float(lower[index]), 'upper': float(upper[index])}
        for index, axis in enumerate(AXES)
    }


def bootstrap_response_model(
    runs_by_fan: dict[int, list[AlignedRun]],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Resample complete runs, refitting response before residual covariance."""
    if samples < 1:
        raise ValueError('Bootstrap samples must be positive.')
    if not runs_by_fan:
        raise ValueError('At least one fan dataset is required.')
    for fan, runs in runs_by_fan.items():
        if len(runs) < 2:
            raise ValueError(f'Fan {fan} requires at least two aligned runs.')

    rng = np.random.default_rng(seed)
    fans = sorted(runs_by_fan)
    lower, upper = RESPONSE_TIME_CONSTANT_BOUNDS
    tau_grid = np.geomspace(lower, upper, 81)

    # Response prediction depends on the original run and candidate tau, not
    # on which bootstrap sample later selects that run. Cache axis residuals
    # and their sufficient statistics once, then refit each resample in O(GN)
    # instead of repeatedly integrating the response model in Python.
    residual_cache: dict[int, dict[int, np.ndarray]] = {}
    sum_cache: dict[int, dict[int, np.ndarray]] = {}
    sumsq_cache: dict[int, dict[int, np.ndarray]] = {}
    raw_cache: dict[int, np.ndarray] = {
        fan: np.stack([run.tracking_error for run in runs])
        for fan, runs in runs_by_fan.items()
    }
    for axis, enabled in enumerate(RESPONSE_ENABLED_AXES):
        if not enabled:
            continue
        residual_cache[axis] = {}
        sum_cache[axis] = {}
        sumsq_cache[axis] = {}
        for fan, runs in runs_by_fan.items():
            by_tau = []
            for tau_value in tau_grid:
                tau = np.zeros(3, dtype=float)
                tau[axis] = tau_value
                by_tau.append(np.stack([
                    (
                        run.actual_xyz
                        - predict_response(run.commanded_t, run.commanded_xyz, tau)
                    )[:, axis]
                    for run in runs
                ]))
            cached = np.stack(by_tau)  # [G, N, K]
            residual_cache[axis][fan] = cached
            sum_cache[axis][fan] = np.sum(cached, axis=2)
            sumsq_cache[axis][fan] = np.sum(cached**2, axis=2)

    variance_samples: dict[int, list[np.ndarray]] = {fan: [] for fan in fans}
    mean_samples: dict[int, list[np.ndarray]] = {fan: [] for fan in fans}
    tau_samples: list[np.ndarray] = []
    r_squared_samples: list[np.ndarray] = []
    samples_valid = 0
    for _ in range(samples):
        selected: dict[int, np.ndarray] = {}
        for fan in fans:
            runs = runs_by_fan[fan]
            selected[fan] = rng.integers(0, len(runs), size=len(runs))
        try:
            tau = np.zeros(3, dtype=float)
            response_sse = np.zeros(3, dtype=float)
            raw_sse = np.zeros(3, dtype=float)
            grid_indices: dict[int, int] = {}
            for axis, enabled in enumerate(RESPONSE_ENABLED_AXES):
                for fan in fans:
                    raw = raw_cache[fan][selected[fan], :, axis]
                    raw_sse[axis] += float(
                        np.sum(raw**2) - np.sum(raw) ** 2 / raw.size
                    )
                if not enabled:
                    response_sse[axis] = raw_sse[axis]
                    continue
                objective = np.zeros(len(tau_grid), dtype=float)
                for fan in fans:
                    indices = selected[fan]
                    sums = np.sum(sum_cache[axis][fan][:, indices], axis=1)
                    sumsqs = np.sum(sumsq_cache[axis][fan][:, indices], axis=1)
                    count = len(indices) * residual_cache[axis][fan].shape[2]
                    objective += sumsqs - sums**2 / count
                grid_index = int(np.argmin(objective))
                grid_indices[axis] = grid_index
                tau[axis] = tau_grid[grid_index]
                response_sse[axis] = objective[grid_index]

            r_squared = np.ones(3, dtype=float)
            for axis in range(3):
                if raw_sse[axis] <= np.finfo(float).eps:
                    r_squared[axis] = (
                        1.0 if response_sse[axis] <= np.finfo(float).eps else -math.inf
                    )
                else:
                    r_squared[axis] = 1.0 - response_sse[axis] / raw_sse[axis]

            iteration_variances: dict[int, np.ndarray] = {}
            iteration_means: dict[int, np.ndarray] = {}
            for fan in fans:
                indices = selected[fan]
                residual_axes = []
                for axis, enabled in enumerate(RESPONSE_ENABLED_AXES):
                    if enabled:
                        residual_axes.append(
                            residual_cache[axis][fan][grid_indices[axis], indices]
                        )
                    else:
                        residual_axes.append(raw_cache[fan][indices, :, axis])
                residuals = np.stack(residual_axes, axis=2)
                stats = planner_residual_statistics(residuals)
                iteration_variances[fan] = stats['stationary_variance']
                iteration_means[fan] = stats['pooled_mean']
        except (FloatingPointError, ValueError):
            continue
        values = [tau, r_squared, *iteration_variances.values(), *iteration_means.values()]
        if not all(np.all(np.isfinite(value)) for value in values):
            continue
        for fan in fans:
            variance_samples[fan].append(iteration_variances[fan])
            mean_samples[fan].append(iteration_means[fan])
        tau_samples.append(tau)
        r_squared_samples.append(r_squared)
        samples_valid += 1

    return {
        'seed': seed,
        'samples_requested': samples,
        'samples_valid': samples_valid,
        'response_time_constant_ci95': _axis_intervals(tau_samples),
        'response_r_squared_ci95': _axis_intervals(r_squared_samples),
        'stationary_residual_variance_ci95': {
            fan: _axis_intervals(values) for fan, values in variance_samples.items()
        },
        'pooled_residual_mean_ci95': {
            fan: _axis_intervals(values) for fan, values in mean_samples.items()
        },
    }


def _axis_intervals_are_valid(
    intervals: dict[str, dict[str, float]] | None,
    *,
    nonnegative: bool,
) -> bool:
    if intervals is None or set(intervals) != set(AXES):
        return False
    for interval in intervals.values():
        lower, upper = interval.get('lower'), interval.get('upper')
        if not all(math.isfinite(value) for value in (lower, upper)) or lower > upper:
            return False
        if nonnegative and lower < 0.0:
            return False
    return True


def _model_is_acceptable(
    fan_results: dict[int, dict[str, Any]],
    response_model: dict[str, Any] | None,
    bootstrap: dict[str, Any] | None,
) -> bool:
    if (
        response_model is None
        or not response_model.get('acceptable', False)
        or bootstrap is None
        or set(fan_results) != set(VALID_FANS)
    ):
        return False
    requested = int(bootstrap.get('samples_requested', 0))
    valid = int(bootstrap.get('samples_valid', 0))
    if requested < BOOTSTRAP_SAMPLES:
        return False
    if valid < math.ceil(MIN_BOOTSTRAP_VALID_FRACTION * requested):
        return False
    if not _axis_intervals_are_valid(
        bootstrap.get('response_time_constant_ci95'), nonnegative=True,
    ):
        return False
    for fan, result in fan_results.items():
        variance = np.asarray(result.get('stationary_residual_variance'), dtype=float)
        residual_mean = np.asarray(result.get('pooled_residual_mean'), dtype=float)
        stationarity = result.get('residual_stationarity') or {}
        if (
            variance.shape != (3,)
            or residual_mean.shape != (3,)
            or not np.all(np.isfinite(variance))
            or not np.all(variance >= 0.0)
            or not np.all(np.isfinite(residual_mean))
            or not stationarity.get('acceptable', False)
        ):
            return False
        if not _axis_intervals_are_valid(
            bootstrap['stationary_residual_variance_ci95'].get(fan), nonnegative=True,
        ):
            return False
        if not _axis_intervals_are_valid(
            bootstrap['pooled_residual_mean_ci95'].get(fan), nonnegative=False,
        ):
            return False
    return True


def _attempt_accounting_is_complete(result: dict[str, Any]) -> bool:
    included = list(result['included_runs'])
    excluded = [run_id for run_id, _reason in result['excluded_runs']]
    all_ids = included + excluded
    return bool(
        result['n_attempts'] == len(all_ids)
        and result['n_runs'] == len(included)
        and len(all_ids) == len(set(all_ids))
    )


def build_campaign_report(
    mode: str,
    fan_results: dict[int, dict[str, Any]],
    response_model: dict[str, Any] | None,
    bootstrap: dict[str, Any] | None,
) -> dict[str, Any]:
    required_runs = PILOT_RUNS if mode == 'pilot' else FINAL_RUNS_PER_FAN
    required_fans = (2,) if mode == 'pilot' else VALID_FANS
    complete = (
        set(fan_results) == set(required_fans)
        and all(fan_results[fan]['n_runs'] >= required_runs for fan in required_fans)
    )
    attempts_accounted_for = all(
        _attempt_accounting_is_complete(result) for result in fan_results.values()
    )
    pilot_clean = not any(result['excluded_runs'] for result in fan_results.values())
    model_ok = _model_is_acceptable(fan_results, response_model, bootstrap)
    accepted = mode == 'final' and complete and attempts_accounted_for and model_ok
    if mode == 'pilot':
        if complete and pilot_clean:
            status = 'PILOT_ONLY'
        elif complete:
            status = 'PILOT_HAS_EXCLUSIONS'
        else:
            status = 'INCOMPLETE_PILOT'
    elif not complete:
        status = 'INCOMPLETE_FINAL_DATASET'
    elif not attempts_accounted_for:
        status = 'INCOMPLETE_ATTEMPT_ACCOUNTING'
    elif not model_ok:
        status = 'MODEL_REJECTED'
    else:
        status = 'ACCEPTED'

    approved_values = None
    if accepted and bootstrap is not None and response_model is not None:
        approved_values = {
            'model': MODEL_NAME,
            'initial_variance': 0.0,
            'response_enabled_axes': list(RESPONSE_ENABLED_AXES),
            'response_time_constant': response_model['time_constant'],
            'stationary_residual_variance_per_fan': {
                fan: [
                    bootstrap['stationary_residual_variance_ci95'][fan][axis]['upper']
                    for axis in AXES
                ]
                for fan in VALID_FANS
            },
            'residual_mean_per_fan': {
                fan: fan_results[fan]['pooled_residual_mean'] for fan in VALID_FANS
            },
        }

    serializable_fans = {}
    for fan, result in fan_results.items():
        serializable_fans[fan] = {
            key: value for key, value in result.items() if not key.startswith('_')
        }
        serializable_fans[fan]['excluded_runs'] = [
            {'run': run_id, 'reason': reason}
            for run_id, reason in result['excluded_runs']
        ]

    return {
        'generated': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'mode': mode,
        'campaign': ACTIVE_CAMPAIGN,
        'status': status,
        'accepted': accepted,
        'attempts_accounted_for': attempts_accounted_for,
        'required_runs_per_fan': required_runs,
        'required_fans': list(required_fans),
        'scenario': _SCENARIO,
        'condition': _CONDITION,
        'profile_signature': PROFILE_SIGNATURE,
        'profile': PROFILE_PAYLOAD,
        'model_name': MODEL_NAME,
        'model': (
            'P_0 = 0; horizontal mean follows dv/dt=(u-v)/tau, dp/dt=v; '
            'vertical mean follows commanded z; future covariance is the '
            'fan-conditioned stationary diagonal response residual'
        ),
        'tracking_response': response_model,
        'model_thresholds': {
            'stationarity_bin_size': STATIONARITY_BIN_SIZE,
            'max_stationarity_ratio': MAX_STATIONARITY_RATIO,
            'response_time_constant_bounds': list(RESPONSE_TIME_CONSTANT_BOUNDS),
            'minimum_response_r_squared': MIN_RESPONSE_R_SQUARED,
            'minimum_valid_bootstrap_fraction': MIN_BOOTSTRAP_VALID_FRACTION,
        },
        'fans': serializable_fans,
        'bootstrap': bootstrap,
        'approved_values': approved_values,
    }


def write_covariance_report(payload: dict[str, Any], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')


def write_calibration_plot(
    results: dict[int, dict[str, Any]],
    report: dict[str, Any],
    out_stem: pathlib.Path,
) -> dict[str, str]:
    """Plot the raw mean, response residual, and residual stationarity evidence."""
    fans = [fan for fan in sorted(results) if results[fan]['mean_residual'] is not None]
    if not fans:
        return {}

    import matplotlib.pyplot as plt

    from visualization.style import PALETTE, save_figure

    axis_colors = (
        PALETTE['ego']['stroke'],
        PALETTE['plan']['stroke'],
        PALETTE['goal']['stroke'],
    )
    figure, axes = plt.subplots(
        len(fans), 2, squeeze=False, figsize=(7.16, max(2.5, 2.25 * len(fans))),
        sharex='col',
    )
    bootstrap = report.get('bootstrap') or {}
    variance_intervals = bootstrap.get('stationary_residual_variance_ci95', {})
    legend_lines = []

    for row, fan in enumerate(fans):
        result = results[fan]
        waypoint = np.arange(len(result['mean_residual']))
        raw_mean = np.asarray(result['raw_mean_tracking_error'], dtype=float)
        residual_mean = np.asarray(result['mean_residual'], dtype=float)
        variance = np.asarray(result['planner_residual_variance'], dtype=float)
        stationary = np.asarray(result['stationary_residual_variance'], dtype=float)

        mean_axis, variance_axis = axes[row]
        for index, (label, color) in enumerate(zip(AXES, axis_colors)):
            mean_axis.plot(
                waypoint, raw_mean[:, index], color=color, alpha=0.25, linestyle='--',
            )
            mean_line, = mean_axis.plot(
                waypoint, residual_mean[:, index], color=color, label=label,
            )
            if row == 0:
                legend_lines.append(mean_line)
            variance_axis.plot(
                waypoint, variance[:, index], color=color, alpha=0.65,
            )
            variance_axis.axhline(
                stationary[index], color=color, linestyle='--', linewidth=1.2,
            )
            interval = variance_intervals.get(fan, {}).get(label)
            if interval is not None:
                variance_axis.axhline(
                    interval['upper'], color=color, linestyle=':', linewidth=1.0,
                )

        mean_axis.axhline(0.0, color='black', linewidth=0.7, alpha=0.5)
        mean_axis.set_ylabel(f'fan {fan}\nerror [m]')
        variance_axis.set_ylabel('variance [m²]')
        ratio = max(result['residual_stationarity']['max_bin_to_pooled_ratio'])
        verdict = 'pass' if result['residual_stationarity']['acceptable'] else 'reject'
        variance_axis.text(
            0.98, 0.95, f'residual stationarity: {verdict}; max ratio={ratio:.2f}',
            transform=variance_axis.transAxes, va='top', ha='right', fontsize=8,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.90, 'pad': 1.5},
        )
        mean_axis.grid(True)
        variance_axis.grid(True)
        variance_axis.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    axes[0, 0].set_title('mean error (faint raw; solid after response)')
    axes[0, 1].set_title('residual variance (dashed estimate; dotted 95% upper)')
    axes[-1, 0].set_xlabel('commanded waypoint')
    axes[-1, 1].set_xlabel('commanded waypoint')
    figure.suptitle(
        f'Crazyflie response/covariance calibration — {report["status"].replace("_", " ")}',
        y=0.995,
    )
    figure.legend(
        legend_lines, AXES, title='axis', ncol=3, frameon=False,
        loc='upper center', bbox_to_anchor=(0.5, 0.94),
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    written = save_figure(figure, out_stem, formats=('png', 'pdf'))
    plt.close(figure)
    return written


def _print_summary(
    mode: str,
    results: dict[int, dict[str, Any]],
    report: dict[str, Any],
) -> None:
    required = PILOT_RUNS if mode == 'pilot' else FINAL_RUNS_PER_FAN
    response = report.get('tracking_response') or {}
    if response:
        tau = ', '.join(
            f'{axis}={value:.3f}s' for axis, value in zip(AXES, response['time_constant'])
        )
        r_squared = ', '.join(
            f'{axis}={value:.3f}' for axis, value in zip(AXES, response['r_squared'])
        )
        print(f'response time constants: {tau}')
        print(f'tracking-error R²: {r_squared}')
        print(f'response fit: {"pass" if response["acceptable"] else "reject"}')
    print(f'{"fan":>4}{"valid":>8}{"attempts":>10}{"excluded":>10}{"target":>8}')
    for fan, result in results.items():
        print(
            f'{fan:>4}{result["n_runs"]:>8}{result["n_attempts"]:>10}'
            f'{len(result["excluded_runs"]):>10}{required:>8}'
        )
        for run_id, reason in result['excluded_runs']:
            print(f'  excluded {run_id}: {reason}')
        if result['stationary_residual_variance'] is not None:
            variance = ', '.join(
                f'{axis}={value:.6f}'
                for axis, value in zip(AXES, result['stationary_residual_variance'])
            )
            ratios = ', '.join(
                f'{axis}={value:.2f}'
                for axis, value in zip(
                    AXES, result['residual_stationarity']['max_bin_to_pooled_ratio'],
                )
            )
            print(f'  stationary residual variance: {variance}')
            print(f'  max residual-bin/pooled ratio: {ratios}')
    print(f'status: {report["status"]}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('pilot', 'final'), default=ACTIVE_CAMPAIGN)
    parser.add_argument('--fans', type=int, nargs='+', choices=VALID_FANS)
    parser.add_argument('--bootstrap-samples', type=int, default=BOOTSTRAP_SAMPLES)
    args = parser.parse_args()

    if args.mode != ACTIVE_CAMPAIGN:
        raise SystemExit(
            f'Refusing {args.mode!r} report while calibration.active_campaign is '
            f'{ACTIVE_CAMPAIGN!r}. Update config.yml deliberately before collecting data.'
        )
    default_fans = (2,) if args.mode == 'pilot' else VALID_FANS
    fans = tuple(args.fans) if args.fans else default_fans
    if args.mode == 'pilot' and fans != (2,):
        raise SystemExit('Pilot mode is restricted to fan 2.')

    results, response_model = calibrate_campaign(fans)
    ready_for_bootstrap = all(result['n_runs'] >= 2 for result in results.values())
    bootstrap = None
    if ready_for_bootstrap:
        bootstrap = bootstrap_response_model(
            {fan: result['_runs'] for fan, result in results.items()},
            samples=args.bootstrap_samples,
            seed=BOOTSTRAP_SEED,
        )

    report = build_campaign_report(args.mode, results, response_model, bootstrap)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prefix = 'pilot_covariance' if args.mode == 'pilot' else 'covariance'
    out_path = EXPERIMENT_DIR / 'calibration' / 'reports' / f'{prefix}_{timestamp}.yml'
    write_covariance_report(report, out_path)
    plot_stem = EXPERIMENT_DIR / 'plots' / f'calibration_{prefix}_{timestamp}'
    written_plots = write_calibration_plot(results, report, plot_stem)
    _print_summary(args.mode, results, report)
    print(f'Wrote {out_path}')
    if written_plots:
        print(f'Wrote {written_plots["png"]} and {written_plots["pdf"]}')

    if args.mode == 'final' and not report['accepted']:
        raise SystemExit(2)
    if report['accepted']:
        print('Approved conservative values:')
        print(yaml.safe_dump(report['approved_values'], sort_keys=False).rstrip())
        print(f'Set uncertainty.source_report to {out_path}. Regenerate all figure8 plans.')


if __name__ == '__main__':
    main()
