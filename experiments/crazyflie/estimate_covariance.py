#!/usr/bin/env python3
"""Estimate figure-eight tracking uncertainty from repeated real flights.

Pilot reports are diagnostic only. Final reports require the configured number
of valid runs for all fan levels and expose conservative bootstrap upper bounds
only when the shared nonnegative uncertainty model passes its fit checks.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import pathlib
from dataclasses import asdict, dataclass
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

VALID_FANS = tuple(int(fan) for fan in _CONFIG['uncertainty']['sigma0_per_fan'])
FIG8_FLIGHT_POINTS = int(_CONFIG['figure8']['flight_points'])
LOG_SAMPLE_HZ = int(_CONFIG['flight']['log_sample_hz'])
PROFILE_SIGNATURE = flight_profile_signature(_CONFIG, 'figure8')
PROFILE_PAYLOAD = flight_profile_payload(_CONFIG, 'figure8')

_CALIBRATION = _CONFIG['calibration']
ACTIVE_CAMPAIGN = str(_CALIBRATION['active_campaign'])
PILOT_RUNS = int(_CALIBRATION['pilot_runs'])
FINAL_RUNS_PER_FAN = int(_CALIBRATION['final_runs_per_fan'])
BOOTSTRAP_SAMPLES = int(_CALIBRATION['bootstrap_samples'])
BOOTSTRAP_SEED = int(_CALIBRATION['bootstrap_seed'])
MIN_R_SQUARED = float(_CALIBRATION['min_r_squared'])
FIT_BIN_SIZE = int(_CALIBRATION['fit_bin_size'])
PLAUSIBLE_POSITION_MARGIN = float(_CALIBRATION['plausible_position_margin'])

_SCENARIO = 'figure8'
_CONDITION = 'deterministic'
_TERMINAL_HOLD_TOLERANCE = 1.0 / LOG_SAMPLE_HZ


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

    pairs = []
    for stem, paths in sorted(attempts.items()):
        pairs.append((
            paths.get('commanded', LOGS_DIR / f'{stem}_commanded.csv'),
            paths.get('actual', LOGS_DIR / f'{stem}_actual.csv'),
        ))
    return pairs


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
    bounds = np.asarray([workspace[axis] for axis in ('x', 'y', 'z')], dtype=float)
    lower = bounds[:, 0] - PLAUSIBLE_POSITION_MARGIN
    upper = bounds[:, 1] + PLAUSIBLE_POSITION_MARGIN
    return bool(np.all((actual_xyz >= lower) & (actual_xyz <= upper)))


def load_run(
    commanded_path: pathlib.Path,
    actual_path: pathlib.Path,
) -> tuple[np.ndarray | None, str | None]:
    """Return per-waypoint XYZ tracking errors or a precise exclusion reason."""
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
            [float(row['x']), float(row['y']), float(row['z'])] for row in mission_rows
        ])
        actual_t = np.asarray([float(row['t']) for row in actual_rows])
        actual_xyz = np.asarray([
            [float(row['x']), float(row['y']), float(row['z'])] for row in actual_rows
        ])
    except (KeyError, ValueError):
        return None, 'invalid numeric log data'

    if not all(np.all(np.isfinite(values)) for values in (
        commanded_t, commanded_xyz, actual_t, actual_xyz,
    )):
        return None, 'non-finite log data'
    if np.any(np.diff(commanded_t) < 0):
        return None, 'commanded timestamps are not monotonic'
    if not _plausible_actual_positions(actual_xyz):
        return None, 'actual position outside plausible workspace envelope'

    order = np.argsort(actual_t)
    actual_t = actual_t[order]
    actual_xyz = actual_xyz[order]
    if np.any(np.diff(actual_t) <= 0):
        return None, 'actual timestamps are not strictly increasing'

    t_min, t_max = actual_t[0], actual_t[-1]
    errors = np.empty((FIG8_FLIGHT_POINTS, 3))
    for k, (t, target) in enumerate(zip(commanded_t, commanded_xyz)):
        terminal_hold = (
            k == FIG8_FLIGHT_POINTS - 1
            and t_max < t <= t_max + _TERMINAL_HOLD_TOLERANCE
        )
        if not (t_min <= t <= t_max) and not terminal_hold:
            return None, (
                f'waypoint {k} arrival time {t:.3f}s outside actual log range '
                f'[{t_min:.3f}, {t_max:.3f}]s -- tracking loss or truncated log'
            )
        interpolated = actual_xyz[-1] if terminal_hold else np.asarray([
            np.interp(t, actual_t, actual_xyz[:, axis]) for axis in range(3)
        ])
        errors[k] = interpolated - target
    return errors, None


def align_runs(fan: int) -> tuple[np.ndarray, list[str], list[tuple[str, str]]]:
    """Align all valid runs by waypoint and account for every excluded attempt."""
    included: list[str] = []
    excluded: list[tuple[str, str]] = []
    error_list: list[np.ndarray] = []
    for commanded_path, actual_path in discover_runs(fan):
        run_path = actual_path if actual_path.exists() else commanded_path
        run_id = run_path.stem.removesuffix('_actual').removesuffix('_commanded')
        errors, reason = load_run(commanded_path, actual_path)
        if errors is None:
            excluded.append((run_id, reason or 'unknown exclusion'))
        else:
            error_list.append(errors)
            included.append(run_id)
    stacked = (
        np.stack(error_list, axis=0)
        if error_list
        else np.empty((0, FIG8_FLIGHT_POINTS, 3))
    )
    return stacked, included, excluded


def compute_error_statistics(
    errors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, centered covariance, second moment, and isotropic MSE."""
    if errors.ndim != 3 or errors.shape[0] < 2 or errors.shape[2] != 3:
        raise ValueError('errors must have shape [N>=2, K, 3].')
    mean_error = errors.mean(axis=0)
    covariance = np.stack([
        np.cov(errors[:, k, :], rowvar=False, bias=False)
        for k in range(errors.shape[1])
    ])
    second_moment = np.einsum('nki,nkj->kij', errors, errors) / errors.shape[0]
    bias_inclusive_variance = np.mean(errors**2, axis=(0, 2))
    return mean_error, covariance, second_moment, bias_inclusive_variance


def compute_mean_and_covariance(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible centered statistics helper."""
    mean_error, covariance, _second_moment, _variance = compute_error_statistics(errors)
    return mean_error, covariance


@dataclass(frozen=True)
class FitResult:
    sigma0: float
    q_var: float
    q_std: float
    r_squared: float
    residual_rms: float


@dataclass(frozen=True)
class JointFitResult:
    sigma0_by_fan: dict[int, float]
    q_var: float
    q_std: float
    r_squared: float
    residual_rms: float


def _nonnegative_least_squares(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve the small NNLS problem exactly by enumerating active variables."""
    n_variables = design.shape[1]
    best_coefficients = np.zeros(n_variables)
    best_residual = float(np.dot(target, target))
    tolerance = 1e-12
    for active_count in range(1, n_variables + 1):
        for active in itertools.combinations(range(n_variables), active_count):
            candidate = np.zeros(n_variables)
            solution, *_ = np.linalg.lstsq(design[:, active], target, rcond=None)
            if np.any(solution < -tolerance):
                continue
            candidate[list(active)] = np.maximum(solution, 0.0)
            residual = float(np.sum((target - design @ candidate) ** 2))
            if residual < best_residual:
                best_coefficients, best_residual = candidate, residual
    return best_coefficients


def fit_shared_sigma0_q(variances_by_fan: dict[int, np.ndarray]) -> JointFitResult:
    """Fit fan-specific nonnegative intercepts and one shared nonnegative Q."""
    if not variances_by_fan:
        raise ValueError('At least one fan variance trace is required.')
    fans = sorted(variances_by_fan)
    rows, targets = [], []
    for fan_index, fan in enumerate(fans):
        variances = np.asarray(variances_by_fan[fan], dtype=float)
        if variances.ndim != 1 or not len(variances) or not np.all(np.isfinite(variances)):
            raise ValueError(f'Fan {fan} variance trace must be a finite 1D array.')
        for k, value in enumerate(variances):
            row = np.zeros(len(fans) + 1)
            row[fan_index] = 1.0
            row[-1] = float(k)
            rows.append(row)
            targets.append(value)
    design = np.asarray(rows)
    target = np.asarray(targets)
    coefficients = _nonnegative_least_squares(design, target)
    fitted = design @ coefficients
    residuals = target - fitted
    ss_res = float(np.sum(residuals**2))
    # Evaluate goodness of fit on ten-waypoint bins. With 20 runs, an
    # individual sample-variance estimate still has substantial noise; bins
    # test the cumulative trend the model represents without letting that
    # point noise make a true linear process look arbitrarily poor. Compare
    # against a separate constant baseline for each fan so between-fan
    # intercept differences cannot inflate R².
    binned_targets, binned_fitted, binned_fans = [], [], []
    offset = 0
    for fan_index, fan in enumerate(fans):
        values = np.asarray(variances_by_fan[fan], dtype=float)
        predicted = fitted[offset:offset + len(values)]
        offset += len(values)
        for start in range(0, len(values), FIT_BIN_SIZE):
            binned_targets.append(float(np.mean(values[start:start + FIT_BIN_SIZE])))
            binned_fitted.append(float(np.mean(predicted[start:start + FIT_BIN_SIZE])))
            binned_fans.append(fan_index)
    binned_targets = np.asarray(binned_targets)
    binned_fitted = np.asarray(binned_fitted)
    binned_fans = np.asarray(binned_fans)
    binned_residual = float(np.sum((binned_targets - binned_fitted) ** 2))
    binned_total = float(sum(
        np.sum((binned_targets[binned_fans == fan_index]
                - np.mean(binned_targets[binned_fans == fan_index])) ** 2)
        for fan_index in range(len(fans))
    ))
    if binned_total > 0:
        r_squared = 1.0 - binned_residual / binned_total
    else:
        raw_total = float(sum(
            np.sum((np.asarray(variances_by_fan[fan])
                    - np.mean(variances_by_fan[fan])) ** 2)
            for fan in fans
        ))
        r_squared = 1.0 - ss_res / raw_total if raw_total > 0 else float('nan')
    q_var = float(coefficients[-1])
    return JointFitResult(
        sigma0_by_fan={fan: float(coefficients[i]) for i, fan in enumerate(fans)},
        q_var=q_var,
        q_std=math.sqrt(q_var),
        r_squared=r_squared,
        residual_rms=float(np.sqrt(np.mean(residuals**2))),
    )


def fit_sigma0_q(covariance_raw: np.ndarray) -> FitResult:
    """Compatibility wrapper for a one-fan centered covariance trace."""
    variances = np.asarray([np.mean(np.diag(cov)) for cov in covariance_raw])
    fit = fit_shared_sigma0_q({0: variances})
    return FitResult(
        sigma0=fit.sigma0_by_fan[0], q_var=fit.q_var, q_std=fit.q_std,
        r_squared=fit.r_squared, residual_rms=fit.residual_rms,
    )


def calibrate_fan(fan: int) -> dict[str, Any]:
    errors, included, excluded = align_runs(fan)
    result: dict[str, Any] = {
        'fan': fan,
        'n_runs': int(errors.shape[0]),
        'n_attempts': len(included) + len(excluded),
        'included_runs': included,
        'excluded_runs': excluded,
        'mean_error': None,
        'centered_covariance': None,
        'second_moment': None,
        'bias_inclusive_variance': None,
        'pooled_sigma0': None,
        '_errors': errors,
    }
    if errors.shape[0] >= 2:
        mean, covariance, second_moment, variance = compute_error_statistics(errors)
        result.update({
            'mean_error': mean.tolist(),
            'centered_covariance': covariance.tolist(),
            'second_moment': second_moment.tolist(),
            'bias_inclusive_variance': variance.tolist(),
            # Single constant-per-fan covariance, pooling every waypoint's
            # bias-inclusive variance -- the model actually validated in the
            # pdSTL paper's real-world section (Sigma_0 pre-characterized
            # from pooled tracking-error residuals per fan, no k-dependence).
            'pooled_sigma0': float(np.mean(variance)),
        })
    return result


def bootstrap_joint_fit(
    errors_by_fan: dict[int, np.ndarray],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if samples < 1:
        raise ValueError('Bootstrap samples must be positive.')
    rng = np.random.default_rng(seed)
    fans = sorted(errors_by_fan)
    sigma_samples = {fan: [] for fan in fans}
    pooled_sigma_samples = {fan: [] for fan in fans}
    q_var_samples: list[float] = []
    q_std_samples: list[float] = []
    for _ in range(samples):
        variances = {}
        for fan in fans:
            errors = errors_by_fan[fan]
            indices = rng.integers(0, errors.shape[0], size=errors.shape[0])
            resampled = errors[indices]
            variances[fan] = np.mean(resampled**2, axis=(0, 2))
        fit = fit_shared_sigma0_q(variances)
        values = [*fit.sigma0_by_fan.values(), fit.q_var, fit.q_std]
        if not np.all(np.isfinite(values)):
            continue
        for fan in fans:
            sigma_samples[fan].append(fit.sigma0_by_fan[fan])
            pooled_sigma_samples[fan].append(float(np.mean(variances[fan])))
        q_var_samples.append(fit.q_var)
        q_std_samples.append(fit.q_std)

    def interval(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        lower, upper = np.percentile(values, [2.5, 97.5])
        return {'lower': float(lower), 'upper': float(upper)}

    return {
        'seed': seed,
        'samples_requested': samples,
        'samples_valid': len(q_var_samples),
        'sigma0_ci95': {fan: interval(values) for fan, values in sigma_samples.items()},
        'q_var_ci95': interval(q_var_samples),
        'q_std_ci95': interval(q_std_samples),
        # Diagnostic-only: CI for the constant pooled-per-fan model (see
        # calibrate_fan's `pooled_sigma0`). Not used for gating or
        # approved_values.
        'pooled_sigma0_ci95': {
            fan: interval(values) for fan, values in pooled_sigma_samples.items()
        },
    }


def _fit_is_acceptable(fit: JointFitResult | None, bootstrap: dict[str, Any] | None) -> bool:
    if fit is None or bootstrap is None:
        return False
    values = [*fit.sigma0_by_fan.values(), fit.q_var, fit.q_std, fit.r_squared]
    intervals = [
        *bootstrap['sigma0_ci95'].values(),
        bootstrap['q_var_ci95'],
        bootstrap['q_std_ci95'],
    ]
    intervals_valid = all(
        interval is not None
        and math.isfinite(interval['lower'])
        and math.isfinite(interval['upper'])
        and 0.0 <= interval['lower'] <= interval['upper']
        for interval in intervals
    )
    return bool(
        np.all(np.isfinite(values))
        and all(value >= 0.0 for value in fit.sigma0_by_fan.values())
        and fit.q_var >= 0.0
        and fit.r_squared >= MIN_R_SQUARED
        and bootstrap['samples_requested'] >= BOOTSTRAP_SAMPLES
        and bootstrap['samples_valid'] >= math.ceil(0.95 * bootstrap['samples_requested'])
        and intervals_valid
    )


def build_campaign_report(
    mode: str,
    fan_results: dict[int, dict[str, Any]],
    fit: JointFitResult | None,
    bootstrap: dict[str, Any] | None,
) -> dict[str, Any]:
    required_runs = PILOT_RUNS if mode == 'pilot' else FINAL_RUNS_PER_FAN
    required_fans = (2,) if mode == 'pilot' else VALID_FANS
    complete = (
        set(fan_results) == set(required_fans)
        and all(fan_results[fan]['n_runs'] >= required_runs for fan in required_fans)
    )
    pilot_clean = not any(
        result['excluded_runs'] for result in fan_results.values()
    )
    fit_ok = _fit_is_acceptable(fit, bootstrap)
    accepted = mode == 'final' and complete and fit_ok
    if mode == 'pilot':
        if complete and pilot_clean:
            status = 'PILOT_ONLY'
        elif complete:
            status = 'PILOT_HAS_EXCLUSIONS'
        else:
            status = 'INCOMPLETE_PILOT'
    elif not complete:
        status = 'INCOMPLETE_FINAL_DATASET'
    elif not fit_ok:
        status = 'POOR_FIT'
    else:
        status = 'ACCEPTED'

    approved_values = None
    if accepted and bootstrap is not None:
        approved_values = {
            'sigma0_per_fan': {
                fan: bootstrap['sigma0_ci95'][fan]['upper'] for fan in VALID_FANS
            },
            'q_std': bootstrap['q_std_ci95']['upper'],
        }

    serializable_fans = {}
    for fan, result in fan_results.items():
        serializable_fans[fan] = {
            key: value for key, value in result.items() if key != '_errors'
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
        'required_runs_per_fan': required_runs,
        'required_fans': list(required_fans),
        'scenario': _SCENARIO,
        'condition': _CONDITION,
        'profile_signature': PROFILE_SIGNATURE,
        'profile': PROFILE_PAYLOAD,
        'model': (
            'E[||error_k||^2]/3 = sigma0_fan + k*q_var; '
            'nonnegative fan intercepts and shared q_var'
        ),
        'fit_thresholds': {
            'min_r_squared': MIN_R_SQUARED,
            'fit_bin_size': FIT_BIN_SIZE,
            'minimum_valid_bootstrap_fraction': 0.95,
        },
        'fans': serializable_fans,
        'joint_fit': asdict(fit) if fit is not None else None,
        'bootstrap': bootstrap,
        'approved_values': approved_values,
        'pooled_model': _pooled_model_summary(fan_results, bootstrap),
    }


def _pooled_model_summary(
    fan_results: dict[int, dict[str, Any]],
    bootstrap: dict[str, Any] | None,
) -> dict[str, Any]:
    """Diagnostic constant-per-fan covariance, no k-dependence.

    This is the model actually validated in the pdSTL paper's real-world
    section (arXiv:2606.19561, Sec. III.C): Sigma_0 pre-characterized from
    pooled tracking-error residuals per fan, held constant across the
    trajectory. It never gates acceptance and never feeds approved_values --
    it exists so a poor `sigma0 + k*q_var` fit (e.g. curvature-driven, not
    linear-in-k, tracking error) still leaves a defensible fallback estimate.
    """
    pooled_ci95 = (bootstrap or {}).get('pooled_sigma0_ci95', {})
    return {
        'description': (
            'E[||error||^2]/3 pooled over all waypoints per fan (no '
            'k-dependence); matches the constant Sigma_0 used in the '
            "paper's real-world Crazyflie validation."
        ),
        'sigma0_per_fan': {
            fan: result.get('pooled_sigma0') for fan, result in fan_results.items()
        },
        'sigma0_ci95': {fan: pooled_ci95.get(fan) for fan in fan_results},
    }


def fit_quality_verdict(result: dict[str, Any]) -> str:
    """Compatibility verdict for legacy one-fit result dictionaries."""
    r_squared = result.get('r_squared')
    if r_squared is None:
        return 'NO DATA'
    if (
        not math.isfinite(r_squared)
        or r_squared < MIN_R_SQUARED
        or result.get('sigma0_fit', -1.0) < 0
        or result.get('q_var_fit', -1.0) < 0
    ):
        return 'POOR FIT'
    return 'GOOD FIT'


def write_covariance_report(payload: dict[str, Any], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')


def _print_summary(mode: str, results: dict[int, dict[str, Any]], report: dict[str, Any]) -> None:
    required = PILOT_RUNS if mode == 'pilot' else FINAL_RUNS_PER_FAN
    print(f'{"fan":>4}{"valid":>8}{"attempts":>10}{"excluded":>10}{"target":>8}')
    for fan, result in results.items():
        print(
            f'{fan:>4}{result["n_runs"]:>8}{result["n_attempts"]:>10}'
            f'{len(result["excluded_runs"]):>10}{required:>8}'
        )
        for run_id, reason in result['excluded_runs']:
            print(f'  excluded {run_id}: {reason}')
    fit = report['joint_fit']
    if fit is not None:
        print(
            f'joint fit: q_std={fit["q_std"]:.6f}, R2={fit["r_squared"]:.3f}, '
            f'residual_rms={fit["residual_rms"]:.6f}'
        )
    pooled = report['pooled_model']['sigma0_per_fan']
    if any(value is not None for value in pooled.values()):
        pooled_str = ', '.join(
            f'fan{fan}={value:.6f}' for fan, value in pooled.items() if value is not None
        )
        print(f'pooled sigma0 (diagnostic, no k-dependence): {pooled_str}')
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

    results = {fan: calibrate_fan(fan) for fan in fans}
    ready_for_fit = all(result['n_runs'] >= 2 for result in results.values())
    fit = None
    bootstrap = None
    if ready_for_fit:
        variances = {
            fan: np.asarray(result['bias_inclusive_variance'])
            for fan, result in results.items()
        }
        fit = fit_shared_sigma0_q(variances)
        bootstrap = bootstrap_joint_fit(
            {fan: result['_errors'] for fan, result in results.items()},
            samples=args.bootstrap_samples,
            seed=BOOTSTRAP_SEED,
        )

    report = build_campaign_report(args.mode, results, fit, bootstrap)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prefix = 'pilot_covariance' if args.mode == 'pilot' else 'covariance'
    out_path = EXPERIMENT_DIR / 'calibration' / 'reports' / f'{prefix}_{timestamp}.yml'
    write_covariance_report(report, out_path)
    _print_summary(args.mode, results, report)
    print(f'Wrote {out_path}')

    if args.mode == 'final' and not report['accepted']:
        raise SystemExit(2)
    if report['accepted']:
        print('Approved conservative values:')
        print(yaml.safe_dump(report['approved_values'], sort_keys=False).rstrip())
        print(f'Set uncertainty.source_report to {out_path}. Regenerate all figure8 plans.')


if __name__ == '__main__':
    main()
