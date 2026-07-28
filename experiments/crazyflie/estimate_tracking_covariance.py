#!/usr/bin/env python3
"""Offline tracking-covariance calibration for the figure8 scenario.

Reads the existing commanded/actual CSVs written by components/flight_logger.py
for repeated `python run.py fly --scenario figure8 --condition deterministic
--fan <L>` flights (no changes to that command, the flight loop, or the
logger). Aligns runs by waypoint, computes the empirical per-waypoint 3D
tracking-error mean/covariance across runs (not across the 10 Hz samples
within one run), fits the existing isotropic Sigma_k = Sigma0 + k*Q
uncertainty model per fan level, and writes calibrated_uncertainty.yml -- the
only artifact a later phase would consume to replace SIGMA0_PER_FAN/Q_STD's
placeholder values.

This script does not touch run.py, components/crazyflie.py,
components/flight_logger.py, src/pdstl, or src/planning, and does not import
torch/ROS/cflib -- only numpy/csv/yaml plus components.config and
components.flight_logger's pure-Python helpers.

Alignment against the *existing*, unmodified CSV format: every complete run
logs exactly 1 + FIG8_FLIGHT_POINTS commanded rows in a fixed order -- row 0
is the pre-mission hover-at-start call (logged before offset correction, not
one of the K figure8 waypoints), and rows 1..FIG8_FLIGHT_POINTS are the
offset-corrected figure8 waypoints, always visited in the same order. Row 0
is dropped; the remaining rows' file order IS the waypoint index k=0..K-1,
with no separate waypoint-index column needed.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import yaml

from components.config import EXPERIMENT_DIR, FIG8_FLIGHT_POINTS, LOGS_DIR, VALID_FANS
from components.flight_logger import _log_prefix

_SCENARIO = 'figure8'
_CONDITION = 'deterministic'
_POOR_FIT_R_SQUARED = 0.5


def _read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(newline='') as fh:
        return list(csv.DictReader(fh))


def discover_runs(fan: int) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """(commanded_path, actual_path) pairs for every logged run at this fan level."""
    prefix = _log_prefix(_CONDITION, _SCENARIO, fan)
    pairs = []
    for actual_path in sorted(LOGS_DIR.glob(f'{prefix}*_actual.csv')):
        commanded_path = pathlib.Path(str(actual_path).replace('_actual.csv', '_commanded.csv'))
        pairs.append((commanded_path, actual_path))
    return pairs


def load_run(
    commanded_path: pathlib.Path, actual_path: pathlib.Path,
) -> tuple[np.ndarray | None, str | None]:
    """Per-waypoint tracking error [K,3] for one run, or (None, reason) if excluded.

    Excludes only for a technical reason (missing file, wrong row count,
    un-interpolable timestamp, non-finite data) -- never for error magnitude.
    """
    if not commanded_path.exists():
        return None, 'missing commanded file'
    if not actual_path.exists():
        return None, 'missing actual file'

    commanded_rows = _read_csv(commanded_path)
    if len(commanded_rows) < 1:
        return None, 'empty commanded file'
    mission_rows = commanded_rows[1:]  # drop the pre-mission hover-at-start row
    if len(mission_rows) != FIG8_FLIGHT_POINTS:
        return None, (
            f'incomplete trajectory: {len(mission_rows)} mission waypoints logged, '
            f'expected {FIG8_FLIGHT_POINTS}'
        )

    actual_rows = _read_csv(actual_path)
    if len(actual_rows) < 2:
        return None, 'actual log has too few samples to interpolate'

    try:
        actual_t = np.array([float(r['t']) for r in actual_rows])
        actual_xyz = np.array([[float(r['x']), float(r['y']), float(r['z'])] for r in actual_rows])
    except (KeyError, ValueError):
        return None, 'invalid actual log data'
    if not np.all(np.isfinite(actual_t)) or not np.all(np.isfinite(actual_xyz)):
        return None, 'non-finite actual position data'

    order = np.argsort(actual_t)
    actual_t = actual_t[order]
    actual_xyz = actual_xyz[order]
    t_min, t_max = actual_t[0], actual_t[-1]

    errors = np.empty((FIG8_FLIGHT_POINTS, 3))
    for k, row in enumerate(mission_rows):
        try:
            t = float(row['t'])
            commanded_xyz = np.array([float(row['x']), float(row['y']), float(row['z'])])
        except (KeyError, ValueError):
            return None, f'invalid commanded row at waypoint {k}'
        if not np.isfinite(t) or not np.all(np.isfinite(commanded_xyz)):
            return None, f'non-finite commanded data at waypoint {k}'
        if not (t_min <= t <= t_max):
            return None, (
                f'waypoint {k} arrival time {t:.3f}s outside actual log range '
                f'[{t_min:.3f}, {t_max:.3f}]s -- tracking loss or truncated log'
            )
        interpolated = np.array([np.interp(t, actual_t, actual_xyz[:, axis]) for axis in range(3)])
        errors[k] = interpolated - commanded_xyz

    return errors, None


def align_runs(fan: int) -> tuple[np.ndarray, list[str], list[tuple[str, str]]]:
    """errors[N,K,3] across every valid run, plus included/excluded run ids."""
    included: list[str] = []
    excluded: list[tuple[str, str]] = []
    error_list: list[np.ndarray] = []

    for commanded_path, actual_path in discover_runs(fan):
        run_id = actual_path.stem.removesuffix('_actual')
        errors, reason = load_run(commanded_path, actual_path)
        if errors is None:
            excluded.append((run_id, reason))
        else:
            error_list.append(errors)
            included.append(run_id)

    errors_arr = (
        np.stack(error_list, axis=0) if error_list
        else np.empty((0, FIG8_FLIGHT_POINTS, 3))
    )
    return errors_arr, included, excluded


def compute_mean_and_covariance(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """mean_error[K,3], covariance_raw[K,3,3] -- sample statistics ACROSS RUNS
    at each waypoint (not across the 10 Hz samples within one run). No
    per-run bias subtraction -- preserves systematic tracking bias in
    mean_error. Requires N>=2 runs (caller's responsibility to check).
    """
    n_runs, n_waypoints, _ = errors.shape
    mean_error = errors.mean(axis=0)
    covariance_raw = np.stack([
        np.cov(errors[:, k, :], rowvar=False, bias=False) for k in range(n_waypoints)
    ])
    return mean_error, covariance_raw


@dataclass
class FitResult:
    sigma0: float
    q_var: float
    q_std: float
    r_squared: float
    residual_rms: float


def fit_sigma0_q(covariance_raw: np.ndarray) -> FitResult:
    """Fit the existing isotropic Sigma_k = sigma0 + k*q_var model against the
    empirical per-waypoint covariance, collapsed to a single isotropic
    variance per waypoint (mean of the 3 diagonal entries, matching the
    model's shared-across-axes assumption). A negative fitted sigma0/q_var is
    reported as-is (not clamped) -- it is itself a poor-fit signal.
    """
    n_waypoints = covariance_raw.shape[0]
    var_isotropic = np.array([np.mean(np.diag(covariance_raw[k])) for k in range(n_waypoints)])
    k_idx = np.arange(n_waypoints, dtype=float)

    q_var, sigma0 = np.polyfit(k_idx, var_isotropic, 1)
    fitted = sigma0 + q_var * k_idx
    residuals = var_isotropic - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((var_isotropic - var_isotropic.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    residual_rms = float(np.sqrt(np.mean(residuals**2)))
    q_std = math.sqrt(q_var) if q_var >= 0 else float('nan')

    return FitResult(
        sigma0=float(sigma0), q_var=float(q_var), q_std=q_std,
        r_squared=r_squared, residual_rms=residual_rms,
    )


def calibrate_fan(fan: int) -> dict:
    """Full pipeline for one fan level: align, compute, fit. Handles zero/one
    valid runs gracefully (no crash) -- fit fields stay None.
    """
    errors, included, excluded = align_runs(fan)
    n_runs = errors.shape[0]

    result: dict = {
        'fan': fan, 'n_runs': n_runs, 'included_runs': included, 'excluded_runs': excluded,
        'mean_error': None,
        'sigma0_fit': None, 'q_var_fit': None, 'q_std_fit': None,
        'r_squared': None, 'residual_rms': None,
    }
    if n_runs < 2:
        return result

    mean_error, covariance_raw = compute_mean_and_covariance(errors)
    fit = fit_sigma0_q(covariance_raw)
    result.update({
        'mean_error': mean_error.tolist(),
        'sigma0_fit': fit.sigma0, 'q_var_fit': fit.q_var, 'q_std_fit': fit.q_std,
        'r_squared': fit.r_squared, 'residual_rms': fit.residual_rms,
    })
    return result


def fit_quality_verdict(result: dict) -> str:
    r_squared = result['r_squared']
    if r_squared is None:
        return 'NO DATA'
    if (
        math.isnan(r_squared) or r_squared < _POOR_FIT_R_SQUARED
        or result['sigma0_fit'] < 0 or result['q_var_fit'] < 0
    ):
        return 'POOR FIT'
    return 'GOOD FIT'


def write_calibrated_uncertainty_yml(results: dict[int, dict], out_path: pathlib.Path) -> None:
    payload = {
        'generated': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'scenario': _SCENARIO,
        'condition': _CONDITION,
        'model': 'Sigma_k = sigma0 + k * q_var (isotropic per-fan fit)',
        'fans': {
            fan: {
                'n_runs': r['n_runs'],
                'sigma0_fit': r['sigma0_fit'],
                'q_var_fit': r['q_var_fit'],
                'q_std_fit': r['q_std_fit'],
                'r_squared': r['r_squared'],
                'residual_rms': r['residual_rms'],
                'fit_quality': fit_quality_verdict(r),
            }
            for fan, r in results.items()
        },
    }
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Estimate figure8 per-waypoint tracking covariance from repeated '
                     'deterministic flights, and fit Sigma_k = Sigma0 + k*Q per fan level.'
    )
    parser.add_argument('--fans', type=int, nargs='+', default=list(VALID_FANS), choices=VALID_FANS)
    args = parser.parse_args()

    results: dict[int, dict] = {}
    print(f'{"fan":>4}{"n_runs":>8}{"excluded":>10}{"sigma0":>14}{"q_std":>14}{"R2":>8}{"resid_rms":>14}  verdict')
    for fan in args.fans:
        result = calibrate_fan(fan)
        results[fan] = result
        for run_id, reason in result['excluded_runs']:
            print(f'  [fan {fan}] excluded run {run_id}: {reason}')

        verdict = fit_quality_verdict(result)
        sigma0_str = f"{result['sigma0_fit']:.6f}" if result['sigma0_fit'] is not None else '--'
        q_std_str = f"{result['q_std_fit']:.6f}" if result['q_std_fit'] is not None else '--'
        r2_str = f"{result['r_squared']:.3f}" if result['r_squared'] is not None else '--'
        resid_str = f"{result['residual_rms']:.6f}" if result['residual_rms'] is not None else '--'
        print(
            f'{fan:>4}{result["n_runs"]:>8}{len(result["excluded_runs"]):>10}'
            f'{sigma0_str:>14}{q_std_str:>14}{r2_str:>8}{resid_str:>14}  {verdict}'
        )
        if verdict == 'POOR FIT':
            print(
                f'  [fan {fan}] Sigma0+kQ fit is poor -- stop here. Do not extend the '
                f'planner to a full per-waypoint covariance schedule based on this data.'
            )

    out_path = EXPERIMENT_DIR / 'calibrated_uncertainty.yml'
    write_calibrated_uncertainty_yml(results, out_path)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
