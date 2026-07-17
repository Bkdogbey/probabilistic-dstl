"""Estimate per-fan Crazyflie tracking-error covariance from flight logs.

Only deterministic baseline runs are used.  Each commanded CSV records the
time at which a blocking ``go_to`` reached a waypoint; the actual Lighthouse
trace is linearly interpolated at those arrival times.  Residual sequences are
then resampled by path progress to the planning horizon and aggregated across
runs into a full 2x2 covariance profile.
"""

from __future__ import annotations

import csv
import datetime
import json
import pathlib
import re

import numpy as np

from components.config import (
    BASELINE_PATH_ID,
    CALIBRATION_DIR,
    LOGS_DIR,
    Q_STD,
    SIGMA0_PER_FAN,
    T,
)

DEFAULT_MIN_RUNS: int = 10
_RUN_RE = re.compile(r'_run(?P<run>\d+)')


def calibration_path(fan: int) -> pathlib.Path:
    return CALIBRATION_DIR / f'uncertainty_fan{fan}.json'


def _read_xy(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f'Empty log: {path}')
    times = np.array([float(r['t']) for r in rows], dtype=float)
    xy = np.array([[float(r['x']), float(r['y'])] for r in rows], dtype=float)
    return times, xy


def _residual_sequence(actual_path: pathlib.Path, n_steps: int) -> np.ndarray:
    commanded_path = pathlib.Path(str(actual_path).replace('_actual.csv', '_commanded.csv'))
    if not commanded_path.exists():
        raise ValueError(f'Missing commanded partner for {actual_path.name}')

    with commanded_path.open(newline='') as fh:
        first = next(csv.DictReader(fh), None)
    logged_path_id = first.get('baseline_path_id') if first else None
    if logged_path_id != BASELINE_PATH_ID:
        raise ValueError(
            f'{actual_path.name}: path id {logged_path_id!r} does not match '
            f'current {BASELINE_PATH_ID!r}'
        )

    actual_t, actual_xy = _read_xy(actual_path)
    commanded_t, commanded_xy = _read_xy(commanded_path)
    valid = (commanded_t >= actual_t[0]) & (commanded_t <= actual_t[-1])
    commanded_t, commanded_xy = commanded_t[valid], commanded_xy[valid]
    if len(commanded_t) < 2:
        raise ValueError(f'Not enough aligned waypoints in {actual_path.name}')

    actual_at_command = np.column_stack([
        np.interp(commanded_t, actual_t, actual_xy[:, axis]) for axis in range(2)
    ])
    residual = actual_at_command - commanded_xy

    # Command logs include takeoff/start and may use a denser deterministic
    # path than the planner.  Resample by normalized mission progress so every
    # run contributes the same T+1 residual locations.
    source_progress = np.linspace(0.0, 1.0, len(residual))
    target_progress = np.linspace(0.0, 1.0, n_steps)
    return np.column_stack([
        np.interp(target_progress, source_progress, residual[:, axis]) for axis in range(2)
    ])


def _nearest_psd(cov: np.ndarray, variance_floor: float) -> np.ndarray:
    cov = (cov + cov.T) / 2.0
    values, vectors = np.linalg.eigh(cov)
    values = np.maximum(values, variance_floor)
    return (vectors * values) @ vectors.T


def calibrate_uncertainty(
    fan: int, *, min_runs: int = DEFAULT_MIN_RUNS, n_steps: int = T + 1,
    variance_floor: float = 1.0e-8,
) -> pathlib.Path:
    """Estimate and save a covariance profile for one fan setting."""
    candidates = sorted(LOGS_DIR.glob(f'deterministic_fan{fan:02d}_run*_actual.csv'))
    usable = [p for p in candidates if '_CRASH' not in p.name]
    sequences: list[np.ndarray] = []
    used_paths: list[pathlib.Path] = []
    skipped: list[str] = []
    for path in usable:
        try:
            sequences.append(_residual_sequence(path, n_steps))
            used_paths.append(path)
        except ValueError as exc:
            skipped.append(str(exc))

    if len(sequences) < min_runs:
        raise ValueError(
            f'Need at least {min_runs} usable deterministic fan-{fan} runs; '
            f'found {len(sequences)} in {LOGS_DIR}.'
        )

    samples = np.stack(sequences)  # [runs, steps, 2]
    mean = samples.mean(axis=0)
    covariances = []
    for step in range(n_steps):
        cov = np.cov(samples[:, step, :], rowvar=False, ddof=1)
        covariances.append(_nearest_psd(cov, variance_floor))
    covariance = np.stack(covariances)

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    out = calibration_path(fan)
    payload = {
        'schema_version': 1,
        'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'fan': fan,
        'condition': 'deterministic',
        'baseline_path_id': BASELINE_PATH_ID,
        'dimensions': ['x', 'y'],
        'runs_used': len(used_paths),
        'run_files': [p.name for p in used_paths],
        'skipped': skipped,
        'n_steps': n_steps,
        'alignment': 'actual interpolated at commanded waypoint arrival; resampled by path progress',
        'variance_floor': variance_floor,
        'mean_residual': mean.tolist(),
        'max_mean_residual_m': float(np.linalg.norm(mean, axis=1).max()),
        'covariance': covariance.tolist(),
    }
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    std_start = np.sqrt(np.diag(covariance[0]))
    std_end = np.sqrt(np.diag(covariance[-1]))
    print(f'Used {len(used_paths)} deterministic fan-{fan} runs.')
    print(f'Start std: x={std_start[0]:.4f} m, y={std_start[1]:.4f} m')
    print(f'End std:   x={std_end[0]:.4f} m, y={std_end[1]:.4f} m')
    max_bias = float(np.linalg.norm(mean, axis=1).max())
    if max_bias > 0.05:
        print(
            f'WARNING: maximum mean residual is {max_bias:.3f} m (> 0.05 m). '
            'This looks like systematic bias; covariance alone does not model it.'
        )
    print(f'Wrote calibration to {out}')
    return out


def load_uncertainty_profile(fan: int, n_steps: int) -> dict | None:
    """Load a generated profile, resampling it if the requested horizon differs."""
    path = calibration_path(fan)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('baseline_path_id') != BASELINE_PATH_ID:
        return None
    covariance = np.asarray(data['covariance'], dtype=float)
    mean = np.asarray(data['mean_residual'], dtype=float)
    if covariance.shape[0] != n_steps:
        source = np.linspace(0.0, 1.0, covariance.shape[0])
        target = np.linspace(0.0, 1.0, n_steps)
        covariance = np.stack([
            np.interp(target, source, covariance[:, i, j])
            for i in range(2) for j in range(2)
        ], axis=-1).reshape(n_steps, 2, 2)
        mean = np.column_stack([
            np.interp(target, source, mean[:, axis]) for axis in range(2)
        ])
    return {'path': path, 'metadata': data, 'mean': mean, 'covariance': covariance}


def uncertainty_signature(fan: int) -> dict:
    """Return metadata used to invalidate plans after recalibration."""
    path = calibration_path(fan)
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    if data is None or data.get('baseline_path_id') != BASELINE_PATH_ID:
        return {
            'uncertainty_model': 'paper_sigma0_shared_process_noise',
            'calibration_generated': None,
            'sigma0': SIGMA0_PER_FAN[fan],
            'q_std': Q_STD,
        }
    return {
        'uncertainty_model': 'empirical_covariance_profile',
        'calibration_generated': data.get('generated'),
    }
