"""Unit tests for the offline figure8 tracking-covariance calibration script,
all synthetic (no Crazyflie/ROS/radio hardware, no real flight logs).
"""

from __future__ import annotations

import csv
import pathlib
import sys

_EXPERIMENT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENT_DIR))

import numpy as np
import pytest
import yaml

import estimate_tracking_covariance as etc


def _commanded_row(t, x, y, z):
    return {
        'condition': 'deterministic', 'scenario': 'figure8', 't': t, 'x': x, 'y': y, 'z': z,
        'outside_obs1': 1, 'outside_obs2': 1, 'outside_obs3': 1, 'safe': 1,
    }


def _write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_run(tmp_path, name, mission_t, mission_xyz, actual_xyz_fn, actual_span=0.1, n_actual=500):
    """Writes a complete (commanded, actual) CSV pair for one run.

    mission_t/mission_xyz: the K figure8 waypoints' arrival times/positions
    (row 0, the takeoff hover, is prepended automatically with distinct
    values so a bug that fails to drop it is detectable).
    """
    commanded_rows = [_commanded_row(0.0, 999.0, 999.0, 999.0)]  # takeoff row, deliberately wild
    for t, (x, y, z) in zip(mission_t, mission_xyz):
        commanded_rows.append(_commanded_row(float(t), float(x), float(y), float(z)))
    commanded_path = tmp_path / f'{name}_commanded.csv'
    _write_csv(commanded_path, commanded_rows)

    t0, t1 = mission_t[0] - actual_span, mission_t[-1] + actual_span
    actual_rows = []
    for t in np.linspace(t0, t1, n_actual):
        x, y, z = actual_xyz_fn(t)
        actual_rows.append(_commanded_row(round(float(t), 6), float(x), float(y), float(z)))
    actual_path = tmp_path / f'{name}_actual.csv'
    _write_csv(actual_path, actual_rows)

    return commanded_path, actual_path


@pytest.fixture(autouse=True)
def _small_k(monkeypatch):
    """Decouple tests from the real config's FIG8_FLIGHT_POINTS value."""
    monkeypatch.setattr(etc, 'FIG8_FLIGHT_POINTS', 5)


def _reference_mission(k=5):
    t = np.arange(k, dtype=float)  # 0,1,2,3,4
    xyz = np.column_stack([np.arange(k, dtype=float), np.zeros(k), np.zeros(k)])  # (0,0,0),(1,0,0),...
    return t, xyz


# ── load_run: interpolation, takeoff-row dropping, exclusion reasons ────────
def test_load_run_interpolates_exactly_at_known_timestamp(tmp_path):
    t, xyz = _reference_mission()
    # Actual position is exactly commanded + 0 everywhere except a linear ramp
    # in x between two known bracketing samples, so linear interpolation at
    # an exact mission timestamp must reproduce the ramp's value exactly.
    commanded_path, actual_path = _build_run(
        tmp_path, 'run', t, xyz, actual_xyz_fn=lambda tt: (tt + 0.5, 0.0, 0.0),
        actual_span=0.0, n_actual=2,
    )
    errors, reason = etc.load_run(commanded_path, actual_path)
    assert reason is None
    assert errors is not None
    assert np.allclose(errors[:, 0], 0.5, atol=1e-6)
    assert np.allclose(errors[:, 1:], 0.0, atol=1e-6)


def test_load_run_drops_takeoff_row(tmp_path):
    t, xyz = _reference_mission()
    commanded_path, actual_path = _build_run(
        tmp_path, 'run', t, xyz, actual_xyz_fn=lambda tt: (tt, 0.0, 0.0),
    )
    errors, reason = etc.load_run(commanded_path, actual_path)
    assert reason is None
    assert errors.shape == (5, 3)
    # If the takeoff row (x=999) leaked in, errors would be enormous.
    assert np.all(np.abs(errors) < 1.0)


def test_load_run_excludes_incomplete_trajectory(tmp_path):
    t, xyz = _reference_mission()
    commanded_path, actual_path = _build_run(
        tmp_path, 'run', t[:-1], xyz[:-1], actual_xyz_fn=lambda tt: (tt, 0.0, 0.0),
    )
    errors, reason = etc.load_run(commanded_path, actual_path)
    assert errors is None
    assert 'incomplete trajectory' in reason


def test_load_run_excludes_missing_actual_file(tmp_path):
    t, xyz = _reference_mission()
    commanded_path, actual_path = _build_run(
        tmp_path, 'run', t, xyz, actual_xyz_fn=lambda tt: (tt, 0.0, 0.0),
    )
    actual_path.unlink()
    errors, reason = etc.load_run(commanded_path, actual_path)
    assert errors is None
    assert reason == 'missing actual file'


def test_load_run_excludes_when_arrival_time_outside_actual_range(tmp_path):
    t, xyz = _reference_mission()
    commanded_path, actual_path = _build_run(
        tmp_path, 'run', t, xyz, actual_xyz_fn=lambda tt: (tt, 0.0, 0.0),
        actual_span=-0.6,  # actual log ends well before the last waypoint's arrival
    )
    errors, reason = etc.load_run(commanded_path, actual_path)
    assert errors is None
    assert 'outside actual log range' in reason


def test_load_run_retains_large_but_finite_errors(tmp_path):
    t, xyz = _reference_mission()
    commanded_path, actual_path = _build_run(
        tmp_path, 'run', t, xyz, actual_xyz_fn=lambda tt: (tt + 5.0, 0.0, 0.0),
    )
    errors, reason = etc.load_run(commanded_path, actual_path)
    assert reason is None
    assert np.allclose(errors[:, 0], 5.0, atol=1e-6)


# ── align_runs: waypoint-index alignment across multiple runs, discovery ────
def test_align_runs_orders_by_file_position_not_row_count(tmp_path, monkeypatch):
    monkeypatch.setattr(etc, 'LOGS_DIR', tmp_path)
    t, xyz = _reference_mission()
    prefix = etc._log_prefix('deterministic', 'figure8', 2)
    _build_run(tmp_path, f'{prefix}01', t, xyz, actual_xyz_fn=lambda tt: (tt + 0.1, 0.0, 0.0))
    _build_run(tmp_path, f'{prefix}02', t, xyz, actual_xyz_fn=lambda tt: (tt + 0.3, 0.0, 0.0))

    errors, included, excluded = etc.align_runs(2)
    assert excluded == []
    assert len(included) == 2
    assert errors.shape == (2, 5, 3)
    assert np.allclose(errors[0, :, 0], 0.1, atol=1e-6)
    assert np.allclose(errors[1, :, 0], 0.3, atol=1e-6)


# ── compute_mean_and_covariance / fit_sigma0_q: pure numpy, no CSVs ─────────
def test_known_mean_error_recovery():
    # 3 runs, constant per-run offset (1.0, -1.0, 2.0) applied identically at
    # every waypoint -- mean_error must recover that constant exactly.
    k, n = 4, 3
    offset = np.array([1.0, -1.0, 2.0])
    errors = np.tile(offset, (n, k, 1)) + np.array([[0.1, -0.1, 0.0]])[:, None, :] * np.arange(n)[:, None, None]
    mean_error, _cov = etc.compute_mean_and_covariance(errors)
    expected_mean = offset + np.array([0.1, -0.1, 0.0]) * (n - 1) / 2
    assert np.allclose(mean_error, expected_mean, atol=1e-8)


def test_known_covariance_recovery():
    # Hand-computable sample covariance: for a single waypoint (k=1), 3 runs
    # with error x in {-1, 0, 1} (mean 0), y and z constant (0 variance).
    errors = np.array([
        [[-1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]],
    ])  # shape [N=3, K=1, 3]
    mean_error, covariance_raw = etc.compute_mean_and_covariance(errors)
    assert np.allclose(mean_error, [[0.0, 0.0, 0.0]])
    # sample variance of [-1,0,1] with ddof=1 is 1.0
    assert covariance_raw[0][0, 0] == pytest.approx(1.0)
    assert covariance_raw[0][1, 1] == pytest.approx(0.0)
    assert covariance_raw[0][2, 2] == pytest.approx(0.0)
    assert np.allclose(covariance_raw[0], covariance_raw[0].T)


def test_fit_recovers_known_linear_variance_growth():
    k = 10
    sigma0, q_var = 0.002, 0.0005
    var_isotropic = sigma0 + q_var * np.arange(k)
    covariance_raw = np.stack([np.eye(3) * v for v in var_isotropic])
    fit = etc.fit_sigma0_q(covariance_raw)
    assert fit.sigma0 == pytest.approx(sigma0, abs=1e-9)
    assert fit.q_var == pytest.approx(q_var, abs=1e-9)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-6)
    assert fit.residual_rms == pytest.approx(0.0, abs=1e-9)


def test_fit_reports_poor_fit_for_nonlinear_data():
    k = 10
    # Oscillating, non-linear-in-k variance -- a linear fit should NOT claim
    # a good match.
    var_isotropic = 0.01 + 0.05 * np.abs(np.sin(np.arange(k)))
    covariance_raw = np.stack([np.eye(3) * v for v in var_isotropic])
    fit = etc.fit_sigma0_q(covariance_raw)
    assert fit.r_squared < 0.5


# ── calibrate_fan / verdict / YAML output ───────────────────────────────────
def test_calibrate_fan_zero_runs_no_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(etc, 'LOGS_DIR', tmp_path)
    result = etc.calibrate_fan(2)
    assert result['n_runs'] == 0
    assert result['sigma0_fit'] is None
    assert etc.fit_quality_verdict(result) == 'NO DATA'


def test_fit_quality_verdict_flags_negative_params():
    result = {'r_squared': 0.9, 'sigma0_fit': -0.001, 'q_var_fit': 0.0001}
    assert etc.fit_quality_verdict(result) == 'POOR FIT'


def test_yaml_round_trip(tmp_path):
    results = {
        2: {
            'n_runs': 12, 'sigma0_fit': 0.001, 'q_var_fit': 0.0001, 'q_std_fit': 0.01,
            'r_squared': 0.87, 'residual_rms': 0.0002,
        },
    }
    out_path = tmp_path / 'calibrated_uncertainty.yml'
    etc.write_calibrated_uncertainty_yml(results, out_path)

    data = yaml.safe_load(out_path.read_text())
    assert data['scenario'] == 'figure8'
    assert data['condition'] == 'deterministic'
    assert data['fans'][2]['n_runs'] == 12
    assert data['fans'][2]['sigma0_fit'] == pytest.approx(0.001)
    assert data['fans'][2]['fit_quality'] == 'GOOD FIT'
