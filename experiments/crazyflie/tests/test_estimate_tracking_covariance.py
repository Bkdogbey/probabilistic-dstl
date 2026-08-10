"""Unit tests for the offline figure8 tracking-covariance calibration script,
all synthetic (no Crazyflie/ROS/radio hardware, no real flight logs).
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
import yaml

from experiments.crazyflie import estimate_covariance as etc


def _commanded_row(t, x, y, z):
    return {
        'condition': 'deterministic', 'scenario': 'figure8',
        'campaign': etc.ACTIVE_CAMPAIGN, 'profile_signature': etc.PROFILE_SIGNATURE,
        't': t, 'x': x, 'y': y, 'z': z,
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
    monkeypatch.setattr(etc, '_plausible_actual_positions', lambda _xyz: True)


def _reference_mission(k=5):
    t = np.arange(k, dtype=float)  # 0,1,2,3,4
    xyz = np.column_stack([np.arange(k, dtype=float), np.zeros(k), np.zeros(k)])  # (0,0,0),(1,0,0),...
    return t, xyz


def _synthetic_response_runs(
    *, fans=(2, 6), n_runs=20, k=12, tau=(0.4, 0.7, 0.0), seed=10,
):
    """Build aligned runs with known first-order response and small noise."""
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, 3.0, k)
    phase = np.linspace(0.0, 2.0 * np.pi, k)
    commands = np.column_stack([
        0.3 * np.sin(phase),
        -1.0 + 0.5 * np.cos(phase),
        0.3 + 0.1 * np.sin(phase / 2.0),
    ])
    mean = etc.predict_response(times, commands, tau)
    return {
        fan: [
            etc.AlignedRun(
                times.copy(), commands.copy(),
                mean + rng.normal(0.0, 0.004 + fan / 10000.0, size=mean.shape),
            )
            for _ in range(n_runs)
        ]
        for fan in fans
    }


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


def test_load_run_holds_last_sample_for_small_terminal_gap(tmp_path):
    t, xyz = _reference_mission()
    commanded_path, actual_path = _build_run(
        tmp_path, 'run', t, xyz, actual_xyz_fn=lambda tt: (tt, 0.0, 0.0),
    )
    actual_rows = [
        row for row in etc._read_csv(actual_path)
        if float(row['t']) <= t[-1] - etc._TERMINAL_HOLD_TOLERANCE / 2
    ]
    _write_csv(actual_path, actual_rows)

    errors, reason = etc.load_run(commanded_path, actual_path)

    assert reason is None
    assert errors is not None
    assert errors[-1, 0] < 0.0
    assert abs(errors[-1, 0]) <= etc._TERMINAL_HOLD_TOLERANCE


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
    prefix = etc.log_prefix('deterministic', 'figure8', 2)
    _build_run(tmp_path, f'{prefix}01', t, xyz, actual_xyz_fn=lambda tt: (tt + 0.1, 0.0, 0.0))
    _build_run(tmp_path, f'{prefix}02', t, xyz, actual_xyz_fn=lambda tt: (tt + 0.3, 0.0, 0.0))

    errors, included, excluded = etc.align_runs(2)
    assert excluded == []
    assert len(included) == 2
    assert errors.shape == (2, 5, 3)
    assert np.allclose(errors[0, :, 0], 0.1, atol=1e-6)
    assert np.allclose(errors[1, :, 0], 0.3, atol=1e-6)


# ── centered tracking-error statistics: pure numpy, no CSVs ─────────────
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


def test_response_fit_recovers_horizontal_time_constants():
    runs = _synthetic_response_runs(n_runs=8, k=30, tau=(0.4, 0.7, 0.0))

    fitted = etc.fit_response_model(runs)

    assert fitted['time_constant'][:2] == pytest.approx([0.4, 0.7], abs=0.04)
    assert fitted['time_constant'][2] == 0.0
    assert fitted['r_squared'][0] > 0.95
    assert fitted['r_squared'][1] > 0.95
    assert fitted['acceptable'] is True


def test_response_residuals_remove_repeatable_lag_not_random_spread():
    runs = _synthetic_response_runs(fans=(2,), n_runs=10, k=30)
    raw = np.stack([run.tracking_error for run in runs[2]])
    fitted = etc.fit_response_model(runs)
    residual = etc.response_residuals(runs[2], fitted)

    assert np.mean(residual[:, :, :2] ** 2) < np.mean(raw[:, :, :2] ** 2) / 10.0
    # A shared deterministic mean correction cannot erase between-run noise.
    assert np.mean(np.var(residual, axis=0, ddof=1)) > 0.0


# ── calibrate_fan / verdict / YAML output ───────────────────────────────────
def test_calibrate_fan_zero_runs_no_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(etc, 'LOGS_DIR', tmp_path)
    result = etc.calibrate_fan(2)
    assert result['n_runs'] == 0
    assert result['mean_residual'] is None
    assert result['bias_inclusive_residual_mse'] is None
    assert result['n_attempts'] == 0


def test_yaml_round_trip(tmp_path):
    payload = {
        'scenario': 'figure8', 'condition': 'deterministic', 'accepted': False,
        'fans': {2: {'n_runs': 5}},
    }
    out_path = tmp_path / 'covariance_report.yml'
    etc.write_covariance_report(payload, out_path)

    data = yaml.safe_load(out_path.read_text())
    assert data['scenario'] == 'figure8'
    assert data['condition'] == 'deterministic'
    assert data['fans'][2]['n_runs'] == 5
    assert data['accepted'] is False


# ── provenance and strict attempt accounting ─────────────────────────────
def test_load_run_rejects_stale_profile_signature(tmp_path):
    t, xyz = _reference_mission()
    commanded_path, actual_path = _build_run(
        tmp_path, 'run', t, xyz, actual_xyz_fn=lambda tt: (tt, 0.0, 0.0),
    )
    rows = etc._read_csv(actual_path)
    rows[0]['profile_signature'] = 'stale'
    _write_csv(actual_path, rows)

    errors, reason = etc.load_run(commanded_path, actual_path)

    assert errors is None
    assert 'profile_signature mismatch' in reason


def test_load_run_rejects_wrong_campaign(tmp_path):
    t, xyz = _reference_mission()
    commanded_path, actual_path = _build_run(
        tmp_path, 'run', t, xyz, actual_xyz_fn=lambda tt: (tt, 0.0, 0.0),
    )
    rows = etc._read_csv(commanded_path)
    for row in rows:
        row['campaign'] = 'final' if etc.ACTIVE_CAMPAIGN == 'pilot' else 'pilot'
    _write_csv(commanded_path, rows)

    errors, reason = etc.load_run(commanded_path, actual_path)

    assert errors is None
    assert 'campaign mismatch' in reason


def test_load_run_rejects_unsafe_actual_sample(tmp_path):
    t, xyz = _reference_mission()
    commanded_path, actual_path = _build_run(
        tmp_path, 'run', t, xyz, actual_xyz_fn=lambda tt: (tt, 0.0, 0.0),
    )
    rows = etc._read_csv(actual_path)
    rows[len(rows) // 2]['safe'] = '0'
    _write_csv(actual_path, rows)

    errors, reason = etc.load_run(commanded_path, actual_path)

    assert errors is None
    assert reason == 'actual trajectory contains an unsafe sample'


def test_discovery_accounts_for_commanded_only_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(etc, 'LOGS_DIR', tmp_path)
    t, xyz = _reference_mission()
    prefix = etc.log_prefix('deterministic', 'figure8', 2)
    _commanded, actual = _build_run(
        tmp_path, f'{prefix}01', t, xyz, actual_xyz_fn=lambda tt: (tt, 0.0, 0.0),
    )
    actual.unlink()

    errors, included, excluded = etc.align_runs(2)

    assert errors.shape == (0, 5, 3)
    assert included == []
    assert len(excluded) == 1
    assert excluded[0][1] == 'missing actual file'


def test_load_run_rejects_implausible_position(tmp_path, monkeypatch):
    monkeypatch.setattr(etc, '_plausible_actual_positions', lambda _xyz: False)
    t, xyz = _reference_mission()
    commanded_path, actual_path = _build_run(
        tmp_path, 'run', t, xyz, actual_xyz_fn=lambda tt: (tt, 0.0, 0.0),
    )

    errors, reason = etc.load_run(commanded_path, actual_path)

    assert errors is None
    assert 'plausible workspace envelope' in reason


# ── stationary empirical covariance and bootstrap ────────────────────
def test_bias_inclusive_mse_preserves_systematic_error():
    errors = np.full((4, 3, 3), 0.2)
    mean, covariance, second_moment, mse = etc.compute_error_statistics(errors)

    assert np.allclose(mean, 0.2)
    assert np.allclose(covariance, 0.0)
    assert np.allclose(np.diagonal(second_moment, axis1=1, axis2=2), 0.04)
    assert np.allclose(mse, 0.04)


def test_planner_variance_preserves_unmodeled_phase_bias():
    residuals = np.zeros((4, 2, 3))
    residuals[:, 0, 0] = -0.2
    residuals[:, 1, 0] = 0.2

    stats = etc.planner_residual_statistics(residuals)

    assert stats['pooled_mean'][0] == pytest.approx(0.0)
    assert stats['centered_stationary_variance'][0] == pytest.approx(0.0)
    assert stats['stationary_variance'][0] == pytest.approx(0.04)
    assert stats['phase_fraction'][0] == pytest.approx(1.0)


def test_stationarity_accepts_constant_per_axis_variance(monkeypatch):
    monkeypatch.setattr(etc, 'STATIONARITY_BIN_SIZE', 2)
    covariance = np.stack([np.diag([0.01, 0.02, 0.03])] * 6)

    diagnostics = etc.stationarity_diagnostics(covariance)

    assert diagnostics['acceptable'] is True
    assert diagnostics['max_bin_to_pooled_ratio'] == pytest.approx([1.0, 1.0, 1.0])


def test_stationarity_rejects_local_variance_spike(monkeypatch):
    monkeypatch.setattr(etc, 'STATIONARITY_BIN_SIZE', 2)
    covariance = np.stack([np.diag([0.01, 0.01, 0.01])] * 10)
    covariance[:2, 0, 0] = 0.20

    diagnostics = etc.stationarity_diagnostics(covariance)

    assert diagnostics['acceptable'] is False
    assert diagnostics['max_bin_to_pooled_ratio'][0] > etc.MAX_STATIONARITY_RATIO


def test_bootstrap_is_reproducible_and_has_all_intervals():
    runs = _synthetic_response_runs(n_runs=8)

    first = etc.bootstrap_response_model(runs, samples=12, seed=123)
    second = etc.bootstrap_response_model(runs, samples=12, seed=123)

    assert first == second
    assert first['samples_valid'] == 12
    assert set(first['stationary_residual_variance_ci95']) == {2, 6}
    assert set(first['stationary_residual_variance_ci95'][2]) == set(etc.AXES)
    assert first['stationary_residual_variance_ci95'][2]['x']['lower'] <= (
        first['stationary_residual_variance_ci95'][2]['x']['upper']
    )
    assert set(first['response_time_constant_ci95']) == set(etc.AXES)


def _campaign_result(fan, n_runs, excluded=()):
    return {
        'fan': fan,
        'n_runs': n_runs,
        'n_attempts': n_runs + len(excluded),
        'included_runs': [f'run-{index}' for index in range(n_runs)],
        'excluded_runs': list(excluded),
        'raw_mean_tracking_error': [],
        'raw_bias_inclusive_mse': [],
        'mean_residual': [],
        'centered_residual_covariance': [],
        'residual_second_moment': [],
        'bias_inclusive_residual_mse': [],
        'pooled_residual_mean': [0.01, -0.01, 0.0],
        'stationary_residual_variance': [0.001 * fan] * 3,
        'residual_stationarity': {
            'acceptable': True,
            'max_bin_to_pooled_ratio': [1.0, 1.0, 1.0],
        },
        '_runs': [],
        '_residuals': np.empty((n_runs, 5, 3)),
    }


def _acceptable_response():
    return {
        'enabled_axes': {'x': True, 'y': True, 'z': False},
        'time_constant': [0.5, 0.55, 0.0],
        'r_squared': [0.85, 0.90, 0.0],
        'minimum_r_squared': etc.MIN_RESPONSE_R_SQUARED,
        'time_constant_bounds': list(etc.RESPONSE_TIME_CONSTANT_BOUNDS),
        'acceptable': True,
    }


def _acceptable_bootstrap():
    return {
        'seed': 1,
        'samples_requested': etc.BOOTSTRAP_SAMPLES,
        'samples_valid': etc.BOOTSTRAP_SAMPLES,
        'response_time_constant_ci95': {
            'x': {'lower': 0.4, 'upper': 0.6},
            'y': {'lower': 0.45, 'upper': 0.65},
            'z': {'lower': 0.0, 'upper': 0.0},
        },
        'response_r_squared_ci95': {
            axis: {'lower': 0.0, 'upper': 1.0} for axis in etc.AXES
        },
        'stationary_residual_variance_ci95': {
            fan: {
                axis: {'lower': 0.0005 * fan, 'upper': 0.0015 * fan}
                for axis in etc.AXES
            }
            for fan in etc.VALID_FANS
        },
        'pooled_residual_mean_ci95': {
            fan: {
                axis: {'lower': -0.02, 'upper': 0.02} for axis in etc.AXES
            }
            for fan in etc.VALID_FANS
        },
    }


def test_final_report_requires_20_valid_runs_for_every_fan(monkeypatch):
    monkeypatch.setattr(etc, 'ACTIVE_CAMPAIGN', 'final')
    bootstrap = _acceptable_bootstrap()
    results = {
        fan: _campaign_result(fan, 20 if fan != 16 else 19)
        for fan in etc.VALID_FANS
    }

    report = etc.build_campaign_report('final', results, _acceptable_response(), bootstrap)

    assert report['accepted'] is False
    assert report['status'] == 'INCOMPLETE_FINAL_DATASET'
    assert report['approved_values'] is None


def test_accepted_final_report_uses_conservative_upper_bounds(monkeypatch):
    monkeypatch.setattr(etc, 'ACTIVE_CAMPAIGN', 'final')
    bootstrap = _acceptable_bootstrap()
    results = {fan: _campaign_result(fan, 20) for fan in etc.VALID_FANS}
    results[2]['excluded_runs'] = [('failed-run', 'run marked VIOLATION')]
    results[2]['n_attempts'] = 21

    report = etc.build_campaign_report('final', results, _acceptable_response(), bootstrap)

    assert report['accepted'] is True
    assert report['status'] == 'ACCEPTED'
    assert report['approved_values']['initial_variance'] == 0.0
    assert report['approved_values']['stationary_residual_variance_per_fan'][16] == (
        pytest.approx([0.024, 0.024, 0.024])
    )
    assert set(report['approved_values']) == {
        'model', 'initial_variance', 'response_enabled_axes',
        'response_time_constant', 'stationary_residual_variance_per_fan',
        'residual_mean_per_fan',
    }
    assert report['fans'][2]['n_attempts'] == 21
    assert report['fans'][2]['excluded_runs'][0]['run'] == 'failed-run'


def test_nonstationary_variance_never_exposes_approved_values(monkeypatch):
    monkeypatch.setattr(etc, 'ACTIVE_CAMPAIGN', 'final')
    bootstrap = _acceptable_bootstrap()
    results = {fan: _campaign_result(fan, 20) for fan in etc.VALID_FANS}
    results[12]['residual_stationarity']['acceptable'] = False

    report = etc.build_campaign_report('final', results, _acceptable_response(), bootstrap)

    assert report['accepted'] is False
    assert report['status'] == 'MODEL_REJECTED'
    assert report['approved_values'] is None


def test_poor_response_fit_never_exposes_approved_values(monkeypatch):
    monkeypatch.setattr(etc, 'ACTIVE_CAMPAIGN', 'final')
    bootstrap = _acceptable_bootstrap()
    response = _acceptable_response()
    response['r_squared'][0] = 0.10
    response['acceptable'] = False
    results = {fan: _campaign_result(fan, 20) for fan in etc.VALID_FANS}

    report = etc.build_campaign_report('final', results, response, bootstrap)

    assert report['accepted'] is False
    assert report['status'] == 'MODEL_REJECTED'
    assert report['approved_values'] is None


def test_pilot_report_is_never_accepted(monkeypatch):
    monkeypatch.setattr(etc, 'ACTIVE_CAMPAIGN', 'pilot')
    result = {2: _campaign_result(2, etc.PILOT_RUNS)}

    report = etc.build_campaign_report('pilot', result, _acceptable_response(), None)

    assert report['accepted'] is False
    assert report['status'] == 'PILOT_ONLY'
    assert report['approved_values'] is None


# ── stationary fan model and strict report accounting ─────────────────
def test_calibrate_fan_separates_residual_mean_from_stationary_variance(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(etc, 'LOGS_DIR', tmp_path)
    monkeypatch.setattr(etc, 'RESPONSE_ENABLED_AXES', (False, False, False))
    t, xyz = _reference_mission()
    prefix = etc.log_prefix('deterministic', 'figure8', 2)
    _build_run(tmp_path, f'{prefix}01', t, xyz, actual_xyz_fn=lambda tt: (tt + 0.1, 0.0, 0.0))
    _build_run(tmp_path, f'{prefix}02', t, xyz, actual_xyz_fn=lambda tt: (tt + 0.3, 0.0, 0.0))

    result = etc.calibrate_fan(2)

    assert result['pooled_residual_mean'] == pytest.approx([0.2, 0.0, 0.0], abs=0.01)
    assert result['stationary_residual_variance'] == pytest.approx([0.01, 0.0, 0.0])
    assert result['residual_stationarity']['acceptable'] is True


def test_calibrate_fan_stationary_residual_variance_none_with_zero_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(etc, 'LOGS_DIR', tmp_path)
    result = etc.calibrate_fan(2)
    assert result['stationary_residual_variance'] is None


def test_bootstrap_reports_per_axis_variance_and_mean_intervals():
    runs = _synthetic_response_runs(n_runs=8)

    bootstrap = etc.bootstrap_response_model(runs, samples=12, seed=123)

    assert set(bootstrap['stationary_residual_variance_ci95']) == {2, 6}
    for fan in (2, 6):
        for axis in etc.AXES:
            variance = bootstrap['stationary_residual_variance_ci95'][fan][axis]
            mean = bootstrap['pooled_residual_mean_ci95'][fan][axis]
            assert 0.0 <= variance['lower'] <= variance['upper']
            assert mean['lower'] <= mean['upper']


def test_incomplete_attempt_accounting_blocks_acceptance(monkeypatch):
    monkeypatch.setattr(etc, 'ACTIVE_CAMPAIGN', 'final')
    bootstrap = _acceptable_bootstrap()
    results = {fan: _campaign_result(fan, 20) for fan in etc.VALID_FANS}
    results[6]['n_attempts'] = 21

    report = etc.build_campaign_report('final', results, _acceptable_response(), bootstrap)

    assert report['accepted'] is False
    assert report['attempts_accounted_for'] is False
    assert report['status'] == 'INCOMPLETE_ATTEMPT_ACCOUNTING'


def test_missing_bootstrap_interval_blocks_acceptance(monkeypatch):
    monkeypatch.setattr(etc, 'ACTIVE_CAMPAIGN', 'final')
    bootstrap = _acceptable_bootstrap()
    del bootstrap['stationary_residual_variance_ci95'][16]['z']
    results = {fan: _campaign_result(fan, 20) for fan in etc.VALID_FANS}

    report = etc.build_campaign_report('final', results, _acceptable_response(), bootstrap)

    assert report['accepted'] is False
    assert report['status'] == 'MODEL_REJECTED'


def test_pilot_report_flags_any_excluded_attempt(monkeypatch):
    monkeypatch.setattr(etc, 'ACTIVE_CAMPAIGN', 'pilot')
    result = {2: _campaign_result(
        2, etc.PILOT_RUNS, excluded=(('bad-run', 'run marked VIOLATION'),),
    )}

    report = etc.build_campaign_report('pilot', result, _acceptable_response(), None)

    assert report['status'] == 'PILOT_HAS_EXCLUSIONS'
    assert report['accepted'] is False
