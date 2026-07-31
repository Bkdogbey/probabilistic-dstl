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
    assert result['mean_error'] is None
    assert result['bias_inclusive_variance'] is None
    assert result['n_attempts'] == 0


def test_fit_quality_verdict_flags_negative_params():
    result = {'r_squared': 0.9, 'sigma0_fit': -0.001, 'q_var_fit': 0.0001}
    assert etc.fit_quality_verdict(result) == 'POOR FIT'


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


# ── bias-inclusive shared fit and bootstrap ──────────────────────────────
def test_bias_inclusive_variance_preserves_systematic_error():
    errors = np.full((4, 3, 3), 0.2)
    mean, covariance, second_moment, variance = etc.compute_error_statistics(errors)

    assert np.allclose(mean, 0.2)
    assert np.allclose(covariance, 0.0)
    assert np.allclose(np.diagonal(second_moment, axis1=1, axis2=2), 0.04)
    assert np.allclose(variance, 0.04)


def test_shared_fit_recovers_fan_intercepts_and_one_q():
    k = np.arange(12, dtype=float)
    variances = {
        2: 0.001 + 0.0002 * k,
        6: 0.004 + 0.0002 * k,
        12: 0.009 + 0.0002 * k,
        16: 0.016 + 0.0002 * k,
    }

    fit = etc.fit_shared_sigma0_q(variances)

    assert fit.q_var == pytest.approx(0.0002, abs=1e-12)
    assert fit.q_std == pytest.approx(np.sqrt(0.0002), abs=1e-12)
    for fan, expected in ((2, 0.001), (6, 0.004), (12, 0.009), (16, 0.016)):
        assert fit.sigma0_by_fan[fan] == pytest.approx(expected, abs=1e-12)
    assert fit.r_squared == pytest.approx(1.0)


def test_shared_fit_never_returns_negative_parameters():
    fit = etc.fit_shared_sigma0_q({
        2: np.array([0.5, 0.3, 0.1]),
        6: np.array([0.2, 0.1, 0.0]),
    })

    assert fit.q_var >= 0.0
    assert all(value >= 0.0 for value in fit.sigma0_by_fan.values())


def test_binned_diagnostic_accepts_noisy_data_from_true_linear_model():
    rng = np.random.default_rng(42)
    variances = {}
    for fan, sigma in zip(etc.VALID_FANS, (0.02, 0.03, 0.04, 0.05)):
        k = np.arange(100)
        errors = rng.normal(
            0.03,
            np.sqrt(sigma**2 + k * 1e-5)[None, :, None],
            size=(20, 100, 3),
        )
        variances[fan] = np.mean(errors**2, axis=(0, 2))

    fit = etc.fit_shared_sigma0_q(variances)

    assert fit.r_squared >= etc.MIN_R_SQUARED
    assert fit.q_var == pytest.approx(1e-5, rel=0.15)


def test_bootstrap_is_reproducible_and_has_all_intervals():
    rng = np.random.default_rng(10)
    errors = {
        fan: rng.normal(0.05, 0.01 + fan / 1000, size=(20, 6, 3))
        for fan in (2, 6)
    }

    first = etc.bootstrap_joint_fit(errors, samples=40, seed=123)
    second = etc.bootstrap_joint_fit(errors, samples=40, seed=123)

    assert first == second
    assert first['samples_valid'] == 40
    assert set(first['sigma0_ci95']) == {2, 6}
    assert first['q_std_ci95']['lower'] <= first['q_std_ci95']['upper']


def _campaign_result(fan, n_runs, excluded=()):
    return {
        'fan': fan,
        'n_runs': n_runs,
        'n_attempts': n_runs + len(excluded),
        'included_runs': [f'run-{index}' for index in range(n_runs)],
        'excluded_runs': list(excluded),
        'mean_error': [],
        'centered_covariance': [],
        'second_moment': [],
        'bias_inclusive_variance': [],
        '_errors': np.empty((n_runs, 5, 3)),
    }


def _acceptable_fit_and_bootstrap():
    fit = etc.JointFitResult(
        sigma0_by_fan={fan: 0.001 * fan for fan in etc.VALID_FANS},
        q_var=0.0001,
        q_std=0.01,
        r_squared=max(0.9, etc.MIN_R_SQUARED),
        residual_rms=0.00001,
    )
    bootstrap = {
        'seed': 1,
        'samples_requested': etc.BOOTSTRAP_SAMPLES,
        'samples_valid': etc.BOOTSTRAP_SAMPLES,
        'sigma0_ci95': {
            fan: {'lower': 0.0005 * fan, 'upper': 0.0015 * fan}
            for fan in etc.VALID_FANS
        },
        'q_var_ci95': {'lower': 0.00005, 'upper': 0.00015},
        'q_std_ci95': {'lower': 0.007, 'upper': 0.013},
    }
    return fit, bootstrap


def test_final_report_requires_20_valid_runs_for_every_fan(monkeypatch):
    monkeypatch.setattr(etc, 'ACTIVE_CAMPAIGN', 'final')
    fit, bootstrap = _acceptable_fit_and_bootstrap()
    results = {
        fan: _campaign_result(fan, 20 if fan != 16 else 19)
        for fan in etc.VALID_FANS
    }

    report = etc.build_campaign_report('final', results, fit, bootstrap)

    assert report['accepted'] is False
    assert report['status'] == 'INCOMPLETE_FINAL_DATASET'
    assert report['approved_values'] is None


def test_accepted_final_report_uses_conservative_upper_bounds(monkeypatch):
    monkeypatch.setattr(etc, 'ACTIVE_CAMPAIGN', 'final')
    fit, bootstrap = _acceptable_fit_and_bootstrap()
    results = {fan: _campaign_result(fan, 20) for fan in etc.VALID_FANS}
    results[2]['excluded_runs'] = [('failed-run', 'run marked VIOLATION')]
    results[2]['n_attempts'] = 21

    report = etc.build_campaign_report('final', results, fit, bootstrap)

    assert report['accepted'] is True
    assert report['status'] == 'ACCEPTED'
    assert report['approved_values']['q_std'] == pytest.approx(0.013)
    assert report['approved_values']['sigma0_per_fan'][16] == pytest.approx(0.024)
    assert report['fans'][2]['n_attempts'] == 21
    assert report['fans'][2]['excluded_runs'][0]['run'] == 'failed-run'


def test_poor_fit_never_exposes_approved_values(monkeypatch):
    monkeypatch.setattr(etc, 'ACTIVE_CAMPAIGN', 'final')
    fit, bootstrap = _acceptable_fit_and_bootstrap()
    fit = etc.JointFitResult(
        sigma0_by_fan=fit.sigma0_by_fan,
        q_var=fit.q_var,
        q_std=fit.q_std,
        r_squared=etc.MIN_R_SQUARED - 0.01,
        residual_rms=fit.residual_rms,
    )
    results = {fan: _campaign_result(fan, 20) for fan in etc.VALID_FANS}

    report = etc.build_campaign_report('final', results, fit, bootstrap)

    assert report['accepted'] is False
    assert report['status'] == 'POOR_FIT'
    assert report['approved_values'] is None


def test_pilot_report_is_never_accepted(monkeypatch):
    monkeypatch.setattr(etc, 'ACTIVE_CAMPAIGN', 'pilot')
    fit, bootstrap = _acceptable_fit_and_bootstrap()
    result = {2: _campaign_result(2, etc.PILOT_RUNS)}
    pilot_fit = etc.JointFitResult(
        sigma0_by_fan={2: fit.sigma0_by_fan[2]},
        q_var=fit.q_var,
        q_std=fit.q_std,
        r_squared=fit.r_squared,
        residual_rms=fit.residual_rms,
    )
    bootstrap['sigma0_ci95'] = {2: bootstrap['sigma0_ci95'][2]}

    report = etc.build_campaign_report('pilot', result, pilot_fit, bootstrap)

    assert report['accepted'] is False
    assert report['status'] == 'PILOT_ONLY'
    assert report['approved_values'] is None


# ── pooled Sigma_0 diagnostic (constant per-fan, no k-dependence) ────────
def test_calibrate_fan_pooled_sigma0_matches_mean_of_bias_inclusive_variance(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(etc, 'LOGS_DIR', tmp_path)
    t, xyz = _reference_mission()
    prefix = etc.log_prefix('deterministic', 'figure8', 2)
    _build_run(tmp_path, f'{prefix}01', t, xyz, actual_xyz_fn=lambda tt: (tt + 0.1, 0.0, 0.0))
    _build_run(tmp_path, f'{prefix}02', t, xyz, actual_xyz_fn=lambda tt: (tt + 0.3, 0.0, 0.0))

    result = etc.calibrate_fan(2)

    assert result['pooled_sigma0'] == pytest.approx(
        float(np.mean(result['bias_inclusive_variance']))
    )


def test_calibrate_fan_pooled_sigma0_none_with_zero_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(etc, 'LOGS_DIR', tmp_path)
    result = etc.calibrate_fan(2)
    assert result['pooled_sigma0'] is None


def test_bootstrap_reports_pooled_sigma0_ci_per_fan():
    rng = np.random.default_rng(10)
    errors = {
        fan: rng.normal(0.05, 0.01 + fan / 1000, size=(20, 6, 3))
        for fan in (2, 6)
    }

    bootstrap = etc.bootstrap_joint_fit(errors, samples=40, seed=123)

    assert set(bootstrap['pooled_sigma0_ci95']) == {2, 6}
    for fan in (2, 6):
        interval = bootstrap['pooled_sigma0_ci95'][fan]
        assert interval['lower'] <= interval['upper']


def test_pooled_model_present_and_diagnostic_only(monkeypatch):
    monkeypatch.setattr(etc, 'ACTIVE_CAMPAIGN', 'final')
    fit, bootstrap = _acceptable_fit_and_bootstrap()
    bootstrap['pooled_sigma0_ci95'] = {
        fan: {'lower': 0.001 * fan, 'upper': 0.002 * fan} for fan in etc.VALID_FANS
    }
    results = {fan: _campaign_result(fan, 20) for fan in etc.VALID_FANS}
    for fan, result in results.items():
        result['pooled_sigma0'] = 0.0015 * fan

    report = etc.build_campaign_report('final', results, fit, bootstrap)

    assert report['pooled_model']['sigma0_per_fan'][16] == pytest.approx(0.024)
    assert report['pooled_model']['sigma0_ci95'][16] == {'lower': 0.016, 'upper': 0.032}
    # Pooled model never gates acceptance or feeds approved_values.
    assert report['accepted'] is True
    assert 'pooled' not in report['approved_values']


def test_pooled_model_survives_missing_bootstrap_key(monkeypatch):
    """Report-building callers that hand-build a bootstrap dict without the
    pooled key (as older tests do) must not crash."""
    monkeypatch.setattr(etc, 'ACTIVE_CAMPAIGN', 'final')
    fit, bootstrap = _acceptable_fit_and_bootstrap()
    assert 'pooled_sigma0_ci95' not in bootstrap
    results = {fan: _campaign_result(fan, 20) for fan in etc.VALID_FANS}

    report = etc.build_campaign_report('final', results, fit, bootstrap)

    assert report['pooled_model']['sigma0_ci95'][16] is None
    assert report['pooled_model']['sigma0_per_fan'][16] is None


def test_pilot_report_flags_any_excluded_attempt(monkeypatch):
    monkeypatch.setattr(etc, 'ACTIVE_CAMPAIGN', 'pilot')
    fit, bootstrap = _acceptable_fit_and_bootstrap()
    result = {2: _campaign_result(
        2, etc.PILOT_RUNS, excluded=(('bad-run', 'run marked VIOLATION'),),
    )}
    pilot_fit = etc.JointFitResult(
        sigma0_by_fan={2: fit.sigma0_by_fan[2]}, q_var=fit.q_var,
        q_std=fit.q_std, r_squared=fit.r_squared, residual_rms=fit.residual_rms,
    )
    bootstrap['sigma0_ci95'] = {2: bootstrap['sigma0_ci95'][2]}

    report = etc.build_campaign_report('pilot', result, pilot_fit, bootstrap)

    assert report['status'] == 'PILOT_HAS_EXCLUSIONS'
    assert report['accepted'] is False
