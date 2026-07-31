"""Offline tests for the smooth 3D figure-eight scenario."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.crazyflie.components.utils import (
    FIG8_DETERMINISTIC_CRUISE_VELOCITY,
    FIG8_FLIGHT_POINTS,
    FIG8_OBSTACLES,
    FIG8_PLOT_POINTS,
    FIG8_T,
    FIG8_X_BOUNDS,
    FIG8_Y_BOUNDS,
    FIG8_Z_BOUNDS,
    SIGMA0_PER_FAN,
    U_MAX,
    VALID_SCENARIOS,
)
from experiments.crazyflie.components.planner import PlanRepository
from experiments.crazyflie.components.planner import (
    build_environment_figure8,
    build_planner_figure8,
    figure8_min_clearances,
    figure8_obstacle_clearance,
    nominal_figure8_waypoints,
    terminal_return_error,
    x0_belief_figure8,
    nominal_initial_controls_figure8,
)


# ── Analytical trajectory ────────────────────────────────────────────────
def test_start_end_and_top_heights():
    wps = nominal_figure8_waypoints(FIG8_T + 1)
    start, end, mid = wps[0], wps[-1], wps[FIG8_T // 2]
    assert start == pytest.approx((0.50, -2.00, 0.20), abs=1e-6)
    assert end == pytest.approx((0.50, -2.00, 0.20), abs=1e-6)
    assert mid == pytest.approx((0.50, 0.00, 0.60), abs=1e-6)


def test_altitude_rises_then_descends_smoothly():
    curve = np.array(nominal_figure8_waypoints(1001))
    z = curve[:, 2]
    assert z.min() == pytest.approx(0.20, abs=1e-6)
    assert z.max() == pytest.approx(0.60, abs=1e-6)
    half = len(z) // 2
    assert np.all(np.diff(z[:half + 1]) >= -1e-12)
    assert np.all(np.diff(z[half:]) <= 1e-12)


def test_altitude_has_zero_slope_at_extrema():
    curve = np.array(nominal_figure8_waypoints(20001))
    for index in (0, 10000, 20000):
        lo = max(index - 2, 0)
        hi = min(index + 2, len(curve) - 1)
        assert abs(curve[hi, 2] - curve[lo, 2]) < 1e-5


def test_canonical_lobe_order():
    curve = np.array(nominal_figure8_waypoints(9))
    expected = np.array([
        (0.50, -2.000, 0.20),
        (0.75, -1.707, 0.205),
        (0.50, -1.000, 0.275),
        (0.25, -0.293, 0.470),
        (0.50, 0.000, 0.600),
        (0.75, -0.293, 0.470),
        (0.50, -1.000, 0.275),
        (0.25, -1.707, 0.205),
        (0.50, -2.000, 0.20),
    ])
    assert curve == pytest.approx(expected, abs=5e-4)


def test_coordinate_ranges_and_finite():
    curve = np.array(nominal_figure8_waypoints(FIG8_PLOT_POINTS))
    assert np.all(np.isfinite(curve))
    assert curve[:, 0].min() >= 0.25 - 1e-3 and curve[:, 0].max() <= 0.75 + 1e-3
    assert curve[:, 1].min() >= -2.00 - 1e-3 and curve[:, 1].max() <= 0.00 + 1e-3
    assert curve[:, 2].min() >= 0.20 - 1e-3 and curve[:, 2].max() <= 0.60 + 1e-3


# ── T+1 states -> T controls contract ────────────────────────────────────
def test_state_and_control_counts():
    wps = nominal_figure8_waypoints(FIG8_T + 1)
    assert len(wps) == FIG8_T + 1
    init_u = nominal_initial_controls_figure8()
    assert tuple(init_u.shape) == (FIG8_T, 3)


def test_no_zero_padding_on_last_control():
    init_u = nominal_initial_controls_figure8()
    last_norm = init_u[-1].norm().item()
    assert last_norm > 0.0


def test_max_control_norm_within_u_max():
    init_u = nominal_initial_controls_figure8()
    norms = init_u.norm(dim=1)
    assert norms.max().item() <= U_MAX


# ── Workspace / clearance ────────────────────────────────────────────────
def test_flight_waypoints_inside_workspace():
    wps = np.array(nominal_figure8_waypoints(FIG8_FLIGHT_POINTS))
    assert len(wps) == 100
    assert np.all(wps[:, 0] >= FIG8_X_BOUNDS[0]) and np.all(wps[:, 0] <= FIG8_X_BOUNDS[1])
    assert np.all(wps[:, 1] >= FIG8_Y_BOUNDS[0]) and np.all(wps[:, 1] <= FIG8_Y_BOUNDS[1])
    assert np.all(wps[:, 2] >= FIG8_Z_BOUNDS[0]) and np.all(wps[:, 2] <= FIG8_Z_BOUNDS[1])


def test_deterministic_flight_uses_good_cruise_speed():
    waypoints = np.array(nominal_figure8_waypoints(FIG8_FLIGHT_POINTS))
    path_length = np.linalg.norm(np.diff(waypoints, axis=0), axis=1).sum()
    duration = path_length / FIG8_DETERMINISTIC_CRUISE_VELOCITY
    assert FIG8_DETERMINISTIC_CRUISE_VELOCITY == pytest.approx(0.30)
    assert 16.0 < duration < 18.0


def test_dense_centerline_does_not_intersect_obstacle_boxes():
    curve = np.array(nominal_figure8_waypoints(FIG8_PLOT_POINTS))
    clearances = figure8_min_clearances(curve)
    assert len(clearances) == len(FIG8_OBSTACLES)
    for name, clearance in clearances.items():
        assert clearance > 0.0, f'{name} clearance not positive: {clearance}'


def test_single_obstacle_clearance_matches_dense_helper():
    curve = np.array(nominal_figure8_waypoints(FIG8_PLOT_POINTS))
    obs = FIG8_OBSTACLES[0]
    c = figure8_obstacle_clearance(curve, obs)
    assert c == pytest.approx(figure8_min_clearances(curve)[obs['name']])


def test_fig8_obstacles_ascending_and_bounded():
    for obs in FIG8_OBSTACLES:
        assert obs['x'][0] < obs['x'][1]
        assert obs['y'][0] < obs['y'][1]
        assert obs['z'][0] < obs['z'][1]


# ── pdSTL environment structure ──────────────────────────────────────────
def test_no_visit_regions():
    env = build_environment_figure8()
    assert env.timed_visit_regions == []
    assert env.visit_regions == []


def test_no_time_windowed_altitude_band():
    env = build_environment_figure8()
    assert env.time_windowed_bounds == []


def test_env_has_no_goal():
    env = build_environment_figure8()
    assert env.goal is None


def test_specification_builds_without_error():
    env = build_environment_figure8()
    spec = env.get_specification(FIG8_T)
    assert spec is not None
    preds = env.get_predicates()
    assert len(preds['timed_visit']) == 0
    assert len(preds['time_windowed_bounds']) == 0
    assert len(preds['obstacles']) == len(FIG8_OBSTACLES)
    assert preds['goal'] is None


# ── Reference-tracking objective (Planner3D._compute_loss) ──────────────
def test_planner_reference_trajectory_attached():
    planner, _dyn, _env = build_planner_figure8()
    assert hasattr(planner, 'reference_trajectory')
    assert tuple(planner.reference_trajectory.shape) == (FIG8_T + 1, 3)
    for key in ('w_ref_xy', 'w_ref_z', 'w_terminal'):
        assert planner.cfg.get(key, 0.0) > 0.0


def test_compute_loss_adds_terms_only_when_reference_set():
    planner, dynamics, env = build_planner_figure8()
    T = planner.T
    mean_trace = torch.zeros(1, T + 1, 3)
    u_seq = torch.zeros(T, 3)
    p_all = torch.tensor(0.5)

    planner.reference_trajectory = None
    loss_without_ref = planner._compute_loss(mean_trace, u_seq, p_all, None)

    planner.reference_trajectory = torch.ones(T + 1, 3)  # differs from mean_trace (zeros)
    loss_with_ref = planner._compute_loss(mean_trace, u_seq, p_all, None)

    assert loss_with_ref.item() > loss_without_ref.item()


# ── x0 belief ─────────────────────────────────────────────────────────────
def test_x0_belief_matches_curve_start():
    mean, cov = x0_belief_figure8(2)
    start = nominal_figure8_waypoints(2)[0]
    assert mean.shape == (3,)
    assert cov.shape == (3, 3)
    assert torch.allclose(mean, torch.tensor(start, dtype=torch.float32), atol=1e-6)
    assert cov[0, 0].item() == pytest.approx(SIGMA0_PER_FAN[2])


# ── Terminal return error ────────────────────────────────────────────────
def test_terminal_return_error_nominal_is_near_zero():
    wps = nominal_figure8_waypoints(FIG8_T + 1)
    assert terminal_return_error(wps) < 1e-6


def test_terminal_return_error_matches_math_dist():
    wps = [(0.0, 0.0, 0.0), (1.0, 2.0, 2.0)]
    assert terminal_return_error(wps) == pytest.approx(3.0)


# ── Saved-plan filename/metadata ─────────────────────────────────────────
def test_waypoints_path_format():
    path = PlanRepository().path(2, 'figure8')
    assert path.name == 'pdstl_figure8_fan2.json'


def test_save_plan_metadata_keys(tmp_path, monkeypatch):
    waypoints = nominal_figure8_waypoints(2)
    repository = PlanRepository(tmp_path / 'waypoints')
    path = repository.save(2, 'figure8', waypoints, {
        'sigma0': 0.001, 'q_std': 0.01, 'rho_before': 0.1, 'rho_after': 0.0,
        'alpha': 0.9, 'T': 5, 'dt': 0.1, 'generated': 'test',
        'return_tolerance': 0.08, 'return_error': 1.4142,
    })

    import json
    data = json.loads(path.read_text())
    required = {
        'fan', 'scenario', 'dt', 'T', 'sigma0', 'q_std',
        'rho_before', 'rho_after', 'alpha', 'generated', 'waypoints',
    }
    assert required.issubset(data.keys())
    assert data['scenario'] == 'figure8'
    assert data['return_tolerance'] == 0.08


# ── Scenario dispatch ─────────────────────────────────────────────────────
def test_scenario_choices():
    assert set(VALID_SCENARIOS) == {'baseline', 'figure8'}
    from experiments.crazyflie.components.planner import get_scenario

    assert {get_scenario(name).name for name in VALID_SCENARIOS} == set(VALID_SCENARIOS)
