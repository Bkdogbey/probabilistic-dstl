"""Offline tests for the smooth 3D figure-eight scenario."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.crazyflie.components import utils as config
from experiments.crazyflie.components.utils import (
    DT,
    FIG8_CHANCE_Y_BOUNDS,
    FIG8_CHANCE_Z_BOUNDS,
    FIG8_CORRIDOR_RADIUS,
    FIG8_DETERMINISTIC_CRUISE_VELOCITY,
    FIG8_FLIGHT_POINTS,
    FIG8_MAX_CONTROL_DELTA,
    FIG8_OBSTACLES,
    FIG8_PLOT_POINTS,
    FIG8_RETURN_TOLERANCE,
    FIG8_T,
    FIG8_X_BOUNDS,
    FIG8_Y_BOUNDS,
    FIG8_Z_BOUNDS,
    INITIAL_VARIANCE,
    RESPONSE_TIME_CONSTANT,
    STATIONARY_RESIDUAL_VARIANCE_PER_FAN,
    U_MAX,
    VALID_SCENARIOS,
    uncertainty_metadata,
)
from experiments.crazyflie.components.planner import PlanRepository
from experiments.crazyflie.components.planner import (
    FigureEightScenario,
    RectangularObstaclePredicate,
    build_environment_figure8,
    build_planner_figure8,
    figure8_min_clearances,
    figure8_obstacle_clearance,
    maximum_control_delta,
    maximum_reference_deviation,
    nominal_figure8_waypoints,
    terminal_return_error,
    x0_belief_figure8,
    nominal_initial_controls_figure8,
)
from pdstl.base import BeliefTrajectory
from planning.environment import normal_cdf
from planning.planner import Planner, TorchGaussianBelief


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


def test_figure8_dynamics_enforces_closed_bounded_control_sequence():
    planner, dynamics, _env = build_planner_figure8()
    unconstrained = torch.randn(FIG8_T, 3)
    controls = dynamics.bound_control(unconstrained)
    mean, covariance = x0_belief_figure8(2)
    trace, _ = dynamics(unconstrained, mean, covariance)
    commanded = dynamics.commanded_trace(trace, controls)
    assert torch.linalg.vector_norm(controls, dim=1).max().item() <= U_MAX + 1e-6
    assert torch.sum(controls, dim=0) == pytest.approx(torch.zeros(3), abs=1e-5)
    assert commanded[0, -1] == pytest.approx(commanded[0, 0], abs=1e-5)


def test_horizontal_mean_lags_command_while_vertical_response_is_instantaneous():
    _planner, dynamics, _env = build_planner_figure8(2)
    mean, covariance = x0_belief_figure8(2)
    physical = torch.zeros(FIG8_T, 3)
    physical[:5] = torch.tensor([0.10, 0.10, 0.10])
    normalized = torch.clamp(physical / dynamics.u_max, -0.99, 0.99)
    unconstrained = torch.atanh(normalized)
    trace, _ = dynamics(unconstrained, mean, covariance)
    commands = dynamics.commanded_trace(trace, dynamics.bound_control(unconstrained))

    assert RESPONSE_TIME_CONSTANT[0] > 0.0
    assert RESPONSE_TIME_CONSTANT[1] > 0.0
    assert RESPONSE_TIME_CONSTANT[2] == 0.0
    assert trace[0, 1, 0] < commands[0, 1, 0]
    assert trace[0, 1, 1] < commands[0, 1, 1]
    assert trace[0, 1, 2] == pytest.approx(commands[0, 1, 2], abs=1e-6)


def test_figure8_residual_covariance_is_zero_then_stationary():
    _planner, dynamics, _env = build_planner_figure8(12)
    mean, covariance = x0_belief_figure8(12)
    _trace, covariance_trace = dynamics(torch.zeros(FIG8_T, 3), mean, covariance)
    expected = torch.diag(torch.tensor(STATIONARY_RESIDUAL_VARIANCE_PER_FAN[12]))

    assert torch.count_nonzero(covariance_trace[0, 0]) == 0
    assert torch.allclose(covariance_trace[0, 1], expected)
    assert torch.allclose(covariance_trace[0, -1], expected)


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


def test_chance_workspace_extends_beyond_commanded_bottom_boundary():
    env = build_environment_figure8()
    assert env.bounds_interval is None
    assert env.bounds['y'] == FIG8_CHANCE_Y_BOUNDS
    assert env.bounds['z'] == FIG8_CHANCE_Z_BOUNDS
    assert env.bounds['y'][0] < FIG8_Y_BOUNDS[0]
    assert env.bounds['z'][0] < FIG8_Z_BOUNDS[0]


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


def test_feasible_candidate_selection_prefers_path_quality():
    assert Planner._candidate_is_better(0.91, 4.0, 0.99, 5.0, 0.90)
    assert not Planner._candidate_is_better(0.99, 6.0, 0.91, 5.0, 0.90)
    assert Planner._candidate_is_better(0.89, 9.0, 0.80, 1.0, 0.90)
    assert not Planner._candidate_is_better(0.89, 0.1, 0.91, 100.0, 0.90)


def test_rectangular_obstacle_probability_is_smooth_outside_union():
    mean = torch.tensor([[0.02, 0.03, 0.04]], requires_grad=True)
    variance = torch.diag(torch.tensor([0.01, 0.01, 0.01])).unsqueeze(0)
    belief = TorchGaussianBelief(mean, variance)
    predicate = RectangularObstaclePredicate({
        'x': [-0.1, 0.1], 'y': [-0.1, 0.1], 'z': [-0.1, 0.1],
    })
    probability = predicate(BeliefTrajectory([belief]))[0, 0, 0]
    diagonal = torch.diagonal(variance, dim1=-2, dim2=-1)
    inside = torch.ones(1)
    for axis, bounds in enumerate(((-0.1, 0.1),) * 3):
        inside = inside * (
            normal_cdf(bounds[1], mean[:, axis], diagonal[:, axis])
            - normal_cdf(bounds[0], mean[:, axis], diagonal[:, axis])
        )
    assert probability.item() == pytest.approx((1.0 - inside).item(), abs=1e-6)
    probability.backward()
    assert torch.all(torch.isfinite(mean.grad))
    assert torch.all(torch.abs(mean.grad) > 0)


# ── x0 belief ─────────────────────────────────────────────────────────────
def test_x0_belief_matches_curve_start():
    mean, cov = x0_belief_figure8(2)
    start = nominal_figure8_waypoints(2)[0]
    assert mean.shape == (3,)
    assert cov.shape == (3, 3)
    assert torch.allclose(mean, torch.tensor(start, dtype=torch.float32), atol=1e-6)
    assert INITIAL_VARIANCE == 0.0
    assert torch.count_nonzero(cov) == 0


# ── Terminal return error ────────────────────────────────────────────────
def test_terminal_return_error_nominal_is_near_zero():
    wps = nominal_figure8_waypoints(FIG8_T + 1)
    assert terminal_return_error(wps) < 1e-6


def test_terminal_return_error_matches_math_dist():
    wps = [(0.0, 0.0, 0.0), (1.0, 2.0, 2.0)]
    assert terminal_return_error(wps) == pytest.approx(3.0)


def test_figure8_validation_rejects_open_path():
    waypoints = nominal_figure8_waypoints(FIG8_T + 1)
    waypoints[-1] = (waypoints[-1][0], waypoints[-1][1], waypoints[-1][2] + 0.10)
    with pytest.raises(ValueError, match='does not close'):
        FigureEightScenario().validate(waypoints)


def test_figure8_validation_rejects_corridor_escape():
    waypoints = nominal_figure8_waypoints(FIG8_T + 1)
    x, y, z = waypoints[25]
    waypoints[25] = (x + FIG8_CORRIDOR_RADIUS + 0.01, y, z)
    with pytest.raises(ValueError, match='leaves figure-eight corridor'):
        FigureEightScenario().validate(waypoints)


def test_figure8_validation_rejects_sharp_control_change():
    waypoints = nominal_figure8_waypoints(FIG8_T + 1)
    x, y, z = waypoints[25]
    waypoints[25] = (x + 0.08, y, z)
    assert maximum_reference_deviation(waypoints) < FIG8_CORRIDOR_RADIUS
    assert maximum_control_delta(waypoints, DT) > FIG8_MAX_CONTROL_DELTA
    with pytest.raises(ValueError, match='not smooth'):
        FigureEightScenario().validate(waypoints)


# ── Saved-plan filename/metadata ─────────────────────────────────────────
def test_waypoints_path_format():
    path = PlanRepository().path(2, 'figure8')
    assert path.name == 'pdstl_figure8_fan2.json'


def test_save_plan_metadata_keys(tmp_path, monkeypatch):
    waypoints = nominal_figure8_waypoints(2)
    repository = PlanRepository(tmp_path / 'waypoints')
    path = repository.save(2, 'figure8', waypoints, {
        **uncertainty_metadata(2),
        'rho_before': 0.1, 'rho_after': 0.0,
        'alpha': 0.9, 'T': 5, 'dt': 0.1, 'generated': 'test',
        'return_tolerance': 0.08, 'return_error': 1.4142,
    })

    import json
    data = json.loads(path.read_text())
    required = {
        'fan', 'scenario', 'dt', 'T', 'uncertainty_model',
        'initial_variance', 'response_enabled_axes', 'response_time_constant',
        'stationary_residual_variance', 'residual_mean', 'source_report',
        'rho_before', 'rho_after', 'alpha', 'generated', 'waypoints',
    }
    assert required.issubset(data.keys())
    assert data['scenario'] == 'figure8'
    assert data['return_tolerance'] == 0.08


def test_flyable_figure8_requires_geometry_metadata(tmp_path, monkeypatch):
    monkeypatch.setitem(config._cfg['uncertainty'], 'source_report', 'accepted-test.yml')
    waypoints = nominal_figure8_waypoints(FIG8_T + 1)
    repository = PlanRepository(tmp_path / 'waypoints')
    metadata = {
        **uncertainty_metadata(2),
        'rho_after': 0.95,
        'alpha': 0.90,
        'return_tolerance': FIG8_RETURN_TOLERANCE,
        'return_error': terminal_return_error(waypoints),
        'corridor_radius': FIG8_CORRIDOR_RADIUS,
        'max_reference_deviation': maximum_reference_deviation(waypoints),
        'max_control_delta_limit': FIG8_MAX_CONTROL_DELTA,
        'max_control_delta': maximum_control_delta(waypoints),
    }
    repository.save(2, 'figure8', waypoints, metadata)
    repository.require_flyable(2, 'figure8')

    metadata.pop('max_control_delta')
    repository.save(2, 'figure8', waypoints, metadata)
    with pytest.raises(RuntimeError, match='geometry validation metadata'):
        repository.require_flyable(2, 'figure8')


# ── Scenario dispatch ─────────────────────────────────────────────────────
def test_scenario_choices():
    assert set(VALID_SCENARIOS) == {'baseline', 'figure8'}
    from experiments.crazyflie.components.planner import get_scenario

    assert {get_scenario(name).name for name in VALID_SCENARIOS} == set(VALID_SCENARIOS)
