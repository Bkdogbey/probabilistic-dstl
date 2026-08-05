from __future__ import annotations

import json
import sys

import pytest

from experiments.crazyflie import estimate_covariance
from experiments.crazyflie.components import utils as config
from experiments.crazyflie.components.planner import PlanRepository, get_scenario
from experiments.crazyflie.components.utils import uncertainty_metadata, validate_waypoints
from experiments.crazyflie.run import ExperimentRunner, RunRequest


@pytest.fixture(autouse=True)
def _accepted_uncertainty_source(monkeypatch):
    monkeypatch.setitem(
        config._cfg['uncertainty'], 'source_report', 'accepted-test-report.yml',
    )


def _request(action: str) -> RunRequest:
    return RunRequest(action, fan=2, condition='deterministic', scenario='baseline')


@pytest.mark.parametrize('action,method', [
    ('plan', '_plan'), ('fly', '_fly'), ('analyze', '_analyze'),
])
def test_runner_dispatches_one_action(monkeypatch, action, method):
    runner = ExperimentRunner(_request(action))
    calls = []
    for candidate in ('_plan', '_fly', '_analyze'):
        monkeypatch.setattr(runner, candidate, lambda name=candidate: calls.append(name))
    runner.run()
    assert calls == [method]


@pytest.mark.parametrize('action,module_name,service_name', [
    ('plan', 'experiments.crazyflie.components.planner', 'PlanningService'),
    ('analyze', 'experiments.crazyflie.components.analysis', 'AnalysisService'),
])
def test_offline_actions_do_not_import_hardware(
    monkeypatch, action, module_name, service_name,
):
    module = __import__(module_name, fromlist=[service_name])
    monkeypatch.setattr(getattr(module, service_name), 'run', lambda *args, **kwargs: None)
    sys.modules.pop('cflib', None)
    sys.modules.pop('ros_sugar', None)
    ExperimentRunner(_request(action)).run()
    assert 'cflib' not in sys.modules
    assert 'ros_sugar' not in sys.modules


def test_scenarios_share_one_planner_contract():
    for name, dimension in [('baseline', 2), ('figure8', 3)]:
        scenario = get_scenario(name)
        assert scenario.dimension == dimension
        assert scenario.start() == scenario.nominal_waypoints(2)[0]
        scenario.validate(scenario.nominal_waypoints(scenario.flight_points))


def test_one_planner_supports_2d_and_3d():
    for name, dimension in [('baseline', 2), ('figure8', 3)]:
        scenario = get_scenario(name)
        planner, dynamics, environment = scenario.build_planner(2)
        mean, covariance = scenario.initial_belief(2)
        assert dynamics.residual_covariance.shape == (dimension, dimension)
        assert mean.shape == (dimension,)
        assert covariance.shape == (dimension, dimension)
        assert planner.env is environment


def test_figure8_uses_narrow_workspace():
    with pytest.raises(ValueError, match='figure8 workspace'):
        validate_waypoints([(1.2, -1.0, 0.3)], scenario='figure8')


def _save_plan(repository: PlanRepository):
    return repository.save(
        2, 'baseline', [(1.0, -2.0, 0.2), (1.0, 0.0, 0.2)],
        {**uncertainty_metadata(2), 'rho_after': 0.95, 'alpha': 0.90},
    )


def test_plan_signature_and_stale_rejection(tmp_path):
    repository = PlanRepository(tmp_path)
    path = _save_plan(repository)
    repository.require_flyable(2)
    payload = json.loads(path.read_text())
    payload['config_signature'] = 'stale'
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match='Stale'):
        repository.require_flyable(2)


def test_uncertainty_change_invalidates_plan(tmp_path, monkeypatch):
    repository = PlanRepository(tmp_path)
    _save_plan(repository)
    monkeypatch.setitem(
        config._cfg['uncertainty']['stationary_residual_variance_per_fan'],
        2,
        [0.002, 0.002, 0.002],
    )
    with pytest.raises(RuntimeError, match='Stale'):
        repository.require_flyable(2)


def test_response_change_invalidates_plan(tmp_path, monkeypatch):
    repository = PlanRepository(tmp_path)
    _save_plan(repository)
    monkeypatch.setitem(
        config._cfg['uncertainty'],
        'response_time_constant',
        [0.70, 0.53, 0.0],
    )
    with pytest.raises(RuntimeError, match='Stale'):
        repository.require_flyable(2)


def test_plan_below_alpha_is_not_flyable(tmp_path):
    repository = PlanRepository(tmp_path)
    path = _save_plan(repository)
    payload = json.loads(path.read_text())
    payload['rho_after'] = 0.8999
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match='below alpha'):
        repository.require_flyable(2)


def test_provisional_uncertainty_plan_is_not_flyable(tmp_path, monkeypatch):
    monkeypatch.setitem(config._cfg['uncertainty'], 'source_report', None)
    repository = PlanRepository(tmp_path)
    _save_plan(repository)
    with pytest.raises(RuntimeError, match='provisional'):
        repository.require_flyable(2)


def test_legacy_plan_is_rejected(tmp_path):
    repository = PlanRepository(tmp_path)
    repository.path(2).write_text(json.dumps({
        'fan': 2, 'scenario': 'baseline', 'rho_after': 0.5,
        'waypoints': [[1.0, -2.0, 0.2]],
    }))
    with pytest.raises(RuntimeError, match='Legacy'):
        repository.require_flyable(2)


def test_covariance_report_never_changes_config(tmp_path):
    before = config._CONFIG_PATH.read_bytes()
    estimate_covariance.write_covariance_report({}, tmp_path / 'report.yml')
    assert config._CONFIG_PATH.read_bytes() == before
