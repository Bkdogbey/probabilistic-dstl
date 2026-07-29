"""Unit tests for 3D-aware flight logging and scenario-suffixed filenames,
all offline (no Crazyflie/ROS/radio hardware).
"""

from __future__ import annotations

import pytest

from experiments.crazyflie.components.flight_logger import (
    FlightLogger,
    _FIG8_OBSTACLES,
    _OBSTACLES,
    _inside,
    _log_prefix,
)
from experiments.crazyflie.components import flight_logger


_OBS = (0.0, 1.0, 0.0, 1.0, 0.0, 0.5)  # x0,x1,y0,y1,z0,z1


def test_above_obstacle_is_safe():
    assert _inside(0.5, 0.5, 0.6, _OBS) is False


def test_inside_full_volume_is_unsafe():
    assert _inside(0.5, 0.5, 0.25, _OBS) is True


def test_outside_footprint_is_safe_regardless_of_altitude():
    assert _inside(2.0, 2.0, 0.25, _OBS) is False


def test_below_floor_mounted_obstacle_is_unsafe():
    # z=0 is the floor -- inside a floor-mounted box's z-range [0, height].
    assert _inside(0.5, 0.5, 0.0, _OBS) is True


def test_baseline_cruise_altitude_matches_2d_check_for_arena_obstacles():
    """Baseline flies at a constant Z_HEIGHT below every current arena
    obstacle's height, so the new 3D check must agree with what the old
    x,y-only check would have given -- this is the regression guarantee
    that 3D volume checking does not change baseline behavior.
    """
    from experiments.crazyflie.components.utils import OBSTACLES, Z_HEIGHT

    for obs in OBSTACLES:
        assert obs['z'][0] <= Z_HEIGHT <= obs['z'][1], (
            f"{obs['name']} height does not bracket Z_HEIGHT; "
            'baseline/3D-check equivalence assumption no longer holds'
        )
        x_mid = (obs['x'][0] + obs['x'][1]) / 2
        y_mid = (obs['y'][0] + obs['y'][1]) / 2
        obs6 = (*obs['x'], *obs['y'], *obs['z'])
        assert _inside(x_mid, y_mid, Z_HEIGHT, obs6) is True


def test_log_prefix_baseline_has_no_scenario_suffix():
    assert _log_prefix('pdstl', 'baseline', 2) == 'pdstl_fan02_run'


def test_log_prefix_scenario_suffixed():
    assert _log_prefix('deterministic', 'figure8', 12) == 'deterministic_figure8_fan12_run'


def test_flight_logger_baseline_filename_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(flight_logger, '_LOGS_DIR', tmp_path)
    logger = FlightLogger('pdstl', fan_speed=2)  # scenario defaults to 'baseline'
    logger.start()
    logger.log_waypoint(0.0, 0.0, 0.0)
    logger.log_waypoint(1.0, 1.0, 1.0)
    logger._actual = list(logger._commanded)  # seed actual so save() doesn't discard
    paths = logger.save()
    assert paths is not None
    cmd_path, act_path = paths
    assert cmd_path.name.startswith('pdstl_fan02_run01')
    assert 'baseline' not in cmd_path.name


def test_flight_logger_figure8_filename_suffixed(tmp_path, monkeypatch):
    monkeypatch.setattr(flight_logger, '_LOGS_DIR', tmp_path)
    logger = FlightLogger('pdstl', fan_speed=2, scenario='figure8')
    logger.start()
    logger.log_waypoint(0.0, 0.0, 0.0)
    logger.log_waypoint(1.0, 1.0, 1.0)
    logger._actual = list(logger._commanded)
    paths = logger.save()
    assert paths is not None
    cmd_path, _act_path = paths
    assert cmd_path.name.startswith('pdstl_figure8_fan02_run01')


def test_flight_logger_row_includes_scenario(tmp_path, monkeypatch):
    monkeypatch.setattr(flight_logger, '_LOGS_DIR', tmp_path)
    logger = FlightLogger('pdstl', fan_speed=2, scenario='figure8')
    logger.start()
    logger.log_waypoint(0.5, -2.0, 0.25)
    assert logger._commanded[0]['scenario'] == 'figure8'


def test_figure8_logger_uses_all_five_scenario_obstacles():
    logger = FlightLogger('deterministic', fan_speed=2, scenario='figure8')
    assert logger.obstacles == _FIG8_OBSTACLES
    assert len(logger.obstacles) == 5
    logger.start()
    logger.log_waypoint(0.5, -2.0, 0.25)
    row = logger._commanded[0]
    assert all(f'outside_obs{i}' in row for i in range(1, 6))


def test_baseline_logger_keeps_the_three_arena_obstacles():
    logger = FlightLogger('deterministic', fan_speed=2, scenario='baseline')
    assert logger.obstacles == _OBSTACLES
    assert len(logger.obstacles) == 3


def test_run_number_scoped_independently_per_scenario(tmp_path, monkeypatch):
    monkeypatch.setattr(flight_logger, '_LOGS_DIR', tmp_path)

    def _one_run(scenario):
        logger = FlightLogger('pdstl', fan_speed=2, scenario=scenario)
        logger.start()
        logger.log_waypoint(0.0, 0.0, 0.0)
        logger.log_waypoint(1.0, 1.0, 1.0)
        logger._actual = list(logger._commanded)
        return logger.save()

    baseline_paths = _one_run('baseline')
    figure8_paths = _one_run('figure8')
    assert 'run01' in baseline_paths[0].name
    assert 'run01' in figure8_paths[0].name  # independent counters, not run02


def test_log_name_regex_parses_scenario_suffixed():
    from experiments.crazyflie.components import flight_logger as log_module

    m = log_module._LOG_NAME_RE.match('pdstl_figure8_fan12_run03_VIOLATION_20260101T000000_actual.csv')
    assert m is not None
    assert m.group('condition') == 'pdstl'
    assert m.group('scenario') == 'figure8'
    assert m.group('fan') == '12'
    assert m.group('run') == '03'


def test_log_name_regex_parses_legacy_baseline():
    from experiments.crazyflie.components import flight_logger as log_module

    m = log_module._LOG_NAME_RE.match('deterministic_fan06_run01_20260101T000000_actual.csv')
    assert m is not None
    assert m.group('condition') == 'deterministic'
    assert m.group('scenario') is None
    assert m.group('fan') == '06'


def test_go_to_accepts_velocity_kwarg():
    pytest.importorskip('cflib')
    import inspect
    from cflib.positioning.position_hl_commander import PositionHlCommander

    sig = inspect.signature(PositionHlCommander.go_to)
    assert 'velocity' in sig.parameters
