from __future__ import annotations

import math
import time

from attrs import define, field
from cflib.positioning.position_hl_commander import PositionHlCommander
from ros_sugar.config import BaseComponentConfig
from ros_sugar.core import BaseComponent

from components import calibration
from components.config import CALIBRATION_HOVER_SECONDS, DRONE_URI, DT, FIG8_FLIGHT_POINTS, FLIGHT_VELOCITY, LAND_Z, POST_GATE_Z, RETURN_Z, SAFE_PATH_FLIGHT_POINTS, START_TOLERANCE, START_XY, TAKEOFF_Z, U_MAX, load_pdstl_waypoints, validate_waypoint_velocities, validate_waypoints_in_bounds
from components.flight_logger import FlightLogger
from components.planning_2d import nominal_safe_waypoints
from components.planning_3d import nominal_figure8_waypoints, nominal_gate_waypoints
from irobot.src.robots.crazyflie.config import CrazyflieConfig as CrazyflieHwConfig
from irobot.src.robots.crazyflie.core.base import CrazyflieBase


def _default_hw_config() -> CrazyflieHwConfig:
    """CrazyflieHwConfig seeded from config.py's DRONE_URI, if set."""
    return CrazyflieHwConfig(uri=DRONE_URI) if DRONE_URI else CrazyflieHwConfig()


def _mission_start_xyz(scenario: str) -> tuple[float, float, float]:
    """The point the drone hovers at during calibration, logs as its first
    waypoint, and returns to before landing.

    figure8's nominal start (0.50, -2.00, 0.25) is not the shared arena's
    START_XY/TAKEOFF_Z -- baseline/gate get the same value they use today.
    """
    if scenario == 'figure8':
        return nominal_figure8_waypoints(n_points=2)[0]
    return (*START_XY, TAKEOFF_Z)


@define(kw_only=True)
class CrazyflieConfig(BaseComponentConfig):
    """Trial settings for one flight, set from run.py's CLI args.

    `condition` selects which plan to fly ('pdstl' = the optimised waypoints for
    `fan_speed`, 'deterministic' = the nominal safe path). `fan_speed` also tags
    the logs. `hw_config` is the irobot-side radio/connection config (uri,
    timeout, ...); its default picks up components/config.py's DRONE_URI, so
    set that once instead of overriding hw_config per run.
    """

    z_hold: float = POST_GATE_Z  # shared with the gate scenario's post-gate descent altitude
    condition: str = 'pdstl'  # 'pdstl' or 'deterministic'
    fan_speed: int = 12  # 2, 6, 12, or 16
    scenario: str = 'baseline'  # 'baseline', 'gate', or 'figure8'
    hw_config: CrazyflieHwConfig = field(factory=_default_hw_config)


class CrazyfliePlanning(BaseComponent):
    def __init__(self, *, component_name, config, **kwargs):
        self.crazyflie = CrazyflieBase(config.hw_config)

        super().__init__(
            component_name=component_name,
            config=config,
            **kwargs,
        )
        # Fly at the planned speed so actual flight matches the belief model's
        # DT/U_MAX timing instead of the commander's arbitrary default.
        self.position_commander = PositionHlCommander(
            self.crazyflie.cf, default_velocity=FLIGHT_VELOCITY,
        )

    def _measured_xy(self) -> tuple[float, float]:
        return self.crazyflie.current_x, self.crazyflie.current_y

    def _calibrate_and_offset(
        self, waypoints: list[tuple[float, float, float]], start_xyz: tuple[float, float, float],
    ) -> list[tuple[float, float, float]]:
        """Hover at the assumed start, measure the real offset, and shift the plan to match.

        Real position rarely matches the offline plan's assumed start exactly
        (tracking drift, imprecise placement). Aborts (raises) if the offset is
        too large to trust rather than silently flying a bad plan.
        """
        start_xy = start_xyz[:2]
        self.position_commander.go_to(*start_xyz)
        measured = calibration.hover_and_measure(self._measured_xy, duration_s=CALIBRATION_HOVER_SECONDS)
        offset = calibration.compute_offset(measured, start_xy)
        print(f'[Calibration] measured={measured} assumed={start_xy} offset={offset}')
        calibration.check_offset_or_abort(offset, START_TOLERANCE)
        return calibration.shift_waypoints(waypoints, offset)

    def _fly_forward_mission(
        self, logger, waypoints: list[tuple[float, float, float]],
        start_xyz: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        waypoints = self._calibrate_and_offset(waypoints, start_xyz)
        logger.log_waypoint(*start_xyz)

        prev = start_xyz
        for x, y, z in waypoints:
            velocity = math.dist(prev, (x, y, z)) / DT
            assert velocity <= U_MAX + 1e-6, (
                f'Leg {prev}->{(x, y, z)} requires {velocity:.3f} m/s > U_MAX={U_MAX} m/s.'
            )
            self.position_commander.go_to(x, y, z, velocity=velocity)
            logger.log_waypoint(x, y, z)
            prev = (x, y, z)

        return waypoints[-1]

    def _return_to_start(
        self, last_xyz: tuple[float, float, float], start_xyz: tuple[float, float, float],
    ) -> None:
        """Fly back to the start and land — not part of the recorded trial.

        Run after the trial's data is already saved, so back-to-back runs don't
        need a manual reset and the return leg doesn't pollute the logged mission.
        Uses the commander's constant default_velocity (not per-leg timing) --
        this leg is untimed and not part of the recorded mission.
        """
        x, y, _z = last_xyz
        start_x, start_y, _start_z = start_xyz
        self.position_commander.go_to(x, y, RETURN_Z)
        time.sleep(1.0)
        self.position_commander.go_to(start_x, start_y, RETURN_Z)
        time.sleep(1.0)
        self.position_commander.go_to(start_x, start_y, LAND_Z)
        time.sleep(1.0)
        self.position_commander._hl_commander.stop()
        time.sleep(1.0)

    def _execute_once(self):
        condition = self.config.condition
        fan_speed = self.config.fan_speed
        scenario = self.config.scenario
        use_optimised = condition == 'pdstl'
        start_xyz = _mission_start_xyz(scenario)

        logger = FlightLogger(condition, fan_speed=fan_speed, scenario=scenario)
        logger.start()
        logger.start_actual_logging(
            lambda: (self.crazyflie.current_x, self.crazyflie.current_y, self.crazyflie.current_z)
        )

        if use_optimised:
            waypoints = load_pdstl_waypoints(fan_speed, scenario=scenario)
        elif scenario == 'gate':
            waypoints = nominal_gate_waypoints(n_points=SAFE_PATH_FLIGHT_POINTS)
        elif scenario == 'figure8':
            waypoints = nominal_figure8_waypoints(n_points=FIG8_FLIGHT_POINTS)
        else:
            waypoints = nominal_safe_waypoints(n_points=SAFE_PATH_FLIGHT_POINTS)
        validate_waypoints_in_bounds(waypoints)
        validate_waypoints_in_bounds([(*start_xyz[:2], RETURN_Z), (*start_xyz[:2], LAND_Z)])
        validate_waypoint_velocities(waypoints, start=start_xyz)

        try:
            last_xyz = self._fly_forward_mission(logger, waypoints, start_xyz)
        except Exception:
            logger.mark_crashed()
            logger.stop_actual_logging()
            logger.save()
            self.position_commander.land()
            raise

        # Forward mission (the recorded trial) is done -- stop logging and save
        # before the return-to-start leg, which is deliberately not recorded.
        logger.stop_actual_logging()
        logger.save()

        try:
            self._return_to_start(last_xyz, start_xyz)
        finally:
            self.position_commander.land()

    def _execution_step(self):
        pass
