from __future__ import annotations

import time

from attrs import define, field
from cflib.positioning.position_hl_commander import PositionHlCommander
from ros_sugar.config import BaseComponentConfig
from ros_sugar.core import BaseComponent

from components import calibration
from components.config import (
    CALIBRATION_HOVER_SECONDS,
    FLIGHT_VELOCITY,
    FLIGHT_X_BOUNDS,
    FLIGHT_Y_BOUNDS,
    LAND_Z,
    RETURN_Z,
    SAFE_PATH_FLIGHT_POINTS,
    START_TOLERANCE,
    START_XY,
    TAKEOFF_Z,
    load_pdstl_waypoints,
    nominal_safe_waypoints,
)
from components.flight_logger import FlightLogger
from irobot.src.robots.crazyflie.config import CrazyflieConfig as CrazyflieHwConfig
from irobot.src.robots.crazyflie.core.base import CrazyflieBase


@define(kw_only=True)
class CrazyflieConfig(BaseComponentConfig):
    """Trial settings for one flight, set from run.py's CLI args.

    `condition` selects which plan to fly ('pdstl' = the optimised waypoints for
    `fan_speed`, 'deterministic' = the nominal safe path). `fan_speed` also tags
    the logs. `hw_config` is the irobot-side radio/connection config (uri,
    timeout, ...); override it to fly a Crazyflie other than the default URI.
    No file editing is needed before a run.
    """

    z_hold: float = 0.3
    condition: str = 'pdstl'  # 'pdstl' or 'deterministic'
    fan_speed: int = 12  # 2, 6, 12, or 16
    hw_config: CrazyflieHwConfig = field(factory=CrazyflieHwConfig)


def _validate_waypoints_inside_flight_area(waypoints: list[tuple[float, float, float]]) -> None:
    x_min, x_max = FLIGHT_X_BOUNDS
    y_min, y_max = FLIGHT_Y_BOUNDS
    outside = [
        (idx, x, y)
        for idx, (x, y, _z) in enumerate(waypoints)
        if not (x_min <= x <= x_max and y_min <= y <= y_max)
    ]
    if outside:
        details = ', '.join(f'#{idx}=({x:.3f}, {y:.3f})' for idx, x, y in outside)
        raise ValueError(
            'Waypoint(s) outside flight area '
            f'x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]: {details}'
        )


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
        self, waypoints: list[tuple[float, float, float]]
    ) -> list[tuple[float, float, float]]:
        """Hover at the assumed start, measure the real offset, and shift the plan to match.

        Real position rarely matches the offline plan's assumed start exactly
        (tracking drift, imprecise placement). Aborts (raises) if the offset is
        too large to trust rather than silently flying a bad plan. This is a
        one-time pre-flight step — not a mid-flight pause.
        """
        self.position_commander.go_to(*START_XY, TAKEOFF_Z)
        measured = calibration.hover_and_measure(self._measured_xy, duration_s=CALIBRATION_HOVER_SECONDS)
        offset = calibration.compute_offset(measured, START_XY)
        print(f'[Calibration] measured={measured} assumed={START_XY} offset={offset}')
        calibration.check_offset_or_abort(offset, START_TOLERANCE)
        return calibration.shift_waypoints(waypoints, offset)

    def _fly_forward_mission(
        self, logger, waypoints: list[tuple[float, float, float]],
    ) -> tuple[float, float, float]:
        """Calibrate once, then fly the given waypoints start->finish; return the last one.

        No mid-flight replanning: the plan is flown exactly as given (offset-
        corrected at the start), so the drone moves continuously from start to
        goal without pausing to re-optimise.
        """
        waypoints = self._calibrate_and_offset(waypoints)
        logger.log_waypoint(*START_XY, TAKEOFF_Z)

        for x, y, z in waypoints:
            self.position_commander.go_to(x, y, z)
            logger.log_waypoint(x, y, z)

        return waypoints[-1]

    def _return_to_start(self, last_xyz: tuple[float, float, float]) -> None:
        """Fly back to the start and land — not part of the recorded trial.

        Run after the trial's data is already saved, so back-to-back runs don't
        need a manual reset and the return leg doesn't pollute the logged mission.
        """
        x, y, _z = last_xyz
        self.position_commander.go_to(x, y, RETURN_Z)
        time.sleep(1.0)
        self.position_commander.go_to(*START_XY, RETURN_Z)
        time.sleep(1.0)
        self.position_commander.go_to(*START_XY, LAND_Z)
        time.sleep(1.0)
        self.position_commander._hl_commander.stop()
        time.sleep(1.0)

    def _execute_once(self):
        condition = self.config.condition
        fan_speed = self.config.fan_speed
        use_optimised = condition == 'pdstl'

        logger = FlightLogger(condition, fan_speed=fan_speed)
        logger.start()
        logger.start_actual_logging(
            lambda: (self.crazyflie.current_x, self.crazyflie.current_y, self.crazyflie.current_z)
        )

        if use_optimised:
            waypoints = load_pdstl_waypoints(fan_speed)
        else:
            waypoints = nominal_safe_waypoints(n_points=SAFE_PATH_FLIGHT_POINTS)
        _validate_waypoints_inside_flight_area(waypoints)
        _validate_waypoints_inside_flight_area([(*START_XY, RETURN_Z), (*START_XY, LAND_Z)])

        try:
            last_xyz = self._fly_forward_mission(logger, waypoints)
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
            self._return_to_start(last_xyz)
        finally:
            self.position_commander.land()

    def _execution_step(self):
        pass
