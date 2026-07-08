from __future__ import annotations

import time

import numpy as np
from cflib.positioning.position_hl_commander import PositionHlCommander
from ros_sugar.core import BaseComponent

from components.flight_logger import FlightLogger
from components.opt_waypoints import WAYPOINTS
from irobot.src.robots.crazyflie.core.base import CrazyflieBase

# Trial configuration: edit these values before each run.
# Path selector: True for the pDSTL-optimised path, False for the original sine path.
USE_OPTIMISED = False
# Condition label: 'deterministic' or 'pdstl'.
CONDITION = 'pdstl'
# Fan speed integer: 2, 6, 12, or 16.
FAN_SPEED = 12

FLIGHT_X_BOUNDS = (-0.5, 1.0)
FLIGHT_Y_BOUNDS = (-2.0, 0.5)
START_Y = -1.5
END_Y = FLIGHT_Y_BOUNDS[1]
RETURN_Z = 0.65
LAND_Z = 0.1
WAYPOINT_DELAY_SECONDS = 0.1


def _sine_waypoints() -> list[tuple[float, float, float]]:
    start_0 = abs(START_Y)
    y_pos = np.linspace(START_Y, END_Y, 10)
    x_pos = 0.5 * np.sin(np.pi * y_pos / start_0)
    return [(float(x), float(y), 0.2) for x, y in zip(x_pos, y_pos)]


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
        self.crazyflie = CrazyflieBase(config)

        super().__init__(
            component_name=component_name,
            config=config,
            **kwargs,
        )
        self.position_commander = PositionHlCommander(self.crazyflie.cf)

    def _execute_once(self):
        logger = FlightLogger(CONDITION, fan_speed=FAN_SPEED)

        logger.start()
        logger.start_actual_logging(
            lambda: (self.crazyflie.current_x, self.crazyflie.current_y, self.crazyflie.current_z)
        )
        waypoints = WAYPOINTS if USE_OPTIMISED else _sine_waypoints()
        _validate_waypoints_inside_flight_area(waypoints)
        _validate_waypoints_inside_flight_area(
            [
                (0.0, START_Y, RETURN_Z),
                (0.0, START_Y, LAND_Z),
            ]
        )
        try:
            for x, y, z in waypoints:
                self.position_commander.go_to(x, y, z)
                logger.log_waypoint(x, y, z)
                time.sleep(WAYPOINT_DELAY_SECONDS)

            self.position_commander.go_to(x, y, RETURN_Z)
            time.sleep(1.0)
            self.position_commander.go_to(0, START_Y, RETURN_Z)
            time.sleep(1.0)
            self.position_commander.go_to(0, START_Y, LAND_Z)
            time.sleep(1.0)
            self.position_commander._hl_commander.stop()
            time.sleep(1.0)

        except Exception:
            logger.mark_crashed()
            raise
        finally:
            logger.stop_actual_logging()
            logger.save()
            self.position_commander.land()

    def _execution_step(self):
        pass
