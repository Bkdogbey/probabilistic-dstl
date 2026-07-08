from __future__ import annotations

import time

import numpy as np
from cflib.positioning.position_hl_commander import PositionHlCommander
from ros_sugar.core import BaseComponent

from components import calibration
from components.environment_config import (
    FLIGHT_X_BOUNDS,
    FLIGHT_Y_BOUNDS,
    REPLAN_CHECKPOINTS,
    START_TOLERANCE,
    START_XY,
    Z_HEIGHT,
    build_planner,
    measured_belief,
)
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

START_Y = START_XY[1]
END_Y = FLIGHT_Y_BOUNDS[1]
TAKEOFF_Z = Z_HEIGHT
RETURN_Z = 0.65
LAND_Z = 0.1
WAYPOINT_DELAY_SECONDS = 0.1
CALIBRATION_HOVER_SECONDS = 2.0


def _sine_waypoints() -> list[tuple[float, float, float]]:
    start_0 = abs(START_Y)
    y_pos = np.linspace(START_Y, END_Y, 10)
    x_pos = 0.5 * np.sin(np.pi * y_pos / start_0)
    return [(float(x), float(y), Z_HEIGHT) for x, y in zip(x_pos, y_pos)]


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

    def _measured_xy(self) -> tuple[float, float]:
        return self.crazyflie.current_x, self.crazyflie.current_y

    def _calibrate_and_offset(self, waypoints: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        """Hover at the assumed start, measure the real offset, and shift the plan to match.

        Real position rarely matches the offline plan's assumed start exactly
        (tracking drift, imprecise placement). Aborts (raises) if the offset
        is too large to trust rather than silently flying a bad plan.
        """
        self.position_commander.go_to(*START_XY, TAKEOFF_Z)
        measured = calibration.hover_and_measure(self._measured_xy, duration_s=CALIBRATION_HOVER_SECONDS)
        offset = calibration.compute_offset(measured, START_XY)
        print(f'[Calibration] measured={measured} assumed={START_XY} offset={offset}')
        calibration.check_offset_or_abort(offset, START_TOLERANCE)
        return calibration.shift_waypoints(waypoints, offset)

    def _replan_from_here(self, remaining_waypoints: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        """Re-optimise the remaining trajectory from the actual measured position.

        Called at each REPLAN_CHECKPOINTS index instead of blindly continuing
        the offline plan — corrects for drift/disturbance accumulated in flight.
        """
        measured = self._measured_xy()
        horizon = max(1, len(remaining_waypoints))
        planner, _dynamics, _env = build_planner(FAN_SPEED, T_horizon=horizon)
        x0_mean, x0_cov = measured_belief(measured)

        best_mean, _best_cov, _best_u, best_p, _history = planner._optimize_window(
            x0_mean, x0_cov, verbose=False,
        )
        positions_xy = best_mean.squeeze(0).cpu().numpy()[1:]  # drop t=0 (== measured)
        new_tail = [(float(x), float(y), Z_HEIGHT) for x, y in positions_xy]
        print(f'[Replan] from measured={measured} -> P(sat)={best_p:.3f}, {len(new_tail)} new waypoints')
        return new_tail

    def _execute_once(self):
        logger = FlightLogger(CONDITION, fan_speed=FAN_SPEED)

        logger.start()
        logger.start_actual_logging(
            lambda: (self.crazyflie.current_x, self.crazyflie.current_y, self.crazyflie.current_z)
        )
        nominal_waypoints = WAYPOINTS if USE_OPTIMISED else _sine_waypoints()
        _validate_waypoints_inside_flight_area(nominal_waypoints)
        _validate_waypoints_inside_flight_area(
            [
                (0.0, START_Y, RETURN_Z),
                (0.0, START_Y, LAND_Z),
            ]
        )
        try:
            waypoints = self._calibrate_and_offset(nominal_waypoints)
            logger.log_waypoint(*START_XY, TAKEOFF_Z)

            i = 0
            while i < len(waypoints):
                x, y, z = waypoints[i]
                self.position_commander.go_to(x, y, z)
                logger.log_waypoint(x, y, z)
                time.sleep(WAYPOINT_DELAY_SECONDS)

                if i in REPLAN_CHECKPOINTS and i < len(waypoints) - 1:
                    new_tail = self._replan_from_here(waypoints[i + 1:])
                    _validate_waypoints_inside_flight_area(new_tail)
                    waypoints = waypoints[: i + 1] + new_tail
                i += 1

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
