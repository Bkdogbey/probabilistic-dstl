from __future__ import annotations

import time

from cflib.positioning.position_hl_commander import PositionHlCommander
from ros_sugar.core import BaseComponent

from components import calibration
from components.environment_config import (
    FLIGHT_X_BOUNDS,
    FLIGHT_Y_BOUNDS,
    REPLAN_CHECKPOINTS,
    SAFE_PATH_FLIGHT_POINTS,
    START_TOLERANCE,
    START_XY,
    Z_HEIGHT,
    build_planner,
    measured_belief,
    nominal_safe_waypoints,
)
from components.flight_logger import FlightLogger
from components.opt_waypoints import WAYPOINTS
from irobot.src.robots.crazyflie.core.base import CrazyflieBase

# Trial settings (condition, fan speed) live in CrazyflieConfig, set from
# main.py's CLI args -- no need to edit this file before a run.

TAKEOFF_Z = Z_HEIGHT
RETURN_Z = 0.65
LAND_Z = 0.1
WAYPOINT_DELAY_SECONDS = 0.1
CALIBRATION_HOVER_SECONDS = 2.0

_nominal_waypoints = nominal_safe_waypoints


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
        planner, _dynamics, _env = build_planner(self.config.fan_speed, T_horizon=horizon)
        x0_mean, x0_cov = measured_belief(measured)

        best_mean, _best_cov, _best_u, best_p, _history = planner._optimize_window(
            x0_mean, x0_cov, verbose=False,
        )
        positions_xy = best_mean.squeeze(0).cpu().numpy()[1:]  # drop t=0 (== measured)
        new_tail = [(float(x), float(y), Z_HEIGHT) for x, y in positions_xy]
        print(f'[Replan] from measured={measured} -> P(sat)={best_p:.3f}, {len(new_tail)} new waypoints')
        return new_tail

    def _fly_forward_mission(
        self, logger, waypoints: list[tuple[float, float, float]], *, replan_enabled: bool,
    ) -> tuple[float, float, float]:
        """Calibrate, fly start->goal (optionally with mid-flight replanning), return the last waypoint flown.

        replan_enabled gates REPLAN_CHECKPOINTS to the pdstl condition only:
        replanning re-optimises via the same pdSTL planner the pdstl condition
        uses, which doesn't belong in the deterministic condition -- its whole
        point is to be an unoptimised baseline flown as-is under real fan
        noise, not one that secretly self-corrects mid-flight.
        """
        waypoints = self._calibrate_and_offset(waypoints)
        logger.log_waypoint(*START_XY, TAKEOFF_Z)

        i = 0
        while i < len(waypoints):
            x, y, z = waypoints[i]
            self.position_commander.go_to(x, y, z)
            logger.log_waypoint(x, y, z)
            time.sleep(WAYPOINT_DELAY_SECONDS)

            if replan_enabled and i in REPLAN_CHECKPOINTS and i < len(waypoints) - 1:
                new_tail = self._replan_from_here(waypoints[i + 1:])
                _validate_waypoints_inside_flight_area(new_tail)
                waypoints = waypoints[: i + 1] + new_tail
            i += 1

        return waypoints[-1]

    def _return_to_start(self, last_xyz: tuple[float, float, float]) -> None:
        """Fly back to the start and land — not part of the recorded trial.

        Run after the trial's data is already saved, so it makes back-to-back
        runs easier without polluting the logged mission with the return leg.
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
        nominal_waypoints = WAYPOINTS if use_optimised else _nominal_waypoints(n_points=SAFE_PATH_FLIGHT_POINTS)
        _validate_waypoints_inside_flight_area(nominal_waypoints)
        _validate_waypoints_inside_flight_area(
            [
                (*START_XY, RETURN_Z),
                (*START_XY, LAND_Z),
            ]
        )

        try:
            last_xyz = self._fly_forward_mission(logger, nominal_waypoints, replan_enabled=use_optimised)
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
