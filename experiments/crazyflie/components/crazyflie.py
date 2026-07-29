from __future__ import annotations

import math
import time
from dataclasses import dataclass

from attr import define, field
from cflib.positioning.position_hl_commander import PositionHlCommander
from ros_sugar.config import BaseComponentConfig
from ros_sugar.core import BaseComponent

from .utils import (
    ARMING_WAIT_SECONDS,
    DRONE_URI,
    DT,
    ESTIMATOR_MIN_BASE_STATIONS,
    ESTIMATOR_RESET_WAIT_SECONDS,
    ESTIMATOR_SAMPLE_COUNT,
    ESTIMATOR_SAMPLE_PERIOD_MS,
    ESTIMATOR_SETTLE_TIMEOUT,
    ESTIMATOR_SPREAD_LIMIT,
    LAND_Z,
    RETURN_Z,
    START_SETTLE_SECONDS,
    TAKEOFF_VELOCITY,
    U_MAX,
    PlanRepository,
    Waypoint,
    shutdown_hardware,
    validate_waypoints,
    validate_waypoint_velocities,
)
from .planner import Scenario, get_scenario
from .flight_logger import FlightLogger
from irobot import (
    CrazyflieBase,
    CrazyflieConfig as CrazyflieHwConfig,
    wait_for_stable_estimate,
)


def _default_hw_config() -> CrazyflieHwConfig:
    """Build the hardware configuration using the configured radio URI."""
    return CrazyflieHwConfig(uri=DRONE_URI) if DRONE_URI else CrazyflieHwConfig()


@dataclass(frozen=True)
class PreparedMission:
    scenario: Scenario
    start: Waypoint
    waypoints: tuple[Waypoint, ...]


class MissionPlanBuilder:
    def build(self, *, condition: str, fan: int, scenario_name: str) -> PreparedMission:
        scenario = get_scenario(scenario_name)
        start = scenario.start()
        if condition == 'pdstl':
            waypoints = PlanRepository().require_flyable(fan, scenario_name).waypoints
        elif condition == 'deterministic':
            waypoints = tuple(scenario.nominal_waypoints(scenario.flight_points))
        else:
            raise ValueError(f'Unknown flight condition {condition!r}.')
        scenario.validate(list(waypoints))
        validate_waypoints([(*start[:2], RETURN_Z), (*start[:2], LAND_Z)])
        validate_waypoint_velocities(list(waypoints), start=start)
        return PreparedMission(scenario, start, waypoints)


class MissionExecutor:
    def __init__(self, commander) -> None:
        self.commander = commander

    def fly_to_start(self, start: Waypoint) -> None:
        """Move from wherever takeoff left the drone to the mission start.

        Mirrors return_and_land's climb-move-descend shape: move at RETURN_Z
        (above every configured obstacle) before descending at the
        destination, so this works regardless of where within the workspace
        the drone took off.
        """
        start_x, start_y, start_z = start
        self.commander.go_to(start_x, start_y, RETURN_Z)
        self.commander.go_to(start_x, start_y, start_z)

    def fly(self, logger, waypoints: list[Waypoint], start: Waypoint) -> Waypoint:
        logger.log_waypoint(*start)
        previous = start
        for waypoint in waypoints:
            velocity = math.dist(previous, waypoint) / DT
            if velocity > U_MAX + 1e-6:
                raise ValueError(f'Leg requires {velocity:.3f} m/s; limit is {U_MAX} m/s.')
            self.commander.go_to(*waypoint, velocity=velocity)
            logger.log_waypoint(*waypoint)
            previous = waypoint
        return waypoints[-1]

    def return_and_land(self, last: Waypoint, start: Waypoint) -> None:
        x, y, _ = last
        start_x, start_y, _ = start
        self.commander.go_to(x, y, RETURN_Z)
        self.commander.go_to(start_x, start_y, RETURN_Z)
        self.commander.go_to(start_x, start_y, LAND_Z)


@define(kw_only=True)
class CrazyflieConfig(BaseComponentConfig):
    """One configured hardware trial."""

    condition: str = 'pdstl'
    fan_speed: int = 12
    scenario: str = 'baseline'
    hw_config: CrazyflieHwConfig = field(factory=_default_hw_config)


class CrazyflieFlightComponent(BaseComponent):
    def __init__(self, *, component_name, config, **kwargs):
        super().__init__(
            component_name=component_name,
            config=config,
            **kwargs,
        )
        # Open hardware only after framework initialization. If BaseComponent
        # construction fails, no radio resource has to be cleaned up.
        self.crazyflie = CrazyflieBase(config.hw_config)
        # Constructed only after the estimator passes preflight so its internal
        # position starts at the real settled pose, not PositionHlCommander's
        # default (0, 0, 0).
        self.position_commander: PositionHlCommander | None = None

    @staticmethod
    def _report_preflight_sample(mean, spread, mask: int) -> None:
        print(
            '[Preflight] '
            f'position=({mean[0]:+.3f}, {mean[1]:+.3f}, {mean[2]:+.3f})  '
            f'spread=({spread[0]:.3f}, {spread[1]:.3f}, {spread[2]:.3f})  '
            f'base_stations={int(mask).bit_count()} (mask {int(mask):#06b})'
        )

    def _wait_for_flight_ready_estimate(self) -> tuple[float, float, float]:
        """Reset the estimator and wait until it settles; return the settled pose.

        The drone can be sitting anywhere -- only a stable lock is required,
        not a specific start coordinate. It flies to the mission start after
        takeoff (see MissionExecutor.fly_to_start).
        """
        return wait_for_stable_estimate(
            self.crazyflie.cf,
            spread_limit=ESTIMATOR_SPREAD_LIMIT,
            min_base_stations=ESTIMATOR_MIN_BASE_STATIONS,
            settle_timeout=ESTIMATOR_SETTLE_TIMEOUT,
            sample_count=ESTIMATOR_SAMPLE_COUNT,
            sample_period_ms=ESTIMATOR_SAMPLE_PERIOD_MS,
            reset_wait_s=ESTIMATOR_RESET_WAIT_SECONDS,
            on_sample=self._report_preflight_sample,
        )

    def _preflight_and_arm(self) -> None:
        """Disarm, wait for a flight-ready estimate, build the commander at
        the settled pose, then arm.
        """
        # Force a safe disarmed preflight and arm only after the estimate
        # has passed every check. (irobot's CrazyflieConfig.auto_arm is
        # False by default, so nothing has armed the drone before now.)
        self.crazyflie.disarm()
        settled = self._wait_for_flight_ready_estimate()
        self.position_commander = PositionHlCommander(
            self.crazyflie.cf,
            x=settled[0], y=settled[1], z=settled[2],
            default_velocity=U_MAX,
            default_height=RETURN_Z,
            controller=PositionHlCommander.CONTROLLER_PID,
        )
        self.crazyflie.arm()
        time.sleep(ARMING_WAIT_SECONDS)

    def _execute_once(self):
        condition = self.config.condition
        fan_speed = self.config.fan_speed
        scenario = self.config.scenario
        mission = MissionPlanBuilder().build(
            condition=condition, fan=fan_speed, scenario_name=scenario,
        )
        start_xyz = mission.start
        waypoints = list(mission.waypoints)

        logger: FlightLogger | None = None
        actual_logging = False
        return_logging = False
        airborne = False

        try:
            self._preflight_and_arm()
            self.position_commander.take_off(RETURN_Z, TAKEOFF_VELOCITY)
            airborne = True

            executor = MissionExecutor(self.position_commander)
            executor.fly_to_start(start_xyz)
            # go_to() returns once its estimated move duration elapses, not once
            # position error has actually converged -- give the final descent
            # leg time to settle so logged data starts from rest, not mid-move.
            time.sleep(START_SETTLE_SECONDS)

            # Takeoff and transit-to-start are preflight, not experiment data.
            logger = FlightLogger(condition, fan_speed=fan_speed, scenario=scenario)
            logger.start()
            logger.start_actual_logging(self.crazyflie.state_snapshot)
            actual_logging = True
            last_xyz = executor.fly(logger, waypoints, start_xyz)

            logger.stop_actual_logging()
            actual_logging = False
            logger.save()

            # Debug capture of return-and-land, saved separately from the
            # mission log (see FlightLogger.start_return_logging).
            logger.start_return_logging(self.crazyflie.state_snapshot)
            return_logging = True
            executor.return_and_land(last_xyz, start_xyz)
        except BaseException:
            if logger is not None and actual_logging:
                logger.mark_crashed()
                logger.stop_actual_logging()
                actual_logging = False
                logger.save()
            raise
        finally:
            try:
                if actual_logging and logger is not None:
                    logger.stop_actual_logging()
            finally:
                try:
                    shutdown_hardware(
                        self.position_commander,
                        self.crazyflie,
                        airborne=airborne,
                        landing_velocity=TAKEOFF_VELOCITY,
                    )
                finally:
                    if return_logging and logger is not None:
                        logger.stop_actual_logging()
                        logger.save_return_phase()

    def _execution_step(self):
        pass
