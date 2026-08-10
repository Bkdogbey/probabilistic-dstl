"""Offline tests for takeoff, return, and landing waypoint profiles."""

from __future__ import annotations

from experiments.crazyflie.components.utils import (
    LAND_Z,
    LANDING_SETTLE_SECONDS,
    LANDING_VELOCITY,
    START_TOLERANCE,
    TAKEOFF_VELOCITY,
    build_return_legs,
    needs_safe_transit,
)


def test_near_start_avoids_high_transit():
    start = (1.0, -2.0, 0.2)
    assert needs_safe_transit((1.0 + START_TOLERANCE / 2, -2.0, 0.0), start) is False
    assert needs_safe_transit((1.0 + START_TOLERANCE * 2, -2.0, 0.0), start) is True


def test_return_retraces_waypoints_at_planned_altitude():
    start = (1.0, -2.0, 0.2)
    waypoints = [start, (0.8, -1.8, 0.2), (0.6, -1.6, 0.2)]

    legs = build_return_legs(waypoints, start)
    commanded = [waypoint for waypoint, _velocity in legs]

    assert commanded == [waypoints[1], start]
    assert all(waypoint[2] == start[2] for waypoint in commanded)


def test_closed_path_does_not_fly_the_whole_mission_again():
    start = (0.5, -2.0, 0.25)
    waypoints = [start, (0.6, -1.5, 0.5), (0.5, -2.0, 0.25)]

    assert build_return_legs(waypoints, start) == []


def test_landing_profile_is_low_slow_and_settled():
    assert LAND_Z <= 0.1
    assert 0.0 < LANDING_VELOCITY < TAKEOFF_VELOCITY
    assert LANDING_SETTLE_SECONDS > 0.0
