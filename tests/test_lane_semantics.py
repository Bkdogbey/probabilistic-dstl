"""Shared physical and temporal semantics for lane execution and pdSTL."""

import torch

from models.rollouts import LaneBeliefTrajectory
from planning.environment import (
    lane_contains_footprint,
    lane_goal_reached,
    lane_has_collision,
    lane_subformulas,
)
from planning.runners import build_environment
from utils import load_config


def test_collision_uses_footprints_and_margins():
    env = build_environment(load_config("configs/scenarios/lane_change.yaml"))
    ego = torch.tensor([0.0, 0.0])
    traffic = torch.tensor([[5.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    assert lane_has_collision(env, ego, traffic)
    traffic[0, 0] += 0.001
    assert not lane_has_collision(env, ego, traffic)


def test_road_contains_the_whole_vehicle_and_taper_front_corner():
    straight = build_environment(
        load_config("configs/scenarios/lane_change.yaml")
    )
    assert lane_contains_footprint(straight, 0.0, -0.9)
    assert not lane_contains_footprint(straight, 0.0, -0.901)

    merge = build_environment(load_config("configs/scenarios/lane_merge.yaml"))
    road, ramp = merge.metadata["road"], merge.metadata["ramp"]
    x = 40.0
    front = x + merge.metadata["ego_vehicle"]["width"] / 2
    slope = (road["lane_divider"] - road["y_min"]) / (
        ramp["end_x"] - ramp["start_x"]
    )
    center_y = road["y_min"] + slope * (front - ramp["start_x"]) + 0.9
    assert lane_contains_footprint(merge, x, center_y)
    assert not lane_contains_footprint(merge, x, center_y - 0.001)


def test_dwell_finishes_inside_window_and_merge_front_is_bounded():
    change = build_environment(
        load_config("configs/scenarios/lane_change.yaml")
    )
    task = change.metadata["task"]
    completion = task["start_end_steps"][0] + task["dwell_steps"]
    mean = torch.tensor([50.0, task["target_center"], 0.0, 0.0])
    assert not lane_goal_reached(
        change, mean, completion - 1, task["dwell_steps"]
    )
    assert lane_goal_reached(change, mean, completion, task["dwell_steps"] + 1)

    merge = build_environment(load_config("configs/scenarios/lane_merge.yaml"))
    limit = (
        merge.metadata["ramp"]["end_x"]
        - merge.metadata["ego_vehicle"]["width"] / 2
    )
    mean[0] = limit
    assert lane_goal_reached(merge, mean, completion, task["dwell_steps"] + 1)
    mean[0] += 0.001
    assert not lane_goal_reached(
        merge, mean, completion, task["dwell_steps"] + 1
    )


def test_completion_formula_matches_sampled_dwell():
    env = build_environment(load_config("configs/scenarios/lane_change.yaml"))
    task = env.metadata["task"]
    steps = env.metadata["task"]["start_end_steps"][1] + 1
    mean = torch.zeros(1, steps, 4)
    start = task["start_end_steps"][0]
    finish = start + task["dwell_steps"]
    mean[0, start : finish + 1, 1] = task["target_center"]
    covariance = torch.zeros(1, steps, 4, 4)
    traffic = torch.zeros(1, steps, 3, 4)
    traffic_covariance = torch.zeros(1, steps, 3, 4, 4)
    beliefs = LaneBeliefTrajectory(
        mean,
        covariance,
        traffic,
        traffic_covariance,
        [vehicle["name"] for vehicle in env.metadata["traffic"]],
    )
    interval = lane_subformulas(env, steps - 1)[
        "complete"
    ].probability_interval(beliefs)
    torch.testing.assert_close(interval, torch.ones(2))
    assert lane_goal_reached(
        env,
        mean[0, finish],
        finish,
        task["dwell_steps"] + 1,
    )
