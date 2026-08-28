"""Composed offline and streaming Always-and-Eventually mission."""

from dataclasses import dataclass

import torch

from models.streaming import mission_example
from pdstl import Always, Eventually, OfflineSource, OnlineSource, Predicate
from visualization.mission import plot_mission_example


@dataclass(frozen=True)
class MissionUpdate:
    """One arrival and the two temporal states used by the mission graph."""

    arrival: int
    always_state: torch.Tensor
    eventually_state: torch.Tensor
    always_output: torch.Tensor | None
    eventually_output: torch.Tensor | None
    mission_output: torch.Tensor | None


def _offline_mission(trace, interval):
    safe = Predicate("safe altitude")
    goal = Predicate("goal reached")
    source = OfflineSource({safe: trace.safe_bounds, goal: trace.goal_bounds})
    always = Always(safe, interval)
    eventually = Eventually(goal, interval)
    mission = always & eventually

    always_bounds = always(source)
    eventually_bounds = eventually(source)
    mission_bounds = mission(source)
    torch.testing.assert_close(
        mission_bounds,
        mission.step(always_bounds, eventually_bounds),
    )
    return safe, goal, always, eventually, mission, (
        always_bounds,
        eventually_bounds,
        mission_bounds,
    )


def _stream_mission(trace, safe, goal, always, eventually, mission):
    source = OnlineSource()
    always_state = eventually_state = None
    updates = []

    for arrival in range(trace.time.shape[0]):
        source.append(
            {
                safe: trace.safe_bounds[:, arrival, :],
                goal: trace.goal_bounds[:, arrival, :],
            }
        )
        always_output, always_state = always.step(
            source.bounds(safe, arrival), always_state
        )
        eventually_output, eventually_state = eventually.step(
            source.bounds(goal, arrival), eventually_state
        )
        mission_output = None
        if always_output is not None and eventually_output is not None:
            mission_output = mission.step(always_output, eventually_output)

        updates.append(
            MissionUpdate(
                arrival=arrival,
                always_state=always_state.clone(),
                eventually_state=eventually_state.clone(),
                always_output=(
                    None if always_output is None else always_output.clone()
                ),
                eventually_output=(
                    None if eventually_output is None else eventually_output.clone()
                ),
                mission_output=(
                    None if mission_output is None else mission_output.clone()
                ),
            )
        )

    complete = [update for update in updates if update.mission_output is not None]
    online_always = torch.stack([update.always_output for update in complete], dim=1)
    online_eventually = torch.stack(
        [update.eventually_output for update in complete], dim=1
    )
    online_mission = torch.stack([update.mission_output for update in complete], dim=1)
    return online_always, online_eventually, online_mission, updates


def _print_results(always_bounds, eventually_bounds, mission_bounds, interval):
    a, b = interval
    print(f"Mission = Always[{a},{b}](safe) AND Eventually[{a},{b}](goal)")
    for anchor in range(mission_bounds.shape[1]):
        always_pair = always_bounds[0, anchor].tolist()
        eventually_pair = eventually_bounds[0, anchor].tolist()
        mission_pair = mission_bounds[0, anchor].tolist()
        print(
            f"k={anchor}: Always=[{always_pair[0]:.2f}, {always_pair[1]:.2f}]  "
            f"Eventually=[{eventually_pair[0]:.2f}, {eventually_pair[1]:.2f}]  "
            f"Mission=[{mission_pair[0]:.2f}, {mission_pair[1]:.2f}]"
        )


def run_mission_example(interval=(0, 2), show=True):
    """Evaluate one composed mission offline and incrementally."""
    trace = mission_example()
    safe, goal, always, eventually, mission, offline = _offline_mission(
        trace, interval
    )
    always_bounds, eventually_bounds, mission_bounds = offline
    online_always, online_eventually, online_mission, updates = _stream_mission(
        trace, safe, goal, always, eventually, mission
    )

    torch.testing.assert_close(online_always, always_bounds)
    torch.testing.assert_close(online_eventually, eventually_bounds)
    torch.testing.assert_close(online_mission, mission_bounds)

    _print_results(always_bounds, eventually_bounds, mission_bounds, interval)
    print("Streaming branch and mission outputs match the offline graph.")
    figure = plot_mission_example(
        trace,
        always_bounds,
        eventually_bounds,
        mission_bounds,
        online_mission,
        interval,
        show=show,
    )
    return trace, offline, online_mission, updates, figure
