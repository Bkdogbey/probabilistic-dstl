"""Deterministic mean-trajectory STL objective for the optional lane study."""

import torch

from pdstl.operators import Maxish, Minish
from planning.environment import lane_collision_extents, lane_dwell_credit


_MIN, _MAX = Minish(), Maxish()


def _reduce(values, operation, beta):
    values = torch.stack(tuple(values))
    return operation(values, beta=beta, dim=0, keepdim=False)


class DeterministicLaneSpecification:
    """Optimize signed robustness while delegating certified reporting to pdSTL."""

    def __init__(self, environment, horizon):
        self.environment = environment
        self.horizon = horizon
        self.reporting = environment.get_specification(horizon)

    def probability_interval(self, beliefs, origin=0, **kwargs):
        return self.reporting.probability_interval(beliefs, origin, **kwargs)

    def smooth_lower(self, beliefs, beta, origin=0, **kwargs):
        del origin, kwargs
        metadata = self.environment.metadata
        ego = beliefs.trace.mean[0]
        traffic = beliefs.traffic_mean[0]
        road = metadata["road"]
        vehicle = metadata["ego_vehicle"]
        half_width = vehicle["width"] / 2
        half_height = vehicle["height"] / 2

        lower = ego[:, 1] - (road["y_min"] + half_height)
        upper = road["y_max"] - half_height - ego[:, 1]
        road_score = _reduce((lower, upper), _MIN, beta)
        ramp = metadata.get("ramp")
        if ramp is not None:
            slope = (road["lane_divider"] - road["y_min"]) / (
                ramp["end_x"] - ramp["start_x"]
            )
            main = ego[:, 1] - (road["lane_divider"] + half_height)
            taper = (
                ego[:, 1]
                - slope * ego[:, 0]
                + slope * ramp["start_x"]
                - road["y_min"]
                - half_height
                - slope * half_width
            )
            road_score = _reduce(
                (road_score, _reduce((main, taper), _MAX, beta)),
                _MIN,
                beta,
            )

        safety = [road_score]
        for index, other in enumerate(metadata["traffic"]):
            longitudinal, lateral = lane_collision_extents(metadata, other)
            delta = traffic[:, index, :2] - ego[:, :2]
            safety.append(
                _reduce(
                    (
                        delta[:, 0].abs() - longitudinal,
                        delta[:, 1].abs() - lateral,
                    ),
                    _MAX,
                    beta,
                )
            )
        safety_trace = _MIN(
            torch.stack(safety), beta=beta, dim=0, keepdim=False
        )

        task = metadata["task"]
        target = (
            task["target_tolerance"]
            - (ego[:, 1] - task["target_center"]).abs()
        )
        step = metadata.get("step", 0)
        streak = metadata.get("streak", 0)
        dwell = task["dwell_steps"]
        credit = lane_dwell_credit(metadata, step, streak)

        def candidate(start, duration):
            finish = start + duration
            values = [
                _MIN(
                    target[start : start + duration + 1],
                    beta=beta,
                    dim=0,
                    keepdim=False,
                )
            ]
            if ramp is not None:
                values.append(ramp["end_x"] - half_width - ego[finish, 0])
            completion = _reduce(values, _MIN, beta)
            safe = _MIN(
                safety_trace[: finish + 1],
                beta=beta,
                dim=0,
                keepdim=False,
            )
            return _reduce((safe, completion), _MIN, beta)

        if credit:
            overall = candidate(0, dwell - (credit - 1))
        else:
            first = max(0, task["start_end_steps"][0] - step)
            last = task["start_end_steps"][1] - dwell - step
            overall = _reduce(
                (candidate(start, dwell) for start in range(first, last + 1)),
                _MAX,
                beta,
            )
        return overall
