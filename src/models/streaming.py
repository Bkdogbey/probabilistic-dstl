"""Small probability-bound traces for sliding and streaming examples."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProbabilityTrace:
    """A named single-batch trace of atomic probability bounds."""

    time: torch.Tensor
    bounds: torch.Tensor
    predicate_name: str


@dataclass(frozen=True)
class MissionTrace:
    """Two aligned atomic traces for the composed mission example."""

    time: torch.Tensor
    safe_bounds: torch.Tensor
    goal_bounds: torch.Tensor


def _trace(rows, predicate_name, dtype):
    bounds = torch.tensor(rows, dtype=dtype).unsqueeze(0)
    return ProbabilityTrace(
        time=torch.arange(bounds.shape[1]),
        bounds=bounds,
        predicate_name=predicate_name,
    )


def sliding_always_example(dtype=torch.float64):
    """Return an 11-step trace with one safety-probability drop at t=4."""
    return _trace(
        [
            [0.96, 0.99],
            [0.95, 0.98],
            [0.94, 0.98],
            [0.93, 0.97],
            [0.72, 0.85],
            [0.92, 0.97],
            [0.95, 0.99],
            [0.96, 0.99],
            [0.94, 0.98],
            [0.95, 0.99],
            [0.96, 0.99],
        ],
        "safe altitude",
        dtype,
    )


def sliding_eventually_example(dtype=torch.float64):
    """Return an 11-step trace with one strong goal opportunity at t=4."""
    return _trace(
        [
            [0.03, 0.07],
            [0.04, 0.08],
            [0.05, 0.09],
            [0.06, 0.10],
            [0.75, 0.88],
            [0.05, 0.10],
            [0.06, 0.11],
            [0.07, 0.12],
            [0.08, 0.13],
            [0.09, 0.14],
            [0.10, 0.15],
        ],
        "goal reached",
        dtype,
    )


def mission_example(dtype=torch.float64):
    """Return aligned safety and goal traces for mission composition."""
    safe = sliding_always_example(dtype)
    goal = sliding_eventually_example(dtype)
    return MissionTrace(
        time=safe.time,
        safe_bounds=safe.bounds,
        goal_bounds=goal.bounds,
    )
