"""Example probability-bound inputs for the pdSTL demonstration in src/main.py.

Twelve discrete time steps for two predicates, "safe" and "goal", each as a
[1, 12, 2] tensor of [lower, upper] probability bounds. These are
illustrative numbers, not the output of any state estimator or dynamics
model.
"""

import torch


def temporal_probability_traces(dtype=torch.float64):
    """Return (time, safe_bounds, goal_bounds) for a 12-step example.

    time is a length-12 tensor of step indices; safe_bounds and goal_bounds
    are each Tensor[1, 12, 2] of [lower, upper] probability bounds.
    """
    time = torch.arange(12)

    safe_bounds = torch.tensor(
        [
            [0.92, 0.98],
            [0.90, 0.96],
            [0.88, 0.95],
            [0.86, 0.94],
            [0.55, 0.72],
            [0.82, 0.90],
            [0.90, 0.97],
            [0.91, 0.98],
            [0.89, 0.96],
            [0.84, 0.92],
            [0.87, 0.94],
            [0.90, 0.97],
        ],
        dtype=dtype,
    ).unsqueeze(0)

    goal_bounds = torch.tensor(
        [
            [0.05, 0.10],
            [0.08, 0.15],
            [0.12, 0.20],
            [0.25, 0.35],
            [0.45, 0.55],
            [0.70, 0.82],
            [0.35, 0.50],
            [0.15, 0.30],
            [0.10, 0.25],
            [0.55, 0.70],
            [0.75, 0.88],
            [0.85, 0.95],
        ],
        dtype=dtype,
    ).unsqueeze(0)

    return time, safe_bounds, goal_bounds
