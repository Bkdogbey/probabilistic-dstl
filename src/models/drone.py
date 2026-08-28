"""A single illustrative drone-altitude scenario.

9 discrete time steps. The altitude mean dips near/below 50m around t=2..4
then climbs past 55m by t=6..8, so both Always(altitude >= 50m) and
Eventually(altitude >= 55m) react visibly over the same trace.

bounds_above_50 / bounds_above_55 are hand-authored [1, T, 2] probability
bounds, not derived from altitude_mean/altitude_std by any formula -- pdSTL
does not currently compute predicate probabilities from a Gaussian belief,
so these stand in for whatever upstream estimator would normally supply
them.
"""

from dataclasses import dataclass

import torch


@dataclass
class DroneAltitudeExample:
    time: torch.Tensor  # [T] step indices
    altitude_mean: torch.Tensor  # [T] meters
    altitude_std: torch.Tensor  # [T] meters, illustrative uncertainty envelope
    bounds_above_50: torch.Tensor  # [1, T, 2] illustrative P(altitude >= 50m)
    bounds_above_55: torch.Tensor  # [1, T, 2] illustrative P(altitude >= 55m)


def drone_altitude_example(dtype=torch.float64):
    time = torch.arange(9)
    altitude_mean = torch.tensor(
        [58, 56, 52, 48, 51, 54, 57, 60, 62], dtype=dtype
    )
    altitude_std = torch.tensor(
        [1.5, 1.5, 2.0, 2.5, 2.0, 1.5, 1.2, 1.0, 1.0], dtype=dtype
    )
    bounds_above_50 = torch.tensor(
        [
            [0.95, 0.98],
            [0.90, 0.96],
            [0.55, 0.75],
            [0.15, 0.35],
            [0.45, 0.65],
            [0.75, 0.88],
            [0.88, 0.95],
            [0.93, 0.97],
            [0.94, 0.98],
        ],
        dtype=dtype,
    ).unsqueeze(0)
    bounds_above_55 = torch.tensor(
        [
            [0.75, 0.88],
            [0.55, 0.70],
            [0.10, 0.25],
            [0.02, 0.08],
            [0.08, 0.20],
            [0.30, 0.48],
            [0.60, 0.78],
            [0.82, 0.92],
            [0.88, 0.95],
        ],
        dtype=dtype,
    ).unsqueeze(0)

    return DroneAltitudeExample(
        time=time,
        altitude_mean=altitude_mean,
        altitude_std=altitude_std,
        bounds_above_50=bounds_above_50,
        bounds_above_55=bounds_above_55,
    )
