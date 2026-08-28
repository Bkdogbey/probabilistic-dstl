"""Two standalone three-step Gaussian altitude beliefs, one per temporal
operator experiment. Atomic probabilities come from the belief itself via
the standard normal CDF (torch.special.ndtr), not hand-authored bounds.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AltitudeBelief:
    time: torch.Tensor  # [T]
    mean: torch.Tensor  # [T] meters
    std: torch.Tensor  # [T] meters
    threshold: float  # meters
    probability_bounds: torch.Tensor  # [1, T, 2], degenerate [p, p]


def _belief(mean, std, threshold, dtype):
    mean = torch.tensor(mean, dtype=dtype)
    std = torch.tensor(std, dtype=dtype)
    time = torch.arange(mean.shape[0])
    p = torch.special.ndtr((mean - threshold) / std)
    probability_bounds = torch.stack([p, p], dim=-1).unsqueeze(0)
    return AltitudeBelief(time, mean, std, threshold, probability_bounds)


def always_altitude_example(dtype=torch.float64):
    """Mean stays above 50m at every step; the belief still assigns risk below it."""
    return _belief([52.0, 53.0, 54.0], [2.0, 2.0, 2.0], 50.0, dtype)


def eventually_altitude_example(dtype=torch.float64):
    """Mean crosses 55m only at t=2."""
    return _belief([52.0, 54.0, 56.0], [1.0, 1.0, 1.0], 55.0, dtype)
