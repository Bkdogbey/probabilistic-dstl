"""Gaussian altitude beliefs used by the offline temporal examples.

``mean_lower`` and ``mean_upper`` describe ambiguity in the conditional
Gaussian mean. ``std`` is the known conditional Gaussian standard deviation.
The atomic probability bounds are derived from those quantities; they are
never assigned by hand.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AltitudeBelief:
    """An ambiguous Gaussian altitude belief over a finite time trace."""

    time: torch.Tensor
    mean_lower: torch.Tensor
    mean_upper: torch.Tensor
    std: torch.Tensor
    threshold: float
    probability_bounds: torch.Tensor


def _build_altitude_belief(
    *,
    time,
    mean_lower,
    mean_upper,
    std,
    threshold,
    dtype,
):
    """Validate trace data and derive exceedance-probability bounds."""
    time_tensor = torch.as_tensor(time)
    mean_lower_tensor = torch.as_tensor(mean_lower, dtype=dtype)
    mean_upper_tensor = torch.as_tensor(mean_upper, dtype=dtype)
    std_tensor = torch.as_tensor(std, dtype=dtype)

    traces = (time_tensor, mean_lower_tensor, mean_upper_tensor, std_tensor)
    if any(trace.ndim != 1 for trace in traces):
        raise ValueError("time, mean bounds, and std must be one-dimensional traces")
    if len({trace.shape[0] for trace in traces}) != 1:
        raise ValueError("time, mean bounds, and std must have equal trace lengths")
    if bool((mean_lower_tensor > mean_upper_tensor).any()):
        raise ValueError("mean_lower must be less than or equal to mean_upper")
    if bool((std_tensor <= 0).any()):
        raise ValueError("std must be strictly positive")

    threshold = float(threshold)
    p_lower = torch.special.ndtr((mean_lower_tensor - threshold) / std_tensor)
    p_upper = torch.special.ndtr((mean_upper_tensor - threshold) / std_tensor)
    probability_bounds = torch.stack((p_lower, p_upper), dim=-1).unsqueeze(0)

    return AltitudeBelief(
        time=time_tensor,
        mean_lower=mean_lower_tensor,
        mean_upper=mean_upper_tensor,
        std=std_tensor,
        threshold=threshold,
        probability_bounds=probability_bounds,
    )


def always_altitude_example(threshold=50.0, dtype=torch.float64):
    """Return the seven-step belief used by the offline Always example."""
    nominal_mean = torch.tensor(
        [54.0, 53.0, 52.0, 51.0, 49.0, 52.0, 54.0], dtype=dtype
    )
    mean_radius = 0.5
    return _build_altitude_belief(
        time=[0, 1, 2, 3, 4, 5, 6],
        mean_lower=nominal_mean - mean_radius,
        mean_upper=nominal_mean + mean_radius,
        std=[1.5] * 7,
        threshold=threshold,
        dtype=dtype,
    )


def eventually_altitude_example(threshold=55.0, dtype=torch.float64):
    """Return the seven-step belief used by the offline Eventually example."""
    nominal_mean = torch.tensor(
        [50.0, 51.0, 53.0, 54.0, 56.0, 57.0, 58.0], dtype=dtype
    )
    mean_radius = 0.5
    return _build_altitude_belief(
        time=[0, 1, 2, 3, 4, 5, 6],
        mean_lower=nominal_mean - mean_radius,
        mean_upper=nominal_mean + mean_radius,
        std=[1.5] * 7,
        threshold=threshold,
        dtype=dtype,
    )
