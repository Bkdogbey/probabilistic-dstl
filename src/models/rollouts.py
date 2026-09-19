from typing import NamedTuple

import torch

from models.beliefs import (
    GaussianBeliefTrajectory,
    ProbabilityBelief,
    interval_probability,
)
from pdstl.base import BeliefTrajectory, check_probability_bounds
from pdstl.predicates import RelativeAxisInterval


class BeliefRollout(NamedTuple):
    """Predicted beliefs (what pdSTL reads) plus optional traces for costs and plots."""

    belief_trajectory: BeliefTrajectory
    nominal_trace: torch.Tensor | None = None  # [1, T+1, D]
    aux: dict[str, torch.Tensor] | None = None


def create_gaussian_belief_trajectory(
    mean, covariance, dtype=None, device=None
):
    """Build Gaussian beliefs from [T], [T, D], or [B, T, D] traces."""
    mean = torch.as_tensor(mean, dtype=dtype, device=device)
    covariance = torch.as_tensor(
        covariance, dtype=mean.dtype, device=mean.device
    )
    if mean.ndim == 1:
        mean, covariance = mean.unsqueeze(-1), covariance.reshape(-1, 1)
    if mean.ndim == 2:
        mean, covariance = mean.unsqueeze(0), covariance.unsqueeze(0)
    if mean.ndim != 3:
        raise ValueError("mean trace must have shape [T], [T,D] or [B,T,D]")
    if covariance.shape[:2] != mean.shape[:2]:
        raise ValueError(
            "covariance must have the same batch and number of steps as mean"
        )
    return GaussianBeliefTrajectory(mean, covariance)


def create_belief_trajectory(traces, dtype=None, device=None):
    """Build beliefs from named [T, 2] supplied probability intervals."""
    if not traces:
        raise ValueError("at least one event trace is required")
    tensors = {}
    for name, trace in traces.items():
        tensor = torch.as_tensor(trace, dtype=dtype, device=device)
        if tensor.ndim != 2 or tensor.shape[-1] != 2:
            raise ValueError(
                f"probability bounds for {name!r} must have shape [T, 2], "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.shape[0] < 1:
            raise ValueError("probability bounds must cover at least one step")
        check_probability_bounds(tensor.unsqueeze(0))
        tensors[name] = tensor

    steps = {name: tensor.shape[0] for name, tensor in tensors.items()}
    if len(set(steps.values())) > 1:
        raise ValueError(
            f"every trace must cover the same number of steps, got {steps}"
        )
    return BeliefTrajectory(
        ProbabilityBelief(
            {name: tensor[t : t + 1] for name, tensor in tensors.items()}
        )
        for t in range(next(iter(steps.values())))
    )


def create_probability_belief_trajectory(
    predicate, bounds, dtype=None, device=None
):
    """Build beliefs from a single event's [T, 2] probability trace."""
    return create_belief_trajectory(
        {predicate.name: bounds}, dtype=dtype, device=device
    )


def gaussian_rollout(dynamics, mean0, cov0):
    """Return rollout(v) -> BeliefRollout of Gaussian beliefs from (mean0, cov0)."""

    def rollout(v):
        mean_trace, cov_trace = dynamics(v, mean0, cov0)
        return BeliefRollout(
            create_gaussian_belief_trajectory(mean_trace[0], cov_trace[0]),
            mean_trace,
            {"mean_trace": mean_trace, "cov_trace": cov_trace},
        )

    return rollout


class LaneBeliefTrajectory(GaussianBeliefTrajectory):
    """Ego beliefs with vectorized relative beliefs for named traffic."""

    def __init__(self, mean, covariance, traffic_mean, traffic_cov, names):
        super().__init__(mean, covariance)
        self.traffic_mean = traffic_mean
        self.traffic_cov = traffic_cov
        self.names = tuple(names)
        self.relative_mean = traffic_mean[..., :2] - mean[..., :2].unsqueeze(2)
        self.relative_cov = traffic_cov[..., :2, :2] + covariance[
            ..., :2, :2
        ].unsqueeze(2)

    def relative_belief(self, vehicle):
        index = self.names.index(vehicle)
        mean = self.relative_mean[:, :, index]
        covariance = self.relative_cov[:, :, index]
        return GaussianBeliefTrajectory(mean, covariance)

    def probability_bounds(self, event):
        if isinstance(event, RelativeAxisInterval):
            index = self.names.index(event.vehicle)
            mean = self.relative_mean[:, :, index, event.dim]
            variance = self.relative_cov[:, :, index, event.dim, event.dim]
            probability = interval_probability(
                event.lower, event.upper, mean, variance
            )
            return torch.stack((probability, probability), dim=-1)
        return super().probability_bounds(event)


def lane_rollout(
    dynamics,
    ego_mean,
    ego_cov,
    traffic_mean,
    traffic_cov,
    traffic_q,
    names,
    speed_bounds,
    accel_bounds,
):
    """Predict ego and constant-velocity traffic with physical controls."""
    traffic_a = dynamics.A

    def rollout(parameters):
        means, covariances = [ego_mean], [ego_cov]
        traffic_means, traffic_covariances = [traffic_mean], [traffic_cov]
        applied = []
        for raw in parameters:
            proposed = dynamics.bound_control(raw)
            speed = means[-1][2]
            lower = torch.maximum(
                speed.new_tensor(accel_bounds[0][0]),
                (speed_bounds[0] - speed) / dynamics.dt,
            )
            upper = torch.minimum(
                speed.new_tensor(accel_bounds[0][1]),
                (speed_bounds[1] - speed) / dynamics.dt,
            )
            longitudinal = torch.clamp(proposed[0], min=lower, max=upper)
            lateral = proposed[1].clamp(*accel_bounds[1])
            control = torch.stack((longitudinal, lateral))
            applied.append(control)
            mean, covariance = dynamics.step(
                means[-1], covariances[-1], control
            )
            means.append(mean)
            covariances.append(covariance)
            traffic_means.append(traffic_means[-1] @ traffic_a.T)
            traffic_covariances.append(
                traffic_a @ traffic_covariances[-1] @ traffic_a.T + traffic_q
            )
        mean_trace = torch.stack(means).unsqueeze(0)
        cov_trace = torch.stack(covariances).unsqueeze(0)
        traffic_mean_trace = torch.stack(traffic_means).unsqueeze(0)
        traffic_cov_trace = torch.stack(traffic_covariances).unsqueeze(0)
        controls = torch.stack(applied)
        return BeliefRollout(
            LaneBeliefTrajectory(
                mean_trace,
                cov_trace,
                traffic_mean_trace,
                traffic_cov_trace,
                names,
            ),
            mean_trace,
            {
                "mean_trace": mean_trace,
                "cov_trace": cov_trace,
                "traffic_mean_trace": traffic_mean_trace,
                "traffic_cov_trace": traffic_cov_trace,
                "applied_controls": controls,
            },
        )

    return rollout
