from typing import NamedTuple

import torch

from models.dynamics import (
    create_enclosure_belief_trajectory,
    create_gaussian_belief_trajectory,
)
from pdstl.base import BeliefTrajectory


class BeliefRollout(NamedTuple):
    """Predicted beliefs for one control sequence, plus traces for costs and plots."""

    belief_trajectory: BeliefTrajectory
    nominal_trace: torch.Tensor  # [1, T+1, D]
    aux: dict

    def detached(self):
        return self._replace(
            nominal_trace=self.nominal_trace.detach(),
            aux={name: trace.detach() for name, trace in self.aux.items()},
        )


def gaussian_rollout(dynamics, mean0, cov0):
    """v -> BeliefRollout of precise Gaussian beliefs; nominal trace is the mean."""

    def rollout(v):
        mean_trace, cov_trace = dynamics(v, mean0, cov0)
        return BeliefRollout(
            create_gaussian_belief_trajectory(mean_trace[0], cov_trace[0]),
            mean_trace,
            {"mean_trace": mean_trace, "cov_trace": cov_trace},
        )

    return rollout


def enclosure_rollout(dynamics, lower0, upper0, cov0, d_lower=None, d_upper=None):
    """v -> BeliefRollout of enclosure beliefs; nominal trace is the midpoint."""

    def rollout(v):
        lower_trace, upper_trace, cov_trace = dynamics.rollout_enclosure(
            v, lower0, upper0, cov0, d_lower, d_upper
        )
        return BeliefRollout(
            create_enclosure_belief_trajectory(lower_trace[0], upper_trace[0], cov_trace[0]),
            (lower_trace + upper_trace) / 2,
            {"lower_trace": lower_trace, "upper_trace": upper_trace, "cov_trace": cov_trace},
        )

    return rollout
