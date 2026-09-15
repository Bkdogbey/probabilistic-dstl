from typing import NamedTuple

import torch

from models.beliefs import create_gaussian_belief_trajectory
from pdstl.base import BeliefTrajectory


class BeliefRollout(NamedTuple):
    """Predicted beliefs, with optional traces for costs and plots.

    Only belief_trajectory defines the semantics consumed by pdSTL. Diagnostics
    neither identify the belief representation nor replace its probability bounds.
    """

    belief_trajectory: BeliefTrajectory
    nominal_trace: torch.Tensor | None = None  # [1, T+1, D], when supplied
    aux: dict[str, torch.Tensor] | None = None

    def detach_diagnostics(self):
        """Detach optional tensors, preserving the original belief trajectory.

        This does not detach beliefs or promise a graph-free rollout.
        """
        return self._replace(
            nominal_trace=(
                None if self.nominal_trace is None else self.nominal_trace.detach()
            ),
            aux=(
                None if self.aux is None
                else {name: trace.detach() for name, trace in self.aux.items()}
            ),
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
