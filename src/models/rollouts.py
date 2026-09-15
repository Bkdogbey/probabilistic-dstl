from typing import NamedTuple

import torch

from models.beliefs import create_gaussian_belief_trajectory
from pdstl.base import BeliefTrajectory


class BeliefRollout(NamedTuple):
    """Predicted beliefs (what pdSTL reads) plus optional traces for costs and plots."""

    belief_trajectory: BeliefTrajectory
    nominal_trace: torch.Tensor | None = None  # [1, T+1, D]
    aux: dict[str, torch.Tensor] | None = None

    def detach_diagnostics(self):
        """Detach nominal_trace and aux; the belief trajectory is kept as is."""
        return self._replace(
            nominal_trace=None if self.nominal_trace is None else self.nominal_trace.detach(),
            aux=None if self.aux is None else {k: t.detach() for k, t in self.aux.items()},
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
