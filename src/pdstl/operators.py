import torch


class STL_Formula(torch.nn.Module):
    """
    Base class for Probabilistic STL formulas.
    Unlike standard STL, inputs are tuples of (lower_bound, upper_bound) representing
    the uncertainty in the system state.
    """

    def __init__(self):
        super(STL_Formula, self).__init__()

    def robustness_trace(self, belief, scale=-1, keepdim=True, **kwargs):
        """
        Compute robustness trace for bounded inputs.

        Args:
           belief: (mean, variance) [batch_size, time_dim, x_dim]

        Returns:
           [B,T,D,2] probability bounds on robustness trace
        """
        raise NotImplementedError("robustness_trace not yet implemented")

    def forward(self, belief, **kwargs):
        """Forward pass delegates to robustness_trace"""
        return self.robustness_trace(belief, **kwargs)


class Minish(torch.nn.Module):
    """Compute minimum (exact or smooth) over specified dimension"""

    def forward(self, x, scale, dim=1, keepdim=True):
        if scale > 0:
            # Smooth minimum using LogSumExp
            return -torch.logsumexp(-x * scale, dim=dim, keepdim=keepdim) / scale
        else:
            # Exact minimum
            return x.min(dim, keepdim=keepdim)[0]


class GreaterThan(STL_Formula):
    """
    Predicate: x >= threshold
    computes P(x >= threshold) for probabilistic bounds
    x ~ N(mean, variance)
    """

    def __init__(self, threshold):
        super(GreaterThan, self).__init__()
        self.threshold = threshold

    def robustness_trace(self, belief_trajectory, **kwargs):
        probs = []

        for t in range(len(belief_trajectory)):
            belief = belief_trajectory[t]  # get the belief at time t
            residual = belief.value() - self.threshold
            prob = belief.probability_of(residual)

            probs.append(prob)

        return torch.stack(
            [prob, prob], dim=-1
        )  # returns upper and lower bounds (same for this predicate)

    def __str__(self):
        return f"x >= {self.threshold}"
