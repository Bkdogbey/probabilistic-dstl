import torch


def normal_cdf(z):
    """Cumulative distribution function for standard normal distribution"""
    return 0.5 * (1 + torch.erf(z / torch.sqrt(torch.tensor(2.0))))


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

    def robustness_trace(self, belief, **kwargs):
        mean, var = belief

        # Compute probability
        std = torch.sqrt(var)
        z = (mean - self.threshold) / std
        prob = normal_cdf(z)
        # NOTE: Where are we returning the robustness measure?

    def __str__(self):
        return f"x >= {self.threshold}"
