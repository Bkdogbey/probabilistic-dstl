import torch
import numpy as np


class STL_Formula(torch.nn.Module):
    """
    Base class for Probabilistic STL formulas.
    Unlike standard STL, inputs are tuples of (lower_bound, upper_bound) representing
    the uncertainty in the system state.
    """

    def __init__(self):
        super(STL_Formula, self).__init__()

    def robustness_trace(self, bounds, pscale=1, scale=-1, keepdim=True, **kwargs):
        """
        Compute robustness trace for bounded inputs.

        Args:
            bounds: tuple of (lower_bound, upper_bound) tensors
            pscale: predicate scale
            scale: smoothing scale for min/max operations
            keepdim: whether to keep dimensions

        Returns:
            tuple of (lower_robustness, upper_robustness)
        """
        raise NotImplementedError("robustness_trace not yet implemented")

    def forward(self, bounds, pscale=1, scale=-1, keepdim=True, **kwargs):
        """Forward pass delegates to robustness_trace"""
        return self.robustness_trace(
            bounds, pscale=pscale, scale=scale, keepdim=keepdim, **kwargs
        )


class Minish(torch.nn.Module):
    """Smoothed minimum operation (or exact min if scale <= 0)"""

    def __init__(self):
        super(Minish, self).__init__()

    def forward(self, x, scale, dim=1, keepdim=True):
        if scale > 0:
            # Smooth minimum using LogSumExp
            return -torch.logsumexp(-x * scale, dim=dim, keepdim=keepdim) / scale
        else:
            # Exact minimum
            return x.min(dim, keepdim=keepdim)[0]


class GreaterThan(STL_Formula):
    """
    Probabilistic predicate: x >= threshold
    For bounds (x_lower, x_upper), robustness is:
    - lower_robustness = x_lower - threshold (pessimistic)
    - upper_robustness = x_upper - threshold (optimistic)
    """

    def __init__(self, threshold):
        super(GreaterThan, self).__init__()
        self.threshold = threshold
        self.threshold = threshold

    def robustness_trace(self, bounds, pscale=1, **kwargs):
        """
        Args:
            bounds: tuple of (lower_bound, upper_bound)
        Returns:
            tuple of (lower_robustness, upper_robustness)
        """
        lower_bound, upper_bound = bounds

        # Lower robustness: pessimistic (use lower bound)
        lower_rob = (lower_bound - self.threshold) * pscale

        # Upper robustness: optimistic (use upper bound)
        upper_rob = (upper_bound - self.threshold) * pscale

        return (lower_rob, upper_rob)

    def __str__(self):
        return f"x >= {self.threshold}"


class Always(STL_Formula):
    """
    Always (□) operator with time interval.

    For Always, we propagate bounds through time:
    - lower_robustness: min over time of lower bounds (most pessimistic)
    - upper_robustness: min over time of upper bounds (least pessimistic)
    """

    def __init__(self, subformula, interval=None):
        super(Always, self).__init__()
        self.subformula = subformula
        self.interval = interval
        self._interval = [0, np.inf] if interval is None else interval

        # RNN setup for bounded interval
        self.rnn_dim = 1 if not interval else interval[-1]
        if self.rnn_dim == np.inf:
            self.rnn_dim = interval[0]

        self.steps = 1 if not interval else interval[-1] - interval[0] + 1

        # Shift matrices for RNN state
        self.M = torch.tensor(np.diag(np.ones(self.rnn_dim - 1), k=1)).float()
        self.b = torch.zeros(self.rnn_dim).unsqueeze(-1).float()
        self.b[-1] = 1.0

        self.operation = Minish()
        self.operation = Minish()

    def _initialize_rnn_cell(self, lower_trace, upper_trace):
        """Initialize RNN hidden state with the last value (padding)"""
        if lower_trace.is_cuda:
            self.M = self.M.cuda()
            self.b = self.b.cuda()

        # Initialize with last value (time-reversed, so first value)
        h0_lower = (
            torch.ones(
                [lower_trace.shape[0], self.rnn_dim, lower_trace.shape[2]],
                device=lower_trace.device,
            )
            * lower_trace[:, :1, :]
        )
        h0_upper = (
            torch.ones(
                [upper_trace.shape[0], self.rnn_dim, upper_trace.shape[2]],
                device=upper_trace.device,
            )
            * upper_trace[:, :1, :]
        )

        # For interval [a, inf), need special handling
        if (self._interval[1] == np.inf) and (self._interval[0] > 0):
            d0_lower = lower_trace[:, :1, :]
            d0_upper = upper_trace[:, :1, :]
            return ((d0_lower, h0_lower), (d0_upper, h0_upper))

        return (h0_lower, h0_upper)

    def _rnn_cell(self, x_lower, x_upper, h_lower, h_upper, scale=-1):
        """
        RNN cell for Always operator with probabilistic bounds.

        Args:
            x_lower, x_upper: current input bounds [batch_size, 1, x_dim]
            h_lower, h_upper: hidden state bounds [batch_size, rnn_dim, x_dim]
            scale: smoothing parameter

        Returns:
            output_bounds, state_bounds
        """
        if self.interval is None:
            # No interval: running minimum
            input_lower = torch.cat([h_lower, x_lower], dim=1)
            input_upper = torch.cat([h_upper, x_upper], dim=1)

            output_lower = self.operation(input_lower, scale, dim=1, keepdim=True)
            output_upper = self.operation(input_upper, scale, dim=1, keepdim=True)

            state_lower = output_lower
            state_upper = output_upper

        elif (self._interval[1] == np.inf) and (self._interval[0] > 0):
            # Interval [a, inf)
            d0_lower, h0_lower = h_lower
            d0_upper, h0_upper = h_upper

            dh_lower = torch.cat([d0_lower, h0_lower[:, :1, :]], dim=1)
            dh_upper = torch.cat([d0_upper, h0_upper[:, :1, :]], dim=1)

            output_lower = self.operation(dh_lower, scale, dim=1, keepdim=True)
            output_upper = self.operation(dh_upper, scale, dim=1, keepdim=True)

            state_lower = (
                output_lower,
                torch.matmul(self.M, h0_lower) + self.b * x_lower,
            )
            state_upper = (
                output_upper,
                torch.matmul(self.M, h0_upper) + self.b * x_upper,
            )

        else:
            # Interval [a, b]
            state_lower = torch.matmul(self.M, h_lower) + self.b * x_lower
            state_upper = torch.matmul(self.M, h_upper) + self.b * x_upper

            h0x_lower = torch.cat([h_lower, x_lower], dim=1)
            h0x_upper = torch.cat([h_upper, x_upper], dim=1)

            input_lower = h0x_lower[:, : self.steps, :]
            input_upper = h0x_upper[:, : self.steps, :]

            output_lower = self.operation(input_lower, scale, dim=1, keepdim=True)
            output_upper = self.operation(input_upper, scale, dim=1, keepdim=True)

        return (output_lower, output_upper), (state_lower, state_upper)

    def _run_cell(self, lower_trace, upper_trace, scale):
        """Run the RNN cell through the entire trace"""
        outputs_lower = []
        outputs_upper = []

        h_lower, h_upper = self._initialize_rnn_cell(lower_trace, upper_trace)

        # Split traces by time
        xs_lower = torch.split(lower_trace, 1, dim=1)
        xs_upper = torch.split(upper_trace, 1, dim=1)
        time_dim = len(xs_lower)

        for i in range(time_dim):
            (o_lower, o_upper), (h_lower, h_upper) = self._rnn_cell(
                xs_lower[i], xs_upper[i], h_lower, h_upper, scale
            )
            outputs_lower.append(o_lower)
            outputs_upper.append(o_upper)

        return (torch.cat(outputs_lower, dim=1), torch.cat(outputs_upper, dim=1))

    def robustness_trace(self, bounds, pscale=1, scale=-1, keepdim=True, **kwargs):
        """
        Compute robustness trace of Always operator.

        Args:
            bounds: tuple of (lower_bound, upper_bound) from parent

        Returns:
            tuple of (lower_robustness_trace, upper_robustness_trace)
        """
        # First compute subformula robustness
        sub_lower, sub_upper = self.subformula(
            bounds, pscale=pscale, scale=scale, keepdim=keepdim, **kwargs
        )

        # Then apply temporal operator
        return self._run_cell(sub_lower, sub_upper, scale=scale)

    def __str__(self):
        return f"□{self._interval}({self.subformula})"
