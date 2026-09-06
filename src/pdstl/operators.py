"""StoRI-style interval equations and matrix-based bounded temporal evaluation.

The direct equations use scale <= 0. Positive scale substitutes log-sum-exp;
those endpoint approximations may leave [0,1] or cross. Reevaluate with scale <= 0
for the defined score. Temporal min/max describe pointwise probabilistic
robustness, not whole-trajectory satisfaction probability.
"""

import math
from numbers import Integral

import torch

from .base import STL_Formula, validate_interval_trace, validate_scale


def _interval(interval):
    if interval is None:
        raise ValueError("provide a finite discrete-time interval [a, b]")
    if not isinstance(interval, (tuple, list)) or len(interval) != 2:
        raise ValueError("interval must be [a, b] with integer endpoints")
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in interval):
        raise ValueError("interval endpoints must be finite nonnegative integers")
    a, b = map(int, interval)
    if not 0 <= a <= b:
        raise ValueError("interval must satisfy 0 <= a <= b")
    return (a, b)


def _error(scale, count):
    validate_scale(scale)
    return math.log(count) / scale if scale > 0 else 0.0


def _align(left, right, left_horizon, right_horizon):
    """Align child traces at their common complete prediction origins."""
    if left.ndim != 3 or right.ndim != 3 or left.shape[-1] != 2 or right.shape[-1] != 2:
        raise ValueError("formula children must return [batch, time, 2] traces")
    if (left.shape[0], left.dtype, left.device) != (right.shape[0], right.dtype, right.device):
        raise ValueError("formula children must share batch, dtype, and device")
    if left.shape[1] + left_horizon != right.shape[1] + right_horizon:
        raise ValueError("formula children must describe the same prediction length")
    length = min(left.shape[1], right.shape[1])
    return left[:, :length], right[:, :length]


class Minish(torch.nn.Module):
    """Minimum, or its unnormalized log-sum-exp approximation."""

    def forward(self, x, scale, dim=1, keepdim=True):
        validate_scale(scale)
        if scale > 0:
            return -torch.logsumexp(-x * scale, dim=dim, keepdim=keepdim) / scale
        return x.min(dim=dim, keepdim=keepdim)[0]


class Maxish(torch.nn.Module):
    """Maximum, or its unnormalized log-sum-exp approximation."""

    def forward(self, x, scale, dim=1, keepdim=True):
        validate_scale(scale)
        if scale > 0:
            return torch.logsumexp(x * scale, dim=dim, keepdim=keepdim) / scale
        return x.max(dim=dim, keepdim=keepdim)[0]


class _LegacyComparison(STL_Formula):
    """Legacy state-bound adapter; the provider must justify its probabilities.

    Kept for migration. It does not infer distributional uncertainty from a
    Gaussian's spread. New code should supply probability intervals to Predicate.
    """

    direction = 1

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def robustness_trace(self, belief_trajectory, **kwargs):
        lower, upper = [], []
        for belief in belief_trajectory:
            if not all(hasattr(belief, attr) for attr in ("lower_bound", "upper_bound", "probability_of")):
                raise TypeError("legacy comparison requires state-bound methods; use Predicate for supplied probability intervals")
            low, high = belief.lower_bound(), belief.upper_bound()
            if self.direction > 0:
                lo, hi = low - self.threshold, high - self.threshold
            else:
                lo, hi = self.threshold - high, self.threshold - low
            lower.append(belief.probability_of(lo))
            upper.append(belief.probability_of(hi))
        if not lower:
            raise ValueError("belief trajectory cannot be empty")
        if any(p.ndim != 3 or p.shape[1:] != (1, 1) for p in lower + upper):
            raise ValueError("legacy comparisons require scalar [batch, 1, 1] values; choose a scalar event")
        result = torch.stack(
            [torch.cat(lower, dim=1).squeeze(-1), torch.cat(upper, dim=1).squeeze(-1)], dim=-1
        )
        return validate_interval_trace(result, "legacy comparison provider")

    def smoothing_error(self, scale):
        validate_scale(scale)
        return 0.0

    def __str__(self):
        return f"x {'>=' if self.direction > 0 else '<='} {self.threshold}"


class GreaterThan(_LegacyComparison):
    """Legacy x >= threshold adapter; prefer supplied Predicate intervals."""


class LessThan(_LegacyComparison):
    """Legacy x <= threshold adapter; prefer supplied Predicate intervals."""

    direction = -1


class Negation(STL_Formula):
    """Complement and swap interval endpoints."""

    def __init__(self, subformula):
        super().__init__()
        self.subformula = subformula

    @property
    def horizon(self):
        return self.subformula.horizon

    def robustness_trace(self, inputs, scale=-1, **kwargs):
        trace = self.subformula.robustness_trace(inputs, scale=scale, **kwargs)
        return torch.stack([1 - trace[..., 1], 1 - trace[..., 0]], dim=-1)

    def smoothing_error(self, scale):
        return self.subformula.smoothing_error(scale)

    def __str__(self):
        return f"¬({self.subformula})"


class _BinaryOperator(STL_Formula):
    def __init__(self, subformula1, subformula2):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.min_op = Minish()
        self.max_op = Maxish()

    @property
    def horizon(self):
        return max(self.subformula1.horizon, self.subformula2.horizon)

    def _traces(self, inputs, scale, **kwargs):
        left = self.subformula1.robustness_trace(inputs, scale=scale, **kwargs)
        right = self.subformula2.robustness_trace(inputs, scale=scale, **kwargs)
        return _align(left, right, self.subformula1.horizon, self.subformula2.horizon)

    def smoothing_error(self, scale):
        # Sum branches can add both child errors; min/max is 1-Lipschitz.
        return (self.subformula1.smoothing_error(scale)
                + self.subformula2.smoothing_error(scale) + _error(scale, 2))

    def __str__(self):
        return f"({self.subformula1}) {self.symbol} ({self.subformula2})"


class And(_BinaryOperator):
    """Fréchet conjunction: [max(0,L1+L2-1), min(U1,U2)]."""

    symbol = "∧"

    def robustness_trace(self, inputs, scale=-1, **kwargs):
        left, right = self._traces(inputs, scale, **kwargs)
        candidate = left[..., 0] + right[..., 0] - 1
        lower = self.max_op(torch.stack([candidate, torch.zeros_like(candidate)], -1), scale, dim=-1, keepdim=False)
        upper = self.min_op(torch.stack([left[..., 1], right[..., 1]], -1), scale, dim=-1, keepdim=False)
        return torch.stack([lower, upper], -1)


class Or(_BinaryOperator):
    """Fréchet disjunction: [max(L1,L2), min(1,U1+U2)]."""

    symbol = "∨"

    def robustness_trace(self, inputs, scale=-1, **kwargs):
        left, right = self._traces(inputs, scale, **kwargs)
        lower = self.max_op(torch.stack([left[..., 0], right[..., 0]], -1), scale, dim=-1, keepdim=False)
        candidate = left[..., 1] + right[..., 1]
        upper = self.min_op(torch.stack([candidate, torch.ones_like(candidate)], -1), scale, dim=-1, keepdim=False)
        return torch.stack([lower, upper], -1)


class Implies(STL_Formula):
    """Implication evaluated as negation followed by disjunction."""

    def __init__(self, subformula1, subformula2):
        super().__init__()
        self.equivalent = Or(Negation(subformula1), subformula2)

    @property
    def horizon(self):
        return self.equivalent.horizon

    def robustness_trace(self, inputs, scale=-1, **kwargs):
        return self.equivalent.robustness_trace(inputs, scale=scale, **kwargs)

    def smoothing_error(self, scale):
        return self.equivalent.smoothing_error(scale)

    def __str__(self):
        return f"({self.equivalent.subformula1.subformula}) ⇒ ({self.equivalent.subformula2})"


class Temporal_Operator(STL_Formula):
    """Bounded future windows using the original shift matrix M and vector b."""

    def __init__(self, subformula, interval=None):
        super().__init__()
        self.subformula = subformula
        self.interval = _interval(interval)
        self._interval = self.interval
        self.rnn_dim = self.interval[1] + 1
        matrix = torch.diag(torch.ones(self.rnn_dim - 1), diagonal=1)
        insertion = torch.zeros(self.rnn_dim, 1)
        insertion[-1, 0] = 1
        self.register_buffer("M", matrix)
        self.register_buffer("b", insertion)
        self.operation = None

    @property
    def horizon(self):
        return self.interval[1] + self.subformula.horizon

    def _initialize_rnn_cell(self, x):
        # Zeros are internal placeholders only. No result is reduced or emitted
        # until every register position contains an actual child score.
        return x.new_zeros(x.shape[0], self.rnn_dim, 2), 0

    def _apply_shift(self, h0, x):
        batch, size, bounds = h0.shape
        flat = h0.permute(0, 2, 1).reshape(-1, size)
        shifted = torch.matmul(flat, self.M.to(h0).t())
        shifted = shifted.reshape(batch, bounds, size).permute(0, 2, 1)
        return shifted + self.b.to(h0).view(1, -1, 1) * x

    def _rnn_cell(self, x, hc, scale=-1, **kwargs):
        h0, count = hc
        new_h0 = self._apply_shift(h0, x)
        count += 1
        output = None
        if count >= self.rnn_dim:
            a, b = self.interval
            window = new_h0[:, :b - a + 1, :]
            output = self.operation(window, scale, dim=1, keepdim=True)
        return output, (new_h0, count)

    def _run_cell(self, x, scale):
        if x.shape[1] < self.rnn_dim:
            raise ValueError(f"temporal window requires {self.rnn_dim} child steps; got {x.shape[1]}")
        outputs = []
        hc = self._initialize_rnn_cell(x)
        for sample in torch.split(x, 1, dim=1):
            output, hc = self._rnn_cell(sample, hc, scale)
            if output is not None:
                outputs.append(output)
        return torch.cat(outputs, dim=1)

    def robustness_trace(self, inputs, scale=-1, **kwargs):
        trace = self.subformula.robustness_trace(inputs, scale=scale, **kwargs)
        reversed_trace = torch.flip(trace, dims=[1])
        output = self._run_cell(reversed_trace, scale)
        return torch.flip(output, dims=[1])

    def smoothing_error(self, scale):
        a, b = self.interval
        return self.subformula.smoothing_error(scale) + _error(scale, b - a + 1)

    def __str__(self):
        return f"{self.symbol}_{list(self.interval)}({self.subformula})"


class Always(Temporal_Operator):
    """Componentwise minimum over a complete future window."""

    symbol = "□"

    def __init__(self, subformula, interval=None):
        super().__init__(subformula, interval)
        self.operation = Minish()
        self.oper = "min"


class Eventually(Temporal_Operator):
    """Componentwise maximum over a complete future window."""

    symbol = "◇"

    def __init__(self, subformula, interval=None):
        super().__init__(subformula, interval)
        self.operation = Maxish()
        self.oper = "max"


class Until(STL_Formula):
    """StoRI Until with inclusive left prefix [t,tau] and Fréchet combination.

    Each candidate has lower=max(0, right_lower + prefix_lower - 1),
    upper=min(right_upper, prefix_upper). Maximize each endpoint over witnesses.
    Only complete witness windows are returned. Prefix reductions are reused;
    this is a separate recurrence, not the unary matrix sliding-window cell.
    """

    def __init__(self, left, right, interval=None):
        super().__init__()
        self.left = left
        self.right = right
        self.interval = _interval(interval)
        self._interval = self.interval
        self.min_op = Minish()
        self.max_op = Maxish()

    @property
    def horizon(self):
        return self.interval[1] + max(self.left.horizon, self.right.horizon)

    def robustness_trace(self, inputs, scale=-1, **kwargs):
        left = self.left.robustness_trace(inputs, scale=scale, **kwargs)
        right = self.right.robustness_trace(inputs, scale=scale, **kwargs)
        left, right = _align(left, right, self.left.horizon, self.right.horizon)
        a, b = self.interval
        if left.shape[1] <= b:
            raise ValueError(f"Until requires at least {b + 1} complete child steps")
        results = []
        for t in range(left.shape[1] - b):
            # A one-element prefix is itself, including in smooth evaluation.
            prefix = left[:, t, :]
            candidates = []
            for offset in range(b + 1):
                if offset:
                    pair = torch.stack([prefix, left[:, t + offset, :]], dim=1)
                    prefix = self.min_op(pair, scale, dim=1, keepdim=False)
                if offset < a:
                    continue
                witness = right[:, t + offset, :]
                candidate = prefix[:, 0] + witness[:, 0] - 1
                lower = self.max_op(torch.stack([candidate, torch.zeros_like(candidate)], -1), scale, dim=-1, keepdim=False)
                upper = self.min_op(torch.stack([prefix[:, 1], witness[:, 1]], -1), scale, dim=-1, keepdim=False)
                candidates.append(torch.stack([lower, upper], -1))
            result = self.max_op(torch.stack(candidates, 1), scale, dim=1, keepdim=False)
            results.append(result)
        return torch.stack(results, dim=1)

    def smoothing_error(self, scale):
        a, b = self.interval
        # The prefix log-sum-exp recurrence equals a reduction over all prefix
        # samples, so its approximation error is log(b+1)/scale, not b*log(2).
        return (self.left.smoothing_error(scale) + self.right.smoothing_error(scale)
                + _error(scale, b + 1) + _error(scale, 2) + _error(scale, b - a + 1))

    def __str__(self):
        return f"({self.left}) U_{list(self.interval)} ({self.right})"
