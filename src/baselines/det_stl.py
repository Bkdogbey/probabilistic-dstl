"""STLCG-style recurrent STL robustness, with a forward-time interface.

Adapted from StanfordASL/stlcg/src/stlcg.py (blob c947ef97fd95631025ce515ec03d9e406ccb047f).
The temporal shift register and hard/log-sum-exp reductions follow STLCG.
Unlike its terminal padding, this adapter returns only complete evaluation
origins, matching the current pdSTL formula horizon and inclusive Until.
This is deterministic robustness guidance, never a probability bound.

MIT License
Copyright (c) 2019 Karen Leung, Nikos Arechiga

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math

import torch
from torch import nn

from pdstl import operators as pd
from pdstl.predicates import (
    AxisInterval,
    HalfSpace,
    RelativeAxisInterval,
    _Rectangle,
)


def _checked_beta(beta):
    if beta is not None and (not math.isfinite(beta) or beta <= 0):
        raise ValueError("auxiliary beta must be finite and positive or None")
    return beta


def _reduce(values, beta, *, minimum, dim=1):
    """Independent deterministic reductions; no probability-bound shifts."""
    if beta is None:
        operation = torch.amin if minimum else torch.amax
        return operation(values, dim=dim, keepdim=True)
    sign = -1 if minimum else 1
    return (
        sign
        * torch.logsumexp(sign * beta * values, dim=dim, keepdim=True)
        / beta
    )


def _interval(interval):
    a, b = (0, math.inf) if interval is None else interval
    if not math.isfinite(a) or a < 0 or int(a) != a:
        raise ValueError("interval start must be a non-negative integer")
    if b != math.inf and (not math.isfinite(b) or int(b) != b):
        raise ValueError("interval end must be an integer or infinity")
    if a > b:
        raise ValueError("interval start must not exceed its end")
    return int(a), math.inf if b == math.inf else int(b)


def _align(left, right):
    n = min(left.shape[1], right.shape[1])
    return left[:, :n], right[:, :n]


class DetSTL_Formula(nn.Module):
    """Evaluate a belief trajectory, or a raw mean tensor [B,T,D].

    Traces are [B,valid_origins,1], in forward time. beta=None uses hard
    robustness; positive beta uses STLCG log-sum-exp smoothing.
    """

    def forward(self, trajectory, beta=None):
        return self.robustness_trace(trajectory, beta=_checked_beta(beta))

    def robustness_trace(self, trajectory, beta=None):
        raise NotImplementedError

    def robustness(self, trajectory, beta=None, origin=0):
        trace = self(trajectory, beta=beta)
        if not isinstance(origin, int) or not 0 <= origin < trace.shape[1]:
            raise ValueError("origin is outside the complete robustness trace")
        return trace[:, origin, 0]

    def __and__(self, other):
        return DetAnd(self, other)

    def __or__(self, other):
        return DetOr(self, other)

    def __invert__(self):
        return DetNegation(self)


class DetPredicate(DetSTL_Formula):
    """Atomic geometry copied from a pdSTL event without registering its module."""

    def __init__(self, event, normalization="none", sigma_floor=1e-3):
        super().__init__()
        if normalization not in ("none", "std"):
            raise ValueError("normalization must be 'none' or 'std'")
        if not math.isfinite(sigma_floor) or sigma_floor <= 0:
            raise ValueError("sigma_floor must be finite and positive")
        self.normalization, self.sigma_floor = normalization, sigma_floor
        self.name = event.name
        if isinstance(event, AxisInterval):
            self.kind = "axis"
            self.lower, self.upper, self.dim = (
                event.lower,
                event.upper,
                event.dim,
            )
            self.vehicle = (
                event.vehicle
                if isinstance(event, RelativeAxisInterval)
                else None
            )
        elif isinstance(event, HalfSpace):
            self.kind, self.a, self.b = "halfspace", tuple(event.a), event.b
        else:
            raise TypeError(
                f"unsupported auxiliary predicate: {type(event).__name__}"
            )

    def _moments(self, trajectory):
        if torch.is_tensor(trajectory):
            if trajectory.ndim != 3 or trajectory.shape[1] == 0:
                raise ValueError("mean trace must have shape [B,T,D], T > 0")
            if self.normalization == "std":
                raise ValueError(
                    "std normalization requires a Gaussian belief trajectory"
                )
            return trajectory, None
        trace = getattr(trajectory, "trace", None)
        if trace is None or not hasattr(trace, "mean"):
            raise ValueError(
                "auxiliary robustness requires a mean-bearing belief trajectory"
            )
        return trace.mean, trace.covariance

    def _standardize(self, margin, variance):
        if self.normalization == "none":
            return margin
        # Floor variance before sqrt: its derivative is otherwise undefined at zero.
        sigma = variance.clamp_min(self.sigma_floor**2).sqrt()
        return margin / sigma

    def robustness_trace(self, trajectory, beta=None):
        mean, covariance = self._moments(trajectory)
        if self.kind == "halfspace":
            normal = mean.new_tensor(self.a)
            if normal.shape != mean.shape[-1:]:
                raise ValueError(
                    "half-space normal must match the state dimension"
                )
            margin = self.b - mean @ normal
            variance = (
                None
                if covariance is None
                else torch.einsum("i,...ij,j->...", normal, covariance, normal)
            )
            return self._standardize(margin, variance).unsqueeze(-1)
        if self.vehicle is not None:
            if not hasattr(trajectory, "relative_mean"):
                raise ValueError(
                    "relative predicates require a lane belief trajectory"
                )
            index = trajectory.names.index(self.vehicle)
            coordinate = trajectory.relative_mean[:, :, index, self.dim]
            variance = trajectory.relative_cov[:, :, index, self.dim, self.dim]
        else:
            if self.dim >= mean.shape[-1]:
                raise ValueError("predicate dimension is outside the state")
            coordinate = mean[..., self.dim]
            variance = (
                None
                if covariance is None
                else covariance[..., self.dim, self.dim]
            )
        margins = []
        if math.isfinite(self.lower):
            margins.append(
                self._standardize(coordinate - self.lower, variance)
            )
        if math.isfinite(self.upper):
            margins.append(
                self._standardize(self.upper - coordinate, variance)
            )
        if not margins:
            return torch.full_like(coordinate.unsqueeze(-1), math.inf)
        return _reduce(
            torch.stack(margins, dim=-1), beta, minimum=True, dim=-1
        )


class _DetBinary(DetSTL_Formula):
    minimum = None

    def __init__(self, left, right):
        super().__init__()
        self.left, self.right = left, right

    def robustness_trace(self, trajectory, beta=None):
        left, right = _align(
            self.left(trajectory, beta), self.right(trajectory, beta)
        )
        return _reduce(
            torch.cat((left, right), dim=-1),
            beta,
            minimum=self.minimum,
            dim=-1,
        )


class DetAnd(_DetBinary):
    minimum = True


class DetOr(_DetBinary):
    minimum = False


class DetNegation(DetSTL_Formula):
    def __init__(self, subformula):
        super().__init__()
        self.subformula = subformula

    def robustness_trace(self, trajectory, beta=None):
        return -self.subformula(trajectory, beta)


class DetImplies(DetOr):
    def __init__(self, left, right):
        super().__init__(DetNegation(left), right)


class DetTemporalOperator(DetSTL_Formula):
    """Reverse-time recurrent scan: suffix accumulator or finite shift register."""

    minimum = None

    def __init__(self, subformula, interval=None):
        super().__init__()
        self.subformula = subformula
        self.interval = _interval(interval)
        a, b = self.interval
        self.lookahead = a if math.isinf(b) else b
        self.rnn_dim = 1 if math.isinf(b) else b + 1
        self.register_buffer(
            "M",
            torch.diag(torch.ones(self.rnn_dim - 1), diagonal=1),
            persistent=False,
        )
        write = torch.zeros(self.rnn_dim, 1)
        write[-1] = 1
        self.register_buffer("write", write, persistent=False)

    def _rnn_cell(self, x, state, beta):
        a, b = self.interval
        if math.isinf(b):
            # Seed with the first observation exactly once (no smoothed padding bias).
            output = (
                x
                if state is None
                else _reduce(
                    torch.cat((state, x), dim=1), beta, minimum=self.minimum
                )
            )
            return output, output
        shifted = torch.matmul(self.M.to(x), state) + self.write.to(x) * x
        output = _reduce(shifted[:, : b - a + 1], beta, minimum=self.minimum)
        return output, shifted

    def robustness_trace(self, trajectory, beta=None):
        values = self.subformula(trajectory, beta)
        n = values.shape[1]
        if n <= self.lookahead:
            raise ValueError(
                "auxiliary temporal window exceeds the available trace"
            )
        reversed_values = values.flip(1)
        state = (
            None
            if math.isinf(self.interval[1])
            else reversed_values[:, :1].expand(-1, self.rnn_dim, -1).clone()
        )
        outputs = []
        for x in reversed_values.split(1, dim=1):
            output, state = self._rnn_cell(x, state, beta)
            outputs.append(output)
        trace = torch.cat(outputs, dim=1).flip(1)
        if math.isinf(self.interval[1]):
            return trace[:, self.interval[0] :]
        return trace[:, : n - self.lookahead]


class DetAlways(DetTemporalOperator):
    minimum = True


class DetEventually(DetTemporalOperator):
    minimum = False


class DetUntil(DetSTL_Formula):
    """Inclusive Until with recurrent prefix and witness accumulators.

    At each origin, carry min(left[t:tau+1]) and then max over witnesses
    tau in [t+a,t+b]. This avoids the upstream quadratic sentinel matrix.
    """

    def __init__(self, left, right, interval=None):
        super().__init__()
        self.left, self.right = left, right
        self.interval = _interval(interval)
        a, b = self.interval
        self.lookahead = a if math.isinf(b) else b

    def robustness_trace(self, trajectory, beta=None):
        left, right = _align(
            self.left(trajectory, beta), self.right(trajectory, beta)
        )
        n = left.shape[1]
        if n <= self.lookahead:
            raise ValueError(
                "auxiliary Until window exceeds the available trace"
            )
        a, b = self.interval
        end = n - 1 if math.isinf(b) else b
        outputs = []
        for t in range(n - self.lookahead):
            prefix, witness = None, None
            for tau in range(t, min(t + end, n - 1) + 1):
                current = left[:, tau : tau + 1]
                prefix = (
                    current
                    if prefix is None
                    else _reduce(
                        torch.cat((prefix, current), dim=1), beta, minimum=True
                    )
                )
                if tau >= t + a:
                    candidate = _reduce(
                        torch.cat((prefix, right[:, tau : tau + 1]), dim=-1),
                        beta,
                        minimum=True,
                        dim=-1,
                    )
                    witness = (
                        candidate
                        if witness is None
                        else _reduce(
                            torch.cat((witness, candidate), dim=1),
                            beta,
                            minimum=False,
                        )
                    )
            outputs.append(witness)
        return torch.cat(outputs, dim=1)


def from_pdstl(spec, normalization="none", sigma_floor=1e-3):
    """Compile the same predicates/windows into an independent recurrent graph.

    Raw mean robustness is a deterministic baseline. 'std' uses Gaussian
    standardized atomic margins as auxiliary optimization guidance.
    """

    def convert(node):
        if isinstance(node, (_Rectangle, pd.Implies)):
            return convert(node.equivalent)
        if isinstance(node, pd.Negation):
            return DetNegation(convert(node.subformula))
        if isinstance(node, (pd.And, pd.Or)):
            cls = DetAnd if isinstance(node, pd.And) else DetOr
            return cls(convert(node.subformula1), convert(node.subformula2))
        if isinstance(node, (pd.Always, pd.Eventually)):
            cls = DetAlways if isinstance(node, pd.Always) else DetEventually
            return cls(convert(node.subformula), node._interval)
        if isinstance(node, pd.Until):
            return DetUntil(
                convert(node.left), convert(node.right), node._interval
            )
        if isinstance(node, (AxisInterval, HalfSpace)):
            return DetPredicate(node, normalization, sigma_floor)
        raise TypeError(
            f"unsupported auxiliary formula: {type(node).__name__}"
        )

    return convert(spec)
