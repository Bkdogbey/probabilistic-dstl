"""pdSTL operators on [B, N, 2] probability-interval traces (Frechet Boolean, windowed temporal)."""

import numpy as np
import torch

from pdstl.base import check_probability_bounds


class STL_Formula(torch.nn.Module):
    """Base formula: robustness_trace(beliefs) -> [B, N, 2] at complete origins."""

    @property
    def is_pointwise(self) -> bool:
        return False

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        raise NotImplementedError("robustness_trace not yet implemented")

    def forward(self, belief_trajectory, **kwargs):
        return self.robustness_trace(belief_trajectory, **kwargs)

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Negation(self)


class Minish(torch.nn.Module):
    """Exact (scale <= 0) or log-sum-exp smooth minimum along dim."""

    def forward(self, x, scale, dim=1, keepdim=True):
        if scale > 0:
            return -torch.logsumexp(-x * scale, dim=dim, keepdim=keepdim) / scale
        return x.min(dim=dim, keepdim=keepdim)[0]


class Maxish(torch.nn.Module):
    """Exact (scale <= 0) or log-sum-exp smooth maximum along dim."""

    def forward(self, x, scale, dim=1, keepdim=True):
        if scale > 0:
            return torch.logsumexp(x * scale, dim=dim, keepdim=keepdim) / scale
        return x.max(dim=dim, keepdim=keepdim)[0]


# --- Atomic events ------------------------------------------------------------


class Predicate(STL_Formula):
    """Atomic event; each belief supplies its probability bounds."""

    def __init__(self, name=None):
        super().__init__()
        self.name = name

    @property
    def is_pointwise(self):
        return True

    def robustness_trace(self, belief_trajectory, validate=True, **kwargs):
        trace = torch.stack([b.probability_bounds(self) for b in belief_trajectory], dim=1)
        if validate:
            check_probability_bounds(trace, self)
        return trace

    def __str__(self):
        return self.name if self.name is not None else type(self).__name__


class _Threshold(Predicate):
    sense = None

    def __init__(self, threshold, dim=0, name=None):
        super().__init__(name=name or f"x[{dim}] {self.sense} {threshold}")
        self.threshold = threshold
        self.dim = dim


class GreaterThan(_Threshold):
    """Event x[dim] >= threshold."""

    sense = ">="


class LessThan(_Threshold):
    """Event x[dim] <= threshold."""

    sense = "<="


# --- Boolean operators ----------------------------------------------------------


def _align(*traces):
    """Truncate traces to their common number of origins."""
    n = min(t.shape[1] for t in traces)
    return tuple(t[:, :n] for t in traces)


def _conjunction(trace1, trace2):
    """[max(0, L1 + L2 - 1), min(U1, U2)]"""
    lower = torch.clamp(trace1[..., 0] + trace2[..., 0] - 1.0, min=0.0)
    upper = torch.minimum(trace1[..., 1], trace2[..., 1])
    return torch.stack([lower, upper], dim=-1)


def _disjunction(trace1, trace2):
    """[max(L1, L2), min(1, U1 + U2)]"""
    lower = torch.maximum(trace1[..., 0], trace2[..., 0])
    upper = torch.clamp(trace1[..., 1] + trace2[..., 1], max=1.0)
    return torch.stack([lower, upper], dim=-1)


def _negation(trace):
    """[1 - U, 1 - L]"""
    return torch.stack([1.0 - trace[..., 1], 1.0 - trace[..., 0]], dim=-1)


class Negation(STL_Formula):
    """¬φ."""

    def __init__(self, subformula):
        super().__init__()
        self.subformula = subformula

    @property
    def is_pointwise(self):
        return self.subformula.is_pointwise

    def robustness_trace(self, belief_trajectory, **kwargs):
        return _negation(self.subformula(belief_trajectory, **kwargs))

    def __str__(self):
        return f"¬({self.subformula})"


class _Binary(STL_Formula):
    combine, symbol = None, None

    def __init__(self, subformula1, subformula2):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    @property
    def is_pointwise(self):
        return self.subformula1.is_pointwise and self.subformula2.is_pointwise

    def robustness_trace(self, belief_trajectory, **kwargs):
        trace1 = self.subformula1(belief_trajectory, **kwargs)
        trace2 = self.subformula2(belief_trajectory, **kwargs)
        return type(self).combine(*_align(trace1, trace2))

    def __str__(self):
        return f"({self.subformula1}) {self.symbol} ({self.subformula2})"


class And(_Binary):
    """φ₁ ∧ φ₂ (Frechet conjunction)."""

    combine, symbol = _conjunction, "∧"


class Or(_Binary):
    """φ₁ ∨ φ₂ (Frechet disjunction)."""

    combine, symbol = _disjunction, "∨"


class Implies(STL_Formula):
    """φ₁ ⇒ φ₂ := ¬φ₁ ∨ φ₂."""

    def __init__(self, subformula1, subformula2):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.equivalent = Or(Negation(subformula1), subformula2)

    @property
    def is_pointwise(self):
        return self.equivalent.is_pointwise

    def robustness_trace(self, belief_trajectory, **kwargs):
        return self.equivalent(belief_trajectory, **kwargs)

    def __str__(self):
        return f"({self.subformula1}) ⇒ ({self.subformula2})"


# --- Temporal operators ---------------------------------------------------------


class Temporal_Operator(STL_Formula):
    """Window reduction (min: Always, max: Eventually) run backwards; scale > 0 smooths."""

    symbol = None  # subclasses set self.operation to Minish() or Maxish()

    def __init__(self, subformula, interval=None):
        super().__init__()
        self.subformula = subformula
        self.interval = interval
        self._interval = [0, np.inf] if interval is None else interval

        a, b = self._interval
        self._unbounded = np.isinf(b)
        # [0, inf) is the same window as no interval: the rest of the trace.
        self._suffix = interval is None or (self._unbounded and int(a) == 0)

        if self._suffix:
            self.rnn_dim = 1
        elif self._unbounded:
            self.rnn_dim = int(a)
        else:
            self.rnn_dim = int(b) + 1

        # Shift register: M drops the oldest slot, b writes the newest one.
        self.register_buffer("M", torch.tensor(np.diag(np.ones(self.rnn_dim - 1), k=1)).float())
        b_vec = torch.zeros(self.rnn_dim, 1)
        b_vec[-1] = 1.0
        self.register_buffer("b", b_vec)

    @property
    def lookahead(self):
        """Future steps one evaluation needs (0 for a suffix window)."""
        a, b = self._interval
        if self._suffix:
            return 0
        return int(a) if self._unbounded else int(b)

    def _initialize_rnn_cell(self, x):
        """Register pre-filled with the first reversed value; those outputs are dropped."""
        h0 = x[:, :1, :].expand(-1, self.rnn_dim, -1).clone()  # [B, rnn_dim, 2]
        if self._unbounded and not self._suffix:  # [a, inf), a > 0
            return ((x[:, :1, :], h0), 0.0)
        return (h0, 0.0)

    def _apply_shift(self, h0, x):
        """M @ h0 + b * x, applied to each interval endpoint."""
        batch, rnn_dim, bounds = h0.shape
        M = self.M.to(dtype=h0.dtype, device=h0.device)
        b = self.b.to(dtype=h0.dtype, device=h0.device)
        h0_flat = h0.permute(0, 2, 1).reshape(-1, rnn_dim)  # [B*2, rnn_dim]
        shifted = torch.matmul(h0_flat, M.t()).reshape(batch, bounds, rnn_dim).permute(0, 2, 1)
        return shifted + b.view(1, -1, 1) * x.squeeze(1).unsqueeze(1)

    def _rnn_cell(self, x, hc, scale=-1):
        h0, _ = hc
        if self._suffix:  # [t, end of trace]
            output = self.operation(torch.cat([h0, x], dim=1), scale, dim=1, keepdim=True)
            state = (output, None)
        elif self._unbounded:  # [a, inf), a > 0
            d0, h0 = h0
            dh = torch.cat([d0, h0[:, :1, :]], dim=1)
            output = self.operation(dh, scale, dim=1, keepdim=True)
            state = ((output, self._apply_shift(h0, x)), None)
        else:  # [a, b]: slot rnn_dim-1-k holds the value k steps ahead
            a, b = int(self._interval[0]), int(self._interval[1])
            new_h0 = self._apply_shift(h0, x)
            output = self.operation(new_h0[:, : b - a + 1, :], scale, dim=1, keepdim=True)
            state = (new_h0, None)
        return output, state

    def _run_cell(self, x, scale):
        outputs, hc = [], self._initialize_rnn_cell(x)
        for x_i in torch.split(x, 1, dim=1):
            o, hc = self._rnn_cell(x_i, hc, scale)
            outputs.append(o)
        return torch.cat(outputs, dim=1)

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        trace = self.subformula(belief_trajectory, scale=scale, keepdim=keepdim, **kwargs)
        if trace.shape[1] - self.lookahead <= 0:
            raise ValueError(
                f"{self}: one evaluation origin needs {self.lookahead + 1} steps, "
                f"child trace has {trace.shape[1]}"
            )
        # Run backwards so each origin sees its future; drop incomplete windows.
        output_reversed = self._run_cell(torch.flip(trace, dims=[1]), scale=scale)
        return torch.flip(output_reversed[:, self.lookahead :], dims=[1])

    def __str__(self):
        if self.interval is None:
            return f"{self.symbol}({self.subformula})"
        return f"{self.symbol}_{self._interval}({self.subformula})"


class Always(Temporal_Operator):
    """□_[a,b] φ: minimum over the window."""

    symbol = "□"

    def __init__(self, subformula, interval=None):
        super().__init__(subformula, interval)
        self.operation = Minish()


class Eventually(Temporal_Operator):
    """♢_[a,b] φ: maximum over the window."""

    symbol = "♢"

    def __init__(self, subformula, interval=None):
        super().__init__(subformula, interval)
        self.operation = Maxish()


class Until(STL_Formula):
    """φ U_[a,b] ψ: max over witnesses τ of Frechet(min φ on [t, τ], ψ(τ))."""

    def __init__(self, left, right, interval=None):
        super().__init__()
        self.left = left
        self.right = right
        self.interval = [0, np.inf] if interval is None else interval
        self._interval = self.interval
        self.min_op = Minish()
        self.max_op = Maxish()

    @property
    def lookahead(self):
        a, b = self._interval
        return int(a) if np.isinf(b) else int(b)

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        phi = self.left(belief_trajectory, scale=scale, keepdim=True, **kwargs)
        psi = self.right(belief_trajectory, scale=scale, keepdim=True, **kwargs)
        phi, psi = _align(phi, psi)

        T = phi.shape[1]
        a = int(self._interval[0])
        b = T - 1 if np.isinf(self._interval[1]) else int(self._interval[1])
        n_valid = T - self.lookahead
        if n_valid <= 0:
            raise ValueError(
                f"{self}: one evaluation origin needs {self.lookahead + 1} steps, "
                f"child traces have {T}"
            )

        results = []
        for t in range(n_valid):
            candidates = []
            for tau in range(t + a, min(t + b, T - 1) + 1):
                # Inclusive: φ must hold from t through the witness τ.
                prefix = self.min_op(phi[:, t : tau + 1, :], scale, dim=1, keepdim=False)
                candidates.append(_conjunction(prefix, psi[:, tau, :]))
            results.append(self.max_op(torch.stack(candidates, dim=1), scale, dim=1, keepdim=False))
        return torch.stack(results, dim=1)

    def __str__(self):
        return f"({self.left}) U_{self._interval} ({self.right})"
