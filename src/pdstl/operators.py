"""pdSTL operators on [B, N, 2] traces: Frechet Boolean, windowed temporal.

Two evaluation modes, selected by `beta`:

* `beta=None` -- exact. The pair is a StoRI probability interval [lower, upper].
* `beta > 0`  -- smooth. The pair is a differentiable surrogate for optimization. It is NOT
  a probability interval: the endpoints need not lie in [0, 1] and need not be ordered. Only
  its lower value is meant to be read, through `smooth_lower()`.

Use `probability_interval()` for the exact result and `smooth_lower()` for the scalar a
planner descends.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from pdstl.base import check_probability_bounds


def _checked_beta(beta):
    """None for exact evaluation, else a finite positive smoothing strength."""
    if beta is None:
        return None
    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError(f"beta must be a finite positive number or None, got {beta}")
    return beta


class STL_Formula(torch.nn.Module):
    """Base formula: robustness_trace(beliefs) -> [B, N, 2] at complete origins."""

    @property
    def is_pointwise(self) -> bool:
        return False

    def robustness_trace(self, belief_trajectory, beta=None, keepdim=True, **kwargs):
        raise NotImplementedError("robustness_trace not yet implemented")

    def forward(self, belief_trajectory, beta=None, **kwargs):
        return self.robustness_trace(belief_trajectory, beta=_checked_beta(beta), **kwargs)

    def probability_interval(self, belief_trajectory, origin=0, **kwargs):
        """The exact pdSTL probability interval [lower, upper] at one origin.

        Named in full because `interval` on a temporal operator is its window [a, b].
        """
        return self(belief_trajectory, beta=None, **kwargs)[0, origin]

    def smooth_lower(self, belief_trajectory, beta, origin=0, **kwargs):
        """The differentiable lower score at one origin: the scalar a planner descends."""
        return self(belief_trajectory, beta=beta, **kwargs)[0, origin, 0]

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Negation(self)


def _smooth_max(x, beta, dim, keepdim):
    """Normalized log-mean-exp: lies in [mean, max] and tends to max as beta grows."""
    return (torch.logsumexp(beta * x, dim=dim, keepdim=keepdim) - math.log(x.shape[dim])) / beta


class Minish(torch.nn.Module):
    """Exact min (beta is None) or normalized smooth min along dim."""

    def forward(self, x, beta=None, dim=1, keepdim=True):
        if beta is None:
            return x.min(dim=dim, keepdim=keepdim)[0]
        return -_smooth_max(-x, beta, dim, keepdim)


class Maxish(torch.nn.Module):
    """Exact max (beta is None) or normalized smooth max along dim."""

    def forward(self, x, beta=None, dim=1, keepdim=True):
        if beta is None:
            return x.max(dim=dim, keepdim=keepdim)[0]
        return _smooth_max(x, beta, dim, keepdim)


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
        # A trajectory that can score every step at once does so; otherwise, step by step.
        score_all = getattr(belief_trajectory, "probability_bounds", None)
        if score_all is not None:
            trace = score_all(self)
        else:
            trace = torch.stack([b.probability_bounds(self) for b in belief_trajectory], dim=1)
        if validate:
            check_probability_bounds(trace, self)
        return trace

    def __str__(self):
        return self.name if self.name is not None else type(self).__name__


# --- Boolean operators ----------------------------------------------------------


_MIN, _MAX = Minish(), Maxish()


def _align(*traces):
    """Truncate traces to their common number of origins."""
    n = min(t.shape[1] for t in traces)
    return tuple(t[:, :n] for t in traces)


def _pair(lower, upper):
    return torch.stack([lower, upper], dim=-1)


def _conjunction(trace1, trace2, beta=None):
    """[max(0, L1 + L2 - 1), min(U1, U2)]; both endpoints smooth when beta is set.

    The upper is smoothed as well as the lower because `Negation` swaps them: without it a
    negated conjunction -- every OutsideRectangle -- would carry a hard minimum in the very
    value the objective differentiates.
    """
    excess = trace1[..., 0] + trace2[..., 0] - 1.0
    lower = torch.clamp(excess, min=0.0) if beta is None else F.softplus(excess, beta=beta)
    upper = _MIN(_pair(trace1[..., 1], trace2[..., 1]), beta, dim=-1, keepdim=False)
    return _pair(lower, upper)


def _disjunction(trace1, trace2, beta=None):
    """[max(L1, L2), min(1, U1 + U2)]; both endpoints smooth when beta is set."""
    lower = _MAX(_pair(trace1[..., 0], trace2[..., 0]), beta, dim=-1, keepdim=False)
    total = trace1[..., 1] + trace2[..., 1]
    upper = _MIN(_pair(total, torch.ones_like(total)), beta, dim=-1, keepdim=False)
    return _pair(lower, upper)


def _negation(trace):
    """[1 - U, 1 - L]: exact in both modes, and it swaps which endpoint is the lower one."""
    return _pair(1.0 - trace[..., 1], 1.0 - trace[..., 0])


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

    def robustness_trace(self, belief_trajectory, beta=None, **kwargs):
        trace1 = self.subformula1(belief_trajectory, beta=beta, **kwargs)
        trace2 = self.subformula2(belief_trajectory, beta=beta, **kwargs)
        return type(self).combine(*_align(trace1, trace2), beta=beta)

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


def _checked_interval(interval):
    """A temporal window [a, b]: whole numbers with 0 <= a <= b. b may be infinite."""
    if interval is None:
        return None
    a, b = interval
    if b != np.inf and (not np.isfinite(b) or b != int(b)):
        raise ValueError(f"interval end must be a whole number or inf, got {list(interval)}")
    if not np.isfinite(a) or a != int(a) or a < 0:
        raise ValueError(f"interval must start at a whole number >= 0, got {list(interval)}")
    if a > b:
        raise ValueError(f"interval must satisfy a <= b, got {list(interval)}")
    return [int(a), np.inf if b == np.inf else int(b)]


class Temporal_Operator(STL_Formula):
    """Window reduction (min: Always, max: Eventually) run backwards; beta > 0 smooths."""

    symbol = None  # subclasses set self.operation to Minish() or Maxish()

    def __init__(self, subformula, interval=None):
        super().__init__()
        self.subformula = subformula
        self.interval = _checked_interval(interval)
        self._interval = [0, np.inf] if self.interval is None else self.interval

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

    def _rnn_cell(self, x, hc, beta=None):
        h0, _ = hc
        if self._suffix:  # [t, end of trace]
            output = self.operation(torch.cat([h0, x], dim=1), beta, dim=1, keepdim=True)
            state = (output, None)
        elif self._unbounded:  # [a, inf), a > 0
            d0, h0 = h0
            dh = torch.cat([d0, h0[:, :1, :]], dim=1)
            output = self.operation(dh, beta, dim=1, keepdim=True)
            state = ((output, self._apply_shift(h0, x)), None)
        else:  # [a, b]: slot rnn_dim-1-k holds the value k steps ahead
            a, b = int(self._interval[0]), int(self._interval[1])
            new_h0 = self._apply_shift(h0, x)
            output = self.operation(new_h0[:, : b - a + 1, :], beta, dim=1, keepdim=True)
            state = (new_h0, None)
        return output, state

    def _run_cell(self, x, beta):
        outputs, hc = [], self._initialize_rnn_cell(x)
        for x_i in torch.split(x, 1, dim=1):
            o, hc = self._rnn_cell(x_i, hc, beta)
            outputs.append(o)
        return torch.cat(outputs, dim=1)

    def robustness_trace(self, belief_trajectory, beta=None, keepdim=True, **kwargs):
        trace = self.subformula(belief_trajectory, beta=beta, keepdim=keepdim, **kwargs)
        if trace.shape[1] - self.lookahead <= 0:
            raise ValueError(
                f"{self}: one evaluation origin needs {self.lookahead + 1} steps, "
                f"child trace has {trace.shape[1]}"
            )
        # Run backwards so each origin sees its future; drop incomplete windows.
        output_reversed = self._run_cell(torch.flip(trace, dims=[1]), beta=beta)
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
        checked = _checked_interval(interval)
        self.interval = [0, np.inf] if checked is None else checked
        self._interval = self.interval
        self.min_op = Minish()
        self.max_op = Maxish()

    @property
    def lookahead(self):
        a, b = self._interval
        return int(a) if np.isinf(b) else int(b)

    def robustness_trace(self, belief_trajectory, beta=None, keepdim=True, **kwargs):
        phi = self.left(belief_trajectory, beta=beta, keepdim=True, **kwargs)
        psi = self.right(belief_trajectory, beta=beta, keepdim=True, **kwargs)
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
                prefix = self.min_op(phi[:, t : tau + 1, :], beta, dim=1, keepdim=False)
                candidates.append(_conjunction(prefix, psi[:, tau, :], beta))
            results.append(self.max_op(torch.stack(candidates, dim=1), beta, dim=1, keepdim=False))
        return torch.stack(results, dim=1)

    def __str__(self):
        return f"({self.left}) U_{self._interval} ({self.right})"
