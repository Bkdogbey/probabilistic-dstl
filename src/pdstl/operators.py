import torch
import numpy as np

from pdstl.base import check_probability_bounds


class STL_Formula(torch.nn.Module):
    """Base formula for pointwise probabilities and temporal robustness."""

    def __init__(self):
        super(STL_Formula, self).__init__()

    @property
    def is_pointwise(self) -> bool:
        """Return whether the formula produces pointwise probabilities."""
        return False

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        """Return ``[B,N,2]`` endpoints at complete evaluation origins."""
        raise NotImplementedError("robustness_trace not yet implemented")

    def forward(self, belief_trajectory, **kwargs):
        """Forward pass delegates to robustness_trace"""
        return self.robustness_trace(belief_trajectory, **kwargs)

    def __str__(self):
        raise NotImplementedError("__str__ not yet implemented")

    def __and__(self, other):
        """Overload & operator for And"""
        return And(self, other)

    def __or__(self, other):
        """Overload | operator for Or"""
        return Or(self, other)

    def __invert__(self):
        """Overload ~ operator for Negation"""
        return Negation(self)


class Minish(torch.nn.Module):
    """Compute an exact or smooth temporal minimum."""

    def forward(self, x, scale, dim=1, keepdim=True):
        """Reduce ``x`` along ``dim`` without mixing interval endpoints."""
        if scale > 0:
            return -torch.logsumexp(-x * scale, dim=dim, keepdim=keepdim) / scale
        else:
            return x.min(dim=dim, keepdim=keepdim)[0]


class Maxish(torch.nn.Module):
    """Compute an exact or smooth temporal maximum."""

    def forward(self, x, scale, dim=1, keepdim=True):
        """Reduce ``x`` along ``dim`` without mixing interval endpoints."""
        if scale > 0:
            return torch.logsumexp(x * scale, dim=dim, keepdim=keepdim) / scale
        else:
            return x.max(dim=dim, keepdim=keepdim)[0]


class Predicate(STL_Formula):
    """Atomic event evaluated independently by each belief."""

    def __init__(self, name=None):
        super().__init__()
        self.name = name

    @property
    def is_pointwise(self):
        return True

    def robustness_trace(self, belief_trajectory, validate=True, **kwargs):
        """Return the pointwise probability interval trace ``[B,T,2]``."""
        bounds = [
            belief_trajectory[t].probability_bounds(self)  # each [B,2]
            for t in range(len(belief_trajectory))
        ]
        trace = torch.stack(bounds, dim=1)  # [B,T,2]

        if validate:
            check_probability_bounds(trace, self)

        return trace

    def __str__(self):
        return self.name if self.name is not None else type(self).__name__


def _event_name(sense, threshold, dim):
    return f"x[{dim}] {sense} {threshold}"


class GreaterThan(Predicate):
    """Predicate ``x[dim] >= threshold``."""

    def __init__(self, threshold, dim=0, name=None):
        if name is None:
            name = _event_name(">=", threshold, dim)
        super().__init__(name=name)
        self.threshold = threshold
        self.dim = dim
        self.sense = ">="


class LessThan(Predicate):
    """Predicate ``x[dim] <= threshold``."""

    def __init__(self, threshold, dim=0, name=None):
        if name is None:
            name = _event_name("<=", threshold, dim)
        super().__init__(name=name)
        self.threshold = threshold
        self.dim = dim
        self.sense = "<="


def _align(*traces):
    """Truncate origin-aligned traces to their common number of valid origins."""
    n = min(t.shape[1] for t in traces)
    return tuple(t[:, :n] for t in traces)


def _require_pointwise(operator, *subformulas):
    """Reject Boolean composition of temporal robustness intervals."""
    for subformula in subformulas:
        if not subformula.is_pointwise:
            raise ValueError(
                f"{operator} only accepts pointwise event formulas; "
                f"got temporal formula {subformula}"
            )


class Negation(STL_Formula):
    """
    Negation: ¬ϕ
    [L, U] -> [1 - U, 1 - L]

    Exact complement of a pointwise probability interval; never smoothed.
    """

    def __init__(self, subformula):
        super(Negation, self).__init__()
        _require_pointwise("Negation", subformula)
        self.subformula = subformula

    @property
    def is_pointwise(self) -> bool:
        return True

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        trace = self.subformula(
            belief_trajectory, scale=scale, keepdim=keepdim, **kwargs
        )
        # trace: [B,T,2]
        # [lower, upper] -> [1 - upper, 1 - lower]
        lower = 1.0 - trace[..., 1]
        upper = 1.0 - trace[..., 0]
        return torch.stack([lower, upper], dim=-1)

    def __str__(self):
        return f"¬({self.subformula})"


class And(STL_Formula):
    """
    Conjunction: ϕ₁ ∧ ϕ₂

    Exact and never smoothed; `scale` is only forwarded to the sub-formulas.

    Both sub-formulas must be pointwise. Exact Fréchet probability bounds:
      lower = max(0, l1 + l2 - 1)
      upper = min(u1, u2)
    """

    def __init__(self, subformula1, subformula2):
        super(And, self).__init__()
        _require_pointwise("And", subformula1, subformula2)
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    @property
    def is_pointwise(self) -> bool:
        return True

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        trace1 = self.subformula1(
            belief_trajectory, scale=scale, keepdim=keepdim, **kwargs
        )
        trace2 = self.subformula2(
            belief_trajectory, scale=scale, keepdim=keepdim, **kwargs
        )
        trace1, trace2 = _align(trace1, trace2)
        l1, u1 = trace1[..., 0], trace1[..., 1]
        l2, u2 = trace2[..., 0], trace2[..., 1]

        # P(A ∩ B) ≥ max(0, P(A) + P(B) - 1) for any dependence.
        lower = torch.clamp(l1 + l2 - 1.0, min=0.0)
        upper = torch.minimum(u1, u2)
        return torch.stack([lower, upper], dim=-1)

    def __str__(self):
        return f"({self.subformula1}) ∧ ({self.subformula2})"


class Or(STL_Formula):
    """
    Disjunction: ϕ₁ ∨ ϕ₂

    Exact and never smoothed; `scale` is only forwarded to the sub-formulas.

    Both sub-formulas must be pointwise. Exact Fréchet/Boole bounds:
      lower = max(l1, l2)
      upper = min(1, u1 + u2)
    """

    def __init__(self, subformula1, subformula2):
        super(Or, self).__init__()
        _require_pointwise("Or", subformula1, subformula2)
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    @property
    def is_pointwise(self) -> bool:
        return True

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        trace1 = self.subformula1(
            belief_trajectory, scale=scale, keepdim=keepdim, **kwargs
        )
        trace2 = self.subformula2(
            belief_trajectory, scale=scale, keepdim=keepdim, **kwargs
        )
        trace1, trace2 = _align(trace1, trace2)
        l1, u1 = trace1[..., 0], trace1[..., 1]
        l2, u2 = trace2[..., 0], trace2[..., 1]

        lower = torch.maximum(l1, l2)
        upper = torch.clamp(u1 + u2, max=1.0)
        return torch.stack([lower, upper], dim=-1)

    def __str__(self):
        return f"({self.subformula1}) ∨ ({self.subformula2})"


class Implies(STL_Formula):
    """
    Implication: ϕ₁ ⇒ ϕ₂
    Defined as: ¬ϕ₁ ∨ ϕ₂, so it inherits Or's exact, unsmoothed bounds.
    """

    def __init__(self, subformula1, subformula2):
        super(Implies, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.equivalent = Or(Negation(subformula1), subformula2)

    @property
    def is_pointwise(self) -> bool:
        return True

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        return self.equivalent(
            belief_trajectory, scale=scale, keepdim=keepdim, **kwargs
        )

    def __str__(self):
        return f"({self.subformula1}) ⇒ ({self.subformula2})"


class Temporal_Operator(STL_Formula):
    """Base endpointwise temporal reduction.

    Positive scales produce optimization surrogates, not reportable intervals.
    """

    def __init__(self, subformula, interval=None):
        super(Temporal_Operator, self).__init__()
        self.subformula = subformula
        self.interval = interval
        self._interval = [0, np.inf] if self.interval is None else self.interval

        a, b = self._interval
        self._unbounded = np.isinf(b)
        # An unbounded interval starting at 0 is the same window as no interval
        # at all: everything from the origin to the end of the available trace.
        self._suffix = self.interval is None or (self._unbounded and int(a) == 0)

        if self._suffix:
            self.rnn_dim = 1
        elif self._unbounded:
            self.rnn_dim = int(a)
        else:
            self.rnn_dim = int(b) + 1

        # Operation set by subclass (Minish or Maxish)
        self.operation = None

        # Shift register: M drops the oldest slot, b writes the newest one.
        self.register_buffer(
            "M", torch.tensor(np.diag(np.ones(self.rnn_dim - 1), k=1)).float()
        )
        b_vec = torch.zeros(self.rnn_dim, 1)
        b_vec[-1] = 1.0
        self.register_buffer("b", b_vec)

    @property
    def lookahead(self):
        """Steps of future beyond the origin that one evaluation needs.

        A suffix window needs none: it covers whatever trace remains.
        """
        a, b = self._interval
        if self._suffix:
            return 0
        return int(a) if self._unbounded else int(b)

    def _initialize_rnn_cell(self, x):
        """
        Initialize hidden state.
        x: [B,T,2] reversed trace
        Returns: (h0, count)

        The register is pre-filled with the first reversed value. Every output
        it reaches is dropped by robustness_trace, so the fill never appears in
        a returned trace.
        """
        h0 = x[:, :1, :].expand(-1, self.rnn_dim, -1).clone()  # [B,rnn_dim,2]
        count = 0.0

        if self._unbounded and not self._suffix:  # [a, inf), a > 0
            d0 = x[:, :1, :]
            return ((d0, h0), count)

        return (h0, count)

    def _apply_shift(self, h0, x):
        """
        Apply M @ h0 + b * x
        h0: [B,rnn_dim,2]
        x: [B,1,2]
        """
        batch, rnn_dim, bounds = h0.shape

        # Runtime state follows the input, not the buffer's stored dtype/device.
        M = self.M.to(dtype=h0.dtype, device=h0.device)
        b = self.b.to(dtype=h0.dtype, device=h0.device)

        # Treat (batch, bounds) as batch dimension for matmul
        h0_reshaped = h0.permute(0, 2, 1)  # [B,2,rnn_dim]

        h0_flat = h0_reshaped.reshape(-1, rnn_dim)  # [B*2,rnn_dim]

        # Shift
        shifted_flat = torch.matmul(h0_flat, M.t())  # [B*2,rnn_dim]
        shifted = shifted_flat.reshape(batch, bounds, rnn_dim)
        shifted = shifted.permute(0, 2, 1)  # [B,rnn_dim,2]

        # Add new value into last position
        b_broadcast = b.view(1, -1, 1)  # [1,rnn_dim,1]
        x_broadcast = x.squeeze(1).unsqueeze(1)  # [B,1,2]

        return shifted + b_broadcast * x_broadcast

    def _rnn_cell(self, x, hc, scale=-1, **kwargs):
        """Must be implemented by subclass"""
        raise NotImplementedError

    def _run_cell(self, x, scale):
        """Run RNN through entire trace"""
        outputs = []
        hc = self._initialize_rnn_cell(x)
        xs = torch.split(x, 1, dim=1)  # list of [B,1,2]

        for xs_i in xs:
            o, hc = self._rnn_cell(xs_i, hc, scale)
            outputs.append(o)

        return torch.cat(outputs, dim=1)  # [B,T,2]

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        trace = self.subformula(
            belief_trajectory, scale=scale, keepdim=keepdim, **kwargs
        )
        n_valid = trace.shape[1] - self.lookahead
        if n_valid <= 0:
            raise ValueError(
                f"{self}: one evaluation origin needs {self.lookahead + 1} steps, "
                f"child trace has {trace.shape[1]}"
            )

        # Evaluate backwards so each origin sees its own future, then drop the
        # origins whose window runs past the end of the trace.
        output_reversed = self._run_cell(torch.flip(trace, dims=[1]), scale=scale)
        return torch.flip(output_reversed[:, self.lookahead :], dims=[1])


class Always(Temporal_Operator):
    """Endpointwise temporal minimum over a future window."""

    def __init__(self, subformula, interval=None):
        super(Always, self).__init__(subformula=subformula, interval=interval)
        self.operation = Minish()
        self.oper = "min"

    def _rnn_cell(self, x, hc, scale=-1, **kwargs):
        """
        Compute running minimum.
        """
        h0, c = hc

        if self.operation is None:
            raise Exception("Operation not initialized")

        # CASE 1: suffix window [t, end of trace]
        if self._suffix:
            # h0: [B,rnn_dim,2], x: [B,1,2]
            input_ = torch.cat([h0, x], dim=1)  # [B,rnn_dim+1,2]
            output = self.operation(input_, scale, dim=1, keepdim=True)  # [B,1,2]
            state = (output, None)

        # CASE 2: unbounded future [a, inf), a > 0
        elif self._unbounded:
            d0, h0 = h0  # unpack tuple state
            dh = torch.cat([d0, h0[:, :1, :]], dim=1)  # [B,2,2]
            output = self.operation(dh, scale, dim=1, keepdim=True)
            new_h0 = self._apply_shift(h0, x)
            state = ((output, new_h0), None)

        # CASE 3: bounded interval [a,b]
        else:
            a, b = int(self._interval[0]), int(self._interval[1])
            new_h0 = self._apply_shift(h0, x)
            # Slot rnn_dim-1-k holds the value k steps ahead, so offsets a..b
            # are the leading b-a+1 slots.
            window = new_h0[:, : b - a + 1, :]
            output = self.operation(window, scale, dim=1, keepdim=True)
            state = (new_h0, None)

        return output, state

    def __str__(self):
        if self.interval is None:
            return f"□({self.subformula})"
        return f"□_{self._interval}({self.subformula})"


class Eventually(Temporal_Operator):
    """Endpointwise temporal maximum over a future window."""

    def __init__(self, subformula, interval=None):
        super(Eventually, self).__init__(subformula=subformula, interval=interval)
        self.operation = Maxish()
        self.oper = "max"

    def _rnn_cell(self, x, hc, scale=-1, **kwargs):
        """
        Compute running maximum.
        """
        h0, c = hc

        if self.operation is None:
            raise Exception("Operation not initialized")

        # Case 1: suffix window [t, end of trace]
        if self._suffix:
            input_ = torch.cat([h0, x], dim=1)
            output = self.operation(input_, scale, dim=1, keepdim=True)
            state = (output, None)

        # Case 2: unbounded future [a, inf), a > 0
        elif self._unbounded:
            d0, h0 = h0
            dh = torch.cat([d0, h0[:, :1, :]], dim=1)
            output = self.operation(dh, scale, dim=1, keepdim=True)
            new_h0 = self._apply_shift(h0, x)
            state = ((output, new_h0), None)

        # Case 3: bounded interval [a,b]
        else:
            a, b = int(self._interval[0]), int(self._interval[1])
            new_h0 = self._apply_shift(h0, x)
            window = new_h0[:, : b - a + 1, :]
            output = self.operation(window, scale, dim=1, keepdim=True)
            state = (new_h0, None)

        return output, state

    def __str__(self):
        if self.interval is None:
            return f"♢({self.subformula})"
        return f"♢_{self._interval}({self.subformula})"


class Until(STL_Formula):
    """Endpointwise max-min Until with an inclusive left prefix."""

    def __init__(self, left, right, interval=None):
        super(Until, self).__init__()
        self.left = left
        self.right = right
        self.interval = [0, np.inf] if interval is None else interval
        self._interval = self.interval

        self.min_op = Minish()
        self.max_op = Maxish()

    @property
    def lookahead(self):
        """Steps of future beyond the origin that one evaluation needs."""
        a, b = self._interval
        return int(a) if np.isinf(b) else int(b)

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        # ϕ and ψ traces: [B,T,2]
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

        results = []  # list of [B,2], one per origin
        for t in range(n_valid):
            candidates = []
            for tau in range(t + a, min(t + b, T - 1) + 1):
                # Inclusive convention: ϕ must hold from the origin t through
                # the witness τ itself, so the prefix slice ends at τ+1.
                prefix = self.min_op(
                    phi[:, t : tau + 1, :], scale, dim=1, keepdim=False
                )  # [B,2]
                psi_tau = psi[:, tau, :]  # [B,2]

                # Both must hold at this witness: endpointwise min, the same
                # reduction Always takes over a window.
                candidates.append(
                    self.min_op(
                        torch.stack([prefix, psi_tau], dim=1),
                        scale,
                        dim=1,
                        keepdim=False,
                    )
                )  # [B,2]

            # best witness in [t+a, t+b]
            best = self.max_op(
                torch.stack(candidates, dim=1), scale, dim=1, keepdim=False
            )  # [B,2]
            results.append(best)

        return torch.stack(results, dim=1)  # [B,n_valid,2]

    def __str__(self):
        return f"({self.left}) U_{self._interval} ({self.right})"
