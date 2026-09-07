import torch
import numpy as np

from pdstl.base import check_probability_bounds


class STL_Formula(torch.nn.Module):
    """
    Base class for Probabilistic STL formulas.
    """

    def __init__(self):
        super(STL_Formula, self).__init__()

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        """
        Compute the probability-bound trace for a belief trajectory.

        Args:
           belief_trajectory: BeliefTrajectory object
           scale: smoothing parameter (scale > 0 for smooth, <= 0 for direct).
              Smooth outputs are approximations and may leave [0,1] or cross;
              evaluate directly for reported intervals.
           keepdim: keep dimensions

        Returns:
           [B,N,2] where [..., 0] is the lower and [..., 1] the upper bound.

        Traces are indexed from origin 0 and contain only origins whose window
        is complete, so N shrinks by an operator's lookahead: N = T for an atom,
        T - b for a bounded [a,b] temporal operator, and the shortest child's
        length for a Boolean operator.
        """
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
    """Compute minimum (exact or smooth) over specified dimension"""

    def forward(self, x, scale, dim=1, keepdim=True):
        """
        The bounds dimension [..., 2] is automatically processed element-wise.
        """
        if scale > 0:
            return -torch.logsumexp(-x * scale, dim=dim, keepdim=keepdim) / scale
        else:
            return x.min(dim=dim, keepdim=keepdim)[0]


class Maxish(torch.nn.Module):
    """Compute maximum (exact or smooth) over specified dimension"""

    def forward(self, x, scale, dim=1, keepdim=True):
        """
        The bounds dimension [..., 2] is automatically processed element-wise.
        """
        if scale > 0:
            return torch.logsumexp(x * scale, dim=dim, keepdim=keepdim) / scale
        else:
            return x.max(dim=dim, keepdim=keepdim)[0]


class Predicate(STL_Formula):
    """
    Atomic predicate: names an event and assembles its trace over time.

    The predicate states the requirement; the belief at each step evaluates it
    under its own uncertainty model (see pdstl.base.Belief). Subclasses add
    whatever describes their event -- a comparison adds dim/threshold/sense --
    and never inspect the belief's internals.

    A bare Predicate("name") carries only an identity, which is what a provider
    of already-computed probability intervals keys on.
    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def robustness_trace(self, belief_trajectory, validate=True, **kwargs):
        """
        Args:
           belief_trajectory: BeliefTrajectory object
           validate: check the assembled bounds are well-formed. Pass False in
              an inner optimisation loop, where the host sync is not worth it.

        Returns:
           [B,T,2] probability bounds, [..., 0] lower and [..., 1] upper.
        """
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


def _event_name(sense, threshold, dim, weights):
    lhs = f"x[{dim}]" if weights is None else f"{list(weights)}·x"
    return f"{lhs} {sense} {threshold}"


class GreaterThan(Predicate):
    """
    Predicate: x[dim] >= threshold, or weights·x >= threshold if weights given.

    The belief evaluates the event; nothing is constructed here.
    """

    def __init__(self, threshold, dim=0, weights=None, name=None):
        if name is None:
            name = _event_name(">=", threshold, dim, weights)
        super().__init__(name=name)
        self.threshold = threshold
        self.dim = dim
        self.weights = weights
        self.sense = ">="


class LessThan(Predicate):
    """
    Predicate: x[dim] <= threshold, or weights·x <= threshold if weights given.
    """

    def __init__(self, threshold, dim=0, weights=None, name=None):
        if name is None:
            name = _event_name("<=", threshold, dim, weights)
        super().__init__(name=name)
        self.threshold = threshold
        self.dim = dim
        self.weights = weights
        self.sense = "<="


def _align(*traces):
    """Truncate origin-aligned traces to their common number of valid origins."""
    n = min(t.shape[1] for t in traces)
    return tuple(t[:, :n] for t in traces)


def _reduce(op, terms, scale):
    """Endpointwise Minish/Maxish over `terms`, honouring the scale convention."""
    return op(torch.stack(terms, dim=-1), scale, dim=-1, keepdim=False)


class Negation(STL_Formula):
    """
    Negation: ¬ϕ
    [L, U] -> [1 - U, 1 - L]
    """

    def __init__(self, subformula):
        super(Negation, self).__init__()
        self.subformula = subformula

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
    Fréchet bounds element-wise:
      lower = max(0, l1 + l2 - 1)
      upper = min(u1, u2)
    """

    def __init__(self, subformula1, subformula2):
        super(And, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.min_op = Minish()
        self.max_op = Maxish()

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

        # P(A ∩ B) ≥ max(0, P(A) + P(B) - 1) holds for any dependence. The
        # product bound would need independence, which sub-formulas over a
        # shared state do not have.
        lower = _reduce(self.max_op, [l1 + l2 - 1.0, torch.zeros_like(l1)], scale)
        upper = _reduce(self.min_op, [u1, u2], scale)
        return torch.stack([lower, upper], dim=-1)

    def __str__(self):
        return f"({self.subformula1}) ∧ ({self.subformula2})"


class Or(STL_Formula):
    """
    Disjunction: ϕ₁ ∨ ϕ₂
    Uses Frechet bounds element-wise:
      lower = max(l1, l2)
      upper = min(u1 + u2, 1)
    """

    def __init__(self, subformula1, subformula2):
        super(Or, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.min_op = Minish()
        self.max_op = Maxish()

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

        lower = _reduce(self.max_op, [l1, l2], scale)
        upper = _reduce(self.min_op, [u1 + u2, torch.ones_like(u1)], scale)
        return torch.stack([lower, upper], dim=-1)

    def __str__(self):
        return f"({self.subformula1}) ∨ ({self.subformula2})"


class Implies(STL_Formula):
    """
    Implication: ϕ₁ ⇒ ϕ₂
    Defined as: ¬ϕ₁ ∨ ϕ₂
    """

    def __init__(self, subformula1, subformula2):
        super(Implies, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.equivalent = Or(Negation(subformula1), subformula2)

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        return self.equivalent(
            belief_trajectory, scale=scale, keepdim=keepdim, **kwargs
        )

    def __str__(self):
        return f"({self.subformula1}) ⇒ ({self.subformula2})"


class Temporal_Operator(STL_Formula):
    """
    Base class for temporal operators.
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
    """
    □_I ϕ: Always operator
    Computes min over time interval.
    """

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
    """
    Eventually operator: ♢_I ϕ
    Computes max over time interval.
    The bounds dimension is processed automatically.
    """

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
    """
    ϕ U_I ψ : Until operator

    Inclusive convention: ϕ must hold from the evaluation origin through the
    witness time τ itself, not merely up to τ-1, so U_[0,0] combines both
    operands at the current time.
    """

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

                lower = _reduce(
                    self.max_op,
                    [prefix[..., 0] + psi_tau[..., 0] - 1.0, torch.zeros_like(prefix[..., 0])],
                    scale,
                )
                upper = _reduce(self.min_op, [prefix[..., 1], psi_tau[..., 1]], scale)
                candidates.append(torch.stack([lower, upper], dim=-1))  # [B,2]

            # best witness in [t+a, t+b]
            best = self.max_op(
                torch.stack(candidates, dim=1), scale, dim=1, keepdim=False
            )  # [B,2]
            results.append(best)

        return torch.stack(results, dim=1)  # [B,n_valid,2]

    def __str__(self):
        return f"({self.left}) U_{self._interval} ({self.right})"
