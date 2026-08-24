"""STL formula structure and the hard probability combination equations.

This module defines *what each STL operation means mathematically*. It does not
know where probabilities come from (``base.py``) and it does not traverse
children or index time (``propagate.py``). Each operator exposes only a small
``combine`` step, so every equation appears exactly once in the codebase.

Scope
-----
Finite, bounded, discrete-time STL only. Temporal operators require an explicit
interval ``[a, b]`` with ``0 <= a <= b``. There is no ``interval=None``, no
``np.inf``, and no unbounded operator: the code matches the theory exactly.

Only the **hard** probability semantics live here. No smoothing, no
log-sum-exp, no differentiable surrogate.

Semantics
---------
Given child enclosures ``[l, u]``, all combinations are dependence-agnostic
Frechet bounds::

    ~A              lower = 1 - u
                    upper = 1 - l

    A and B         lower = max(0, l1 + l2 - 1)
                    upper = min(u1, u2)

    A or B          lower = max(l1, l2)
                    upper = min(1, u1 + u2)

    G_[a,b] A       lower = max(0, sum_j l_j - (n - 1))     n = b - a + 1
                    upper = min_j u_j

    F_[a,b] A       lower = max_j l_j
                    upper = min(1, sum_j u_j)

No independence is assumed anywhere, in particular not across time.

Formula objects are plain classes, not ``torch.nn.Module``. Autograd operates on
tensor operations regardless of whether the object performing them is a Module,
so a future differentiable semantics does not require this to change. A formula
tree is a syntax tree, not a neural network; only genuinely learnable semantic
parameters would justify revisiting that.
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Sequence

import torch

__all__ = [
    "Always",
    "And",
    "Eventually",
    "Negation",
    "Or",
    "Predicate",
    "STLFormula",
    "TemporalOperator",
]


def _clamp(bounds: torch.Tensor) -> torch.Tensor:
    """Guard a combination result against float drift outside ``[0, 1]``.

    Each equation below already applies its own ``max(0, .)`` / ``min(1, .)``
    where the mathematics calls for one, and each already preserves
    ``lower <= upper`` (for ``And``: ``l1 + l2 - 1 <= u1 + u2 - 1 <=
    min(u1, u2)`` because ``u1, u2 <= 1``). This final clamp exists only so
    accumulated float error cannot produce a technically invalid interval,
    which is what lets ``validate_bounds`` stay strict at the source boundary.
    """
    return bounds.clamp(0.0, 1.0)


class STLFormula(ABC):
    """Base class for bounded-time pdSTL formulas."""

    @abstractmethod
    def horizon(self) -> int:
        """Required lookahead ``H(phi)`` in discrete steps.

        ``H(mu) = 0``; ``H(~phi) = H(phi)``;
        ``H(phi1 and phi2) = H(phi1 or phi2) = max(H(phi1), H(phi2))``;
        ``H(G_[a,b] phi) = H(F_[a,b] phi) = b + H(phi)``.

        A formula evaluated at time ``t`` reads source data over
        ``t ... t + H(phi)``.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def tag(self) -> Hashable:
        """Small hashable structural label used to build event identity keys."""
        raise NotImplementedError

    def __and__(self, other: STLFormula) -> And:
        return And(self, other)

    def __or__(self, other: STLFormula) -> Or:
        return Or(self, other)

    def __invert__(self) -> Negation:
        return Negation(self)

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self})"


class Predicate(STLFormula):
    """The atomic object ``mu_i : h_i(x) >= 0``.

    Two usage modes, both valid:

    ``Predicate(h=callable, name=...)``
        *State-resolved predicate.* ``h`` is the real ``h_i``, and a future
        model-based source (Gaussian, sampling, learned) evaluates it against a
        state representation to derive ``P(E_{i,k})``.

    ``Predicate(h=None, name="...")``
        *Symbolic predicate.* The user supplies event probabilities directly
        with no state model behind them. ``h_i`` exists mathematically but is
        not available to the code. This is the mode ``TableProbabilitySource``
        uses.

    Either way the hard core never calls ``h``: translating a state model into
    ``P(E_{i,k})`` is the source's job, which is exactly the separation this
    package is built around.

    Example
    -------
    >>> mu = Predicate(lambda x: x[..., 0] - 5.0, name="x_ge_5")

    Identity
    --------
    Each predicate carries a stable ``uid`` from a module-level counter, and
    uses default identity equality. So:

    - repeated use of the **same object** is the same atomic event;
    - two separately constructed predicates are **never** the same event, even
      with an identical callable, threshold, or name;
    - ``name`` is human-readable presentation only and carries no mathematical
      meaning.

    A counter is used rather than ``id()``, which can be recycled by the
    allocator and is not reproducible across runs.
    """

    _uid_counter = itertools.count()

    def __init__(self, h: Callable | None = None, name: str | None = None) -> None:
        if h is not None and not callable(h):
            raise TypeError(f"h must be callable or None, got {type(h).__name__}")
        self.h = h
        self.name = name
        self.uid: int = next(Predicate._uid_counter)

    def horizon(self) -> int:
        return 0

    @property
    def tag(self) -> Hashable:
        return ("atom", self.uid)

    def __str__(self) -> str:
        return self.name if self.name is not None else f"mu_{self.uid}"


class Negation(STLFormula):
    """Negation ``~mu``, restricted to atomic predicates.

    The target theory works in negation normal form, so this class deliberately
    refuses to define arbitrary formula negation rather than silently inventing
    a semantics for it. The restriction also makes the complement identity
    ``A and ~A = empty`` / ``A or ~A = universe`` exactly decidable during
    propagation.
    """

    def __init__(self, subformula: STLFormula) -> None:
        if not isinstance(subformula, Predicate):
            raise TypeError(
                "negation is only defined for atomic predicates in this "
                f"implementation, got {type(subformula).__name__}; the target "
                "semantics is negation normal form, so push negations down to "
                "the predicates instead of negating a compound formula"
            )
        self.subformula = subformula

    def horizon(self) -> int:
        return self.subformula.horizon()

    @property
    def tag(self) -> Hashable:
        return ("not",)

    def combine(self, value: torch.Tensor) -> torch.Tensor:
        """``[l, u] -> [1 - u, 1 - l]``. Shape ``[B, 2]`` in, ``[B, 2]`` out."""
        lower = 1.0 - value[..., 1]
        upper = 1.0 - value[..., 0]
        return _clamp(torch.stack([lower, upper], dim=-1))

    def __str__(self) -> str:
        return f"¬{self.subformula}"


class _BinaryOperator(STLFormula):
    """Shared structure for the two-argument Boolean operators."""

    def __init__(self, left: STLFormula, right: STLFormula) -> None:
        for side, operand in (("left", left), ("right", right)):
            if not isinstance(operand, STLFormula):
                raise TypeError(
                    f"{side} operand must be an STLFormula, got "
                    f"{type(operand).__name__}"
                )
        self.left = left
        self.right = right

    def horizon(self) -> int:
        return max(self.left.horizon(), self.right.horizon())


class And(_BinaryOperator):
    """Conjunction ``phi1 and phi2`` under dependence-agnostic Frechet bounds."""

    @property
    def tag(self) -> Hashable:
        return ("and",)

    def combine(self, value1: torch.Tensor, value2: torch.Tensor) -> torch.Tensor:
        """``lower = max(0, l1 + l2 - 1)``, ``upper = min(u1, u2)``.

        Shapes ``[B, 2]`` in, ``[B, 2]`` out. Symmetric in its arguments, which
        is what lets propagation canonicalise the operand order.
        """
        l1, u1 = value1[..., 0], value1[..., 1]
        l2, u2 = value2[..., 0], value2[..., 1]
        lower = torch.clamp(l1 + l2 - 1.0, min=0.0)
        upper = torch.minimum(u1, u2)
        return _clamp(torch.stack([lower, upper], dim=-1))

    def __str__(self) -> str:
        return f"({self.left}) ∧ ({self.right})"


class Or(_BinaryOperator):
    """Disjunction ``phi1 or phi2`` under dependence-agnostic Frechet bounds."""

    @property
    def tag(self) -> Hashable:
        return ("or",)

    def combine(self, value1: torch.Tensor, value2: torch.Tensor) -> torch.Tensor:
        """``lower = max(l1, l2)``, ``upper = min(1, u1 + u2)``.

        Shapes ``[B, 2]`` in, ``[B, 2]`` out. Symmetric in its arguments.
        """
        l1, u1 = value1[..., 0], value1[..., 1]
        l2, u2 = value2[..., 0], value2[..., 1]
        lower = torch.maximum(l1, l2)
        upper = torch.clamp(u1 + u2, max=1.0)
        return _clamp(torch.stack([lower, upper], dim=-1))

    def __str__(self) -> str:
        return f"({self.left}) ∨ ({self.right})"


class TemporalOperator(STLFormula):
    """Base class for the bounded temporal operators.

    ``interval`` is required and must satisfy ``0 <= a <= b`` with integral
    endpoints. Unbounded time is not representable by design.
    """

    def __init__(self, subformula: STLFormula, interval: Sequence[int]) -> None:
        if not isinstance(subformula, STLFormula):
            raise TypeError(
                f"subformula must be an STLFormula, got {type(subformula).__name__}"
            )

        try:
            endpoints = tuple(interval)
        except TypeError:
            raise TypeError(
                f"interval must be a sequence [a, b], got {interval!r}; "
                f"this branch implements bounded-time STL only, so there is no "
                f"unbounded default"
            ) from None

        if len(endpoints) != 2:
            raise ValueError(
                f"interval must have exactly 2 endpoints [a, b], got {endpoints!r}"
            )

        a, b = endpoints
        for label, value in (("a", a), ("b", b)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"interval endpoint {label} must be a number, got {value!r}"
                )
            if not math.isfinite(value):
                raise ValueError(
                    f"interval endpoint {label} must be finite; this branch "
                    f"implements bounded-time STL only, so infinity is not a "
                    f"valid endpoint, got {value!r}"
                )
            if float(value) != int(value):
                raise ValueError(
                    f"interval endpoint {label} must be integral (bounded "
                    f"discrete time), got {value!r}"
                )
        a, b = int(a), int(b)

        if a < 0:
            raise ValueError(f"interval must satisfy 0 <= a <= b, got a={a}")
        if a > b:
            raise ValueError(f"interval must satisfy 0 <= a <= b, got [{a}, {b}]")

        self.subformula = subformula
        self.interval: tuple[int, int] = (a, b)

    @property
    def a(self) -> int:
        return self.interval[0]

    @property
    def b(self) -> int:
        return self.interval[1]

    def horizon(self) -> int:
        return self.b + self.subformula.horizon()


class Always(TemporalOperator):
    """``G_[a,b] phi``: the subformula holds at every time in the window."""

    @property
    def tag(self) -> Hashable:
        return ("always", self.a, self.b)

    def combine(self, stacked: torch.Tensor) -> torch.Tensor:
        """``lower = max(0, sum_j l_j - (n - 1))``, ``upper = min_j u_j``.

        Parameters
        ----------
        stacked : torch.Tensor
            Shape ``[B, n, 2]``: the child enclosures at the ``n = b - a + 1``
            times in the window.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 2]``. No temporal independence is assumed.
        """
        n = stacked.shape[-2]
        lower = torch.clamp(stacked[..., 0].sum(dim=-1) - (n - 1), min=0.0)
        upper = stacked[..., 1].amin(dim=-1)
        return _clamp(torch.stack([lower, upper], dim=-1))

    def __str__(self) -> str:
        return f"□[{self.a},{self.b}]({self.subformula})"


class Eventually(TemporalOperator):
    """``F_[a,b] phi``: the subformula holds at some time in the window."""

    @property
    def tag(self) -> Hashable:
        return ("eventually", self.a, self.b)

    def combine(self, stacked: torch.Tensor) -> torch.Tensor:
        """``lower = max_j l_j``, ``upper = min(1, sum_j u_j)``.

        Parameters
        ----------
        stacked : torch.Tensor
            Shape ``[B, n, 2]``: the child enclosures at the ``n = b - a + 1``
            times in the window.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 2]``. No temporal independence is assumed.
        """
        lower = stacked[..., 0].amax(dim=-1)
        upper = torch.clamp(stacked[..., 1].sum(dim=-1), max=1.0)
        return _clamp(torch.stack([lower, upper], dim=-1))

    def __str__(self) -> str:
        return f"♢[{self.a},{self.b}]({self.subformula})"
