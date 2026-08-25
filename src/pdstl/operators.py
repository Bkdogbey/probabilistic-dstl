"""Bounded discrete-time STL formulas and hard probability semantics.

Operators combine probability enclosures with dependence-agnostic Frechet
bounds. Temporal intervals are finite and explicit; no smoothing or
independence assumptions are used.

Boolean combinations consume ``[B, 2]`` bounds; temporal combinations consume
windows of shape ``[B, n, 2]``.
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
    "Until",
    "frechet_intersection",
    "frechet_union",
]


def _clamp(bounds: torch.Tensor) -> torch.Tensor:
    """Clamp combination results against floating-point drift.

    The equations already enforce their mathematical extrema; this final clamp
    only prevents accumulated numerical error from leaving ``[0, 1]``.
    """
    return bounds.clamp(0.0, 1.0)


def frechet_intersection(stacked: torch.Tensor) -> torch.Tensor:
    """Dependence-agnostic bounds on ``P(E_1 and ... and E_n)``.

    ``lower = max(0, sum_i l_i - (n - 1))`` and ``upper = min_i u_i``.

    Both extrema are attained by some joint distribution with the given
    marginals, so the enclosure is the tightest one available without a
    dependence model.

    Parameters
    ----------
    stacked : torch.Tensor
        Shape ``[B, n, 2]``: the ``n`` enclosures being intersected.

    Returns
    -------
    torch.Tensor
        Shape ``[B, 2]``.
    """
    n = stacked.shape[-2]
    lower = torch.clamp(stacked[..., 0].sum(dim=-1) - (n - 1), min=0.0)
    upper = stacked[..., 1].amin(dim=-1)
    return _clamp(torch.stack([lower, upper], dim=-1))


def frechet_union(stacked: torch.Tensor) -> torch.Tensor:
    """Dependence-agnostic bounds on ``P(E_1 or ... or E_n)``.

    ``lower = max_i l_i`` and ``upper = min(1, sum_i u_i)``. The dual of
    :func:`frechet_intersection`, and equally tight.

    Parameters
    ----------
    stacked : torch.Tensor
        Shape ``[B, n, 2]``: the ``n`` enclosures being unioned.

    Returns
    -------
    torch.Tensor
        Shape ``[B, 2]``.
    """
    lower = stacked[..., 0].amax(dim=-1)
    upper = torch.clamp(stacked[..., 1].sum(dim=-1), max=1.0)
    return _clamp(torch.stack([lower, upper], dim=-1))


def _validate_interval(interval: Sequence[int]) -> tuple[int, int]:
    """Validate a finite discrete interval ``[a, b]`` with ``0 <= a <= b``.

    Shared by :class:`TemporalOperator` and :class:`Until`, which both carry a
    bounded window but differ in arity.
    """
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

    return a, b


class STLFormula(ABC):
    """Base class for bounded-time pdSTL formulas."""

    @abstractmethod
    def horizon(self) -> int:
        """Required lookahead ``H(phi)`` in discrete steps.

        ``H(mu) = 0`` and ``H(~phi) = H(phi)``. Boolean operators take the
        maximum child horizon; ``G_[a,b]`` and ``F_[a,b]`` add ``b`` to the
        child horizon; :class:`Until` reads its two children over different
        offsets and so has its own rule.

        Evaluation at ``t`` reads source data through ``t + H(phi)``.
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

    ``Predicate(h=callable, name=...)``
        A state-resolved predicate available to model-based sources.

    ``Predicate(h=None, name="...")``
        A symbolic predicate whose probabilities are supplied directly.

    The evaluator never calls ``h``; deriving atomic probabilities belongs to
    the source. Event identity follows object identity through a stable ``uid``:
    reusing one predicate means the same event, while separately constructed
    predicates remain distinct regardless of callable or name. The name is
    presentation only; the counter-backed ``uid`` avoids recycled ``id()``
    values.
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
    """Negation ``~mu``, restricted to atoms by NNF semantics.

    Keeping negation at atoms also makes exact complements structurally
    recognizable during evaluation.
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
        """Return ``[1 - u, 1 - l]`` from bounds ``[l, u]`` of shape ``[B, 2]``."""
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
        """The two-operand case of :func:`frechet_intersection`.

        Shapes ``[B, 2]`` in, ``[B, 2]`` out. Symmetric in its arguments, which
        is what lets propagation canonicalise the operand order.
        """
        return frechet_intersection(torch.stack([value1, value2], dim=-2))

    def __str__(self) -> str:
        return f"({self.left}) ∧ ({self.right})"


class Or(_BinaryOperator):
    """Disjunction ``phi1 or phi2`` under dependence-agnostic Frechet bounds."""

    @property
    def tag(self) -> Hashable:
        return ("or",)

    def combine(self, value1: torch.Tensor, value2: torch.Tensor) -> torch.Tensor:
        """The two-operand case of :func:`frechet_union`.

        Shapes ``[B, 2]`` in, ``[B, 2]`` out. Symmetric in its arguments.
        """
        return frechet_union(torch.stack([value1, value2], dim=-2))

    def __str__(self) -> str:
        return f"({self.left}) ∨ ({self.right})"


class TemporalOperator(STLFormula):
    """Base class for finite intervals ``[a, b]`` with ``0 <= a <= b``.

    Endpoints must be integral; unbounded time is not representable.
    """

    def __init__(self, subformula: STLFormula, interval: Sequence[int]) -> None:
        if not isinstance(subformula, STLFormula):
            raise TypeError(
                f"subformula must be an STLFormula, got {type(subformula).__name__}"
            )

        self.subformula = subformula
        self.interval: tuple[int, int] = _validate_interval(interval)

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
        """Intersect the window via :func:`frechet_intersection`.

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
        return frechet_intersection(stacked)

    def __str__(self) -> str:
        return f"□[{self.a},{self.b}]({self.subformula})"


class Eventually(TemporalOperator):
    """``F_[a,b] phi``: the subformula holds at some time in the window."""

    @property
    def tag(self) -> Hashable:
        return ("eventually", self.a, self.b)

    def combine(self, stacked: torch.Tensor) -> torch.Tensor:
        """Union the window via :func:`frechet_union`.

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
        return frechet_union(stacked)

    def __str__(self) -> str:
        return f"♢[{self.a},{self.b}]({self.subformula})"


class Until(_BinaryOperator):
    """``phi1 U_[a,b] phi2``: bounded strong until.

    Strong until requires ``phi2`` to actually occur somewhere in ``[a, b]``,
    with ``phi1`` holding at every step strictly before it. The satisfaction
    event at time ``k`` is::

        E_U = union_{j=a..b} C_j,
        C_j = E_{phi2, k+j} intersect (intersect_{r=0..j-1} E_{phi1, k+r})

    The ``phi1`` prefix always starts at ``r = 0``, not at ``r = a``: the
    formula is not satisfied by a run that violates ``phi1`` before the window
    opens. For ``j = 0`` the prefix is empty, so ``C_0 = E_{phi2, k}`` and
    ``phi1 U_[0,0] phi2`` is exactly ``phi2``.

    This is a two-argument temporal operator, so it shares the operand
    validation of the Boolean binaries rather than the single-``subformula``
    shape of :class:`TemporalOperator`.
    """

    def __init__(
        self, left: STLFormula, right: STLFormula, interval: Sequence[int]
    ) -> None:
        super().__init__(left, right)
        self.interval: tuple[int, int] = _validate_interval(interval)

    @property
    def a(self) -> int:
        return self.interval[0]

    @property
    def b(self) -> int:
        return self.interval[1]

    def horizon(self) -> int:
        """The actual lookahead required, which is not ``b + max(H1, H2)``.

        ``phi2`` is read at offsets up to ``b`` and ``phi1`` only at offsets up
        to ``b - 1``, so::

            H = H(phi2)                                  if b == 0
            H = max(b + H(phi2), b - 1 + H(phi1))        if b > 0

        When ``b == 0`` the prefix is empty and ``phi1`` is never read at all,
        so its own lookahead does not enter.
        """
        if self.b == 0:
            return self.right.horizon()
        return max(
            self.b + self.right.horizon(),
            self.b - 1 + self.left.horizon(),
        )

    @property
    def tag(self) -> Hashable:
        return ("until", self.a, self.b)

    def __str__(self) -> str:
        return f"({self.left}) U[{self.a},{self.b}] ({self.right})"
