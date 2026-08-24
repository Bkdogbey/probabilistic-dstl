"""User input contract for pdSTL.

This module defines *what probabilistic information a user must supply*. It says
nothing about STL syntax (``operators.py``) and nothing about how a formula is
traversed (``propagate.py``).

The pdSTL core is deliberately agnostic about where atomic probabilities come
from. A source may wrap exact probabilities, a Gaussian belief, samples,
statistical confidence intervals, learned predictions, or numbers a human typed
in. None of that reaches the semantics.

Bound representation
--------------------
Probability bounds are plain tensors with trailing dimension 2::

    bounds[..., 0] = lower
    bounds[..., 1] = upper

There is no wrapper class; a tensor is sufficient.

Batch-shape contract
--------------------
A source must return a *consistent* leading batch shape for every predicate-time
query participating in one evaluation::

    atomic bounds  ->  [B, 2]
    formula trace  ->  [B, T_valid, 2]

The core does not broadcast. Temporal evaluation stacks several time-indexed
values, and ``torch.stack`` does not broadcast mismatched batch dimensions, so a
ragged source would fail with an opaque shape error deep inside propagation.
``PropagationContext`` instead records the batch size of the first atom it sees
and reports a clear error if a later atom disagrees.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence

import torch

__all__ = ["ProbabilitySource", "TableProbabilitySource", "validate_bounds"]


def validate_bounds(bounds: torch.Tensor, *, context: str = "") -> torch.Tensor:
    """Validate a probability-bound tensor and return it unchanged.

    Enforces the atomic contract::

        0 <= lower <= P(E) <= upper <= 1

    Validation is **strict**: there is no numerical tolerance. Admitting
    something like ``[-5e-7, 0.8]`` would mean the core carries an invalid
    probability interval, which makes the invariant ambiguous everywhere
    downstream. Sources are responsible for clamping their own numerics before
    returning them. The complementary half of this rule lives in
    ``operators.py``, where every combination result is clamped to ``[0, 1]``,
    so float drift is removed where it is produced rather than tolerated where
    it is checked.

    Parameters
    ----------
    bounds : torch.Tensor
        Shape ``[B, 2]`` with ``[..., 0] = lower`` and ``[..., 1] = upper``.
    context : str
        Free-text description of the caller, included in error messages.

    Returns
    -------
    torch.Tensor
        The same tensor, unmodified.

    Raises
    ------
    TypeError
        If ``bounds`` is not a tensor at all.
    ValueError
        If the shape or the values violate the contract.
    """
    where = f" ({context})" if context else ""

    if not isinstance(bounds, torch.Tensor):
        raise TypeError(
            f"probability bounds must be a torch.Tensor, got "
            f"{type(bounds).__name__}{where}"
        )
    if bounds.ndim != 2:
        raise ValueError(
            f"probability bounds must have shape [B, 2], got ndim="
            f"{bounds.ndim} with shape {tuple(bounds.shape)}{where}"
        )
    if bounds.shape[-1] != 2:
        raise ValueError(
            f"probability bounds must have trailing dimension 2 "
            f"[lower, upper], got shape {tuple(bounds.shape)}{where}"
        )
    if not torch.isfinite(bounds).all():
        raise ValueError(f"probability bounds contain non-finite values: {bounds}{where}")

    lower = bounds[..., 0]
    upper = bounds[..., 1]

    if bool((lower < 0.0).any()):
        raise ValueError(f"lower bound must be >= 0, got {bounds}{where}")
    if bool((upper > 1.0).any()):
        raise ValueError(f"upper bound must be <= 1, got {bounds}{where}")
    if bool((lower > upper).any()):
        raise ValueError(f"lower bound must be <= upper bound, got {bounds}{where}")

    return bounds


class ProbabilitySource(ABC):
    """Abstract supplier of atomic predicate probability bounds.

    A source answers one question: for predicate ``mu_i`` at discrete time
    ``k``, what is a valid enclosure of ``P(E_{i,k})``?

    Implementations are duck-typed on the predicate: only ``predicate.uid`` is
    required, so this module does not import ``operators.py``.
    """

    @property
    @abstractmethod
    def horizon(self) -> int:
        """Largest valid discrete time index.

        Valid times are ``0 ... horizon`` **inclusive**, so a source covering
        four time steps has ``horizon == 3``.
        """
        raise NotImplementedError

    @abstractmethod
    def bounds(self, predicate, time: int) -> torch.Tensor:
        """Return probability bounds for the atomic event ``E_{i,k}``.

        Parameters
        ----------
        predicate : Predicate
            The atomic predicate ``mu_i``. Only its ``uid`` is meaningful here.
        time : int
            Discrete time ``k``, in ``0 ... horizon``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 2]`` satisfying ``0 <= lower <= P(E_{i,k}) <= upper <= 1``.
        """
        raise NotImplementedError


class TableProbabilitySource(ProbabilitySource):
    """A small table-driven source for testing and examples.

    Semantic correctness must be testable without any dynamics model, so this
    source lets bounds be written down directly::

        mu = Predicate(name="mu")
        source = TableProbabilitySource({(mu, 0): (0.9, 0.9), (mu, 1): (0.9, 0.9)})

    Entries are keyed on ``(predicate.uid, time)`` and validated eagerly on
    insertion, so a malformed interval is reported at the point it was written
    rather than during evaluation.
    """

    def __init__(
        self,
        table: Mapping[tuple, Sequence[float]] | None = None,
        horizon: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.dtype = dtype
        self._entries: dict[tuple[int, int], torch.Tensor] = {}
        self._names: dict[int, str] = {}
        self._explicit_horizon = horizon

        if horizon is not None and horizon < 0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")

        for (predicate, time), interval in (table or {}).items():
            self.set(predicate, time, *self._unpack(interval))

    @staticmethod
    def _unpack(interval: Iterable[float]) -> tuple[float, float]:
        values = tuple(interval)
        if len(values) != 2:
            raise ValueError(
                f"a probability interval must have exactly 2 entries "
                f"[lower, upper], got {values}"
            )
        return values  # type: ignore[return-value]

    def set(self, predicate, time: int, lower: float, upper: float) -> None:
        """Record ``P(E_{i,k}) in [lower, upper]``."""
        if not isinstance(time, int) or isinstance(time, bool):
            raise TypeError(f"time must be an int, got {time!r}")
        if time < 0:
            raise ValueError(f"time must be >= 0, got {time}")

        bounds = torch.tensor([[float(lower), float(upper)]], dtype=self.dtype)
        validate_bounds(bounds, context=f"{predicate} at time {time}")

        self._entries[(predicate.uid, time)] = bounds
        self._names[predicate.uid] = str(predicate)

    @property
    def horizon(self) -> int:
        if self._explicit_horizon is not None:
            return self._explicit_horizon
        if not self._entries:
            raise ValueError(
                "cannot infer horizon from an empty table; pass horizon= explicitly"
            )
        return max(time for _, time in self._entries)

    def bounds(self, predicate, time: int) -> torch.Tensor:
        try:
            return self._entries[(predicate.uid, time)]
        except KeyError:
            known = sorted(t for uid, t in self._entries if uid == predicate.uid)
            raise KeyError(
                f"no probability bounds recorded for {predicate} at time {time}; "
                f"this predicate has entries at times {known}"
            ) from None
