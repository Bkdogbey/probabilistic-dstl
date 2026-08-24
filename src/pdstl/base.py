"""Probability input interfaces for pdSTL.

A :class:`ProbabilitySource` supplies bounds ``[lower, upper]`` on the
satisfaction probability of an atomic predicate at a discrete time. Atomic
bounds use shape ``[B, 2]`` with a consistent batch size per evaluation.
Formula traces use shape ``[B, T_valid, 2]``.

The trailing entries are ``bounds[..., 0] = lower`` and
``bounds[..., 1] = upper``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence

import torch

__all__ = ["ProbabilitySource", "TableProbabilitySource", "validate_bounds"]


def validate_bounds(bounds: torch.Tensor, *, context: str = "") -> torch.Tensor:
    """Validate bounds of shape ``[B, 2]`` and return them unchanged.

    The check strictly enforces finite values and
    ``0 <= lower <= upper <= 1``. Sources must clamp their own numerical drift.
    No tolerance is applied at this boundary, so the invariant remains exact
    throughout evaluation.

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
    """Supply bounds on atomic predicate satisfaction probabilities.

    For predicate ``mu_i`` at time ``k``, a source encloses ``P(E_{i,k})``.
    Predicate arguments are duck-typed; only ``predicate.uid`` is required.
    """

    @property
    @abstractmethod
    def horizon(self) -> int:
        """Largest valid discrete time index.

        Valid times are ``0 ... horizon`` inclusive.
        """
        raise NotImplementedError

    @abstractmethod
    def bounds(self, predicate, time: int) -> torch.Tensor:
        """Return bounds for atomic event ``E_{i,k}`` with shape ``[B, 2]``.

        Parameters
        ----------
        predicate : Predicate
            The atomic predicate ``mu_i``. Only its ``uid`` is meaningful here.
        time : int
            Discrete time ``k``, in ``0 ... horizon``.

        Returns
        -------
        torch.Tensor
            Bounds satisfying ``0 <= lower <= P(E_{i,k}) <= upper <= 1``.
        """
        raise NotImplementedError


class TableProbabilitySource(ProbabilitySource):
    """Table-driven probability bounds for tests and examples.

    Entries are keyed by ``(predicate.uid, time)`` and validated on insertion::

        mu = Predicate(name="mu")
        source = TableProbabilitySource({(mu, 0): (0.9, 0.9), (mu, 1): (0.9, 0.9)})

    Eager validation reports malformed intervals where they are defined.
    The horizon is explicit when provided and otherwise inferred from the
    largest recorded time.
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
