"""A differentiable Gaussian atomic-probability provider for pdSTL synthesis.

This module is a synthesis-side *provider*, not a semantics layer: it does
not define new STL syntax, and it is not a :class:`~pdstl.base.ProbabilitySource`.
It bridges a differentiable Gaussian trajectory (``mean``, ``covariance``
tensors that may themselves depend on upstream control parameters) to the
tensor contract :class:`~pdstl.graph.CompiledFormula` already expects --
``dict[predicate.uid, Tensor[B, N+1, 2]]``.

The returned traces must retain their autograd graph back to ``mean`` and
``covariance``, so this module deliberately never imports
:mod:`pdstl.base` / :mod:`pdstl.propagate`, and never routes through
:func:`pdstl.graph.materialize_atom_traces` (which is built around the
non-differentiable :class:`~pdstl.base.ProbabilitySource` contract). The
intended flow is::

    mean, covariance tensors (may require grad)
              |
              v
      gaussian_atom_traces(...)
              |
              v
      {predicate.uid: Tensor[B, N+1, 2]}
              |
              v
      CompiledFormula(...)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .operators import Predicate

__all__ = [
    "GaussianHalfspace",
    "gaussian_atom_traces",
    "gaussian_halfspace_probability",
]


@dataclass(frozen=True)
class GaussianHalfspace:
    """Describes the atom ``mu : a^T x - b >= 0`` for the Gaussian provider.

    This does not create a new STL predicate class -- ``predicate`` is an
    existing :class:`~pdstl.operators.Predicate` instance (the formula atom);
    this object only tells :func:`gaussian_atom_traces` how to compute that
    predicate's probability under a Gaussian state distribution.
    """

    predicate: Predicate
    normal: torch.Tensor  # [D], the "a" in a^T x - b >= 0
    threshold: float | torch.Tensor  # the "b"

    def __post_init__(self) -> None:
        if not isinstance(self.predicate, Predicate):
            raise TypeError(
                f"predicate must be a pdstl.Predicate, got {type(self.predicate).__name__}"
            )
        normal = torch.as_tensor(self.normal)
        if normal.ndim != 1:
            raise ValueError(f"normal must be a 1-D tensor [D], got shape {tuple(normal.shape)}")
        object.__setattr__(self, "normal", normal)


def gaussian_halfspace_probability(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    normal: torch.Tensor,
    threshold: float | torch.Tensor,
    *,
    variance_tol: float = 1e-6,
) -> torch.Tensor:
    """``P(a^T X - b >= 0)`` for ``X ~ N(mean, covariance)``, pointwise over ``[B, T]``.

    Parameters
    ----------
    mean : torch.Tensor
        Shape ``[B, T, D]``.
    covariance : torch.Tensor
        Shape ``[B, T, D, D]``.
    normal : torch.Tensor
        Shape ``[D]``.
    threshold : float or torch.Tensor
        Scalar ``b``.
    variance_tol : float
        Projected variances below ``-variance_tol`` are treated as a non-PSD
        covariance and raise; roundoff within ``[-variance_tol, 0)`` is
        silently clamped to exactly ``0``.

    Returns
    -------
    torch.Tensor
        Shape ``[B, T]``. For ``v_R = a^T Sigma a > 0``,
        ``Phi((a^T mean - b) / sqrt(v_R))``. For ``v_R == 0`` (a deterministic
        residual), exactly ``1.0`` if ``a^T mean - b >= 0`` else ``0.0`` --
        not an epsilon-fudged approximation.

    Notes
    -----
    Gradients back to ``mean``/``covariance`` are preserved throughout. The
    zero-variance branch is computed via a safe (never-zero) divisor before
    any division happens, so the *discarded* stochastic-branch value is
    always finite -- ``torch.where``'s backward multiplies each branch's
    local gradient by its own mask, and ``0 * NaN`` is ``NaN``, so a NaN
    forward value in the unselected branch would still poison the gradient
    even though it is "not selected". Making that branch's forward value
    merely wrong-but-finite (never NaN/Inf) at the discarded indices avoids
    that failure mode entirely.
    """
    normal = normal.to(dtype=mean.dtype, device=mean.device)

    m_R = torch.einsum("d,btd->bt", normal, mean) - threshold
    v_R_raw = torch.einsum("d,btde,e->bt", normal, covariance, normal)

    if bool((v_R_raw < -variance_tol).any()):
        raise ValueError(
            "covariance projection a^T Sigma a is negative beyond the "
            f"numerical tolerance ({variance_tol}); minimum observed value "
            f"was {v_R_raw.min()!r}, which indicates a non-PSD covariance "
            "matrix rather than roundoff"
        )
    v_R = torch.clamp(v_R_raw, min=0.0)

    positive = v_R > 0
    safe_v = torch.where(positive, v_R, torch.ones_like(v_R))
    z = m_R / torch.sqrt(safe_v)
    p_stochastic = torch.special.ndtr(z)

    p_deterministic = (m_R >= 0).to(dtype=mean.dtype)

    return torch.where(positive, p_stochastic, p_deterministic)


def gaussian_atom_traces(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    halfspaces: Sequence[GaussianHalfspace],
) -> dict[int, torch.Tensor]:
    """Materialize one differentiable ``[B, T, 2]`` trace per halfspace's predicate.

    Each trace is the degenerate interval ``[p, p]`` -- this Gaussian model
    reports an exact probability, not an enclosure, so lower and upper
    coincide. Both entries trace back to the same underlying computation, so
    gradient contributions from :class:`~pdstl.graph.CompiledFormula` reading
    either side correctly accumulate at ``mean``/``covariance``.

    Does not query a :class:`~pdstl.base.ProbabilitySource` and does not call
    :func:`pdstl.graph.materialize_atom_traces` -- the returned tensors are
    computed directly from ``mean``/``covariance`` so their autograd graph is
    preserved.
    """
    traces: dict[int, torch.Tensor] = {}
    for halfspace in halfspaces:
        uid = halfspace.predicate.uid
        if uid in traces:
            raise ValueError(
                f"duplicate halfspace for predicate {halfspace.predicate} (uid={uid})"
            )
        p = gaussian_halfspace_probability(mean, covariance, halfspace.normal, halfspace.threshold)
        traces[uid] = torch.stack([p, p], dim=-1)
    return traces
