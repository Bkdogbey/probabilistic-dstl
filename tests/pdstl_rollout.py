"""A tiny differentiable single-integrator rollout, shared by
test_pdstl_end_to_end.py and run_pdstl_hard_optimization.py.

This is a vertical-slice diagnostic helper for the first end-to-end
differentiable pdSTL experiment -- not part of the pdstl package, and not a
substitute for the legacy (currently broken) src/planning package, which is
untouched. It exists here, shared, rather than duplicated per file, because
its exact numerics must match between the finite-difference-verified test
and the diagnostic optimization script; two independently written copies
would risk silent drift.

    x_{k+1} = x_k + dt * u_k
    u_k = u_max * tanh(v_k)

    mean_{k+1} = mean_k + dt * u_k
    covariance_{k+1} = covariance_k + process_noise

Single-trajectory only (batch size 1): every user of this helper in this
phase is a single Adam-optimized control sequence, not a batched rollout.
`covariance` never depends on `v` in this model -- all gradient signal from
any formula built on top of this rollout flows through the mean-residual
term of the Gaussian atomic probability, never through the variance term.
"""

from __future__ import annotations

import torch

__all__ = ["differentiable_rollout"]


def differentiable_rollout(
    v: torch.Tensor,
    x0_mean: torch.Tensor,
    x0_cov: torch.Tensor,
    *,
    dt: float,
    u_max: float,
    process_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out ``N`` steps from ``v: [N, D]``.

    Parameters
    ----------
    v : torch.Tensor
        Shape ``[N, D]``, unconstrained control parameters.
    x0_mean : torch.Tensor
        Shape ``[D]``.
    x0_cov : torch.Tensor
        Shape ``[D, D]``.
    dt, u_max : float
    process_noise : torch.Tensor
        Shape ``[D, D]`` (``Q``), added once per step.

    Returns
    -------
    mean : torch.Tensor
        Shape ``[1, N+1, D]``.
    covariance : torch.Tensor
        Shape ``[1, N+1, D, D]``.
    """
    means = [x0_mean]
    covs = [x0_cov]
    mean_k = x0_mean
    cov_k = x0_cov
    for k in range(v.shape[0]):
        u_k = u_max * torch.tanh(v[k])
        mean_k = mean_k + dt * u_k
        cov_k = cov_k + process_noise
        means.append(mean_k)
        covs.append(cov_k)

    mean = torch.stack(means, dim=0).unsqueeze(0)  # [1, N+1, D]
    covariance = torch.stack(covs, dim=0).unsqueeze(0)  # [1, N+1, D, D]
    return mean, covariance
