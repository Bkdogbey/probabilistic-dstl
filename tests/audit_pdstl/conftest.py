"""Shared fixtures and mathematical references for the pdSTL audit."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pdstl.base import Belief  # noqa: E402
from pdstl.operators import STL_Formula  # noqa: E402


class MinimalBelief(Belief):
    """Smallest concrete implementation of the documented Belief contract."""

    def __init__(self, value: float = 0.0):
        self._value = torch.tensor([[[value]]], dtype=torch.float64)

    def value(self):
        return self._value

    def probability_of(self, residual):
        return (residual >= 0).to(dtype=torch.float64)


class ConstantFormula(STL_Formula):
    """Formula returning a caller-supplied [B,T,2] trace."""

    def __init__(self, trace):
        super().__init__()
        # Temporal_Operator stores its shift matrix as float32, so use the
        # implementation's native dtype rather than introducing a test-only
        # dtype mismatch.
        if torch.is_tensor(trace):
            tensor = trace.to(dtype=torch.float32)
        else:
            tensor = torch.as_tensor(np.asarray(trace, dtype=float), dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        self.trace = tensor

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        return self.trace


def exact_temporal(trace, interval, operation):
    """Direct temporal reference with the implementation's terminal-repeat convention."""

    trace = np.asarray(trace, dtype=float)
    horizon = trace.shape[0]
    a, b = interval
    b = horizon - 1 if np.isinf(b) else int(b)
    out = np.full_like(trace, np.nan)
    for t in range(horizon):
        start = t + int(a)
        end = min(t + b, horizon - 1)
        if start <= end:
            reducer = np.min if operation == "min" else np.max
            out[t] = reducer(trace[start : end + 1], axis=0)
        else:
            # The finite-trace RNN pads unavailable future positions with the
            # terminal sample. The audit records this convention explicitly.
            out[t] = trace[-1]
    return out


def exact_until(phi, psi, interval):
    """Inclusive-prefix StoRI Until oracle from the audit brief."""

    phi = np.asarray(phi, dtype=float)
    psi = np.asarray(psi, dtype=float)
    horizon = phi.shape[0]
    a, b = interval
    b = horizon - 1 if np.isinf(b) else int(b)
    out = np.zeros_like(phi)

    for t in range(horizon):
        candidates = []
        for tau in range(t + int(a), min(t + b, horizon - 1) + 1):
            prefix = np.min(phi[t : tau + 1], axis=0)
            lower = max(psi[tau, 0] + prefix[0] - 1.0, 0.0)
            upper = min(psi[tau, 1], prefix[1])
            candidates.append((lower, upper))
        if candidates:
            out[t, 0] = max(value[0] for value in candidates)
            out[t, 1] = max(value[1] for value in candidates)
    return out
