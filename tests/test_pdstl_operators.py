"""Unit test for the pdSTL And operator's robustness lower bound.

Confirms And.robustness_trace uses the sound Frechet bound
(max(l1+l2-1, 0)), not the independence-assuming product (l1*l2) that was
in place before this fix. The product form silently overestimates the joint
satisfaction probability for correlated (non-independent) sub-formulas, which
is exactly how this operator is used elsewhere (obstacle-safety predicates
chained over the same trajectory belief).
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from pdstl.operators import And, STL_Formula  # noqa: E402


class _ConstantFormula(STL_Formula):
    """A fake STL formula returning a fixed [B,T,2] robustness trace, ignoring the input trajectory."""

    def __init__(self, lower: float, upper: float):
        super().__init__()
        self._trace = torch.tensor([[[lower, upper]]])

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        return self._trace


def _and_bounds(l1: float, u1: float, l2: float, u2: float) -> tuple[float, float]:
    formula = And(_ConstantFormula(l1, u1), _ConstantFormula(l2, u2))
    trace = formula(None)
    return trace[0, 0, 0].item(), trace[0, 0, 1].item()


def test_and_uses_frechet_lower_bound_not_product():
    # l1=0.6, l2=0.7: Frechet = max(0.6+0.7-1, 0) = 0.3; product would give 0.42.
    lower, upper = _and_bounds(0.6, 0.9, 0.7, 0.95)
    assert lower == pytest.approx(0.3)
    assert upper == pytest.approx(0.9)


def test_and_lower_bound_clamped_at_zero():
    # l1=0.2, l2=0.3: l1+l2-1 = -0.5 -> clamped to 0. Product would give 0.06 (nonzero).
    lower, _ = _and_bounds(0.2, 0.5, 0.3, 0.6)
    assert lower == pytest.approx(0.0)


def test_and_upper_bound_is_min():
    _, upper = _and_bounds(0.1, 0.4, 0.2, 0.3)
    assert upper == pytest.approx(0.3)
