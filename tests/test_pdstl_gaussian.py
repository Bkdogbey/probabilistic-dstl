"""Numeric, autograd, and finite-difference tests for pdstl.gaussian.

pdstl.gaussian bridges a differentiable Gaussian trajectory to the tensor
contract CompiledFormula expects. It is a synthesis-side atomic probability
provider, not a semantics layer and not a ProbabilitySource -- it never
imports pdstl.base/pdstl.propagate and never routes through
pdstl.graph.materialize_atom_traces, so gradients back to mean/covariance
survive.
"""

import math

import pytest
import torch

from pdstl import Predicate
from pdstl.gaussian import GaussianHalfspace, gaussian_atom_traces, gaussian_halfspace_probability


def scalar(mean_value, var_value, normal_value=1.0, threshold=0.0):
    """A [1,1,1] mean/[1,1,1,1] covariance pair for one scalar atom."""
    mean = torch.tensor([[[mean_value]]])
    covariance = torch.tensor([[[[var_value]]]])
    normal = torch.tensor([normal_value])
    return mean, covariance, normal, threshold


# ---------------------------------------------------------------------------
# Numeric cases
# ---------------------------------------------------------------------------


def test_symmetric_residual_gives_probability_half():
    mean, covariance, normal, threshold = scalar(mean_value=0.0, var_value=0.25)
    p = gaussian_halfspace_probability(mean, covariance, normal, threshold)
    assert p.item() == pytest.approx(0.5, abs=1e-6)


def test_clearly_positive_residual_gives_high_probability():
    # z = 3 / 1 = 3 -> Phi(3) ~= 0.9987
    mean, covariance, normal, threshold = scalar(mean_value=3.0, var_value=1.0)
    p = gaussian_halfspace_probability(mean, covariance, normal, threshold)
    assert p.item() > 0.99
    assert p.item() == pytest.approx(0.9986501, abs=1e-5)


def test_clearly_negative_residual_gives_low_probability():
    mean, covariance, normal, threshold = scalar(mean_value=-3.0, var_value=1.0)
    p = gaussian_halfspace_probability(mean, covariance, normal, threshold)
    assert p.item() < 0.01
    assert p.item() == pytest.approx(0.0013499, abs=1e-5)


def test_zero_variance_positive_residual_is_exactly_one():
    mean, covariance, normal, threshold = scalar(mean_value=0.5, var_value=0.0)
    p = gaussian_halfspace_probability(mean, covariance, normal, threshold)
    assert p.item() == 1.0


def test_zero_variance_negative_residual_is_exactly_zero():
    mean, covariance, normal, threshold = scalar(mean_value=-0.5, var_value=0.0)
    p = gaussian_halfspace_probability(mean, covariance, normal, threshold)
    assert p.item() == 0.0


def test_zero_variance_zero_residual_uses_closed_halfspace_convention():
    """m_R == 0 exactly at v_R == 0: h(x) >= 0 is closed, so p == 1.0."""
    mean, covariance, normal, threshold = scalar(mean_value=0.0, var_value=0.0)
    p = gaussian_halfspace_probability(mean, covariance, normal, threshold)
    assert p.item() == 1.0


def test_covariance_projection_uses_full_quadratic_form():
    """Non-diagonal Sigma, non-axis-aligned normal: v_R = a^T Sigma a exactly."""
    mean = torch.tensor([[[1.0, 1.0]]])
    covariance = torch.tensor([[[[2.0, 0.5], [0.5, 1.0]]]])
    normal = torch.tensor([1.0, 2.0])
    threshold = 0.0

    a = normal
    sigma = covariance[0, 0]
    expected_v_r = float(a @ sigma @ a)  # 1*2*1 + 1*2*0.5*2 + 4*1 = 2 + 2 + 4 = 8
    assert expected_v_r == pytest.approx(8.0)

    m_r = float(a @ mean[0, 0]) - threshold  # 1 + 2 = 3
    expected_p = 0.5 * (1.0 + math.erf((m_r / math.sqrt(expected_v_r)) / math.sqrt(2.0)))

    p = gaussian_halfspace_probability(mean, covariance, normal, threshold)
    assert p.item() == pytest.approx(expected_p, abs=1e-5)


def test_batched_trajectory_shapes():
    batch, time, dim = 3, 4, 2
    mean = torch.randn(batch, time, dim)
    covariance = torch.eye(dim).expand(batch, time, dim, dim) * 0.1
    normal = torch.tensor([1.0, 0.0])

    p = gaussian_halfspace_probability(mean, covariance, normal, 0.0)
    assert p.shape == (batch, time)
    assert torch.all((p >= 0) & (p <= 1))

    mu = Predicate(name="mu")
    hs = GaussianHalfspace(predicate=mu, normal=normal, threshold=0.0)
    traces = gaussian_atom_traces(mean, covariance, [hs])
    assert traces[mu.uid].shape == (batch, time, 2)


def test_negative_variance_beyond_tolerance_raises():
    normal = torch.tensor([1.0, 0.0])
    covariance = torch.tensor([[[[-1.0, 0.0], [0.0, 1.0]]]])  # a^T Sigma a = -1
    with pytest.raises(ValueError, match="negative beyond"):
        gaussian_halfspace_probability(torch.zeros(1, 1, 2), covariance, normal, 0.0)


def test_small_negative_variance_roundoff_is_clamped():
    normal = torch.tensor([1.0])
    covariance = torch.tensor([[[[-1e-9]]]])  # within the default 1e-6 tolerance
    mean = torch.tensor([[[0.5]]])
    p = gaussian_halfspace_probability(mean, covariance, normal, 0.0)
    # Treated as zero variance: deterministic branch, positive residual.
    assert p.item() == 1.0


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------


def test_gradient_flows_to_mean_and_covariance():
    mean = torch.tensor([[[0.3]]], requires_grad=True)
    covariance = (torch.ones(1, 1, 1, 1) * 0.09).requires_grad_(True)
    normal = torch.tensor([1.0])

    p = gaussian_halfspace_probability(mean, covariance, normal, 0.0)
    p.sum().backward()

    assert mean.grad is not None
    assert covariance.grad is not None
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(covariance.grad).all()
    assert mean.grad.norm().item() > 0
    assert covariance.grad.norm().item() > 0


def test_zero_variance_branch_has_zero_local_gradient_and_no_nan_leak():
    """The discarded stochastic branch must not poison the selected one with NaN."""
    mean = torch.tensor([[[0.5], [0.5]]], requires_grad=True)  # two batch rows
    covariance = torch.tensor([[[[0.0]], [[0.09]]]])  # row 0 deterministic, row 1 stochastic
    normal = torch.tensor([1.0])

    p = gaussian_halfspace_probability(mean, covariance, normal, 0.0)
    p.sum().backward()

    assert torch.isfinite(mean.grad).all()
    assert mean.grad[0, 0, 0].item() == 0.0  # deterministic branch: zero local gradient
    assert mean.grad[0, 1, 0].item() != 0.0  # stochastic branch: nonzero local gradient


# ---------------------------------------------------------------------------
# Finite-difference check (atomic probability alone)
# ---------------------------------------------------------------------------


def test_finite_difference_matches_autograd_for_atomic_probability():
    mean = torch.tensor([[[0.3, -0.2]]], requires_grad=True)  # z ~ O(1), interior of the CDF
    covariance = torch.tensor([[[[0.09, 0.0], [0.0, 0.16]]]])
    normal = torch.tensor([1.0, 1.0])
    threshold = 0.1

    p = gaussian_halfspace_probability(mean, covariance, normal, threshold)
    p.sum().backward()
    analytic = mean.grad.clone()

    h = 1e-4
    numeric = torch.zeros_like(mean).squeeze(0).squeeze(0)
    flat_mean = mean.detach().clone()
    for i in range(mean.shape[-1]):
        plus = flat_mean.clone()
        plus[0, 0, i] += h
        minus = flat_mean.clone()
        minus[0, 0, i] -= h
        p_plus = gaussian_halfspace_probability(plus, covariance, normal, threshold)
        p_minus = gaussian_halfspace_probability(minus, covariance, normal, threshold)
        numeric[i] = (p_plus.sum() - p_minus.sum()) / (2 * h)

    assert torch.allclose(analytic[0, 0], numeric, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# gaussian_atom_traces
# ---------------------------------------------------------------------------


def test_gaussian_atom_traces_returns_degenerate_interval():
    mu = Predicate(name="mu")
    nu = Predicate(name="nu")
    mean = torch.tensor([[[0.5, -0.5]]])
    covariance = torch.eye(2).expand(1, 1, 2, 2) * 0.1

    halfspaces = [
        GaussianHalfspace(predicate=mu, normal=torch.tensor([1.0, 0.0]), threshold=0.0),
        GaussianHalfspace(predicate=nu, normal=torch.tensor([0.0, 1.0]), threshold=0.0),
    ]
    traces = gaussian_atom_traces(mean, covariance, halfspaces)

    assert set(traces) == {mu.uid, nu.uid}
    for uid, trace in traces.items():
        assert trace.shape == (1, 1, 2)
        assert trace[..., 0].item() == pytest.approx(trace[..., 1].item())


def test_gaussian_atom_traces_rejects_duplicate_predicates():
    mu = Predicate(name="mu")
    mean = torch.zeros(1, 1, 1)
    covariance = torch.ones(1, 1, 1, 1) * 0.1
    halfspaces = [
        GaussianHalfspace(predicate=mu, normal=torch.tensor([1.0]), threshold=0.0),
        GaussianHalfspace(predicate=mu, normal=torch.tensor([1.0]), threshold=0.5),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        gaussian_atom_traces(mean, covariance, halfspaces)


# ---------------------------------------------------------------------------
# GaussianHalfspace construction
# ---------------------------------------------------------------------------


def test_gaussian_halfspace_rejects_non_predicate():
    with pytest.raises(TypeError, match="Predicate"):
        GaussianHalfspace(predicate="not a predicate", normal=torch.tensor([1.0]), threshold=0.0)


def test_gaussian_halfspace_rejects_non_1d_normal():
    mu = Predicate(name="mu")
    with pytest.raises(ValueError, match="1-D"):
        GaussianHalfspace(predicate=mu, normal=torch.tensor([[1.0]]), threshold=0.0)


def test_gaussian_halfspace_is_frozen():
    mu = Predicate(name="mu")
    hs = GaussianHalfspace(predicate=mu, normal=torch.tensor([1.0]), threshold=0.0)
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError subclasses AttributeError
        hs.threshold = 1.0
