"""Tests for the probability-bound input foundation: sources and validation.

Covers requirement groups A (validate_bounds), B (OfflineSource), and
C (OnlineSource).
"""

import pytest
import torch

from pdstl import (
    OfflineSource,
    OnlineSource,
    Predicate,
    validate_bounds,
)

# ---------------------------------------------------------------------------
# A. validate_bounds
# ---------------------------------------------------------------------------


def test_lower_below_zero_is_rejected():
    with pytest.raises(ValueError, match="lower bound must be >= 0"):
        validate_bounds(torch.tensor([[-0.1, 0.8]]))


def test_upper_above_one_is_rejected():
    with pytest.raises(ValueError, match="upper bound must be <= 1"):
        validate_bounds(torch.tensor([[0.2, 1.3]]))


def test_lower_above_upper_is_rejected():
    with pytest.raises(ValueError, match="lower bound must be <= upper bound"):
        validate_bounds(torch.tensor([[0.8, 0.2]]))


def test_malformed_bound_shapes_are_rejected():
    with pytest.raises(ValueError, match="trailing dimension 2"):
        validate_bounds(torch.tensor([[0.1, 0.5, 0.9]]))
    with pytest.raises(ValueError, match="trailing dimension 2"):
        validate_bounds(torch.tensor(0.5))
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        validate_bounds([[0.1, 0.5]])


def test_non_finite_bounds_are_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        validate_bounds(torch.tensor([[float("nan"), 0.5]]))
    with pytest.raises(ValueError, match="non-finite"):
        validate_bounds(torch.tensor([[0.1, float("inf")]]))


# ---------------------------------------------------------------------------
# B. OfflineSource
# ---------------------------------------------------------------------------


def test_offline_bounds_retrieval():
    mu = Predicate(name="mu")
    trace = torch.tensor([[0.25, 0.75], [0.4, 0.6], [0.1, 0.9]]).unsqueeze(0)
    source = OfflineSource({mu: trace})

    assert len(source) == 3
    assert source.bounds(mu, 0)[0].tolist() == pytest.approx([0.25, 0.75])
    assert source.bounds(mu, 2)[0].tolist() == pytest.approx([0.1, 0.9])


def test_offline_exact_probability_is_a_degenerate_interval():
    mu = Predicate(name="mu")
    source = OfflineSource({mu: torch.tensor([[0.7, 0.7]]).unsqueeze(0)})

    assert source.bounds(mu, 0)[0].tolist() == pytest.approx([0.7, 0.7])


def test_offline_multiple_predicates_are_independent():
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = OfflineSource(
        {
            a: torch.tensor([[0.6, 0.9]]).unsqueeze(0),
            b: torch.tensor([[0.1, 0.2]]).unsqueeze(0),
        }
    )

    assert source.bounds(a, 0)[0].tolist() == pytest.approx([0.6, 0.9])
    assert source.bounds(b, 0)[0].tolist() == pytest.approx([0.1, 0.2])


def test_offline_batched_inputs_are_preserved():
    mu = Predicate(name="mu")
    trace = torch.tensor([[0.3, 0.5], [0.3, 0.5], [0.3, 0.5]]).unsqueeze(1)  # [3, 1, 2]
    source = OfflineSource({mu: trace})

    bounds = source.bounds(mu, 0)
    assert bounds.shape == (3, 2)
    assert torch.allclose(bounds, torch.tensor([[0.3, 0.5]] * 3))


def test_offline_rejects_bounds_outside_unit_interval():
    mu = Predicate(name="mu")
    with pytest.raises(ValueError, match="upper bound must be <= 1"):
        OfflineSource({mu: torch.tensor([[0.5, 1.5]]).unsqueeze(0)})


def test_offline_rejects_lower_above_upper():
    mu = Predicate(name="mu")
    with pytest.raises(ValueError, match="lower bound must be <= upper bound"):
        OfflineSource({mu: torch.tensor([[0.9, 0.1]]).unsqueeze(0)})


def test_offline_rejects_malformed_trace_shape():
    mu = Predicate(name="mu")
    with pytest.raises(ValueError, match="trailing dimension 2"):
        OfflineSource({mu: torch.tensor([[0.1, 0.5, 0.9]]).unsqueeze(0)})


def test_offline_rejects_inconsistent_trace_lengths():
    a = Predicate(name="A")
    b = Predicate(name="B")
    with pytest.raises(ValueError, match="batch size and length"):
        OfflineSource(
            {
                a: torch.tensor([[0.5, 0.5], [0.5, 0.5]]).unsqueeze(0),  # T=2
                b: torch.tensor([[0.5, 0.5]]).unsqueeze(0),  # T=1
            }
        )


def test_offline_rejects_inconsistent_batch_sizes():
    a = Predicate(name="A")
    b = Predicate(name="B")
    with pytest.raises(ValueError, match="batch size and length"):
        OfflineSource(
            {
                a: torch.tensor([[0.5, 0.5]]).unsqueeze(0),  # B=1
                b: torch.tensor([[0.5, 0.5]] * 2).unsqueeze(1),  # B=2
            }
        )


def test_offline_preserves_dtype_and_device():
    mu = Predicate(name="mu")
    trace = torch.tensor([[0.5, 0.5]], dtype=torch.float64).unsqueeze(0)
    source = OfflineSource({mu: trace})

    bounds = source.bounds(mu, 0)
    assert bounds.dtype == torch.float64
    assert bounds.device == trace.device


def test_offline_preserves_autograd():
    mu = Predicate(name="mu")
    trace = torch.tensor([[[0.5, 0.5]]], requires_grad=True)
    source = OfflineSource({mu: trace})

    source.bounds(mu, 0).sum().backward()

    assert trace.grad is not None


# ---------------------------------------------------------------------------
# C. OnlineSource
# ---------------------------------------------------------------------------


def test_online_append_and_retrieval():
    mu = Predicate(name="mu")
    source = OnlineSource()
    source.append({mu: torch.tensor([[0.25, 0.75]])})
    source.append({mu: torch.tensor([[0.4, 0.6]])})

    assert len(source) == 2
    assert source.bounds(mu, 0)[0].tolist() == pytest.approx([0.25, 0.75])
    assert source.bounds(mu, 1)[0].tolist() == pytest.approx([0.4, 0.6])


def test_online_exact_probability_is_a_degenerate_interval():
    mu = Predicate(name="mu")
    source = OnlineSource()
    source.append({mu: torch.tensor([[0.7, 0.7]])})

    assert source.bounds(mu, 0)[0].tolist() == pytest.approx([0.7, 0.7])


def test_online_multiple_predicates_are_independent():
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = OnlineSource()
    source.append({a: torch.tensor([[0.6, 0.9]]), b: torch.tensor([[0.1, 0.2]])})

    assert source.bounds(a, 0)[0].tolist() == pytest.approx([0.6, 0.9])
    assert source.bounds(b, 0)[0].tolist() == pytest.approx([0.1, 0.2])


def test_online_batched_inputs_are_preserved():
    mu = Predicate(name="mu")
    source = OnlineSource()
    source.append({mu: torch.tensor([[0.3, 0.5]] * 4)})

    bounds = source.bounds(mu, 0)
    assert bounds.shape == (4, 2)


def test_online_rejects_bounds_outside_unit_interval():
    mu = Predicate(name="mu")
    source = OnlineSource()
    with pytest.raises(ValueError, match="lower bound must be >= 0"):
        source.append({mu: torch.tensor([[-0.1, 0.5]])})


def test_online_rejects_lower_above_upper():
    mu = Predicate(name="mu")
    source = OnlineSource()
    with pytest.raises(ValueError, match="lower bound must be <= upper bound"):
        source.append({mu: torch.tensor([[0.9, 0.1]])})


def test_online_rejects_malformed_step_shape():
    mu = Predicate(name="mu")
    source = OnlineSource()
    with pytest.raises(ValueError, match="trailing dimension 2"):
        source.append({mu: torch.tensor([[0.1, 0.5, 0.9]])})


def test_online_rejects_inconsistent_predicate_set():
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = OnlineSource()
    source.append({a: torch.tensor([[0.5, 0.5]])})

    with pytest.raises(ValueError, match="expected predicates"):
        source.append({b: torch.tensor([[0.5, 0.5]])})


def test_online_rejects_inconsistent_batch_size():
    mu = Predicate(name="mu")
    source = OnlineSource()
    source.append({mu: torch.tensor([[0.5, 0.5]])})

    with pytest.raises(ValueError, match="batch size changed"):
        source.append({mu: torch.tensor([[0.5, 0.5]] * 2)})


def test_online_preserves_dtype_and_device():
    mu = Predicate(name="mu")
    step = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    source = OnlineSource()
    source.append({mu: step})

    bounds = source.bounds(mu, 0)
    assert bounds.dtype == torch.float64
    assert bounds.device == step.device


def test_online_preserves_autograd():
    mu = Predicate(name="mu")
    step = torch.tensor([[0.5, 0.5]], requires_grad=True)
    source = OnlineSource()
    source.append({mu: step})

    source.bounds(mu, 0).sum().backward()

    assert step.grad is not None
