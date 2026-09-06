from pathlib import Path

import pytest
import torch


@pytest.fixture
def legacy(monkeypatch):
    # Legacy models/utils are source-tree helpers, outside the installed core.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    from models.dynamics import GaussianBelief
    from utils import create_belief_trajectory
    return GaussianBelief, create_belief_trajectory


def test_zero_variance_inclusive_threshold_and_finite_gradients(legacy):
    from pdstl.operators import GreaterThan, LessThan
    _, create = legacy
    means = torch.tensor([-1., 0., 1.], dtype=torch.float64, requires_grad=True)
    variance = torch.zeros_like(means, requires_grad=True)
    beliefs = create(means, variance)
    above = GreaterThan(0)(beliefs)
    below = LessThan(0)(beliefs)
    torch.testing.assert_close(above, torch.tensor([[[0., 0.], [1., 1.], [1., 1.]]], dtype=means.dtype))
    torch.testing.assert_close(below, torch.tensor([[[1., 1.], [1., 1.], [0., 0.]]], dtype=means.dtype))
    (above.sum() + below.sum()).backward()
    assert torch.isfinite(means.grad).all()
    assert torch.isfinite(variance.grad).all()


def test_legacy_wrapper_preserves_nonzero_gradient(legacy):
    from pdstl.operators import GreaterThan
    _, create = legacy
    means = torch.tensor([0.2, 0.4], dtype=torch.float64, requires_grad=True)
    variance = torch.ones_like(means, requires_grad=True)
    output = GreaterThan(0)(create(means, variance))
    output.sum().backward()
    assert output.dtype == means.dtype
    assert torch.all(means.grad > 0)
    assert torch.isfinite(variance.grad).all()


def test_negative_variance_rejected(legacy):
    gaussian, _ = legacy
    with pytest.raises(ValueError, match="nonnegative"):
        gaussian(torch.zeros(1, 1, 1), -torch.ones(1, 1, 1))
