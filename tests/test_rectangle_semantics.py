"""Rectangle probability semantics and the gradients the planner depends on.

A rectangle is `AxisInterval(x) AND AxisInterval(y)`, with "outside" its negation. These tests
pin two things the refactor must never break:

* the HARD interval is the Frechet enclosure `[max(0, p_x + p_y - 1), min(p_x, p_y)]`, computed
  from exact marginals with no independence assumption;
* the SMOOTH lower bound carries gradient where the hard clamp does not.

The smooth path has a limited reach, and one test pins that honestly rather than pretending
otherwise -- see `test_smooth_goal_gradient_is_an_approach_phase_fix`.
"""

import pytest
import torch

from models.beliefs import GaussianBelief
from pdstl.base import BeliefTrajectory
from pdstl.predicates import AxisInterval, InsideRectangle, OutsideRectangle

RECT = ([4.0, 6.0], [3.0, 5.0])


def evaluate(rectangle, mean, covariance, scale=-1):
    """Probability interval [B, 2] of a rectangle event under one belief."""
    belief = GaussianBelief(mean, covariance)
    return rectangle(BeliefTrajectory([belief]), scale=scale)[:, 0]


# --- Hard semantics ---------------------------------------------------------


@pytest.mark.parametrize("point", [[5.0, 4.0], [4.0, 3.0], [6.0, 5.0]])
def test_deterministic_belief_inside_the_rectangle_is_certain(point):
    """Zero variance, mean inside (corners included: the rectangle is closed)."""
    bounds = evaluate(InsideRectangle(*RECT), torch.tensor([point]), torch.zeros(1, 2))
    assert bounds.tolist() == [[1.0, 1.0]]


@pytest.mark.parametrize("point", [[0.0, 0.0], [5.0, 5.01], [3.99, 4.0]])
def test_deterministic_belief_outside_the_rectangle_is_impossible(point):
    bounds = evaluate(InsideRectangle(*RECT), torch.tensor([point]), torch.zeros(1, 2))
    assert bounds.tolist() == [[0.0, 0.0]]


def test_outside_is_exactly_the_complement_of_inside():
    torch.manual_seed(0)
    mean = torch.randn(32, 2) * 2 + torch.tensor([5.0, 4.0])
    covariance = torch.diag_embed(torch.rand(32, 2) + 0.05)

    inside = evaluate(InsideRectangle(*RECT), mean, covariance)
    outside = evaluate(OutsideRectangle(*RECT), mean, covariance)

    torch.testing.assert_close(outside[:, 0], 1.0 - inside[:, 1])
    torch.testing.assert_close(outside[:, 1], 1.0 - inside[:, 0])


def test_bounds_stay_valid_under_strong_correlation():
    """Frechet needs no independence, so a full covariance must not break the enclosure."""
    mean = torch.tensor([[5.0, 4.0]], dtype=torch.float64)
    for rho in (-0.95, -0.5, 0.0, 0.5, 0.95):
        covariance = torch.tensor([[[0.6, rho * 0.6], [rho * 0.6, 0.6]]], dtype=torch.float64)
        for rectangle in (InsideRectangle(*RECT), OutsideRectangle(*RECT)):
            lower, upper = evaluate(rectangle, mean, covariance)[0].tolist()
            assert 0.0 <= lower <= upper <= 1.0, f"rho={rho} {rectangle}"


def test_monte_carlo_probability_lies_inside_the_analytical_interval():
    mean = torch.tensor([[5.1, 3.9]], dtype=torch.float64)
    covariance = torch.tensor([[[0.5, -0.35], [-0.35, 0.45]]], dtype=torch.float64)
    lower, upper = evaluate(InsideRectangle(*RECT), mean, covariance)[0].tolist()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        samples = torch.distributions.MultivariateNormal(mean[0], covariance[0]).sample((200_000,))
    (x_lo, x_hi), (y_lo, y_hi) = RECT
    hit = (
        (samples[:, 0] >= x_lo) & (samples[:, 0] <= x_hi)
        & (samples[:, 1] >= y_lo) & (samples[:, 1] <= y_hi)
    ).double().mean().item()

    tolerance = 3.0 * (hit * (1 - hit) / 200_000) ** 0.5
    assert lower - tolerance <= hit <= upper + tolerance, f"{hit} outside [{lower}, {upper}]"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x_range": [6.0, 4.0], "y_range": [3.0, 5.0]},   # reversed
        {"x_range": [4.0, 4.0], "y_range": [3.0, 5.0]},   # degenerate
        {"x_range": [4.0, 6.0], "y_range": [3.0, 5.0], "dims": (1, 1)},
        {"x_range": [4.0, 6.0], "y_range": [3.0, 5.0], "dims": (0, -1)},
    ],
)
def test_invalid_geometry_raises_clearly(kwargs):
    with pytest.raises(ValueError):
        InsideRectangle(**kwargs)


@pytest.mark.parametrize("args", [(6.0, 4.0), (4.0, 4.0)])
def test_axis_interval_rejects_invalid_bounds(args):
    with pytest.raises(ValueError, match="min < max"):
        AxisInterval(*args, dim=0)


def test_batch_shape_and_dtype_are_preserved():
    mean = torch.randn(9, 2, dtype=torch.float64) + torch.tensor([5.0, 4.0], dtype=torch.float64)
    covariance = torch.full((9, 2), 0.3, dtype=torch.float64)
    bounds = evaluate(InsideRectangle(*RECT), mean, covariance)
    assert bounds.shape == (9, 2) and bounds.dtype is torch.float64


def test_rectangle_over_non_default_state_dimensions():
    """A 4-state double integrator measures position on dims (0, 1) only."""
    mean = torch.tensor([[5.0, 4.0, 9.9, -9.9]])
    bounds = evaluate(InsideRectangle(*RECT, dims=(0, 1)), mean, torch.zeros(1, 4))
    assert bounds.tolist() == [[1.0, 1.0]]


# --- Gradients --------------------------------------------------------------


def _goal_gradient(mean_value, scale, goal=([8.5, 9.5], [8.5, 9.5]), sigma_sq=0.04):
    mean = torch.tensor([[mean_value, mean_value]], requires_grad=True)
    lower = evaluate(InsideRectangle(*goal), mean, torch.tensor([[sigma_sq, sigma_sq]]), scale)[0, 0]
    lower.backward()
    return lower.item(), mean.grad.abs().sum().item()


def test_smooth_inside_goal_score_has_a_finite_gradient():
    value, gradient = _goal_gradient(8.4, scale=1.0)
    assert torch.isfinite(torch.tensor([value, gradient])).all()
    assert gradient > 0.0


def test_smooth_goal_gradient_is_an_approach_phase_fix():
    """Softplus revives the gradient near the goal -- and only near it.

    Within roughly 3 sigma the hard clamp is what kills the gradient, and smoothing fixes it.
    Further out the Gaussian CDF itself underflows to exactly zero in float32, so no operator
    smoothing can help; the goal-directed initialisation is what covers that regime. This test
    documents both halves so nobody mistakes the far-field zero for a regression.
    """
    _, near_hard = _goal_gradient(8.4, scale=-1)
    _, near_smooth = _goal_gradient(8.4, scale=1.0)
    assert near_hard == 0.0, "hard clamp gives no gradient on approach"
    assert near_smooth > 1e-3, "softplus must revive it"

    _, far_smooth = _goal_gradient(1.0, scale=1.0)
    assert far_smooth == 0.0, "far field is lost to CDF underflow, not to the clamp"


def test_outside_rectangle_gradient_pushes_away_from_the_block():
    """Safety gradient must point out of the obstacle along the escaping axis."""
    block = ([4.0, 6.0], [0.0, 4.0])
    for mean_value, expect_x in [([6.2, 2.0], True), ([5.0, 4.3], False)]:
        mean = torch.tensor([mean_value], requires_grad=True)
        lower = evaluate(OutsideRectangle(*block), mean, torch.tensor([[0.09, 0.09]]), 20.0)[0, 0]
        lower.backward()
        gx, gy = mean.grad[0].tolist()
        assert torch.isfinite(mean.grad).all()
        if expect_x:
            # Beside the block: moving further out in x raises the safety probability.
            assert gx > 0.1, f"no lateral escape gradient at {mean_value}"
        else:
            # Centred in x: p_x sits on its plateau, so only y can help. Not a defect.
            assert abs(gx) < 1e-6 and gy > 0.1


def test_hard_evaluation_carries_no_gradient_into_the_clamped_region():
    mean = torch.tensor([[1.0, 1.0]], requires_grad=True)
    lower = evaluate(InsideRectangle([8.5, 9.5], [8.5, 9.5]), mean, torch.tensor([[0.04, 0.04]]))[0, 0]
    assert lower.item() == 0.0
    lower.backward()
    assert mean.grad.abs().sum().item() == 0.0
