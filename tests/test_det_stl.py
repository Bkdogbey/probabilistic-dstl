"""Independent window references for the recurrent deterministic baseline."""

import math

import pytest
import torch

from baselines.det_stl import DetUntil, from_pdstl
from models.rollouts import (
    LaneBeliefTrajectory,
    create_gaussian_belief_trajectory,
)
from pdstl.operators import (
    Always,
    And,
    Eventually,
    Implies,
    Negation,
    Or,
    Predicate,
    Until,
)
from pdstl.predicates import (
    AxisInterval,
    GreaterThan,
    HalfSpace,
    InsideRectangle,
    OutsideRectangle,
    RelativeAxisInterval,
)


def reduce_reference(values, beta, minimum, dim):
    if beta is None:
        return (torch.amin if minimum else torch.amax)(values, dim=dim)
    sign = -1 if minimum else 1
    return sign * torch.logsumexp(sign * beta * values, dim=dim) / beta


@pytest.mark.parametrize(
    "operator,minimum", [(Always, True), (Eventually, False)]
)
@pytest.mark.parametrize(
    "window", [[0, 0], [0, 1], [1, 3], None, [0, math.inf], [2, math.inf]]
)
@pytest.mark.parametrize("beta", [None, 2.0])
def test_temporal_recurrence_matches_direct_windows(
    operator, minimum, window, beta
):
    mean = torch.tensor(
        [
            [[3.0], [-2.0], [1.0], [4.0], [-1.0]],
            [[-1.0], [2.0], [5.0], [0.0], [3.0]],
        ],
        dtype=torch.float64,
    )
    spec = operator(GreaterThan(0), window)
    baseline = from_pdstl(spec)
    actual = baseline(mean, beta=beta)[..., 0]
    a, b = (0, math.inf) if window is None else window
    lookahead = a if math.isinf(b) else b
    expected = torch.stack(
        [
            reduce_reference(
                mean[:, t + a : None if math.isinf(b) else t + b + 1, 0],
                beta,
                minimum,
                1,
            )
            for t in range(mean.shape[1] - lookahead)
        ],
        dim=1,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "window", [[0, 0], [0, 2], [1, 3], None, [2, math.inf]]
)
@pytest.mark.parametrize("beta", [None, 3.0])
def test_until_recurrence_matches_inclusive_witness_definition(window, beta):
    mean = torch.tensor(
        [[[3.0, -1.0], [2.0, 4.0], [-2.0, 2.0], [1.0, 3.0], [4.0, 0.0]]],
        dtype=torch.float64,
    )
    spec = Until(GreaterThan(0, dim=0), GreaterThan(0, dim=1), window)
    baseline = from_pdstl(spec)
    assert isinstance(baseline, DetUntil)
    a, b = (0, math.inf) if window is None else window
    lookahead = a if math.isinf(b) else b
    expected = []
    for t in range(mean.shape[1] - lookahead):
        candidates = []
        for tau in range(t + a, mean.shape[1] if math.isinf(b) else t + b + 1):
            prefix = reduce_reference(mean[:, t : tau + 1, 0], beta, True, 1)
            candidates.append(
                reduce_reference(
                    torch.stack((prefix, mean[:, tau, 1]), dim=1),
                    beta,
                    True,
                    1,
                )
            )
        expected.append(
            reduce_reference(torch.stack(candidates, dim=1), beta, False, 1)
        )
    torch.testing.assert_close(
        baseline(mean, beta)[..., 0], torch.stack(expected, dim=1)
    )


def test_until_requires_left_at_the_witness():
    mean = torch.tensor([[[1.0, -2.0], [-3.0, 4.0]]])
    spec = Until(GreaterThan(0, 0), GreaterThan(0, 1), [1, 1])
    torch.testing.assert_close(
        from_pdstl(spec).robustness(mean), torch.tensor([-3.0])
    )


@pytest.mark.parametrize("beta", [None, 2.0])
def test_nested_windows_preserve_complete_origins(beta):
    mean = torch.tensor(
        [[[0.0], [2.0], [1.0], [-1.0], [3.0], [4.0]]], dtype=torch.float64
    )
    spec = Eventually(Always(GreaterThan(0), [0, 2]), [1, 2])
    inner = torch.stack(
        [
            reduce_reference(mean[:, t : t + 3, 0], beta, True, 1)
            for t in range(4)
        ],
        dim=1,
    )
    expected = torch.stack(
        [
            reduce_reference(inner[:, t + 1 : t + 3], beta, False, 1)
            for t in range(2)
        ],
        dim=1,
    )
    torch.testing.assert_close(from_pdstl(spec)(mean, beta)[..., 0], expected)
    assert (
        spec(
            create_gaussian_belief_trajectory(mean, torch.zeros_like(mean))
        ).shape[1]
        == 2
    )


def test_boolean_geometry_and_conversion_do_not_change_pdstl():
    mean = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    beliefs = create_gaussian_belief_trajectory(mean, torch.ones_like(mean))
    goal = InsideRectangle([1.5, 2.5], [-0.5, 0.5])
    avoid = OutsideRectangle([0.75, 1.25], [-0.5, 0.5])
    spec = And(Always(avoid, [1, 2]), Eventually(goal, [1, 2]))
    original = spec(beliefs).clone()
    state = {k: v.clone() for k, v in spec.state_dict().items()}
    auxiliary = from_pdstl(spec, normalization="std")
    torch.testing.assert_close(
        auxiliary.robustness(beliefs), torch.tensor([-0.25])
    )
    auxiliary.double()  # must not move/convert the source graph's buffers
    torch.testing.assert_close(spec(beliefs), original)
    for name, value in spec.state_dict().items():
        torch.testing.assert_close(value, state[name], check_dtype=True)


def test_implication_and_negation():
    mean = torch.tensor([[[2.0, -1.0], [-2.0, 3.0]]])
    left, right = GreaterThan(0, 0), GreaterThan(0, 1)
    torch.testing.assert_close(
        from_pdstl(Implies(left, right))(mean)[..., 0],
        torch.maximum(-mean[..., 0], mean[..., 1]),
    )
    torch.testing.assert_close(
        from_pdstl(Negation(Or(left, right)))(mean)[..., 0],
        -torch.maximum(mean[..., 0], mean[..., 1]),
    )


def test_halfspace_projection_and_zero_variance_gradients():
    mean = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]], requires_grad=True)
    covariance = torch.zeros(1, 2, 2, 2, requires_grad=True)
    beliefs = create_gaussian_belief_trajectory(mean, covariance)
    auxiliary = from_pdstl(
        HalfSpace([2.0, 1.0], 5.0), normalization="std", sigma_floor=0.5
    )
    rho = auxiliary(beliefs)
    torch.testing.assert_close(rho[..., 0], torch.tensor([[2.0, 0.0]]))
    rho.sum().backward()
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(covariance.grad).all()


def test_relative_lane_margin_uses_relative_variance():
    mean = torch.zeros(1, 3, 4)
    traffic = torch.tensor(
        [
            [
                [[3.0, 0.0, 0.0, 0.0]],
                [[4.0, 0.0, 0.0, 0.0]],
                [[5.0, 0.0, 0.0, 0.0]],
            ]
        ]
    )
    ego_cov = torch.eye(4).expand(1, 3, 4, 4)
    traffic_cov = (3 * torch.eye(4)).expand(1, 3, 1, 4, 4)
    beliefs = LaneBeliefTrajectory(
        mean, ego_cov, traffic, traffic_cov, ["car"]
    )
    spec = RelativeAxisInterval("car", 2, math.inf, dim=0)
    torch.testing.assert_close(
        from_pdstl(spec, normalization="std")(beliefs)[..., 0],
        torch.tensor([[0.5, 1.0, 1.5]]),
    )


def test_smooth_recurrent_gradient_matches_direct_window_gradient():
    mean = torch.tensor(
        [[[1.0], [2.0], [-1.0], [3.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    auxiliary = from_pdstl(Always(GreaterThan(0), [1, 3]))
    actual = auxiliary.robustness(mean, beta=2.0).sum()
    expected = reduce_reference(mean[:, 1:4, 0], 2.0, True, 1).sum()
    grad = torch.autograd.grad(actual, mean, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(expected, mean)[0]
    torch.testing.assert_close(grad, reference_grad)


def test_bad_inputs_fail_before_incomplete_or_unsupported_evaluation():
    with pytest.raises(TypeError, match="unsupported auxiliary"):
        from_pdstl(Predicate("p"))
    with pytest.raises(ValueError, match="available trace"):
        from_pdstl(Always(GreaterThan(0), [0, 3]))(torch.zeros(1, 3, 1))
    with pytest.raises(ValueError, match="Gaussian"):
        from_pdstl(GreaterThan(0), normalization="std")(torch.zeros(1, 3, 1))
    with pytest.raises(ValueError, match="beta"):
        from_pdstl(GreaterThan(0))(torch.zeros(1, 3, 1), beta=-1)


def test_two_sided_axis_margin():
    mean = torch.tensor([[[-2.0], [0.0], [3.0]]])
    torch.testing.assert_close(
        from_pdstl(AxisInterval(-1, 2))(mean)[..., 0],
        torch.tensor([[-1.0, 1.0, -1.0]]),
    )
