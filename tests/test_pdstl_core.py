"""Verification for the pdSTL core: Gaussian atom, trace assembly, Boolean and
temporal bounds, the finite-trace rule, and gradient flow."""

import numpy as np
import pytest
import torch
from scipy.stats import norm

from models.dynamics import GaussianBelief
from pdstl.base import (
    BeliefTrajectory,
    OnlineBeliefTrajectory,
    ProbabilityBelief,
    create_probability_belief_trajectory,
)
from pdstl.operators import (
    Always,
    And,
    Eventually,
    GreaterThan,
    Implies,
    LessThan,
    Negation,
    Or,
    Predicate,
    Until,
)


def supplied(*rows):
    """Trajectory of supplied bounds; each row is {event: (lower, upper)}."""
    return BeliefTrajectory(
        [
            ProbabilityBelief({k: torch.tensor([[lo, hi]]) for k, (lo, hi) in r.items()})
            for r in rows
        ]
    )


def scalar(values, event="p"):
    """Trajectory of exact supplied probabilities for one event."""
    return supplied(*[{event: (v, v)} for v in values])


def gaussian(mean, var):
    """Trajectory of exact Gaussian steps (collapsed bound = mean) from [T, D]
    mean and [T, D] or [T, D, D] var."""
    mean = torch.as_tensor(mean, dtype=torch.float64)
    var = torch.as_tensor(var, dtype=torch.float64)
    return BeliefTrajectory(
        [
            GaussianBelief(mean[t : t + 1], mean[t : t + 1], var[t : t + 1])
            for t in range(len(mean))
        ]
    )


# --- 1. Gaussian predicate -------------------------------------------------


def test_gaussian_matches_the_analytic_probability():
    mean, var = [[45.0], [55.0], [50.0]], [[4.0], [4.0], [9.0]]
    traj = gaussian(mean, var)

    got = GreaterThan(50.0)(traj)[0].numpy()
    exact = 1.0 - norm.cdf((50.0 - np.array(mean)[:, 0]) / np.sqrt(np.array(var)[:, 0]))

    np.testing.assert_allclose(got[:, 0], exact, atol=1e-9)
    np.testing.assert_allclose(got[:, 1], exact, atol=1e-9)  # exact -> [p, p]


def test_less_than_is_the_complement():
    traj = gaussian([[45.0], [55.0]], [[4.0], [4.0]])

    gt = GreaterThan(50.0)(traj)[0].numpy()
    lt = LessThan(50.0)(traj)[0].numpy()

    np.testing.assert_allclose(lt[:, 0], 1.0 - gt[:, 0], atol=1e-9)


@pytest.mark.parametrize(
    "m, c, expect_ge",
    [(3.0, 1.0, 1.0), (1.0, 3.0, 0.0), (2.0, 2.0, 1.0)],  # equality is inclusive
)
def test_zero_variance_is_an_inclusive_deterministic_comparison(m, c, expect_ge):
    traj = gaussian([[m]], [[0.0]])

    assert GreaterThan(c)(traj)[0, 0, 0].item() == expect_ge
    assert LessThan(c)(traj)[0, 0, 0].item() == (1.0 if m <= c else 0.0)


def test_negative_variance_is_rejected():
    with pytest.raises(ValueError, match="covariance must be non-negative"):
        gaussian([[1.0]], [[-2.0]])


def test_full_covariance_selects_the_requested_marginal():
    mean = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    covariance = torch.tensor([[[4.0, 3.0], [3.0, 9.0]]], dtype=torch.float64)
    trajectory = BeliefTrajectory([GaussianBelief(mean, mean, covariance)])

    got = GreaterThan(0.0, dim=1)(trajectory)[0, 0, 0].item()

    assert got == pytest.approx(norm.cdf(2.0 / 3.0), abs=1e-9)


@pytest.mark.parametrize(
    "lower,upper,covariance,message",
    [
        (torch.zeros(2), torch.zeros(2), torch.ones(2), r"lower must have shape \[B,D\]"),
        (torch.zeros(2, 2), torch.zeros(2, 2), torch.ones(3, 2), "diagonal covariance"),
        (torch.zeros(2, 2), torch.zeros(2, 2), torch.ones(2, 2, 3), r"full covariance must have shape \[B,D,D\]"),
        (torch.zeros(2, 2), torch.zeros(2, 2), torch.ones(2, 3, 3), r"full covariance must have shape \[B,D,D\]"),
    ],
)
def test_gaussian_belief_rejects_invalid_shapes(lower, upper, covariance, message):
    with pytest.raises(ValueError, match=message):
        GaussianBelief(lower, upper, covariance)


def test_gaussian_belief_rejects_an_invalid_predicate_dimension():
    trajectory = BeliefTrajectory(
        [GaussianBelief(torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, 2))]
    )

    with pytest.raises(ValueError, match="dimension 2 is outside"):
        GreaterThan(0.0, dim=2)(trajectory)


def test_unsupported_event_fails_clearly():
    with pytest.raises(ValueError, match="cannot evaluate"):
        Predicate("safe")(gaussian([[1.0]], [[1.0]]))


# --- 2. Trace assembly and validation --------------------------------------


def test_supplied_bounds_assemble_into_a_trace():
    traj = scalar([0.2, 0.5, 0.9])

    trace = Predicate("p")(traj)

    assert trace.shape == (1, 3, 2)
    np.testing.assert_allclose(trace[0, :, 0].numpy(), [0.2, 0.5, 0.9])


def test_events_are_served_by_name():
    traj = supplied({"safe": (0.9, 0.97), "reach": (0.1, 0.42)})

    np.testing.assert_allclose(Predicate("safe")(traj)[0, 0].numpy(), [0.9, 0.97])
    with pytest.raises(ValueError, match="no supplied bounds for event 'other'"):
        Predicate("other")(traj)


@pytest.mark.parametrize(
    "bad", [(0.8, 0.3), (float("nan"), 0.5), (-0.2, 0.5), (0.5, 1.4)]
)
def test_malformed_bounds_are_rejected_naming_the_step(bad):
    traj = supplied({"p": (0.4, 0.6)}, {"p": bad})

    with pytest.raises(ValueError, match="at step 1"):
        Predicate("p")(traj)


def test_validation_can_be_disabled():
    traj = supplied({"p": (0.8, 0.3)})

    np.testing.assert_allclose(
        Predicate("p")(traj, validate=False)[0, 0].numpy(), [0.8, 0.3]
    )


def test_appended_container_matches_the_offline_one():
    values = [0.9, 0.2, 0.7, 0.4, 0.6]
    spec = Always(Predicate("p"), interval=[0, 1])

    offline = spec(scalar(values))
    online = OnlineBeliefTrajectory()
    for belief in scalar(values):
        online.append(belief)

    torch.testing.assert_close(spec(online), offline)


# --- 3. Boolean bounds -----------------------------------------------------


def test_and_uses_frechet_not_the_product():
    # P(A)=0.9, P(B)=0.1 admits P(A and B)=0, so the product 0.09 is unsound.
    traj = supplied({"a": (0.9, 0.9), "b": (0.1, 0.1)})

    lower, upper = And(Predicate("a"), Predicate("b"))(traj)[0, 0].tolist()

    assert lower == pytest.approx(0.0)
    assert upper == pytest.approx(0.1)


def test_boolean_bounds_on_hand_calculated_values():
    traj = supplied({"a": (0.7, 0.8), "b": (0.6, 0.9)})
    a, b = Predicate("a"), Predicate("b")

    np.testing.assert_allclose(
        And(a, b)(traj)[0, 0].numpy(), [0.7 + 0.6 - 1, 0.8], atol=1e-6
    )
    np.testing.assert_allclose(
        Negation(a)(traj)[0, 0].numpy(), [1 - 0.8, 1 - 0.7], atol=1e-6
    )

    # Or, chosen so the upper endpoint does not saturate
    small = supplied({"a": (0.2, 0.3), "b": (0.1, 0.4)})
    np.testing.assert_allclose(
        Or(a, b)(small)[0, 0].numpy(), [0.2, 0.3 + 0.4], atol=1e-6
    )
    # and a case where it does
    assert Or(a, b)(traj)[0, 0, 1].item() == pytest.approx(1.0)


def test_nested_pointwise_boolean_composes_frechet_at_each_level():
    """And(And(a,b), Or(b,c)), evaluated standalone (not merely as a side
    effect of being wrapped in a temporal operator): is_pointwise propagates
    correctly through two levels of nesting, and the numeric result matches
    hand-computed nested Frechet bounds."""
    traj = supplied({"a": (0.7, 0.8), "b": (0.6, 0.9), "c": (0.3, 0.5)})
    a, b, c = Predicate("a"), Predicate("b"), Predicate("c")

    inner_and = And(a, b)
    inner_or = Or(b, c)
    nested = And(inner_and, inner_or)

    assert inner_and.is_pointwise and inner_or.is_pointwise and nested.is_pointwise

    # inner_and = [max(0, 0.7+0.6-1), min(0.8,0.9)] = [0.3, 0.8]
    # inner_or  = [max(0.6,0.3), min(1, 0.9+0.5)]   = [0.6, 1.0]
    # nested    = [max(0, 0.3+0.6-1), min(0.8,1.0)] = [0.0, 0.8]
    np.testing.assert_allclose(inner_and(traj)[0, 0].numpy(), [0.3, 0.8], atol=1e-6)
    np.testing.assert_allclose(inner_or(traj)[0, 0].numpy(), [0.6, 1.0], atol=1e-6)
    np.testing.assert_allclose(nested(traj)[0, 0].numpy(), [0.0, 0.8], atol=1e-6)


def test_implies_is_negation_then_or():
    traj = supplied({"a": (0.7, 0.8), "b": (0.6, 0.9)})
    a, b = Predicate("a"), Predicate("b")

    torch.testing.assert_close(Implies(a, b)(traj), Or(Negation(a), b)(traj))


def test_boolean_bounds_are_unchanged_by_scale():
    traj = supplied({"a": (0.7, 0.8), "b": (0.6, 0.9)})
    a, b = Predicate("a"), Predicate("b")

    for spec in (And(a, b), Or(a, b), Implies(a, b), Negation(a)):
        torch.testing.assert_close(spec(traj, scale=5.0), spec(traj, scale=-1))


def test_pointwise_flag_tracks_the_formula_kind():
    a, b = Predicate("a"), Predicate("b")
    windowed = Always(a, interval=[0, 1])

    assert a.is_pointwise
    assert Negation(a).is_pointwise
    assert And(a, b).is_pointwise
    assert Implies(a, b).is_pointwise

    assert not windowed.is_pointwise
    assert not Until(a, b).is_pointwise


def test_boolean_over_temporal_is_rejected_at_construction():
    a, b = Predicate("a"), Predicate("b")
    windowed = Always(a, interval=[0, 1])

    for build in (
        lambda: And(windowed, b),
        lambda: Or(a, windowed),
        lambda: Implies(windowed, b),
        lambda: Implies(a, windowed),
        lambda: Negation(windowed),
    ):
        with pytest.raises(ValueError, match="only accepts pointwise"):
            build()


# --- 4. Always / Eventually ------------------------------------------------


@pytest.mark.parametrize("interval", [[0, 1], [1, 2], [2, 3]])
def test_always_and_eventually_match_direct_window_reductions(interval):
    # Distinct lower and upper values, so each endpoint is reduced on its own.
    rng = np.random.default_rng(0)
    lows = rng.uniform(0.0, 0.5, size=9).tolist()
    highs = rng.uniform(0.5, 1.0, size=9).tolist()
    traj = supplied(*[{"p": (lo, hi)} for lo, hi in zip(lows, highs)])
    a, b = interval
    n = len(lows) - b

    got_min = Always(Predicate("p"), interval=interval)(traj)[0].tolist()
    got_max = Eventually(Predicate("p"), interval=interval)(traj)[0].tolist()

    assert len(got_min) == n and len(got_max) == n
    np.testing.assert_allclose(
        got_min,
        [
            [min(lows[t + a : t + b + 1]), min(highs[t + a : t + b + 1])]
            for t in range(n)
        ],
    )
    np.testing.assert_allclose(
        got_max,
        [
            [max(lows[t + a : t + b + 1]), max(highs[t + a : t + b + 1])]
            for t in range(n)
        ],
    )


def test_smooth_temporal_output_is_differentiable():
    p = torch.tensor([[0.2, 0.8]], requires_grad=True)
    values = [p, p * 0.5, p * 0.75, p * 0.9]
    traj = BeliefTrajectory([ProbabilityBelief({"p": v}) for v in values])

    Always(Predicate("p"), interval=[0, 2])(traj, scale=3.0)[0, 0, 0].backward()

    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert p.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "operator,smooth_reduce",
    [
        (
            Always,
            lambda window, scale: -torch.logsumexp(-window * scale, dim=0)
            / scale,
        ),
        (
            Eventually,
            lambda window, scale: torch.logsumexp(window * scale, dim=0) / scale,
        ),
    ],
)
def test_smooth_always_and_eventually_match_their_endpointwise_surrogates(
    operator, smooth_reduce
):
    rows = [
        {"p": (0.2, 0.7)},
        {"p": (0.5, 0.8)},
        {"p": (0.1, 0.9)},
        {"p": (0.6, 0.95)},
    ]
    trajectory = supplied(*rows)
    pointwise = Predicate("p")(trajectory)[0]
    scale = 3.0

    got = operator(Predicate("p"), interval=[0, 2])(
        trajectory, scale=scale
    )[0]
    expected = torch.stack(
        [smooth_reduce(pointwise[t : t + 3], scale) for t in range(2)]
    )

    torch.testing.assert_close(got, expected)


def test_nested_temporal_lookahead_composes():
    values = [0.9, 0.2, 0.7, 0.4, 0.6, 0.8, 0.3, 0.5]
    traj = scalar(values)

    spec = Always(Eventually(Predicate("p"), interval=[0, 2]), interval=[0, 3])
    got = spec(traj)[0, :, 0].tolist()

    n = len(values) - 2 - 3
    inner = [max(values[t : t + 3]) for t in range(len(values) - 2)]
    assert len(got) == n
    np.testing.assert_allclose(got, [min(inner[t : t + 4]) for t in range(n)])


def test_nested_eventually_of_always_composes():
    values = [0.9, 0.2, 0.7, 0.4, 0.6, 0.8, 0.3, 0.5]
    traj = scalar(values)

    spec = Eventually(Always(Predicate("p"), interval=[0, 1]), interval=[0, 2])
    trace = spec(traj)

    # lookahead adds: the inner window costs 1 step, the outer 2 more
    n = len(values) - 1 - 2
    assert trace.shape == (1, n, 2)
    inner = [min(values[t : t + 2]) for t in range(len(values) - 1)]
    np.testing.assert_allclose(
        trace[0, :, 0].tolist(), [max(inner[t : t + 3]) for t in range(n)]
    )


def test_suffix_windows_cover_whatever_trace_remains():
    values = [0.9, 0.2, 0.7, 0.4]
    traj = scalar(values)
    ref = [min(values[t:]) for t in range(len(values))]

    for spec in (
        Always(Predicate("p")),
        Always(Predicate("p"), interval=[0, np.inf]),  # used to raise OverflowError
    ):
        got = spec(traj)[0, :, 0].tolist()
        assert len(got) == len(values)
        np.testing.assert_allclose(got, ref)


def test_unbounded_start_offsets_the_suffix():
    values = [0.9, 0.2, 0.7, 0.4, 0.6]
    got = Always(Predicate("p"), interval=[2, np.inf])(scalar(values))[0, :, 0].tolist()

    assert len(got) == len(values) - 2
    np.testing.assert_allclose(got, [min(values[t + 2 :]) for t in range(len(values) - 2)])


# --- 5. Until --------------------------------------------------------------


def until_reference(left, right, a, b):
    """Inclusive-prefix Until, written independently of the implementation."""
    T = len(left)
    out = []
    for t in range(T - (b if np.isfinite(b) else a)):
        best = None
        hi = min(t + (T - 1 if not np.isfinite(b) else b), T - 1)
        for tau in range(t + a, hi + 1):
            prefix = min(left[t : tau + 1])  # inclusive of tau
            cand = min(prefix, right[tau])
            best = cand if best is None else max(best, cand)
        out.append(best)
    return out


UNTIL_LEFT = [0.9, 0.8, 0.4, 0.7, 0.95, 0.6]
UNTIL_RIGHT = [0.1, 0.3, 0.9, 0.2, 0.5, 0.85]


def until_trajectory():
    return supplied(
        *[{"l": (lo, lo), "r": (hi, hi)} for lo, hi in zip(UNTIL_LEFT, UNTIL_RIGHT)]
    )


@pytest.mark.parametrize("interval", [[0, 0], [0, 1], [1, 2]])
def test_until_matches_the_inclusive_reference(interval):
    got = Until(Predicate("l"), Predicate("r"), interval=interval)(until_trajectory())

    np.testing.assert_allclose(
        got[0, :, 0].tolist(),
        until_reference(UNTIL_LEFT, UNTIL_RIGHT, *interval),
        atol=1e-6,
    )


def test_until_smooth_approaches_the_exact_reduction():
    spec = Until(Predicate("l"), Predicate("r"), interval=[0, 2])
    traj = until_trajectory()

    exact = spec(traj, scale=-1)
    smooth = spec(traj, scale=400.0)

    torch.testing.assert_close(smooth, exact, atol=5e-2, rtol=0)
    assert not torch.equal(smooth, exact)  # a surrogate, not the same tensor


def test_until_reduces_both_endpoints_with_the_exact_and_smooth_max_min():
    trajectory = supplied(
        {"l": (0.7, 0.9), "r": (0.1, 0.3)},
        {"l": (0.4, 0.8), "r": (0.6, 0.75)},
        {"l": (0.8, 0.95), "r": (0.5, 0.85)},
    )
    formula = Until(Predicate("l"), Predicate("r"), interval=[0, 2])

    exact = formula(trajectory, scale=-1)[0, 0]
    torch.testing.assert_close(exact, torch.tensor([0.4, 0.8]))

    left = Predicate("l")(trajectory)[0]
    right = Predicate("r")(trajectory)[0]
    scale = 4.0
    candidates = []
    for witness in range(3):
        prefix = -torch.logsumexp(
            -left[: witness + 1] * scale, dim=0
        ) / scale
        candidates.append(
            -torch.logsumexp(
                -torch.stack((prefix, right[witness])) * scale, dim=0
            )
            / scale
        )
    expected_smooth = torch.logsumexp(
        torch.stack(candidates) * scale, dim=0
    ) / scale

    torch.testing.assert_close(
        formula(trajectory, scale=scale)[0, 0], expected_smooth
    )


def test_until_at_zero_zero_combines_both_operands_now():
    traj = supplied({"l": (0.9, 0.9), "r": (0.4, 0.4)})

    lower, upper = Until(Predicate("l"), Predicate("r"), interval=[0, 0])(traj)[0, 0]

    # endpointwise min of prefix and witness, not a Frechet conjunction
    assert lower.item() == pytest.approx(min(0.9, 0.4))
    assert upper.item() == pytest.approx(min(0.9, 0.4))


# --- 6. Finite trace -------------------------------------------------------


def test_insufficient_prediction_length_is_rejected():
    traj = scalar([0.5, 0.5, 0.5])

    with pytest.raises(ValueError, match="needs 5 steps"):
        Always(Predicate("p"), interval=[0, 4])(traj)
    with pytest.raises(ValueError, match="needs 4 steps"):
        Until(Predicate("p"), Predicate("p"), interval=[0, 3])(traj)


def test_temporal_operator_accepts_a_nested_pointwise_boolean():
    traj = supplied(
        {"a": (0.7, 0.8), "b": (0.6, 0.9)},
        {"a": (0.9, 0.95), "b": (0.8, 0.85)},
        {"a": (0.5, 0.7), "b": (0.7, 0.9)},
    )
    pointwise = And(Predicate("a"), Or(Predicate("a"), Predicate("b")))

    trace = Always(pointwise, interval=[0, 1])(traj)

    assert trace.shape == (1, 2, 2)
    torch.testing.assert_close(trace[0, 0], torch.tensor([0.4, 0.8]))


# --- 7. dtype, device and gradients ----------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved_through_a_temporal_formula(dtype):
    mean = torch.tensor([[45.0], [55.0], [60.0]], dtype=dtype)
    var = torch.full((3, 1), 4.0, dtype=dtype)
    traj = BeliefTrajectory(
        [GaussianBelief(mean[t : t + 1], mean[t : t + 1], var[t : t + 1]) for t in range(3)]
    )

    assert Always(GreaterThan(50.0), interval=[0, 1])(traj).dtype == dtype


def test_gradients_reach_the_mean_through_a_composed_formula():
    mean = torch.tensor([[45.0], [55.0], [60.0], [48.0]], requires_grad=True)
    var = torch.full((4, 1), 4.0)
    traj = BeliefTrajectory(
        [GaussianBelief(mean[t : t + 1], mean[t : t + 1], var[t : t + 1]) for t in range(4)]
    )
    spec = Always(
        And(GreaterThan(50.0), LessThan(70.0)), interval=[0, 1]
    )

    spec(traj)[0, 0, 0].backward()

    assert mean.grad is not None
    assert torch.isfinite(mean.grad).all()


def test_smooth_formula_passes_a_finite_difference_gradcheck():
    def f(mean):
        var = torch.full_like(mean, 4.0)
        traj = BeliefTrajectory(
            [
                GaussianBelief(mean[t : t + 1], mean[t : t + 1], var[t : t + 1])
                for t in range(len(mean))
            ]
        )
        spec = Always(GreaterThan(50.0), interval=[0, 1])
        return spec(traj, scale=2.0)[0, 0, 0]

    # well separated from ties, so min/logsumexp is smooth and nonsingular
    mean = torch.tensor([[47.0], [53.0], [58.0]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (mean,), eps=1e-6, atol=1e-4)


def test_supplied_bounds_keep_their_graph():
    p = torch.tensor([[0.3, 0.7]], requires_grad=True)
    traj = BeliefTrajectory([ProbabilityBelief({"p": p})] * 3)

    Always(Predicate("p"), interval=[0, 1])(traj)[0, 0, 0].backward()

    assert p.grad is not None and p.grad.abs().sum() > 0


# --- 8. Differentiable control feeding a formula ---------------------------


def test_control_to_mean_to_formula_is_differentiable():
    u = torch.tensor([1.0, 1.0, 1.0, 1.0], requires_grad=True)
    mean = torch.cumsum(u, dim=0).unsqueeze(-1)  # [T, 1]
    var = torch.full_like(mean, 0.25)
    traj = BeliefTrajectory(
        [
            GaussianBelief(mean[t : t + 1], mean[t : t + 1], var[t : t + 1])
            for t in range(len(u))
        ]
    )
    spec = Eventually(GreaterThan(2.5), interval=[0, 2])

    spec(traj)[0, 0, 0].backward()

    assert u.grad is not None
    assert torch.isfinite(u.grad).all()
    assert u.grad.abs().sum() > 0


# --- 9. Supplied probability-bound trajectories -----------------------------


def test_supplied_bounds_factory_keys_every_step_by_the_predicate_event():
    predicate = GreaterThan(50.0)
    bounds = [[0.2, 0.4], [0.6, 0.8], [0.5, 0.9]]

    traj = create_probability_belief_trajectory(predicate, bounds)

    assert len(traj) == 3
    assert set(traj[0].bounds) == {predicate.name}
    torch.testing.assert_close(
        predicate(traj)[0], torch.tensor(bounds, dtype=torch.float32)
    )


@pytest.mark.parametrize(
    "bounds,message",
    [
        ([0.2, 0.4], r"\[T, 2\]"),
        ([[0.2, 0.4, 0.6]], r"\[T, 2\]"),
        (np.zeros((0, 2)), "at least one step"),
        ([[float("nan"), 0.4]], "malformed"),
        ([[0.5, 1.5]], "malformed"),
        ([[0.8, 0.3]], "malformed"),
        ([[-0.2, 0.4]], "malformed"),
    ],
)
def test_supplied_bounds_factory_rejects_malformed_input(bounds, message):
    with pytest.raises(ValueError, match=message):
        create_probability_belief_trajectory(GreaterThan(50.0), bounds)


def test_supplied_bounds_factory_preserves_dtype_device_and_gradients():
    predicate = GreaterThan(50.0)
    bounds = torch.tensor(
        [[0.2, 0.4], [0.6, 0.8], [0.5, 0.9]],
        dtype=torch.float64,
        requires_grad=True,
    )

    traj = create_probability_belief_trajectory(predicate, bounds)
    temporal = Always(predicate, interval=[0, 1])(traj, scale=-1)

    assert temporal.dtype == torch.float64
    assert temporal.device == bounds.device
    temporal.sum().backward()
    assert bounds.grad is not None
    assert torch.isfinite(bounds.grad).all()
    assert bounds.grad.abs().sum() > 0
