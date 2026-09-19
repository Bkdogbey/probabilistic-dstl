"""Verification for the pdSTL core: Gaussian atom, trace assembly, Boolean and
temporal bounds, the finite-trace rule, and gradient flow."""

import numpy as np
import pytest
import torch
from scipy.stats import norm

from models.beliefs import GaussianBelief, ProbabilityBelief
from models.rollouts import create_probability_belief_trajectory
from pdstl.base import (
    BeliefTrajectory,
    OnlineBeliefTrajectory,
)
from pdstl.operators import (
    Always,
    And,
    Eventually,
    Implies,
    Maxish,
    Minish,
    Negation,
    Or,
    Predicate,
    Until,
)
from pdstl.predicates import GreaterThan, LessThan


def supplied(*rows):
    """Trajectory of supplied bounds; each row is {event: (lower, upper)}."""
    return BeliefTrajectory(
        [
            ProbabilityBelief(
                {k: torch.tensor([[lo, hi]]) for k, (lo, hi) in r.items()}
            )
            for r in rows
        ]
    )


def scalar(values, event="p"):
    """Trajectory of exact supplied probabilities for one event."""
    return supplied(*[{event: (v, v)} for v in values])


def gaussian(mean, var):
    """Trajectory of precise Gaussian steps from [T, D] mean and [T, D] or
    [T, D, D] var."""
    mean = torch.as_tensor(mean, dtype=torch.float64)
    var = torch.as_tensor(var, dtype=torch.float64)
    return BeliefTrajectory(
        [
            GaussianBelief(mean[t : t + 1], var[t : t + 1])
            for t in range(len(mean))
        ]
    )


# --- 1. Gaussian predicate -------------------------------------------------


def test_gaussian_matches_the_analytic_probability():
    mean, var = [[45.0], [55.0], [50.0]], [[4.0], [4.0], [9.0]]
    traj = gaussian(mean, var)

    got = GreaterThan(50.0)(traj)[0].numpy()
    exact = 1.0 - norm.cdf(
        (50.0 - np.array(mean)[:, 0]) / np.sqrt(np.array(var)[:, 0])
    )

    np.testing.assert_allclose(got[:, 0], exact, atol=1e-9)
    np.testing.assert_allclose(got[:, 1], exact, atol=1e-9)  # exact -> [p, p]


def test_less_than_is_the_complement():
    traj = gaussian([[45.0], [55.0]], [[4.0], [4.0]])

    gt = GreaterThan(50.0)(traj)[0].numpy()
    lt = LessThan(50.0)(traj)[0].numpy()

    np.testing.assert_allclose(lt[:, 0], 1.0 - gt[:, 0], atol=1e-9)


@pytest.mark.parametrize(
    "m, c, expect_ge",
    [
        (3.0, 1.0, 1.0),
        (1.0, 3.0, 0.0),
        (2.0, 2.0, 1.0),
    ],  # equality is inclusive
)
def test_zero_variance_is_an_inclusive_deterministic_comparison(
    m, c, expect_ge
):
    traj = gaussian([[m]], [[0.0]])

    assert GreaterThan(c)(traj)[0, 0, 0].item() == expect_ge
    assert LessThan(c)(traj)[0, 0, 0].item() == (1.0 if m <= c else 0.0)


def test_negative_variance_is_rejected():
    with pytest.raises(ValueError, match="positive semi-definite"):
        gaussian([[1.0]], [[-2.0]])


def test_full_covariance_selects_the_requested_marginal():
    mean = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    covariance = torch.tensor([[[4.0, 3.0], [3.0, 9.0]]], dtype=torch.float64)
    trajectory = BeliefTrajectory([GaussianBelief(mean, covariance)])

    got = GreaterThan(0.0, dim=1)(trajectory)[0, 0, 0].item()

    assert got == pytest.approx(norm.cdf(2.0 / 3.0), abs=1e-9)


@pytest.mark.parametrize(
    "mean,covariance,message",
    [
        (torch.zeros(2), torch.ones(2), "mean must have shape"),
        (torch.zeros(2, 2), torch.ones(3, 2), "covariance must be"),
        (torch.zeros(2, 2), torch.ones(2, 2, 3), "covariance must be"),
        (torch.zeros(2, 2), torch.ones(2, 3, 3), "covariance must be"),
    ],
)
def test_gaussian_belief_rejects_invalid_shapes(mean, covariance, message):
    with pytest.raises(ValueError, match=message):
        GaussianBelief(mean, covariance)


def test_gaussian_belief_rejects_an_invalid_predicate_dimension():
    trajectory = BeliefTrajectory(
        [GaussianBelief(torch.zeros(1, 2), torch.ones(1, 2))]
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

    np.testing.assert_allclose(
        Predicate("safe")(traj)[0, 0].numpy(), [0.9, 0.97]
    )
    with pytest.raises(
        ValueError, match="no supplied bounds for event 'other'"
    ):
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

    assert (
        inner_and.is_pointwise
        and inner_or.is_pointwise
        and nested.is_pointwise
    )

    # inner_and = [max(0, 0.7+0.6-1), min(0.8,0.9)] = [0.3, 0.8]
    # inner_or  = [max(0.6,0.3), min(1, 0.9+0.5)]   = [0.6, 1.0]
    # nested    = [max(0, 0.3+0.6-1), min(0.8,1.0)] = [0.0, 0.8]
    np.testing.assert_allclose(
        inner_and(traj)[0, 0].numpy(), [0.3, 0.8], atol=1e-6
    )
    np.testing.assert_allclose(
        inner_or(traj)[0, 0].numpy(), [0.6, 1.0], atol=1e-6
    )
    np.testing.assert_allclose(
        nested(traj)[0, 0].numpy(), [0.0, 0.8], atol=1e-6
    )


def test_implies_is_negation_then_or():
    traj = supplied({"a": (0.7, 0.8), "b": (0.6, 0.9)})
    a, b = Predicate("a"), Predicate("b")

    torch.testing.assert_close(Implies(a, b)(traj), Or(Negation(a), b)(traj))


def test_both_conjunction_endpoints_are_smoothed():
    """Negation swaps the endpoints, so a hard upper would leave negated conjunctions
    -- every OutsideRectangle -- only partly differentiable."""
    traj = supplied({"a": (0.7, 0.8), "b": (0.6, 0.9)})
    a, b = Predicate("a"), Predicate("b")

    exact = And(a, b)(traj, beta=None)[0, 0]
    smooth = And(a, b)(traj, beta=5.0)[0, 0]

    torch.testing.assert_close(exact, torch.tensor([0.3, 0.8]))
    torch.testing.assert_close(
        smooth[0], torch.nn.functional.softplus(torch.tensor(0.3), beta=5.0)
    )
    assert smooth[1] != exact[1], "the upper endpoint must be a smooth min"


def test_disjunction_and_implication_are_smoothed():
    """Implies is built from Or, so a hard Or would make implications partly differentiable."""
    traj = supplied({"a": (0.7, 0.8), "b": (0.6, 0.9)})
    a, b = Predicate("a"), Predicate("b")

    for spec in (Or(a, b), Implies(a, b)):
        exact, smooth = spec(traj, beta=None)[0, 0], spec(traj, beta=5.0)[0, 0]
        assert not torch.allclose(exact, smooth), f"{spec} is not smoothed"


def test_negation_is_exact_in_both_modes():
    """Negation only swaps and complements, so it introduces no smoothing of its own."""
    traj = supplied({"a": (0.7, 0.8), "b": (0.6, 0.9)})
    spec = Negation(Predicate("a"))
    torch.testing.assert_close(spec(traj, beta=5.0), spec(traj, beta=None))


def test_smoothed_operators_converge_to_the_exact_values_as_beta_grows():
    traj = supplied({"a": (0.7, 0.8), "b": (0.6, 0.9)})
    a, b = Predicate("a"), Predicate("b")

    for spec in (And(a, b), Or(a, b), Implies(a, b)):
        exact = spec(traj, beta=None)[0, 0]
        errors = [
            float((spec(traj, beta=beta)[0, 0] - exact).abs().max())
            for beta in (5.0, 50.0, 500.0)
        ]
        assert errors == sorted(errors, reverse=True), (
            f"{spec} does not converge"
        )
        assert errors[-1] < 1e-2


def test_smooth_and_escapes_the_exact_zero_clamp():
    lower = torch.tensor([[0.3], [0.4]], requires_grad=True)  # L1 + L2 - 1 < 0
    upper = torch.ones(2, 1)

    def conjunction_lower(beta):
        traj = BeliefTrajectory(
            [
                ProbabilityBelief(
                    {
                        "a": torch.stack([lower[0], upper[0]], dim=-1),
                        "b": torch.stack([lower[1], upper[1]], dim=-1),
                    }
                )
            ]
        )
        return And(Predicate("a"), Predicate("b"))(traj, beta=beta)[0, 0, 0]

    exact = conjunction_lower(None)
    (exact_grad,) = torch.autograd.grad(exact, lower)
    smooth = conjunction_lower(10.0)
    (smooth_grad,) = torch.autograd.grad(smooth, lower)

    assert exact.item() == 0.0 and exact_grad.abs().sum() == 0
    assert (
        smooth.item() > 0
        and torch.isfinite(smooth_grad).all()
        and (smooth_grad > 0).all()
    )


def test_smooth_min_and_max_converge_to_the_exact_values_as_beta_grows():
    x = torch.tensor([[0.2, 0.9, 0.5, 0.7]])
    errors = []
    for beta in (1.0, 10.0, 100.0, 1000.0):
        smooth_min, smooth_max = (
            Minish()(x, beta)[0, 0],
            Maxish()(x, beta)[0, 0],
        )
        assert (
            x.min() <= smooth_min <= x.mean() <= smooth_max <= x.max()
        )  # normalized
        errors.append(max(smooth_min - x.min(), x.max() - smooth_max).item())
        assert errors[-1] <= np.log(4) / beta + 1e-6
    assert errors == sorted(errors, reverse=True)
    assert (
        Minish()(x, None)[0, 0] == x.min()
        and Maxish()(x, None)[0, 0] == x.max()
    )


def test_pointwise_flag_tracks_the_formula_kind():
    a, b = Predicate("a"), Predicate("b")
    windowed = Always(a, interval=[0, 1])

    assert a.is_pointwise
    assert Negation(a).is_pointwise
    assert And(a, b).is_pointwise
    assert Implies(a, b).is_pointwise

    assert not windowed.is_pointwise
    assert not Until(a, b).is_pointwise
    assert not And(windowed, b).is_pointwise
    assert not Negation(windowed).is_pointwise
    assert not Implies(a, windowed).is_pointwise


A_TRACE = [(0.9, 0.95), (0.7, 0.8), (0.85, 0.9), (0.6, 0.75), (0.95, 1.0)]
B_TRACE = [(0.2, 0.4), (0.5, 0.6), (0.3, 0.35), (0.8, 0.9), (0.1, 0.2)]


def ab_trajectory():
    return supplied(*[{"a": a, "b": b} for a, b in zip(A_TRACE, B_TRACE)])


def window_ref(trace, a, b, reduce):
    return [
        [
            reduce(x[0] for x in trace[t + a : t + b + 1]),
            reduce(x[1] for x in trace[t + a : t + b + 1]),
        ]
        for t in range(len(trace) - b)
    ]


def and_ref(x, y):
    return [max(0.0, x[0] + y[0] - 1.0), min(x[1], y[1])]


def or_ref(x, y):
    return [max(x[0], y[0]), min(1.0, x[1] + y[1])]


def not_ref(x):
    return [1.0 - x[1], 1.0 - x[0]]


def test_boolean_combines_temporal_children_at_the_same_origins():
    a, b = Predicate("a"), Predicate("b")
    always_a = Always(a, interval=[0, 2])  # 3 origins
    eventually_b = Eventually(b, interval=[0, 1])  # 4 origins
    alw = window_ref(A_TRACE, 0, 2, min)
    ev = window_ref(B_TRACE, 0, 1, max)

    got_and = And(always_a, eventually_b)(ab_trajectory())[0]
    got_or = Or(always_a, eventually_b)(ab_trajectory())[0]

    assert got_and.shape == got_or.shape == (3, 2)
    np.testing.assert_allclose(
        got_and.tolist(), [and_ref(alw[t], ev[t]) for t in range(3)], atol=1e-6
    )
    np.testing.assert_allclose(
        got_or.tolist(), [or_ref(alw[t], ev[t]) for t in range(3)], atol=1e-6
    )


def test_boolean_combines_temporal_and_atomic_children():
    a, b = Predicate("a"), Predicate("b")
    ev = window_ref(A_TRACE, 0, 1, max)

    got = And(Eventually(a, interval=[0, 1]), b)(ab_trajectory())[0]

    np.testing.assert_allclose(
        got.tolist(), [and_ref(ev[t], B_TRACE[t]) for t in range(4)], atol=1e-6
    )


def test_negation_of_a_temporal_formula_is_its_dual():
    a = Predicate("a")
    traj = ab_trajectory()

    got = Negation(Always(a, interval=[0, 2]))(traj)

    np.testing.assert_allclose(
        got[0].tolist(),
        [not_ref(x) for x in window_ref(A_TRACE, 0, 2, min)],
        atol=1e-6,
    )
    torch.testing.assert_close(
        got, Eventually(Negation(a), interval=[0, 2])(traj)
    )


def test_implies_over_temporal_children_is_negation_then_or():
    a, b = Predicate("a"), Predicate("b")
    left, right = Always(a, interval=[0, 1]), Eventually(b, interval=[0, 2])
    traj = ab_trajectory()

    torch.testing.assert_close(
        Implies(left, right)(traj), Or(Negation(left), right)(traj)
    )


def test_nested_boolean_and_temporal_composition_matches_the_reference():
    a, b = Predicate("a"), Predicate("b")
    inner = Or(
        Negation(Always(a, interval=[0, 2])),
        And(Eventually(b, interval=[0, 1]), a),
    )
    formula = Eventually(inner, interval=[0, 1])

    alw = window_ref(A_TRACE, 0, 2, min)
    ev = window_ref(B_TRACE, 0, 1, max)
    inner_ref = [
        or_ref(not_ref(alw[t]), and_ref(ev[t], A_TRACE[t])) for t in range(3)
    ]

    got = formula(ab_trajectory())[0]

    np.testing.assert_allclose(
        got.tolist(), window_ref(inner_ref, 0, 1, max), atol=1e-6
    )


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

    Always(Predicate("p"), interval=[0, 2])(traj, beta=3.0)[0, 0, 0].backward()

    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert p.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "operator,smooth_reduce",
    [
        (
            Always,
            lambda window, beta: (
                -(torch.logsumexp(-window * beta, dim=0) - np.log(len(window)))
                / beta
            ),
        ),
        (
            Eventually,
            lambda window, beta: (
                (torch.logsumexp(window * beta, dim=0) - np.log(len(window)))
                / beta
            ),
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
    beta = 3.0

    got = operator(Predicate("p"), interval=[0, 2])(trajectory, beta=beta)[0]
    expected = torch.stack(
        [smooth_reduce(pointwise[t : t + 3], beta) for t in range(2)]
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
        Always(
            Predicate("p"), interval=[0, np.inf]
        ),  # used to raise OverflowError
    ):
        got = spec(traj)[0, :, 0].tolist()
        assert len(got) == len(values)
        np.testing.assert_allclose(got, ref)


def test_unbounded_start_offsets_the_suffix():
    values = [0.9, 0.2, 0.7, 0.4, 0.6]
    got = Always(Predicate("p"), interval=[2, np.inf])(scalar(values))[
        0, :, 0
    ].tolist()

    assert len(got) == len(values) - 2
    np.testing.assert_allclose(
        got, [min(values[t + 2 :]) for t in range(len(values) - 2)]
    )


# --- 5. Until --------------------------------------------------------------


def until_reference(left, right, a, b):
    """[lower, upper] per origin, straight from the Until equations."""
    T = len(left)
    out = []
    for t in range(T - b):
        lowers, uppers = [], []
        for tau in range(t + a, t + b + 1):
            prefix_lower = min(lo for lo, _ in left[t : tau + 1])
            prefix_upper = min(hi for _, hi in left[t : tau + 1])
            lowers.append(max(0.0, prefix_lower + right[tau][0] - 1.0))
            uppers.append(min(prefix_upper, right[tau][1]))
        out.append([max(lowers), max(uppers)])
    return out


UNTIL_LEFT = [
    (0.9, 0.95),
    (0.8, 0.9),
    (0.4, 0.7),
    (0.7, 0.8),
    (0.95, 1.0),
    (0.6, 0.75),
]
UNTIL_RIGHT = [
    (0.1, 0.2),
    (0.3, 0.6),
    (0.9, 0.95),
    (0.2, 0.4),
    (0.5, 0.7),
    (0.85, 0.9),
]


def until_trajectory():
    return supplied(
        *[{"l": lo, "r": hi} for lo, hi in zip(UNTIL_LEFT, UNTIL_RIGHT)]
    )


@pytest.mark.parametrize("interval", [[0, 0], [0, 1], [1, 2], [0, 3]])
def test_until_matches_the_reference_equations(interval):
    got = Until(Predicate("l"), Predicate("r"), interval=interval)(
        until_trajectory()
    )

    np.testing.assert_allclose(
        got[0].tolist(),
        until_reference(UNTIL_LEFT, UNTIL_RIGHT, *interval),
        atol=1e-6,
    )


def test_until_lower_bound_is_a_frechet_conjunction_of_prefix_and_witness():
    traj = supplied({"l": (0.6, 0.9), "r": (0.6, 0.8)})

    lower, upper = Until(Predicate("l"), Predicate("r"), interval=[0, 0])(
        traj
    )[0, 0]

    assert lower.item() == pytest.approx(0.6 + 0.6 - 1.0)  # 0.2, not min = 0.6
    assert upper.item() == pytest.approx(0.8)


def test_until_smooth_approaches_the_exact_reduction():
    spec = Until(Predicate("l"), Predicate("r"), interval=[0, 2])
    traj = until_trajectory()

    exact = spec(traj, beta=None)
    smooth = spec(traj, beta=400.0)

    torch.testing.assert_close(smooth, exact, atol=5e-2, rtol=0)
    assert not torch.equal(smooth, exact)  # a surrogate, not the same tensor


def test_until_smooth_matches_its_reference():
    trajectory = supplied(
        {"l": (0.7, 0.9), "r": (0.1, 0.3)},
        {"l": (0.4, 0.8), "r": (0.6, 0.75)},
        {"l": (0.8, 0.95), "r": (0.5, 0.85)},
    )
    formula = Until(Predicate("l"), Predicate("r"), interval=[0, 2])

    exact = formula(trajectory, beta=None)[0, 0]
    # witnesses: [max(0, .7+.1-1), min(.9,.3)], [max(0, .4+.6-1), min(.8,.75)],
    #            [max(0, .4+.5-1), min(.8,.85)]
    torch.testing.assert_close(exact, torch.tensor([0.0, 0.8]))

    left = Predicate("l")(trajectory)[0]
    right = Predicate("r")(trajectory)[0]
    beta = 4.0
    candidates = []
    for witness in range(3):
        window = left[: witness + 1]
        prefix = (
            -(torch.logsumexp(-window * beta, dim=0) - np.log(len(window)))
            / beta
        )
        candidates.append(
            torch.stack(
                (
                    torch.nn.functional.softplus(
                        prefix[0] + right[witness, 0] - 1.0, beta=beta
                    ),
                    -(
                        torch.logsumexp(
                            -torch.stack((prefix[1], right[witness, 1]))
                            * beta,
                            dim=0,
                        )
                        - np.log(2)
                    )
                    / beta,
                )
            )
        )
    expected_smooth = (
        torch.logsumexp(torch.stack(candidates) * beta, dim=0) - np.log(3)
    ) / beta

    torch.testing.assert_close(
        formula(trajectory, beta=beta)[0, 0], expected_smooth
    )


def test_until_at_zero_zero_combines_both_operands_now():
    traj = supplied({"l": (0.9, 0.9), "r": (0.4, 0.4)})

    lower, upper = Until(Predicate("l"), Predicate("r"), interval=[0, 0])(
        traj
    )[0, 0]

    assert lower.item() == pytest.approx(0.9 + 0.4 - 1.0)
    assert upper.item() == pytest.approx(0.4)


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
        [GaussianBelief(mean[t : t + 1], var[t : t + 1]) for t in range(3)]
    )

    assert Always(GreaterThan(50.0), interval=[0, 1])(traj).dtype == dtype


def test_gradients_reach_the_mean_through_a_composed_formula():
    mean = torch.tensor([[45.0], [55.0], [60.0], [48.0]], requires_grad=True)
    var = torch.full((4, 1), 4.0)
    traj = BeliefTrajectory(
        [GaussianBelief(mean[t : t + 1], var[t : t + 1]) for t in range(4)]
    )
    spec = Always(And(GreaterThan(50.0), LessThan(70.0)), interval=[0, 1])

    spec(traj)[0, 0, 0].backward()

    assert mean.grad is not None
    assert torch.isfinite(mean.grad).all()


def test_smooth_formula_passes_a_finite_difference_gradcheck():
    def f(mean):
        var = torch.full_like(mean, 4.0)
        traj = BeliefTrajectory(
            [
                GaussianBelief(mean[t : t + 1], var[t : t + 1])
                for t in range(len(mean))
            ]
        )
        spec = Always(GreaterThan(50.0), interval=[0, 1])
        return spec(traj, beta=2.0)[0, 0, 0]

    # well separated from ties, so min/logsumexp is smooth and nonsingular
    mean = torch.tensor(
        [[47.0], [53.0], [58.0]], dtype=torch.float64, requires_grad=True
    )
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
            GaussianBelief(mean[t : t + 1], var[t : t + 1])
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
    temporal = Always(predicate, interval=[0, 1])(traj, beta=None)

    assert temporal.dtype == torch.float64
    assert temporal.device == bounds.device
    temporal.sum().backward()
    assert bounds.grad is not None
    assert torch.isfinite(bounds.grad).all()
    assert bounds.grad.abs().sum() > 0
