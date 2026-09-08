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


def gaussian(mean, var, confidence_level=0.0):
    """Trajectory of Gaussian steps from [T, D] mean and [T, D] or [T, D, D] var."""
    mean = torch.as_tensor(mean, dtype=torch.float64)
    var = torch.as_tensor(var, dtype=torch.float64)
    return BeliefTrajectory(
        [GaussianBelief(mean[t : t + 1], var[t : t + 1], confidence_level) for t in range(len(mean))]
    )


def interval_gaussian(mean, var, k):
    """Trajectory of k-sigma-displaced Gaussian steps, same shapes as `gaussian`."""
    mean = torch.as_tensor(mean, dtype=torch.float64)
    var = torch.as_tensor(var, dtype=torch.float64)
    return BeliefTrajectory(
        [
            GaussianBelief(mean[t : t + 1], var[t : t + 1], confidence_level=k)
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
    traj = gaussian([[1.0]], [[-2.0]])

    with pytest.raises(ValueError, match="negative projected variance"):
        GreaterThan(0.0)(traj)


def test_affine_event_uses_the_full_covariance():
    # Strongly correlated components: assuming independence changes the answer.
    mu = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    cov = torch.tensor([[[4.0, 3.0], [3.0, 9.0]]], dtype=torch.float64)
    w = [1.0, -1.0]
    traj = BeliefTrajectory([GaussianBelief(mu, cov, confidence_level=0.0)])

    got = GreaterThan(0.0, weights=w)(traj)[0, 0, 0].item()

    w_t = torch.tensor(w, dtype=torch.float64)
    m = (mu[0] * w_t).sum().item()
    v = (w_t @ cov[0] @ w_t).item()  # 4 - 2*3 + 9 = 7
    assert v == pytest.approx(7.0)
    assert got == pytest.approx(1.0 - norm.cdf((0.0 - m) / np.sqrt(v)), abs=1e-9)

    # the independence-assuming value differs, so the test would catch it
    v_diag = (w_t**2 * torch.diagonal(cov[0])).sum().item()  # 4 + 9 = 13
    assert got != pytest.approx(1.0 - norm.cdf((0.0 - m) / np.sqrt(v_diag)), abs=1e-6)


def test_unsupported_event_fails_clearly():
    with pytest.raises(ValueError, match="cannot evaluate"):
        Predicate("safe")(gaussian([[1.0]], [[1.0]]))


# --- 1b. Interval Gaussian predicate ---------------------------------------

MEAN_3 = [[45.0], [55.0], [50.0]]
VAR_3 = [[4.0], [4.0], [9.0]]


def test_zero_k_reproduces_the_exact_gaussian():
    exact = GreaterThan(50.0)(gaussian(MEAN_3, VAR_3))[0].numpy()

    got = GreaterThan(50.0)(interval_gaussian(MEAN_3, VAR_3, k=0.0))[0].numpy()

    np.testing.assert_allclose(got, exact, atol=1e-12)


@pytest.mark.parametrize("k", [0.5, 1.0, 2.0])
def test_endpoints_straddle_the_exact_probability(k):
    exact = GreaterThan(50.0)(gaussian(MEAN_3, VAR_3))[0, :, 0].numpy()

    got = GreaterThan(50.0)(interval_gaussian(MEAN_3, VAR_3, k=k))[0].numpy()

    # strict, because every step here has positive variance
    assert (got[:, 0] < exact).all()
    assert (exact < got[:, 1]).all()


def test_endpoints_are_the_displaced_mean_probabilities():
    k, threshold = 2.0, 50.0
    mean, var = np.array(MEAN_3)[:, 0], np.array(VAR_3)[:, 0]
    sigma = np.sqrt(var)

    got = GreaterThan(threshold)(interval_gaussian(MEAN_3, VAR_3, k=k))[0].numpy()

    # displacing the mean by k*sigma is the same as shifting z by k
    lower = 1.0 - norm.cdf((threshold - (mean - k * sigma)) / sigma)
    upper = 1.0 - norm.cdf((threshold - (mean + k * sigma)) / sigma)
    np.testing.assert_allclose(got[:, 0], lower, atol=1e-9)
    np.testing.assert_allclose(got[:, 1], upper, atol=1e-9)


def test_band_widens_monotonically_with_k():
    widths = []
    for k in (0.0, 0.5, 1.0, 2.0, 4.0):
        trace = GreaterThan(50.0)(interval_gaussian(MEAN_3, VAR_3, k=k))[0].numpy()
        widths.append(trace[:, 1] - trace[:, 0])

    for narrow, wide in zip(widths, widths[1:]):
        assert (wide > narrow).all()


def test_pessimism_flips_direction_with_the_sense():
    k = 1.5
    traj = interval_gaussian([[45.0]], [[4.0]], k=k)

    gt = GreaterThan(50.0)(traj)[0, 0].numpy()
    lt = LessThan(50.0)(traj)[0, 0].numpy()

    # The pessimistic mean for x >= c is the optimistic mean for x <= c, so the
    # two bands are reflections of one another.
    np.testing.assert_allclose(lt[0], 1.0 - gt[1], atol=1e-9)
    np.testing.assert_allclose(lt[1], 1.0 - gt[0], atol=1e-9)


@pytest.mark.parametrize("k", [0.0, 2.0])
@pytest.mark.parametrize(
    "m, c, expect_ge", [(3.0, 1.0, 1.0), (1.0, 3.0, 0.0), (2.0, 2.0, 1.0)]
)
def test_zero_variance_stays_deterministic_at_any_k(k, m, c, expect_ge):
    traj = interval_gaussian([[m]], [[0.0]], k=k)

    got = GreaterThan(c)(traj)[0, 0]

    assert got[0].item() == expect_ge  # the displacement is k*sigma = 0
    assert got[1].item() == expect_ge


@pytest.mark.parametrize("k", [-1.0, float("inf"), float("nan")])
def test_invalid_k_is_rejected(k):
    with pytest.raises(ValueError, match="confidence_level must be finite and non-negative"):
        GaussianBelief(torch.zeros(1, 1), torch.ones(1, 1), confidence_level=k)


def test_bounds_are_well_formed_through_a_temporal_formula():
    # An ordered, in-range band is what check_probability_bounds demands, and
    # Always must not disturb that.
    trace = Always(GreaterThan(50.0), interval=[0, 1])(
        interval_gaussian(MEAN_3, VAR_3, k=3.0)
    )

    assert (trace[..., 0] <= trace[..., 1]).all()
    assert (trace >= 0.0).all() and (trace <= 1.0).all()


def test_gradients_reach_the_mean_through_an_interval_belief():
    mean = torch.tensor(
        [[48.0], [52.0], [51.0]], dtype=torch.float64, requires_grad=True
    )
    var = torch.full((3, 1), 4.0, dtype=torch.float64)
    traj = BeliefTrajectory(
        [
            GaussianBelief(mean[t : t + 1], var[t : t + 1], confidence_level=2.0)
            for t in range(3)
        ]
    )

    Always(GreaterThan(50.0), interval=[0, 2])(traj)[..., 0].sum().backward()

    assert torch.isfinite(mean.grad).all()
    assert (mean.grad != 0).any()


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


def test_implies_is_negation_then_or():
    traj = supplied({"a": (0.7, 0.8), "b": (0.6, 0.9)})
    a, b = Predicate("a"), Predicate("b")

    torch.testing.assert_close(Implies(a, b)(traj), Or(Negation(a), b)(traj))


# --- 4. Always / Eventually ------------------------------------------------


@pytest.mark.parametrize("interval", [[0, 1], [1, 2], [2, 3]])
def test_always_and_eventually_match_direct_window_reductions(interval):
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1, size=9).tolist()
    traj = scalar(values)
    a, b = interval
    n = len(values) - b

    got_min = Always(Predicate("p"), interval=interval)(traj)[0, :, 0].tolist()
    got_max = Eventually(Predicate("p"), interval=interval)(traj)[0, :, 0].tolist()

    assert len(got_min) == n
    np.testing.assert_allclose(got_min, [min(values[t + a : t + b + 1]) for t in range(n)])
    np.testing.assert_allclose(got_max, [max(values[t + a : t + b + 1]) for t in range(n)])


def test_nested_temporal_lookahead_composes():
    values = [0.9, 0.2, 0.7, 0.4, 0.6, 0.8, 0.3, 0.5]
    traj = scalar(values)

    spec = Always(Eventually(Predicate("p"), interval=[0, 2]), interval=[0, 3])
    got = spec(traj)[0, :, 0].tolist()

    n = len(values) - 2 - 3
    inner = [max(values[t : t + 3]) for t in range(len(values) - 2)]
    assert len(got) == n
    np.testing.assert_allclose(got, [min(inner[t : t + 4]) for t in range(n)])


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
            cand = max(0.0, prefix + right[tau] - 1.0)
            best = cand if best is None else max(best, cand)
        out.append(best)
    return out


@pytest.mark.parametrize("interval", [[0, 0], [0, 1], [1, 2]])
def test_until_matches_the_inclusive_reference(interval):
    left = [0.9, 0.8, 0.4, 0.7, 0.95, 0.6]
    right = [0.1, 0.3, 0.9, 0.2, 0.5, 0.85]
    traj = supplied(*[{"l": (l, l), "r": (r, r)} for l, r in zip(left, right)])

    got = Until(Predicate("l"), Predicate("r"), interval=interval)(traj)[0, :, 0]

    np.testing.assert_allclose(
        got.tolist(), until_reference(left, right, *interval), atol=1e-6
    )


def test_until_at_zero_zero_combines_both_operands_now():
    traj = supplied({"l": (0.9, 0.9), "r": (0.4, 0.4)})

    lower, upper = Until(Predicate("l"), Predicate("r"), interval=[0, 0])(traj)[0, 0]

    assert lower.item() == pytest.approx(max(0.0, 0.9 + 0.4 - 1.0))  # Frechet
    assert upper.item() == pytest.approx(min(0.9, 0.4))


# --- 6. Finite trace -------------------------------------------------------


def test_insufficient_prediction_length_is_rejected():
    traj = scalar([0.5, 0.5, 0.5])

    with pytest.raises(ValueError, match="needs 5 steps"):
        Always(Predicate("p"), interval=[0, 4])(traj)
    with pytest.raises(ValueError, match="needs 4 steps"):
        Until(Predicate("p"), Predicate("p"), interval=[0, 3])(traj)


def test_boolean_operators_align_to_the_shortest_child():
    values = [0.9, 0.2, 0.7, 0.4, 0.6]
    traj = scalar(values)
    atom = Predicate("p")  # length 5
    windowed = Always(atom, interval=[0, 2])  # length 3

    combined = And(windowed, atom)(traj)

    assert combined.shape[1] == 3
    # origin 0 of each child really is origin 0 of the conjunction
    expected = max(0.0, min(values[0:3]) + values[0] - 1.0)
    assert combined[0, 0, 0].item() == pytest.approx(expected)


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
    spec = And(Always(GreaterThan(50.0), interval=[0, 1]), LessThan(70.0))

    spec(traj)[0, 0, 0].backward()

    assert mean.grad is not None
    assert torch.isfinite(mean.grad).all()


def test_smooth_formula_passes_a_finite_difference_gradcheck():
    def f(mean):
        var = torch.full_like(mean, 4.0)
        traj = BeliefTrajectory(
            [GaussianBelief(mean[t : t + 1], var[t : t + 1]) for t in range(len(mean))]
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
        [GaussianBelief(mean[t : t + 1], var[t : t + 1]) for t in range(len(u))]
    )
    spec = Eventually(GreaterThan(2.5), interval=[0, 2])

    spec(traj)[0, 0, 0].backward()

    assert u.grad is not None
    assert torch.isfinite(u.grad).all()
    assert u.grad.abs().sum() > 0
