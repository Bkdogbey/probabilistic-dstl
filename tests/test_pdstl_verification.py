"""Numerical core of the pdSTL verification suite (``src/verification.py``).

These tests check the *numbers* the verification scripts report and plot --
never the rendered images. They import the same builders ``src/main.py`` calls,
so what is asserted here is exactly what appears in the figures.

What is being verified, per demonstration:

A. the Gaussian halfspace probability, and that ``F`` is the temporal union;
B. that the conjunction of two halfspace events on one random variable is
   handled *exactly* and is emphatically **not** a product of probabilities,
   and that ``G`` is the temporal intersection with the post-reduction operand
   count;
C. mean/covariance propagation, backend agreement, autograd through the whole
   stochastic pipeline, that direct optimization of the hard lower bound
   improves it, and that the known hard zero-gradient plateau is unchanged.

Headless rendering is pinned in ``tests/conftest.py`` (``MPLBACKEND=Agg``), so
importing the verification suite here never tries to open a window.
"""

import math

import numpy as np
import pytest
import torch

from models.dynamics import bound_controls, linear_gaussian_rollout
from pdstl import (
    Predicate,
    TableProbabilitySource,
    TensorProbabilitySource,
    compile_formula,
    compile_recurrent_formula,
)
from tests.pdstl_rollout import differentiable_rollout
from verification import (
    analytic_band_probability,
    analytic_eventually_bounds,
    backend_bounds,
    build_always_scenario,
    build_eventually_scenario,
    build_stochastic_scenario,
    initial_controls,
    optimize_hard_lower_bound,
    propagate,
    run_zero_gradient_diagnostic,
)

TOL = 1e-10


def _phi(z):
    """Standard normal CDF, computed independently of the provider."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Shared: the tensor -> ProbabilitySource adapter
# ---------------------------------------------------------------------------


def test_tensor_source_matches_an_equivalent_table_source():
    a = Predicate(name="A")
    b = Predicate(name="B")
    rows = {
        a: [(0.10, 0.30), (0.40, 0.55), (0.90, 0.95)],
        b: [(0.20, 0.20), (0.05, 0.65), (0.70, 0.80)],
    }

    table = TableProbabilitySource(
        {(p, k): interval for p, intervals in rows.items() for k, interval in enumerate(intervals)},
        horizon=2,
    )
    traces = {
        p.uid: torch.tensor(intervals, dtype=torch.float32).unsqueeze(0)
        for p, intervals in rows.items()
    }
    tensor_source = TensorProbabilitySource(traces)

    assert tensor_source.horizon == table.horizon
    for p in rows:
        for k in range(3):
            assert torch.equal(tensor_source.bounds(p, k), table.bounds(p, k))


def test_tensor_source_rejects_ragged_traces_and_out_of_range_times():
    a = Predicate(name="A")
    with pytest.raises(ValueError, match="same number of times"):
        TensorProbabilitySource(
            {0: torch.zeros(1, 3, 2), 1: torch.zeros(1, 4, 2)}
        )
    source = TensorProbabilitySource({a.uid: torch.zeros(1, 3, 2)})
    with pytest.raises(KeyError):
        source.bounds(a, 5)


def test_tensor_source_preserves_autograd():
    a = Predicate(name="A")
    p = torch.full((1, 3), 0.4, requires_grad=True)
    trace = torch.stack([p, p], dim=-1)
    source = TensorProbabilitySource({a.uid: trace})
    source.bounds(a, 1).sum().backward()
    assert p.grad is not None and torch.any(p.grad != 0)


# ---------------------------------------------------------------------------
# Verification A -- Eventually is the temporal union
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eventually():
    return build_eventually_scenario()


def test_a_atom_probability_matches_the_analytic_gaussian_cdf(eventually):
    threshold = eventually["config"]["threshold"]
    expected = [
        _phi((mu - threshold) / sigma)
        for mu, sigma in zip(eventually["mean_trace"], eventually["sigma_trace"])
    ]
    assert eventually["atom_probabilities"] == pytest.approx(expected, abs=1e-12)


def test_a_probabilities_are_small_before_the_window_and_vary_inside(eventually):
    a, b = eventually["interval"]
    p = eventually["atom_probabilities"]
    assert p[: a].max() < 1e-3, "x >= 8 should be unlikely before the window opens"
    window = p[a : b + 1]
    assert window.max() - window.min() > 0.25, "atomic probabilities must vary visibly"
    assert window.max() < 0.999 and window.min() > 0.0


def test_a_eventually_is_max_lower_and_summed_upper(eventually):
    bounds = backend_bounds(eventually["formula"], eventually["traces"], eventually["horizon"])
    expected_lower, expected_upper = analytic_eventually_bounds(
        eventually["atom_probabilities"], eventually["interval"]
    )
    for backend, (lower, upper) in bounds.items():
        assert lower == pytest.approx(expected_lower, abs=1e-12), backend
        assert upper == pytest.approx(expected_upper, abs=1e-12), backend


def test_a_upper_bound_is_informative_rather_than_saturated(eventually):
    """The window's probabilities sum to < 1, so ``U_F`` is not pinned at 1."""
    a, b = eventually["interval"]
    total = eventually["atom_probabilities"][a : b + 1].sum()
    assert total < 1.0
    _, upper = backend_bounds(
        eventually["formula"], eventually["traces"], eventually["horizon"]
    )["recurrent"]
    assert upper == pytest.approx(total, abs=1e-12)
    assert upper < 1.0


def test_a_backends_agree(eventually):
    """``backend_bounds`` raises on disagreement; this pins the values too."""
    bounds = backend_bounds(eventually["formula"], eventually["traces"], eventually["horizon"])
    assert bounds["reference"] == pytest.approx(bounds["compiled"], abs=TOL)
    assert bounds["reference"] == pytest.approx(bounds["recurrent"], abs=TOL)


# ---------------------------------------------------------------------------
# Verification B -- conjunction, then temporal intersection
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def always():
    return build_always_scenario()


def test_b_halfspace_probabilities_match_the_analytic_cdfs(always):
    low, high = always["thresholds"]
    mu, sigma = always["mu_trace"], always["sigma_trace"]
    assert always["p_low"] == pytest.approx(
        [_phi((m - low) / s) for m, s in zip(mu, sigma)], abs=1e-12
    )
    assert always["p_high"] == pytest.approx(
        [_phi((high - m) / s) for m, s in zip(mu, sigma)], abs=1e-12
    )


def test_b_conjunction_lower_bound_is_exactly_the_band_probability(always):
    """For two halfspaces on one Gaussian, the Frechet lower bound is exact.

    ``max(0, p_low + p_high - 1) == Phi((5-mu)/s) - Phi((3-mu)/s)`` identically,
    because ``p_low + p_high - 1`` *is* that difference. The analytic value is
    an external reference: it is never supplied to pdSTL.
    """
    exact = analytic_band_probability(
        always["mu_trace"], always["sigma_trace"], *always["thresholds"]
    )
    assert always["band_lower"] == pytest.approx(exact, abs=1e-12)


def test_b_conjunction_is_not_a_product(always):
    """The independence assumption is measurably wrong, and is not used.

    At the widened-uncertainty step the product of the two marginals differs
    from the true band probability by far more than any floating-point slack,
    and -- crucially -- it *overestimates* it, so it would be unsound as a
    lower bound.
    """
    product = always["p_low"] * always["p_high"]
    exact = always["analytic_band"]

    widened = int(np.argmax(always["sigma_trace"]))
    assert abs(product[widened] - exact[widened]) > 0.02
    assert product[widened] > exact[widened], "independence overestimates here"
    # pdSTL's own lower bound is exact at that step, so it is not the product.
    assert always["band_lower"][widened] == pytest.approx(exact[widened], abs=1e-12)
    assert always["band_lower"][widened] != pytest.approx(product[widened], abs=1e-3)


def test_b_conjunction_encloses_the_truth_at_every_time(always):
    exact = always["analytic_band"]
    assert np.all(always["band_lower"] <= exact + 1e-12)
    assert np.all(always["band_upper"] >= exact - 1e-12)


def test_b_always_is_the_frechet_temporal_intersection(always):
    a, b = always["interval"]
    window_lower = always["band_lower"][a : b + 1]
    window_upper = always["band_upper"][a : b + 1]
    m = len(window_lower)  # post-reduction count: distinct events, one per time

    expected_lower = max(0.0, window_lower.sum() - (m - 1))
    expected_upper = window_upper.min()

    bounds = backend_bounds(always["formula"], always["traces"], always["horizon"])
    for backend, (lower, upper) in bounds.items():
        assert lower == pytest.approx(expected_lower, abs=1e-12), backend
        assert upper == pytest.approx(expected_upper, abs=1e-12), backend


def test_b_post_reduction_operand_count_is_the_window_length(always):
    """No event identity collapses here, so ``m`` is the full window length.

    Guards the distinction the semantics turns on: the offset ``m - 1`` must
    come from the number of *distinct surviving* events, which happens to equal
    the syntactic window length only because these five events really are
    distinct.
    """
    a, b = always["interval"]
    m = b - a + 1
    lower = backend_bounds(always["formula"], always["traces"], always["horizon"])["recurrent"][0]
    window_sum = always["band_lower"][a : b + 1].sum()

    assert lower == pytest.approx(max(0.0, window_sum - (m - 1)), abs=1e-12)
    # A wrong operand count would move the bound by a whole unit.
    assert lower != pytest.approx(max(0.0, window_sum - m), abs=1e-3)


def test_b_bounds_are_non_degenerate(always):
    """The scenario must actually exercise Frechet, not sit at 1.0."""
    lower, upper = backend_bounds(always["formula"], always["traces"], always["horizon"])["recurrent"]
    assert 0.0 < lower < upper < 1.0


# ---------------------------------------------------------------------------
# Verification C -- the complete stochastic-system pipeline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stochastic():
    scenario = build_stochastic_scenario()
    v = initial_controls(scenario)
    mean, covariance, traces = propagate(scenario, v)
    return scenario, v, mean, covariance, traces


def test_c_rollout_matches_the_closed_form_propagation(stochastic):
    scenario, v, mean, covariance, _ = stochastic
    config = scenario["config"]
    dt, u_max = float(config["dt"]), float(config["u_max"])
    q_var = float(config["q_std"]) ** 2

    controls = bound_controls(v, u_max)
    # Single integrator: mu_k = mu_0 + dt * sum_{j<k} u_j, Sigma_k = k * Q.
    expected_mean = torch.cumsum(controls * dt, dim=0)
    assert mean[0, 0] == pytest.approx(scenario["x0_mean"].tolist(), abs=1e-12)
    assert mean[0, 1:].detach().numpy() == pytest.approx(expected_mean.detach().numpy(), abs=1e-12)
    for k in range(scenario["horizon"] + 1):
        assert covariance[0, k, 0, 0].item() == pytest.approx(k * q_var, abs=1e-12)
        assert covariance[0, k, 0, 1].item() == pytest.approx(0.0, abs=1e-12)


def test_c_rollout_reproduces_the_existing_differentiable_rollout():
    """Pin the general propagator to the one the rest of the suite already uses.

    ``linear_gaussian_rollout`` generalizes ``differentiable_rollout`` (explicit
    ``A``/``B``, and the control bounding split out). For ``A = I``, ``B = dt I``
    the two must agree *exactly*, so the generalization cannot silently drift.
    """
    n, dim, dt, u_max = 5, 2, 0.7, 1.3
    generator = torch.Generator().manual_seed(20260825)
    v = torch.randn(n, dim, dtype=torch.float64, generator=generator)
    q = torch.diag(torch.tensor([0.09, 0.04], dtype=torch.float64))
    x0_mean = torch.zeros(dim, dtype=torch.float64)
    x0_cov = torch.zeros(dim, dim, dtype=torch.float64)

    expected_mean, expected_cov = differentiable_rollout(
        v, x0_mean, x0_cov, dt=dt, u_max=u_max, process_noise=q
    )
    mean, covariance = linear_gaussian_rollout(
        bound_controls(v, u_max),
        x0_mean,
        x0_cov,
        torch.eye(dim, dtype=torch.float64),
        dt * torch.eye(dim, dtype=torch.float64),
        q,
    )

    assert torch.equal(mean, expected_mean)
    assert torch.equal(covariance, expected_cov)


def test_c_atomic_trace_shapes(stochastic):
    scenario, _, mean, covariance, traces = stochastic
    horizon, dim = scenario["horizon"], scenario["dim"]

    assert mean.shape == (1, horizon + 1, dim)
    assert covariance.shape == (1, horizon + 1, dim, dim)
    # Two rectangles x four faces, each face its own atomic predicate.
    assert len(traces) == 8
    for trace in traces.values():
        assert trace.shape == (1, horizon + 1, 2)
        assert torch.all(trace[..., 0] <= trace[..., 1])
        assert torch.all((trace >= 0.0) & (trace <= 1.0))


def test_c_formula_bound_shapes(stochastic):
    scenario, _, _, _, traces = stochastic
    horizon = scenario["horizon"]
    for key in ("phi_safe", "phi_goal", "phi"):
        formula = scenario[key]
        out = compile_recurrent_formula(formula, horizon=horizon)(traces)
        assert out.shape == (1, horizon - formula.horizon() + 1, 2)


def test_c_all_three_backends_agree(stochastic):
    scenario, _, _, _, traces = stochastic
    for key in ("phi_safe", "phi_goal", "phi"):
        bounds = backend_bounds(scenario[key], traces, scenario["horizon"])
        assert bounds["reference"] == pytest.approx(bounds["compiled"], abs=TOL)
        assert bounds["reference"] == pytest.approx(bounds["recurrent"], abs=TOL)


def test_c_forward_bounds_are_on_an_active_branch(stochastic):
    """The C1/C2 scenario must not start in the clamped, gradient-free region."""
    scenario, _, _, _, traces = stochastic
    horizon = scenario["horizon"]
    lower_g = backend_bounds(scenario["phi_safe"], traces, horizon)["recurrent"][0]
    lower_f = backend_bounds(scenario["phi_goal"], traces, horizon)["recurrent"][0]
    lower_phi, upper_phi = backend_bounds(scenario["phi"], traces, horizon)["recurrent"]

    assert lower_g + lower_f - 1.0 > 0.0, "top-level conjunction must be active"
    assert lower_phi == pytest.approx(lower_g + lower_f - 1.0, abs=1e-12)
    assert upper_phi == pytest.approx(min(
        backend_bounds(scenario["phi_safe"], traces, horizon)["recurrent"][1],
        backend_bounds(scenario["phi_goal"], traces, horizon)["recurrent"][1],
    ), abs=1e-12)
    assert 0.0 < lower_g < 1.0, "G Safe should exercise Frechet, not saturate at 1"


def test_c_autograd_is_finite_and_nonzero_on_the_active_branch():
    scenario = build_stochastic_scenario()
    v = initial_controls(scenario, requires_grad=True)
    _, _, traces = propagate(scenario, v)
    out = compile_recurrent_formula(scenario["phi"], horizon=scenario["horizon"])(traces)

    lower = out[0, 0, 0]
    assert lower.item() > 0.0
    lower.backward()

    assert v.grad is not None
    assert torch.isfinite(v.grad).all()
    assert v.grad.norm().item() > 1e-6


def test_c_compiled_and_recurrent_gradients_match():
    grads = []
    for factory in (compile_formula, compile_recurrent_formula):
        scenario = build_stochastic_scenario()
        v = initial_controls(scenario, requires_grad=True)
        _, _, traces = propagate(scenario, v)
        factory(scenario["phi"], horizon=scenario["horizon"])(traces)[0, 0, 0].backward()
        grads.append(v.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-10)


def test_c_optimization_improves_the_hard_lower_bound():
    """Direct maximization of the exact hard lower bound -- no surrogate."""
    scenario = build_stochastic_scenario()
    result = optimize_hard_lower_bound(scenario, iterations=40)
    history = result["history"]

    assert history["lower"][-1] > history["lower"][0] + 1e-3
    assert all(math.isfinite(x) for x in history["lower"])
    assert all(0.0 <= x <= 1.0 for x in history["lower"])
    assert all(
        lower <= upper + 1e-9 for lower, upper in zip(history["lower"], history["upper"])
    )
    assert history["grad_norm"][0] > 1e-6
    # The loss is exactly the negated bound, not a transformed objective.
    assert history["loss"] == pytest.approx([-x for x in history["lower"]], abs=1e-12)


def test_c_optimized_controls_stay_within_their_box():
    scenario = build_stochastic_scenario()
    result = optimize_hard_lower_bound(scenario, iterations=20)
    u_max = float(scenario["config"]["u_max"])
    controls = bound_controls(result["v"], u_max)
    assert torch.all(controls.abs() <= u_max)


# ---------------------------------------------------------------------------
# Verification C3 -- the known hard zero-gradient plateau, unchanged
# ---------------------------------------------------------------------------


def test_c3_negative_preclamp_gives_zero_bound_and_zero_gradient():
    """Expected behavior of the exact hard semantics, reported not repaired.

    Below the clamp the Frechet lower bound is identically zero, so it carries
    no gradient. This test exists to keep that fact pinned: if it ever starts
    failing, a smoothing/margin has been introduced somewhere.
    """
    result = run_zero_gradient_diagnostic()

    assert result["pre_clamp"] < 0.0
    assert result["p_lower"] == 0.0
    assert result["grad_norm"] == 0.0
    assert result["lower_f"] == 0.0
    # Safety alone is still perfectly informative; only the conjunction collapses.
    assert result["lower_g"] > 0.9
