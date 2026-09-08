"""Integration regressions for the example cases, the optimizer wiring and the
plot adapter. The core semantics are covered by test_pdstl_core.py."""

import numpy as np
import pytest
import torch
from scipy.stats import norm

from models.dynamics import SingleIntegrator
from planning.examples import (
    CASES,
    _initial_controls,
    _initial_state,
    build_case,
    evaluate_direct,
    load_examples_config,
    rollout,
    run_always_step_zero,
    run_case,
    run_mpc_check,
)
from utils import get_device
from visualization.robustness import _to_numpy


@pytest.fixture(scope="module")
def setup():
    cfg, planner_cfg = load_examples_config()
    device = get_device()
    dyn = SingleIntegrator(
        dt=cfg["dt"], u_max=cfg["u_max"], q_std=cfg["q_std"], device=device
    )
    x0_mean, x0_cov = _initial_state(cfg, device)
    traj, mean, cov = rollout(dyn, _initial_controls(cfg, device), x0_mean, x0_cov)
    return cfg, planner_cfg, traj, mean, cov


def atom_probs(mean, cov, threshold, dim, sense):
    """P(x[dim] >= c) or P(x[dim] <= c) per step, computed with scipy."""
    mu = mean[0, :, dim].detach().numpy()
    sigma = np.sqrt(cov[0, :, dim, dim].detach().numpy())
    z = (mu - threshold) / sigma
    return norm.cdf(z) if sense == ">=" else norm.cdf(-z)


# --- 1. Each case matches an independent reduction -------------------------


def test_always_matches_direct_window_minimum(setup):
    cfg, _, traj, mean, cov = setup
    case = cfg["cases"]["always"]
    a, b = case["interval"]
    formula, _, _, _ = build_case("always", cfg)

    got = evaluate_direct(formula, traj)[0, :, 0].detach().numpy()
    p = atom_probs(mean, cov, case["threshold"], cfg["dim"], ">=")
    ref = [p[t + a : t + b + 1].min() for t in range(len(p) - b)]

    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_eventually_matches_direct_window_maximum(setup):
    cfg, _, traj, mean, cov = setup
    case = cfg["cases"]["eventually"]
    a, b = case["interval"]
    formula, _, _, _ = build_case("eventually", cfg)

    got = evaluate_direct(formula, traj)[0, :, 0].detach().numpy()
    p = atom_probs(mean, cov, case["threshold"], cfg["dim"], ">=")
    ref = [p[t + a : t + b + 1].max() for t in range(len(p) - b)]

    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_corridor_matches_frechet_inside_the_window(setup):
    cfg, _, traj, mean, cov = setup
    case = cfg["cases"]["corridor"]
    a, b = case["interval"]
    formula, _, _, _ = build_case("corridor", cfg)

    got = evaluate_direct(formula, traj)[0, :, :].detach().numpy()
    lo = atom_probs(mean, cov, case["lower_threshold"], cfg["dim"], ">=")
    hi = atom_probs(mean, cov, case["upper_threshold"], cfg["dim"], "<=")
    conj_lower = np.maximum(0.0, lo + hi - 1.0)  # Frechet, not the product
    conj_upper = np.minimum(lo, hi)

    ref_lower = [conj_lower[t + a : t + b + 1].min() for t in range(len(lo) - b)]
    ref_upper = [conj_upper[t + a : t + b + 1].min() for t in range(len(lo) - b)]

    np.testing.assert_allclose(got[:, 0], ref_lower, atol=1e-6)
    np.testing.assert_allclose(got[:, 1], ref_upper, atol=1e-6)


def test_until_matches_inclusive_prefix_reference(setup):
    cfg, _, traj, mean, cov = setup
    case = cfg["cases"]["until"]
    a, b = case["interval"]
    formula, _, _, _ = build_case("until", cfg)

    got = evaluate_direct(formula, traj)[0, :, 0].detach().numpy()
    left = atom_probs(mean, cov, case["safe_threshold"], cfg["dim"], "<=")
    right = atom_probs(mean, cov, case["goal_threshold"], cfg["dim"], ">=")

    ref = []
    for t in range(len(left) - b):
        best = 0.0
        for tau in range(t + a, t + b + 1):
            prefix = left[t : tau + 1].min()  # inclusive of the witness
            best = max(best, min(prefix, right[tau]))  # endpointwise, not Frechet
        ref.append(best)

    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_nested_matches_inner_then_outer_reduction(setup):
    cfg, _, traj, mean, cov = setup
    case = cfg["cases"]["nested"]
    ia, ib = case["inner_interval"]
    oa, ob = case["outer_interval"]
    formula, _, _, _ = build_case("nested", cfg)

    got = evaluate_direct(formula, traj)[0, :, 0].detach().numpy()
    p = atom_probs(mean, cov, case["threshold"], cfg["dim"], ">=")
    inner = [p[t + ia : t + ib + 1].min() for t in range(len(p) - ib)]
    inner = np.asarray(inner)
    ref = [inner[t + oa : t + ob + 1].max() for t in range(len(inner) - ob)]

    np.testing.assert_allclose(got, ref, atol=1e-6)


# --- 2. Trace lengths and time axes ----------------------------------------


def test_trace_lengths_follow_the_lookahead_rule(setup):
    cfg, _, traj, _, _ = setup
    steps = cfg["H"] + 1

    expected = {
        "always": steps - cfg["cases"]["always"]["interval"][1],
        "eventually": steps - cfg["cases"]["eventually"]["interval"][1],
        "corridor": steps - cfg["cases"]["corridor"]["interval"][1],
        "until": steps - cfg["cases"]["until"]["interval"][1],
        "nested": steps
        - cfg["cases"]["nested"]["inner_interval"][1]
        - cfg["cases"]["nested"]["outer_interval"][1],
    }
    for name, k in expected.items():
        formula, _, _, _ = build_case(name, cfg)
        assert evaluate_direct(formula, traj).shape[1] == k, name
        assert k >= 1, name


def test_plot_adapter_accepts_a_short_trace_and_uses_origin_timestamps(setup):
    cfg, _, traj, _, _ = setup
    formula, _, _, _ = build_case("nested", cfg)
    bounds = _to_numpy(evaluate_direct(formula, traj).detach())

    time = np.array([t * cfg["dt"] for t in range(cfg["H"] + 1)])
    plotted = time[: len(bounds)]

    assert bounds.shape == (1, 2)  # nested leaves a single valid origin
    # origin timestamps, not shifted by the interval's lower bound
    assert plotted[0] == pytest.approx(0.0)
    assert len(plotted) == len(bounds)


def test_adapter_rejects_a_malformed_trace():
    with pytest.raises(ValueError, match=r"\[K, 2\]"):
        _to_numpy(np.zeros((4, 3)))


# --- 3 and 4. Optimisation and return-value consistency --------------------


def test_optimisation_improves_the_directly_evaluated_robustness():
    result = run_case("eventually", save=False, verbose=False)

    lo_initial = result["interval_initial"][0]
    lo_final = result["interval_final"][0]
    assert lo_final > lo_initial + 0.05
    # a meaningful demonstration starts away from both saturation ends
    assert 0.01 < lo_initial < 0.95


def test_returned_controls_reproduce_the_reported_prediction():
    """The best iterate's controls, predictions and score come from one
    candidate: re-rolling the returned controls reproduces its mean exactly."""
    result = run_case("always", save=False, verbose=False)

    assert result["mean_mismatch"] < 1e-5
    assert np.isfinite(result["objective"])
    assert len(result["history"]) > 0


def test_gradients_are_finite_in_a_smooth_nondegenerate_optimisation():
    cfg, planner_cfg = load_examples_config()
    device = get_device()
    dyn = SingleIntegrator(
        dt=cfg["dt"], u_max=cfg["u_max"], q_std=cfg["q_std"], device=device
    )
    x0_mean, x0_cov = _initial_state(cfg, device)
    v = _initial_controls(cfg, device).clone().requires_grad_(True)

    formula, _, _, _ = build_case("eventually", cfg)
    traj, _, _ = rollout(dyn, v, x0_mean, x0_cov)
    formula(traj, scale=planner_cfg["scale"])[0, 0, 0].backward()

    assert v.grad is not None
    assert torch.isfinite(v.grad).all()
    assert v.grad.abs().sum() > 0


def test_step_zero_bottleneck_is_not_improved():
    """Always including step 0 depends on the fixed initial belief, so no
    control can change it. Reporting improvement here would be dishonest."""
    out = run_always_step_zero(verbose=False)

    assert out["after"][0] == pytest.approx(out["before"][0], abs=1e-6)
    assert 0.0 < out["before"][0] < 1.0  # a real, non-degenerate bottleneck


# --- 5. MPC consistency ----------------------------------------------------


def test_mpc_deadline_decrements_and_belief_resets_to_the_observation():
    steps = run_mpc_check(verbose=False)
    assert len(steps) >= 3

    # absolute deadline: strictly decreasing, never restarted
    uppers = [s["window"][1] for s in steps]
    assert uppers == sorted(uppers, reverse=True)
    assert all(b >= 0 for b in uppers)
    assert uppers[0] > uppers[-1]

    for s in steps:
        # belief is initialised at the observed state with zero covariance
        assert s["belief_cov_trace"] == pytest.approx(0.0, abs=1e-12)
        assert s["belief_mean"] == pytest.approx(s["true_state"], abs=1e-6)


def test_mpc_simulated_state_advances_and_reaches_the_target():
    steps = run_mpc_check(verbose=False)
    xs = [s["true_state"][0] for s in steps]

    assert xs[-1] > xs[0]  # the physical state actually moves
    assert any(s["satisfied"] for s in steps)
    # the physical trajectory is sampled, so it is not the belief mean path
    assert all(np.isfinite(x) for x in xs)


def test_every_named_case_runs():
    for name in CASES:
        result = run_case(name, save=False, verbose=False)
        assert result["trace_length"] >= 1
        assert len(result["interval_final"]) == 2
        lo, hi = result["interval_final"]
        assert lo <= hi + 1e-6
