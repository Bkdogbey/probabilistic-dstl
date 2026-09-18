"""Integration regressions for the example cases, the optimizer wiring and the
plot adapter. The core semantics are covered by test_pdstl_core.py."""

import numpy as np
import pytest
import torch
from scipy.stats import norm

from planning.examples import CASES, build_case, run_case
from planning.runners import setup_problem
from utils import load_config

from visualization.robustness import _to_numpy


@pytest.fixture(scope="module")
def setup():
    cfg = load_config("configs/scenarios/planning_examples.yaml")
    problem = setup_problem(cfg, device="cpu")
    predicted = problem.rollout(torch.zeros(cfg["H"], 2))
    mean, cov = predicted.aux["mean_trace"], predicted.aux["cov_trace"]
    return cfg, problem.planner.cfg, predicted.belief_trajectory, mean, cov


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

    got = formula(traj)[0, :, 0].detach().numpy()
    p = atom_probs(mean, cov, case["threshold"], cfg["dim"], ">=")
    ref = [p[t + a : t + b + 1].min() for t in range(len(p) - b)]

    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_eventually_matches_direct_window_maximum(setup):
    cfg, _, traj, mean, cov = setup
    case = cfg["cases"]["eventually"]
    a, b = case["interval"]
    formula, _, _, _ = build_case("eventually", cfg)

    got = formula(traj)[0, :, 0].detach().numpy()
    p = atom_probs(mean, cov, case["threshold"], cfg["dim"], ">=")
    ref = [p[t + a : t + b + 1].max() for t in range(len(p) - b)]

    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_corridor_matches_frechet_inside_the_window(setup):
    cfg, _, traj, mean, cov = setup
    case = cfg["cases"]["corridor"]
    a, b = case["interval"]
    formula, _, _, _ = build_case("corridor", cfg)

    got = formula(traj)[0, :, :].detach().numpy()
    lo = atom_probs(mean, cov, case["lower_threshold"], cfg["dim"], ">=")
    hi = atom_probs(mean, cov, case["upper_threshold"], cfg["dim"], "<=")
    conj_lower = np.maximum(0.0, lo + hi - 1.0)  # Frechet, not the product
    conj_upper = np.minimum(lo, hi)

    ref_lower = [conj_lower[t + a : t + b + 1].min() for t in range(len(lo) - b)]
    ref_upper = [conj_upper[t + a : t + b + 1].min() for t in range(len(lo) - b)]

    np.testing.assert_allclose(got[:, 0], ref_lower, atol=1e-6)
    np.testing.assert_allclose(got[:, 1], ref_upper, atol=1e-6)


def test_until_matches_the_frechet_witness_reference(setup):
    cfg, _, traj, mean, cov = setup
    case = cfg["cases"]["until"]
    a, b = case["interval"]
    formula, _, _, _ = build_case("until", cfg)

    got = formula(traj)[0, :, 0].detach().numpy()
    left = atom_probs(mean, cov, case["safe_threshold"], cfg["dim"], "<=")
    right = atom_probs(mean, cov, case["goal_threshold"], cfg["dim"], ">=")

    ref = []
    for t in range(len(left) - b):
        best = 0.0
        for tau in range(t + a, t + b + 1):
            prefix = left[t : tau + 1].min()  # inclusive of the witness
            best = max(best, max(0.0, prefix + right[tau] - 1.0))
        ref.append(best)

    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_nested_matches_inner_then_outer_reduction(setup):
    cfg, _, traj, mean, cov = setup
    case = cfg["cases"]["nested"]
    ia, ib = case["inner_interval"]
    oa, ob = case["outer_interval"]
    formula, _, _, _ = build_case("nested", cfg)

    got = formula(traj)[0, :, 0].detach().numpy()
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
        assert formula(traj).shape[1] == k, name
        assert k >= 1, name


def test_plot_adapter_accepts_a_short_trace_and_uses_origin_timestamps(setup):
    cfg, _, traj, _, _ = setup
    formula, _, _, _ = build_case("nested", cfg)
    bounds = _to_numpy(formula(traj).detach())

    time = np.array([t * cfg["dt"] for t in range(cfg["H"] + 1)])
    plotted = time[: len(bounds)]

    assert bounds.shape == (1, 2)  # nested leaves a single valid origin
    # origin timestamps, not shifted by the interval's lower bound
    assert plotted[0] == pytest.approx(0.0)
    assert len(plotted) == len(bounds)


def test_adapter_rejects_a_malformed_trace():
    with pytest.raises(ValueError, match=r"\[K, 2\]"):
        _to_numpy(np.zeros((4, 3)))




@pytest.mark.parametrize("name", list(CASES))
def test_demonstration_returns_a_replayable_plan(name):
    cfg = load_config("configs/scenarios/planning_examples.yaml")
    problem = setup_problem(cfg, device="cpu")
    spec, _, _, _ = build_case(name, cfg)
    result = run_case(name, show=False, save=False)
    replay = problem.planner.evaluate_controls(problem.rollout, result.controls, spec=spec)
    assert replay.hard_interval == pytest.approx(result.hard_interval, abs=1e-6)
    assert result.loss_history
    assert torch.isfinite(result.controls).all()
