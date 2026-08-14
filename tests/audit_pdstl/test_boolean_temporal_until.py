"""Audit Groups D-F: Boolean, temporal, and Until semantics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pdstl.operators import Always, And, Eventually, Negation, Or, STL_Formula, Until

from conftest import ConstantFormula, exact_temporal, exact_until


def _evaluate(formula, scale=-1):
    return formula(None, scale=scale).detach().cpu().numpy()[0]


class _TypedConstantFormula(STL_Formula):
    """Constant test formula that preserves a supplied tensor exactly."""

    def __init__(self, trace):
        super().__init__()
        self.trace = trace

    def robustness_trace(self, belief_trajectory, **kwargs):
        return self.trace


def test_d1_negation_swaps_and_complements_bounds():
    observed = _evaluate(Negation(ConstantFormula([[0.2, 0.7]])))
    np.testing.assert_allclose(observed, [[0.3, 0.8]], atol=1e-7)


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([0.6, 0.9], [0.7, 0.95], [0.3, 0.9]),
        ([0.2, 0.5], [0.3, 0.6], [0.0, 0.5]),
        ([1.0, 1.0], [0.7, 0.8], [0.7, 0.8]),
    ],
)
def test_d2_and_matches_frechet(left, right, expected):
    observed = _evaluate(And(ConstantFormula([left]), ConstantFormula([right])))
    np.testing.assert_allclose(observed, [expected], atol=1e-7)


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([0.2, 0.4], [0.3, 0.6], [0.3, 1.0]),
        ([0.1, 0.2], [0.3, 0.4], [0.3, 0.6]),
    ],
)
def test_d3_or_matches_frechet(left, right, expected):
    observed = _evaluate(Or(ConstantFormula([left]), ConstantFormula([right])))
    np.testing.assert_allclose(observed, [expected], atol=1e-7)


def test_d4_boolean_interval_invariants_for_seeded_cases():
    rng = np.random.default_rng(20260814)
    for _ in range(100):
        values = np.sort(rng.random((2, 2)), axis=1)
        left, right = values[0], values[1]
        formulas = [
            Negation(ConstantFormula([left])),
            And(ConstantFormula([left]), ConstantFormula([right])),
            Or(ConstantFormula([left]), ConstantFormula([right])),
        ]
        for formula in formulas:
            lower, upper = _evaluate(formula)[0]
            assert 0.0 <= lower <= upper <= 1.0


TRACE = np.array(
    [[0.15, 0.35], [0.75, 0.85], [0.30, 0.65], [0.90, 0.95], [0.45, 0.70]],
    dtype=float,
)
INCREASING_TRACE = np.array(
    [[0.10, 0.20], [0.25, 0.35], [0.40, 0.55], [0.60, 0.75], [0.80, 0.95]]
)
DECREASING_TRACE = INCREASING_TRACE[::-1].copy()


@pytest.mark.parametrize("interval", [[0, 0], [0, 1], [1, 2], [2, 3], [1, 3]])
@pytest.mark.parametrize("operator,operation", [(Always, "min"), (Eventually, "max")])
@pytest.mark.parametrize(
    "trace", [INCREASING_TRACE, DECREASING_TRACE], ids=["increasing", "decreasing"]
)
def test_e1_e2_bounded_temporal_matches_direct_windows(
    operator, operation, interval, trace
):
    observed = _evaluate(operator(ConstantFormula(trace), interval=interval))
    expected = exact_temporal(trace, interval, operation)
    assert observed == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("operator,operation", [(Always, "min"), (Eventually, "max")])
def test_e3_global_temporal_matches_suffix_reference(operator, operation):
    observed = _evaluate(operator(ConstantFormula(TRACE), interval=None))
    expected = exact_temporal(TRACE, [0, np.inf], operation)
    assert observed == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("operator", [Always, Eventually])
@pytest.mark.xfail(
    strict=True, reason="E4: explicit [0, inf] reaches int(inf) conversion"
)
def test_e4_explicit_zero_infinity_matches_global(operator):
    explicit = _evaluate(operator(ConstantFormula(TRACE), interval=[0, np.inf]))
    implicit = _evaluate(operator(ConstantFormula(TRACE), interval=None))
    assert explicit == pytest.approx(implicit, abs=1e-7)


@pytest.mark.parametrize("interval", [[1, np.inf], [2, np.inf]])
@pytest.mark.parametrize("operator,operation", [(Always, "min"), (Eventually, "max")])
def test_e5_delayed_unbounded_matches_direct_reference(operator, operation, interval):
    observed = _evaluate(operator(ConstantFormula(TRACE), interval=interval))
    expected = exact_temporal(TRACE, interval, operation)
    assert observed == pytest.approx(expected, abs=1e-7)


def test_f1_until_single_time_uses_inclusive_stori_frechet_candidate():
    phi = ConstantFormula([[0.6, 0.6]])
    psi = ConstantFormula([[0.7, 0.7]])
    observed = _evaluate(Until(phi, psi, interval=[0, 0]))
    np.testing.assert_allclose(observed, [[0.3, 0.6]], atol=1e-7)


def test_f2_until_prefix_includes_tau():
    phi_trace = np.array([[0.9, 0.9], [0.1, 0.1], [0.8, 0.8]])
    psi_trace = np.array([[0.2, 0.2], [0.9, 0.9], [0.4, 0.4]])
    observed = _evaluate(
        Until(ConstantFormula(phi_trace), ConstantFormula(psi_trace), interval=[1, 1])
    )
    expected = exact_until(phi_trace, psi_trace, [1, 1])
    assert observed == pytest.approx(expected, abs=1e-7)
    assert observed[0] == pytest.approx([0.0, 0.1], abs=1e-7)


PHI = np.array(
    [[0.85, 0.95], [0.55, 0.80], [0.70, 0.90], [0.25, 0.60], [0.65, 0.75]]
)
PSI = np.array(
    [[0.20, 0.45], [0.75, 0.90], [0.35, 0.55], [0.80, 0.95], [0.50, 0.85]]
)
PHI_ALT = np.array(
    [[0.40, 0.70], [0.90, 0.95], [0.15, 0.45], [0.60, 0.80], [0.35, 0.65]]
)
PSI_ALT = np.array(
    [[0.65, 0.85], [0.10, 0.30], [0.80, 0.90], [0.30, 0.55], [0.75, 0.95]]
)


@pytest.mark.parametrize(
    "phi_trace,psi_trace",
    [(PHI, PSI), (PHI_ALT, PSI_ALT)],
    ids=["primary", "alternate"],
)
@pytest.mark.parametrize(
    "interval", [[0, 0], [0, 1], [0, 2], [1, 2], [0, np.inf]]
)
def test_f3_until_matches_direct_stori_reference_all_times(
    interval, phi_trace, psi_trace
):
    observed = _evaluate(
        Until(
            ConstantFormula(phi_trace),
            ConstantFormula(psi_trace),
            interval=interval,
        )
    )
    expected = exact_until(phi_trace, psi_trace, interval)
    assert observed == pytest.approx(expected, abs=1e-7)


def test_f4_until_matches_randomized_reference_and_preserves_invariants():
    rng = np.random.default_rng(140826)
    intervals = [[0, 0], [0, 1], [0, 2], [1, 2], [0, np.inf]]
    for _ in range(100):
        phi = np.sort(rng.random((5, 2)), axis=1)
        psi = np.sort(rng.random((5, 2)), axis=1)
        for interval in intervals:
            observed = _evaluate(
                Until(ConstantFormula(phi), ConstantFormula(psi), interval=interval)
            )
            expected = exact_until(phi, psi, interval)
            assert observed == pytest.approx(expected, abs=1e-7)
            assert np.isfinite(observed).all()
            assert np.all(observed[..., 0] >= 0.0)
            assert np.all(observed[..., 0] <= observed[..., 1])
            assert np.all(observed[..., 1] <= 1.0)


def test_f4_until_preserves_batch_shape_dtype_and_device():
    phi = torch.tensor(
        [
            [[0.6, 0.8], [0.4, 0.7], [0.9, 0.95]],
            [[0.2, 0.5], [0.8, 0.9], [0.3, 0.6]],
        ],
        dtype=torch.float64,
    )
    psi = torch.tensor(
        [
            [[0.7, 0.9], [0.3, 0.6], [0.8, 0.9]],
            [[0.5, 0.7], [0.6, 0.8], [0.4, 0.65]],
        ],
        dtype=torch.float64,
    )
    observed = Until(
        _TypedConstantFormula(phi), _TypedConstantFormula(psi), interval=[0, 2]
    )(None)
    expected = np.stack(
        [exact_until(left, right, [0, 2]) for left, right in zip(phi, psi)]
    )
    assert observed.shape == (2, 3, 2)
    assert observed.dtype == phi.dtype
    assert observed.device == phi.device
    assert observed.detach().cpu().numpy() == pytest.approx(expected, abs=1e-12)
