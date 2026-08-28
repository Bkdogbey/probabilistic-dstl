"""Tests for the bounded temporal operators: TemporalOperator, Always, Eventually.

Hard mode (``smooth=False``) must reproduce the Frechet window semantics exactly
and return valid probability intervals. Smooth mode (``smooth=True``) is only an
optimization surrogate: it is checked for convergence to the hard result and for
gradient reach, never asserted to be a certified bound.
"""

import random

import pytest
import torch

from pdstl import (
    Always,
    Eventually,
    Not,
    OfflineSource,
    OnlineSource,
    Predicate,
    TemporalOperator,
)


def _offline(trace, predicate=None):
    """An OfflineSource holding a single [B, T, 2] trace."""
    a = predicate if predicate is not None else Predicate("A")
    return a, OfflineSource({a: trace})


def _trace(rows):
    """A [1, T, 2] trace from a list of [lower, upper] pairs."""
    return torch.tensor(rows, dtype=torch.float64).unsqueeze(0)


# Hand-written references for the Frechet window rules, deliberately written in
# plain Python so they cannot share a bug with the tensor implementation.


def _always_reference(window):
    lowers = [lower for lower, _ in window]
    uppers = [upper for _, upper in window]
    return [max(0.0, sum(lowers) - (len(window) - 1)), min(uppers)]


def _eventually_reference(window):
    lowers = [lower for lower, _ in window]
    uppers = [upper for _, upper in window]
    return [max(lowers), min(1.0, sum(uppers))]


def _windows(rows, a, b):
    """The active window at every anchor of a length-T row list."""
    return [rows[k + a : k + b + 1] for k in range(len(rows) - b)]


def _flat(rows):
    """Flatten a list of [lower, upper] pairs; pytest.approx rejects nesting."""
    return [value for row in rows for value in row]


# ---------------------------------------------------------------------------
# 1. Interval validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interval", [(2, 1), (5, 0)])
def test_interval_with_lower_above_upper_is_rejected(interval):
    with pytest.raises(ValueError, match="a <= b"):
        Always(Predicate("A"), interval)


@pytest.mark.parametrize("interval", [(-1, 2), (-3, -1)])
def test_negative_interval_lower_endpoint_is_rejected(interval):
    with pytest.raises(ValueError, match=">= 0"):
        Eventually(Predicate("A"), interval)


@pytest.mark.parametrize("interval", [(0.0, 2), (1, 2.5), (True, 2)])
def test_non_integer_interval_endpoints_are_rejected(interval):
    with pytest.raises(TypeError, match="integers"):
        Always(Predicate("A"), interval)


def test_degenerate_singleton_interval_is_accepted():
    formula = Always(Predicate("A"), (2, 2))
    assert formula.interval == (2, 2)


def test_temporal_operator_is_abstract():
    with pytest.raises(TypeError):
        TemporalOperator(Predicate("A"), (0, 1))


def test_child_is_registered_as_a_submodule():
    a = Predicate("A")
    formula = Always(a, (0, 2))
    assert dict(formula.named_children()) == {"child": a}


# ---------------------------------------------------------------------------
# 2-4. Frechet window semantics
# ---------------------------------------------------------------------------


def test_always_matches_frechet_intersection():
    rows = [[0.9, 0.95], [0.8, 0.9], [0.85, 1.0], [0.7, 0.75]]
    a, source = _offline(_trace(rows))

    out = Always(a, (0, 2))(source)

    expected = [_always_reference(window) for window in _windows(rows, 0, 2)]
    assert out.shape == (1, 2, 2)
    assert _flat(out[0].tolist()) == pytest.approx(_flat(expected))
    # Spot-check the first anchor against the equation written out by hand.
    assert out[0, 0].tolist() == pytest.approx([0.9 + 0.8 + 0.85 - 2.0, 0.9])


def test_always_lower_bound_clamps_at_zero():
    a, source = _offline(_trace([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]))

    out = Always(a, (0, 2))(source)

    # 0.1 + 0.2 + 0.3 - 2 = -1.4, clamped to 0.
    assert out[0, 0].tolist() == pytest.approx([0.0, 0.4])


def test_eventually_matches_frechet_union():
    rows = [[0.1, 0.2], [0.3, 0.35], [0.2, 0.4], [0.05, 0.1]]
    a, source = _offline(_trace(rows))

    out = Eventually(a, (0, 2))(source)

    expected = [_eventually_reference(window) for window in _windows(rows, 0, 2)]
    assert out.shape == (1, 2, 2)
    assert _flat(out[0].tolist()) == pytest.approx(_flat(expected))
    assert out[0, 0].tolist() == pytest.approx([0.3, 0.2 + 0.35 + 0.4])


def test_eventually_upper_bound_clamps_at_one():
    a, source = _offline(_trace([[0.4, 0.6], [0.5, 0.7], [0.3, 0.8]]))

    out = Eventually(a, (0, 2))(source)

    # 0.6 + 0.7 + 0.8 = 2.1, clamped to 1.
    assert out[0, 0].tolist() == pytest.approx([0.5, 1.0])


def test_nonzero_interval_lower_endpoint_selects_the_shifted_window():
    rows = [[0.05 * k, 0.5 + 0.05 * k] for k in range(7)]
    a, source = _offline(_trace(rows))

    out = Always(a, (2, 4))(source)

    expected = [_always_reference(window) for window in _windows(rows, 2, 4)]
    assert out.shape == (1, 3, 2)
    assert _flat(out[0].tolist()) == pytest.approx(_flat(expected))


def test_nonzero_interval_lower_endpoint_ignores_steps_before_it():
    """Anchor k must not see child_trace[k] .. child_trace[k + a - 1]."""
    rows = [[0.0, 0.0], [0.0, 0.0], [0.9, 0.95], [0.8, 0.9], [0.85, 0.9]]
    a, source = _offline(_trace(rows))

    out = Eventually(a, (2, 4))(source)

    # The two zero rows sit outside [2, 4], so they cannot reach the output.
    assert out[0, 0].tolist() == pytest.approx(_eventually_reference(rows[2:5]))


# ---------------------------------------------------------------------------
# 5-8. Shapes, empty traces, batching, dtype/device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("length", "interval"),
    [(5, (0, 0)), (5, (0, 2)), (7, (2, 4)), (4, (3, 3)), (6, (1, 5))],
)
def test_output_shape_is_b_t_minus_b_2(length, interval):
    _, b = interval
    trace = torch.rand(3, length, 2).sort(dim=-1).values
    a, source = _offline(trace)

    out = Always(a, interval)(source)

    assert out.shape == (3, max(length - b, 0), 2)


@pytest.mark.parametrize("length", [1, 2, 3])
def test_incomplete_trace_returns_an_empty_output(length):
    """T <= b is not an error: the window has simply not filled yet."""
    trace = torch.rand(2, length, 2).sort(dim=-1).values
    a, source = _offline(trace)

    out = Always(a, (0, 3))(source)

    assert out.shape == (2, 0, 2)


def test_incomplete_online_trace_returns_an_empty_output():
    a = Predicate("A")
    source = OnlineSource()
    source.append({a: torch.tensor([[0.4, 0.6]])})

    assert Eventually(a, (0, 2))(source).shape == (1, 0, 2)


def test_batched_traces_are_reduced_independently():
    rows_first = [[0.9, 0.95], [0.8, 0.9], [0.85, 1.0]]
    rows_second = [[0.1, 0.2], [0.3, 0.35], [0.2, 0.4]]
    trace = torch.tensor([rows_first, rows_second], dtype=torch.float64)
    a, source = _offline(trace)

    out = Always(a, (0, 2))(source)

    assert out.shape == (2, 1, 2)
    assert out[0, 0].tolist() == pytest.approx(_always_reference(rows_first))
    assert out[1, 0].tolist() == pytest.approx(_always_reference(rows_second))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("interval", [(0, 2), (0, 8)])
def test_dtype_and_device_are_preserved(dtype, interval):
    trace = torch.rand(2, 5, 2, dtype=dtype, device="cpu").sort(dim=-1).values
    a, source = _offline(trace)

    out = Eventually(a, interval)(source)

    assert out.dtype == dtype
    assert out.device == trace.device


# ---------------------------------------------------------------------------
# 9-10. forward() == step(), and offline == online
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("formula_type", [Always, Eventually])
@pytest.mark.parametrize("interval", [(0, 2), (2, 4), (1, 1)])
def test_forward_equals_a_manual_step_loop(formula_type, interval):
    trace = torch.rand(3, 8, 2, dtype=torch.float64).sort(dim=-1).values
    a, source = _offline(trace)
    formula = formula_type(a, interval)

    state = None
    stepped = []
    for current_bounds in torch.unbind(trace, dim=1):
        output, state = formula.step(current_bounds, state)
        if output is not None:
            stepped.append(output)

    assert torch.equal(formula(source), torch.stack(stepped, dim=1))


def test_step_returns_none_until_the_window_is_full():
    formula = Always(Predicate("A"), (1, 3))
    state = None

    for t in range(6):
        output, state = formula.step(torch.rand(2, 2).sort(dim=-1).values, state)
        assert (output is None) == (t < 3)
        assert state.shape == (2, min(t + 1, 4), 2)


def test_state_holds_the_raw_recent_child_bounds():
    """The hidden state is the raw window, not an accumulated probability."""
    formula = Always(Predicate("A"), (0, 2))
    rows = [[0.9, 0.95], [0.8, 0.9], [0.85, 1.0], [0.7, 0.75]]

    state = None
    for row in rows:
        _, state = formula.step(torch.tensor([row], dtype=torch.float64), state)

    assert state.shape == (1, 3, 2)
    assert _flat(state[0].tolist()) == pytest.approx(_flat(rows[1:]))


def test_step_does_not_store_state_on_the_module():
    formula = Always(Predicate("A"), (0, 2))
    before = set(formula.__dict__) | set(formula.state_dict())

    state = None
    for _ in range(4):
        _, state = formula.step(torch.rand(1, 2).sort(dim=-1).values, state)

    assert set(formula.__dict__) | set(formula.state_dict()) == before
    assert list(formula.buffers()) == []


@pytest.mark.parametrize("formula_type", [Always, Eventually])
def test_online_appends_reproduce_the_offline_result(formula_type):
    rows = [[0.9, 0.95], [0.8, 0.9], [0.85, 1.0], [0.7, 0.75], [0.6, 0.65]]
    trace = _trace(rows)

    a = Predicate("A")
    formula = formula_type(a, (0, 2))
    offline = formula(OfflineSource({a: trace}))

    online_source = OnlineSource()
    for step, row in enumerate(rows, start=1):
        online_source.append({a: torch.tensor([row], dtype=torch.float64)})
        online = formula(online_source)

        assert online.shape == (1, max(step - 2, 0), 2)
        assert torch.equal(online, offline[:, : online.shape[1], :])


# ---------------------------------------------------------------------------
# 11-12. Composition and interval validity
# ---------------------------------------------------------------------------


def test_nested_temporal_and_boolean_formulas_keep_correct_shapes():
    a, b = Predicate("A"), Predicate("B")
    source = OfflineSource(
        {
            a: torch.rand(2, 6, 2, dtype=torch.float64).sort(dim=-1).values,
            b: torch.rand(2, 6, 2, dtype=torch.float64).sort(dim=-1).values,
        }
    )

    # And keeps T = 6, Always[0,1] drops 1 -> 5, Eventually[0,2] drops 2 -> 3.
    assert Eventually(Always(a & b, (0, 1)), (0, 2))(source).shape == (2, 3, 2)
    assert Not(Always(a, (0, 2)))(source).shape == (2, 4, 2)
    assert (Always(a, (0, 2)) | Eventually(b, (0, 2)))(source).shape == (2, 4, 2)
    assert Always(Eventually(a, (1, 2)), (0, 1))(source).shape == (2, 3, 2)


def test_nested_temporal_operators_compose_the_window_semantics():
    rows = [[0.9, 0.95], [0.8, 0.9], [0.85, 1.0], [0.7, 0.75]]
    a, source = _offline(_trace(rows))

    inner = [_always_reference(window) for window in _windows(rows, 0, 1)]
    expected = [_eventually_reference(window) for window in _windows(inner, 0, 1)]

    out = Eventually(Always(a, (0, 1)), (0, 1))(source)

    assert _flat(out[0].tolist()) == pytest.approx(_flat(expected))


@pytest.mark.parametrize("formula_type", [Always, Eventually])
@pytest.mark.parametrize("interval", [(0, 0), (0, 3), (1, 4)])
def test_hard_outputs_are_valid_probability_intervals(formula_type, interval):
    rng = random.Random(20260827)
    rows = []
    for _ in range(9):
        lower = rng.random()
        rows.append([lower, lower + (1.0 - lower) * rng.random()])
    a, source = _offline(_trace(rows))

    out = formula_type(a, interval)(source)

    lower, upper = out[..., 0], out[..., 1]
    assert bool((lower >= 0.0).all())
    assert bool((upper <= 1.0).all())
    assert bool((lower <= upper).all())


# ---------------------------------------------------------------------------
# 13-17. Smooth mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("formula_type", [Always, Eventually])
def test_smooth_approaches_hard_as_beta_increases(formula_type):
    trace = torch.rand(2, 7, 2, dtype=torch.float64).sort(dim=-1).values
    a, source = _offline(trace)
    formula = formula_type(a, (0, 2))

    hard = formula(source)
    errors = [
        (formula(source, smooth=True, beta=beta) - hard).abs().max().item()
        for beta in (1.0, 5.0, 20.0, 100.0, 500.0)
    ]

    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < errors[0]
    assert errors[-1] < 1e-2


@pytest.mark.parametrize("formula_type", [Always, Eventually])
def test_smooth_mode_gradients_are_finite(formula_type):
    trace = torch.rand(2, 7, 2, dtype=torch.float64).sort(dim=-1).values
    trace.requires_grad_(True)
    a, source = _offline(trace)

    formula_type(a, (1, 3))(source, smooth=True, beta=20.0).sum().backward()

    assert trace.grad is not None
    assert bool(torch.isfinite(trace.grad).all())


def test_smooth_gradients_reach_every_input_in_the_active_window():
    # T = b + 1 = 5 gives exactly one anchor, so the active window is [2, 4].
    trace = torch.full((1, 5, 2), 0.5, dtype=torch.float64)
    trace[..., 1] = 0.8
    trace.requires_grad_(True)
    a, source = _offline(trace)

    out = Always(a, (2, 4))(source, smooth=True, beta=10.0)
    assert out.shape == (1, 1, 2)
    out[0, 0, 0].backward()

    lower_grad = trace.grad[0, :, 0]
    assert bool((lower_grad[2:5] != 0).all()), lower_grad
    assert bool((lower_grad[0:2] == 0).all()), lower_grad
    # The lower bound of Always does not depend on any upper bound.
    assert bool((trace.grad[0, :, 1] == 0).all())


def test_smooth_eventually_lower_gradient_reaches_every_window_step():
    trace = torch.tensor(
        [[[0.2, 0.5], [0.3, 0.5], [0.25, 0.5]]], dtype=torch.float64, requires_grad=True
    )
    a, source = _offline(trace)

    Eventually(a, (0, 2))(source, smooth=True, beta=10.0)[0, 0, 0].backward()

    # Hard amax would send gradient only to the argmax step (index 1).
    assert bool((trace.grad[0, :, 0] > 0).all()), trace.grad


def test_violated_always_window_is_flat_in_hard_mode_but_not_in_smooth_mode():
    rows = [[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]  # sum(lower) - 2 = -1.7

    hard_trace = _trace(rows).requires_grad_(True)
    a, hard_source = _offline(hard_trace)
    Always(a, (0, 2))(hard_source)[0, 0, 0].backward()
    assert bool((hard_trace.grad[0, :, 0] == 0).all())

    smooth_trace = _trace(rows).requires_grad_(True)
    b_pred, smooth_source = _offline(smooth_trace)
    Always(b_pred, (0, 2))(smooth_source, smooth=True, beta=5.0)[0, 0, 0].backward()
    assert bool((smooth_trace.grad[0, :, 0] > 0).all()), smooth_trace.grad


@pytest.mark.parametrize("formula_type", [Always, Eventually])
@pytest.mark.parametrize("interval", [(0, 2), (1, 2)])
def test_gradcheck_in_double_precision(formula_type, interval):
    a = Predicate("A")

    def evaluate(trace):
        return formula_type(a, interval)(
            OfflineSource({a: trace}), smooth=True, beta=3.0
        )

    # Interior values: validate_bounds must survive gradcheck's +/- eps probes.
    trace = torch.tensor(
        [[[0.30, 0.70], [0.45, 0.62], [0.38, 0.80], [0.52, 0.66]]],
        dtype=torch.float64,
        requires_grad=True,
    )

    assert torch.autograd.gradcheck(evaluate, (trace,), eps=1e-6, atol=1e-7)


def test_smooth_gradients_reach_the_probability_source_through_a_deep_formula():
    a_trace = torch.rand(2, 6, 2, dtype=torch.float64).sort(dim=-1).values
    b_trace = torch.rand(2, 6, 2, dtype=torch.float64).sort(dim=-1).values
    a_trace.requires_grad_(True)
    b_trace.requires_grad_(True)

    a, b = Predicate("A"), Predicate("B")
    source = OfflineSource({a: a_trace, b: b_trace})
    formula = Always(Eventually(a & b, (0, 1)), (0, 2))

    formula(source, smooth=True, beta=10.0)[..., 0].sum().backward()

    assert bool((a_trace.grad != 0).any())
    assert bool((b_trace.grad != 0).any())
    assert bool(torch.isfinite(a_trace.grad).all())
    assert bool(torch.isfinite(b_trace.grad).all())


def test_smooth_mode_is_not_asserted_to_be_a_certified_bound():
    """Finite beta may leave the hard interval; hard mode must be rerun."""
    a, source = _offline(_trace([[0.0, 0.5], [0.0, 0.5], [0.0, 0.5]]))
    formula = Always(a, (0, 2))

    hard = formula(source)
    smooth = formula(source, smooth=True, beta=2.0)

    assert hard[0, 0, 0].item() == pytest.approx(0.0)
    assert smooth[0, 0, 0].item() > hard[0, 0, 0].item()


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def test_str_shows_the_operator_and_interval():
    a = Predicate("A")
    assert str(Always(a, (0, 2))) == "□[0,2](A)"
    assert str(Eventually(a, (1, 3))) == "◇[1,3](A)"
