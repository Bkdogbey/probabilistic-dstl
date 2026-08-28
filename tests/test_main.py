"""Tests for src/main.py: the drone-altitude pdSTL pipeline.

Hand-computed references live here, not in main.py.
"""

import importlib

import torch

import main as main_module
from main import run_pipeline


def _always_reference(window):
    lowers = [lower for lower, _ in window]
    uppers = [upper for _, upper in window]
    return [max(0.0, sum(lowers) - (len(window) - 1)), min(uppers)]


def _eventually_reference(window):
    lowers = [lower for lower, _ in window]
    uppers = [upper for _, upper in window]
    return [max(lowers), min(1.0, sum(uppers))]


def _windows(rows, a, b):
    return [rows[k + a : k + b + 1] for k in range(len(rows) - b)]


def test_predicate_returns_supplied_bounds_unchanged():
    result = run_pipeline()
    model = result["model"]

    assert torch.equal(result["bounds_50"], model.bounds_above_50)
    assert torch.equal(result["bounds_55"], model.bounds_above_55)


def test_not_uses_one_minus_upper_one_minus_lower():
    result = run_pipeline()
    bounds = result["bounds_50"]

    torch.testing.assert_close(result["not_above_50"][..., 0], 1 - bounds[..., 1])
    torch.testing.assert_close(result["not_above_50"][..., 1], 1 - bounds[..., 0])


def test_and_uses_frechet_intersection():
    result = run_pipeline()
    l1, u1 = result["bounds_50"][..., 0], result["bounds_50"][..., 1]
    l2, u2 = result["bounds_55"][..., 0], result["bounds_55"][..., 1]

    torch.testing.assert_close(result["and_bounds"][..., 0], torch.clamp(l1 + l2 - 1, min=0.0))
    torch.testing.assert_close(result["and_bounds"][..., 1], torch.minimum(u1, u2))


def test_or_uses_frechet_union():
    result = run_pipeline()
    l1, u1 = result["bounds_50"][..., 0], result["bounds_50"][..., 1]
    l2, u2 = result["bounds_55"][..., 0], result["bounds_55"][..., 1]

    torch.testing.assert_close(result["or_bounds"][..., 0], torch.maximum(l1, l2))
    torch.testing.assert_close(result["or_bounds"][..., 1], torch.clamp(u1 + u2, max=1.0))


def test_every_always_window_matches_hand_calculation():
    result = run_pipeline()
    actual = result["always_bounds"][0]
    rows = result["bounds_50"][0].tolist()

    expected = torch.tensor([_always_reference(w) for w in _windows(rows, 0, 2)], dtype=actual.dtype)
    torch.testing.assert_close(actual, expected)


def test_every_eventually_window_matches_hand_calculation():
    result = run_pipeline()
    actual = result["eventually_bounds"][0]
    rows = result["bounds_55"][0].tolist()

    expected = torch.tensor([_eventually_reference(w) for w in _windows(rows, 0, 2)], dtype=actual.dtype)
    torch.testing.assert_close(actual, expected)


def test_window_state_length_sequence_is_one_two_three_three():
    result = run_pipeline()
    always_above_50 = result["always_above_50"]

    window_state = None
    lengths = []
    for current_bounds in result["bounds_50"].unbind(dim=1):
        _, window_state = always_above_50.step(current_bounds, window_state)
        lengths.append(window_state.shape[1])

    assert lengths[:5] == [1, 2, 3, 3, 3]


def test_state_after_fourth_input_holds_x1_x2_x3():
    result = run_pipeline()
    always_above_50 = result["always_above_50"]
    bounds = result["bounds_50"]

    window_state = None
    for current_bounds in bounds[:, :4, :].unbind(dim=1):
        _, window_state = always_above_50.step(current_bounds, window_state)

    torch.testing.assert_close(window_state, bounds[:, 1:4, :])


def test_incremental_always_equals_offline_always():
    result = run_pipeline()

    torch.testing.assert_close(result["always_incremental"], result["always_bounds"])


def test_incremental_eventually_equals_offline_eventually():
    result = run_pipeline()

    torch.testing.assert_close(result["eventually_incremental"], result["eventually_bounds"])


def test_importing_main_has_no_side_effects(capsys):
    importlib.reload(main_module)

    captured = capsys.readouterr()
    assert captured.out == ""
