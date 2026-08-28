"""Tests for src/models/temporal_examples.py: the example input traces."""

import torch

from models.temporal_examples import temporal_probability_traces


def test_time_has_length_twelve():
    time, _, _ = temporal_probability_traces()

    assert time.shape == (12,)


def test_bounds_have_expected_shape():
    _, safe_bounds, goal_bounds = temporal_probability_traces()

    assert safe_bounds.shape == (1, 12, 2)
    assert goal_bounds.shape == (1, 12, 2)


def test_bounds_are_finite():
    _, safe_bounds, goal_bounds = temporal_probability_traces()

    assert bool(torch.isfinite(safe_bounds).all())
    assert bool(torch.isfinite(goal_bounds).all())


def test_bounds_satisfy_the_probability_interval_invariant():
    _, safe_bounds, goal_bounds = temporal_probability_traces()

    for bounds in (safe_bounds, goal_bounds):
        lower, upper = bounds[..., 0], bounds[..., 1]
        assert bool((lower >= 0.0).all())
        assert bool((upper <= 1.0).all())
        assert bool((lower <= upper).all())


def test_dtype_is_preserved():
    _, safe_bounds, goal_bounds = temporal_probability_traces(dtype=torch.float32)

    assert safe_bounds.dtype == torch.float32
    assert goal_bounds.dtype == torch.float32
