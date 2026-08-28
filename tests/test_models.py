"""Tests for src/models/drone.py."""

import torch

from models.drone import drone_altitude_example


def test_time_has_nine_steps():
    model = drone_altitude_example()

    assert model.time.shape == (9,)


def test_bounds_have_expected_shape():
    model = drone_altitude_example()

    assert model.bounds_above_50.shape == (1, 9, 2)
    assert model.bounds_above_55.shape == (1, 9, 2)


def test_altitude_mean_and_std_have_expected_shape():
    model = drone_altitude_example()

    assert model.altitude_mean.shape == (9,)
    assert model.altitude_std.shape == (9,)


def test_bounds_are_finite():
    model = drone_altitude_example()

    assert bool(torch.isfinite(model.bounds_above_50).all())
    assert bool(torch.isfinite(model.bounds_above_55).all())


def test_bounds_satisfy_the_probability_interval_invariant():
    model = drone_altitude_example()

    for bounds in (model.bounds_above_50, model.bounds_above_55):
        lower, upper = bounds[..., 0], bounds[..., 1]
        assert bool((lower >= 0.0).all())
        assert bool((upper <= 1.0).all())
        assert bool((lower <= upper).all())


def test_dtype_is_preserved():
    model = drone_altitude_example(dtype=torch.float32)

    assert model.bounds_above_50.dtype == torch.float32
    assert model.altitude_mean.dtype == torch.float32
