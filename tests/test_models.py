"""Tests for src/models/boolean.py and src/models/drone.py."""

import torch

from models.boolean import boolean_example
from models.drone import always_altitude_example, eventually_altitude_example


def test_boolean_example_returns_the_specified_bounds():
    bounds_a, bounds_b = boolean_example()

    torch.testing.assert_close(bounds_a, torch.tensor([[[0.60, 0.90]]], dtype=bounds_a.dtype))
    torch.testing.assert_close(bounds_b, torch.tensor([[[0.70, 0.95]]], dtype=bounds_b.dtype))


def test_boolean_example_shape_is_one_one_two():
    bounds_a, bounds_b = boolean_example()

    assert bounds_a.shape == (1, 1, 2)
    assert bounds_b.shape == (1, 1, 2)


def test_always_model_probabilities_match_the_normal_cdf():
    model = always_altitude_example()

    expected = torch.special.ndtr((model.mean - model.threshold) / model.std)
    torch.testing.assert_close(model.probability_bounds[0, :, 0], expected)


def test_eventually_model_probabilities_match_the_normal_cdf():
    model = eventually_altitude_example()

    expected = torch.special.ndtr((model.mean - model.threshold) / model.std)
    torch.testing.assert_close(model.probability_bounds[0, :, 0], expected)


def test_atomic_bounds_have_shape_one_three_two():
    for model in (always_altitude_example(), eventually_altitude_example()):
        assert model.probability_bounds.shape == (1, 3, 2)


def test_atomic_lower_equals_upper():
    for model in (always_altitude_example(), eventually_altitude_example()):
        torch.testing.assert_close(model.probability_bounds[..., 0], model.probability_bounds[..., 1])


def test_atomic_bounds_are_valid_probabilities():
    for model in (always_altitude_example(), eventually_altitude_example()):
        bounds = model.probability_bounds
        assert bool((bounds >= 0.0).all())
        assert bool((bounds <= 1.0).all())
