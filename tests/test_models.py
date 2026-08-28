"""Tests for the standalone Boolean and Gaussian altitude inputs."""

import pytest
import torch

from models import drone
from models.boolean import boolean_example


def test_boolean_example_returns_the_specified_bounds():
    bounds_a, bounds_b = boolean_example()

    torch.testing.assert_close(
        bounds_a, torch.tensor([[[0.60, 0.90]]], dtype=bounds_a.dtype)
    )
    torch.testing.assert_close(
        bounds_b, torch.tensor([[[0.70, 0.95]]], dtype=bounds_b.dtype)
    )


@pytest.mark.parametrize(
    "model",
    [drone.always_altitude_example(), drone.eventually_altitude_example()],
)
def test_atomic_bounds_are_derived_from_ambiguous_gaussian_means(model):
    expected_lower = torch.special.ndtr(
        (model.mean_lower - model.threshold) / model.std
    )
    expected_upper = torch.special.ndtr(
        (model.mean_upper - model.threshold) / model.std
    )

    torch.testing.assert_close(model.probability_bounds[0, :, 0], expected_lower)
    torch.testing.assert_close(model.probability_bounds[0, :, 1], expected_upper)


def test_atomic_bounds_have_seven_steps_and_are_ordered():
    for model in (
        drone.always_altitude_example(),
        drone.eventually_altitude_example(),
    ):
        bounds = model.probability_bounds
        assert bounds.shape == (1, 7, 2)
        assert bool((bounds[..., 0] <= bounds[..., 1]).all())
        assert bool((bounds[..., 0] < bounds[..., 1]).all())
        assert bool((bounds >= 0).all())
        assert bool((bounds <= 1).all())


def test_changing_threshold_changes_generated_probabilities():
    low_threshold = drone.always_altitude_example(threshold=45.0)
    high_threshold = drone.always_altitude_example(threshold=55.0)

    assert not torch.equal(
        low_threshold.probability_bounds,
        high_threshold.probability_bounds,
    )
    assert bool(
        (
            low_threshold.probability_bounds
            > high_threshold.probability_bounds
        ).all()
    )


def _build_test_belief(**overrides):
    values = {
        "time": [0, 1],
        "mean_lower": [49.0, 50.0],
        "mean_upper": [50.0, 51.0],
        "std": [1.0, 1.0],
        "threshold": 50.0,
        "dtype": torch.float64,
    }
    values.update(overrides)
    return drone._build_altitude_belief(**values)


def test_belief_builder_rejects_unequal_trace_lengths():
    with pytest.raises(ValueError, match="equal trace lengths"):
        _build_test_belief(std=[1.0])


def test_belief_builder_rejects_reversed_mean_bounds():
    with pytest.raises(ValueError, match="mean_lower"):
        _build_test_belief(mean_lower=[51.0, 50.0])


@pytest.mark.parametrize("std", [[0.0, 1.0], [-1.0, 1.0]])
def test_belief_builder_rejects_nonpositive_standard_deviation(std):
    with pytest.raises(ValueError, match="strictly positive"):
        _build_test_belief(std=std)
