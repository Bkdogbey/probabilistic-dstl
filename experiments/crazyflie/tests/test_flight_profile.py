"""Tests for flight-profile provenance signatures."""

from __future__ import annotations

import copy

from experiments.crazyflie.components import utils
from experiments.crazyflie.components.profile import (
    flight_profile_payload,
    flight_profile_signature,
)


def test_signature_is_stable_for_same_configuration():
    config = copy.deepcopy(utils._cfg)
    assert flight_profile_signature(config, 'figure8') == flight_profile_signature(
        config, 'figure8',
    )


def test_signature_changes_with_obstacle_position():
    original = copy.deepcopy(utils._cfg)
    changed = copy.deepcopy(original)
    changed['figure8']['obstacles'][0]['x'][0] += 0.01
    assert flight_profile_signature(original, 'figure8') != flight_profile_signature(
        changed, 'figure8',
    )


def test_signature_changes_with_speed_path_and_estimator():
    original = copy.deepcopy(utils._cfg)
    mutations = []
    speed = copy.deepcopy(original)
    speed['figure8']['deterministic_cruise_velocity'] += 0.01
    mutations.append(speed)
    path = copy.deepcopy(original)
    path['figure8']['path']['crossing_height'] += 0.01
    mutations.append(path)
    estimator = copy.deepcopy(original)
    estimator['flight']['estimator']['spread_limit'] += 0.01
    mutations.append(estimator)

    signature = flight_profile_signature(original, 'figure8')
    assert all(flight_profile_signature(config, 'figure8') != signature for config in mutations)


def test_payload_contains_required_provenance_inputs():
    payload = flight_profile_payload(utils._cfg, 'figure8')
    assert {
        'trajectory_version', 'path', 'deterministic_cruise_velocity', 'flight_points',
        'estimator', 'obstacles', 'log_sample_hz',
    }.issubset(payload)
