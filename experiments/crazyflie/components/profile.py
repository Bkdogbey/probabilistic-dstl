"""Stable provenance identifiers for Crazyflie flight datasets."""

from __future__ import annotations

import hashlib
import json
from typing import Any


FLIGHT_PROFILE_SCHEMA_VERSION = 2


def flight_profile_payload(config: dict[str, Any], scenario: str) -> dict[str, Any]:
    """Return the physical/logging inputs that make two runs comparable."""
    flight = config['flight']
    common = {
        'schema_version': FLIGHT_PROFILE_SCHEMA_VERSION,
        'scenario': scenario,
        'log_sample_hz': int(flight['log_sample_hz']),
        'estimator': flight['estimator'],
        'start_position': {
            'tolerance': flight['start_position_tolerance'],
            'timeout': flight['start_position_timeout'],
            'settle_seconds': flight['start_settle_seconds'],
        },
        'controller': 'PositionHlCommander.CONTROLLER_PID',
    }
    if scenario == 'figure8':
        figure8 = config['figure8']
        return {
            **common,
            'trajectory_version': int(figure8['trajectory_version']),
            'workspace': figure8['workspace'],
            'path': figure8['path'],
            'deterministic_cruise_velocity': figure8['deterministic_cruise_velocity'],
            'flight_points': int(figure8['flight_points']),
            'obstacles': figure8['obstacles'],
        }
    if scenario == 'baseline':
        return {
            **common,
            'arena': config['arena'],
            'deterministic_path': config['deterministic_path'],
        }
    raise ValueError(f'Unknown Crazyflie scenario {scenario!r}.')


def flight_profile_signature(config: dict[str, Any], scenario: str) -> str:
    payload = flight_profile_payload(config, scenario)
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()
