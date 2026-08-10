"""Offline tests for the experiment-specific positioning safety policy."""

from __future__ import annotations

import pytest

from experiments.crazyflie.components.utils import shutdown_hardware


class _FailingCommander:
    def __init__(self) -> None:
        self.land_calls = 0

    def land(self, *, velocity: float) -> None:
        self.land_calls += 1
        raise RuntimeError('landing failed')


class _FakeHardware:
    def __init__(self) -> None:
        self.disarm_calls = 0
        self.disconnect_calls = 0

    def disarm(self) -> None:
        self.disarm_calls += 1

    def disconnect(self) -> None:
        self.disconnect_calls += 1


def test_shutdown_disarms_and_disconnects_even_when_landing_raises():
    commander = _FailingCommander()
    hardware = _FakeHardware()

    with pytest.raises(RuntimeError, match='landing failed'):
        shutdown_hardware(
            commander, hardware, airborne=True, landing_velocity=0.3,
        )

    assert commander.land_calls == 1
    assert hardware.disarm_calls == 1
    assert hardware.disconnect_calls == 1


def test_shutdown_disarms_and_disconnects_when_never_airborne():
    hardware = _FakeHardware()

    shutdown_hardware(None, hardware, airborne=False, landing_velocity=0.3)

    assert hardware.disarm_calls == 1
    assert hardware.disconnect_calls == 1
