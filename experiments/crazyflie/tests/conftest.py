import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config


@pytest.fixture
def cfg():
    """The tracked template with short waits, so simulated flights take seconds."""
    base = config.load(config.HERE / "config.example.yml")
    return replace(
        base,
        measured=True,  # synthetic test geometry, not a hardware measurement
        flight=replace(
            base.flight,
            takeoff_speed=1.0,
            transit_speed=1.0,
            land_speed=1.0,
        ),
    )
