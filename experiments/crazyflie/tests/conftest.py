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
        flight=replace(
            base.flight,
            settle=0.1,
            hold_end=0.2,
            takeoff_speed=1.0,
            transit_speed=1.0,
            land_speed=1.0,
        ),
    )
