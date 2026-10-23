import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules["cflib"] = None  # tests never need cflib; its imports stay lazy

import config  # noqa: E402  (after the sys.path setup above)


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


@pytest.fixture
def short_plan(cfg, tmp_path):
    """A checked two-piece, 1.2 s plan near the start, saved to a file."""
    from geometry import Task
    from trajectory import check_plan, new_plan, save_plan

    start = np.array(cfg.start)
    waypoints = [start, start + [0.02, 0.01], start + [0.04, 0.02]]
    plan = new_plan(cfg, waypoints, [0.6, 0.6], "reference", "short")
    flight_cfg = cfg.for_plan(plan)
    check_plan(flight_cfg, Task(flight_cfg), plan)
    return save_plan(plan, tmp_path / "short.json"), plan
