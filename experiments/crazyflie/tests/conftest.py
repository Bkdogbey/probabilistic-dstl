import sys
from pathlib import Path

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    """The shipped config with fast timings, a small pdSTL problem and all
    outputs under tmp_path."""
    c = config.load()
    c["planning"]["s_path"]["speed"] = 2.0
    c["planning"]["pdstl"].update(
        H=24,
        dt=0.5,
        monte_carlo_samples=200,
        planner={"max_iters": 30, "lr": 0.05, "alpha": 0.9},
    )
    c["flight"].update(
        takeoff_speed=5.0,
        transit_speed=5.0,
        land_speed=5.0,
        stream_hz=50,
        start_timeout_s=1.0,
        settle_s=0.05,
        hold_end_s=0.05,
        max_speed=3.0,
    )
    c["positioning"].update(reset_wait_s=0.01, window=3, timeout_s=0.5)
    c["logging"]["dir"] = str(tmp_path / "logs")
    c["logging"]["plans_dir"] = str(tmp_path / "plans")
    return c


@pytest.fixture
def cfg_file(cfg, tmp_path):
    """Write a (possibly edited) config to disk; returns its path."""

    def write(c=None):
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(c or cfg))
        return str(path)

    return write
