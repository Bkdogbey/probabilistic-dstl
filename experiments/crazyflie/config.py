"""The experiment configuration as one typed, frozen object."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))  # the pdSTL core (models, planning, ...)


@dataclass(frozen=True)
class Box:
    x: tuple[float, float]
    y: tuple[float, float]
    name: str = ""

    def grown(self, margin):
        return Box(
            (self.x[0] - margin, self.x[1] + margin),
            (self.y[0] - margin, self.y[1] + margin),
            self.name,
        )

    @property
    def centre(self):
        return ((self.x[0] + self.x[1]) / 2, (self.y[0] + self.y[1]) / 2)


@dataclass(frozen=True)
class Flight:
    connect_timeout: float
    takeoff_speed: float
    transit_speed: float
    transit_z: float
    land_speed: float
    home_tolerance: float
    start_tolerance: float
    start_speed: float
    start_timeout: float
    settle: float
    hold_end: float
    stale_s: float
    max_lateness_s: float
    geofence_margin: float
    position_spread: float
    position_timeout: float


@dataclass(frozen=True)
class Config:
    measured: bool
    radio_uri: str
    altitude: float
    altitude_tolerance: float
    airframe_radius: float
    workspace: Box
    obstacles: tuple[Box, ...]
    start: tuple[float, float]
    goal: Box
    reference: dict
    duration: float
    dt: float
    stream_hz: int
    u_max: float
    fans: dict
    fan_position: list | None
    fan_direction: list | None
    excitation: dict
    flight: Flight
    model: dict
    planner: dict
    mean_model: dict

    @property
    def steps(self):
        """Mission horizon H in planning steps."""
        return round(self.duration / self.dt)

    @property
    def repeats(self):
        """Streamed commands per planning step."""
        return round(self.stream_hz * self.dt)

    @property
    def hash(self):
        text = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:12]

    def ready_for_data(self, fan):
        """Why identification/evaluation flights may not fly yet, or None."""
        if not self.measured:
            return "config.yml has measured: false"
        if self.fans.get(fan) is None:
            return f"fan setting {fan!r} is not chosen"
        if fan != "off" and None in (self.fan_position, self.fan_direction):
            return "fan position/direction are not measured"
        return None


def load(path=HERE / "config.yml"):
    raw = yaml.safe_load(Path(path).read_text())
    cfg = Config(
        **{
            **raw,
            "workspace": Box(**raw["workspace"]),
            "obstacles": tuple(Box(**box) for box in raw["obstacles"]),
            "start": tuple(raw["start"]),
            "goal": Box(**raw["goal"], name="goal"),
            "flight": Flight(**raw["flight"]),
        }
    )
    if abs(cfg.steps * cfg.dt - cfg.duration) > 1e-9:
        raise ValueError("duration must be a whole number of dt steps")
    if abs(cfg.repeats - cfg.stream_hz * cfg.dt) > 1e-9:
        raise ValueError("stream_hz * dt must be a whole number")
    return cfg


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
