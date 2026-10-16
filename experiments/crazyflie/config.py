"""The experiment configuration: defaults here, local values in config.yml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Workspace:
    x: tuple = (0.0, 1.0)
    y: tuple = (-2.0, 0.0)
    z_max: float = 0.6


@dataclass(frozen=True)
class Box:
    name: str
    x: tuple
    y: tuple


@dataclass(frozen=True)
class PathSpec:
    speed: float = 0.25  # m/s along the path
    centre: tuple = (0.5, -1.0)  # middle of the figure-8
    width: float = 0.6  # side to side
    height: float = 1.6  # bottom tip to top tip


@dataclass(frozen=True)
class Control:
    gain: float = 1.0  # 1/s on the position error
    v_max: float = 0.40  # m/s command limit
    rate: float = 20.0  # Hz
    max_error: float = 0.30  # m behind the reference before aborting
    tol: float = 0.10  # m from the end point to finish
    end_timeout: float = 10.0  # s allowed to settle at the end


@dataclass(frozen=True)
class Flight:
    connect_timeout: float = 10.0
    transit_speed: float = 0.20  # m/s from takeoff to the path start
    settle_spread: float = 0.08  # m of jitter allowed over 1 s before takeoff
    settle_timeout: float = 20.0  # s to wait for the estimate to settle
    takeoff_s: float = 3.0
    stale_s: float = 0.25  # abort if the state is older than this
    fence_margin: float = 0.05  # m allowed outside the workspace


@dataclass(frozen=True)
class Config:
    radio_uri: str = "radio://0/80/2M/E7E7E7E85"
    altitude: float = 0.30
    workspace: Workspace = field(default_factory=Workspace)
    path: PathSpec = field(default_factory=PathSpec)
    control: Control = field(default_factory=Control)
    flight: Flight = field(default_factory=Flight)
    airframe_radius: float = 0.07  # m; obstacles grow and walls move in by this
    obstacles: tuple = ()  # Box, ...
    goal: Box = Box("goal", (0.35, 0.65), (-0.35, -0.05))
    dt: float = 0.2  # s per step of the specification


def _tuple(value):
    if isinstance(value, (list, tuple)):
        return tuple(_tuple(v) for v in value)
    return value


def load(path=HERE / "config.yml"):
    """Read config.yml; unknown keys are an error so typos do not pass."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    parts = {"workspace": Workspace, "path": PathSpec,
             "control": Control, "flight": Flight}
    for name, cls in parts.items():
        raw[name] = cls(**{k: _tuple(v) for k, v in (raw.get(name) or {}).items()})
    raw["obstacles"] = tuple(_box(b) for b in raw.get("obstacles") or [])
    if "goal" in raw:
        raw["goal"] = _box({"name": "goal", **raw["goal"]})
    return Config(**raw)


def _box(raw):
    return Box(**{k: _tuple(v) for k, v in raw.items()})
