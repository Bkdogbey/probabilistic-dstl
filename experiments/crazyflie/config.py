"""The experiment configuration as one typed, frozen object."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))  # the pdSTL core (models, planning, ...)


FAN_SETTINGS = (0, 6, 12, 18)  # old experiment: off plus three active settings


def fan_setting(value):
    """Numeric physical setting; 'off' is readable in older trial metadata."""
    setting = 0 if value == "off" else int(value)
    if setting not in FAN_SETTINGS:
        raise ValueError(f"fan must be one of {FAN_SETTINGS}")
    return setting


@dataclass(frozen=True)
class Box:
    x: tuple[float, float]
    y: tuple[float, float]
    name: str = ""
    height: float = 0.0  # obstacles only

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
    connect_timeout: float = 10.0
    takeoff_speed: float = 0.3
    transit_speed: float = 0.3
    land_speed: float = 0.15
    start_tolerance: float = 0.05
    start_altitude_tolerance: float = 0.05
    start_speed: float = 0.10
    start_window: float = 1.0
    start_timeout: float = 15.0
    stale_s: float = 0.5
    geofence_margin: float = 0.10
    position_spread: float = 0.05
    position_timeout: float = 15.0


# Routine defaults live here; config.yml contains arena measurements and the
# settings changed between experiments. Existing full configs still work.
DEFAULTS = {
    "log_hz": 20,
    "clearance_margin": 0.01,
    "fan_position": None,
    "fan_direction": None,
    "model": {
        "holdout_every": 3, "bootstrap": 1000, "seed": 0,
        "min_flights_warning": 8,
    },
    "planner": {
        "max_iters": 400, "alpha": 0.90, "w_lim": 100.0,
        "snap_fraction": 0.01, "lr_grid": [0.003, 0.01, 0.03],
        "sweep_iters": 100,
    },
    "mean_model": {"beta": 50.0, "min_robustness": 0.0},
}  # fmt: skip


PLAN_FIELDS = (
    "workspace", "obstacles", "start", "goal", "airframe_radius", "reference",
    "altitude", "duration", "dt", "v_max", "a_max", "clearance_margin",
    "planner", "mean_model",
)  # fmt: skip


def _digest(value):
    text = json.dumps(value, sort_keys=True, default=asdict)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


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
    reference: str
    duration: float
    dt: float
    log_hz: int
    v_max: float
    a_max: float
    clearance_margin: float
    fan_position: list | None
    fan_direction: list | None
    flight: Flight
    model: dict
    planner: dict
    mean_model: dict
    model_report: str | None = None
    fan: int = 0

    @property
    def steps(self):
        """Mission horizon H in planning steps."""
        return round(self.duration / self.dt)

    @property
    def hash(self):
        """Whole configuration; recorded with every trial."""
        return _digest(asdict(self))

    @property
    def plan_hash(self):
        """Only what defines a plan; flight tuning and fan settings may change."""
        return _digest({name: getattr(self, name) for name in PLAN_FIELDS})

    @property
    def execution_hash(self):
        """Only geometry, altitude, timing and the curve limits."""
        fields = (
            "workspace",
            "obstacles",
            "airframe_radius",
            "altitude",
            "dt",
            "v_max",
            "a_max",
            "clearance_margin",
        )
        return _digest({name: getattr(self, name) for name in fields})

    def for_plan(self, plan):
        """The saved trajectory owns its start, altitude and timing."""
        return replace(
            self,
            start=tuple(plan["start"]),
            altitude=plan["altitude"],
            dt=plan["dt"],
            duration=float(sum(plan["durations"])),
        )

    def ready_for_data(self, fan):
        """Why recorded data is excluded from fitting/evaluation, or None."""
        if not self.measured:
            return "config.yml has measured: false"
        fan_setting(fan)
        return None


def load(path=HERE / "config.yml"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Create it from the template, then edit it:\n"
            f"  cd {HERE} && cp config.example.yml {path.name}"
        )
    raw = {**deepcopy(DEFAULTS), **yaml.safe_load(path.read_text())}
    raw["fan"] = fan_setting(raw.get("fan", 0))
    if not isinstance(raw["reference"], str):
        raise TypeError(
            "reference must name the supplied trajectory file, e.g. reference: plans/reference.json"
        )
    reference = Path(raw["reference"])
    raw["reference"] = str((path.parent / reference).resolve())
    # Timing/pose are read from the frozen file, not duplicated in YAML.
    from trajectory import read_plan

    supplied = read_plan(raw["reference"])
    raw.update(
        start=tuple(supplied["start"]),
        dt=supplied["dt"],
        altitude=supplied["altitude"],
        duration=float(sum(supplied["durations"])),
    )
    if raw.get("model_report"):
        raw["model_report"] = str(
            (path.parent / raw["model_report"]).resolve()
        )
    for name in ("model", "planner", "mean_model"):
        raw[name] = {**DEFAULTS[name], **(raw[name] or {})}
    return Config(
        **{
            **raw,
            "workspace": Box(**raw["workspace"]),
            "obstacles": tuple(Box(**box) for box in raw["obstacles"]),
            "start": tuple(raw["start"]),
            "goal": Box(**raw["goal"], name="goal"),
            "flight": Flight(**(raw.get("flight") or {})),
        }
    )


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
