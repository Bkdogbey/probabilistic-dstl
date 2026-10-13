"""Load and validate ``config.yaml``.

Every pipeline stage reads the same validated dict, so a missing or
invalid value fails here, before planning or flight, with its key name.
"""

from __future__ import annotations

from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / "config.yaml"

REQUIRED = {
    "radio": ["uri"],
    "planning": ["method", "start", "goal", "height", "s_path", "pdstl"],
    "planning.s_path": ["amplitude", "speed", "dt"],
    "planning.pdstl": [
        "dynamics",
        "H",
        "dt",
        "u_max",
        "q_std",
        "x0_var",
        "goal_half_width",
        "seed",
        "monte_carlo_samples",
        "planner",
    ],
    "bounds": ["x", "y", "z", "margin"],
    "environment": ["obstacles", "rings"],
    "environment.obstacles": ["enabled", "clearance", "boxes"],
    "environment.rings": ["enabled", "regions"],
    "flight": [
        "connect_timeout_s",
        "takeoff_speed",
        "transit_speed",
        "land_speed",
        "home_tolerance",
        "transit_z",
        "stream_hz",
        "start_tolerance",
        "start_timeout_s",
        "settle_s",
        "hold_end_s",
        "max_speed",
        "max_lateness_s",
    ],
    "positioning": ["reset_wait_s", "window", "spread_limit", "timeout_s"],
    "telemetry": ["period_ms", "stale_s", "queue_size"],
    "logging": ["dir", "plans_dir"],
}
POSITIVE = {
    "planning": ["height"],
    "planning.s_path": ["speed", "dt"],
    "planning.pdstl": ["H", "dt", "u_max", "q_std", "goal_half_width"],
    "flight": [
        "connect_timeout_s",
        "takeoff_speed",
        "transit_speed",
        "land_speed",
        "stream_hz",
        "start_tolerance",
        "start_timeout_s",
        "max_speed",
        "max_lateness_s",
    ],
    "positioning": ["window", "spread_limit", "timeout_s"],
    "telemetry": ["period_ms", "stale_s", "queue_size"],
}


def section(cfg, dotted):
    """``section(cfg, "planning.pdstl")`` -> ``cfg["planning"]["pdstl"]``."""
    for key in dotted.split("."):
        cfg = cfg.get(key) if isinstance(cfg, dict) else None
    return cfg


def _check_keys(cfg):
    for where, keys in REQUIRED.items():
        found = section(cfg, where) or {}
        missing = [k for k in keys if k not in found]
        if missing:
            raise ValueError(f"config '{where}' is missing {missing}")
    for where, keys in POSITIVE.items():
        for key in keys:
            if not section(cfg, where)[key] > 0:
                raise ValueError(f"config '{where}.{key}' must be > 0")


def _check_geometry(cfg):
    bounds = cfg["bounds"]
    for axis in "xyz":
        lo, hi = bounds[axis]
        if not lo < hi:
            raise ValueError(f"config 'bounds.{axis}' must satisfy lo < hi")
    if bounds["margin"] < 0:
        raise ValueError("config 'bounds.margin' must be >= 0")
    if not bounds["z"][0] <= cfg["flight"]["transit_z"] <= bounds["z"][1]:
        raise ValueError("config 'flight.transit_z' must lie in bounds.z")
    planning = cfg["planning"]
    if list(planning["start"]) == list(planning["goal"]):
        raise ValueError("config 'planning.start' and 'goal' must differ")
    tel = cfg["telemetry"]
    if tel["period_ms"] % 10:
        raise ValueError(
            "config 'telemetry.period_ms' must be a multiple of 10"
        )
    if not tel["stale_s"] > tel["period_ms"] / 1000:
        raise ValueError("config 'telemetry.stale_s' must exceed one period")
    for ring in cfg["environment"]["rings"]["regions"]:
        if len(ring["center"]) != 3 or not ring["radius"] > 0:
            raise ValueError(f"config ring {ring.get('name')} is invalid")


def load(path=DEFAULT, method=None):
    """Load, optionally override ``planning.method``, and validate.

    Args:
        path: YAML file.
        method: If given, replaces ``planning.method``.

    Returns:
        The config dict.

    Raises:
        ValueError: A required key is missing or a value is invalid.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or not cfg.get("name"):
        raise ValueError("config needs a non-empty 'name'")
    if method is not None and isinstance(cfg.get("planning"), dict):
        cfg["planning"]["method"] = method
    _check_keys(cfg)
    _check_geometry(cfg)
    return cfg


def resolve(cfg, key):
    """Absolute directory for ``logging.<key>`` (relative to this folder)."""
    path = Path(cfg["logging"][key])
    return path if path.is_absolute() else HERE / path
