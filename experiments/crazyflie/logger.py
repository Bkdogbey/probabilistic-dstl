"""Trial records: commands.csv, state.csv and metadata.json per attempt.

Rows are kept in memory while flying (callbacks never touch the disk) and
written by `finish`, which `flight.run_flight` always calls. metadata.json
is written at creation too, so an attempt that never flies is recorded.
"""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
from config import HERE, fan_setting, git_commit
from trajectory import file_hash

LOGS = HERE / "logs"
STATE = ["t_host", "device_ms", "x", "y", "z", "vx", "vy", "yaw"]
COMMANDS = [
    "k",
    "step",
    "t_ref",
    "t_sent",
    "lateness",
    "vx",
    "vy",
    "yaw",
    "vx_body",
    "vy_body",
    "z",
]
CLOCK_WRAP = 1 << 24  # cflib log timestamps are a 24-bit ms counter


class TrialLog:
    def __init__(self, metadata, root=LOGS):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        fan_tag = (
            f"_fan{fan_setting(metadata['fan']):02d}"
            if "fan" in metadata
            else ""
        )
        self.id = (
            f"{stamp}_{metadata['condition']}{fan_tag}_{metadata['purpose']}"
        )
        self.dir = root / self.id
        self.dir.mkdir(parents=True)
        self.meta = {
            **metadata,
            "trial_id": self.id,
            "started": stamp,
            "outcome": "started",
        }
        self.phase = "connect"
        self.states, self.commands, self.events = [], [], []
        self._lock = threading.Lock()
        self._write_meta()

    @classmethod
    def for_flight(
        cls,
        cfg,
        plan,
        path,
        *,
        purpose,
        fan,
        dry_run=False,
        note="",
        root=LOGS,
    ):
        """All provenance is assembled once, before the ROS component starts."""
        return cls(
            {
                "purpose": purpose,
                "condition": plan["condition"],
                "plan_id": plan["id"],
                "plan_path": str(path.resolve()),
                "plan_hash": file_hash(path),
                "model_id": plan["model_id"],
                "model_hash": plan.get("model_hash"),
                "fan": fan_setting(fan),
                "fan_position": cfg.fan_position,
                "fan_direction": cfg.fan_direction,
                "dry_run": dry_run,
                "note": note,
                "dt": cfg.dt,
                "steps": len(plan["velocities"]),
                "repeats": cfg.repeats,
                "stream_hz": cfg.stream_hz,
                "timing_policy": "trajectory_step_deadline",
                "altitude": cfg.altitude,
                "altitude_tolerance": cfg.altitude_tolerance,
                "altitude_average_s": 0.25,
                "config_hash": cfg.hash,
                "plan_config_hash": cfg.plan_hash,
                "execution_hash": cfg.execution_hash,
                "data_readiness": cfg.ready_for_data(fan),
                "config": asdict(cfg),
                "git_commit": git_commit(),
            },
            root=root,
        )

    def state(self, row):
        with self._lock:
            self.states.append(row)

    def command(self, row):
        with self._lock:
            self.commands.append(row)

    def battery(self, volts):
        self.meta.setdefault("battery_start", volts)
        self.meta["battery_end"] = volts

    def event(self, text):
        self.events.append(f"{self.phase}: {text}")

    def set_phase(self, phase):
        self.phase = phase

    def mission_start(self, t0, state):
        self.meta["mission_t0"] = t0
        self.meta["start_state"] = [float(v) for v in state]

    def mission_end(self, timestamp):
        """The execution boundary, before any return or cleanup command."""
        self.meta["mission_t_end"] = timestamp

    def finish(self, outcome, reason=None, kind=None, phase=None):
        """Write every file; returns the final metadata."""
        self.meta.update(
            outcome=outcome,
            abort_reason=reason,
            abort_kind=kind,
            abort_phase=phase if reason else None,
            events=self.events,
            finished=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        )
        with self._lock:
            _write_csv(self.dir / "state.csv", STATE, self.states)
            _write_csv(self.dir / "commands.csv", COMMANDS, self.commands)
        self._write_meta()
        return self.meta

    def _write_meta(self):
        (self.dir / "metadata.json").write_text(
            json.dumps(self.meta, indent=1, default=str)
        )


def _write_csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def _read_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    data = np.array(rows[1:], dtype=float).reshape(-1, len(rows[0]))
    return {name: data[:, i] for i, name in enumerate(rows[0])}


def load_trial(directory):
    """{'meta', 'state', 'commands'}; columns are numpy arrays."""
    meta = json.loads((directory / "metadata.json").read_text())
    files = {n: directory / f"{n}.csv" for n in ("state", "commands")}
    trial = {"meta": meta, "dir": directory}
    for name, path in files.items():
        trial[name] = _read_csv(path) if path.exists() else None
    return trial


def all_trials(root=LOGS):
    """Every recorded trial, oldest first."""
    return [
        load_trial(d)
        for d in sorted(root.glob("*/"))
        if (d / "metadata.json").exists()
    ]


def state_times(state):
    """Onboard sample times on the host clock.

    Device times are unwrapped, then shifted by the smallest observed
    receive delay, so radio jitter does not blur the sample spacing.
    """
    ms = state["device_ms"].astype(np.int64)
    steps = np.diff(ms)
    steps[steps < -CLOCK_WRAP // 2] += CLOCK_WRAP
    device = (ms[0] + np.concatenate([[0], np.cumsum(steps)])) / 1000.0
    return device + np.min(state["t_host"] - device)


def aligned_states(trial):
    """States [N+1, 4] (px, py, vx, vy) at each plan step and controls [N, 2].

    Step k starts when command k was due: mission_t0 + k * dt. Only steps
    whose commands were all sent and whose end is covered by state data
    are returned.
    """
    meta, cmd, s = trial["meta"], trial["commands"], trial["state"]
    mission = cmd["step"] >= 0
    first = mission & (cmd["k"] % meta["repeats"] == 0)
    controls = np.stack([cmd["vx"][first], cmd["vy"][first]], 1)
    counts = np.bincount(
        cmd["step"][mission].astype(int), minlength=len(controls)
    )
    n = int(np.cumprod(counts == meta["repeats"]).sum())
    t = state_times(s)
    grid = meta["mission_t0"] + meta["dt"] * np.arange(n + 1)
    grid = grid[grid <= min(t[-1], meta.get("mission_t_end", t[-1]))]
    states = np.stack(
        [np.interp(grid, t, s[c]) for c in ("x", "y", "vx", "vy")], 1
    )
    return states, controls[: len(grid) - 1]


def find_trial(ref="latest", root=LOGS):
    """A trial by id, unique id fragment, or "latest"."""
    trials = sorted(
        d for d in root.glob("*/") if (d / "metadata.json").exists()
    )
    matches = (
        trials[-1:]
        if ref == "latest"
        else [d for d in trials if ref in d.name]
    )
    if len(matches) != 1:
        raise SystemExit(
            f"{len(matches)} trials match {ref!r}"
            + (f": {[d.name for d in matches]}" if matches else f" in {root}")
        )
    return load_trial(matches[0])
