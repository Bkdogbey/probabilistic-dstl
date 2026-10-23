"""Trial records: state.csv, target.csv and metadata.json per attempt.

Rows are kept in memory while flying (callbacks never touch the disk) and
written by `finish`, which `flight.run_flight` always calls. metadata.json
is written at creation too, so an attempt that never flies is recorded.

`target.csv` holds the onboard controller's setpoint (`ctrltarget`). It is
used to fit when playback of the uploaded curve really started.
"""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone

import minsnap
import numpy as np
from config import HERE, fan_setting, git_commit
from scipy.optimize import least_squares
from trajectory import file_hash

LOGS = HERE / "logs"
STATE = ["t_host", "device_ms", "x", "y", "z", "vx", "vy", "yaw"]
TARGET = ["t_host", "device_ms", "tx", "ty", "tz"]
CLOCK_WRAP = 1 << 24  # cflib log timestamps are a 24-bit ms counter
ALIGN_SEARCH_S = 0.5  # t_play is searched within this of t_cmd_after
ALIGN_SE_MAX = 0.02  # s; a larger standard error is not identifiable


class TrialLog:
    def __init__(self, metadata, root=LOGS):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        fan_tag = (
            f"_fan-{fan_setting(metadata['fan'])}" if "fan" in metadata else ""
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
        self.states, self.targets, self.events = [], [], []
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
        """All provenance is assembled once, before the ROS component starts.

        The plan's rows are copied in, so a trial carries its own curve.
        """
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
                "dry_run": dry_run,
                "note": note,
                "dt": cfg.dt,
                "duration": float(sum(plan["durations"])),
                "durations": plan["durations"],
                "coefficients": plan["coefficients"],
                "log_hz": cfg.log_hz,
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

    def target(self, row):
        with self._lock:
            self.targets.append(row)

    def battery(self, volts):
        self.meta.setdefault("battery_start", volts)
        self.meta["battery_end"] = volts

    def event(self, text):
        self.events.append(f"{self.phase}: {text}")

    def set_phase(self, phase):
        self.phase = phase

    def mission_start(self, t0, state):
        """t0 is the host time after start_trajectory; it schedules monitoring."""
        self.meta["mission_t0"] = t0
        self.meta["start_state"] = [float(v) for v in state]

    def mission_end(self, timestamp):
        """The execution boundary, before any return or cleanup command."""
        self.meta["mission_t_end"] = timestamp

    def finish(self, outcome, reason=None, kind=None, phase=None):
        """Fit the playback start, write every file; returns the metadata."""
        self.meta.update(
            outcome=outcome,
            abort_reason=reason,
            abort_kind=kind,
            abort_phase=phase if reason else None,
            events=self.events,
            finished=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        )
        with self._lock:
            state = _columns(STATE, self.states)
            target = _columns(TARGET, self.targets)
            _write_csv(self.dir / "state.csv", STATE, self.states)
            _write_csv(self.dir / "target.csv", TARGET, self.targets)
        if "t_cmd_after" in self.meta and "coefficients" in self.meta:
            self.meta.update(align_playback(self.meta, state, target))
        self._write_meta()
        return self.meta

    def _write_meta(self):
        (self.dir / "metadata.json").write_text(
            json.dumps(self.meta, indent=1, default=str)
        )


def _columns(names, rows):
    data = np.array(rows, dtype=float).reshape(-1, len(names))
    return {name: data[:, i] for i, name in enumerate(names)}


def _write_csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def _read_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return _columns(rows[0], rows[1:])


def load_trial(directory):
    """{'meta', 'state', 'target'}; columns are numpy arrays."""
    meta = json.loads((directory / "metadata.json").read_text())
    trial = {"meta": meta, "dir": directory}
    for name in ("state", "target"):
        path = directory / f"{name}.csv"
        trial[name] = _read_csv(path) if path.exists() else None
    return trial


def all_trials(root=LOGS):
    """Every recorded trial, oldest first."""
    return [
        load_trial(d)
        for d in sorted(root.glob("*/"))
        if (d / "metadata.json").exists()
    ]


def _unwrap(device_ms):
    ms = np.asarray(device_ms).astype(np.int64)
    steps = np.diff(ms)
    steps[steps < -CLOCK_WRAP // 2] += CLOCK_WRAP
    return (ms[0] + np.concatenate([[0], np.cumsum(steps)])) / 1000.0


def device_times(state, target=None):
    """Onboard sample times of the state and target blocks on the host clock.

    Both blocks are stamped by the same onboard clock. Device times are
    unwrapped, then shifted by the smallest receive delay seen in either
    block, so radio jitter does not blur the sample spacing. A missing or
    empty block gives None.
    """
    blocks = [
        b if b is not None and len(b["t_host"]) else None
        for b in (state, target)
    ]
    device = [None if b is None else _unwrap(b["device_ms"]) for b in blocks]
    offset = min(
        np.min(b["t_host"] - d)
        for b, d in zip(blocks, device)
        if b is not None
    )
    return [None if d is None else d + offset for d in device]


def state_times(state):
    return device_times(state)[0]


def align_playback(meta, state, target):
    """Fit when, and at what rate, the drone played the curve.

    The logged ctrltarget xy is matched to the plan's r(tau), with playback
    time tau = rate * (t - t_play) on the logged clock, over the whole
    mission. The drone's playback clock and its log timestamps can differ
    by about 1%, so the rate is fitted with the start. Before playback the
    setpoint holds the start and afterwards the end, so r is clamped to
    [0, T]. A coarse search over t_play starts a least-squares fit of both;
    the fit's standard error of t_play decides identifiability, and an
    unidentifiable fit falls back to the host command time.
    """
    t_cmd = meta["t_cmd_after"]
    result = {
        "t_play": t_cmd,
        "t_play_rate": 1.0,
        "alignment": "host_fallback",
        "t_play_offset": 0.0,
        "alignment_rms": None,
        "alignment_max": None,
        "alignment_se": None,
        "identifiable": False,
    }
    if target is None or len(target["t_host"]) < 10:
        return result
    rows = np.asarray(meta["coefficients"], dtype=float)
    t = device_times(state, target)[1]
    end = meta.get("mission_t_end", t_cmd + meta["duration"])
    keep = (t >= t_cmd - 1.0) & (t <= end)
    t = t[keep]
    xy = np.stack([target["tx"][keep], target["ty"][keep]], 1)
    if len(t) < 10:
        return result

    def residual(theta):
        t_play, rate = theta
        tau = rate * (t - t_play)
        return (xy - minsnap.evaluate(rows, tau)[:, :2]).ravel()

    starts = t_cmd + np.arange(-ALIGN_SEARCH_S, ALIGN_SEARCH_S, 0.01)
    start = min(starts, key=lambda t0: np.sum(residual((t0, 1.0)) ** 2))
    fit = least_squares(residual, [start, 1.0], x_scale=[0.01, 0.001])
    variance = np.sum(fit.fun**2) / max(len(fit.fun) - 2, 1)
    try:
        cov = variance * np.linalg.inv(fit.jac.T @ fit.jac)
        se = float(np.sqrt(cov[0, 0]))
    except np.linalg.LinAlgError:
        se = float("inf")
    result["alignment_se"] = se
    if se <= ALIGN_SE_MAX:
        t_play, rate = fit.x
        result.update(
            t_play=float(t_play),
            t_play_rate=float(rate),
            t_play_offset=float(t_play - t_cmd),
            alignment="ctrltarget",
            identifiable=True,
        )
    theta = (result["t_play"], result["t_play_rate"])
    distance = np.linalg.norm(residual(theta).reshape(-1, 2), axis=1)
    result.update(
        alignment_rms=float(np.sqrt(np.mean(distance**2))),
        alignment_max=float(distance.max()),
    )
    return result


def playback_time(meta, t):
    """Curve time of logged (host-clock) times t."""
    return meta.get("t_play_rate", 1.0) * (np.asarray(t) - meta["t_play"])


def logged_time(meta, tau):
    """Logged (host-clock) time of curve time tau."""
    return meta["t_play"] + np.asarray(tau) / meta.get("t_play_rate", 1.0)


def aligned_states(trial):
    """Measured xy, reference r(t_k) and tracking error e_k, each [N, 2].

    Step k is curve time k * dt, logged at logged_time(k * dt); r comes from
    the stored (uploaded) coefficients and e_k = p_k - r(t_k). Only steps
    covered by state data, up to the mission end, are returned.
    """
    meta, s = trial["meta"], trial["state"]
    rows = np.asarray(meta["coefficients"], dtype=float)
    t = device_times(s, trial.get("target"))[0]
    tk = minsnap.grid_times(rows, meta["dt"])
    grid = logged_time(meta, tk)
    end = min(t[-1], meta.get("mission_t_end", t[-1]))
    n = int(np.sum(grid <= end))
    p = np.stack([np.interp(grid[:n], t, s[c]) for c in ("x", "y")], 1)
    r = minsnap.evaluate(rows, tk[:n])[:, :2]
    return p, r, p - r


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
