"""Flight analysis: load run directories, compute metrics, compare runs.

Used by ``run.py analyze`` (and by ``FlightLog`` for the summary metrics):

- one run: print outcome, plan probability, tracking error (vs the active
  controller target), ring passage and data quality (device-time gaps,
  command lateness, dropped rows); save ``flight.png/.pdf``.
- ``--all``: ``<logs>/runs.csv`` (one row per attempt) and
  ``<logs>/repeats.png/.pdf`` (completed runs of the latest run's plan,
  overlaid) to judge repeatability.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

import environment
import trajectory
from telemetry import CLOCK_WRAP, COLUMNS, unwrap_ms

RUN_FIELDS = [
    "run_id",
    "method",
    "p_lower",
    "outcome",
    "last_phase",
    "dry_run",
    "duration_s",
    "rmse_m",
    "max_m",
    "rings_passed",
    "dropped_rows",
    "gaps",
    "max_gap_ms",
    "max_lateness_s",
    "error",
]


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _floats(rows, key):
    return np.array([float(r[key]) if r[key] != "" else np.nan for r in rows])


def _packets(rows, columns):
    """One log block: unwrapped device ms, receive time, phase, values."""
    return {
        "device_ms": unwrap_ms([int(r["device_ms"]) for r in rows]),
        "t_recv": _floats(rows, "t_recv"),
        "phase": np.array([r["phase"] for r in rows], dtype=str),
        **{c: _floats(rows, c) for c in columns},
    }


def _commands(rows):
    return {
        "t_send": _floats(rows, "t_send"),
        "t_ref": _floats(rows, "t_ref"),
        "xyz": np.column_stack([_floats(rows, c) for c in "xyz"]).reshape(
            -1, 3
        ),
        "lateness": _floats(rows, "lateness_s"),
        "phase": np.array([r["phase"] for r in rows], dtype=str),
    }


def _nearest(source_ms, target_ms, values, tolerance_ms):
    """``values`` at the ``source_ms`` sample nearest each ``target_ms``
    (NaN when none lies within ``tolerance_ms``)."""
    out = np.full(len(target_ms), np.nan)
    if not len(source_ms):
        return out
    last = len(source_ms) - 1
    right = np.clip(np.searchsorted(source_ms, target_ms), 0, last)
    left = np.clip(right - 1, 0, None)
    closer_left = np.abs(source_ms[left] - target_ms) <= np.abs(
        source_ms[right] - target_ms
    )
    pick = np.where(closer_left, left, right)
    close = np.abs(source_ms[pick] - target_ms) <= tolerance_ms
    out[close] = values[pick][close]
    return out


def _join(state, alt, period_ms):
    """Flown xyz and active target xyz per ``state`` packet.

    z comes from the ``alt`` packet nearest on the device clock (within
    1.5 periods, else NaN). Both blocks share the device clock; their
    separately unwrapped counters are re-aligned if a wrap fell between
    their first samples.
    """
    alt_ms = alt["device_ms"]
    if len(alt_ms) and len(state["device_ms"]):
        shift = state["device_ms"][0] - alt_ms[0]
        alt_ms = alt_ms + CLOCK_WRAP * np.round(shift / CLOCK_WRAP)
    tol = 1.5 * period_ms
    z = _nearest(alt_ms, state["device_ms"], alt["z"], tol)
    tz = _nearest(alt_ms, state["device_ms"], alt["target_z"], tol)
    flown = np.column_stack([state["x"], state["y"], z]).reshape(-1, 3)
    target = np.column_stack(
        [state["target_x"], state["target_y"], tz]
    ).reshape(-1, 3)
    return flown, target


def load_run(run_dir):
    """Load everything one attempt wrote.

    Args:
        run_dir: A ``logs/<run_id>`` directory.

    Returns:
        Dict with ``dir``, ``cfg``, ``summary`` (``None`` while the run is
        unfinished), ``plan`` ``(t, xyz)``, ``plan_info`` (``plan.json``),
        ``commands``, ``state``, ``alt``, ``events``, and per ``state``
        packet the ``flown`` and active ``target`` xyz.
    """
    run_dir = Path(run_dir)
    summary_file = run_dir / "summary.json"
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    state = _packets(_read_csv(run_dir / "state.csv"), COLUMNS["state"])
    alt = _packets(_read_csv(run_dir / "alt.csv"), COLUMNS["alt"])
    flown, target = _join(state, alt, cfg["telemetry"]["period_ms"])
    return {
        "dir": run_dir,
        "cfg": cfg,
        "summary": (
            json.loads(summary_file.read_text())
            if summary_file.exists()
            else None
        ),
        "plan": trajectory.load_csv(run_dir / "trajectory.csv"),
        "plan_info": (
            json.loads((run_dir / "plan.json").read_text())
            if (run_dir / "plan.json").exists()
            else {}
        ),
        "commands": _commands(_read_csv(run_dir / "commands.csv")),
        "state": state,
        "alt": alt,
        "events": _read_csv(run_dir / "events.csv"),
        "flown": flown,
        "target": target,
    }


def trajectory_segment(run):
    """Flown vs active target during the trajectory phase.

    Both come from the same device packets, so the comparison needs no
    host-clock alignment.

    Returns:
        ``(tau, flown, target)`` with ``tau`` the device time since the
        first trajectory packet (s), or ``None`` without trajectory data.
    """
    state = run["state"]
    k = (state["phase"] == "trajectory") & np.isfinite(run["flown"][:, 2])
    if k.sum() < 2:
        return None
    ms = state["device_ms"][k]
    return (ms - ms[0]) / 1000.0, run["flown"][k], run["target"][k]


def data_quality(run):
    """Device-time gaps per block and command lateness."""
    period = run["cfg"]["telemetry"]["period_ms"]
    blocks = {}
    for name in ("state", "alt"):
        ms = run[name]["device_ms"]
        steps = np.diff(ms)
        span_s = (ms[-1] - ms[0]) / 1000 if len(ms) > 1 else 0.0
        blocks[name] = {
            "packets": int(len(ms)),
            "gaps": int((steps > 1.5 * period).sum()),
            "max_gap_ms": int(steps.max()) if len(steps) else None,
            "rate_hz": round((len(ms) - 1) / span_s, 2) if span_s else None,
        }
    late = run["commands"]["lateness"]
    late = late[np.isfinite(late)]
    lateness = {"streamed": int(len(late))}
    if len(late):
        lateness.update(
            p50_s=float(np.percentile(late, 50)),
            p99_s=float(np.percentile(late, 99)),
            max_s=float(late.max()),
        )
    return {"blocks": blocks, "lateness": lateness}


def metrics(run):
    """Tracking error, ring passage and data quality for one run.

    Tracking error is the estimate minus the active controller target
    from the same packet: flight performance, not process noise.

    Returns:
        ``{"tracking": {...} | None, "rings": [...] | None, "data": {...}}``.
    """
    data = data_quality(run)
    segment = trajectory_segment(run)
    if segment is None:
        return {"tracking": None, "rings": None, "data": data}
    _, flown, target = segment
    error = flown - target
    norm = np.linalg.norm(error, axis=1)
    tracking = {
        "reference": "active controller target (same packet)",
        "rmse_m": float(np.sqrt(np.mean(norm**2))),
        "max_m": float(norm.max()),
        "rmse_xyz_m": np.sqrt(np.mean(error**2, axis=0)).round(4).tolist(),
        "samples": int(len(norm)),
    }
    rings = run["cfg"]["environment"]["rings"]
    if not rings["enabled"]:
        return {"tracking": tracking, "rings": None, "data": data}
    distances = environment.ring_distances(flown, rings["regions"])
    passage = [
        {
            "name": ring["name"],
            "min_distance_m": d,
            "passed": d <= ring["radius"],
        }
        for ring, d in zip(rings["regions"], distances)
    ]
    return {"tracking": tracking, "rings": passage, "data": data}


def run_dirs(log_root):
    """Run directories under ``log_root``, oldest first."""
    log_root = Path(log_root)
    if not log_root.exists():
        return []
    return sorted(
        (d for d in log_root.iterdir() if (d / "config.yaml").exists()),
        key=lambda d: d.name,
    )


def summary_row(run):
    """One ``runs.csv`` row for a run (unfinished runs are recomputed)."""
    s = run["summary"] or {
        "run_id": run["dir"].name,
        "outcome": "unfinished",
        **metrics(run),
    }
    tracking = s.get("tracking") or {}
    rings = s.get("rings")
    data = s.get("data") or {}
    blocks = data.get("blocks", {}).values()
    max_gaps = [b["max_gap_ms"] for b in blocks if b["max_gap_ms"]]
    return {
        "run_id": s["run_id"],
        "method": run["plan_info"].get("method"),
        "p_lower": run["plan_info"].get("p_lower"),
        "outcome": s["outcome"],
        "last_phase": s.get("last_phase"),
        "dry_run": s.get("dry_run"),
        "duration_s": s.get("duration_s"),
        "rmse_m": tracking.get("rmse_m"),
        "max_m": tracking.get("max_m"),
        "rings_passed": (
            None if rings is None else sum(r["passed"] for r in rings)
        ),
        "dropped_rows": sum(data.get("dropped_rows", {}).values()),
        "gaps": sum(b["gaps"] for b in blocks),
        "max_gap_ms": max(max_gaps, default=None),
        "max_lateness_s": data.get("lateness", {}).get("max_s"),
        "error": s.get("error"),
    }


def write_table(rows, path):
    """Write ``runs.csv``."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def repeats_of(run, runs):
    """Completed runs that flew exactly the same planned trajectory."""
    plan = (run["dir"] / "trajectory.csv").read_bytes()
    return [
        r
        for r in runs
        if r["summary"]
        and r["summary"]["outcome"] == "completed"
        and (r["dir"] / "trajectory.csv").read_bytes() == plan
    ]


def _print_run(run):
    s = run["summary"] or {"outcome": "unfinished", **metrics(run)}
    method = run["plan_info"].get("method", "?")
    print(f"{run['dir'].name} [{method}]: {s['outcome']}", end="")
    if s.get("error"):
        print(f" ({s['error']})", end="")
    print()
    if "p_lower" in run["plan_info"]:
        print(f"  plan P(phi) >= {run['plan_info']['p_lower']:.3f}")
    if s.get("tracking"):
        tr = s["tracking"]
        print(f"  tracking RMSE {tr['rmse_m']:.3f} m, max {tr['max_m']:.3f} m")
    for ring in s.get("rings") or []:
        mark = "passed" if ring["passed"] else "MISSED"
        print(f"  {ring['name']}: {ring['min_distance_m']:.3f} m {mark}")
    data = s.get("data") or {}
    for name, block in data.get("blocks", {}).items():
        print(
            f"  {name}: {block['packets']} packets, {block['gaps']} gaps "
            f"(max {block['max_gap_ms']} ms), {block['rate_hz']} Hz"
        )
    late = data.get("lateness", {})
    if late.get("streamed"):
        print(
            f"  command lateness p99 {1000 * late['p99_s']:.1f} ms, "
            f"max {1000 * late['max_s']:.1f} ms"
        )
    dropped = sum(data.get("dropped_rows", {}).values())
    if dropped or data.get("writer_errors"):
        print(
            f"  dropped rows {dropped}, writer errors "
            f"{len(data.get('writer_errors', []))}"
        )


def report(logs, run_dir=None, all_runs=False, show=True):
    """Print and plot one run; with ``all_runs`` also tabulate every run.

    Args:
        logs: Directory holding the run directories.
        run_dir: Run to report; defaults to the latest.
        all_runs: Also write ``runs.csv`` and the repeats overlay.
        show: Show figures interactively before saving.

    Returns:
        Process exit code (1 if there is nothing to analyse).
    """
    import plots  # matplotlib only when plotting

    logs = Path(logs)
    dirs = run_dirs(logs)
    if run_dir is None and not dirs:
        print(f"No runs in {logs}")
        return 1
    run = load_run(run_dir or dirs[-1])
    _print_run(run)
    plots.plot_run(run, run["dir"] / "flight", show=show)
    if not all_runs:
        return 0

    runs = [load_run(d) for d in dirs]
    write_table([summary_row(r) for r in runs], logs / "runs.csv")
    print(f"{len(runs)} runs -> {logs / 'runs.csv'}")
    repeats = repeats_of(run, runs)
    if repeats:
        plots.plot_repeats(repeats, logs / "repeats", show=show)
        print(f"{len(repeats)} completed run(s) of this plan overlaid")
    return 0
