"""Flight analysis: load run directories, compute metrics, compare runs.

Used by ``run.py analyze`` (and by ``FlightLog`` for the summary metrics):

- one run: print outcome, plan probability, tracking error and ring
  passage; save ``flight.png/.pdf`` in the run directory.
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
    "error",
]


def _read_log(path, columns):
    """Read a flight CSV into ``t[N]``, ``phase[N]`` and ``xyz[N, 3]``."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "t": np.array([float(r["t"]) for r in rows]),
        "phase": np.array([r["phase"] for r in rows], dtype=str),
        "xyz": np.array(
            [[float(r[c]) for c in columns] for r in rows]
        ).reshape(-1, 3),
    }


def load_run(run_dir):
    """Load everything one attempt wrote.

    Args:
        run_dir: A ``logs/<run_id>`` directory.

    Returns:
        Dict with ``dir``, ``cfg``, ``summary`` (``None`` while the run is
        unfinished), ``plan`` ``(t, xyz)``, ``plan_info`` (``plan.json``),
        ``commands`` and ``telemetry``
        (each ``{"t", "phase", "xyz"}``).
    """
    run_dir = Path(run_dir)
    summary_file = run_dir / "summary.json"
    return {
        "dir": run_dir,
        "cfg": yaml.safe_load((run_dir / "config.yaml").read_text()),
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
        "commands": _read_log(
            run_dir / "commands.csv", ["x_cmd", "y_cmd", "z_cmd"]
        ),
        "telemetry": _read_log(run_dir / "telemetry.csv", ["x", "y", "z"]),
    }


def trajectory_segment(run):
    """Flown vs commanded positions during the trajectory phase.

    Returns:
        ``(tau, flown, reference)`` with ``tau`` the time since the first
        streamed setpoint and ``reference`` the commanded position
        interpolated at each telemetry sample, or ``None`` if the run never
        streamed the trajectory.
    """
    cmd, tel = run["commands"], run["telemetry"]
    c = cmd["phase"] == "trajectory"
    if c.sum() < 2:
        return None
    t_cmd, xyz_cmd = cmd["t"][c], cmd["xyz"][c]
    k = (tel["phase"] == "trajectory") & (tel["t"] >= t_cmd[0])
    k &= tel["t"] <= t_cmd[-1]
    if not k.any():
        return None
    reference = np.column_stack(
        [np.interp(tel["t"][k], t_cmd, xyz_cmd[:, i]) for i in range(3)]
    )
    return tel["t"][k] - t_cmd[0], tel["xyz"][k], reference


def metrics(run):
    """Tracking error and ring passage for one run.

    Returns:
        ``{"tracking": {...} | None, "rings": [...] | None}``.
    """
    segment = trajectory_segment(run)
    if segment is None:
        return {"tracking": None, "rings": None}
    _, flown, reference = segment
    error = flown - reference
    norm = np.linalg.norm(error, axis=1)
    tracking = {
        "rmse_m": float(np.sqrt(np.mean(norm**2))),
        "max_m": float(norm.max()),
        "rmse_xyz_m": np.sqrt(np.mean(error**2, axis=0)).round(4).tolist(),
        "samples": int(len(norm)),
    }
    rings = run["cfg"]["environment"]["rings"]
    if not rings["enabled"]:
        return {"tracking": tracking, "rings": None}
    distances = environment.ring_distances(flown, rings["regions"])
    passage = [
        {
            "name": ring["name"],
            "min_distance_m": d,
            "passed": d <= ring["radius"],
        }
        for ring, d in zip(rings["regions"], distances)
    ]
    return {"tracking": tracking, "rings": passage}


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
