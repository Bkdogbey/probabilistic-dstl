"""Per-attempt flight logs.

Each attempt gets its own directory ``<logging.dir>/<run_id>/`` holding
``config.yaml``, the planned ``trajectory.csv`` and ``plan.json``, the
streamed ``commands.csv``, the sampled ``telemetry.csv`` and a final
``summary.json``. CSV rows are flushed as they are written so a crash or
hard kill still leaves the partial logs on disk.
"""

from __future__ import annotations

import csv
import json
import logging
import secrets
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

import analysis
import config
import trajectory

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
COMMAND_FIELDS = ["t", "phase", "x_cmd", "y_cmd", "z_cmd"]
TELEMETRY_FIELDS = ["t", "phase", "x", "y", "z", "vx", "vy", "yaw"]


def _utc_now():
    return datetime.now(timezone.utc)


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=HERE,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class FlightLog:
    """Owns one run directory, its CSV files and the telemetry thread."""

    def __init__(self, cfg, t, xyz, plan_info=None, dry_run=False):
        self.started = _utc_now()
        stamp = self.started.strftime("%Y%m%dT%H%M%SZ")
        method = cfg["planning"]["method"]
        self.run_id = f"{stamp}_{cfg['name']}_{method}_{secrets.token_hex(2)}"
        self.dir = config.resolve(cfg, "dir") / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)

        self.cfg = cfg
        self.meta = {
            "method": method,
            "plan_signature": (plan_info or {}).get("signature"),
            "uri": cfg["radio"]["uri"],
            "dry_run": dry_run,
            "git_commit": _git_commit(),
        }
        with open(self.dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        trajectory.save_csv(self.dir / "trajectory.csv", t, xyz)
        if plan_info is not None:
            (self.dir / "plan.json").write_text(
                json.dumps(plan_info, indent=2)
            )

        self.phase = "init"
        self.t0 = time.monotonic()
        self._lock = threading.Lock()
        self._cmd_file = open(self.dir / "commands.csv", "w", newline="")
        self._tel_file = open(self.dir / "telemetry.csv", "w", newline="")
        self._cmd = csv.writer(self._cmd_file)
        self._tel = csv.writer(self._tel_file)
        self._cmd.writerow(COMMAND_FIELDS)
        self._tel.writerow(TELEMETRY_FIELDS)
        self._n_cmd = 0
        self._n_tel = 0
        self._stop = threading.Event()
        self._thread = None

    def now(self):
        return time.monotonic() - self.t0

    def set_phase(self, phase):
        self.phase = phase

    def command(self, xyz):
        """Record a commanded position at the current time and phase."""
        row = [self.now(), self.phase, *(float(v) for v in xyz)]
        with self._lock:
            if self._cmd_file.closed:
                return
            self._cmd.writerow(row)
            self._cmd_file.flush()
            self._n_cmd += 1

    def state(self, drone):
        """Record the drone's current estimated state."""
        row = [
            self.now(),
            self.phase,
            drone.current_x,
            drone.current_y,
            drone.current_z,
            drone.current_vx,
            drone.current_vy,
            drone.current_yaw,
        ]
        with self._lock:
            if self._tel_file.closed:
                return
            self._tel.writerow(row)
            self._tel_file.flush()
            self._n_tel += 1

    def start_telemetry(self, drone):
        period = 1.0 / self.cfg["logging"]["telemetry_hz"]

        def loop():
            while not self._stop.is_set():
                self.state(drone)
                self._stop.wait(period)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_telemetry(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _metrics(self):
        """Tracking and ring metrics from the files just written."""
        try:
            return analysis.metrics(analysis.load_run(self.dir))
        except Exception as exc:  # keep the summary even if analysis fails
            logger.exception("Could not compute run metrics")
            return {"tracking": None, "rings": None, "metrics_error": str(exc)}

    def finish(self, outcome, error=None, phase=None, traceback=None):
        """Close the CSVs and write summary.json; returns the summary.

        ``phase`` is the last phase reached before any cleanup landing.
        """
        self.stop_telemetry()
        with self._lock:
            self._cmd_file.close()
            self._tel_file.close()
        summary = {
            "run_id": self.run_id,
            "outcome": outcome,
            "last_phase": phase or self.phase,
            "error": error,
            "traceback": traceback,
            "started_utc": self.started.isoformat(),
            "ended_utc": _utc_now().isoformat(),
            "duration_s": self.now(),
            "command_rows": self._n_cmd,
            "telemetry_rows": self._n_tel,
            **self._metrics(),
            **self.meta,
        }
        with open(self.dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return summary
