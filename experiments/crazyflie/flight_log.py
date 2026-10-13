"""Per-attempt flight logs, written by one queued writer thread.

Each attempt gets its own directory ``<logging.dir>/<run_id>/`` holding
``config.yaml``, the planned ``trajectory.csv`` and ``plan.json``, and:

- ``commands.csv``: every command (send time, reference time, lateness)
- ``state.csv`` / ``alt.csv``: one row per device log packet (device and
  host receive time)
- ``events.csv``: phase changes, aborts and log-block errors
- ``summary.json``: outcome, metrics and data quality

Producers (the executor, telemetry callbacks) only ``put_nowait`` rows into
a bounded queue; a full queue drops the row and counts it. The writer
flushes after every drained batch, so a crash leaves at most one batch
unwritten.
"""

from __future__ import annotations

import csv
import json
import logging
import queue
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
FILES = {
    "commands": [
        "t_send",
        "t_ref",
        "k",
        "x",
        "y",
        "z",
        "yaw",
        "lateness_s",
        "phase",
    ],
    "state": [
        "device_ms",
        "t_recv",
        "x",
        "y",
        "vx",
        "vy",
        "target_x",
        "target_y",
        "phase",
    ],
    "alt": ["device_ms", "t_recv", "z", "vz", "target_z", "vbat", "phase"],
    "events": ["t", "kind", "detail"],
}
_STOP = object()


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
    """One run directory, its CSV files and the writer thread."""

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
        self._queue = queue.Queue(maxsize=cfg["telemetry"]["queue_size"])
        self._files = {
            name: open(self.dir / f"{name}.csv", "w", newline="")
            for name in FILES
        }
        self._writers = {
            name: csv.writer(f) for name, f in self._files.items()
        }
        for name, header in FILES.items():
            self._writers[name].writerow(header)
        self._rows = dict.fromkeys(FILES, 0)
        self._dropped = dict.fromkeys(FILES, 0)
        self._writer_errors = []
        self._max_depth = 0
        self._max_write_lag = 0.0
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    # --- producers (never block) ------------------------------------------

    def now(self):
        """Seconds since the run started (host monotonic clock)."""
        return time.monotonic() - self.t0

    def enqueue(self, name, row):
        """Queue one row for ``<name>.csv``; drop and count it if full."""
        try:
            self._queue.put_nowait((name, row))
        except queue.Full:
            self._dropped[name] += 1
            return
        self._max_depth = max(self._max_depth, self._queue.qsize())

    def set_phase(self, phase):
        self.phase = phase
        self.event("phase", phase)

    def event(self, kind, detail=""):
        self.enqueue("events", [self.now(), kind, detail])

    def command(self, xyz, t_ref=None, k=None, lateness=None):
        """Record a sent position command (yaw is always 0)."""
        x, y, z = (float(v) for v in xyz)
        self.enqueue(
            "commands",
            [self.now(), t_ref, k, x, y, z, 0.0, lateness, self.phase],
        )

    def packet(self, block, device_ms, t_recv, values):
        """``StateFeed`` sink: one row per device packet."""
        row = [device_ms, t_recv - self.t0, *values, self.phase]
        self.enqueue(block, row)

    def block_error(self, block, message):
        """``StateFeed`` error hook."""
        self.event("log_block_error", f"{block}: {message}")

    # --- writer -------------------------------------------------------------

    def _write_loop(self):
        while True:
            batch = [self._queue.get()]
            while True:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            done = False
            for item in batch:
                if item is _STOP:
                    done = True
                else:
                    self._write(*item)
            for f in self._files.values():
                f.flush()
            if done:
                return

    def _write(self, name, row):
        try:
            self._writers[name].writerow(row)
            self._rows[name] += 1
            if name in ("state", "alt"):
                self._max_write_lag = max(
                    self._max_write_lag, self.now() - row[1]
                )
        except Exception as exc:  # recorded in the summary, never raised
            if len(self._writer_errors) < 20:
                self._writer_errors.append(f"{name}: {exc}")

    def _close(self):
        self._queue.put(_STOP)
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            self._writer_errors.append("writer did not finish in 10 s")
        for f in self._files.values():
            f.close()

    # --- summary ------------------------------------------------------------

    def _metrics(self):
        """Tracking, ring and data-quality metrics from the written files."""
        try:
            return analysis.metrics(analysis.load_run(self.dir))
        except Exception as exc:  # keep the summary even if analysis fails
            logger.exception("Could not compute run metrics")
            return {
                "tracking": None,
                "rings": None,
                "data": {},
                "metrics_error": str(exc),
            }

    def finish(self, outcome, error=None, phase=None, traceback=None):
        """Drain and close the files, then write summary.json.

        ``phase`` is the last phase reached before any cleanup landing.
        """
        if error:
            self.event("abort", error)
        self._close()
        metrics = self._metrics()
        metrics["data"].update(
            rows=dict(self._rows),
            dropped_rows=dict(self._dropped),
            writer_errors=list(self._writer_errors),
            max_queue_depth=self._max_depth,
            max_write_lag_s=round(self._max_write_lag, 4),
        )
        summary = {
            "run_id": self.run_id,
            "outcome": outcome,
            "last_phase": phase or self.phase,
            "error": error,
            "traceback": traceback,
            "started_utc": self.started.isoformat(),
            "ended_utc": _utc_now().isoformat(),
            "duration_s": self.now(),
            **metrics,
            **self.meta,
        }
        with open(self.dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return summary
