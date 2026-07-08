"""
Flight data logger for Crazyflie experiment trials.

Two conditions:
    deterministic  — original sine path (USE_OPTIMISED = False)
    pdstl          — pDSTL-optimised path (USE_OPTIMISED = True)

Usage (from crazyflie.py):
    logger = FlightLogger("pdstl", fan_speed=12)
    logger.start()
    logger.start_actual_logging(lambda: (cf_base.current_x,
                                         cf_base.current_y,
                                         cf_base.current_z))
    ...
    logger.log_waypoint(x, y, z)   # called at each go_to
    ...
    logger.stop_actual_logging()
    logger.save()

Output files (logs/ directory, next to this file):
    <condition>_fan<XX>_run<NN>_<ts>_commanded.csv  — one row per go_to call
    <condition>_fan<XX>_run<NN>_<ts>_actual.csv     — 10 Hz sampled Lighthouse position
    (crashed trials get _CRASH_ between run tag and timestamp)

Columns (both files):
    condition, t, x, y, z, outside_obs1, outside_obs2, outside_obs3, safe
"""

from __future__ import annotations

import csv
import pathlib
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

# ── Valid condition labels ──────────────────────────────────────────────────
CONDITIONS = (
    'deterministic',
    'pdstl',
)

# ── Obstacles (x_min, x_max, y_min, y_max) — must match generate_waypoints.py ─
_OBSTACLES: list[tuple[float, float, float, float]] = [
    (-0.165,  0.165, -1.144, -0.941),  # OBS-1 (red)
    (-0.432, -0.102, -0.179,  0.049),  # OBS-2 (blue)
    ( 0.114,  0.343, -0.611, -0.421),  # OBS-3 (green)
]

# ── Log directory: next to this file ────────────────────────────────────────
_LOGS_DIR = pathlib.Path(__file__).parent / 'logs'

# Actual-position sampling rate (Hz) — CrazyflieBase updates at 20 Hz so 10 is safe
_SAMPLE_HZ = 10

# Regex to find existing run files for auto-increment
_RUN_RE = re.compile(r'_run(\d+)_')


def _inside(x: float, y: float, obs: tuple[float, float, float, float]) -> bool:
    """Return True if point (x, y) is inside rectangular obstacle."""
    x0, x1, y0, y1 = obs
    return x0 <= x <= x1 and y0 <= y <= y1


def _safety_row(condition: str, t: float, x: float, y: float, z: float,
                obstacles: list[tuple[float, float, float, float]]) -> dict:
    outside = [not _inside(x, y, obs) for obs in obstacles]
    row: dict = {
        'condition': condition,
        't': t,
        'x': round(x, 6),
        'y': round(y, 6),
        'z': round(z, 6),
    }
    for i, out in enumerate(outside, start=1):
        row[f'outside_obs{i}'] = int(out)
    row['safe'] = int(all(outside))
    return row


@dataclass
class FlightLogger:
    """
    Logs commanded waypoints and continuous Lighthouse position for one trial.

    Two output files are written on save():
      - <condition>_fan<XX>_run<NN>_<ts>_commanded.csv  one row per go_to call
      - <condition>_fan<XX>_run<NN>_<ts>_actual.csv     10 Hz sampled real position

    The run number NN is auto-incremented by scanning the logs directory for
    existing files with the same (condition, fan_speed) so that successive calls
    of main.py automatically produce run01, run02, … without any manual editing.
    """

    condition: str
    fan_speed: int = 0
    obstacles: list[tuple[float, float, float, float]] = field(
        default_factory=lambda: list(_OBSTACLES)
    )

    _t0: float = field(init=False, default=0.0)
    _commanded: list[dict] = field(init=False, default_factory=list)
    _actual: list[dict] = field(init=False, default_factory=list)
    _stop_event: threading.Event = field(init=False, default_factory=threading.Event)
    _thread: threading.Thread | None = field(init=False, default=None)
    _crashed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(
                f'Unknown condition {self.condition!r}. '
                f'Choose from: {CONDITIONS}'
            )

    def start(self) -> None:
        """Record start time. Call just before the first go_to."""
        self._t0 = time.monotonic()
        self._commanded = []
        self._actual = []

    def start_actual_logging(
        self, get_pos: Callable[[], tuple[float, float, float]]
    ) -> None:
        """
        Begin sampling actual Lighthouse position at _SAMPLE_HZ in a background thread.

        Args:
            get_pos: callable returning (x, y, z) from CrazyflieBase.current_x/y/z.
                     Example: lambda: (cf_base.current_x, cf_base.current_y, cf_base.current_z)
        """
        self._stop_event.clear()
        interval = 1.0 / _SAMPLE_HZ

        def _loop() -> None:
            while not self._stop_event.is_set():
                tick = time.monotonic()
                x, y, z = get_pos()
                t = round(tick - self._t0, 4)
                self._actual.append(
                    _safety_row(self.condition, t, x, y, z, self.obstacles)
                )
                elapsed = time.monotonic() - tick
                remaining = interval - elapsed
                if remaining > 0:
                    self._stop_event.wait(timeout=remaining)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop_actual_logging(self) -> None:
        """Stop the background sampling thread. Call before save()."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def log_waypoint(self, x: float, y: float, z: float) -> None:
        """Log one commanded waypoint with elapsed time and safety flags."""
        t = round(time.monotonic() - self._t0, 4)
        self._commanded.append(
            _safety_row(self.condition, t, x, y, z, self.obstacles)
        )

    def mark_crashed(self) -> None:
        """Call from the exception handler to flag this trial as a crash."""
        self._crashed = True

    def _next_run_number(self) -> int:
        """Scan logs dir and return the next available run number for this cell."""
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        prefix = f'{self.condition}_fan{self.fan_speed:02d}_run'
        existing = list(_LOGS_DIR.glob(f'{prefix}*_actual.csv'))
        run_nums = [
            int(m.group(1))
            for f in existing
            if (m := _RUN_RE.search(f.name))
        ]
        return max(run_nums, default=0) + 1

    def save(self) -> tuple[pathlib.Path, pathlib.Path]:
        """Write both CSVs and return (commanded_path, actual_path)."""
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        run_num = self._next_run_number()

        fan_tag = f'fan{self.fan_speed:02d}'
        run_tag = f'run{run_num:02d}'
        crash_tag = '_CRASH' if self._crashed else ''
        stem = f'{self.condition}_{fan_tag}_{run_tag}{crash_tag}_{ts}'
        cmd_path = _LOGS_DIR / f'{stem}_commanded.csv'
        act_path = _LOGS_DIR / f'{stem}_actual.csv'

        self._write_csv(cmd_path, self._commanded, label='commanded')
        self._write_csv(act_path, self._actual, label='actual')

        print(f'[FlightLogger] Run {run_num:02d} | condition={self.condition} | fan={self.fan_speed}')
        if self._crashed:
            print('[FlightLogger] *** CRASH flagged — partial data saved ***')

        if self._actual:
            xs = [r['x'] for r in self._actual]
            ys = [r['y'] for r in self._actual]
            zs = [r['z'] for r in self._actual]
            print(
                f'[FlightLogger] Mean position — '
                f'x={sum(xs)/len(xs):.3f} m  '
                f'y={sum(ys)/len(ys):.3f} m  '
                f'z={sum(zs)/len(zs):.3f} m  '
                f'({len(self._actual)} samples)'
            )

        return cmd_path, act_path

    def _write_csv(self, path: pathlib.Path, rows: list[dict], label: str) -> None:
        if not rows:
            print(f'[FlightLogger] No {label} data to save.')
            return
        fieldnames = list(rows[0].keys())
        with path.open('w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        n_unsafe = sum(1 for r in rows if not r['safe'])
        print(
            f'[FlightLogger] {label}: {len(rows)} rows '
            f'({n_unsafe} unsafe) → {path}'
        )
