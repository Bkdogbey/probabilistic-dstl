"""
Flight data logger for Crazyflie experiment trials.

Two conditions (set via `python run.py fly --condition ...`):
    deterministic  — nominal safe path, no optimisation
    pdstl          — pDSTL-optimised path

See components/crazyflie.py for real usage (start/start_actual_logging/
log_waypoint/stop_actual_logging/save, in that order).

Output files (logs/ directory, next to this file):
    <condition>[_<scenario>]_fan<XX>_run<NN>_<ts>_commanded.csv  — one row per go_to call
    <condition>[_<scenario>]_fan<XX>_run<NN>_<ts>_actual.csv     — 10 Hz sampled Lighthouse position
    (the `_<scenario>` segment is omitted entirely for scenario='baseline',
    exactly matching today's filenames -- 'gate'/'figure8' get it inserted
    between condition and fan, same convention as components/config.py's
    _waypoints_path; crashed trials get _CRASH, trials with any unsafe sample
    get _VIOLATION, both inserted between the run tag and timestamp, in that
    order)

A trial whose actual position never moves more than START_TOLERANCE from its
own first sample (e.g. a calibration abort that fires before any real flight)
isn't saved at all — see save()'s early return.

Columns (both files):
    condition, scenario, t, x, y, z, outside_obs1, outside_obs2, outside_obs3, safe

Collision classification checks x, y AND z simultaneously against each
obstacle's full 3D box (obstacles default to components.config.OBSTACLES,
which already carries a canonical z-extent for every obstacle) -- a drone
flying above or below an obstacle's z-range is correctly logged as safe even
if its x,y falls inside the footprint.
"""

from __future__ import annotations

import csv
import math
import pathlib
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from components.config import LOGS_DIR
from components.config import OBSTACLES as _ENV_OBSTACLES
from components.config import START_TOLERANCE

# ── Valid condition labels ──────────────────────────────────────────────────
CONDITIONS = (
    'deterministic',
    'pdstl',
)

# ── Obstacles (x_min, x_max, y_min, y_max, z_min, z_max), sourced from the
#    arena geometry config in config.yml — nothing hardcoded here. Every
#    obstacle in components.config.OBSTACLES already carries a canonical
#    'z' extent (floor-mounted or explicit), so no new parsing is needed
#    here to make collision checks 3D-aware. ─────────────────────────────
_OBSTACLES: list[tuple[float, float, float, float, float, float]] = [
    (obs['x'][0], obs['x'][1], obs['y'][0], obs['y'][1], obs['z'][0], obs['z'][1])
    for obs in _ENV_OBSTACLES
]

# ── Log directory (shared with analyze_logs.py) ─────────────────────────────
_LOGS_DIR = LOGS_DIR

# Actual-position sampling rate (Hz) — CrazyflieBase updates at 20 Hz so 10 is safe
_SAMPLE_HZ = 10

# Regex to find existing run files for auto-increment
_RUN_RE = re.compile(r'_run(\d+)_')


def _inside(x: float, y: float, z: float, obs: tuple[float, float, float, float, float, float]) -> bool:
    """Return True if point (x, y, z) is inside the rectangular obstacle box.

    Checking all three axes (not just x, y) means a drone flying above or
    below an obstacle's z-range is correctly classified as safe, even with
    its x, y inside the footprint -- x,y-only would incorrectly flag that as
    unsafe for any 3D scenario that varies altitude.
    """
    x0, x1, y0, y1, z0, z1 = obs
    return x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1


def _safety_row(condition: str, scenario: str, t: float, x: float, y: float, z: float,
                obstacles: list[tuple[float, float, float, float, float, float]]) -> dict:
    outside = [not _inside(x, y, z, obs) for obs in obstacles]
    row: dict = {
        'condition': condition,
        'scenario': scenario,
        't': t,
        'x': round(x, 6),
        'y': round(y, 6),
        'z': round(z, 6),
    }
    for i, out in enumerate(outside, start=1):
        row[f'outside_obs{i}'] = int(out)
    row['safe'] = int(all(outside))
    return row


def _log_prefix(condition: str, scenario: str, fan_speed: int) -> str:
    """Filename prefix shared by save()'s stem and _next_run_number()'s glob.

    Mirrors components.config._waypoints_path's suffix convention exactly:
    scenario='baseline' adds nothing (today's filenames, unchanged),
    'gate'/'figure8' insert a '_<scenario>' segment before 'fan', so their
    run-numbering is scoped independently of baseline at the same fan level.
    """
    suffix = '' if scenario == 'baseline' else f'_{scenario}'
    return f'{condition}{suffix}_fan{fan_speed:02d}_run'


@dataclass
class FlightLogger:
    """
    Logs commanded waypoints and continuous Lighthouse position for one trial.

    Two output files are written on save() -- unless the trial never left the
    start region, in which case nothing is written (see _never_left_start()):
      - <condition>_fan<XX>_run<NN>_<ts>_commanded.csv  one row per go_to call
      - <condition>_fan<XX>_run<NN>_<ts>_actual.csv     10 Hz sampled real position

    The run number NN is auto-incremented by scanning the logs directory for
    existing files with the same (condition, fan_speed) so that successive calls
    of run.py automatically produce run01, run02, … without any manual editing.
    """

    condition: str
    fan_speed: int = 0
    scenario: str = 'baseline'
    obstacles: list[tuple[float, float, float, float, float, float]] = field(
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
                    _safety_row(self.condition, self.scenario, t, x, y, z, self.obstacles)
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
            _safety_row(self.condition, self.scenario, t, x, y, z, self.obstacles)
        )

    def mark_crashed(self) -> None:
        """Call from the exception handler to flag this trial as a crash."""
        self._crashed = True

    def _never_left_start(self) -> bool:
        """True if the actual trace never moved beyond START_TOLERANCE from its own first sample.

        Compares against the trial's own first sample rather than the nominal
        START_XY: a calibration-abort trial's hover position can itself differ
        from START_XY by more than START_TOLERANCE (that's why calibration
        aborted) without the drone having flown anywhere afterward -- what
        matters here is whether it moved *during the trial*, not where it
        happened to start.
        """
        if not self._actual:
            return True
        x0, y0 = self._actual[0]['x'], self._actual[0]['y']
        return all(
            math.hypot(r['x'] - x0, r['y'] - y0) <= START_TOLERANCE for r in self._actual
        )

    def _next_run_number(self) -> int:
        """Scan logs dir and return the next available run number for this cell."""
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        prefix = _log_prefix(self.condition, self.scenario, self.fan_speed)
        existing = list(_LOGS_DIR.glob(f'{prefix}*_actual.csv'))
        run_nums = [
            int(m.group(1))
            for f in existing
            if (m := _RUN_RE.search(f.name))
        ]
        return max(run_nums, default=0) + 1

    def save(self) -> tuple[pathlib.Path, pathlib.Path] | None:
        """Write both CSVs and return (commanded_path, actual_path).

        Returns None (writing nothing, consuming no run number) if the trial
        never left the start region -- see _never_left_start(). Checked before
        _next_run_number() is even called, so the next real trial still gets
        the correct next run number.
        """
        if self._never_left_start():
            print(
                f'[FlightLogger] Discarded: never moved beyond {START_TOLERANCE} m '
                f'from start -- not saved, run number not consumed.'
            )
            return None

        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        run_num = self._next_run_number()

        crash_tag = '_CRASH' if self._crashed else ''
        violated = any(not r['safe'] for r in self._actual)
        violation_tag = '_VIOLATION' if violated else ''
        prefix = _log_prefix(self.condition, self.scenario, self.fan_speed)
        stem = f'{prefix}{run_num:02d}{crash_tag}{violation_tag}_{ts}'
        cmd_path = _LOGS_DIR / f'{stem}_commanded.csv'
        act_path = _LOGS_DIR / f'{stem}_actual.csv'

        self._write_csv(cmd_path, self._commanded, label='commanded')
        self._write_csv(act_path, self._actual, label='actual')

        print(f'[FlightLogger] Run {run_num:02d} | condition={self.condition} | '
              f'scenario={self.scenario} | fan={self.fan_speed}')
        if self._crashed:
            print('[FlightLogger] *** CRASH flagged — partial data saved ***')
        if violated:
            print('[FlightLogger] *** VIOLATION flagged — an actual sample entered an obstacle ***')

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
