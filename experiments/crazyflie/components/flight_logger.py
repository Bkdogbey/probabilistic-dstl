"""Threaded XYZ flight logging and log-file persistence."""

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

from .utils import (
    FIG8_OBSTACLES as _ENV_FIG8_OBSTACLES,
    LOGS_2D_DIR,
    OBSTACLES as _ENV_OBSTACLES,
    START_TOLERANCE,
    VALID_SCENARIOS,
    logs_directory,
)

CONDITIONS = (
    'deterministic',
    'pdstl',
)

_OBSTACLES: list[tuple[float, float, float, float, float, float]] = [
    (obs['x'][0], obs['x'][1], obs['y'][0], obs['y'][1], obs['z'][0], obs['z'][1])
    for obs in _ENV_OBSTACLES
]
_FIG8_OBSTACLES: list[tuple[float, float, float, float, float, float]] = [
    (obs['x'][0], obs['x'][1], obs['y'][0], obs['y'][1], obs['z'][0], obs['z'][1])
    for obs in _ENV_FIG8_OBSTACLES
]

_LOGS_DIR: pathlib.Path | None = None
_SAMPLE_HZ = 10
_RUN_RE = re.compile(r'_run(\d+)_')
_SCENARIO_ALT = '|'.join(s for s in VALID_SCENARIOS if s != 'baseline')
_LOG_NAME_RE = re.compile(
    rf'^(?P<condition>{"|".join(CONDITIONS)})(?:_(?P<scenario>{_SCENARIO_ALT}))?'
    rf'_fan(?P<fan>\d+)_run(?P<run>\d+)(?P<crash>_CRASH)?(?P<violation>_VIOLATION)?_'
)


@dataclass(frozen=True)
class LogIdentity:
    condition: str
    scenario: str
    fan: int
    run: int
    crashed: bool
    violated: bool


class FlightLogRepository:
    def __init__(self, directory: pathlib.Path | None = None) -> None:
        self.directory = directory or LOGS_2D_DIR

    @staticmethod
    def prefix(condition: str, scenario: str, fan: int) -> str:
        suffix = '' if scenario == 'baseline' else f'_{scenario}'
        return f'{condition}{suffix}_fan{fan:02d}_run'

    def next_run_number(self, condition: str, scenario: str, fan: int) -> int:
        self.directory.mkdir(parents=True, exist_ok=True)
        prefix = self.prefix(condition, scenario, fan)
        run_numbers = [
            int(match.group(1))
            for path in self.directory.glob(f'{prefix}*_actual.csv')
            if (match := _RUN_RE.search(path.name))
        ]
        return max(run_numbers, default=0) + 1

    def find(
        self, condition: str, scenario: str, fan: int, run: int | None = None,
    ) -> tuple[pathlib.Path, pathlib.Path | None, LogIdentity]:
        prefix = self.prefix(condition, scenario, fan)
        candidates = []
        for path in self.directory.glob(f'{prefix}*_actual.csv'):
            identity = self.parse(path)
            if run is None or identity.run == run:
                candidates.append((identity.run, path, identity))
        if not candidates:
            raise FileNotFoundError(
                f'No logs for condition={condition}, scenario={scenario}, fan={fan}'
                + (f', run={run}' if run is not None else '') + f' in {self.directory}'
            )
        _, actual_path, identity = max(candidates, key=lambda item: item[0])
        commanded_path = actual_path.with_name(actual_path.name.replace('_actual.csv', '_commanded.csv'))
        return actual_path, commanded_path if commanded_path.exists() else None, identity

    @staticmethod
    def parse(path: pathlib.Path) -> LogIdentity:
        match = _LOG_NAME_RE.match(path.name)
        if match is None:
            raise ValueError(f'Unrecognised log filename: {path.name}')
        return LogIdentity(
            condition=match.group('condition'),
            scenario=match.group('scenario') or 'baseline',
            fan=int(match.group('fan')),
            run=int(match.group('run')),
            crashed=match.group('crash') is not None,
            violated=match.group('violation') is not None,
        )

    @staticmethod
    def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
        if not rows:
            return
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


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

    Uses the same scenario suffix convention as waypoint plans:
    scenario='baseline' adds nothing (today's filenames, unchanged),
    Figure-eight inserts a '_figure8' segment before 'fan', so its
    run-numbering is scoped independently of baseline at the same fan level.
    """
    return FlightLogRepository.prefix(condition, scenario, fan_speed)


@dataclass
class FlightLogger:
    """
    Logs commanded waypoints and continuous Lighthouse position for one trial.

    Two output files are written on save() -- unless the trial never left the
    start region, in which case nothing is written (see _never_left_start()):
      - <condition>_fan<XX>_run<NN>_<ts>_commanded.csv  one row per go_to call
      - <condition>_fan<XX>_run<NN>_<ts>_actual.csv     10 Hz sampled real position

    Run numbers auto-increment within each condition, scenario, and fan cell.
    """

    condition: str
    fan_speed: int = 0
    scenario: str = 'baseline'
    obstacles: list[tuple[float, float, float, float, float, float]] | None = None

    _t0: float = field(init=False, default=0.0)
    _commanded: list[dict] = field(init=False, default_factory=list)
    _actual: list[dict] = field(init=False, default_factory=list)
    _return_phase: list[dict] = field(init=False, default_factory=list)
    _stop_event: threading.Event = field(init=False, default_factory=threading.Event)
    _thread: threading.Thread | None = field(init=False, default=None)
    _crashed: bool = field(init=False, default=False)
    _stem: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(
                f'Unknown condition {self.condition!r}. '
                f'Choose from: {CONDITIONS}'
            )
        if self.obstacles is None:
            source = _FIG8_OBSTACLES if self.scenario == 'figure8' else _OBSTACLES
            self.obstacles = list(source)

    def start(self) -> None:
        """Record start time. Call just before the first go_to."""
        self._t0 = time.monotonic()
        self._commanded = []
        self._actual = []
        self._return_phase = []

    def start_actual_logging(
        self, get_pos: Callable[[], tuple[float, float, float]]
    ) -> None:
        """
        Begin sampling actual Lighthouse position at _SAMPLE_HZ in a background thread.

        Args:
            get_pos: callable returning (x, y, z). Pass CrazyflieBase's
                     state_snapshot method so all axes come from one radio packet.
        """
        self._start_logging(get_pos, self._actual)

    def start_return_logging(
        self, get_pos: Callable[[], tuple[float, float, float]]
    ) -> None:
        """Sample position through the return-and-land phase, for debugging.

        Written to a separate <stem>_return.csv on save_return_phase() --
        kept out of _actual.csv so it never affects analysis or calibration,
        which assume only the forward mission is logged.
        """
        self._start_logging(get_pos, self._return_phase)

    def _start_logging(
        self, get_pos: Callable[[], tuple[float, float, float]], target: list[dict],
    ) -> None:
        self._stop_event.clear()
        interval = 1.0 / _SAMPLE_HZ

        def _loop() -> None:
            while not self._stop_event.is_set():
                tick = time.monotonic()
                x, y, z = get_pos()
                t = round(tick - self._t0, 4)
                target.append(
                    _safety_row(self.condition, self.scenario, t, x, y, z, self.obstacles)
                )
                elapsed = time.monotonic() - tick
                remaining = interval - elapsed
                if remaining > 0:
                    self._stop_event.wait(timeout=remaining)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop_actual_logging(self) -> None:
        """Stop whichever background sampling thread is running. Call before save()."""
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
        START_XY: the drone may legitimately take off anywhere in the tracked
        workspace, so its actual start can itself differ from START_XY by more
        than START_TOLERANCE without anything having gone wrong -- what
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
        return FlightLogRepository(_LOGS_DIR or logs_directory(self.scenario)).next_run_number(
            self.condition, self.scenario, self.fan_speed,
        )

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

        directory = _LOGS_DIR or logs_directory(self.scenario)
        directory.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        run_num = self._next_run_number()

        crash_tag = '_CRASH' if self._crashed else ''
        violated = any(not r['safe'] for r in self._actual)
        violation_tag = '_VIOLATION' if violated else ''
        prefix = _log_prefix(self.condition, self.scenario, self.fan_speed)
        stem = f'{prefix}{run_num:02d}{crash_tag}{violation_tag}_{ts}'
        self._stem = stem
        cmd_path = directory / f'{stem}_commanded.csv'
        act_path = directory / f'{stem}_actual.csv'

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

    def save_return_phase(self) -> pathlib.Path | None:
        """Write the return-and-land phase to <stem>_return.csv, for debugging.

        Reuses the stem save() chose for the mission files, so the two stay
        correlated. Returns None if there's nothing to write (no data, or
        save() was never called so no stem exists yet).
        """
        if not self._return_phase or self._stem is None:
            return None
        directory = _LOGS_DIR or logs_directory(self.scenario)
        path = directory / f'{self._stem}_return.csv'
        self._write_csv(path, self._return_phase, label='return')
        return path

    def _write_csv(self, path: pathlib.Path, rows: list[dict], label: str) -> None:
        if not rows:
            print(f'[FlightLogger] No {label} data to save.')
            return
        FlightLogRepository.write_csv(path, rows)
        n_unsafe = sum(1 for r in rows if not r['safe'])
        print(
            f'[FlightLogger] {label}: {len(rows)} rows '
            f'({n_unsafe} unsafe) → {path}'
        )
