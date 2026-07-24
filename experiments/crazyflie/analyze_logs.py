"""Post-flight analysis for the Crazyflie experiment.

Reads the commanded/actual CSVs written by components/flight_logger.py and
plots the actual flown path against the planned one (pdSTL-optimised or
nominal safe path) on the same arena drawing used by waypoint_planning.py, so
a flight can be visually compared to the offline plan it was supposed to fly.

Called by `run.py analyze`; has no hardware/ROS dependency (only torch/numpy,
via components.planning_2d, plus matplotlib) -- same weight class as `plan`.
Only ever plots against the 2D baseline plan/environment -- there's no
--scenario flag here yet, so a gate-scenario run's plot would draw the wrong
arena.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re

import numpy as np

from components.config import (
    EXPERIMENT_DIR,
    LOGS_DIR,
    VALID_FANS,
    load_pdstl_waypoints,
    pdstl_plan_meta,
)
from components.flight_logger import CONDITIONS
from components.planning_2d import build_environment_2d, nominal_safe_waypoints

_LOGS_DIR = LOGS_DIR
_RUN_RE = re.compile(r'_run(\d+)_')
_LOG_NAME_RE = re.compile(
    r'^(?P<condition>\w+?)_fan(?P<fan>\d+)_run(?P<run>\d+)(?P<crash>_CRASH)?(?:_VIOLATION)?_'
)


def _find_run(
    condition: str, fan: int, run: int | None,
) -> tuple[pathlib.Path, pathlib.Path | None, int, bool]:
    """Locate the actual/commanded CSVs for one (condition, fan, run) triple.

    run=None picks the highest-numbered run for that cell (the latest trial).
    """
    prefix = f'{condition}_fan{fan:02d}_run'
    tagged = [
        (int(m.group(1)), path)
        for path in _LOGS_DIR.glob(f'{prefix}*_actual.csv')
        if (m := _RUN_RE.search(path.name))
    ]
    if run is not None:
        tagged = [(n, p) for n, p in tagged if n == run]
    if not tagged:
        raise FileNotFoundError(
            f'No logs for condition={condition} fan={fan}'
            + (f' run={run}' if run is not None else '') + f' in {_LOGS_DIR}'
        )
    run_num, actual_path = max(tagged, key=lambda t: t[0])
    crashed = '_CRASH' in actual_path.name
    commanded_path = pathlib.Path(str(actual_path).replace('_actual.csv', '_commanded.csv'))
    return actual_path, (commanded_path if commanded_path.exists() else None), run_num, crashed


def _read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(newline='') as fh:
        return list(csv.DictReader(fh))


def _summarize(rows: list[dict]) -> dict:
    unsafe = sum(1 for r in rows if int(r['safe']) == 0)
    return {
        'n': len(rows),
        'duration_s': float(rows[-1]['t']) - float(rows[0]['t']),
        'unsafe_samples': unsafe,
        'unsafe_frac': unsafe / len(rows),
    }


def _planned_path(condition: str, fan: int) -> np.ndarray:
    wps = load_pdstl_waypoints(fan) if condition == 'pdstl' else nominal_safe_waypoints()
    return np.array(wps)[:, :2]


def plot_run(condition: str, fan: int, run: int | None = None) -> pathlib.Path:
    """Plot one logged run against its planned path; returns the saved PNG path."""
    from waypoint_planning import _draw_env_2d  # same flat arena drawing as baseline `plan`

    import matplotlib.pyplot as plt

    from visualization.style import PALETTE, figsize as _figsize, save_figure

    actual_path, commanded_path, run_num, crashed = _find_run(condition, fan, run)
    actual_rows = _read_csv(actual_path)
    commanded_rows = _read_csv(commanded_path) if commanded_path else []
    summary = _summarize(actual_rows)
    planned_xy = _planned_path(condition, fan)

    fig, ax = plt.subplots(figsize=_figsize("single", aspect=1.0))
    _draw_env_2d(ax, build_environment_2d())

    title = f'{condition} fan {fan}  —  run {run_num:02d}'
    if crashed:
        title += '  (CRASH)'
    ax.set_title(title)

    ax.plot(
        planned_xy[:, 0], planned_xy[:, 1],
        color=PALETTE["plan"]["stroke"], linestyle='--', lw=1.5, alpha=0.7, label='planned',
    )

    ax_x = [float(r['x']) for r in actual_rows]
    ax_y = [float(r['y']) for r in actual_rows]
    safe = [int(r['safe']) for r in actual_rows]
    ax.plot(
        ax_x, ax_y,
        color=PALETTE["ego"]["stroke"], linestyle='-', lw=1.2, alpha=0.8, label='actual (flown)',
    )
    unsafe_xy = [(x, y) for x, y, s in zip(ax_x, ax_y, safe) if not s]
    if unsafe_xy:
        ux, uy = zip(*unsafe_xy)
        ax.scatter(ux, uy, c=PALETTE["obs_static"]["stroke"], s=20, zorder=5, label='unsafe sample')

    if commanded_rows:
        cx = [float(r['x']) for r in commanded_rows]
        cy = [float(r['y']) for r in commanded_rows]
        ax.plot(
            cx, cy, 'o',
            color=PALETTE["goal"]["stroke"], ms=4, alpha=0.6, label='commanded waypoint',
        )

    ax.legend()
    fig.tight_layout()

    out_path = EXPERIMENT_DIR / 'plots' / f'{condition}_fan{fan:02d}_run{run_num:02d}_actual.png'
    save_figure(fig, out_path.with_suffix(""), formats=("png", "pdf"))
    plt.close(fig)

    print(
        f'[{condition} fan{fan} run{run_num:02d}] {summary["n"]} samples, '
        f'{summary["duration_s"]:.1f}s, {summary["unsafe_samples"]} unsafe '
        f'({summary["unsafe_frac"] * 100:.1f}%)' + (' -- CRASH' if crashed else '')
    )
    print(f'Plot saved to {out_path}')
    return out_path


def plot_all() -> list[pathlib.Path]:
    """Plot the latest run of every (condition, fan) pair that has logs."""
    out = []
    for condition in CONDITIONS:
        for fan in VALID_FANS:
            try:
                out.append(plot_run(condition, fan))
            except FileNotFoundError:
                continue
    return out


def _run_stats(actual_path: pathlib.Path) -> dict:
    """Parse one run's filename + CSV into a flat stats row.

    condition/fan/run/crashed come from the filename (same convention
    flight_logger.py writes and _find_run parses); n/duration/unsafe come from
    the actual-position CSV itself.
    """
    m = _LOG_NAME_RE.match(actual_path.name)
    if not m:
        raise ValueError(f'Unrecognised log filename: {actual_path.name}')
    condition = m.group('condition')
    fan = int(m.group('fan'))
    row = {
        'condition': condition,
        'fan': fan,
        'run': int(m.group('run')),
        'crashed': m.group('crash') is not None,
    }
    row.update(_summarize(_read_csv(actual_path)))
    if condition == 'pdstl':
        try:
            meta = pdstl_plan_meta(fan)
        except FileNotFoundError:
            meta = {}
        row['rho_before'] = meta.get('rho_before')
        row['rho_after'] = meta.get('rho_after')
    else:
        row['rho_before'] = None
        row['rho_after'] = None
    return row


def summarize_all() -> list[dict]:
    """Print a per-run table and a per-(condition, fan) rollup across ALL logged runs.

    Unlike plot_all/plot_run (latest run per cell, one PNG each), this covers
    every run ever logged -- the cross-run view for judging progress across a
    batch of real flights, not just the most recent trial.
    """
    paths = sorted(_LOGS_DIR.glob('*_actual.csv'))
    if not paths:
        print(f'No logs yet in {_LOGS_DIR}.')
        return []

    rows = [_run_stats(p) for p in paths]
    rows.sort(key=lambda r: (r['condition'], r['fan'], r['run']))

    print(f'{"condition":<14}{"fan":>4}{"run":>4}{"crash":>7}{"n":>6}{"dur_s":>8}'
          f'{"unsafe":>8}{"unsafe%":>9}{"rho_after":>11}')
    for r in rows:
        rho = '' if r['rho_after'] is None else f'{r["rho_after"]:.3f}'
        print(f'{r["condition"]:<14}{r["fan"]:>4}{r["run"]:>4}{"Y" if r["crashed"] else "":>7}'
              f'{r["n"]:>6}{r["duration_s"]:>8.1f}{r["unsafe_samples"]:>8}'
              f'{r["unsafe_frac"] * 100:>8.1f}%{rho:>11}')

    print(f'\n{"condition":<14}{"fan":>4}{"runs":>6}{"crashed":>9}{"crash%":>8}'
          f'{"mean unsafe%":>14}{"mean dur_s":>12}')
    cells: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        cells.setdefault((r['condition'], r['fan']), []).append(r)
    for (condition, fan), cell_rows in sorted(cells.items()):
        n = len(cell_rows)
        n_crashed = sum(1 for r in cell_rows if r['crashed'])
        mean_unsafe = sum(r['unsafe_frac'] for r in cell_rows) / n
        mean_dur = sum(r['duration_s'] for r in cell_rows) / n
        print(f'{condition:<14}{fan:>4}{n:>6}{n_crashed:>9}{n_crashed / n * 100:>7.1f}%'
              f'{mean_unsafe * 100:>13.1f}%{mean_dur:>12.1f}')

    return rows


def run_analyze(
    condition: str | None, fan: int | None, run: int | None, all_: bool, summary: bool = False,
) -> None:
    if summary:
        summarize_all()
        return
    if all_:
        plot_all()
        return
    if condition is None or fan is None:
        raise SystemExit('--condition and --fan are required unless --all is given')
    plot_run(condition, fan, run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot logged Crazyflie flight(s) against their planned path'
    )
    parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument('--fan', type=int, choices=VALID_FANS)
    parser.add_argument('--run', type=int, default=None, help='Run number; defaults to the latest')
    parser.add_argument('--all', action='store_true',
                        help='Plot the latest run of every condition/fan pair with logs')
    parser.add_argument('--summary', action='store_true',
                        help='Print a cross-run table + per-condition/fan rollup instead of plotting')
    args = parser.parse_args()
    run_analyze(args.condition, args.fan, args.run, args.all, args.summary)


if __name__ == '__main__':
    main()
