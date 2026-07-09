"""Post-flight analysis for the Crazyflie experiment.

Reads the commanded/actual CSVs written by components/flight_logger.py and
plots the actual flown path against the planned one (pdSTL-optimised or
nominal safe path) on the same arena drawing used by waypoint_planning.py, so
a flight can be visually compared to the offline plan it was supposed to fly.

Called by `run.py analyze`; has no hardware/ROS dependency (only torch/numpy,
via components.config, plus matplotlib) -- same weight class as `plan`.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re

import numpy as np

from components.config import (
    EXPERIMENT_DIR,
    VALID_FANS,
    build_environment,
    load_pdstl_waypoints,
    nominal_safe_waypoints,
)
from components.flight_logger import CONDITIONS

_LOGS_DIR = EXPERIMENT_DIR / 'components' / 'logs'
_RUN_RE = re.compile(r'_run(\d+)_')


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
    from waypoint_planning import _draw_env  # reuse the same arena drawing as `plan`

    import matplotlib.pyplot as plt

    actual_path, commanded_path, run_num, crashed = _find_run(condition, fan, run)
    actual_rows = _read_csv(actual_path)
    commanded_rows = _read_csv(commanded_path) if commanded_path else []
    summary = _summarize(actual_rows)
    planned_xy = _planned_path(condition, fan)

    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_env(ax, build_environment())

    title = f'{condition} fan {fan}  —  run {run_num:02d}'
    if crashed:
        title += '  (CRASH)'
    ax.set_title(title)

    ax.plot(planned_xy[:, 0], planned_xy[:, 1], 'k--', lw=1.5, alpha=0.7, label='planned')

    ax_x = [float(r['x']) for r in actual_rows]
    ax_y = [float(r['y']) for r in actual_rows]
    safe = [int(r['safe']) for r in actual_rows]
    ax.plot(ax_x, ax_y, 'b-', lw=1.2, alpha=0.8, label='actual (flown)')
    unsafe_xy = [(x, y) for x, y, s in zip(ax_x, ax_y, safe) if not s]
    if unsafe_xy:
        ux, uy = zip(*unsafe_xy)
        ax.scatter(ux, uy, c='red', s=20, zorder=5, label='unsafe sample')

    if commanded_rows:
        cx = [float(r['x']) for r in commanded_rows]
        cy = [float(r['y']) for r in commanded_rows]
        ax.plot(cx, cy, 'go', ms=4, alpha=0.6, label='commanded waypoint')

    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = EXPERIMENT_DIR / 'plots' / f'{condition}_fan{fan:02d}_run{run_num:02d}_actual.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
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


def run_analyze(condition: str | None, fan: int | None, run: int | None, all_: bool) -> None:
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
    args = parser.parse_args()
    run_analyze(args.condition, args.fan, args.run, args.all)


if __name__ == '__main__':
    main()
