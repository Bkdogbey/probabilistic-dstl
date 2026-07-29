"""Post-flight plotting and cross-run summaries."""

from __future__ import annotations

import csv
import pathlib

import numpy as np

from .utils import (
    EXPERIMENT_DIR,
    VALID_FANS,
    PlanRepository,
    logs_directory,
)
from .flight_logger import CONDITIONS, FlightLogRepository
from .planner import build_environment_baseline, get_scenario


def _find_run(
    condition: str, fan: int, run: int | None, scenario: str = 'baseline',
) -> tuple[pathlib.Path, pathlib.Path | None, int, bool]:
    """Locate the actual/commanded CSVs for one (condition, scenario, fan, run) tuple.

    run=None picks the highest-numbered run for that cell (the latest trial).
    """
    actual_path, commanded_path, identity = FlightLogRepository(logs_directory(scenario)).find(
        condition, scenario, fan, run,
    )
    return actual_path, commanded_path, identity.run, identity.crashed


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


def _planned_path(condition: str, fan: int, scenario: str = 'baseline') -> np.ndarray:
    scenario_model = get_scenario(scenario)
    wps = (
        PlanRepository().load(fan, scenario).waypoints
        if condition == 'pdstl'
        else scenario_model.nominal_waypoints(scenario_model.flight_points)
    )
    return np.array(wps)


def _plot_run_2d(condition: str, fan: int, run: int | None = None) -> pathlib.Path:
    """Plot one logged baseline run against its planned path; returns the saved PNG path.

    Unchanged from before --scenario existed -- always the flat 2D baseline
    arena, only ever called for scenario='baseline'.
    """
    from .utils import draw_environment_2d

    import matplotlib.pyplot as plt

    from visualization.style import PALETTE, figsize as _figsize, save_figure

    actual_path, commanded_path, run_num, crashed = _find_run(condition, fan, run, scenario='baseline')
    actual_rows = _read_csv(actual_path)
    commanded_rows = _read_csv(commanded_path) if commanded_path else []
    summary = _summarize(actual_rows)
    planned_xy = _planned_path(condition, fan)[:, :2]

    fig, ax = plt.subplots(figsize=_figsize("single", aspect=1.0))
    draw_environment_2d(ax, build_environment_baseline())

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


def _plot_run_3d(condition: str, fan: int, run: int | None, scenario: str) -> pathlib.Path:
    """Plot one logged figure-eight run in 3D against its planned path and
    environment (not the 2D baseline arena)."""
    from .utils import draw_environment_3d

    import numpy as np
    import matplotlib.pyplot as plt

    from visualization.style import PALETTE, save_figure

    env = get_scenario(scenario).build_environment()

    actual_path, commanded_path, run_num, crashed = _find_run(condition, fan, run, scenario=scenario)
    actual_rows = _read_csv(actual_path)
    commanded_rows = _read_csv(commanded_path) if commanded_path else []
    summary = _summarize(actual_rows)
    planned_xyz = _planned_path(condition, fan, scenario=scenario)

    actual_xyz = np.array([[float(r['x']), float(r['y']), float(r['z'])] for r in actual_rows])
    fit_points = np.vstack([planned_xyz, actual_xyz])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=22, azim=-50)
    draw_environment_3d(ax, env, fit_points=fit_points)

    title = f'{condition} {scenario} fan {fan}  —  run {run_num:02d}'
    if crashed:
        title += '  (CRASH)'
    ax.set_title(title)

    ax.plot(
        planned_xyz[:, 0], planned_xyz[:, 1], planned_xyz[:, 2],
        color=PALETTE["plan"]["stroke"], linestyle='--', lw=1.5, alpha=0.7, label='planned',
    )

    ax_x = [float(r['x']) for r in actual_rows]
    ax_y = [float(r['y']) for r in actual_rows]
    ax_z = [float(r['z']) for r in actual_rows]
    safe = [int(r['safe']) for r in actual_rows]
    ax.plot(
        ax_x, ax_y, ax_z,
        color=PALETTE["ego"]["stroke"], linestyle='-', lw=1.2, alpha=0.8, label='actual (flown)',
    )
    unsafe_xyz = [(x, y, z) for x, y, z, s in zip(ax_x, ax_y, ax_z, safe) if not s]
    if unsafe_xyz:
        ux, uy, uz = zip(*unsafe_xyz)
        ax.scatter(ux, uy, uz, c=PALETTE["obs_static"]["stroke"], s=20, zorder=5, label='unsafe sample')

    if commanded_rows:
        cx = [float(r['x']) for r in commanded_rows]
        cy = [float(r['y']) for r in commanded_rows]
        cz = [float(r['z']) for r in commanded_rows]
        ax.plot(
            cx, cy, cz, 'o',
            color=PALETTE["goal"]["stroke"], ms=4, alpha=0.6, label='commanded waypoint',
        )

    ax.legend()
    fig.tight_layout()

    out_path = EXPERIMENT_DIR / 'plots' / f'{condition}_{scenario}_fan{fan:02d}_run{run_num:02d}_actual.png'
    save_figure(fig, out_path.with_suffix(""), formats=("png", "pdf"))
    plt.close(fig)

    print(
        f'[{condition} {scenario} fan{fan} run{run_num:02d}] {summary["n"]} samples, '
        f'{summary["duration_s"]:.1f}s, {summary["unsafe_samples"]} unsafe '
        f'({summary["unsafe_frac"] * 100:.1f}%)' + (' -- CRASH' if crashed else '')
    )
    print(f'Plot saved to {out_path}')
    return out_path


def plot_run(
    condition: str, fan: int, run: int | None = None, scenario: str = 'baseline',
) -> pathlib.Path:
    """Plot one logged run against its planned path; returns the saved PNG path.

    scenario='baseline' (default) keeps the original flat 2D arena plot,
    byte-identical to before scenario selection. Figure-eight dispatches to
    a 3D plot against their own environment.
    """
    if scenario == 'baseline':
        return _plot_run_2d(condition, fan, run)
    return _plot_run_3d(condition, fan, run, scenario)


def plot_all(scenario: str = 'baseline') -> list[pathlib.Path]:
    """Plot the latest run of every (condition, fan) pair that has logs, for one scenario."""
    out = []
    for condition in CONDITIONS:
        for fan in VALID_FANS:
            try:
                out.append(plot_run(condition, fan, scenario=scenario))
            except FileNotFoundError:
                continue
    return out


def _run_stats(actual_path: pathlib.Path) -> dict:
    """Parse one run's filename + CSV into a flat stats row.

    condition/scenario/fan/run/crashed come from the filename (same
    convention flight_logger.py writes and _find_run parses); n/duration/
    unsafe come from the actual-position CSV itself.
    """
    identity = FlightLogRepository.parse(actual_path)
    condition = identity.condition
    scenario = identity.scenario
    fan = identity.fan
    row = {
        'condition': condition,
        'scenario': scenario,
        'fan': fan,
        'run': identity.run,
        'crashed': identity.crashed,
    }
    row.update(_summarize(_read_csv(actual_path)))
    if condition == 'pdstl':
        try:
            meta = PlanRepository().load(fan, scenario).metadata
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
    paths = sorted(
        path
        for scenario in ('baseline', 'figure8')
        for path in logs_directory(scenario).glob('*_actual.csv')
    )
    if not paths:
        print('No Crazyflie logs yet.')
        return []

    rows = [_run_stats(p) for p in paths]
    rows.sort(key=lambda r: (r['condition'], r['scenario'], r['fan'], r['run']))

    print(f'{"condition":<14}{"scenario":<10}{"fan":>4}{"run":>4}{"crash":>7}{"n":>6}{"dur_s":>8}'
          f'{"unsafe":>8}{"unsafe%":>9}{"rho_after":>11}')
    for r in rows:
        rho = '' if r['rho_after'] is None else f'{r["rho_after"]:.3f}'
        print(f'{r["condition"]:<14}{r["scenario"]:<10}{r["fan"]:>4}{r["run"]:>4}'
              f'{"Y" if r["crashed"] else "":>7}'
              f'{r["n"]:>6}{r["duration_s"]:>8.1f}{r["unsafe_samples"]:>8}'
              f'{r["unsafe_frac"] * 100:>8.1f}%{rho:>11}')

    print(f'\n{"condition":<14}{"scenario":<10}{"fan":>4}{"runs":>6}{"crashed":>9}{"crash%":>8}'
          f'{"mean unsafe%":>14}{"mean dur_s":>12}')
    cells: dict[tuple[str, str, int], list[dict]] = {}
    for r in rows:
        cells.setdefault((r['condition'], r['scenario'], r['fan']), []).append(r)
    for (condition, scenario, fan), cell_rows in sorted(cells.items()):
        n = len(cell_rows)
        n_crashed = sum(1 for r in cell_rows if r['crashed'])
        mean_unsafe = sum(r['unsafe_frac'] for r in cell_rows) / n
        mean_dur = sum(r['duration_s'] for r in cell_rows) / n
        print(f'{condition:<14}{scenario:<10}{fan:>4}{n:>6}{n_crashed:>9}{n_crashed / n * 100:>7.1f}%'
              f'{mean_unsafe * 100:>13.1f}%{mean_dur:>12.1f}')

    return rows


class AnalysisService:
    def run(
        self,
        *,
        condition: str,
        fan: int,
        scenario: str,
        mode: str = 'latest',
        run_number: int | None = None,
    ) -> None:
        if mode == 'summary':
            summarize_all()
        elif mode == 'all':
            plot_all(scenario)
        elif mode == 'latest':
            plot_run(condition, fan, run_number, scenario)
        else:
            raise ValueError(f'Unknown analysis mode {mode!r}.')
