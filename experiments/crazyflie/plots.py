"""Figures for planned paths and flown runs.

``plot_plan`` is drawn before every flight; ``plot_run`` after every
attempt (and by ``analysis.py``); ``plot_repeats`` overlays repeated runs
of one plan. Each figure is saved as PNG (300 dpi) and PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import analysis
import trajectory
from planners import CHI2_95_2D

# Same roles and colours as src/visualization/figures.py.
COLORS = {
    "planned": "tab:orange",
    "executed": "tab:blue",
    "obstacle": "tab:red",
    "ring": "tab:green",
    "workspace": "tab:gray",
}


def _finish(fig, save_path, show):
    """Optionally show, then save PNG and PDF next to ``save_path``."""
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show(block=True)
    if save_path is not None:
        stem = Path(save_path).with_suffix("")
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix(".png"), dpi=300)
        fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    return fig


def _style(ax):
    ax.grid(color=COLORS["workspace"], alpha=0.16, linewidth=0.6)
    ax.tick_params(labelsize=9)


def draw_arena(ax, cfg):
    """Top view of bounds, obstacles and rings.

    Disabled groups are drawn dashed and labelled "(off)" so the figure
    always shows what is physically there and what is enforced.
    """
    bounds = cfg["bounds"]
    (x0, x1), (y0, y1) = bounds["x"], bounds["y"]
    ax.add_patch(
        patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor=COLORS["workspace"],
            linewidth=1.3,
            label="flight bounds",
        )
    )
    obstacles = cfg["environment"]["obstacles"]
    on = obstacles["enabled"]
    for box in obstacles["boxes"]:
        (bx0, bx1), (by0, by1) = box["x"], box["y"]
        ax.add_patch(
            patches.Rectangle(
                (bx0, by0),
                bx1 - bx0,
                by1 - by0,
                facecolor=COLORS["obstacle"],
                alpha=0.35 if on else 0.12,
                edgecolor=COLORS["obstacle"],
                linestyle="-" if on else "--",
                label="obstacle" if on else "obstacle (off)",
            )
        )
        ax.annotate(
            f"{box['name']}\n{box['height']:.2f} m",
            ((bx0 + bx1) / 2, (by0 + by1) / 2),
            ha="center",
            va="center",
            fontsize=7,
            color="0.25",
        )
    rings = cfg["environment"]["rings"]
    on = rings["enabled"]
    for ring in rings["regions"]:
        cx, cy, cz = ring["center"]
        ax.add_patch(
            patches.Circle(
                (cx, cy),
                ring["radius"],
                fill=False,
                edgecolor=COLORS["ring"],
                linewidth=1.5,
                linestyle="-" if on else "--",
                label="ring" if on else "ring (off)",
            )
        )
        ax.annotate(
            f"{ring['name']}\nz={cz:.2f}",
            (cx, cy - ring["radius"] - 0.02),
            ha="center",
            va="top",
            fontsize=7,
            color="0.25",
        )
    margin = 0.1
    ax.set_xlim(x0 - margin, x1 + margin)
    ax.set_ylim(y0 - margin, y1 + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _style(ax)


def _legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8, **kwargs)


def _endpoints(ax, xyz):
    ax.plot(*xyz[0, :2], "o", color="black", markersize=6, label="start")
    ax.plot(*xyz[-1, :2], "s", color="black", markersize=6, label="end")


def _uncertainty(ax, t, xyz, info):
    """pdSTL 95 % position ellipses about once per second."""
    cov = np.asarray(info.get("cov_xy", []))
    if not len(cov):
        return
    every = max(1, int(round(1.0 / np.median(np.diff(t)))))
    for k in range(every, len(cov), every):
        values, vectors = np.linalg.eigh(cov[k])
        width, height = 2 * CHI2_95_2D * np.sqrt(np.maximum(values, 0))
        angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
        ax.add_patch(
            patches.Ellipse(
                xyz[k, :2],
                width,
                height,
                angle=angle,
                fill=False,
                edgecolor=COLORS["planned"],
                linewidth=0.8,
                alpha=0.7,
                label="95 % ellipse",
            )
        )


def _goal(ax, cfg, info):
    """The pdSTL goal square (only pdSTL plans to a region)."""
    if info.get("method") != "pdstl":
        return
    (gx, gy), h = (
        cfg["planning"]["goal"],
        cfg["planning"]["pdstl"]["goal_half_width"],
    )
    ax.add_patch(
        patches.Rectangle(
            (gx - h, gy - h),
            2 * h,
            2 * h,
            facecolor="tab:green",
            alpha=0.25,
            edgecolor="none",
            label="goal region",
        )
    )


def _plan_title(cfg, t, info):
    title = f"{cfg['name']} [{info.get('method', '?')}]: {t[-1]:.1f} s"
    if "p_lower" in info:
        title += f", P(phi) in [{info['p_lower']:.3f}, {info['p_upper']:.3f}]"
    if "monte_carlo" in info:
        title += f", Monte Carlo {info['monte_carlo']['rate']:.3f}"
    return title


def plot_plan(cfg, t, xyz, info=None, save_path=None, show=False):
    """Planned path over the arena, with height and speed against time.

    Args:
        cfg: Full config (bounds and environment are drawn).
        t: Times, shape ``[N]``.
        xyz: Planned positions, shape ``[N, 3]``.
        info: Planner metadata; pdSTL plans add their goal region, 95 %
            ellipses and probability bounds.
        save_path: Stem for ``.png``/``.pdf``; ``None`` to not save.
        show: Block on an interactive window before saving.

    Returns:
        The matplotlib figure (already closed).
    """
    info = info or {}
    fig = plt.figure(figsize=(10, 6.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1])
    top = fig.add_subplot(grid[:, 0])
    draw_arena(top, cfg)
    _goal(top, cfg, info)
    _uncertainty(top, t, xyz, info)
    top.plot(*xyz[:, :2].T, color=COLORS["planned"], lw=2, label="plan")
    every = max(1, int(round(1.0 / np.median(np.diff(t)))))
    top.plot(
        *xyz[::every, :2].T,
        ".",
        color=COLORS["planned"],
        markersize=5,
        label="1 s marks",
    )
    _endpoints(top, xyz)
    _legend(top, loc="upper left")

    z_ax = fig.add_subplot(grid[0, 1])
    z_ax.plot(t, xyz[:, 2], color=COLORS["planned"], lw=2)
    z_ax.set_ylim(*cfg["bounds"]["z"])
    z_ax.set_ylabel("z [m]")
    _style(z_ax)

    v_ax = fig.add_subplot(grid[1, 1], sharex=z_ax)
    v_ax.step(
        t[1:],
        trajectory.speeds(t, xyz),
        where="pre",
        color=COLORS["planned"],
        lw=2,
    )
    v_ax.axhline(
        cfg["flight"]["max_speed"],
        color=COLORS["workspace"],
        ls="--",
        lw=1,
        label="max speed",
    )
    v_ax.set_ylim(0, cfg["flight"]["max_speed"] * 1.15)
    v_ax.set_xlabel("time [s]")
    v_ax.set_ylabel("speed [m/s]")
    _legend(v_ax, loc="lower right")
    _style(v_ax)

    fig.suptitle(_plan_title(cfg, t, info), fontsize=11)
    return _finish(fig, save_path, show)


def plot_run(run, save_path=None, show=False):
    """One attempt: plan vs flown path, per-axis traces and tracking error.

    Args:
        run: Output of ``analysis.load_run``.
        save_path: Stem for ``.png``/``.pdf``; ``None`` to not save.
        show: Block on an interactive window before saving.
    """
    cfg = run["cfg"]
    t_plan, xyz_plan = run["plan"]
    segment = analysis.trajectory_segment(run)

    fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(5, 2, width_ratios=[1.1, 1])
    top = fig.add_subplot(grid[:, 0])
    draw_arena(top, cfg)
    _goal(top, cfg, run["plan_info"])
    _uncertainty(top, t_plan, xyz_plan, run["plan_info"])
    top.plot(*run["flown"][:, :2].T, color="0.6", lw=0.8, label="flown (all)")
    top.plot(
        *xyz_plan[:, :2].T,
        color=COLORS["planned"],
        lw=1.5,
        ls="--",
        label="plan",
        zorder=3,
    )
    if segment is not None:
        top.plot(
            *segment[1][:, :2].T,
            color=COLORS["executed"],
            lw=1.5,
            label="flown (trajectory)",
        )
    _endpoints(top, xyz_plan)
    _legend(top, loc="upper left")

    axes = [fig.add_subplot(grid[i, 1]) for i in range(5)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    for i, name in enumerate("xyz"):
        ax = axes[i]
        ax.plot(
            t_plan,
            xyz_plan[:, i],
            color=COLORS["planned"],
            ls="--",
            lw=1.5,
            label="plan",
            zorder=3,
        )
        if segment is not None:
            ax.plot(
                segment[0],
                segment[1][:, i],
                color=COLORS["executed"],
                lw=1.5,
                label="flown",
            )
        ax.set_ylabel(f"{name} [m]")
        _style(ax)
    _legend(axes[0], loc="best")
    err_ax = axes[3]
    if segment is not None:
        error = np.linalg.norm(segment[1] - segment[2], axis=1)
        err_ax.plot(segment[0], error, color=COLORS["executed"], lw=1.5)
    err_ax.set_ylabel("error vs\ntarget [m]")
    _style(err_ax)
    _timing(axes[4], run)

    s = run["summary"] or {"outcome": "unfinished"}
    method = run["plan_info"].get("method", "?")
    title = f"{run['dir'].name} [{method}]: {s['outcome']}"
    if s.get("tracking"):
        title += f", RMSE {s['tracking']['rmse_m']:.3f} m"
    fig.suptitle(title, fontsize=11)
    return _finish(fig, save_path, show)


def _timing(ax, run):
    """Command lateness over reference time, with the abort limit."""
    cmd = run["commands"]
    streamed = np.isfinite(cmd["lateness"])
    ax.plot(
        cmd["t_ref"][streamed],
        1000 * cmd["lateness"][streamed],
        color=COLORS["executed"],
        lw=1,
    )
    limit = run["cfg"]["flight"]["max_lateness_s"]
    ax.axhline(
        1000 * limit, color=COLORS["obstacle"], ls="--", lw=1, label="limit"
    )
    gaps = sum(
        b["gaps"] for b in analysis.data_quality(run)["blocks"].values()
    )
    ax.set_title(f"device-time gaps: {gaps}", fontsize=8, loc="right")
    ax.set_ylabel("lateness [ms]")
    ax.set_xlabel("trajectory time [s]")
    _legend(ax, loc="upper right")
    _style(ax)


def plot_repeats(runs, save_path=None, show=False):
    """Completed runs of one plan overlaid, with their tracking errors.

    Args:
        runs: ``analysis.load_run`` outputs that share one plan.
        save_path: Stem for ``.png``/``.pdf``; ``None`` to not save.
        show: Block on an interactive window before saving.
    """
    cfg = runs[-1]["cfg"]
    t_plan, xyz_plan = runs[-1]["plan"]
    fig = plt.figure(figsize=(11, 6.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.1, 1])
    top = fig.add_subplot(grid[0, 0])
    err_ax = fig.add_subplot(grid[0, 1])
    draw_arena(top, cfg)
    errors = []
    for run in runs:
        segment = analysis.trajectory_segment(run)
        if segment is None:
            continue
        tau, flown, reference = segment
        top.plot(
            *flown[:, :2].T,
            color=COLORS["executed"],
            lw=1,
            alpha=0.4,
            label="flown runs",
        )
        error = np.linalg.norm(flown - reference, axis=1)
        err_ax.plot(
            tau,
            error,
            color=COLORS["executed"],
            lw=1,
            alpha=0.4,
            label="single run",
        )
        errors.append((tau, error))
    top.plot(
        *xyz_plan[:, :2].T,
        color=COLORS["planned"],
        lw=2,
        ls="--",
        label="plan",
        zorder=3,
    )
    _endpoints(top, xyz_plan)
    _legend(top, loc="upper left")

    if errors:
        grid_t = np.linspace(0, min(e[0][-1] for e in errors), 200)
        mean = np.mean([np.interp(grid_t, *e) for e in errors], axis=0)
        err_ax.plot(grid_t, mean, color="black", lw=2, label="mean")
    err_ax.set_xlabel("trajectory time [s]")
    err_ax.set_ylabel("tracking error [m]")
    _legend(err_ax, loc="upper right")
    _style(err_ax)
    fig.suptitle(f"{len(errors)} completed run(s) of one plan", fontsize=11)
    return _finish(fig, save_path, show)
