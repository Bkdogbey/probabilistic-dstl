"""GIFs: a reach-avoid plan step by step, and a lane execution."""

from pathlib import Path

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from visualization.figures import (
    COLORS,
    RHO_INTERVAL,
    RHO_LOWER,
    RHO_UPPER,
    _draw_ego_vehicle,
    _draw_environment,
    _ellipse,
    _ellipse_parameters,
    _move_obstacle,
    _move_ego_vehicle,
    _np,
    _style,
    _unique_legend,
)


def _finish_animation(fig, movie, filename, fps, show):
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show(block=True)
    if filename is not None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        movie.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def _frames(last_step, fps):
    return [0] * fps + list(range(1, last_step + 1)) + [last_step] * fps


def animate_reach_avoid(
    result,
    env,
    *,
    dt,
    title="Reach–Avoid",
    filename=None,
    fps=6,
    show=False,
):
    """Reveal one optimized belief trajectory inside its environment."""
    mean = _np(result.rollout.aux["mean_trace"])[0]
    covariance = _np(result.rollout.aux["cov_trace"])[0]
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    _draw_environment(ax, env)
    ax.plot(
        mean[:, 0],
        mean[:, 1],
        color=COLORS["mean"],
        alpha=0.15,
    )
    ax.scatter(
        [mean[0, 0]],
        [mean[0, 1]],
        marker="o",
        s=55,
        color="black",
        zorder=10,
    )
    (path,) = ax.plot(
        [],
        [],
        color=COLORS["mean"],
        linewidth=2.2,
        zorder=8,
    )
    (point,) = ax.plot(
        [],
        [],
        marker="s",
        color=COLORS["mean"],
        markersize=6,
        zorder=10,
    )
    ellipse = _ellipse(ax, mean[0], covariance[0])
    status = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax.set_title(
        f"{title} | {RHO_LOWER} = {result.hard_interval[0]:.3f}",
        fontweight="bold",
    )

    def update(step):
        path.set_data(mean[: step + 1, 0], mean[: step + 1, 1])
        point.set_data([mean[step, 0]], [mean[step, 1]])
        width, height, angle = _ellipse_parameters(covariance[step])
        ellipse.set_center(mean[step, :2])
        ellipse.set_width(width)
        ellipse.set_height(height)
        ellipse.set_angle(angle)
        status.set_text(
            f"t = {step * dt:.1f} s  ·  predicted step {step}/{len(mean) - 1}"
        )
        return path, point, ellipse, status

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=_frames(len(mean) - 1, fps),
        interval=1000 / fps,
        blit=False,
    )
    _finish_animation(fig, movie, filename, fps, show)
    return fig, ax, movie


def animate_mpc(result, env, *, dt, filename=None, fps=6, show=False):
    """Replay a lane execution: path, current plan, traffic, and robustness."""
    fig, (ax_map, ax_score) = plt.subplots(
        1,
        2,
        figsize=(11, 5),
        layout="constrained",
        gridspec_kw={"width_ratios": (1.4, 1)},
    )
    moving = _draw_environment(ax_map, env, lane=True)
    states = np.asarray([_np(state[0])[:2] for state in result.states])
    ego_patch = _draw_ego_vehicle(ax_map, env, states[0])
    plans = result.window_plans
    intervals = np.asarray([plan.hard_interval for plan in plans]).reshape(
        -1, 2
    )
    applied = _np(result.applied_controls)
    for index, _vehicle in enumerate(env.metadata["traffic"]):
        centers = np.asarray(
            [_np(state[3][index])[:2] for state in result.states]
        )
        ax_map.plot(
            centers[:, 0],
            centers[:, 1],
            color=COLORS["traffic"],
            linestyle=":",
            linewidth=1.2,
            label="Traffic path" if index == 0 else "_nolegend_",
        )
    ax_map.set_xlim(states[0, 0] - 15, states[0, 0] + 45)
    ax_map.scatter(
        states[0, 0],
        states[0, 1],
        color=COLORS["executed"],
        s=28,
        label="Start",
        zorder=5,
    )
    (executed,) = ax_map.plot(
        [],
        [],
        color=COLORS["executed"],
        linewidth=2,
        marker="o",
        markersize=3,
        label="Executed",
    )
    (planned,) = ax_map.plot(
        [], [], color=COLORS["planned"], linewidth=1.5, label="Plan"
    )
    (lower_bound,) = ax_score.plot(
        [],
        [],
        color=COLORS["goal"],
        linewidth=1.7,
        marker="o",
        markersize=3,
        label=RHO_LOWER,
    )
    (upper_bound,) = ax_score.plot(
        [],
        [],
        color=COLORS["goal"],
        linewidth=1.2,
        linestyle="--",
        label=RHO_UPPER,
    )
    ax_score.set(
        xlabel="time [s]",
        ylabel="robustness per window",
        xlim=(0, max(dt, len(plans) * dt)),
    )
    task = env.metadata["task"]
    start, end = task["start_end_steps"]
    ax_score.axvspan(
        start * dt,
        (end - task["dwell_steps"]) * dt,
        color=COLORS["goal"],
        alpha=0.10,
        label="Dwell-start window",
    )
    ax_score.axvline(
        end * dt,
        color=COLORS["obstacle"],
        linestyle="--",
        linewidth=1,
        label="Completion deadline",
    )
    _style(ax_score, probability=True)
    status = ax_score.text(
        0.03,
        0.05,
        "",
        transform=ax_score.transAxes,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    _unique_legend(
        ax_map, loc="lower center", bbox_to_anchor=(0.5, 1.14), ncol=5
    )
    _unique_legend(ax_score, loc="best")

    def update(step):
        executed.set_data(states[: step + 1, 0], states[: step + 1, 1])
        if plans:
            index = min(step, len(plans) - 1)
            trace = _np(plans[index].rollout.aux["mean_trace"])[0]
            planned.set_data(trace[:, 0], trace[:, 1])
        lower_bound.set_data(dt * np.arange(step), intervals[:step, 0])
        upper_bound.set_data(dt * np.arange(step), intervals[:step, 1])
        _move_obstacle(moving, env, step, result.states[step])
        _move_ego_vehicle(ego_patch, states[step])
        ax_map.set_xlim(states[step, 0] - 15, states[step, 0] + 45)
        ax_map.set_title(
            f"Executed step {step}/{len(plans)}  ·  time {step * dt:.1f} s",
            fontsize=11,
        )
        interval = (
            "—"
            if step == 0
            else f"[{intervals[step - 1, 0]:.3f}, {intervals[step - 1, 1]:.3f}]"
        )
        if step == 0:
            control = "No control applied yet"
        else:
            values = ", ".join(f"{value:.2f}" for value in applied[step - 1])
            control = f"Applied control: ({values}) m/s²"
        ending = (
            f"\nOutcome: {result.stopped_reason.replace('_', ' ')}"
            if step == len(plans)
            else ""
        )
        status.set_text(f"{RHO_INTERVAL} = {interval}\n{control}{ending}")
        return executed, planned, lower_bound, upper_bound, status

    # Hold the first and final frames briefly.
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=_frames(len(plans), fps),
        interval=1000 / fps,
        blit=False,
    )
    _finish_animation(fig, movie, filename, fps, show)
    return fig, (ax_map, ax_score), movie
