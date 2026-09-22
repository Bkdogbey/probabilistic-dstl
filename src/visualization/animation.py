"""Replay a stored MPC result, one executed step at a time."""

from pathlib import Path

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from pdstl.predicates import GreaterThan
from visualization.planning import (
    COLORS,
    _draw_ego_vehicle,
    _draw_environment,
    _ellipse,
    _ellipse_parameters,
    _interval_text,
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


def animate_altitude(
    result, *, dt, threshold, filename=None, fps=6, show=False
):
    """Reveal the altitude belief and atomic probability over predicted time."""
    mean = _np(result.rollout.aux["mean_trace"])[0, :, 0]
    variance = _np(result.rollout.aux["cov_trace"])[0, :, 0, 0]
    time = dt * np.arange(len(mean))
    sigma = np.sqrt(np.maximum(variance, 0))
    bounds = _np(
        GreaterThan(threshold, dim=0)(result.rollout.belief_trajectory)
    )[0]
    fig, (ax_state, ax_prob) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, layout="constrained"
    )
    ax_state.fill_between(
        time,
        mean - 1.96 * sigma,
        mean + 1.96 * sigma,
        color=COLORS["ellipse"],
        alpha=0.25,
        label="95% belief band",
    )
    ax_state.plot(time, mean, color=COLORS["mean"], alpha=0.25)
    ax_state.axhline(
        threshold,
        color=COLORS["obstacle"],
        linestyle="--",
        label="Safety threshold",
    )
    (state_line,) = ax_state.plot(
        [],
        [],
        color=COLORS["mean"],
        linewidth=2,
        label="Predicted belief mean",
    )
    (state_point,) = ax_state.plot(
        [], [], marker="o", color=COLORS["mean"], markersize=5
    )
    ax_state.set_ylabel("altitude [m]")
    ax_prob.plot(time, bounds[:, 0], color=COLORS["score"], alpha=0.25)
    (prob_line,) = ax_prob.plot(
        [], [], color=COLORS["score"], linewidth=2, label="Atomic probability"
    )
    ax_prob.set(xlabel="time [s]", ylabel="atomic probability")
    status = ax_prob.text(
        0.02,
        0.05,
        "",
        transform=ax_prob.transAxes,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    for ax in (ax_state, ax_prob):
        _style(ax, probability=ax is ax_prob)
        _unique_legend(ax, loc="best")

    def update(step):
        state_line.set_data(time[: step + 1], mean[: step + 1])
        state_point.set_data([time[step]], [mean[step]])
        prob_line.set_data(time[: step + 1], bounds[: step + 1, 0])
        status.set_text(
            f"Predicted step {step}/{len(mean) - 1}  ·  "
            f"atomic P = {bounds[step, 0]:.3f}\n"
            f"{_interval_text(result.hard_interval)}"
        )
        return state_line, state_point, prob_line, status

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=_frames(len(mean) - 1, fps),
        interval=1000 / fps,
        blit=False,
    )
    _finish_animation(fig, movie, filename, fps, show)
    return fig, (ax_state, ax_prob), movie


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
        f"{title} | P↓(φ)={result.hard_interval[0]:.3f}",
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


def animate_mpc(
    result, env, *, dt, filename=None, lane=False, fps=6, show=False
):
    """Show and save the path, latest plan, moving vehicle, and hard lower bound."""
    fig, (ax_map, ax_score) = plt.subplots(
        1,
        2,
        figsize=(11, 5),
        layout="constrained",
        gridspec_kw={"width_ratios": (1.4, 1)},
    )
    moving = _draw_environment(ax_map, env, lane=lane)
    states = np.asarray([_np(state[0])[:2] for state in result.states])
    ego_patch = _draw_ego_vehicle(ax_map, env, states[0]) if lane else None
    plans = result.window_plans
    intervals = np.asarray([plan.hard_interval for plan in plans]).reshape(
        -1, 2
    )
    applied = _np(result.applied_controls)
    if lane:
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
        [],
        [],
        color=COLORS["planned"],
        linewidth=1.5,
        label="Plan",
    )
    (lower_bound,) = ax_score.plot(
        [],
        [],
        color=COLORS["goal"],
        linewidth=1.7,
        marker="o",
        markersize=3,
        label="Hard lower",
    )
    (upper_bound,) = ax_score.plot(
        [],
        [],
        color=COLORS["goal"],
        linewidth=1.2,
        linestyle="--",
        label="Hard upper",
    )
    ax_score.set(
        xlabel="time [s]",
        ylabel="lower satisfaction bound",
        xlim=(0, max(dt, len(plans) * dt)),
    )
    if lane:
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
    if lane:
        _unique_legend(
            ax_map,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.14),
            ncol=5,
        )
    else:
        _unique_legend(ax_map, loc="best")
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
        if ego_patch is not None:
            _move_ego_vehicle(ego_patch, states[step])
        if lane:
            ax_map.set_xlim(states[step, 0] - 15, states[step, 0] + 45)
        ax_map.set_title(
            f"Executed step {step}/{len(plans)}  ·  time {step * dt:.1f} s",
            fontsize=11,
        )
        hard = (
            "—"
            if step == 0
            else f"[{intervals[step - 1, 0]:.3f}, {intervals[step - 1, 1]:.3f}]"
        )
        if step == 0:
            control = "No control applied yet"
        else:
            values = ", ".join(f"{value:.2f}" for value in applied[step - 1])
            unit = "m/s²" if lane else "m/s"
            control = f"Applied control: ({values}) {unit}"
        ending = (
            f"\nStopping reason: {result.stopped_reason.replace('_', ' ')}"
            if step == len(plans)
            else ""
        )
        status.set_text(f"Certified pdSTL: {hard}\n{control}{ending}")
        return executed, planned, lower_bound, upper_bound, status

    # Hold the first and final states briefly, while retaining every executed step.
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=_frames(len(plans), fps),
        interval=1000 / fps,
        blit=False,
    )
    _finish_animation(fig, movie, filename, fps, show)
    return fig, (ax_map, ax_score), movie
