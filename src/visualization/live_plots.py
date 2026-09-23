"""Optional live execution and optimizer progress views."""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from visualization.planning import (
    COLORS,
    _draw_ego_vehicle,
    _draw_environment,
    _ellipse,
    _ellipse_indices,
    _ellipse_parameters,
    _move_obstacle,
    _move_ego_vehicle,
    _np,
    _style,
    _unique_legend,
)


def _update_control_lines(ax, lines, controls):
    if not lines:
        palette = (COLORS["mean"], COLORS["planned"], COLORS["goal"])
        for dimension in range(controls.shape[1]):
            (line,) = ax.plot(
                [],
                [],
                color=palette[dimension % len(palette)],
                label=("Longitudinal" if dimension == 0 else "Lateral"),
            )
            lines.append(line)
        _unique_legend(ax, loc="best")
    for dimension, line in enumerate(lines):
        line.set_data(np.arange(len(controls)), controls[:, dimension])
    ax.set_xlim(0, max(1, len(controls) - 1))
    ax.relim()
    ax.autoscale_view(scalex=False)


def _refresh(fig, interactive):
    if interactive:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)


def _state_beliefs(env, state, lane):
    values = [(state[0], state[1], COLORS["mean"])]
    if lane:
        values.extend(
            (
                state[3][index],
                state[4][index],
                COLORS["traffic"],
            )
            for index, vehicle in enumerate(env.metadata["traffic"])
        )
    return values


def _draw_belief_outlines(ax, env, state, lane):
    if state[1] is None:
        return []
    outlines = []
    for mean, covariance, color in _state_beliefs(env, state, lane):
        width, height, angle = _ellipse_parameters(_np(covariance))
        outline = patches.Ellipse(
            _np(mean)[:2],
            width,
            height,
            angle=angle,
            fill=False,
            edgecolor=color,
            alpha=0.7,
            linewidth=1.2,
        )
        ax.add_patch(outline)
        outlines.append(outline)
    return outlines


def _update_belief_outlines(outlines, env, state, lane):
    for outline, (mean, covariance, _) in zip(
        outlines, _state_beliefs(env, state, lane)
    ):
        width, height, angle = _ellipse_parameters(_np(covariance))
        outline.center = _np(mean)[:2]
        outline.width = width
        outline.height = height
        outline.angle = angle


def _candidate_trace(dynamics, state, controls):
    point = _np(state[0]).copy()
    candidate = [point[:2].copy()]
    for control in _np(controls):
        point = _np(dynamics.A) @ point + _np(dynamics.B) @ control
        candidate.append(point[:2].copy())
    return np.asarray(candidate)


def _follow_lane(ax, x, lane):
    if lane:
        ax.set_xlim(x - 15, x + 45)


def _mark_lane_window(ax, env, lane, dt):
    if not lane:
        return dt
    task = env.metadata["task"]
    start, end = task["start_end_steps"]
    latest_start = end - task["dwell_steps"]
    ax.axvspan(
        start * dt,
        latest_start * dt,
        color=COLORS["goal"],
        alpha=0.12,
        label="Dwell-start window",
    )
    ax.axvline(
        end * dt,
        color=COLORS["goal"],
        linestyle="--",
        linewidth=1,
        label="Completion deadline",
    )
    return end * dt


def create_optimization_view(label, *, max_iters, control_unit="m/s"):
    """Show sampled gradient-descent iterates from the planner callback."""
    fig, (ax_loss, ax_score, ax_controls) = plt.subplots(
        3, 1, figsize=(7, 7), layout="constrained"
    )
    (loss_line,) = ax_loss.plot(
        [], [], color=COLORS["planned"], label="Optimization loss"
    )
    (smooth_line,) = ax_score.plot(
        [], [], color=COLORS["score"], label="Smooth score"
    )
    (hard_line,) = ax_score.plot(
        [],
        [],
        color=COLORS["goal"],
        label="Hard lower",
    )
    ax_loss.set(
        xlabel="gradient descent iteration", ylabel="optimization loss"
    )
    ax_score.set(xlabel="gradient descent iteration", ylabel="score")
    ax_controls.set(
        xlabel="prediction step", ylabel=f"current control [{control_unit}]"
    )
    for ax in (ax_loss, ax_score):
        ax.set_xlim(0, max_iters)
        _style(ax)
        _unique_legend(ax, loc="best")
    _style(ax_controls)
    beta_text = ax_score.text(
        0.02,
        0.05,
        "",
        transform=ax_score.transAxes,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    interactive = matplotlib.get_backend().lower() != "agg"
    if interactive:
        fig.show()
    window_number = None
    iterations, losses, smooth_scores, hard_scores = [], [], [], []
    control_lines = []

    def redraw():
        loss_line.set_data(iterations, losses)
        smooth_line.set_data(iterations, smooth_scores)
        hard_line.set_data(iterations, hard_scores)
        for ax in (ax_loss, ax_score):
            ax.relim()
            ax.autoscale_view(scalex=False)
        _refresh(fig, interactive)

    def on_iteration(window, iteration, record):
        nonlocal window_number
        if window_number != window:
            window_number = window
            iterations.clear()
            losses.clear()
            smooth_scores.clear()
            hard_scores.clear()
            ax_loss.set_title(f"{label}: planning window {window + 1}")
        iterations.append(iteration + 1)
        losses.append(record.loss)
        smooth_scores.append(record.smooth_lower)
        hard_scores.append(record.hard_interval[0])
        _update_control_lines(ax_controls, control_lines, _np(record.controls))
        beta_text.set_text(f"smoothing beta: {record.beta:.2f}")
        print(
            f"{label} window {window + 1}, iteration {iteration + 1}/"
            f"{max_iters}: loss {record.loss:.4f}, "
            f"smooth lower {record.smooth_lower:.4f}, "
            f"hard lower {record.hard_interval[0]:.4f}",
            flush=True,
        )
        redraw()

    def finish_window(window, plan):
        loss_line.set_data(
            np.arange(1, len(plan.loss_history) + 1), plan.loss_history
        )
        ax_loss.relim()
        ax_loss.autoscale_view(scalex=False)
        beta_text.set_text(f"final smoothing beta: {plan.smoothing_beta:.2f}")
        print(
            f"{label} window {window + 1} complete after "
            f"{len(plan.loss_history)} iterations: "
            f"hard interval [{plan.hard_interval[0]:.4f}, "
            f"{plan.hard_interval[1]:.4f}]",
            flush=True,
        )
        _refresh(fig, interactive)

    return fig, (ax_loss, ax_score, ax_controls), on_iteration, finish_window


def create_reach_avoid_live_view(
    env,
    evaluate_controls,
    *,
    dt,
    title="Reach–Avoid",
    max_iters,
    ellipse_every=8,
    alpha=None,
):
    """Live candidate trajectory with separate smooth and hard scores."""
    fig, (ax_map, ax_scores) = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
        layout="constrained",
        gridspec_kw={"width_ratios": (1.45, 1)},
    )
    _draw_environment(ax_map, env)
    (path,) = ax_map.plot(
        [],
        [],
        color=COLORS["mean"],
        linewidth=2.2,
        label="Candidate belief mean",
        zorder=8,
    )
    start = ax_map.scatter(
        [], [], color="black", s=55, marker="o", label="Start", zorder=10
    )
    terminal = ax_map.scatter(
        [],
        [],
        color=COLORS["mean"],
        s=55,
        marker="s",
        label="Terminal belief mean",
        zorder=10,
    )
    (smooth_line,) = ax_scores.plot(
        [],
        [],
        color=COLORS["score"],
        linewidth=1.7,
        label="Smooth surrogate",
    )
    (hard_line,) = ax_scores.plot(
        [],
        [],
        color=COLORS["mean"],
        linewidth=1.7,
        label="Hard lower score",
    )
    if alpha is not None:
        ax_scores.axhline(
            alpha,
            color=COLORS["obstacle"],
            linestyle="--",
            linewidth=1.1,
            label=f"Required alpha={alpha:.2f}",
        )
    ax_scores.set(
        xlabel="optimizer iteration",
        ylabel="score",
        ylim=(-0.03, 1.03),
        xlim=(1, max(1, max_iters)),
        title="Optimization diagnostics",
    )
    _style(ax_scores, probability=True)
    _unique_legend(ax_scores, loc="best")
    interactive = matplotlib.get_backend().lower() != "agg"
    if interactive:
        fig.show()
    ellipses = []
    iterations, smooth_scores, hard_scores = [], [], []

    def render(iteration, controls, hard_interval):
        plan = evaluate_controls(controls)
        mean = _np(plan.rollout.aux["mean_trace"])[0]
        covariance = _np(plan.rollout.aux["cov_trace"])[0]
        path.set_data(mean[:, 0], mean[:, 1])
        start.set_offsets(mean[:1, :2])
        terminal.set_offsets(mean[-1:, :2])
        while ellipses:
            ellipses.pop().remove()
        for index in _ellipse_indices(mean, ellipse_every):
            ellipses.append(_ellipse(ax_map, mean[index], covariance[index]))
        ax_map.set_title(
            f"{title} | iteration {iteration + 1}/{max_iters} | "
            f"P↓(φ)={hard_interval[0]:.3f}",
            fontweight="bold",
        )
        _refresh(fig, interactive)

    def on_iteration(iteration, record):
        iterations.append(iteration + 1)
        smooth_scores.append(record.smooth_lower)
        hard_scores.append(record.hard_interval[0])
        smooth_line.set_data(iterations, smooth_scores)
        hard_line.set_data(iterations, hard_scores)
        render(iteration, record.controls, record.hard_interval)
        print(
            f"{title} iteration {iteration + 1}/{max_iters}: "
            f"loss {record.loss:.4f}, "
            f"hard lower {record.hard_interval[0]:.4f}",
            flush=True,
        )

    def finish(result):
        candidates = np.arange(len(result.smooth_history))
        smooth_line.set_data(candidates, result.smooth_history)
        hard_line.set_data(candidates, result.hard_lower_history)
        ax_scores.set_xlim(0, max(1, len(candidates) - 1))
        render(
            result.selected_iteration, result.controls, result.hard_interval
        )

    return fig, (ax_map, ax_scores), on_iteration, finish


def create_live_view(
    env,
    initial_state,
    *,
    dt,
    lane=False,
    dynamics=None,
    max_iters=40,
    label="MPC",
):
    """One road-centered figure for candidate plans and executed steps."""
    fig = plt.figure(figsize=(14, 6), layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=(2.8, 1.0))
    ax_map = fig.add_subplot(grid[:, 0])
    ax_loss = fig.add_subplot(grid[0, 1])
    ax_score = fig.add_subplot(grid[1, 1])
    ax_windows = fig.add_subplot(grid[2, 1])
    moving = _draw_environment(ax_map, env, lane=lane)
    initial = _np(initial_state[0])[:2]
    ego_patch = _draw_ego_vehicle(ax_map, env, initial) if lane else None
    executed = [initial]
    window_scores = []
    (line_exec,) = ax_map.plot(
        [initial[0]],
        [initial[1]],
        color=COLORS["executed"],
        linewidth=2,
        marker="o",
        markersize=3,
        label="Executed",
    )
    (line_plan,) = ax_map.plot(
        [],
        [],
        color=COLORS["planned"],
        linewidth=1.5,
        label="Plan",
    )
    (line_loss,) = ax_loss.plot(
        [],
        [],
        color=COLORS["planned"],
        marker="o",
        markersize=3,
        label="Optimization loss",
    )
    (line_smooth,) = ax_score.plot(
        [],
        [],
        color=COLORS["score"],
        marker="o",
        markersize=3,
        label="Smooth score",
    )
    (line_hard,) = ax_score.plot(
        [],
        [],
        color=COLORS["goal"],
        marker="o",
        markersize=3,
        label="Hard lower",
    )
    (line_windows,) = ax_windows.plot(
        [],
        [],
        color=COLORS["score"],
        linewidth=1.5,
        marker="o",
        markersize=3,
        label="Hard lower",
    )
    ax_loss.set(xlabel="iteration", ylabel="loss", xlim=(0, max_iters))
    ax_score.set(xlabel="iteration", ylabel="lower score", xlim=(0, max_iters))
    ax_windows.set(xlabel="executed time [s]", ylabel="hard lower bound")
    final_time = _mark_lane_window(ax_windows, env, lane, dt)
    ax_windows.set_xlim(0, final_time)
    for ax in (ax_loss, ax_score, ax_windows):
        _style(ax, probability=ax is ax_windows)
        _unique_legend(ax, loc="best")
    _unique_legend(
        ax_map, loc="lower center", bbox_to_anchor=(0.5, 1.16), ncol=5
    )
    _follow_lane(ax_map, initial[0], lane)
    belief_outlines = _draw_belief_outlines(ax_map, env, initial_state, lane)
    interactive = matplotlib.get_backend().lower() != "agg"
    if interactive:
        fig.show()
    current_state = initial_state
    iterations, losses, smooth, hard = [], [], [], []
    current_window = None

    def on_iteration(window, iteration, record):
        nonlocal current_window
        if current_window != window:
            current_window = window
            iterations.clear()
            losses.clear()
            smooth.clear()
            hard.clear()
            ax_map.set_title(
                f"Planning window {window + 1} at t={window * dt:.1f} s"
            )
        iterations.append(iteration + 1)
        losses.append(record.loss)
        smooth.append(record.smooth_lower)
        hard.append(record.hard_interval[0])
        line_loss.set_data(iterations, losses)
        line_smooth.set_data(iterations, smooth)
        line_hard.set_data(iterations, hard)
        for ax in (ax_loss, ax_score):
            ax.relim()
            ax.autoscale_view(scalex=False)
        if dynamics is not None:
            candidate = _candidate_trace(
                dynamics, current_state, record.controls
            )
            line_plan.set_data(candidate[:, 0], candidate[:, 1])
        print(
            f"{label} window {window + 1}, iteration {iteration + 1}/{max_iters}: "
            f"loss {record.loss:.4f}, smooth lower {record.smooth_lower:.4f}, "
            f"hard lower {record.hard_interval[0]:.4f}",
            flush=True,
        )
        _refresh(fig, interactive)

    def finish_window(window, plan):
        print(
            f"{label} window {window + 1} complete after "
            f"{len(plan.loss_history)} iterations: "
            f"hard interval [{plan.hard_interval[0]:.4f}, "
            f"{plan.hard_interval[1]:.4f}]",
            flush=True,
        )

    def on_step(step, state, plan):
        nonlocal current_state
        current_state = state
        executed.append(_np(state[0])[:2])
        points = np.asarray(executed)
        line_exec.set_data(points[:, 0], points[:, 1])
        trace = _np(plan.rollout.aux["mean_trace"])[0]
        line_plan.set_data(trace[:, 0], trace[:, 1])
        window_scores.append(plan.hard_interval[0])
        line_windows.set_data(
            dt * np.arange(1, len(window_scores) + 1), window_scores
        )
        ax_windows.set_xlim(0, max(final_time, dt * (len(window_scores) + 1)))
        _move_obstacle(moving, env, step + 1, state)
        if ego_patch is not None:
            _move_ego_vehicle(ego_patch, points[-1])
        _follow_lane(ax_map, points[-1, 0], lane)
        _update_belief_outlines(belief_outlines, env, state, lane)
        ax_map.set_title(
            f"Executed step {step + 1} at t={(step + 1) * dt:.1f} s"
        )
        _refresh(fig, interactive)

    on_step.on_iteration = on_iteration
    on_step.finish_window = finish_window
    return fig, (ax_map, ax_loss, ax_score, ax_windows), on_step
