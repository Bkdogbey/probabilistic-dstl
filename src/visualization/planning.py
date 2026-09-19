"""Publication figures from completed pdSTL planning results."""

from functools import reduce
from math import log, sqrt
from pathlib import Path

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from pdstl.operators import And
from pdstl.predicates import GreaterThan
from planning.environment import MovingRectangleRegion, reach_avoid_events

COLORS = {
    "mean": "tab:blue",
    "planned": "tab:orange",
    "executed": "tab:blue",
    "goal": "tab:green",
    "obstacle": "tab:red",
    "workspace": "tab:gray",
    "score": "tab:purple",
    "ellipse": "tab:cyan",
    "lead": "tab:red",
    "target_front": "tab:purple",
    "target_rear": "tab:brown",
}
CHI95_RADIUS = sqrt(-2 * log(0.05))


class VehiclePatch(patches.Polygon):
    """Top-down car silhouette within its configured collision footprint."""

    def __init__(self, x, y, width, height, **kwargs):
        self._x, self._y = x, y
        self._width, self._height = width, height
        super().__init__(self._vertices(x, y), closed=True, **kwargs)

    def _vertices(self, x, y):
        w, h = self._width, self._height
        return [
            (x, y + 0.20 * h),
            (x + 0.10 * w, y + 0.05 * h),
            (x + 0.72 * w, y + 0.05 * h),
            (x + w, y + 0.28 * h),
            (x + w, y + 0.72 * h),
            (x + 0.72 * w, y + 0.95 * h),
            (x + 0.10 * w, y + 0.95 * h),
            (x, y + 0.80 * h),
        ]

    def set_xy(self, xy):
        coordinates = np.asarray(xy)
        if coordinates.ndim == 2:
            return super().set_xy(coordinates)
        self._x, self._y = coordinates
        return super().set_xy(self._vertices(self._x, self._y))

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height


def _np(value):
    return (
        value.detach().cpu().numpy()
        if torch.is_tensor(value)
        else np.asarray(value)
    )


def _unique_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(
            unique.values(),
            unique.keys(),
            fontsize=9,
            framealpha=0.9,
            **kwargs,
        )


def _finish(fig, save_path, show):
    """Let the user inspect the figure, then save both publication formats."""
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show(block=True)
    if save_path is not None:
        stem = Path(save_path).with_suffix("")
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix(".png"), dpi=300)
        fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    return fig


def _style(ax, *, probability=False):
    ax.grid(color="#888888", alpha=0.16, linewidth=0.6)
    ax.tick_params(labelsize=9)
    if probability:
        ax.set_ylim(-0.03, 1.03)


def _interval_text(interval):
    lower, upper = interval
    return f"pdSTL probability interval: [{lower:.3f}, {upper:.3f}]"


def _ellipse_parameters(covariance):
    """Width, height, and angle of a joint 95% two-dimensional belief ellipse."""
    covariance = np.asarray(covariance)[:2, :2]
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = np.maximum(values[order], 0), vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2 * CHI95_RADIUS * np.sqrt(values)
    return width, height, angle


def _ellipse(ax, mean, covariance, *, label=None):
    width, height, angle = _ellipse_parameters(covariance)
    ellipse = patches.Ellipse(
        mean[:2],
        width,
        height,
        angle=angle,
        facecolor=COLORS["ellipse"],
        edgecolor=COLORS["mean"],
        alpha=0.27,
        linewidth=0.8,
        label=label,
    )
    ax.add_patch(ellipse)
    return ellipse


def _draw_environment(ax, env, *, lane=False):
    """Draw configured geometry; return patches for moving traffic, if any."""
    moving = None
    if lane:
        road = env.metadata["road"]
        ramp = env.metadata.get("ramp")
        if ramp is None:
            ax.axhspan(road["y_min"], road["y_max"], color="#f2f2f7")
            for y in (road["y_min"], road["y_max"]):
                ax.axhline(y, color=COLORS["workspace"], linewidth=1.3)
            ax.axhline(
                road["lane_divider"],
                color=COLORS["workspace"],
                linestyle="--",
                linewidth=1.1,
                label="Lane divider",
            )
        else:
            extent = env.single_region("goal").x
            start, end = ramp["start_x"], ramp["end_x"]
            x = np.asarray([extent[0], start, end, extent[1]])
            lower = np.asarray(
                [
                    road["y_min"],
                    road["y_min"],
                    road["lane_divider"],
                    road["lane_divider"],
                ]
            )
            ax.fill_between(x, lower, road["y_max"], color="#f2f2f7")
            ax.plot(x, lower, color=COLORS["workspace"], linewidth=1.3)
            ax.axhline(road["y_max"], color=COLORS["workspace"], linewidth=1.3)
            ax.plot(
                [extent[0], start],
                [road["lane_divider"], road["lane_divider"]],
                color=COLORS["workspace"],
                linestyle="--",
                linewidth=1.1,
                label="Ramp merge line",
            )
            ax.scatter(
                [end],
                [road["lane_divider"]],
                marker="x",
                color=COLORS["workspace"],
                label="Ramp end",
            )
        goal = env.single_region("goal")
        ax.add_patch(
            patches.Rectangle(
                (goal.xmin, goal.ymin),
                goal.xmax - goal.xmin,
                goal.ymax - goal.ymin,
                facecolor=COLORS["goal"],
                edgecolor="none",
                alpha=0.10,
                label="Goal lane",
            )
        )
        moving = {}
        for vehicle in env.metadata["traffic"]:
            name = vehicle["name"]
            patch = VehiclePatch(
                vehicle["x0"] - vehicle["width"] / 2,
                vehicle["y"] - vehicle["height"] / 2,
                vehicle["width"],
                vehicle["height"],
                facecolor=COLORS.get(name, COLORS["obstacle"]),
                edgecolor=COLORS.get(name, COLORS["obstacle"]),
                alpha=0.65,
                label=name.replace("_", " ").title(),
            )
            ax.add_patch(patch)
            moving[name] = patch
    for region in env.regions.values():
        if isinstance(region, MovingRectangleRegion):
            x, y = _np(region.centers[0])
            moving = patches.Rectangle(
                (x - region.width / 2, y - region.height / 2),
                region.width,
                region.height,
                facecolor=COLORS["obstacle"],
                edgecolor=COLORS["obstacle"],
                alpha=0.5,
                label="Moving obstacle",
            )
            ax.add_patch(moving)
        elif region.role == "workspace" and not lane:
            ax.add_patch(
                patches.Rectangle(
                    (region.xmin, region.ymin),
                    region.xmax - region.xmin,
                    region.ymax - region.ymin,
                    fill=False,
                    edgecolor=COLORS["workspace"],
                    linewidth=1.2,
                    label="Workspace",
                )
            )
            ax.set_xlim(region.xmin - 0.5, region.xmax + 0.5)
            ax.set_ylim(region.ymin - 0.5, region.ymax + 0.5)
        elif region.role == "goal" and not lane:
            ax.add_patch(
                patches.Rectangle(
                    (region.xmin, region.ymin),
                    region.xmax - region.xmin,
                    region.ymax - region.ymin,
                    facecolor=COLORS["goal"],
                    edgecolor=COLORS["goal"],
                    alpha=0.30,
                    label="Goal",
                )
            )
        elif region.role == "obstacle":
            ax.add_patch(
                patches.Rectangle(
                    (region.xmin, region.ymin),
                    region.xmax - region.xmin,
                    region.ymax - region.ymin,
                    facecolor=region.style.get("color", COLORS["obstacle"]),
                    edgecolor=COLORS["obstacle"],
                    alpha=0.35,
                    label="Obstacles",
                )
            )
    ax.set_xlabel("x position [m]")
    ax.set_ylabel("y position [m]")
    ax.set_aspect(1.4 if lane else "equal", adjustable="box")
    _style(ax)
    return moving


def _draw_ego_vehicle(ax, env, position):
    """Show the ego footprint at its current sampled position."""
    size = env.metadata["ego_vehicle"]
    patch = VehiclePatch(
        position[0] - size["width"] / 2,
        position[1] - size["height"] / 2,
        size["width"],
        size["height"],
        facecolor=COLORS["executed"],
        edgecolor=COLORS["executed"],
        alpha=0.7,
        label="Ego",
        zorder=6,
    )
    ax.add_patch(patch)
    return patch


def _move_ego_vehicle(patch, position):
    patch.set_xy(
        (
            position[0] - patch.get_width() / 2,
            position[1] - patch.get_height() / 2,
        )
    )


def _move_obstacle(patch, env, step, state=None):
    if patch is None:
        return
    if isinstance(patch, dict):
        for index, vehicle in enumerate(env.metadata["traffic"]):
            if state is not None and len(state) > 3:
                center = _np(state[3][index])[:2]
            else:
                center = (
                    vehicle["x0"]
                    + vehicle["speed"] * env.metadata["dt"] * step,
                    vehicle["y"],
                )
            patch[vehicle["name"]].set_xy(
                (
                    center[0] - vehicle["width"] / 2,
                    center[1] - vehicle["height"] / 2,
                )
            )
        return
    obstacle = next(
        r for r in env.regions.values() if isinstance(r, MovingRectangleRegion)
    )
    center = _np(obstacle.centers[min(step, len(obstacle.centers) - 1)])
    patch.set_xy(
        (center[0] - obstacle.width / 2, center[1] - obstacle.height / 2)
    )


def _selected_windows(count, maximum=5):
    if count <= maximum:
        return list(range(count))
    return sorted(set(np.linspace(0, count - 1, maximum, dtype=int)))


def _plot_bounds(ax, time, bounds, color, label):
    values = _np(bounds)
    if values.ndim == 3:
        values = values[0]
    time = np.asarray(time)[: len(values)]
    ax.fill_between(
        time, values[:, 0], values[:, 1], color=color, alpha=0.20, label=label
    )
    ax.plot(
        time,
        values[:, 0],
        color=color,
        linewidth=1.4,
        marker="o",
        markersize=2.5,
    )
    if not np.allclose(values[:, 0], values[:, 1]):
        ax.plot(time, values[:, 1], color=color, linewidth=1.1, linestyle="--")


def plot_altitude_safety(
    result, *, initial=None, dt, threshold, u_max, save_path=None, show=False
):
    """Predicted belief, atomic probability, controls, and final interval."""
    fig, axes = plt.subplots(
        3, 1, figsize=(8, 8), sharex=True, layout="constrained"
    )
    ax_state, ax_prob, ax_control = axes
    atom = GreaterThan(threshold, dim=0)
    for plan, name, color in (
        (initial, "Initial", COLORS["workspace"]),
        (result, "Optimized", COLORS["mean"]),
    ):
        if plan is None:
            continue
        mean = _np(plan.rollout.aux["mean_trace"])[0, :, 0]
        variance = _np(plan.rollout.aux["cov_trace"])[0, :, 0, 0]
        time = np.arange(len(mean)) * dt
        sigma = np.sqrt(np.maximum(variance, 0))
        ax_state.fill_between(
            time,
            mean - 1.96 * sigma,
            mean + 1.96 * sigma,
            color=color,
            alpha=0.17,
            label=f"{name} 95% belief band",
        )
        ax_state.plot(
            time,
            mean,
            color=color,
            linewidth=1.8,
            label=f"{name} predicted belief mean",
        )
        _plot_bounds(
            ax_prob,
            time,
            atom(plan.rollout.belief_trajectory),
            color,
            f"{name} atomic probability",
        )
        controls = _np(plan.controls)[:, 0]
        ax_control.step(
            time[:-1],
            controls,
            where="post",
            color=color,
            linewidth=1.7,
            label=f"{name} controls",
        )
    ax_state.axhline(
        threshold,
        color=COLORS["obstacle"],
        linestyle="--",
        linewidth=1.1,
        label="Safety threshold",
    )
    ax_state.set_ylabel("altitude [m]")
    ax_prob.set_ylabel("atomic probability")
    ax_prob.text(
        0.02,
        0.05,
        _interval_text(result.hard_interval),
        transform=ax_prob.transAxes,
        fontsize=9,
    )
    ax_control.set_ylabel("control [m/s]")
    ax_control.set_xlabel("time [s]")
    ax_control.set_ylim(-1.1 * u_max, 1.1 * u_max)
    for ax in axes:
        _style(ax, probability=ax is ax_prob)
        _unique_legend(ax, loc="best")
    _finish(fig, save_path, show)
    return fig, axes


def plot_reach_avoid(
    result, env, *, dt, ellipse_every=10, save_path=None, show=False
):
    """Workspace plan, predicate intervals, and optimization history."""
    fig = plt.figure(figsize=(11, 7), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 1))
    ax_map = fig.add_subplot(grid[:, 0])
    ax_prob = fig.add_subplot(grid[0, 1])
    ax_loss = fig.add_subplot(grid[1, 1])
    _draw_environment(ax_map, env)
    mean = _np(result.rollout.aux["mean_trace"])[0]
    covariance = _np(result.rollout.aux["cov_trace"])[0]
    ax_map.plot(
        mean[:, 0],
        mean[:, 1],
        color=COLORS["mean"],
        linewidth=2,
        label="Predicted belief mean",
    )
    ax_map.scatter(
        [mean[0, 0]],
        [mean[0, 1]],
        marker="o",
        s=32,
        color=COLORS["executed"],
        label="Start",
        zorder=4,
    )
    indices = sorted(
        set(range(0, len(mean), max(1, ellipse_every))) | {len(mean) - 1}
    )
    for index in indices:
        _ellipse(
            ax_map,
            mean[index],
            covariance[index],
            label="95% belief ellipse" if index == indices[0] else None,
        )
    _unique_legend(ax_map, loc="best")
    events = reach_avoid_events(env)
    belief = result.rollout.belief_trajectory
    time = np.arange(len(belief)) * dt
    obstacles = events["obstacles"]
    traces = [
        (events["goal"](belief), COLORS["goal"], "Goal probability interval")
    ]
    if obstacles:
        traces.append(
            (
                reduce(And, obstacles)(belief),
                COLORS["obstacle"],
                "Obstacle avoidance probability interval",
            )
        )
    traces.append(
        (
            events["workspace"](belief),
            COLORS["workspace"],
            "Workspace probability interval",
        )
    )
    for bounds, color, label in traces:
        _plot_bounds(ax_prob, time, bounds, color, label)
    ax_prob.set(xlabel="time [s]", ylabel="predicate probability interval")
    _style(ax_prob, probability=True)
    _unique_legend(ax_prob, loc="best")
    ax_loss.plot(
        np.arange(1, len(result.loss_history) + 1),
        result.loss_history,
        color=COLORS["score"],
        linewidth=1.6,
        label="Optimization loss",
    )
    ax_loss.text(
        0.02,
        0.05,
        f"{_interval_text(result.hard_interval)}\n"
        f"Smooth lower score: {result.smooth_lower:.3f}",
        transform=ax_loss.transAxes,
        fontsize=9,
    )
    ax_loss.set(xlabel="optimization iteration", ylabel="optimization loss")
    _style(ax_loss)
    _unique_legend(ax_loss, loc="best")
    _finish(fig, save_path, show)
    return fig, (ax_map, ax_prob, ax_loss)


def _plot_execution(result, env, *, dt, lane, save_path, show):
    fig = plt.figure(
        figsize=(14, 6) if lane else (11, 7), layout="constrained"
    )
    if lane:
        grid = fig.add_gridspec(2, 2, height_ratios=(0.55, 1))
        ax_map = fig.add_subplot(grid[0, :])
        ax_score = fig.add_subplot(grid[1, 0])
        ax_control = fig.add_subplot(grid[1, 1])
    else:
        grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1))
        ax_map = fig.add_subplot(grid[:, 0])
        ax_score = fig.add_subplot(grid[0, 1])
        ax_control = fig.add_subplot(grid[1, 1])
    moving = _draw_environment(ax_map, env, lane=lane)
    means = np.asarray([_np(state[0])[:2] for state in result.states])
    ax_map.plot(
        means[:, 0],
        means[:, 1],
        color=COLORS["executed"],
        linewidth=2,
        label="Executed trajectory",
        zorder=5,
    )
    ax_map.scatter(
        [means[0, 0]],
        [means[0, 1]],
        color=COLORS["executed"],
        s=26,
        label="Start",
        zorder=6,
    )
    for index in _selected_windows(len(result.window_plans)):
        plan = _np(result.window_plans[index].rollout.aux["mean_trace"])[0]
        ax_map.plot(
            plan[:, 0],
            plan[:, 1],
            color=COLORS["planned"],
            linewidth=1.1,
            alpha=0.6,
            label="Predicted planning window",
        )
    if lane:
        _draw_ego_vehicle(ax_map, env, means[-1])
        traffic_positions = []
        for index, vehicle in enumerate(env.metadata["traffic"]):
            positions = np.asarray(
                [_np(state[3][index])[:2] for state in result.states]
            )
            traffic_positions.append(positions)
            ax_map.plot(
                positions[:, 0],
                positions[:, 1],
                color=COLORS.get(vehicle["name"], COLORS["obstacle"]),
                linestyle=":",
                linewidth=1.2,
                label=f"{vehicle['name'].replace('_', ' ').title()} path",
            )
        _move_obstacle(moving, env, len(result.states) - 1, result.states[-1])
        all_x = np.concatenate(
            [
                means[:, 0],
                *(positions[:, 0] for positions in traffic_positions),
            ]
        )
        ax_map.set_xlim(all_x.min() - 8, all_x.max() + 8)
    reason = {
        "goal_reached": "goal reached",
        "max_steps": "step limit reached",
        "deadline_missed": "lane-change deadline missed",
        "collision": "collision",
        "road_violation": "road boundary violated",
    }
    ax_map.text(
        0.98 if lane else 0.02,
        0.95 if lane else 0.02,
        f"Stopping reason: {reason.get(result.stopped_reason, result.stopped_reason)}",
        transform=ax_map.transAxes,
        fontsize=9,
        ha="right" if lane else "left",
        va="top" if lane else "baseline",
    )
    if lane:
        _unique_legend(
            ax_map,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=6,
        )
    else:
        _unique_legend(ax_map, loc="best")
    scores = [plan.hard_interval[0] for plan in result.window_plans]
    time = dt * np.arange(len(scores))
    ax_score.plot(
        time,
        scores,
        color=COLORS["score"],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Lower satisfaction bound",
    )
    ax_score.set(xlabel="time [s]", ylabel="lower satisfaction bound")
    _style(ax_score, probability=True)
    _unique_legend(ax_score, loc="best")
    controls = _np(result.applied_controls)
    for dimension in range(controls.shape[1]):
        ax_control.step(
            dt * np.arange(len(controls)),
            controls[:, dimension],
            where="post",
            linewidth=1.4,
            label=f"Applied control {dimension + 1}",
        )
    ax_control.set(
        xlabel="time [s]",
        ylabel="applied control [m/s²]" if lane else "applied control [m/s]",
    )
    _style(ax_control)
    _unique_legend(ax_control, loc="best")
    _finish(fig, save_path, show)
    return fig, (ax_map, ax_score, ax_control)


def plot_mpc_execution(result, env, *, dt, save_path=None, show=False):
    """Executed reach-avoid path, selected plans, bounds, and controls."""
    return _plot_execution(
        result, env, dt=dt, lane=False, save_path=save_path, show=show
    )


def plot_lane_merge(result, env, *, dt, save_path=None, show=False):
    """Lane geometry, moving vehicle, executed path, plans, and bounds."""
    return _plot_execution(
        result, env, dt=dt, lane=True, save_path=save_path, show=show
    )
