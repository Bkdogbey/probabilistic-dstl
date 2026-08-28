"""Static views of bounded sliding-window and streaming evaluation."""

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch


def _as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else value


def _plot_bounds(ax, time, bounds, *, lower_label, upper_label):
    values = bounds[0] if bounds.ndim == 3 else bounds
    lower = _as_numpy(values[:, 0])
    upper = _as_numpy(values[:, 1])
    time = _as_numpy(time)
    ax.plot(time, lower, color="tab:blue", marker="o", label=lower_label)
    ax.plot(
        time,
        upper,
        color="tab:orange",
        linestyle="--",
        marker="o",
        label=upper_label,
    )
    ax.fill_between(time, lower, upper, color="tab:blue", alpha=0.15)


def plot_sliding_windows(trace, temporal_bounds, interval, formula_label, show=True):
    """Plot the atomic trace and all complete offline temporal windows."""
    fig, (ax_atomic, ax_temporal) = plt.subplots(2, 1, figsize=(9, 7))
    _plot_bounds(
        ax_atomic,
        trace.time,
        trace.bounds,
        lower_label="atomic lower",
        upper_label="atomic upper",
    )

    a, b = interval
    for anchor in range(temporal_bounds.shape[1]):
        ax_atomic.axvspan(
            anchor + a,
            anchor + b,
            color="purple" if anchor % 2 == 0 else "gray",
            alpha=0.045,
        )
    ax_atomic.set_title(f"Atomic bounds and sliding [{a},{b}] windows")
    ax_atomic.set_ylabel("Probability")
    ax_atomic.set_xticks(_as_numpy(trace.time))
    ax_atomic.set_ylim(0, 1)
    ax_atomic.grid(True, alpha=0.3)
    ax_atomic.legend(loc="best")

    output_count = temporal_bounds.shape[1]
    anchor_time = trace.time[:output_count]
    _plot_bounds(
        ax_temporal,
        anchor_time,
        temporal_bounds,
        lower_label="temporal lower",
        upper_label="temporal upper",
    )
    ax_temporal.set_title(formula_label)
    ax_temporal.set_xlabel("STL anchor k")
    ax_temporal.set_ylabel("Probability")
    ax_temporal.set_xticks(_as_numpy(anchor_time))
    ax_temporal.set_ylim(0, 1)
    ax_temporal.grid(True, alpha=0.3)
    ax_temporal.legend(loc="best")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def _snapshot_indices(update_count, b):
    candidates = [0, b - 1, b, b + 1, update_count - 2, update_count - 1]
    return list(dict.fromkeys(index for index in candidates if 0 <= index < update_count))


def plot_streaming_updates(
    trace,
    updates,
    offline_bounds,
    online_bounds,
    interval,
    formula_label,
    show=True,
):
    """Show selected recurrent states and offline/online output agreement."""
    a, b = interval
    selected = _snapshot_indices(len(updates), b)
    fig = plt.figure(figsize=(12, 9))
    grid = fig.add_gridspec(3, 3, height_ratios=(1, 1, 0.9))

    for position, update_index in enumerate(selected):
        update = updates[update_index]
        ax = fig.add_subplot(grid[position // 3, position % 3])
        arrived = trace.bounds[0, : update.arrival + 1]
        state_start = update.arrival - update.window_state.shape[1] + 1

        for time, (lower, upper) in enumerate(arrived.tolist()):
            if time == update.arrival:
                color = "tab:orange"
            elif time >= state_start:
                color = "tab:blue"
            else:
                color = "lightgray"
            ax.vlines(time, lower, upper, color=color, linewidth=7, alpha=0.8)

        if update.output is None:
            status = "window filling"
        else:
            lower, upper = update.output[0].tolist()
            status = f"output=[{lower:.2f}, {upper:.2f}]"
        ax.set_title(f"arrival t={update.arrival}\n{status}", fontsize=9)
        ax.set_xlim(-0.5, len(trace.time) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(trace.time)))
        ax.grid(True, alpha=0.25)

    comparison = fig.add_subplot(grid[2, :])
    availability_time = trace.time[b : b + offline_bounds.shape[1]]
    offline_lower = _as_numpy(offline_bounds[0, :, 0])
    offline_upper = _as_numpy(offline_bounds[0, :, 1])
    online_lower = _as_numpy(online_bounds[0, :, 0])
    online_upper = _as_numpy(online_bounds[0, :, 1])
    availability_time = _as_numpy(availability_time)

    comparison.plot(
        availability_time,
        offline_lower,
        color="purple",
        label="offline lower",
    )
    comparison.plot(
        availability_time,
        offline_upper,
        color="red",
        linestyle="--",
        label="offline upper",
    )
    comparison.scatter(
        availability_time,
        online_lower,
        color="black",
        marker="x",
        label="online lower",
        zorder=3,
    )
    comparison.scatter(
        availability_time,
        online_upper,
        facecolors="none",
        edgecolors="black",
        label="online upper",
        zorder=3,
    )
    comparison.fill_between(
        availability_time,
        offline_lower,
        offline_upper,
        color="purple",
        alpha=0.12,
    )
    comparison.set_title(f"{formula_label}: output available at t = k + {b}")
    comparison.set_xlabel("Arrival time")
    comparison.set_ylabel("Probability")
    comparison.set_ylim(0, 1)
    comparison.set_xticks(availability_time)
    comparison.grid(True, alpha=0.3)
    comparison.legend(loc="best", ncol=2)

    fig.suptitle(
        f"Streaming state retains {b + 1} entries; active offsets are [{a},{b}]",
        y=0.995,
    )
    fig.legend(
        handles=[
            Patch(color="tab:blue", label="retained state"),
            Patch(color="tab:orange", label="newest entry"),
            Patch(color="lightgray", label="expired entry"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    if show:
        plt.show()
    return fig
