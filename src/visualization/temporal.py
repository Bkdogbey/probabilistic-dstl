"""Plotting for the drone-altitude scenario.
"""

import matplotlib.pyplot as plt
import torch

COLOR_PRIMARY = "tab:blue"
COLOR_SECONDARY = "tab:orange"
COLOR_COMBINED = "purple"


def _as_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x


def plot_bounds(ax, time, bounds, label, color):
    """Shade [lower, upper] with a solid lower line and a dashed upper line.

    bounds is Tensor[1, T, 2] or Tensor[T, 2]; time is length T.
    """
    values = bounds[0] if bounds.dim() == 3 else bounds
    lower, upper = _as_numpy(values[:, 0]), _as_numpy(values[:, 1])
    t = _as_numpy(time)

    ax.fill_between(t, lower, upper, color=color, alpha=0.15, label=label)
    ax.plot(t, lower, color=color, linestyle="-", marker="o", markersize=4)
    ax.plot(t, upper, color=color, linestyle="--", marker="o", markersize=4)


def _style_probability_axis(ax, time, title):
    ax.set_title(title)
    ax.set_xlabel("Time step k")
    ax.set_ylabel("Probability bound")
    ax.set_xticks(_as_numpy(time))
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)


def _plot_signal(ax, time, mean, std, thresholds):
    t, m, s = _as_numpy(time), _as_numpy(mean), _as_numpy(std)
    ax.plot(t, m, color="black", marker="o", markersize=4, label="altitude mean")
    ax.fill_between(t, m - s, m + s, color="gray", alpha=0.2, label="mean +/- std")
    for th in thresholds:
        ax.axhline(th, color="red", linestyle="--", linewidth=1, label=f"{th} m")
    ax.set_title("Altitude signal")
    ax.set_xlabel("Time step k")
    ax.set_ylabel("Altitude (m)")
    ax.set_xticks(t)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def plot_predicates_and_boolean(
    time, mean, std, bounds_50, bounds_55, boolean_results, thresholds=(50, 55), show=True
):
    """Figure 1: signal, both atomic predicate bounds, and selected Boolean
    outputs, all on the same time axis.

    boolean_results is a list of (label, bounds, color) tuples.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    _plot_signal(axes[0, 0], time, mean, std, thresholds)

    plot_bounds(axes[0, 1], time, bounds_50, "altitude >= 50m", COLOR_PRIMARY)
    _style_probability_axis(axes[0, 1], time, "P(altitude >= 50m)")

    plot_bounds(axes[1, 0], time, bounds_55, "altitude >= 55m", COLOR_SECONDARY)
    _style_probability_axis(axes[1, 0], time, "P(altitude >= 55m)")

    for label, bounds, color in boolean_results:
        plot_bounds(axes[1, 1], time, bounds, label, color)
    _style_probability_axis(axes[1, 1], time, "Selected Boolean outputs")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_temporal_operator(
    time, mean, std, threshold, atomic_bounds, operator_bounds, anchor_time, operator_label, show=True
):
    """Figures 2/3: signal+threshold, the atomic predicate over `time`, and
    the operator output over `anchor_time` (shorter than `time` by the
    window width -- anchor k summarizes [k, k+b] of the atomic trace).
    """
    fig, (ax_signal, ax_atomic, ax_operator) = plt.subplots(3, 1, figsize=(8, 9))

    _plot_signal(ax_signal, time, mean, std, [threshold])

    plot_bounds(ax_atomic, time, atomic_bounds, "atomic bounds", COLOR_PRIMARY)
    _style_probability_axis(ax_atomic, time, "Atomic predicate bounds")

    plot_bounds(ax_operator, anchor_time, operator_bounds, operator_label, COLOR_COMBINED)
    _style_probability_axis(ax_operator, anchor_time, operator_label)

    fig.suptitle(operator_label)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_online_window(time, bounds, snapshots, show=True):
    """Figure 4: one subplot per (arrival_time, window_start_index,
    output_available) snapshot. Highlights the retained
    [window_start_index, arrival_time] slice of `bounds`; the index just
    before window_start_index (if any) is marked as just-expired. Indices
    past arrival_time haven't arrived yet and aren't drawn.
    """
    values = bounds[0] if bounds.dim() == 3 else bounds
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5), sharey=True)
    axes = [axes] if n == 1 else list(axes)

    x_max = max(arrival for arrival, _, _ in snapshots)
    for ax, (arrival_time, window_start, output_available) in zip(axes, snapshots):
        for i in range(arrival_time + 1):
            lower, upper = values[i, 0].item(), values[i, 1].item()
            if i < window_start:
                ax.vlines(i, lower, upper, color="lightgray", linewidth=6)
                ax.plot(i, (lower + upper) / 2, marker="x", color="red", markersize=10)
            else:
                ax.vlines(i, lower, upper, color=COLOR_PRIMARY, linewidth=6, alpha=0.7)
        status = "output available" if output_available else "window filling"
        ax.set_title(f"t={arrival_time}\n{status}", fontsize=9)
        ax.set_xlim(-0.6, x_max + 0.6)
        ax.set_xticks(range(x_max + 1))
        ax.set_xlabel("index")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Probability bound")
    axes[0].set_ylim(0, 1.02)
    fig.suptitle("Online sliding window (arrival time = when the window last filled)")
    fig.tight_layout()

    if show:
        plt.show()
    return fig
