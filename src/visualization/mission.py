"""Visualization for the composed Always-and-Eventually mission."""

import matplotlib.pyplot as plt


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def _plot_pair(ax, time, bounds, label, color):
    values = bounds[0]
    ax.plot(time, _numpy(values[:, 0]), color=color, marker="o", label=f"{label} lower")
    ax.plot(
        time,
        _numpy(values[:, 1]),
        color=color,
        linestyle="--",
        marker="o",
        label=f"{label} upper",
    )


def plot_mission_example(
    trace,
    always_bounds,
    eventually_bounds,
    mission_bounds,
    online_mission,
    interval,
    show=True,
):
    """Plot atomic inputs, temporal branches, and their mission conjunction."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    time = _numpy(trace.time)

    _plot_pair(axes[0], time, trace.safe_bounds, "safe", "tab:blue")
    _plot_pair(axes[0], time, trace.goal_bounds, "goal", "tab:orange")
    axes[0].set_title("Atomic predicate-probability bounds")

    anchor_count = mission_bounds.shape[1]
    anchors = time[:anchor_count]
    _plot_pair(axes[1], anchors, always_bounds, "Always", "tab:blue")
    _plot_pair(axes[1], anchors, eventually_bounds, "Eventually", "tab:orange")
    axes[1].set_title("Two temporal branches")

    _plot_pair(axes[2], anchors, mission_bounds, "offline mission", "purple")
    axes[2].scatter(
        anchors,
        _numpy(online_mission[0, :, 0]),
        color="black",
        marker="x",
        label="streaming lower",
        zorder=3,
    )
    axes[2].scatter(
        anchors,
        _numpy(online_mission[0, :, 1]),
        facecolors="none",
        edgecolors="black",
        label="streaming upper",
        zorder=3,
    )
    axes[2].fill_between(
        anchors,
        _numpy(mission_bounds[0, :, 0]),
        _numpy(mission_bounds[0, :, 1]),
        color="purple",
        alpha=0.12,
    )
    axes[2].set_title(
        f"Always AND Eventually mission; streaming output arrives at t=k+{interval[1]}"
    )
    axes[2].set_xlabel("STL anchor k")

    for ax in axes:
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", ncol=2)

    fig.tight_layout()
    if show:
        plt.show()
    return fig
