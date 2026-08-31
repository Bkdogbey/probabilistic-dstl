"""Visualization for bounded strong Until."""

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


def plot_until_example(
    trace,
    candidate_bounds,
    offline_bounds,
    online_bounds,
    formula,
    show=True,
):
    """Plot atomic inputs, candidate events, and final Until bounds."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    time = _numpy(trace.time)
    anchors = time[: offline_bounds.shape[1]]

    _plot_pair(axes[0], time, trace.safe_bounds, "safe", "tab:blue")
    _plot_pair(axes[0], time, trace.goal_bounds, "goal", "tab:orange")
    axes[0].set_title("Atomic predicate-probability bounds")

    colors = ("tab:green", "tab:red", "tab:purple", "tab:brown")
    for index, offset in enumerate(range(formula.a, formula.b + 1)):
        _plot_pair(
            axes[1],
            anchors,
            candidate_bounds[:, :, index, :],
            f"C{offset}",
            colors[index % len(colors)],
        )
    axes[1].set_title(
        "Candidate Cj: goal at k+j AND safe through the goal step"
    )

    _plot_pair(axes[2], anchors, offline_bounds, "offline Until", "purple")
    axes[2].scatter(
        anchors,
        _numpy(online_bounds[0, :, 0]),
        color="black",
        marker="x",
        label="streaming lower",
        zorder=3,
    )
    axes[2].scatter(
        anchors,
        _numpy(online_bounds[0, :, 1]),
        facecolors="none",
        edgecolors="black",
        label="streaming upper",
        zorder=3,
    )
    axes[2].fill_between(
        anchors,
        _numpy(offline_bounds[0, :, 0]),
        _numpy(offline_bounds[0, :, 1]),
        color="purple",
        alpha=0.12,
    )
    axes[2].set_title(f"{formula}; streaming output arrives at t=k+{formula.b}")
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
