"""Visualization for an already-evaluated offline temporal example."""

import matplotlib.pyplot as plt
import torch


def _as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else value


def plot_temporal_example(
    model,
    atomic_bounds,
    temporal_bounds,
    interval,
    formula_label,
    show=True,
):
    """Plot physical, atomic, and temporal bounds without evaluating them."""
    fig, (ax_belief, ax_atomic, ax_temporal) = plt.subplots(
        3, 1, figsize=(8, 10)
    )

    time = _as_numpy(model.time)
    mean_lower = _as_numpy(model.mean_lower)
    mean_upper = _as_numpy(model.mean_upper)
    nominal_mean = (mean_lower + mean_upper) / 2

    ax_belief.plot(
        time,
        nominal_mean,
        color="black",
        marker="o",
        label="nominal mean",
    )
    ax_belief.fill_between(
        time,
        mean_lower,
        mean_upper,
        color="gray",
        alpha=0.3,
        label="admissible Gaussian mean",
    )
    ax_belief.axhline(
        model.threshold,
        color="red",
        linestyle="--",
        label=f"threshold = {model.threshold:g} m",
    )
    ax_belief.set_title("Altitude belief model")
    ax_belief.set_ylabel("Altitude (m)")
    ax_belief.set_xticks(time)
    ax_belief.grid(True, alpha=0.3)
    ax_belief.legend(loc="best")

    atomic_lower = _as_numpy(atomic_bounds[0, :, 0])
    atomic_upper = _as_numpy(atomic_bounds[0, :, 1])
    ax_atomic.plot(
        time,
        atomic_lower,
        color="tab:blue",
        marker="o",
        label="lower bound",
    )
    ax_atomic.plot(
        time,
        atomic_upper,
        color="tab:orange",
        linestyle="--",
        marker="o",
        label="upper bound",
    )
    ax_atomic.fill_between(
        time, atomic_lower, atomic_upper, color="tab:blue", alpha=0.15
    )
    ax_atomic.set_title(
        f"Atomic bounds: P(altitude >= {model.threshold:g} m)"
    )
    ax_atomic.set_ylabel("Probability")
    ax_atomic.set_xticks(time)
    ax_atomic.set_ylim(0, 1)
    ax_atomic.grid(True, alpha=0.3)
    ax_atomic.legend(loc="best")

    output_count = temporal_bounds.shape[1]
    ax_temporal.set_title(f"Temporal bounds: {formula_label}")
    ax_temporal.set_xlabel("STL anchor k")
    ax_temporal.set_ylabel("Probability")
    ax_temporal.set_ylim(0, 1)
    ax_temporal.grid(True, alpha=0.3)

    if output_count == 0:
        ax_temporal.text(
            0.5,
            0.5,
            f"No complete windows for interval [{interval[0]},{interval[1]}]",
            ha="center",
            va="center",
            transform=ax_temporal.transAxes,
        )
    else:
        anchor_time = time[:output_count]
        temporal_lower = _as_numpy(temporal_bounds[0, :, 0])
        temporal_upper = _as_numpy(temporal_bounds[0, :, 1])
        ax_temporal.plot(
            anchor_time,
            temporal_lower,
            color="purple",
            marker="o",
            label="lower bound",
        )
        ax_temporal.plot(
            anchor_time,
            temporal_upper,
            color="red",
            linestyle="--",
            marker="o",
            label="upper bound",
        )
        ax_temporal.fill_between(
            anchor_time,
            temporal_lower,
            temporal_upper,
            color="purple",
            alpha=0.15,
        )
        ax_temporal.set_xticks(anchor_time)
        ax_temporal.legend(loc="best")

    fig.tight_layout()
    if show:
        plt.show()
    return fig
