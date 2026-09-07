"""Shared plots for evaluated Gaussian altitude traces."""

import matplotlib.pyplot as plt
import numpy as np
import torch


def _as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _plot_interval(ax, time, bounds):
    lower, upper = bounds.T
    ax.plot(time, lower, color="tab:blue", marker="o", label="lower endpoint")
    ax.plot(
        time,
        upper,
        color="tab:orange",
        linestyle="--",
        marker=".",
        label="upper endpoint",
    )
    ax.fill_between(time, lower, upper, color="tab:blue", alpha=0.15)
    ax.set_ylim(-0.03, 1.03)


def plot_temporal_example(
    time,
    mean,
    variance,
    atomic_trace,
    temporal_trace,
    threshold,
    formula_label,
    show=True,
):
    """Plot a scalar trace and its valid temporal origins; return the figure."""
    time, mean, variance = map(_as_numpy, (time, mean, variance))
    atomic = _as_numpy(atomic_trace)[0]
    temporal = _as_numpy(temporal_trace)[0]
    if mean.shape != time.shape or variance.shape != time.shape:
        raise ValueError("time, mean, and variance must have matching scalar traces")
    if atomic.shape != (len(time), 2) or temporal.shape[1:] != (2,):
        raise ValueError("expected atomic [1,T,2] and temporal [1,K,2] traces")
    if not 0 < len(temporal) <= len(time):
        raise ValueError("temporal trace must contain valid origins within the input")
    fig, (state_ax, atomic_ax, temporal_ax) = plt.subplots(
        3, 1, figsize=(8, 9), layout="constrained"
    )
    std = np.sqrt(variance)
    state_ax.plot(time, mean, color="black", marker="o", label="mean altitude")
    state_ax.fill_between(
        time,
        mean - std,
        mean + std,
        color="gray",
        alpha=0.25,
        label="mean ± one standard deviation",
    )
    state_ax.axhline(
        threshold, color="tab:red", linestyle="--", label=f"threshold = {threshold:g} m"
    )
    state_ax.set_title("Gaussian altitude predictions")
    state_ax.set_ylabel("Altitude (m)")
    state_ax.set_xlabel("Prediction time (s)")
    _plot_interval(atomic_ax, time, atomic)
    atomic_ax.set_title(f"Atomic probability: altitude ≥ {threshold:g} m")
    atomic_ax.set_ylabel("Probability")
    atomic_ax.set_xlabel("Prediction time (s)")
    # Complete-window outputs are indexed by the origin, including when a > 0.
    origin_time = time[: len(temporal)]
    _plot_interval(temporal_ax, origin_time, temporal)
    temporal_ax.set_title(formula_label)
    temporal_ax.set_ylabel("Stochastic robustness")
    temporal_ax.set_xlabel("Window origin (s)")
    for ax in fig.axes:
        ax.set_xticks(time if ax is not temporal_ax else origin_time)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    if show:
        plt.show()
    return fig
