"""Shared visualisation for the offline Gaussian temporal examples."""

import matplotlib.pyplot as plt
import numpy as np
import torch


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _bounds(ax, time, trace, title):
    lower, upper = trace.T
    ax.plot(time, lower, "o-", label="lower probability")
    ax.plot(time, upper, "o--", label="upper probability")
    ax.fill_between(time, lower, upper, alpha=0.15)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_temporal_example(
    time,
    mean,
    variance,
    confidence_level,
    atomic_trace,
    temporal_trace,
    predicate,
    formula,
    interval,
    show=True,
):
    """Plot model uncertainty, predicate bounds, and temporal bounds."""
    time, mean, variance = map(_numpy, (time, mean, variance))
    atomic, temporal = _numpy(atomic_trace)[0], _numpy(temporal_trace)[0]
    if atomic.shape != (len(time), 2) or temporal.ndim != 2 or temporal.shape[1] != 2:
        raise ValueError("expected atomic [B,T,2] and temporal [B,K,2] traces")

    std = np.sqrt(variance)
    lower_state = mean - confidence_level * std
    upper_state = mean + confidence_level * std
    comparison = f"x {predicate.sense} {predicate.threshold:g}"
    origin_time = time[: len(temporal)]

    fig, (state_ax, atomic_ax, temporal_ax) = plt.subplots(
        3, 1, figsize=(8, 9), layout="constrained"
    )
    state_ax.fill_between(time, lower_state, upper_state, alpha=0.18, label="belief band")
    state_ax.plot(time, mean, color="black", linestyle=":", label="mean")
    state_ax.plot(time, lower_state, label=rf"$\mu - {confidence_level:g}\sigma$")
    state_ax.plot(time, upper_state, linestyle="--", label=rf"$\mu + {confidence_level:g}\sigma$")
    state_ax.axhline(predicate.threshold, color="tab:red", linestyle="--", label="threshold")
    state_ax.set_title("Gaussian state prediction")
    state_ax.set_ylabel("state")
    state_ax.legend(fontsize=8)
    state_ax.grid(alpha=0.3)

    _bounds(atomic_ax, time, atomic, f"Atomic predicate probability: {comparison}")
    atomic_ax.set_ylabel("probability")
    _bounds(
        temporal_ax,
        origin_time,
        temporal,
        f"{type(formula).__name__}{list(interval)}({comparison}) at valid origins",
    )
    temporal_ax.set_ylabel("stochastic robustness")
    temporal_ax.set_xlabel("time")
    if show:
        plt.show()
    return fig
