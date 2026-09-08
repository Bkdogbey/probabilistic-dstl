"""Presentation helpers for the offline pdSTL examples."""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch


LOWER_COLOR = "tab:blue"
UPPER_COLOR = "tab:orange"

ATOMIC_YLABEL = "probability bounds"
TEMPORAL_YLABEL = "pdSTL stochastic robustness"


def _numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _trace(value, name):
    value = _numpy(value)
    if value.ndim != 3 or value.shape[0] < 1 or value.shape[-1] != 2:
        raise ValueError(f"{name} must have shape [B,T,2]")
    return value[0]


def _plot_state(ax, time, mean, sigma, threshold, title):
    """Draw the upstream mean trace, its state endpoints, and the threshold."""
    lower, upper = mean - sigma, mean + sigma
    ax.fill_between(time, lower, upper, step="post", color=LOWER_COLOR, alpha=0.12)
    ax.step(
        time, lower, where="post", color=LOWER_COLOR, linewidth=1.5, alpha=0.9,
        label="lower endpoint",
    )
    ax.step(
        time, upper, where="post", color=UPPER_COLOR, linewidth=1.5, linestyle="--",
        alpha=0.9, label="upper endpoint",
    )
    ax.step(
        time, mean, where="post", marker="o", markersize=5, color="black",
        linewidth=2.0, label="mean",
    )
    ax.axhline(
        threshold, color="tab:red", linewidth=1.4, linestyle=":", label="threshold"
    )
    ax.set_title(title)
    ax.set_ylabel("state")
    ax.margins(y=0.30)  # headroom so the legend clears the band
    ax.legend(loc="upper left", fontsize=9, ncol=4)
    ax.grid(alpha=0.25)


def _plot_interval(ax, time, trace, title, ylabel):
    """Draw one [T,2] endpoint trace at its own valid evaluation origins."""
    origins = time[: len(trace)]
    lower, upper = trace.T
    ax.step(
        origins, lower, where="post", marker="o", markersize=4, color=LOWER_COLOR,
        linewidth=2.0, label="lower",
    )
    ax.step(
        origins, upper, where="post", marker="s", markersize=4, color=UPPER_COLOR,
        linewidth=2.0, linestyle="--", label="upper",
    )
    ax.fill_between(
        origins, lower, upper, step="post", color=LOWER_COLOR, alpha=0.12
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)


def print_temporal_results(title, atomic_trace, temporal_trace, inner_trace=None):
    """Print every endpoint pair of each trace."""
    traces = [(ATOMIC_YLABEL, atomic_trace)]
    if inner_trace is not None:
        traces.append(("inner robustness", inner_trace))
    traces.append(("temporal robustness", temporal_trace))

    print(f"\n{title}")
    for label, trace in traces:
        values = _numpy(trace)
        print(f"{label} shape: {tuple(values.shape)}")
        for step, (lower, upper) in enumerate(values[0]):
            print(f"  t={step}  [{lower:.4f}, {upper:.4f}]")


def plot_temporal_example(
    time,
    mean,
    sigma,
    threshold,
    atomic_label,
    atomic_trace,
    temporal_label,
    temporal_trace,
    inner_label=None,
    inner_trace=None,
    show=True,
):
    """Plot the state signal, its atomic bounds, and the temporal outputs.

    All panels share one time axis. Temporal traces are drawn only at their
    valid evaluation origins, so a shorter trace ends early rather than being
    stretched across the horizon.
    """
    time, mean, sigma = _numpy(time), _numpy(mean), _numpy(sigma)
    atomic = _trace(atomic_trace, "atomic_trace")
    if not (time.shape == mean.shape == sigma.shape) or len(time) != len(atomic):
        raise ValueError("time, mean, and sigma must match each other and the atomic trace")

    panels = [(f"Atomic probability bounds: {atomic_label}", atomic, ATOMIC_YLABEL)]
    if inner_trace is not None:
        panels.append(
            (f"Inner robustness: {inner_label}", _trace(inner_trace, "inner_trace"),
             TEMPORAL_YLABEL)
        )
    panels.append(
        (f"Temporal robustness: {temporal_label}",
         _trace(temporal_trace, "temporal_trace"), TEMPORAL_YLABEL)
    )

    for title, trace, _ in panels[1:]:
        if len(trace) > len(atomic):
            raise ValueError(f"{title} has more origins than the atomic trace")

    panel_count = len(panels) + 1
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(9, 2.7 * panel_count),
        layout="constrained",
        sharex=True,
    )

    _plot_state(axes[0], time, mean, sigma, threshold, "Upstream state prediction")
    for ax, (title, trace, ylabel) in zip(axes[1:], panels):
        _plot_interval(ax, time, trace, title, ylabel)

    axes[-1].set_xlim(time[0] - 0.25, time[-1] + 0.25)
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[-1].set_xlabel("time step")

    if show:
        plt.show()
    return fig
