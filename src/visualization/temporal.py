"""Presentation helpers for the offline pdSTL probability-bound examples."""

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


def _plot_interval(ax, trace, title, ylabel):
    """Draw one [T,2] endpoint trace at its own valid evaluation origins."""
    time = np.arange(len(trace))
    lower, upper = trace.T
    ax.step(
        time,
        lower,
        where="post",
        marker="o",
        markersize=4,
        color=LOWER_COLOR,
        linewidth=2.0,
        label="lower",
    )
    ax.step(
        time,
        upper,
        where="post",
        marker="s",
        markersize=4,
        color=UPPER_COLOR,
        linewidth=2.0,
        linestyle="--",
        label="upper",
    )
    ax.fill_between(
        time, lower, upper, step="post", color=LOWER_COLOR, alpha=0.12
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
    atomic_label,
    atomic_trace,
    temporal_label,
    temporal_trace,
    inner_label=None,
    inner_trace=None,
    show=True,
):
    """Plot supplied probability bounds and the temporal outputs above them.

    Every trace is drawn only at its valid evaluation origins; the panels share
    one x-axis so shorter temporal traces end early instead of being stretched.
    """
    atomic = _trace(atomic_trace, "atomic_trace")
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

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(9, 2.7 * len(panels)),
        layout="constrained",
        sharex=True,
    )

    for ax, (title, trace, ylabel) in zip(axes, panels):
        _plot_interval(ax, trace, title, ylabel)

    axes[-1].set_xlim(-0.25, len(atomic) - 0.75)
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[-1].set_xlabel("time step")

    if show:
        plt.show()
    return fig
