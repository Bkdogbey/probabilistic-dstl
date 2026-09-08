"""Presentation helpers for the offline bounded-interval pdSTL examples."""

import matplotlib.pyplot as plt
import numpy as np
import torch


LOWER_COLOR = "tab:blue"
UPPER_COLOR = "tab:orange"


def _numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _trace(value, name):
    value = _numpy(value)
    if value.ndim != 3 or value.shape[0] < 1 or value.shape[-1] != 2:
        raise ValueError(f"{name} must have shape [B,T,2]")
    return value[0]


def _plot_interval(ax, time, trace, title, ylabel):
    lower, upper = trace.T
    ax.step(
        time, lower, where="post", color=LOWER_COLOR, linewidth=2.0, label="lower"
    )
    ax.step(
        time,
        upper,
        where="post",
        color=UPPER_COLOR,
        linewidth=2.0,
        linestyle="--",
        label="upper",
    )
    ax.fill_between(time, lower, upper, step="post", color=LOWER_COLOR, alpha=0.12)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)


def print_temporal_results(title, atomic_trace, temporal_trace, inner_trace=None):
    """Print trace shapes and the first three endpoint pairs."""
    traces = [("atomic probability", atomic_trace)]
    if inner_trace is not None:
        traces.append(("inner robustness", inner_trace))
    traces.append(("temporal robustness", temporal_trace))

    print(f"\n{title}")
    for label, trace in traces:
        values = _numpy(trace)
        print(f"{label} shape: {tuple(values.shape)}")
        print(f"{label} endpoints: {values[0, :3]}")


def plot_temporal_example(
    title,
    time,
    lower_state,
    upper_state,
    threshold,
    atomic_trace,
    temporal_trace,
    inner_trace=None,
    show=True,
):
    """Plot state bounds, atomic probability bounds, and temporal output.

    Every temporal trace is plotted only at its valid evaluation origins.
    """
    time, lower_state, upper_state = map(
        _numpy, (time, lower_state, upper_state)
    )
    atomic = _trace(atomic_trace, "atomic_trace")
    temporal = _trace(temporal_trace, "temporal_trace")
    matching_shapes = lower_state.shape == upper_state.shape == time.shape
    if time.ndim != 1 or not matching_shapes:
        raise ValueError(
            "time, lower_state, and upper_state must be matching scalar traces"
        )
    if atomic.shape != (len(time), 2):
        raise ValueError("atomic_trace must span the complete time vector")
    inner = None if inner_trace is None else _trace(inner_trace, "inner_trace")

    panel_count = 4 if inner is not None else 3
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(9, 2.7 * panel_count),
        layout="constrained",
        sharex=False,
    )

    state_ax = axes[0]
    state_ax.fill_between(
        time, lower_state, upper_state, step="post", color=LOWER_COLOR, alpha=0.12
    )
    state_ax.step(
        time,
        lower_state,
        where="post",
        color=LOWER_COLOR,
        linewidth=1.7,
        label="lower bound",
    )
    state_ax.step(
        time,
        upper_state,
        where="post",
        color=UPPER_COLOR,
        linewidth=1.7,
        linestyle="--",
        label="upper bound",
    )
    state_ax.axhline(
        threshold,
        color="tab:red",
        linewidth=1.4,
        linestyle=":",
        label="threshold",
    )
    state_ax.set_title("State bounds")
    state_ax.set_ylabel("state")
    state_ax.legend(loc="best", fontsize=9, ncol=2)
    state_ax.grid(alpha=0.25)

    _plot_interval(
        axes[1],
        time,
        atomic,
        f"Atomic probability bounds: x ≥ {threshold:g}",
        "probability",
    )

    final_axis = axes[-1]
    if inner is not None:
        _plot_interval(
            axes[2],
            time[: len(inner)],
            inner,
            "Inner robustness: Always",
            "pdSTL stochastic robustness",
        )

    _plot_interval(
        final_axis,
        time[: len(temporal)],
        temporal,
        f"{title} temporal robustness",
        "pdSTL stochastic robustness",
    )
    final_axis.set_xlabel("time step")

    if show:
        plt.show()
    return fig
