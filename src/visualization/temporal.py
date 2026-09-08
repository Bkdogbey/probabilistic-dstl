"""Presentation helpers for the offline Gaussian pdSTL examples."""

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


def _comparison_math(predicate):
    relation = r"\geq" if predicate.sense == ">=" else r"\leq"
    return rf"x_{{{predicate.dim}}} {relation} {predicate.threshold:g}"


def _formula_math(config, predicate):
    operator = config["operator"]
    symbol = r"\mathbf{G}" if operator == "always" else r"\mathbf{F}"
    seconds = "interval_sec" in config
    interval = config["interval_sec" if seconds else "interval_steps"]
    unit = r"\,\mathrm{s}" if seconds else ""
    interval_text = rf"[{interval[0]:g}, {interval[1]:g}]{unit}"
    child = config.get("child")
    child_text = (
        _comparison_math(predicate)
        if child is None
        else _formula_math(child, predicate)
    )
    return rf"{symbol}_{{{interval_text}}}\left({child_text}\right)"


def _plot_interval(ax, time, trace, title, ylabel):
    lower, upper = trace.T
    ax.plot(time, lower, color=LOWER_COLOR, linewidth=2.0, label="lower")
    ax.plot(
        time,
        upper,
        color=UPPER_COLOR,
        linewidth=2.0,
        linestyle="--",
        label="upper",
    )
    ax.fill_between(time, lower, upper, color=LOWER_COLOR, alpha=0.12)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)


def print_temporal_results(result):
    """Print trace shapes and the first three endpoint pairs."""
    traces = [("atomic probability", result.atomic_trace)]
    if result.inner_trace is not None:
        traces.append(("inner robustness", result.inner_trace))
    traces.append(("temporal robustness", result.temporal_trace))

    print(f"\n{result.title}")
    for label, trace in traces:
        values = _numpy(trace)
        print(f"{label} shape: {tuple(values.shape)}")
        print(f"{label} endpoints: {values[0, :3]}")


def plot_temporal_example(result, show=True):
    """Plot a result at each trace's valid evaluation origins."""
    time, mean, variance = map(
        _numpy, (result.time, result.mean, result.variance)
    )
    atomic = _trace(result.atomic_trace, "atomic_trace")
    temporal = _trace(result.temporal_trace, "temporal_trace")
    matching_shapes = mean.shape == time.shape == variance.shape
    if time.ndim != 1 or not matching_shapes:
        raise ValueError(
            "time, mean, and variance must be matching scalar traces"
        )
    if atomic.shape != (len(time), 2):
        raise ValueError("atomic_trace must span the complete time vector")
    inner = (
        None
        if result.inner_trace is None
        else _trace(result.inner_trace, "inner_trace")
    )
    panel_count = 4 if inner is not None else 3
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(9, 2.7 * panel_count),
        layout="constrained",
        sharex=False,
    )

    std = np.sqrt(variance)
    lower_state = mean - result.sigma_multiplier * std
    upper_state = mean + result.sigma_multiplier * std
    state_ax = axes[0]
    state_ax.fill_between(
        time, lower_state, upper_state, color=LOWER_COLOR, alpha=0.12
    )
    state_ax.plot(time, mean, color="black", linewidth=2.0, label=r"$\mu$")
    state_ax.plot(
        time,
        lower_state,
        color=LOWER_COLOR,
        linewidth=1.7,
        label=rf"$\mu-{result.sigma_multiplier:g}\sigma$",
    )
    state_ax.plot(
        time,
        upper_state,
        color=UPPER_COLOR,
        linewidth=1.7,
        linestyle="--",
        label=rf"$\mu+{result.sigma_multiplier:g}\sigma$",
    )
    state_ax.axhline(
        result.predicate.threshold,
        color="tab:red",
        linewidth=1.4,
        linestyle=":",
        label="threshold",
    )
    state_ax.set_title("Gaussian state prediction")
    state_ax.set_ylabel("state")
    state_ax.legend(loc="best", fontsize=9, ncol=2)
    state_ax.grid(alpha=0.25)

    predicate_math = _comparison_math(result.predicate)
    _plot_interval(
        axes[1],
        time,
        atomic,
        rf"Pointwise probability interval: ${predicate_math}$",
        "probability",
    )

    final_axis = axes[-1]
    if inner is not None:
        inner_title = _formula_math(
            result.formula_config["child"], result.predicate
        )
        _plot_interval(
            axes[2],
            time[: len(inner)],
            inner,
            rf"Inner robustness: ${inner_title}$",
            "pdSTL stochastic robustness",
        )

    _plot_interval(
        final_axis,
        time[: len(temporal)],
        temporal,
        rf"Temporal robustness: "
        rf"${_formula_math(result.formula_config, result.predicate)}$",
        "pdSTL stochastic robustness",
    )
    time_label = "time (s)" if result.signal_type == "linear" else "time step"
    final_axis.set_xlabel(time_label)

    if show:
        plt.show()
    return fig


def present_temporal_example(result, show=True):
    """Print and plot one evaluated example."""
    print_temporal_results(result)
    return plot_temporal_example(result, show=show)
