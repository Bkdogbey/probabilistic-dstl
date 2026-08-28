"""Offline sliding-window and true incremental temporal examples."""

from dataclasses import dataclass

import torch

from models.streaming import sliding_always_example, sliding_eventually_example
from pdstl import Always, Eventually, OfflineSource, OnlineSource, Predicate
from visualization.streaming import (
    plot_sliding_windows,
    plot_streaming_animation,
    plot_streaming_updates,
)


@dataclass(frozen=True)
class StreamingUpdate:
    """One arrival, its retained temporal state, and its optional output."""

    arrival: int
    window_state: torch.Tensor
    output: torch.Tensor | None


def _offline(trace, operator_type, interval):
    predicate = Predicate(trace.predicate_name)
    source = OfflineSource({predicate: trace.bounds})
    operator = operator_type(predicate, interval)
    return predicate, operator, operator(source)


def _stream(trace, predicate, operator):
    source = OnlineSource()
    window_state = None
    outputs = []
    updates = []

    for arrival, incoming in enumerate(trace.bounds.unbind(dim=1)):
        source.append({predicate: incoming})
        current_bounds = source.bounds(predicate, arrival)
        output, window_state = operator.step(current_bounds, window_state)
        updates.append(
            StreamingUpdate(
                arrival=arrival,
                window_state=window_state.clone(),
                output=None if output is None else output.clone(),
            )
        )
        if output is not None:
            outputs.append(output)

    online_bounds = (
        torch.stack(outputs, dim=1) if outputs else trace.bounds[:, :0, :]
    )
    return online_bounds, updates


def _print_sliding_results(operator, temporal_bounds):
    a, b = operator.interval
    print(f"{operator}: {temporal_bounds.shape[1]} complete windows")
    for anchor, (lower, upper) in enumerate(temporal_bounds[0].tolist()):
        print(
            f"k={anchor}, window=[{anchor + a},{anchor + b}]: "
            f"[{lower:.4f}, {upper:.4f}]"
        )


def run_sliding_always_example(interval=(0, 5), show=True):
    """Evaluate all offline windows of the 11-step Always trace."""
    trace = sliding_always_example()
    _, operator, temporal_bounds = _offline(trace, Always, interval)
    _print_sliding_results(operator, temporal_bounds)
    formula_label = f"Always[{interval[0]},{interval[1]}]({trace.predicate_name})"
    figure = plot_sliding_windows(
        trace,
        temporal_bounds,
        interval,
        formula_label,
        show=show,
    )
    return trace, temporal_bounds, figure


def _evaluate_streaming(trace, operator_type, interval):
    predicate, operator, offline_bounds = _offline(trace, operator_type, interval)
    online_bounds, updates = _stream(trace, predicate, operator)
    torch.testing.assert_close(online_bounds, offline_bounds)
    return operator, offline_bounds, online_bounds, updates


def _run_streaming(trace, operator_type, interval, show):
    operator, offline_bounds, online_bounds, updates = _evaluate_streaming(
        trace, operator_type, interval
    )

    print(f"Streaming {operator}")
    for update in updates:
        if update.output is None:
            output = "window filling"
        else:
            lower, upper = update.output[0].tolist()
            output = f"[{lower:.4f}, {upper:.4f}]"
        print(
            f"t={update.arrival}, state length={update.window_state.shape[1]}: "
            f"{output}"
        )
    print("Online step() outputs match the complete offline trace.")

    formula_label = (
        f"{operator_type.__name__}[{interval[0]},{interval[1]}]"
        f"({trace.predicate_name})"
    )
    figure = plot_streaming_updates(
        trace,
        updates,
        offline_bounds,
        online_bounds,
        interval,
        formula_label,
        show=show,
    )
    return trace, offline_bounds, online_bounds, updates, figure


def run_streaming_always_example(interval=(0, 5), show=True):
    """Stream the Always trace through one persistent temporal state."""
    return _run_streaming(sliding_always_example(), Always, interval, show)


def run_streaming_eventually_example(interval=(0, 5), show=True):
    """Stream the Eventually trace through one persistent temporal state."""
    return _run_streaming(sliding_eventually_example(), Eventually, interval, show)


def run_streaming_always_animation(
    interval=(0, 5),
    frame_interval_ms=900,
    repeat=True,
    show=True,
):
    """Animate how an Always window fills, shifts, and emits bounds."""
    trace = sliding_always_example()
    operator, offline_bounds, online_bounds, updates = _evaluate_streaming(
        trace, Always, interval
    )
    print(f"Animating {operator}: one frame per incoming interval.")
    figure, movie = plot_streaming_animation(
        trace,
        updates,
        interval,
        f"Always[{interval[0]},{interval[1]}]({trace.predicate_name})",
        frame_interval_ms=frame_interval_ms,
        repeat=repeat,
        show=show,
    )
    return trace, offline_bounds, online_bounds, updates, figure, movie
