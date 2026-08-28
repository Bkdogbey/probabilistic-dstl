"""Independent offline Boolean and temporal examples."""

import torch

from models.boolean import boolean_example
from models.drone import always_altitude_example, eventually_altitude_example
from pdstl import Always, Eventually, OfflineSource, Predicate
from visualization.temporal import plot_temporal_example


def run_boolean_example():
    """Evaluate one hand-checkable predicate/Boolean example."""
    bounds_a, bounds_b = boolean_example()
    a = Predicate("A")
    b = Predicate("B")
    source = OfflineSource({a: bounds_a, b: bounds_b})

    not_a = (~a)(source)
    and_ab = (a & b)(source)
    or_ab = (a | b)(source)

    dtype = bounds_a.dtype
    torch.testing.assert_close(not_a, torch.tensor([[[0.10, 0.40]]], dtype=dtype))
    torch.testing.assert_close(and_ab, torch.tensor([[[0.30, 0.90]]], dtype=dtype))
    torch.testing.assert_close(or_ab, torch.tensor([[[0.70, 1.00]]], dtype=dtype))

    print("A = [0.60, 0.90]   B = [0.70, 0.95]")
    print(f"not A   = [{not_a[0, 0, 0]:.2f}, {not_a[0, 0, 1]:.2f}]")
    print(f"A AND B = [{and_ab[0, 0, 0]:.2f}, {and_ab[0, 0, 1]:.2f}]")
    print(f"A OR B  = [{or_ab[0, 0, 0]:.2f}, {or_ab[0, 0, 1]:.2f}]")
    return not_a, and_ab, or_ab


def _print_atomic_bounds(model, atomic_bounds):
    print("Atomic predicate bounds")
    for time, bounds in zip(model.time.tolist(), atomic_bounds[0].tolist()):
        lower, upper = bounds
        print(f"t={time}: [{lower:.4f}, {upper:.4f}]")


def _print_temporal_bounds(formula_label, interval, temporal_bounds, trace_length):
    print(formula_label)
    if temporal_bounds.shape[1] == 0:
        print(
            f"No complete temporal windows: interval [{interval[0]},{interval[1]}] "
            f"needs at least {interval[1] + 1} time steps; trace has {trace_length}."
        )
        return

    a, b = interval
    for anchor, bounds in enumerate(temporal_bounds[0].tolist()):
        lower, upper = bounds
        print(
            f"k={anchor}, window=[{anchor + a},{anchor + b}]: "
            f"[{lower:.4f}, {upper:.4f}]"
        )


def _run_temporal(*, model_factory, operator_type, threshold, interval, show):
    model = model_factory(threshold=threshold)
    predicate = Predicate(f"altitude >= {float(threshold):g} m")
    source = OfflineSource({predicate: model.probability_bounds})
    atomic_bounds = predicate(source)
    temporal_operator = operator_type(predicate, interval)
    temporal_bounds = temporal_operator(source)
    operator_name = operator_type.__name__
    formula_label = f"{operator_name}[{interval[0]},{interval[1]}]({predicate})"

    _print_atomic_bounds(model, atomic_bounds)
    _print_temporal_bounds(
        formula_label,
        interval,
        temporal_bounds,
        trace_length=model.time.shape[0],
    )
    print("Atomic intervals bound the marginal predicate probability at each time.")
    print("Temporal intervals also account for unknown dependence across time.")

    figure = plot_temporal_example(
        model,
        atomic_bounds,
        temporal_bounds,
        interval,
        formula_label,
        show=show,
    )
    return model, atomic_bounds, temporal_bounds, figure


def run_always_example(threshold, interval, show=True):
    """Evaluate the configurable offline altitude-Always example."""
    return _run_temporal(
        model_factory=always_altitude_example,
        operator_type=Always,
        threshold=threshold,
        interval=interval,
        show=show,
    )


def run_eventually_example(threshold, interval, show=True):
    """Evaluate the configurable offline altitude-Eventually example."""
    return _run_temporal(
        model_factory=eventually_altitude_example,
        operator_type=Eventually,
        threshold=threshold,
        interval=interval,
        show=show,
    )
