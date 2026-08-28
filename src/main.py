import torch

from models.boolean import boolean_example
from models.drone import always_altitude_example, eventually_altitude_example
from pdstl import Always, Eventually, OfflineSource, Predicate
from utils import skip_run
from visualization.temporal import plot_temporal_example

RUN_BOOLEAN = "run"
RUN_ALWAYS = "run"
RUN_EVENTUALLY = "run"

ALWAYS_THRESHOLD = 50.0
ALWAYS_INTERVAL = (0, 2)

EVENTUALLY_THRESHOLD = 55.0
EVENTUALLY_INTERVAL = (0, 2)


def run_boolean_example():
    bounds_a, bounds_b = boolean_example()
    a = Predicate("A")
    b = Predicate("B")
    source = OfflineSource({a: bounds_a, b: bounds_b})

    torch.testing.assert_close(a(source), bounds_a)
    torch.testing.assert_close(b(source), bounds_b)

    not_a = (~a)(source)
    and_ab = (a & b)(source)
    or_ab = (a | b)(source)

    dtype = bounds_a.dtype
    torch.testing.assert_close(not_a, torch.tensor([[[0.10, 0.40]]], dtype=dtype))
    torch.testing.assert_close(and_ab, torch.tensor([[[0.30, 0.90]]], dtype=dtype))
    torch.testing.assert_close(or_ab, torch.tensor([[[0.70, 1.00]]], dtype=dtype))

    print("A = [0.60, 0.90]   B = [0.70, 0.95]")
    print(f"not A   = {not_a[0, 0].tolist()}   (expected [0.10, 0.40])")
    print(f"A AND B = {and_ab[0, 0].tolist()}   (expected [0.30, 0.90])")
    print(f"A OR B  = {or_ab[0, 0].tolist()}   (expected [0.70, 1.00])")
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
        window = f"[{anchor + a},{anchor + b}]"
        print(f"k={anchor}, window={window}: [{lower:.4f}, {upper:.4f}]")


def _print_temporal_interpretation(operator_name):
    print("Atomic intervals bound the marginal predicate probability at each time.")
    print("Temporal intervals bound the formula-satisfaction probability because")
    print("dependence between different time events is not specified.")
    influence = "low" if operator_name == "Always" else "high"
    print(
        f"For {operator_name}, a {influence} atomic bound affects every window "
        "containing that time."
    )


def _run_temporal(
    *,
    model_factory,
    operator_type,
    threshold,
    interval,
    show,
):
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
    _print_temporal_interpretation(operator_name)

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
    return _run_temporal(
        model_factory=always_altitude_example,
        operator_type=Always,
        threshold=threshold,
        interval=interval,
        show=show,
    )


def run_eventually_example(threshold, interval, show=True):
    return _run_temporal(
        model_factory=eventually_altitude_example,
        operator_type=Eventually,
        threshold=threshold,
        interval=interval,
        show=show,
    )


def main(
    *,
    show=True,
    run_boolean=RUN_BOOLEAN,
    run_always=RUN_ALWAYS,
    run_eventually=RUN_EVENTUALLY,
):
    with skip_run(run_boolean, "Boolean operators") as check, check():
        run_boolean_example()

    with skip_run(run_always, "Always") as check, check():
        run_always_example(
            threshold=ALWAYS_THRESHOLD,
            interval=ALWAYS_INTERVAL,
            show=show,
        )

    with skip_run(run_eventually, "Eventually") as check, check():
        run_eventually_example(
            threshold=EVENTUALLY_THRESHOLD,
            interval=EVENTUALLY_INTERVAL,
            show=show,
        )


if __name__ == "__main__":
    main()
