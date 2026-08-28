"""The pdSTL demonstration entry point.

Run with::

    python src/main.py

Three independent experiments, each toggled by flipping "run"/"skip" in
main()'s skip_run(...) calls. A skipped experiment builds nothing and
prints nothing. The Always/Eventually windows are plain literals passed to
Always(...)/Eventually(...) below -- edit them directly to experiment; the
verification, labels, and prints all derive from the constructed operator
and its actual output, so they stay correct for any window or trace length.
"""

import torch

from models.boolean import boolean_example
from models.drone import always_altitude_example, eventually_altitude_example
from pdstl import And, Always, Eventually, Not, OfflineSource, Or, Predicate
from utils import skip_run
from visualization.temporal import plot_temporal_example


def run_boolean_example():
    bounds_a, bounds_b = boolean_example()
    a = Predicate("A")
    b = Predicate("B")
    source = OfflineSource({a: bounds_a, b: bounds_b})

    torch.testing.assert_close(a(source), bounds_a)
    torch.testing.assert_close(b(source), bounds_b)

    not_a = Not(a)(source)
    and_ab = And(a, b)(source)
    or_ab = Or(a, b)(source)

    dtype = bounds_a.dtype
    torch.testing.assert_close(not_a, torch.tensor([[[0.10, 0.40]]], dtype=dtype))
    torch.testing.assert_close(and_ab, torch.tensor([[[0.30, 0.90]]], dtype=dtype))
    torch.testing.assert_close(or_ab, torch.tensor([[[0.70, 1.00]]], dtype=dtype))

    print("A = [0.60, 0.90]   B = [0.70, 0.95]")
    print(f"not A   = {not_a[0, 0].tolist()}   (expected [0.10, 0.40])")
    print(f"A AND B = {and_ab[0, 0].tolist()}   (expected [0.30, 0.90])")
    print(f"A OR B  = {or_ab[0, 0].tolist()}   (expected [0.70, 1.00])")


def _always_windows(p, a, b):
    """[lower, upper] for every valid Always[a,b] window of 1-D tensor p.

    Shape [n, 2], n = max(len(p) - b, 0) -- correct for any window or trace
    length, including n == 0 when the window doesn't fit yet.
    """
    width = b - a + 1
    shifted = p[a:]
    if shifted.shape[0] < width:
        return torch.empty(0, 2, dtype=p.dtype)
    windows = shifted.unfold(0, width, 1)
    lower = torch.clamp(windows.sum(dim=-1) - (width - 1), min=0.0)
    upper = windows.amin(dim=-1)
    return torch.stack([lower, upper], dim=-1)


def _eventually_windows(p, a, b):
    """[lower, upper] for every valid Eventually[a,b] window of 1-D tensor p.

    Shape [n, 2], n = max(len(p) - b, 0).
    """
    width = b - a + 1
    shifted = p[a:]
    if shifted.shape[0] < width:
        return torch.empty(0, 2, dtype=p.dtype)
    windows = shifted.unfold(0, width, 1)
    lower = windows.amax(dim=-1)
    upper = torch.clamp(windows.sum(dim=-1), max=1.0)
    return torch.stack([lower, upper], dim=-1)


def _print_windows(operator, result, available_steps):
    if result.shape[1] == 0:
        print(f"{operator}: no output yet -- needs {operator.b + 1} steps, "
              f"only {available_steps} available")
        return False
    for k in range(result.shape[1]):
        lower, upper = result[0, k].tolist()
        print(f"{operator} at k={k}: [{lower:.4f}, {upper:.4f}]")
    return True


def run_always_example():
    model = always_altitude_example()
    predicate = Predicate("altitude >= 50 m")
    source = OfflineSource({predicate: model.probability_bounds})

    atomic = predicate(source)
    always_op = Always(predicate, (0, 2))
    result = always_op(source)

    p = atomic[0, :, 0]
    expected = _always_windows(p, always_op.a, always_op.b).unsqueeze(0)
    torch.testing.assert_close(result, expected)

    print(f"atomic probabilities: {p.tolist()}")
    if not _print_windows(always_op, result, p.shape[0]):
        return
    print("The plotted altitude line is the belief mean, not a deterministic")
    print("realized path. A mean above 50 m does not imply probability one")
    print("because the Gaussian belief still assigns probability below 50 m.")

    plot_temporal_example(model, atomic, result, str(always_op))


def run_eventually_example():
    model = eventually_altitude_example()
    predicate = Predicate("altitude >= 55 m")
    source = OfflineSource({predicate: model.probability_bounds})

    atomic = predicate(source)
    eventually_op = Eventually(predicate, (0, 2))
    result = eventually_op(source)

    p = atomic[0, :, 0]
    expected = _eventually_windows(p, eventually_op.a, eventually_op.b).unsqueeze(0)
    torch.testing.assert_close(result, expected)

    print(f"atomic probabilities: {p.tolist()}")
    if not _print_windows(eventually_op, result, p.shape[0]):
        return
    print("The belief mean crosses 55 m at t=2, but this does not make the")
    print("event certain. The uncertain altitude can still be below 55 m.")

    plot_temporal_example(model, atomic, result, str(eventually_op))


def main():
    with skip_run("run", "Boolean operators") as check, check():
        run_boolean_example()

    with skip_run("run", "Always above 50 m") as check, check():
        run_always_example()

    with skip_run("run", "Eventually above 55 m") as check, check():
        run_eventually_example()


if __name__ == "__main__":
    main()
