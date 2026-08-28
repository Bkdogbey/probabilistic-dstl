"""The pdSTL demonstration entry point.

Run with::

    python src/main.py

Three independent experiments, each toggled by flipping "run"/"skip" in
main()'s skip_run(...) calls. A skipped experiment builds nothing and
prints nothing.
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


def run_always_example():
    model = always_altitude_example()
    predicate = Predicate("altitude >= 50 m")
    source = OfflineSource({predicate: model.probability_bounds})

    atomic = predicate(source)
    result = Always(predicate, (0, 2))(source)

    p = atomic[0, :, 0]
    lower = torch.clamp(p.sum() - 2, min=0.0)
    upper = p.amin()
    expected = torch.stack([lower, upper]).reshape(1, 1, 2)
    torch.testing.assert_close(result, expected)

    print(f"p0, p1, p2 = {p.tolist()}")
    print(f"Always[0,2](altitude >= 50 m) = {result[0, 0].tolist()}"
          f"   (expected [{lower.item():.4f}, {upper.item():.4f}])")
    print("The plotted altitude line is the belief mean, not a deterministic")
    print("realized path. A mean above 50 m does not imply probability one")
    print("because the Gaussian belief still assigns probability below 50 m.")

    plot_temporal_example(model, atomic, result, "Always[0,2](altitude >= 50 m)")


def run_eventually_example():
    model = eventually_altitude_example()
    predicate = Predicate("altitude >= 55 m")
    source = OfflineSource({predicate: model.probability_bounds})

    atomic = predicate(source)
    result = Eventually(predicate, (0, 2))(source)

    p = atomic[0, :, 0]
    lower = p.amax()
    upper = torch.clamp(p.sum(), max=1.0)
    expected = torch.stack([lower, upper]).reshape(1, 1, 2)
    torch.testing.assert_close(result, expected)

    print(f"p0, p1, p2 = {p.tolist()}")
    print(f"Eventually[0,2](altitude >= 55 m) = {result[0, 0].tolist()}"
          f"   (expected [{lower.item():.4f}, {upper.item():.4f}])")
    print("The belief mean crosses 55 m at t=2, but this does not make the")
    print("event certain. The uncertain altitude can still be below 55 m.")

    plot_temporal_example(model, atomic, result, "Eventually[0,2](altitude >= 55 m)")


def main():
    with skip_run("run", "Boolean operators") as check, check():
        run_boolean_example()

    with skip_run("skip", "Always above 50 m") as check, check():
        run_always_example()

    with skip_run("skip", "Eventually above 55 m") as check, check():
        run_eventually_example()


if __name__ == "__main__":
    main()
