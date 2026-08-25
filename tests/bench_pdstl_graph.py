"""Baseline timing: reference interpreter vs. compiled hard-probability graph.

Diagnostic only -- not collected by pytest (no ``test_`` prefix) and asserts
nothing about speed. This phase's acceptance criterion is exact equivalence,
not performance; this script exists only to record a starting point before
any future vectorization/masking work on the temporal kernels.

Run directly:

    python tests/bench_pdstl_graph.py
"""

import time

from pdstl import Always, And, Eventually, Predicate, TableProbabilitySource, Until, evaluate
from pdstl.graph import compile_formula, materialize_atom_traces


def _random_wide_source(preds, horizon):
    table = {}
    for i, p in enumerate(preds):
        for t in range(horizon + 1):
            table[(p, t)] = (0.4, 0.9)
    return TableProbabilitySource(table, horizon=horizon)


def _time_reference(formula, source, repeats):
    start = time.perf_counter()
    for _ in range(repeats):
        evaluate(formula, source)
    return (time.perf_counter() - start) / repeats


def _time_compiled(formula, source, horizon, repeats, include_compile):
    if include_compile:
        start = time.perf_counter()
        for _ in range(repeats):
            compiled = compile_formula(formula, horizon=horizon)
            traces = materialize_atom_traces(formula, source, horizon)
            compiled(traces)
        return (time.perf_counter() - start) / repeats

    compiled = compile_formula(formula, horizon=horizon)
    traces = materialize_atom_traces(formula, source, horizon)
    start = time.perf_counter()
    for _ in range(repeats):
        compiled(traces)
    return (time.perf_counter() - start) / repeats


def _report(label, formula, preds, horizon, repeats=5):
    source = _random_wide_source(preds, horizon)

    ref_time = _time_reference(formula, source, repeats)
    compiled_incl = _time_compiled(formula, source, horizon, repeats, include_compile=True)
    compiled_amortized = _time_compiled(formula, source, horizon, repeats, include_compile=False)

    print(
        f"{label:40s} N={horizon:4d}  "
        f"reference={ref_time * 1e3:8.3f}ms  "
        f"compiled(+compile)={compiled_incl * 1e3:8.3f}ms  "
        f"compiled(amortized)={compiled_amortized * 1e3:8.3f}ms"
    )


def main() -> None:
    a, b = Predicate(name="A"), Predicate(name="B")
    preds = [a, b]

    print("--- Always/Eventually, fixed window width, growing trace length T ---")
    for horizon in (50, 200, 800):
        _report("G[0,5](A & B)", Always(And(a, b), [0, 5]), preds, horizon)

    print("\n--- Always, fixed trace length, growing window width W ---")
    for width in (2, 8, 32):
        horizon = 200
        _report(f"G[0,{width}](A)", Always(a, [0, width]), preds, horizon)

    print("\n--- Until, fixed trace length, growing window width W ---")
    for width in (2, 8, 32):
        horizon = 200
        _report(f"A U[0,{width}] B", Until(a, b, [0, width]), preds, horizon)

    print("\n--- Nested formula, growing trace length T ---")
    for horizon in (50, 200, 800):
        nested = Eventually(And(a, Always(b, [1, 3])), [0, 5])
        _report("F[0,5](A & G[1,3]B)", nested, preds, horizon)


if __name__ == "__main__":
    main()
