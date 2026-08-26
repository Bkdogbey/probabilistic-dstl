"""Timing and scaling: reference vs. compiled graph vs. recurrent evaluator.

Diagnostic only -- not collected by pytest (no ``test_`` prefix) and asserts
nothing about speed. The acceptance criterion for the recurrent layer is exact
equivalence with the other two backends plus a genuinely formula-structured
backward-time unfolding, not wall-clock superiority.

What the numbers here are actually good for:

* confirming the recurrent backend's cost grows *linearly* in the trace length
  and in the temporal window width, rather than quadratically;
* confirming ``Until`` does not recompute a prefix per candidate -- the
  reported prefix-update count stays ``T + b - 1`` as the window grows;
* recording the recurrent state size, which depends only on the formula's
  windows and never on the horizon.

Expect the recurrent backend to be *slower* in wall clock than the compiled
graph: it walks time in a Python loop with small tensor operations per step,
while the compiled graph issues one vectorized fold over the whole trace. That
is the honest cost of an explicit temporal unfolding at these problem sizes.

Run directly::

    python tests/bench_pdstl_recurrent.py
"""

import time

from pdstl import (
    Always,
    And,
    Eventually,
    Predicate,
    TableProbabilitySource,
    Until,
    compile_recurrent_formula,
    evaluate,
)
from pdstl.graph import compile_formula, materialize_atom_traces


def _wide_source(preds, horizon):
    table = {(p, t): (0.4, 0.9) for p in preds for t in range(horizon + 1)}
    return TableProbabilitySource(table, horizon=horizon)


def _time(fn, repeats):
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def _report(label, formula, preds, horizon, repeats=3, reference=True):
    """Time all three backends on one formula.

    ``reference=False`` skips the per-time reference interpreter, which is
    quadratic in the Until window and becomes impractically slow well before
    the point where the two vectorized backends part company.
    """
    source = _wide_source(preds, horizon)
    traces = materialize_atom_traces(formula, source, horizon)

    compiled = compile_formula(formula, horizon=horizon)
    recurrent = compile_recurrent_formula(formula, horizon=horizon)

    if reference:
        reference_ms = f"{_time(lambda: evaluate(formula, source), repeats) * 1e3:9.3f}ms"
    else:
        reference_ms = f"{'skipped':>11s}"
    compiled_ms = _time(lambda: compiled(traces), repeats) * 1e3
    recurrent_ms = _time(lambda: recurrent(traces), repeats) * 1e3

    scan_steps = sum(cell.n_state_updates for cell in recurrent.temporal_cells)
    print(
        f"{label:28s} N={horizon:4d}  T={recurrent.valid_length:4d}  "
        f"reference={reference_ms}  compiled={compiled_ms:8.3f}ms  "
        f"recurrent={recurrent_ms:8.3f}ms  "
        f"state={recurrent.recurrent_state_size:3d}  scan_steps={scan_steps:5d}"
    )


def main() -> None:
    a, b = Predicate(name="A"), Predicate(name="B")
    preds = [a, b]

    print("--- Always/Eventually, fixed window width, growing trace length T ---")
    for horizon in (50, 200, 800):
        _report("G[0,5](A & B)", Always(And(a, b), [0, 5]), preds, horizon)
    for horizon in (50, 200, 800):
        _report("F[0,5](A & B)", Eventually(And(a, b), [0, 5]), preds, horizon)

    print("\n--- Always, fixed trace length, growing window width W ---")
    for width in (2, 8, 32):
        _report(f"G[0,{width}](A)", Always(a, [0, width]), preds, 200)

    print("\n--- Eventually, fixed trace length, growing window width W ---")
    for width in (2, 8, 32):
        _report(f"F[0,{width}](A)", Eventually(a, [0, width]), preds, 200)

    print("\n--- Until, growing trace length T (fixed window) ---")
    for horizon in (50, 200, 800):
        _report("A U[0,5] B", Until(a, b, [0, 5]), preds, horizon)

    print("\n--- Until, fixed trace length, growing window width W ---")
    print("    (prefix ladder is advanced once per step: scan_steps ~ T + b)")
    for width in (2, 8, 32):
        _report(f"A U[0,{width}] B", Until(a, b, [0, width]), preds, 200)

    print("\n--- Until with a > 0 (common-prefix upper tightening active) ---")
    for width in (4, 16):
        _report(f"A U[2,{width}] B", Until(a, b, [2, width]), preds, 200)

    print("\n--- Nested formula, growing trace length T ---")
    for horizon in (50, 200, 800):
        nested = Eventually(And(a, Always(b, [1, 3])), [0, 5])
        _report("F[0,5](A & G[1,3]B)", nested, preds, horizon)

    print("\n--- Until window scaling: O(T*W) recurrent vs. O(T*W^2) compiled ---")
    print("    (the compiled graph builds sum_j (j+1) candidate operands; the")
    print("     recurrent ladder reads every prefix out of one shared state)")
    for width in (16, 32, 64, 128, 200):
        _report(f"A U[0,{width}] B", Until(a, b, [0, width]), preds, 400, reference=False)

    print("\n--- Recurrent state size is independent of the horizon ---")
    formula = Until(a, Always(b, [0, 3]), [1, 6])
    for horizon in (20, 100, 500):
        compiled = compile_recurrent_formula(formula, horizon=horizon)
        widths = [cell.state_width for cell in compiled.temporal_cells]
        print(
            f"    A U[1,6] G[0,3]B     N={horizon:4d}  "
            f"T={compiled.valid_length:4d}  cells={compiled.n_cells}  "
            f"temporal_state_widths={widths}"
        )


if __name__ == "__main__":
    main()
