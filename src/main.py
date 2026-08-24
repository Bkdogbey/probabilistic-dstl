"""Foundational pdSTL examples.

Three small demonstrations of the probability-first core, in the spirit of the
original pdSTL examples but with no planning, no dynamics model and no plotting:

1. manually supplied atomic probabilities;
2. Boolean composition;
3. Always / Eventually.

Run with::

    python src/main.py
"""

import sys

from pdstl import (
    Always,
    Eventually,
    Predicate,
    TableProbabilitySource,
    evaluate,
)

# Formula strings use the usual logic symbols, which the default Windows
# console codepage cannot encode.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def show(formula, source, note=""):
    """Print the probability enclosure of ``formula`` at time 0."""
    lower, upper = evaluate(formula, source)[0, 0].tolist()
    suffix = f"    {note}" if note else ""
    print(f"  {formula!s:<34} = [{lower:.3f}, {upper:.3f}]{suffix}")


def banner(title):
    print(f"\n{title}\n{'-' * len(title)}")


# =============================================================================
# EXAMPLE 1: Manually supplied atomic probabilities
# =============================================================================


def example_atomic():
    """A predicate whose event probabilities are simply written down.

    No state model is involved: this is a symbolic predicate, and the source
    supplies P(E) directly.
    """
    banner("Example 1: manually supplied atomic probabilities")

    mu = Predicate(name="mu")
    source = TableProbabilitySource(
        {
            (mu, 0): (0.90, 0.90),  # known exactly
            (mu, 1): (0.60, 0.85),  # only bracketed
            (mu, 2): (0.20, 0.40),
        }
    )

    print(f"  source horizon = {source.horizon} (valid times 0 ... {source.horizon})")
    trace = evaluate(mu, source)
    print(f"  trace shape    = {tuple(trace.shape)}  [B, T_valid, 2]")
    for k in range(trace.shape[1]):
        lower, upper = trace[0, k].tolist()
        print(f"    P(E_mu,{k}) in [{lower:.3f}, {upper:.3f}]")


# =============================================================================
# EXAMPLE 2: Boolean composition
# =============================================================================


def example_boolean():
    """Frechet bounds, plus the two exact structural identities."""
    banner("Example 2: Boolean composition")

    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource(
        {(a, 0): (0.6, 0.9), (b, 0): (0.7, 0.95)}, horizon=0
    )

    print("  A = [0.600, 0.900]   B = [0.700, 0.950]")
    show(a & b, source, "max(0, l1+l2-1), min(u1, u2)")
    show(a | b, source, "max(l1, l2), min(1, u1+u2)")
    show(a & a, source, "repetition identity: A and A = A")
    show(a | a, source, "repetition identity: A or A = A")

    # A separate source so the complement example uses the paper's numbers.
    c = Predicate(name="C")
    complement_source = TableProbabilitySource({(c, 0): (0.4, 0.7)}, horizon=0)

    print("\n  C = [0.400, 0.700]")
    show(~c, complement_source, "[l, u] -> [1-u, 1-l]")
    show(c & ~c, complement_source, "complement identity (Frechet: [0.000, 0.600])")
    show(c | ~c, complement_source, "complement identity (Frechet: [0.400, 1.000])")


# =============================================================================
# EXAMPLE 3: Always / Eventually
# =============================================================================


def example_temporal():
    """Bounded temporal operators, with no assumption of independence."""
    banner("Example 3: Always / Eventually")

    a = Predicate(name="A")
    source = TableProbabilitySource({(a, 0): (0.9, 0.9), (a, 1): (0.9, 0.9)})

    print("  P(A_0) = P(A_1) = 0.900")
    show(Always(a, interval=[0, 1]), source, "sum(l) - (n-1), min(u)")
    show(Eventually(a, interval=[0, 1]), source, "max(l), min(1, sum(u))")
    print("  (an independence assumption would give 0.81, not 0.80)")

    banner("Example 3b: horizon and the valid trace")

    long_source = TableProbabilitySource({(a, k): (0.9, 0.95) for k in range(6)})
    for formula in [a, Always(a, [0, 1]), Always(Eventually(a, [0, 2]), [0, 1])]:
        trace = evaluate(formula, long_source)
        print(
            f"  {formula!s:<34} H={formula.horizon()}  "
            f"trace {tuple(trace.shape)}  "
            f"(= horizon {long_source.horizon} - H + 1)"
        )
    print("  out-of-range tail times are omitted, never padded")


if __name__ == "__main__":
    example_atomic()
    example_boolean()
    example_temporal()
    print()
