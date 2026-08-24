"""Enclosure soundness against explicitly constructed joint distributions.

The other test files check worked examples. This one checks the property the
theory actually claims: for any joint distribution consistent with the supplied
atomic marginals, the computed interval must *contain* the true probability.

Each test builds a real joint distribution over the indicator outcomes, derives
the exact marginals and the exact truth, feeds only the marginals to the source,
and asserts containment.
"""

import itertools
import random

import pytest

from pdstl import (
    Always,
    Eventually,
    Predicate,
    TableProbabilitySource,
    evaluate,
)

SEED = 20260824
N_TRIALS = 200


def random_simplex(rng, n):
    """Uniform-ish sample from the probability simplex of dimension ``n``."""
    weights = [rng.expovariate(1.0) for _ in range(n)]
    total = sum(weights)
    return [w / total for w in weights]


def marginal(joint, outcomes, index):
    """P(indicator ``index`` is 1) under ``joint`` over ``outcomes``."""
    return sum(p for outcome, p in zip(outcomes, joint) if outcome[index] == 1)


def assert_encloses(interval, truth, label):
    lower, upper = interval
    assert lower <= truth + 1e-6, f"{label}: lower {lower} exceeds truth {truth}"
    assert upper >= truth - 1e-6, f"{label}: upper {upper} below truth {truth}"


# ---------------------------------------------------------------------------
# Boolean operators against a 4-outcome joint
# ---------------------------------------------------------------------------


def test_conjunction_and_disjunction_enclose_the_true_probability():
    rng = random.Random(SEED)
    outcomes = list(itertools.product([0, 1], repeat=2))

    a = Predicate(name="A")
    b = Predicate(name="B")

    for _ in range(N_TRIALS):
        joint = random_simplex(rng, 4)

        p_a = marginal(joint, outcomes, 0)
        p_b = marginal(joint, outcomes, 1)
        p_and = sum(p for o, p in zip(outcomes, joint) if o[0] == 1 and o[1] == 1)
        p_or = sum(p for o, p in zip(outcomes, joint) if o[0] == 1 or o[1] == 1)

        source = TableProbabilitySource(
            {(a, 0): (p_a, p_a), (b, 0): (p_b, p_b)}, horizon=0
        )

        assert_encloses(evaluate(a & b, source)[0, 0].tolist(), p_and, "A and B")
        assert_encloses(evaluate(a | b, source)[0, 0].tolist(), p_or, "A or B")


def test_negation_encloses_the_true_complement_probability():
    rng = random.Random(SEED + 1)
    a = Predicate(name="A")

    for _ in range(N_TRIALS):
        p_a = rng.random()
        source = TableProbabilitySource({(a, 0): (p_a, p_a)}, horizon=0)
        assert_encloses(evaluate(~a, source)[0, 0].tolist(), 1.0 - p_a, "not A")


# ---------------------------------------------------------------------------
# Temporal operators against an 8-outcome joint
# ---------------------------------------------------------------------------


def test_always_and_eventually_enclose_the_true_probability():
    rng = random.Random(SEED + 2)
    outcomes = list(itertools.product([0, 1], repeat=3))

    a = Predicate(name="A")

    for _ in range(N_TRIALS):
        joint = random_simplex(rng, 8)

        marginals = [marginal(joint, outcomes, k) for k in range(3)]
        p_all = sum(p for o, p in zip(outcomes, joint) if all(o))
        p_any = sum(p for o, p in zip(outcomes, joint) if any(o))

        source = TableProbabilitySource(
            {(a, k): (m, m) for k, m in enumerate(marginals)}
        )

        assert_encloses(
            evaluate(Always(a, [0, 2]), source)[0, 0].tolist(), p_all, "G[0,2] A"
        )
        assert_encloses(
            evaluate(Eventually(a, [0, 2]), source)[0, 0].tolist(), p_any, "F[0,2] A"
        )


def test_soundness_holds_for_genuinely_wide_marginals():
    """Containment is not an artifact of exact point marginals.

    Each marginal is inflated into a real interval before evaluation. A widened
    input can only widen the result, so the true probability must still land
    inside.
    """
    rng = random.Random(SEED + 3)
    outcomes = list(itertools.product([0, 1], repeat=3))

    a = Predicate(name="A")

    for _ in range(N_TRIALS):
        joint = random_simplex(rng, 8)
        delta = rng.uniform(0.0, 0.25)

        marginals = [marginal(joint, outcomes, k) for k in range(3)]
        p_all = sum(p for o, p in zip(outcomes, joint) if all(o))
        p_any = sum(p for o, p in zip(outcomes, joint) if any(o))

        source = TableProbabilitySource(
            {
                (a, k): (max(0.0, m - delta), min(1.0, m + delta))
                for k, m in enumerate(marginals)
            }
        )

        assert_encloses(
            evaluate(Always(a, [0, 2]), source)[0, 0].tolist(), p_all, "G[0,2] A wide"
        )
        assert_encloses(
            evaluate(Eventually(a, [0, 2]), source)[0, 0].tolist(),
            p_any,
            "F[0,2] A wide",
        )


def test_nested_formula_encloses_the_true_probability():
    """Recursive Frechet composition stays sound, if not sharp.

    ``(A and B) or C`` over a joint on three indicators at one time.
    """
    rng = random.Random(SEED + 4)
    outcomes = list(itertools.product([0, 1], repeat=3))

    a = Predicate(name="A")
    b = Predicate(name="B")
    c = Predicate(name="C")

    for _ in range(N_TRIALS):
        joint = random_simplex(rng, 8)

        marginals = [marginal(joint, outcomes, k) for k in range(3)]
        truth = sum(
            p
            for o, p in zip(outcomes, joint)
            if (o[0] == 1 and o[1] == 1) or o[2] == 1
        )

        source = TableProbabilitySource(
            {
                (predicate, 0): (m, m)
                for predicate, m in zip((a, b, c), marginals)
            },
            horizon=0,
        )

        assert_encloses(
            evaluate((a & b) | c, source)[0, 0].tolist(), truth, "(A and B) or C"
        )


def test_bounds_are_sound_but_not_claimed_sharp():
    """Documents the known looseness of recursive composition.

    ``(A and B) or (A and ~B)`` is exactly ``A``, but the recursive bound does
    not discover that: the two disjuncts are distinct events, so generic
    Frechet applies. This is expected and sound, just not tight.
    """
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource(
        {(a, 0): (0.5, 0.5), (b, 0): (0.5, 0.5)}, horizon=0
    )

    exact = evaluate(a, source)[0, 0].tolist()
    recursive = evaluate((a & b) | (a & ~b), source)[0, 0].tolist()

    assert exact == pytest.approx([0.5, 0.5])
    # Sound: the true value 0.5 is still enclosed.
    assert recursive[0] <= 0.5 <= recursive[1]
    # But strictly wider than the exact answer.
    assert recursive[0] < exact[0] or recursive[1] > exact[1]
