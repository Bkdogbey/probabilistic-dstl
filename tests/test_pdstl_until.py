"""Bounded strong until: ``phi1 U_[a,b] phi2``.

The satisfaction event at time ``k`` is

    E_U = union_{j=a..b} C_j,
    C_j = E_{phi2, k+j} intersect (intersect_{r=0..j-1} E_{phi1, k+r})

with the ``phi1`` prefix always starting at ``r = 0``. Candidates are combined
with the shared finite-intersection Frechet rule and unioned with the shared
finite-union rule; for ``a > 0`` the union's upper bound is additionally
tightened by the common prefix ``P_a``, which every candidate contains.

Covers the analytical worked examples, the horizon algebra, the deterministic
Boolean limit, and soundness against explicit joint distributions.
"""

import itertools
import random

import pytest

from pdstl import (
    Always,
    Eventually,
    Predicate,
    TableProbabilitySource,
    Until,
    evaluate,
)
from pdstl.propagate import EvaluationContext

SEED = 20260824
N_TRIALS = 200


def enclosure(formula, source, time=0):
    """The ``[lower, upper]`` pair of ``formula`` at ``time``, as floats."""
    return evaluate(formula, source)[0, time].tolist()


@pytest.fixture
def worked_example():
    """A0 = [.80, .90], A1 = [.70, .85], B1 = [.80, .90], B2 = [.50, .70]."""
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource(
        {
            (a, 0): (0.80, 0.90),
            (a, 1): (0.70, 0.85),
            (b, 1): (0.80, 0.90),
            (b, 2): (0.50, 0.70),
        }
    )
    return a, b, source


# ---------------------------------------------------------------------------
# Analytical worked examples
# ---------------------------------------------------------------------------


def test_until_11_matches_the_analytical_interval(worked_example):
    """``C_1 = B_1 ∩ A_0``: lower max(0, .80 + .80 - 1) = .60, upper min(.90, .90) = .90."""
    a, b, source = worked_example
    assert enclosure(Until(a, b, [1, 1]), source) == pytest.approx([0.60, 0.90])


def test_until_12_upper_is_tightened_by_the_common_prefix(worked_example):
    """``A U[1,2] B = [0.60, 0.90]``, and the upper is 0.90 rather than 1.00.

    ``C_1 = B_1 ∩ A_0 = [.60, .90]`` and ``C_2 = B_2 ∩ A_0 ∩ A_1``, whose lower
    is max(0, .50 + .80 + .70 - 2) = 0 and whose upper is min(.70, .90, .85) =
    .70. The candidate union alone would report upper min(1, .90 + .70) = 1.00.
    Every candidate contains the common prefix ``P_1 = A_0``, so ``E_U ⊆ A_0``
    and the upper drops back to ``U_{A_0} = 0.90``.
    """
    a, b, source = worked_example

    assert enclosure(Until(a, b, [1, 2]), source) == pytest.approx([0.60, 0.90])
    assert enclosure(Until(a, b, [1, 2]), source) != pytest.approx([0.60, 1.00])


def test_until_12_lower_bound_is_the_candidate_union_lower(worked_example):
    """The prefix tightens only the upper bound; the lower is untouched."""
    a, b, source = worked_example

    lower_11 = enclosure(Until(a, b, [1, 1]), source)[0]
    lower_12 = enclosure(Until(a, b, [1, 2]), source)[0]

    # max over candidates: max(.60, 0) = .60, the same as the single candidate.
    assert lower_12 == pytest.approx(0.60)
    assert lower_12 == pytest.approx(lower_11)


def test_until_22_intersects_the_whole_prefix(worked_example):
    """``C_2 = B_2 ∩ A_0 ∩ A_1``: lower max(0, 2.00 - 2) = 0, upper .70.

    ``P_2 = A_0 ∩ A_1`` has upper .85, which does not bite here because the
    candidate's own upper is already the smaller .70.
    """
    a, b, source = worked_example
    assert enclosure(Until(a, b, [2, 2]), source) == pytest.approx([0.0, 0.70])


def test_until_00_is_the_right_operand():
    """``phi1 U[0,0] phi2 ≡ phi2``, numerically and by event key."""
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource(
        {(a, 0): (0.50, 0.50), (b, 0): (0.30, 0.60)}, horizon=0
    )

    assert enclosure(Until(a, b, [0, 0]), source) == pytest.approx([0.30, 0.60])

    context = EvaluationContext(source)
    _, until_key = context._evaluate_with_key(Until(a, b, [0, 0]), 0)
    _, right_key = context._evaluate_with_key(b, 0)
    assert until_key == right_key


def test_until_00_identity_survives_into_surrounding_structure():
    """Because the key is preserved, ``(A U[0,0] B) ∧ B`` is just ``B``."""
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource(
        {(a, 0): (0.50, 0.50), (b, 0): (0.30, 0.60)}, horizon=0
    )

    assert enclosure(Until(a, b, [0, 0]) & b, source) == pytest.approx([0.30, 0.60])
    assert enclosure(Until(a, b, [0, 0]) & ~b, source) == pytest.approx([0.0, 0.0])


def test_until_prefix_starts_at_zero_not_at_a():
    """With ``A`` impossible at time 0, ``A U[1,1] B`` is impossible.

    The prefix for ``j = 1`` is ``{A_0}``. An implementation that started the
    prefix at ``r = a`` would leave it empty here and report ``B_1``'s own
    interval ``[0.9, 0.9]``.
    """
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource({(a, 0): (0.0, 0.0), (b, 1): (0.9, 0.9)})

    assert enclosure(Until(a, b, [1, 1]), source) == pytest.approx([0.0, 0.0])
    assert enclosure(Until(a, b, [1, 1]), source) != pytest.approx([0.9, 0.9])


def test_until_with_a_zero_needs_no_prefix_tightening():
    """``C_0 = B_0`` has no prefix, so the union upper stands as computed.

    ``C_0 = [.10, .20]``, ``C_1 = B_1 ∩ A_0 = [max(0, .30 + .50 - 1), min(.40,
    .60)] = [0, .40]``, union ``[max(.10, 0), min(1, .20 + .40)] = [.10, .60]``.
    """
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource(
        {
            (a, 0): (0.50, 0.60),
            (b, 0): (0.10, 0.20),
            (b, 1): (0.30, 0.40),
        }
    )

    assert enclosure(Until(a, b, [0, 1]), source) == pytest.approx([0.10, 0.60])


# ---------------------------------------------------------------------------
# Horizon
# ---------------------------------------------------------------------------


def test_until_horizon_uses_the_actual_required_lookahead():
    a = Predicate(name="A")
    b = Predicate(name="B")

    # b > 0: max(b + H(phi2), b - 1 + H(phi1)).
    assert Until(a, b, [0, 2]).horizon() == 2
    assert Until(a, b, [1, 2]).horizon() == 2
    assert Until(Always(a, [0, 3]), b, [1, 2]).horizon() == 4
    assert Until(Always(a, [0, 2]), Eventually(b, [0, 1]), [3, 4]).horizon() == 5

    # The looser b + max(H1, H2) would have claimed 5 and 6 respectively.
    assert Until(Always(a, [0, 3]), b, [1, 2]).horizon() != 5
    assert Until(Always(a, [0, 2]), Eventually(b, [0, 1]), [3, 4]).horizon() != 6


def test_until_horizon_with_b_zero_ignores_the_left_lookahead():
    """``phi1`` is never read when ``b = 0``, so only ``H(phi2)`` matters."""
    a = Predicate(name="A")
    b = Predicate(name="B")

    assert Until(a, b, [0, 0]).horizon() == 0
    assert Until(Always(a, [0, 5]), b, [0, 0]).horizon() == 0
    assert Until(Always(a, [0, 5]), Eventually(b, [0, 2]), [0, 0]).horizon() == 2


def test_until_valid_trace_length_matches_the_horizon():
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource(
        {(p, k): (0.5, 0.7) for p in (a, b) for k in range(6)}
    )

    for formula in [
        Until(a, b, [0, 0]),
        Until(a, b, [1, 2]),
        Until(a, b, [0, 3]),
        Until(Always(a, [0, 1]), b, [1, 2]),
    ]:
        trace = evaluate(formula, source)
        expected = source.horizon - formula.horizon() + 1
        assert trace.shape == (1, expected, 2), f"{formula} produced {trace.shape}"


def test_until_source_is_queried_only_where_needed():
    """``phi1`` at offsets 0..b-1 and ``phi2`` at offsets a..b, nothing else."""
    queried = []

    class RecordingSource(TableProbabilitySource):
        def bounds(self, predicate, time):
            queried.append((str(predicate), time))
            return super().bounds(predicate, time)

    a = Predicate(name="A")
    b = Predicate(name="B")
    source = RecordingSource(
        {(p, k): (0.5, 0.7) for p in (a, b) for k in range(3)}, horizon=2
    )

    evaluate(Until(a, b, [1, 2]), source)

    assert sorted(set(queried)) == [("A", 0), ("A", 1), ("B", 1), ("B", 2)]


def test_until_source_too_short_is_an_error():
    a = Predicate(name="A")
    b = Predicate(name="B")
    source = TableProbabilitySource(
        {(p, k): (0.5, 0.7) for p in (a, b) for k in range(2)}
    )

    with pytest.raises(ValueError, match="too short"):
        evaluate(Until(a, b, [0, 4]), source)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_until_validates_its_operands():
    a = Predicate(name="A")

    with pytest.raises(TypeError, match="right operand must be an STLFormula"):
        Until(a, "B", [0, 1])
    with pytest.raises(TypeError, match="left operand must be an STLFormula"):
        Until("A", a, [0, 1])


def test_until_validates_its_interval():
    a = Predicate(name="A")
    b = Predicate(name="B")

    with pytest.raises(ValueError, match=r"0 <= a <= b"):
        Until(a, b, [2, 1])
    with pytest.raises(ValueError, match=r"0 <= a <= b"):
        Until(a, b, [-1, 3])
    with pytest.raises(ValueError, match="exactly 2 endpoints"):
        Until(a, b, [0, 1, 2])
    with pytest.raises(ValueError, match="must be integral"):
        Until(a, b, [0, 1.5])
    with pytest.raises(ValueError, match="must be finite"):
        Until(a, b, [0, float("inf")])
    with pytest.raises(TypeError, match="no unbounded default"):
        Until(a, b, None)
    with pytest.raises(TypeError):
        Until(a, b)


# ---------------------------------------------------------------------------
# Deterministic Boolean limit
# ---------------------------------------------------------------------------


def boolean_stl(formula, signal, time):
    """Reference ordinary (non-probabilistic) discrete-time STL, with until.

    ``signal`` maps ``predicate.uid -> list of 0/1 truth values``. Written
    directly from the textbook semantics, independently of the implementation
    under test.
    """
    from pdstl.operators import And, Negation, Or, TemporalOperator
    from pdstl.operators import Predicate as P

    if isinstance(formula, P):
        return bool(signal[formula.uid][time])
    if isinstance(formula, Negation):
        return not boolean_stl(formula.subformula, signal, time)
    if isinstance(formula, And):
        return boolean_stl(formula.left, signal, time) and boolean_stl(
            formula.right, signal, time
        )
    if isinstance(formula, Or):
        return boolean_stl(formula.left, signal, time) or boolean_stl(
            formula.right, signal, time
        )
    if isinstance(formula, Until):
        return any(
            boolean_stl(formula.right, signal, time + j)
            and all(
                boolean_stl(formula.left, signal, time + r) for r in range(j)
            )
            for j in range(formula.a, formula.b + 1)
        )
    if isinstance(formula, TemporalOperator):
        window = range(time + formula.a, time + formula.b + 1)
        results = [boolean_stl(formula.subformula, signal, k) for k in window]
        return all(results) if isinstance(formula, Always) else any(results)
    raise TypeError(type(formula).__name__)


def test_until_with_zero_one_probabilities_reproduces_boolean_stl():
    rng = random.Random(SEED)
    horizon = 6

    a = Predicate(name="A")
    b = Predicate(name="B")

    formulas = [
        Until(a, b, [0, 0]),
        Until(a, b, [0, 1]),
        Until(a, b, [1, 1]),
        Until(a, b, [1, 3]),
        Until(a, b, [2, 3]),
        Until(a, a, [0, 2]),
        Until(~a, b, [0, 2]),
        Until(a & b, b, [1, 2]),
        Until(a, Always(b, [0, 1]), [1, 2]),
        Until(Eventually(a, [0, 1]), b, [1, 2]),
        Always(Until(a, b, [0, 1]), [0, 1]),
        Until(a, b, [0, 2]) | Always(a, [0, 1]),
    ]

    for _ in range(40):
        signal = {
            a.uid: [rng.randint(0, 1) for _ in range(horizon + 1)],
            b.uid: [rng.randint(0, 1) for _ in range(horizon + 1)],
        }
        table = {}
        for predicate in (a, b):
            for k in range(horizon + 1):
                truth = float(signal[predicate.uid][k])
                table[(predicate, k)] = (truth, truth)
        source = TableProbabilitySource(table, horizon=horizon)

        for formula in formulas:
            trace = evaluate(formula, source)
            for time in range(trace.shape[1]):
                lower, upper = trace[0, time].tolist()
                expected = float(boolean_stl(formula, signal, time))

                assert lower == pytest.approx(upper), (
                    f"{formula} at t={time} is not degenerate: [{lower}, {upper}]"
                )
                assert lower == pytest.approx(expected), (
                    f"{formula} at t={time}: pdSTL {lower} vs Boolean {expected}"
                )


def test_deterministic_until_covers_both_truth_values():
    """Sanity check that the Boolean comparison is not vacuous."""
    a = Predicate(name="A")
    b = Predicate(name="B")
    seen = set()

    for bits in itertools.product([0.0, 1.0], repeat=4):
        source = TableProbabilitySource(
            {
                (a, 0): (bits[0], bits[0]),
                (a, 1): (bits[1], bits[1]),
                (b, 1): (bits[2], bits[2]),
                (b, 2): (bits[3], bits[3]),
            }
        )
        seen.add(tuple(evaluate(Until(a, b, [1, 2]), source)[0, 0].tolist()))

    assert seen == {(0.0, 0.0), (1.0, 1.0)}


# ---------------------------------------------------------------------------
# Soundness against explicit joint distributions
# ---------------------------------------------------------------------------


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


def test_until_12_encloses_the_true_probability_under_dependence():
    """Indicators ``(A_0, A_1, B_1, B_2)`` with an arbitrary joint law.

    Truth is ``P((B_1 ∧ A_0) ∨ (B_2 ∧ A_0 ∧ A_1))``. Only the four marginals
    reach the evaluator, so the enclosure must hold for *every* dependence
    structure consistent with them -- including the ones that make the prefix
    tightening bite.
    """
    rng = random.Random(SEED + 5)
    outcomes = list(itertools.product([0, 1], repeat=4))

    a = Predicate(name="A")
    b = Predicate(name="B")

    for _ in range(N_TRIALS):
        joint = random_simplex(rng, 16)

        a0, a1, b1, b2 = (marginal(joint, outcomes, i) for i in range(4))
        truth = sum(
            p
            for o, p in zip(outcomes, joint)
            if (o[2] == 1 and o[0] == 1) or (o[3] == 1 and o[0] == 1 and o[1] == 1)
        )

        source = TableProbabilitySource(
            {
                (a, 0): (a0, a0),
                (a, 1): (a1, a1),
                (b, 1): (b1, b1),
                (b, 2): (b2, b2),
            }
        )

        assert_encloses(
            evaluate(Until(a, b, [1, 2]), source)[0, 0].tolist(), truth, "A U[1,2] B"
        )


def test_until_01_encloses_the_true_probability_under_dependence():
    """Indicators ``(A_0, B_0, B_1)``; truth is ``P(B_0 ∨ (B_1 ∧ A_0))``."""
    rng = random.Random(SEED + 6)
    outcomes = list(itertools.product([0, 1], repeat=3))

    a = Predicate(name="A")
    b = Predicate(name="B")

    for _ in range(N_TRIALS):
        joint = random_simplex(rng, 8)

        a0, b0, b1 = (marginal(joint, outcomes, i) for i in range(3))
        truth = sum(
            p
            for o, p in zip(outcomes, joint)
            if o[1] == 1 or (o[2] == 1 and o[0] == 1)
        )

        source = TableProbabilitySource(
            {(a, 0): (a0, a0), (b, 0): (b0, b0), (b, 1): (b1, b1)}
        )

        assert_encloses(
            evaluate(Until(a, b, [0, 1]), source)[0, 0].tolist(), truth, "A U[0,1] B"
        )


def test_until_soundness_holds_for_genuinely_wide_marginals():
    """Containment is not an artifact of exact point marginals."""
    rng = random.Random(SEED + 7)
    outcomes = list(itertools.product([0, 1], repeat=4))

    a = Predicate(name="A")
    b = Predicate(name="B")

    for _ in range(N_TRIALS):
        joint = random_simplex(rng, 16)
        delta = rng.uniform(0.0, 0.25)

        margins = [marginal(joint, outcomes, i) for i in range(4)]
        truth = sum(
            p
            for o, p in zip(outcomes, joint)
            if (o[2] == 1 and o[0] == 1) or (o[3] == 1 and o[0] == 1 and o[1] == 1)
        )

        def widen(m):
            return (max(0.0, m - delta), min(1.0, m + delta))

        source = TableProbabilitySource(
            {
                (a, 0): widen(margins[0]),
                (a, 1): widen(margins[1]),
                (b, 1): widen(margins[2]),
                (b, 2): widen(margins[3]),
            }
        )

        assert_encloses(
            evaluate(Until(a, b, [1, 2]), source)[0, 0].tolist(),
            truth,
            "A U[1,2] B wide",
        )
