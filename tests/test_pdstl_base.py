"""Validation of the user input contract, and the source query behaviour.

Covers requirement groups A (input validation) and B (predicate/source).
"""

import pytest
import torch

from pdstl import (
    Always,
    Eventually,
    Predicate,
    TableProbabilitySource,
    evaluate,
    validate_bounds,
)

# ---------------------------------------------------------------------------
# A. Input validation
# ---------------------------------------------------------------------------


def test_lower_below_zero_is_rejected():
    with pytest.raises(ValueError, match="lower bound must be >= 0"):
        validate_bounds(torch.tensor([[-0.1, 0.8]]))


def test_upper_above_one_is_rejected():
    with pytest.raises(ValueError, match="upper bound must be <= 1"):
        validate_bounds(torch.tensor([[0.2, 1.3]]))


def test_lower_above_upper_is_rejected():
    with pytest.raises(ValueError, match="lower bound must be <= upper bound"):
        validate_bounds(torch.tensor([[0.8, 0.2]]))


def test_validation_is_strict_with_no_tolerance():
    """A near-miss is still a miss.

    Admitting ``[-5e-7, 0.8]`` would let the core carry an invalid interval and
    make the ``0 <= l <= u <= 1`` invariant ambiguous downstream. Sources clamp
    their own numerics; the boundary check does not forgive.
    """
    with pytest.raises(ValueError, match="lower bound must be >= 0"):
        validate_bounds(torch.tensor([[-5e-7, 0.8]], dtype=torch.float64))
    with pytest.raises(ValueError, match="upper bound must be <= 1"):
        validate_bounds(torch.tensor([[0.2, 1.0 + 5e-7]], dtype=torch.float64))


def test_malformed_bound_shapes_are_rejected():
    with pytest.raises(ValueError, match="trailing dimension 2"):
        validate_bounds(torch.tensor([[0.1, 0.5, 0.9]]))
    with pytest.raises(ValueError, match=r"shape \[B, 2\]"):
        validate_bounds(torch.tensor([0.1, 0.5]))
    with pytest.raises(ValueError, match=r"shape \[B, 2\]"):
        validate_bounds(torch.tensor(0.5))
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        validate_bounds([[0.1, 0.5]])


def test_non_finite_bounds_are_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        validate_bounds(torch.tensor([[float("nan"), 0.5]]))
    with pytest.raises(ValueError, match="non-finite"):
        validate_bounds(torch.tensor([[0.1, float("inf")]]))


def test_table_source_validates_eagerly_on_insertion():
    mu = Predicate(name="mu")
    with pytest.raises(ValueError, match="lower bound must be <= upper bound"):
        TableProbabilitySource({(mu, 0): (0.9, 0.1)})
    with pytest.raises(ValueError, match="exactly 2 entries"):
        TableProbabilitySource({(mu, 0): (0.1, 0.5, 0.9)})


def test_malformed_temporal_intervals_are_rejected():
    mu = Predicate(name="mu")
    with pytest.raises(ValueError, match=r"0 <= a <= b"):
        Always(mu, interval=[2, 1])
    with pytest.raises(ValueError, match=r"0 <= a <= b"):
        Eventually(mu, interval=[-1, 3])
    with pytest.raises(ValueError, match="exactly 2 endpoints"):
        Always(mu, interval=[0, 1, 2])
    with pytest.raises(ValueError, match="must be integral"):
        Always(mu, interval=[0, 1.5])


def test_unbounded_intervals_are_not_representable():
    """This branch is bounded-time only: no ``None``, no infinity."""
    mu = Predicate(name="mu")
    with pytest.raises(TypeError):
        Always(mu)
    with pytest.raises(TypeError, match="no unbounded default"):
        Always(mu, interval=None)
    with pytest.raises(ValueError, match="must be finite"):
        Always(mu, interval=[0, float("inf")])


# ---------------------------------------------------------------------------
# B. Predicate / source
# ---------------------------------------------------------------------------


class CountingSource(TableProbabilitySource):
    """Table source that records every ``bounds`` call it receives."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queries = []

    def bounds(self, predicate, time):
        self.queries.append((predicate.uid, time))
        return super().bounds(predicate, time)


def test_atom_returns_exactly_the_supplied_bounds():
    mu = Predicate(name="mu")
    source = TableProbabilitySource({(mu, 0): (0.25, 0.75), (mu, 1): (0.4, 0.6)})

    trace = evaluate(mu, source)

    assert trace.shape == (1, 2, 2)
    assert trace[0, 0].tolist() == pytest.approx([0.25, 0.75])
    assert trace[0, 1].tolist() == pytest.approx([0.4, 0.6])


def test_repeated_predicate_time_event_is_queried_once():
    mu = Predicate(name="mu")
    source = CountingSource({(mu, 0): (0.6, 0.9)}, horizon=0)

    # mu appears four times, all at time 0.
    phi = (mu & mu) | (mu & mu)
    evaluate(phi, source)

    assert source.queries == [(mu.uid, 0)]


def test_negation_does_not_trigger_a_second_query():
    """``~A`` is derived from the cached positive atom, never re-queried."""
    mu = Predicate(name="mu")
    source = CountingSource({(mu, 0): (0.4, 0.7)}, horizon=0)

    evaluate(mu | ~mu, source)

    assert source.queries == [(mu.uid, 0)]


def test_each_predicate_time_pair_is_queried_exactly_once():
    mu = Predicate(name="mu")
    source = CountingSource({(mu, k): (0.9, 0.95) for k in range(3)})

    # G[0,2] mu and F[0,2] mu read the same three atoms.
    evaluate(Always(mu, [0, 2]) & Eventually(mu, [0, 2]), source)

    assert sorted(source.queries) == [(mu.uid, 0), (mu.uid, 1), (mu.uid, 2)]


def test_separately_created_predicates_are_distinct_events():
    """Identical name and callable do not make two predicates the same event."""
    mu1 = Predicate(lambda x: x[..., 0], name="same")
    mu2 = Predicate(lambda x: x[..., 0], name="same")

    assert mu1.uid != mu2.uid

    source = TableProbabilitySource(
        {(mu1, 0): (0.6, 0.9), (mu2, 0): (0.6, 0.9)}, horizon=0
    )

    # mu1 & mu1 collapses by identity; mu1 & mu2 must not.
    assert evaluate(mu1 & mu1, source)[0, 0].tolist() == pytest.approx([0.6, 0.9])
    assert evaluate(mu1 & mu2, source)[0, 0].tolist() == pytest.approx([0.2, 0.9])


def test_missing_table_entry_is_reported_not_fabricated():
    mu = Predicate(name="mu")
    source = TableProbabilitySource({(mu, 0): (0.5, 0.5)}, horizon=1)

    with pytest.raises(KeyError, match="no probability bounds recorded"):
        evaluate(mu, source)


def test_inconsistent_batch_size_is_reported_clearly():
    class RaggedSource(TableProbabilitySource):
        def bounds(self, predicate, time):
            single = super().bounds(predicate, time)
            return single if time == 0 else single.repeat(3, 1)

    mu = Predicate(name="mu")
    source = RaggedSource({(mu, 0): (0.5, 0.5), (mu, 1): (0.5, 0.5)})

    with pytest.raises(ValueError, match="inconsistent batch size"):
        evaluate(Always(mu, [0, 1]), source)


def test_batched_source_is_supported():
    class BatchedSource(TableProbabilitySource):
        def bounds(self, predicate, time):
            return super().bounds(predicate, time).repeat(4, 1)

    mu = Predicate(name="mu")
    source = BatchedSource({(mu, k): (0.9, 0.9) for k in range(2)})

    trace = evaluate(Always(mu, [0, 1]), source)

    assert trace.shape == (4, 1, 2)
    assert trace[:, 0, 0].tolist() == pytest.approx([0.8] * 4)
