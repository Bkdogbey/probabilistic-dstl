"""Verification for the pdSTL input contract and atomic predicates.

Covers input handling, event selection, probability intervals and gradient
preservation -- the four things the new Belief/Predicate split is responsible
for. Boolean and temporal equations are exercised only as pass-through, since
this change does not touch them.
"""

import numpy as np
import pytest
import torch
from scipy.stats import norm

from models.dynamics import GaussianBelief, piecewise_signal
from pdstl.base import (
    BeliefTrajectory,
    OnlineBeliefTrajectory,
    ProbabilityBelief,
    check_probability_bounds,
)
from pdstl.operators import Always, Eventually, GreaterThan, LessThan, Predicate
from utils import create_belief_trajectory


def gaussian_traj(mean, var, k=1.0):
    """BeliefTrajectory of [B, D] Gaussian steps from [T, D] arrays."""
    mean = torch.as_tensor(mean, dtype=torch.float32)
    var = torch.as_tensor(var, dtype=torch.float32)
    return BeliefTrajectory(
        [
            GaussianBelief(mean[t : t + 1], var[t : t + 1], confidence_level=k)
            for t in range(mean.shape[0])
        ]
    )


# ---------------------------------------------------------------------------
# Input handling: shapes and the per-element convention
# ---------------------------------------------------------------------------


def test_atom_returns_batch_time_bounds():
    _, mean, var = piecewise_signal()
    traj = create_belief_trajectory(mean, var)

    trace = GreaterThan(50.0)(traj)

    assert trace.shape == (1, len(mean), 2)


def test_multidimensional_state_selects_the_named_component():
    """D > 1 with an explicit dim. This is what crashed before: the atom had no
    component index and leaked a [B, T, D, 2] tensor into the temporal cells."""
    mean = np.array([[0.0, 10.0, -5.0]] * 4)  # T=4, D=3
    var = np.ones((4, 3))
    traj = gaussian_traj(mean, var, k=0.0)

    on_dim1 = GreaterThan(9.0, dim=1)(traj)
    on_dim2 = GreaterThan(9.0, dim=2)(traj)

    assert on_dim1.shape == (1, 4, 2)
    assert on_dim2.shape == (1, 4, 2)
    # dim 1 sits above the threshold, dim 2 far below: different components,
    # so the index is honoured rather than silently collapsed.
    assert on_dim1[0, 0, 0] > 0.8
    assert on_dim2[0, 0, 0] < 0.2


def test_multidimensional_state_composes_into_a_temporal_operator():
    mean = np.array([[0.0, 10.0, -5.0]] * 4)
    var = np.ones((4, 3))
    traj = gaussian_traj(mean, var, k=0.0)

    trace = Always(GreaterThan(9.0, dim=1), interval=[0, 2])(traj)

    assert trace.shape == (1, 4, 2)
    assert torch.isfinite(trace).all()


def test_trajectory_indexing_length_suffix_and_append():
    _, mean, var = piecewise_signal()
    traj = create_belief_trajectory(mean, var)

    assert len(traj) == len(mean)
    assert traj[0] is traj.beliefs[0]

    tail = traj.suffix(2)
    assert isinstance(tail, BeliefTrajectory)
    assert len(tail) == len(mean) - 2

    online = OnlineBeliefTrajectory()
    for belief in traj:
        online.append(belief)
    assert len(online) == len(traj)
    # suffix keeps the streaming container's own type
    assert isinstance(online.suffix(1), OnlineBeliefTrajectory)
    assert GreaterThan(50.0)(online).shape == (1, len(mean), 2)


# ---------------------------------------------------------------------------
# Probability intervals
# ---------------------------------------------------------------------------


def test_interval_is_ordered_and_brackets_the_exact_probability():
    mean = np.array([[45.0], [55.0], [50.0]])
    var = np.array([[4.0], [4.0], [9.0]])
    traj = gaussian_traj(mean, var, k=1.0)

    trace = GreaterThan(50.0)(traj)[0].detach().numpy()
    exact = 1.0 - norm.cdf((50.0 - mean[:, 0]) / np.sqrt(var[:, 0]))

    assert np.all(trace[:, 0] <= trace[:, 1])
    assert np.all((trace >= 0.0) & (trace <= 1.0))
    assert np.all(trace[:, 0] <= exact + 1e-6)
    assert np.all(exact <= trace[:, 1] + 1e-6)


def test_zero_ambiguity_gives_the_exact_probability_as_a_singleton():
    mean = np.array([[45.0], [55.0]])
    var = np.array([[4.0], [4.0]])
    traj = gaussian_traj(mean, var, k=0.0)

    trace = GreaterThan(50.0)(traj)[0].detach().numpy()
    exact = 1.0 - norm.cdf((50.0 - mean[:, 0]) / np.sqrt(var[:, 0]))

    np.testing.assert_allclose(trace[:, 0], trace[:, 1], atol=1e-7)
    np.testing.assert_allclose(trace[:, 0], exact, atol=1e-6)


def test_less_than_is_the_complement_of_greater_than():
    mean = np.array([[45.0], [55.0], [50.0]])
    var = np.array([[4.0], [4.0], [9.0]])
    traj = gaussian_traj(mean, var, k=1.5)

    gt = GreaterThan(50.0)(traj)[0].detach().numpy()
    lt = LessThan(50.0)(traj)[0].detach().numpy()

    # P(x <= c) endpoints are the swapped complements of P(x >= c)
    np.testing.assert_allclose(lt[:, 0], 1.0 - gt[:, 1], atol=1e-6)
    np.testing.assert_allclose(lt[:, 1], 1.0 - gt[:, 0], atol=1e-6)


def test_wider_ambiguity_gives_a_wider_interval_around_the_same_exact_value():
    mean = np.array([[48.0]])
    var = np.array([[4.0]])

    narrow = GreaterThan(50.0)(gaussian_traj(mean, var, k=0.5))[0, 0]
    wide = GreaterThan(50.0)(gaussian_traj(mean, var, k=2.0))[0, 0]

    assert wide[0] < narrow[0]
    assert wide[1] > narrow[1]


# ---------------------------------------------------------------------------
# Event selection
# ---------------------------------------------------------------------------


def test_supplied_intervals_are_served_by_event_name():
    step = ProbabilityBelief(
        {
            "safe": torch.tensor([[0.90, 0.97]]),
            "reach": torch.tensor([[0.10, 0.42]]),
        }
    )
    traj = BeliefTrajectory([step] * 5)

    safe = Always(Predicate("safe"), interval=[0, 3])(traj)
    reach = Eventually(Predicate("reach"), interval=[0, 3])(traj)

    assert safe.shape == (1, 5, 2)
    np.testing.assert_allclose(safe[0, 0].detach().numpy(), [0.90, 0.97], atol=1e-6)
    np.testing.assert_allclose(reach[0, 0].detach().numpy(), [0.10, 0.42], atol=1e-6)


def test_unknown_event_raises_instead_of_returning_the_wrong_bounds():
    traj = BeliefTrajectory([ProbabilityBelief({"safe": torch.tensor([[0.9, 0.97]])})])

    with pytest.raises(ValueError, match="no bounds for event 'reach'"):
        Predicate("reach")(traj)

    # A comparison predicate names itself, so it cannot collide with "safe"
    with pytest.raises(ValueError, match="no bounds for event"):
        GreaterThan(50.0)(traj)


def test_gaussian_belief_rejects_an_event_it_cannot_evaluate():
    traj = gaussian_traj(np.array([[50.0]]), np.array([[4.0]]))

    with pytest.raises(ValueError, match="comparison predicates"):
        Predicate("safe")(traj)


def test_differently_thresholded_atoms_are_different_events():
    assert GreaterThan(50.0).name != GreaterThan(60.0).name
    assert GreaterThan(50.0, dim=0).name != GreaterThan(50.0, dim=1).name
    assert GreaterThan(50.0).name != LessThan(50.0).name


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad, match",
    [
        ([[0.8, 0.3]], "lower > upper"),
        ([[float("nan"), 0.5]], "non-finite"),
        ([[-0.2, 0.5]], r"outside \[0, 1\]"),
        ([[0.5, 1.4]], r"outside \[0, 1\]"),
    ],
)
def test_malformed_supplied_bounds_are_rejected(bad, match):
    traj = BeliefTrajectory([ProbabilityBelief({"safe": torch.tensor(bad)})])

    with pytest.raises(ValueError, match=match):
        Predicate("safe")(traj)


def test_validation_names_the_offending_step():
    good = torch.tensor([[0.4, 0.6]])
    bad = torch.tensor([[0.8, 0.3]])
    traj = BeliefTrajectory(
        [ProbabilityBelief({"safe": b}) for b in (good, good, bad, good)]
    )

    with pytest.raises(ValueError, match="at step 2"):
        Predicate("safe")(traj)


def test_validation_can_be_disabled_in_the_optimiser_loop():
    traj = BeliefTrajectory([ProbabilityBelief({"safe": torch.tensor([[0.8, 0.3]])})])

    trace = Predicate("safe")(traj, validate=False)

    np.testing.assert_allclose(trace[0, 0].detach().numpy(), [0.8, 0.3], atol=1e-9)


def test_validation_tolerates_cdf_round_off():
    # Endpoints a hair outside [0, 1] are round-off, not malformed input.
    check_probability_bounds(torch.tensor([[[-1e-9, 1.0 + 1e-9]]]))


def test_bad_supplied_shape_is_rejected_at_construction():
    with pytest.raises(ValueError, match=r"\[batch, 2\]"):
        ProbabilityBelief({"safe": torch.tensor([0.9, 0.97])})


def test_mismatched_mean_and_var_shapes_raise_instead_of_going_non_finite():
    # [B, D] and [D, D] have the same rank when B == D, so this must be caught
    # rather than broadcast into a zero variance.
    traj = BeliefTrajectory(
        [GaussianBelief(torch.tensor([[1.0, 2.0]]), torch.eye(2) * 0.25)]
    )

    with pytest.raises(ValueError, match="disagree"):
        GreaterThan(1.0)(traj)


# ---------------------------------------------------------------------------
# dtype, device and gradient preservation
# ---------------------------------------------------------------------------


def test_gradients_flow_from_a_temporal_formula_back_to_the_mean():
    mean = torch.tensor([[45.0], [55.0], [60.0], [48.0]], requires_grad=True)
    var = torch.full((4, 1), 4.0)
    traj = BeliefTrajectory(
        [
            GaussianBelief(mean[t : t + 1], var[t : t + 1], confidence_level=1.0)
            for t in range(4)
        ]
    )

    Always(GreaterThan(50.0), interval=[0, 2])(traj)[0, 0, 0].backward()

    assert mean.grad is not None
    assert torch.isfinite(mean.grad).all()
    assert mean.grad.abs().sum() > 0


def test_supplied_intervals_keep_their_graph():
    p = torch.tensor([[0.3, 0.7]], requires_grad=True)
    traj = BeliefTrajectory([ProbabilityBelief({"safe": p})] * 3)

    Always(Predicate("safe"), interval=[0, 2])(traj)[0, 0, 0].backward()

    assert p.grad is not None
    assert p.grad.abs().sum() > 0


def test_dtype_is_preserved_rather_than_forced_to_float32():
    mean = torch.tensor([[45.0], [55.0]], dtype=torch.float64)
    var = torch.full((2, 1), 4.0, dtype=torch.float64)
    traj = BeliefTrajectory(
        [GaussianBelief(mean[t : t + 1], var[t : t + 1]) for t in range(2)]
    )

    trace = GreaterThan(50.0)(traj)

    assert trace.dtype == torch.float64


def test_create_belief_trajectory_honours_dtype_and_element_shape():
    _, mean, var = piecewise_signal()
    traj = create_belief_trajectory(mean, var, dtype=torch.float64)

    assert traj[0].value().shape == (1, 1)  # [B, D], one prediction step
    assert GreaterThan(50.0)(traj).dtype == torch.float64
