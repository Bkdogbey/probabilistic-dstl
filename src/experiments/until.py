"""Offline and streaming bounded strong-Until example."""

from dataclasses import dataclass

import torch

from models.streaming import mission_example
from pdstl import OfflineSource, OnlineSource, Predicate, Until, UntilState
from visualization.until import plot_until_example


@dataclass(frozen=True)
class UntilUpdate:
    """One input arrival, the retained two-branch state, and optional output."""

    arrival: int
    state: UntilState
    output: torch.Tensor | None


def _offline(trace, interval):
    safe = Predicate("safe altitude")
    goal = Predicate("goal reached")
    source = OfflineSource({safe: trace.safe_bounds, goal: trace.goal_bounds})
    formula = Until(safe, goal, interval)
    bounds = formula(source)

    candidates = []
    for anchor in range(bounds.shape[1]):
        candidates.append(
            formula.candidate_bounds(
                trace.safe_bounds[:, anchor : anchor + formula.b, :],
                trace.goal_bounds[:, anchor : anchor + formula.b + 1, :],
            )
        )
    candidate_bounds = torch.stack(candidates, dim=1)
    return safe, goal, formula, bounds, candidate_bounds


def _stream(trace, safe, goal, formula):
    source = OnlineSource()
    state = None
    outputs = []
    updates = []

    for arrival in range(trace.time.shape[0]):
        source.append(
            {
                safe: trace.safe_bounds[:, arrival, :],
                goal: trace.goal_bounds[:, arrival, :],
            }
        )
        output, state = formula.step(
            source.bounds(safe, arrival),
            source.bounds(goal, arrival),
            state,
        )
        updates.append(
            UntilUpdate(
                arrival=arrival,
                state=UntilState(state.left.clone(), state.right.clone()),
                output=None if output is None else output.clone(),
            )
        )
        if output is not None:
            outputs.append(output)

    online_bounds = (
        torch.stack(outputs, dim=1) if outputs else trace.safe_bounds[:, :0, :]
    )
    return online_bounds, updates


def _print_results(formula, candidate_bounds, until_bounds):
    print(f"Until mission = {formula}")
    for anchor in range(until_bounds.shape[1]):
        candidates = "  ".join(
            f"C{offset}=[{pair[0]:.2f}, {pair[1]:.2f}]"
            for offset, pair in zip(
                range(formula.a, formula.b + 1),
                candidate_bounds[0, anchor].tolist(),
            )
        )
        lower, upper = until_bounds[0, anchor].tolist()
        print(f"k={anchor}: {candidates}  Until=[{lower:.2f}, {upper:.2f}]")


def run_until_example(interval=(1, 2), show=True):
    """Evaluate ``safe U[1,2] goal`` offline and one arrival at a time."""
    trace = mission_example()
    safe, goal, formula, offline_bounds, candidates = _offline(trace, interval)
    online_bounds, updates = _stream(trace, safe, goal, formula)
    torch.testing.assert_close(online_bounds, offline_bounds)

    _print_results(formula, candidates, offline_bounds)
    print("Streaming Until outputs match the complete offline trace.")
    figure = plot_until_example(
        trace,
        candidates,
        offline_bounds,
        online_bounds,
        formula,
        show=show,
    )
    return trace, candidates, offline_bounds, online_bounds, updates, figure
