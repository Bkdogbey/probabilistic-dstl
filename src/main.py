"""The pdSTL demonstration entry point.

Run the whole thing with::

    python src/main.py

Four examples, each toggled independently by flipping ``"run"`` / ``"skip"``
in the ``skip_run`` calls in ``main()`` below, building up one drone-altitude
scenario:

1. Predicate       -- a single physically meaningful predicate.
2. Boolean         -- combine two predicates with And / Or / Not.
3. Always          -- a bounded temporal operator alone.
4. Always + Eventually -- combine temporal operators into a mission, verify
   the offline result against hand-computed references and against an
   OnlineSource replay, and plot it.

Verification (the ``_reference_*`` functions and the ``torch.testing.
assert_close`` calls) lives with the example that first needs it: ``Always``
is checked in Example 3, ``Eventually`` and the mission conjunction in
Example 4. All plotting is one reusable call to
``visualization.plot_formula_bounds`` -- this file has no plotting logic of
its own.
"""

import torch

from models.temporal_examples import temporal_probability_traces
from pdstl import Always, Eventually, OfflineSource, OnlineSource, Predicate
from utils import skip_run
from visualization.probability_bounds import (
    COLOR_COMBINED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    plot_formula_bounds,
)

# ---------------------------------------------------------------------------
# Independent verification: plain tensor expressions, not the pdSTL operators
# ---------------------------------------------------------------------------


def _reference_always(bounds, a, b):
    """Hand-computed Frechet intersection over each [a, b] window of bounds[0].

    L(k) = max(0, sum_{j=a..b} lower[k+j] - (b - a)), U(k) = min_{j=a..b} upper[k+j].
    """
    lower, upper = bounds[0, :, 0], bounds[0, :, 1]
    window = b - a + 1
    outputs = [
        torch.stack(
            [
                torch.clamp(lower[k + a : k + b + 1].sum() - (window - 1), min=0.0),
                upper[k + a : k + b + 1].amin(),
            ]
        )
        for k in range(bounds.shape[1] - b)
    ]
    return torch.stack(outputs).unsqueeze(0)


def _reference_eventually(bounds, a, b):
    """Hand-computed Frechet union over each [a, b] window of bounds[0].

    L(k) = max_{j=a..b} lower[k+j], U(k) = min(1, sum_{j=a..b} upper[k+j]).
    """
    lower, upper = bounds[0, :, 0], bounds[0, :, 1]
    outputs = [
        torch.stack(
            [
                lower[k + a : k + b + 1].amax(),
                torch.clamp(upper[k + a : k + b + 1].sum(), max=1.0),
            ]
        )
        for k in range(bounds.shape[1] - b)
    ]
    return torch.stack(outputs).unsqueeze(0)


def _reference_and(left_bounds, right_bounds):
    """Hand-computed Frechet conjunction, pointwise over matching anchors.

    L(k) = max(0, L1(k) + L2(k) - 1), U(k) = min(U1(k), U2(k)).
    """
    l1, u1 = left_bounds[..., 0], left_bounds[..., 1]
    l2, u2 = right_bounds[..., 0], right_bounds[..., 1]
    lower = torch.clamp(l1 + l2 - 1, min=0.0)
    upper = torch.minimum(u1, u2)
    return torch.stack([lower, upper], dim=-1)


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------


def banner(title):
    print(f"\n{title}\n{'-' * len(title)}")


def show(formula, source, note=""):
    """Print the probability enclosure of ``formula`` at time 0."""
    lower, upper = formula(source)[0, 0].tolist()
    suffix = f"    {note}" if note else ""
    print(f"  {formula!s:<45} = [{lower:.3f}, {upper:.3f}]{suffix}")


# ---------------------------------------------------------------------------
# Example 1: Predicate
# ---------------------------------------------------------------------------


def example_predicate():
    """A single predicate: is the drone above the minimum safe altitude?"""
    banner("Example 1: Predicate -- altitude >= 50m")

    altitude_safe = Predicate("altitude >= 50m")
    trace = torch.tensor([[0.90, 0.90], [0.60, 0.85], [0.20, 0.40]]).unsqueeze(0)
    source = OfflineSource({altitude_safe: trace})

    bounds = altitude_safe(source)
    narration = [
        "sensor certain the drone is above 50m",
        "growing uncertainty about the drone's altitude",
        "likely descended below 50m",
    ]

    print(f"  formula: {altitude_safe}")
    print(f"  shape:   {tuple(bounds.shape)}  [B, T, 2]")
    for k in range(bounds.shape[1]):
        lower, upper = bounds[0, k].tolist()
        print(f"    t={k}: P(altitude >= 50m) in [{lower:.2f}, {upper:.2f}]   ({narration[k]})")

    plot_formula_bounds(
        str(altitude_safe),
        [(torch.arange(bounds.shape[1]), bounds, altitude_safe.name, COLOR_PRIMARY)],
    )


# ---------------------------------------------------------------------------
# Example 2: Boolean operators
# ---------------------------------------------------------------------------


def example_boolean():
    """Boolean composition of two physically meaningful predicates."""
    banner("Example 2: Boolean operators -- altitude and battery")

    altitude_safe = Predicate("altitude >= 50m")
    battery_ok = Predicate("battery >= 20%")
    source = OfflineSource(
        {
            altitude_safe: torch.tensor([[0.6, 0.9]]).unsqueeze(0),
            battery_ok: torch.tensor([[0.7, 0.95]]).unsqueeze(0),
        }
    )

    print("  P(altitude >= 50m) = [0.600, 0.900]   P(battery >= 20%) = [0.700, 0.950]")
    show(altitude_safe & battery_ok, source, "safe to continue: max(0, l1+l2-1), min(u1, u2)")
    show(altitude_safe | battery_ok, source, "at least one holds: max(l1, l2), min(1, u1+u2)")
    show(altitude_safe & altitude_safe, source, "repetition identity: A and A = A")
    show(altitude_safe | altitude_safe, source, "repetition identity: A or A = A")

    print("\n  P(battery >= 20%) = [0.700, 0.950]")
    show(~battery_ok, source, "battery low: [l, u] -> [1-u, 1-l]")
    show(battery_ok & ~battery_ok, source, "complement identity (Frechet: [0.000, 0.300])")
    show(battery_ok | ~battery_ok, source, "complement identity (Frechet: [0.700, 1.000])")


# ---------------------------------------------------------------------------
# Example 3: Always (temporal, alone)
# ---------------------------------------------------------------------------


def example_always():
    """Always alone: does the drone maintain safe altitude throughout a window?

    The raw altitude trace dips at t=2; the plot shows how that dip drags
    down every Always window anchor that overlaps it (k=0,1,2), and how the
    bound recovers once the window slides past it (k=3).
    """
    banner("Example 3: Always -- altitude stays safe over a sliding window")

    altitude_safe = Predicate("altitude >= 50m")
    trace = torch.tensor(
        [
            [0.95, 0.98],
            [0.92, 0.97],
            [0.60, 0.75],
            [0.85, 0.93],
            [0.90, 0.96],
            [0.88, 0.95],
        ]
    ).unsqueeze(0)
    source = OfflineSource({altitude_safe: trace})

    always_safe = Always(altitude_safe, (0, 2))
    bounds = always_safe(source)
    torch.testing.assert_close(bounds, _reference_always(trace, 0, 2))

    print(f"  formula: {always_safe}")
    print(f"  shape:   {tuple(bounds.shape)}  [B, T, 2]  (raw trace has {trace.shape[1]} time steps)")
    print("  ✓ matches the hand-computed temporal Frechet intersection")

    plot_formula_bounds(
        str(always_safe),
        [
            (torch.arange(trace.shape[1]), trace, altitude_safe.name, COLOR_PRIMARY),
            (torch.arange(bounds.shape[1]), bounds, str(always_safe), COLOR_COMBINED),
        ],
    )


# ---------------------------------------------------------------------------
# Example 4: Always + Eventually (combined mission)
# ---------------------------------------------------------------------------


def run_examples():
    """Build models -> OfflineSource -> pdSTL formulas -> hard bounds.

    Returns a dict of the predicates, source, formulas, output traces, and
    independent reference traces, for reuse by both example_combined() and
    the tests.
    """
    time, safe_bounds, goal_bounds = temporal_probability_traces()

    safe = Predicate("altitude >= 50m")
    goal = Predicate("reached landing zone")
    source = OfflineSource({safe: safe_bounds, goal: goal_bounds})

    always_safe = Always(safe, (0, 2))
    eventually_goal = Eventually(goal, (0, 2))
    mission = always_safe & eventually_goal

    always_safe_bounds = always_safe(source)
    eventually_goal_bounds = eventually_goal(source)
    mission_bounds = mission(source)

    reference_always = _reference_always(safe_bounds, 0, 2)
    reference_eventually = _reference_eventually(goal_bounds, 0, 2)
    reference_mission = _reference_and(reference_always, reference_eventually)

    return {
        "time": time,
        "output_time": torch.arange(always_safe_bounds.shape[1]),
        "safe": safe,
        "goal": goal,
        "safe_bounds": safe_bounds,
        "goal_bounds": goal_bounds,
        "source": source,
        "always_safe": always_safe,
        "eventually_goal": eventually_goal,
        "mission": mission,
        "always_safe_bounds": always_safe_bounds,
        "eventually_goal_bounds": eventually_goal_bounds,
        "mission_bounds": mission_bounds,
        "reference_always": reference_always,
        "reference_eventually": reference_eventually,
        "reference_mission": reference_mission,
    }


def verify_online(state):
    """Stream safe/goal bounds through OnlineSource one step at a time and
    check every online output equals the matching prefix of the offline
    result, for Always, Eventually, and the mission conjunction.
    """
    safe, goal = state["safe"], state["goal"]
    safe_bounds, goal_bounds = state["safe_bounds"], state["goal_bounds"]
    always_safe, eventually_goal, mission = (
        state["always_safe"],
        state["eventually_goal"],
        state["mission"],
    )

    online = OnlineSource()
    for k in range(safe_bounds.shape[1]):
        online.append({safe: safe_bounds[:, k, :], goal: goal_bounds[:, k, :]})

        for formula, offline_bounds in (
            (always_safe, state["always_safe_bounds"]),
            (eventually_goal, state["eventually_goal_bounds"]),
            (mission, state["mission_bounds"]),
        ):
            online_bounds = formula(online)
            t = online_bounds.shape[1]
            torch.testing.assert_close(online_bounds, offline_bounds[:, :t, :])


def example_combined():
    """Combine Always and Eventually: stay safe, and eventually land."""
    banner("Example 4: Always + Eventually -- stay safe and eventually land")

    state = run_examples()

    torch.testing.assert_close(state["eventually_goal_bounds"], state["reference_eventually"])
    torch.testing.assert_close(state["mission_bounds"], state["reference_mission"])

    print(f"  formula: {state['always_safe']}")
    print(f"  formula: {state['eventually_goal']}")
    print(f"  formula: {state['mission']}")
    print(f"  shape:   {tuple(state['mission_bounds'].shape)}  [B, T, 2]")
    print("  ✓ Eventually matches the hand-computed temporal Frechet union")
    print("  ✓ mission matches the hand-computed Boolean Frechet conjunction")

    verify_online(state)
    print("  ✓ every online prefix matches the offline trace")

    time, output_time = state["time"], state["output_time"]
    plot_formula_bounds(
        str(state["always_safe"]),
        [
            (time, state["safe_bounds"], state["safe"].name, COLOR_PRIMARY),
            (output_time, state["always_safe_bounds"], str(state["always_safe"]), COLOR_COMBINED),
        ],
    )
    plot_formula_bounds(
        str(state["eventually_goal"]),
        [
            (time, state["goal_bounds"], state["goal"].name, COLOR_SECONDARY),
            (output_time, state["eventually_goal_bounds"], str(state["eventually_goal"]), COLOR_COMBINED),
        ],
    )
    plot_formula_bounds(
        str(state["mission"]),
        [
            (output_time, state["always_safe_bounds"], str(state["always_safe"]), COLOR_PRIMARY),
            (output_time, state["eventually_goal_bounds"], str(state["eventually_goal"]), COLOR_SECONDARY),
            (output_time, state["mission_bounds"], str(state["mission"]), COLOR_COMBINED),
        ],
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def main():
    with skip_run("run", "Example 1: Predicate") as check, check():
        example_predicate()

    with skip_run("run", "Example 2: Boolean operators") as check, check():
        example_boolean()

    with skip_run("run", "Example 3: Always (temporal)") as check, check():
        example_always()

    with skip_run("run", "Example 4: Always + Eventually (combined)") as check, check():
        example_combined()


if __name__ == "__main__":
    main()
