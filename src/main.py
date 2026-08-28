
import torch

from models.drone import drone_altitude_example
from pdstl import Always, And, Eventually, Not, OfflineSource, Or, Predicate
from utils import skip_run
from visualization.temporal import (
    plot_online_window,
    plot_predicates_and_boolean,
    plot_temporal_operator,
)


def _run_online(operator, atomic_bounds):
    """Replay atomic_bounds[:, t, :] through operator.step() one t at a time.

    Returns (incremental, snapshots): incremental is Tensor[B, T-b, 2], the
    same shape/values as operator(source); snapshots is one
    (arrival_time, window_start_index, output_available) tuple per t,
    where output at anchor k = t - b becomes available the instant t
    arrives (window_start_index = max(0, t - b)).
    """
    window_state = None
    outputs = []
    snapshots = []

    for t, current_bounds in enumerate(atomic_bounds.unbind(dim=1)):
        output, window_state = operator.step(current_bounds, window_state)
        snapshots.append((t, max(0, t - operator.b), output is not None))
        if output is not None:
            outputs.append(output)

    return torch.stack(outputs, dim=1), snapshots


def run_pipeline():
    """Build the drone scenario -> predicates -> OfflineSource -> formulas
    -> offline and incremental (step()) results. Returns a dict for both
    main() and the tests.
    """
    model = drone_altitude_example()

    above_50 = Predicate("altitude >= 50 m")
    above_55 = Predicate("altitude >= 55 m")
    source = OfflineSource({above_50: model.bounds_above_50, above_55: model.bounds_above_55})

    bounds_50 = above_50(source)
    bounds_55 = above_55(source)
    not_above_50 = Not(above_50)(source)
    and_bounds = And(above_50, above_55)(source)
    or_bounds = Or(above_50, above_55)(source)

    always_above_50 = Always(above_50, (0, 2))
    eventually_above_55 = Eventually(above_55, (0, 2))
    mission = always_above_50 & eventually_above_55

    always_bounds = always_above_50(source)
    eventually_bounds = eventually_above_55(source)
    mission_bounds = mission(source)
    anchor_time = model.time[: always_bounds.shape[1]]

    always_incremental, always_snapshots = _run_online(always_above_50, bounds_50)
    eventually_incremental, eventually_snapshots = _run_online(eventually_above_55, bounds_55)

    return {
        "model": model,
        "above_50": above_50,
        "above_55": above_55,
        "source": source,
        "bounds_50": bounds_50,
        "bounds_55": bounds_55,
        "not_above_50": not_above_50,
        "and_bounds": and_bounds,
        "or_bounds": or_bounds,
        "always_above_50": always_above_50,
        "eventually_above_55": eventually_above_55,
        "mission": mission,
        "always_bounds": always_bounds,
        "eventually_bounds": eventually_bounds,
        "mission_bounds": mission_bounds,
        "anchor_time": anchor_time,
        "always_incremental": always_incremental,
        "always_snapshots": always_snapshots,
        "eventually_incremental": eventually_incremental,
        "eventually_snapshots": eventually_snapshots,
    }


def main():
    result = run_pipeline()
    model = result["model"]

    with skip_run("skip", "Predicates and Boolean operators") as check, check():
        print(f"  {result['above_50']}: {result['bounds_50'][0, 0].tolist()} -> {result['bounds_50'][0, -1].tolist()}")
        print(f"  {result['above_55']}: {result['bounds_55'][0, 0].tolist()} -> {result['bounds_55'][0, -1].tolist()}")
        print(f"  above_50 & above_55 at t=0: {result['and_bounds'][0, 0].tolist()}"
              "  (Frechet -- not aware that >=55m implies >=50m)")
        plot_predicates_and_boolean(
            model.time, model.altitude_mean, model.altitude_std,
            result["bounds_50"], result["bounds_55"],
            [
                ("~above_50", result["not_above_50"], "tab:green"),
                ("above_50 & above_55", result["and_bounds"], "tab:red"),
                ("above_50 | above_55", result["or_bounds"], "tab:purple"),
            ],
        )

    with skip_run("skip", "Always[0,2](altitude >= 50m)") as check, check():
        print(f"  formula: {result['always_above_50']}   shape: {tuple(result['always_bounds'].shape)}")
        plot_temporal_operator(
            model.time, model.altitude_mean, model.altitude_std, 50,
            result["bounds_50"], result["always_bounds"], result["anchor_time"],
            str(result["always_above_50"]),
        )

    with skip_run("skip", "Eventually[0,2](altitude >= 55m)") as check, check():
        print(f"  formula: {result['eventually_above_55']}   shape: {tuple(result['eventually_bounds'].shape)}")
        plot_temporal_operator(
            model.time, model.altitude_mean, model.altitude_std, 55,
            result["bounds_55"], result["eventually_bounds"], result["anchor_time"],
            str(result["eventually_above_55"]),
        )

    with skip_run("run", "Mission + online step()") as check, check():
        print(f"  formula: {result['mission']}   shape: {tuple(result['mission_bounds'].shape)}")
        torch.testing.assert_close(result["always_incremental"], result["always_bounds"])
        torch.testing.assert_close(result["eventually_incremental"], result["eventually_bounds"])
        print("  ✓ incremental step() outputs equal offline forward() outputs")
        plot_online_window(model.time, result["bounds_50"], result["always_snapshots"][:4])


if __name__ == "__main__":
    main()
