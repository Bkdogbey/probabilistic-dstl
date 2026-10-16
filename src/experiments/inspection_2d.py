"""2D inspection: pdSTL vs deterministic STLCG on specs R1-R3.

The map and settings come from configs/scenarios/inspection_2d.yaml. Run one
step at a time on CPU:

    PDSTL_DEVICE=cpu python -m experiments.inspection_2d checks
"""

import argparse
import csv
from functools import reduce

import numpy as np
import torch

from experiments.reach_avoid import tracking_controls
from models.rollouts import create_gaussian_belief_trajectory
from pdstl.operators import Always, Eventually, Or, Until
from planning.environment import (
    Environment,
    RectangleRegion,
    _FreeGrid,
    inside,
    safety_event,
)
from planning.runners import RESULTS_DIR, setup_problem
from utils import load_config

CONFIG = "configs/scenarios/inspection_2d.yaml"
OUTPUTS = RESULTS_DIR / "inspection_2d"
INSPECT = ("P1", "P2")


def build_environment(cfg):
    """Workspace, racks, targets P1, P2, C and goal D; the start must be safe."""
    env = Environment()
    env.set_bounds(**cfg["bounds"])
    for obstacle in cfg["obstacles"]:
        env.add_obstacle(**obstacle)
    for name, box in cfg["regions"].items():
        env.add_region(
            RectangleRegion(
                name,
                "goal" if name == "D" else "target",
                *map(float, (*box["x_range"], *box["y_range"])),
            )
        )
    # The specs start at step 1, so step 0 is checked here.
    if not satisfied(safety_event(env), [cfg["x0_mean"][:2]]):
        raise ValueError(f"start {cfg['x0_mean'][:2]} is not safe")
    return env


def build_specs(env, cfg):
    """R1-R3, each with its named parts.

    Returns:
        {"R1": (formula, parts), ...}; parts maps safety, ordering,
        inspection and deadline (those that apply) to sub-formulas, in the
        order failures are classified.
    """
    H, dwell, deadline = cfg["H"], cfg["dwell"], cfg["deadline"]
    region = {name: inside(env.region(name)) for name in cfg["regions"]}
    inspect = reduce(Or, [Always(region[n], [0, dwell]) for n in INSPECT])
    safety = Always(safety_event(env), [1, H])
    reach = Eventually(region["D"], [1, deadline])
    inspection = Eventually(inspect, [1, H - dwell])
    r1 = safety & reach
    r3 = (
        r1
        & Until(~region["C"], inspect, [1, H - dwell])
        & Until(~region["D"], region["C"], [1, deadline])
    )
    return {
        "R1": (r1, {"safety": safety, "deadline": reach}),
        "R2": (
            r1 & inspection,
            {"safety": safety, "inspection": inspection, "deadline": reach},
        ),
        "R3": (
            r3,
            {
                "safety": safety,
                "ordering": ordering(region, inspect, cfg),
                "inspection": inspection,
                "deadline": reach,
            },
        ),
    }


def ordering(region, inspect, cfg):
    """R3's order (inspect before C, C before D), for failure classes only.

    Unlike R3's strong Untils, it must not fail just because the inspection
    or C never happens: those failures belong to the inspection and
    deadline parts. R3 holds exactly when all its parts hold.

    Args:
        region: {name: inside(region)} for P1, P2, C, D.
        inspect: Dwell in P1 or P2, Always[0, dwell] P1 | Always[0, dwell] P2.
        cfg: Scenario config (H, dwell, deadline).
    """
    H, deadline = cfg["H"], cfg["deadline"]
    c, d = region["C"], region["D"]
    # Weak Until: phi W psi = (phi U psi) | Always phi.
    inspect_first = Until(~c, inspect, [1, H - cfg["dwell"]]) | Always(
        ~c, [0, H]
    )
    c_first = Until(~d, c, [1, deadline]) | Always(~d, [0, H])
    return inspect_first & c_first


def satisfied(formula, states):
    """Boolean truth at step 0 of [T, D] or [N, T, D] noise-free states."""
    states = torch.as_tensor(states, dtype=torch.float32)
    if states.ndim == 2:
        states = states.unsqueeze(0)
    beliefs = create_gaussian_belief_trajectory(
        states, torch.zeros_like(states)
    )
    return formula(beliefs)[:, 0, 0] > 0.5


# Routes and seeds


def prescribed_route(env, cfg, stops):
    """Collision-free route from the start through `stops`, in that order.

    Args:
        env: The inspection Environment.
        cfg: Scenario config (x0_mean, route).
        stops: [x, y] points or region names (their centres).

    Returns:
        (length in m over _FreeGrid, corner points after the start, index
        of each stop's corner).
    """
    grid = _FreeGrid(env, **cfg["route"])
    points = [cfg["x0_mean"][:2]] + [
        env.region(stop).centre if isinstance(stop, str) else stop
        for stop in stops
    ]
    cells = [grid.cell(point) for point in points]
    length, corners, ends = 0.0, [], []
    for a, b in zip(cells, cells[1:]):
        distance, parent = grid.search(a)
        length += distance[b] * grid.step  # Dijkstra counts cells
        corners += grid.shortcut(grid.trace(parent, a, b))
        ends.append(len(corners) - 1)
    return length, corners, ends


def route_reference(cfg, corners, hold_at, hold, arrival):
    """[H+1, 2] positions along the corners, at rest at both leg ends.

    Each leg (to corners[hold_at], then to the last corner) accelerates and
    brakes at one constant rate, the same on both legs. The reference rests
    `hold` steps at corners[hold_at] and ends at arrival * H.
    """
    points = np.array([cfg["x0_mean"][:2], *corners])
    arc = np.r_[0, np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
    legs = np.array([arc[hold_at + 1], arc[-1] - arc[hold_at + 1]])
    # Rest-to-rest time grows as sqrt(length) at a fixed acceleration.
    time = (arrival * cfg["H"] - hold) * np.sqrt(legs) / np.sqrt(legs).sum()
    t = np.arange(cfg["H"] + 1.0)
    distance = legs[0] * _bang_bang(t / time[0]) + legs[1] * _bang_bang(
        (t - time[0] - hold) / time[1]
    )
    return np.stack(
        [np.interp(distance, arc, points[:, k]) for k in (0, 1)], 1
    )


def _bang_bang(tau):
    """Rest-to-rest progress in [0, 1]: accelerate, then brake equally."""
    tau = np.clip(tau, 0, 1)
    return np.where(tau < 0.5, 2 * tau**2, 1 - 2 * (1 - tau) ** 2)


def route_seed(env, cfg, dyn, name):
    """Reference that holds in P1 or P2, and the controls tracking it."""
    stops = cfg["routes"][name]["stops"]
    _, corners, ends = prescribed_route(env, cfg, stops)
    hold_at = ends[next(i for i, s in enumerate(stops) if s in INSPECT)]
    reference = route_reference(cfg, corners, hold_at, **cfg["seeds"])
    return reference, tracking_controls(cfg, dyn, reference)


def _inside(env, positions, name):
    region, (x, y) = env.region(name), np.asarray(positions)[:, :2].T
    return (
        (region.xmin <= x)
        & (x <= region.xmax)
        & (region.ymin <= y)
        & (y <= region.ymax)
    )


def longest_stay(env, positions, name):
    """Longest run of consecutive samples inside a region."""
    best = run = 0
    for flag in _inside(env, positions, name):
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def first_inside(env, positions, name):
    """Index of the first sample inside a region."""
    hits = np.flatnonzero(_inside(env, positions, name))
    if not len(hits):
        raise ValueError(f"the path never enters {name}")
    return int(hits[0])


def problem(cfg, method="pdstl", **overrides):
    """setup_problem for one method's planner settings and noise."""
    return setup_problem(
        {**cfg, **overrides, "planner": cfg["methods"][method]}
    )


def _mean(s, controls):
    return s.planner.evaluate_controls(s.rollout, controls, spec=s.spec)


def _write(rows, name):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS / name, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUTPUTS / name}", flush=True)


def _print(rows):
    keys = list(rows[0])
    print("  ".join(f"{k:>12}" for k in keys))
    for row in rows:
        print(
            "  ".join(
                f"{v:>12.3f}" if isinstance(v, float) else f"{v!s:>12}"
                for v in row.values()
            )
        )


# Step 2: checks


def checks(cfg):
    """Route lengths, seed dwell, nominal R3 feasibility and a noise table."""
    env = build_environment(cfg)
    r3, parts = build_specs(env, cfg)["R3"]
    s = problem(cfg, **cfg["nominal"])
    s.spec = r3
    rows = []
    for name, route in cfg["routes"].items():
        grid_m, corners, _ = prescribed_route(env, cfg, route["stops"])
        reference, controls = route_seed(env, cfg, s.dyn, name)
        mean = _mean(s, controls).rollout.aux["mean_trace"][0, :, :2]
        target = next(stop for stop in route["stops"] if stop in INSPECT)
        rows.append(
            {
                "route": name,
                "grid_m": grid_m,
                "path_m": float(
                    np.linalg.norm(np.diff(reference, axis=0), axis=1).sum()
                ),
                "ref_dwell": longest_stay(env, reference, target) - 1,
                "seed_dwell": longest_stay(env, mean, target) - 1,
                "seed_safe": bool(satisfied(parts["safety"], mean)),
                "seed_R3": bool(satisfied(r3, mean)),
            }
        )
    print("\nRoutes (dwell in steps, needs >= {})".format(cfg["dwell"]))
    _print(rows)

    nominal = []
    for name in cfg["starts"]:
        s.init_guess = route_seed(env, cfg, s.dyn, name)[1]
        result = s.planner.optimize_window(
            s.rollout, spec=r3, init_guess=s.init_guess
        )
        mean = result.rollout.aux["mean_trace"][0, :, :2]
        nominal.append(
            {
                "start": name,
                "initial": result.initial_hard_lower,
                "final": result.hard_interval[0],
                "mean_R3": bool(satisfied(r3, mean)),
                "time_s": result.planning_time,
            }
        )
    print("\nNominal pdSTL on R3 (near-zero noise)")
    _print(nominal)

    noise = []
    for q_std in cfg["noise_candidates"]:
        # Open loop, the covariance does not depend on the controls.
        noisy = problem(cfg, q_std=q_std)
        cov = noisy.dyn(torch.zeros(cfg["H"], 2), *noisy.state)[1][0]
        sigma = cov[:, 0, 0].sqrt() * 100
        for name, route in cfg["routes"].items():
            reference, _ = route_seed(env, cfg, noisy.dyn, name)
            narrow = np.linalg.norm(reference - route["narrow"], axis=1)
            at_narrow = int(narrow.argmin())
            at_d = first_inside(env, reference, "D")
            noise.append(
                {
                    "q_std": q_std,
                    "route": name,
                    "t_narrow_s": at_narrow * cfg["dt"],
                    "sigma_narrow_cm": float(sigma[at_narrow]),
                    "t_D_s": at_d * cfg["dt"],
                    "sigma_D_cm": float(sigma[at_d]),
                }
            )
    print("\nPosition sigma along each reference")
    _print(noise)
    _write(rows, "checks_routes.csv")
    _write(nominal, "checks_nominal.csv")
    _write(noise, "checks_noise.csv")


STEPS = {"checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args()
    STEPS[args.step](load_config(args.config))


if __name__ == "__main__":
    main()
