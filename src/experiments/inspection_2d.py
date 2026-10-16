"""2D inspection: pdSTL vs deterministic STLCG on specs R1-R3.

STLCG plans from two seeds once per spec and selects by mean robustness.
pdSTL refines both STLCG plans at each noise level and selects by exact score.

The map and settings come from configs/scenarios/inspection_2d.yaml. Run one
step at a time on CPU:

    PDSTL_DEVICE=cpu python -m experiments.inspection_2d check
"""

import argparse
import csv
from functools import reduce
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import torch

from baselines.det_stl import from_pdstl
from experiments.reach_avoid import tracking_controls
from models.rollouts import (
    create_gaussian_belief_trajectory,
    sample_trajectories,
)
from pdstl.operators import Always, Eventually, Or, Until
from planning.environment import (
    Environment,
    RectangleRegion,
    _FreeGrid,
    inside,
    safety_event,
)
from planning.planner import Planner
from planning.runners import RESULTS_DIR, monte_carlo, setup_problem
from utils import load_config

CONFIG = "configs/scenarios/inspection_2d.yaml"
OUTPUTS = RESULTS_DIR / "inspection_2d"
INSPECT = ("P1", "P2")
# Named parts, in the order a failed rollout is classified.
PARTS = ("safety", "ordering", "inspection", "deadline")
# Five evaluation cases; pdSTL refines both STLCG seed plans in each case.
CASES = [("R1", "medium"), ("R2", "medium"), ("R3", "medium")] + [
    ("R3", "low"),
    ("R3", "high"),
]


def build_environment(cfg):
    """Racks, targets P1, P2, C and goal D; the start must be safe."""
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
    for a, b in pairwise(cells):
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


class InspectionPlanner(Planner):
    """Preserve supplied controls, including near-saturated STLCG controls."""

    def _init_controls(self, init_guess):
        if init_guess is None:
            return super()._init_controls(init_guess)
        parameters = self._control_parameters(
            init_guess, margin=torch.finfo(self.dyn.B.dtype).eps
        )
        return torch.nn.Parameter(parameters.detach().clone())


def problem(cfg, method="pdstl", planner=None, **overrides):
    """setup_problem with one method's planner settings (plus `planner`)."""
    settings = {**cfg["methods"][method], **(planner or {})}
    s = setup_problem({**cfg, **overrides, "planner": settings})
    s.planner = InspectionPlanner(s.dyn, cfg["H"], settings)
    return s


def _mean(s, controls):
    return s.planner.evaluate_controls(s.rollout, controls, spec=s.spec)


def _write(rows, name):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS / name, "w", newline="", encoding="utf-8") as file:
        fields = dict.fromkeys(key for row in rows for key in row)
        writer = csv.DictWriter(
            file, fieldnames=list(fields), lineterminator="\n"
        )
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
        grid_m, _, _ = prescribed_route(env, cfg, route["stops"])
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

    nominal, _, _ = stlcg_plan(cfg, env, ("R3", r3))
    print("\nNominal STLCG on R3 (no noise)")
    _print(nominal)

    noise = []
    for q_std in cfg["noise"].values():
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


# Steps 3-4: planning runs


def noise_level(cfg, noise):
    """q_std of a named noise level."""
    q_std = cfg["noise"][noise]
    if q_std is None:
        raise ValueError(f"set noise.{noise} in {CONFIG} first")
    return q_std


def run_start(
    cfg,
    env,
    spec,
    method,
    start,
    noise="none",
    planner=None,
    *,
    init_guess=None,
    stlcg_time=0.0,
):
    """Plan one start and describe it.

    Args:
        spec: (name, formula).
        planner: Optional planner overrides (the lr pilot).
        init_guess: One STLCG seed plan's controls, required for pdSTL.
        stlcg_time: Total optimizer time to select the STLCG plan.

    Returns:
        (row, controls or None, [(iteration, seconds, controls), ...]).
    """
    name, formula = spec
    if method == "stlcg":
        s = problem(cfg, method, planner, q_std=0.0, x0_cov_scale=0.0)
        _, guess = route_seed(env, cfg, s.dyn, start)
    else:
        if init_guess is None:
            raise ValueError("pdSTL requires a STLCG seed plan's controls")
        s = problem(cfg, method, planner, q_std=noise_level(cfg, noise))
        guess = init_guess
    log = [(-1, 0.0, guess)]

    def observe(iteration, record):
        if record.elapsed > cfg["time_limit"]:
            raise TimeoutError
        log.append((iteration, record.elapsed, record.controls))

    row = {"spec": name, "noise": noise, "method": method, "start": start}
    try:
        result = s.planner.optimize_window(
            s.rollout,
            spec=formula,
            init_guess=guess,
            on_iteration=observe,
            callback_every=cfg["log_every"],
        )
    except TimeoutError:
        return {**row, "solver_status": "timeout"}, None, log
    except (RuntimeError, ValueError) as error:
        return (
            {**row, "solver_status": "error", "error": str(error)},
            None,
            log,
        )
    mean = result.rollout.aux["mean_trace"][0]
    pickup = [p for p in INSPECT if longest_stay(env, mean, p) > cfg["dwell"]]
    row.update(
        solver_status="completed",
        mean_satisfies_spec=bool(satisfied(formula, mean)),
        pickup=pickup[0] if pickup else "neither",
        slot_traversed=slot_traversed(env, mean),
        stlcg_time=(result.planning_time if method == "stlcg" else stlcg_time),
        refine_time=(result.planning_time if method == "pdstl" else 0.0),
        planning_time=stlcg_time + result.planning_time,
        control_effort=float(result.controls.square().sum() * cfg["dt"]),
        initial_score=result.initial_hard_lower,
        final_score=result.hard_interval[0],
        own_score=own_score(method, formula, result),
    )
    # The best pdSTL iterate may precede the last optimizer iterate.
    log.append(
        (result.selected_iteration, result.planning_time, result.controls)
    )
    return row, result.controls, log


def own_score(method, formula, result):
    """Each method's selection criterion: hard pdSTL or mean robustness."""
    if method == "pdstl":
        return result.hard_interval[0]
    mean = result.rollout.aux["mean_trace"]
    return from_pdstl(formula).robustness(mean, beta=None).item()


def slot_traversed(env, positions):
    """Whether a sample lies in the gap between the two racks."""
    lower, upper = env.by_role("obstacle")
    x, y = np.asarray(positions)[:, :2].T
    return bool(
        np.any(
            (lower.xmin <= x)
            & (x <= lower.xmax)
            & (lower.ymax < y)
            & (y < upper.ymin)
        )
    )


def stlcg_plan(cfg, env, spec, planner=None):
    """Run both noise-free seeds and select by final hard mean robustness."""
    group, plans, logs = [], {}, {}
    elapsed = 0.0
    for start in cfg["starts"]:
        row, controls, log = run_start(
            cfg, env, spec, "stlcg", start, planner=planner
        )
        print(row, flush=True)
        group.append(row)
        plans[_key(row)] = controls
        logs[_key(row)] = [(i, t + elapsed, u) for i, t, u in log]
        elapsed += row.get("planning_time", log[-1][1])
    done = [r for r in group if r["solver_status"] == "completed"]
    if not done:
        raise RuntimeError(f"{spec[0]}: neither STLCG seed completed")
    best = max(done, key=lambda r: r["own_score"])
    for row in group:
        row["selected"] = row is best
    # Selection costs both seed runs, even when the first seed wins.
    best.update(stlcg_time=elapsed, planning_time=elapsed)
    key = _key(best)
    logs[key].append((logs[key][-1][0], elapsed, plans[key]))
    return group, plans, logs


def pilot(cfg):
    """Tune STLCG on noise-free R2, then pdSTL on its plan at medium noise."""
    env = build_environment(cfg)
    spec = ("R2", build_specs(env, cfg)["R2"][0])
    rows, candidates = [], []
    for lr in cfg["lr_grid"]:
        group, plans, _ = stlcg_plan(cfg, env, spec, {"lr": lr})
        rows += [{"lr": lr, **row} for row in group]
        best = next(r for r in group if r["selected"])
        candidates.append((lr, best, plans[_key(best)]))
    stlcg_lr, base, controls = max(
        candidates, key=lambda candidate: candidate[1]["own_score"]
    )
    for lr in cfg["lr_grid"]:
        row, _, _ = run_start(
            cfg,
            env,
            spec,
            "pdstl",
            base["start"],
            "medium",
            {"lr": lr},
            init_guess=controls,
            stlcg_time=base["stlcg_time"],
        )
        rows.append({"lr": lr, **row})
        print(row, flush=True)
    _write(rows, "lr_pilot.csv")
    refinements = [
        r
        for r in rows
        if r["method"] == "pdstl" and r["solver_status"] == "completed"
    ]
    if not refinements:
        raise RuntimeError("no pdSTL pilot refinement completed")
    pdstl_lr = max(refinements, key=lambda r: r["own_score"])["lr"]
    print(f"Freeze in YAML: stlcg.lr={stlcg_lr}, pdstl.lr={pdstl_lr}")
    return {"stlcg": stlcg_lr, "pdstl": pdstl_lr}


def runs(cfg):
    """Six STLCG seed runs, then ten refinements selected by exact pdSTL."""
    env = build_environment(cfg)
    specs = build_specs(env, cfg)
    rows, convergence, plans, bases = [], [], {}, {}
    for name, (formula, _) in specs.items():
        group, controls, logs = stlcg_plan(cfg, env, (name, formula))
        rows += group
        plans.update({k: u for k, u in controls.items() if u is not None})
        if any(r["solver_status"] != "completed" for r in group):
            raise RuntimeError(f"{name}: both STLCG seed plans are required")
        bases[name] = group
        for row in group:
            for spec, noise in CASES:
                if spec == name:
                    convergence += _score_log(
                        cfg, row, formula, logs[_key(row)], noise=noise
                    )
    for name, noise in CASES:
        stlcg_time = next(r for r in bases[name] if r["selected"])[
            "stlcg_time"
        ]
        group, logs, refine_time = [], {}, 0.0
        for base in bases[name]:
            row, controls, log = run_start(
                cfg,
                env,
                (name, specs[name][0]),
                "pdstl",
                base["start"],
                noise,
                init_guess=plans[_key(base)],
                stlcg_time=stlcg_time,
            )
            group.append(row)
            logs[_key(row)] = [
                (i, t + stlcg_time + refine_time, u) for i, t, u in log
            ]
            refine_time += row.get("refine_time", log[-1][1])
            if controls is not None:
                plans[_key(row)] = controls
        done = [r for r in group if r["solver_status"] == "completed"]
        best = max(done, key=lambda r: r["own_score"], default=None)
        if best is not None:
            best.update(
                refine_time=refine_time, planning_time=stlcg_time + refine_time
            )
            key = _key(best)
            logs[key].append(
                (logs[key][-1][0], best["planning_time"], plans[key])
            )
        for row in group:
            row["selected"] = row is best
            print(row, flush=True)
            convergence += _score_log(
                cfg, row, specs[name][0], logs[_key(row)]
            )
        rows += group
    _write(rows, "runs.csv")
    _write(convergence, "convergence.csv")
    torch.save(plans, OUTPUTS / "plans.pt")


def _score_log(cfg, row, formula, log, *, noise=None):
    """Hard pdSTL scores at evaluation noise, outside optimizer timing."""
    noise = noise or row["noise"]
    s = problem(cfg, q_std=noise_level(cfg, noise))
    keys = {k: row[k] for k in ("spec", "noise", "method", "start")}
    keys["noise"] = noise
    return [
        {
            **keys,
            "iteration": iteration,
            "time_s": seconds,
            "score": s.planner.evaluate_controls(
                s.rollout, controls, spec=formula
            ).hard_interval[0],
        }
        for iteration, seconds, controls in log
    ]


def selected_runs():
    """Reuse each selected STLCG plan at all of its evaluation noise levels."""
    selected = {}
    for row in _read("runs.csv"):
        if row["selected"] != "True":
            continue
        if row["method"] == "stlcg":
            for spec, noise in CASES:
                if spec == row["spec"]:
                    selected[(spec, noise, "stlcg")] = row
        else:
            selected[(row["spec"], row["noise"], "pdstl")] = row
    return selected


def _key(row):
    return "/".join(row[k] for k in ("spec", "noise", "method", "start"))


# Step 5: Monte Carlo


def validate(cfg):
    """Monte Carlo of every selected plan, with failure flags and classes."""
    env = build_environment(cfg)
    specs = build_specs(env, cfg)
    plans = torch.load(OUTPUTS / "plans.pt")
    samples = cfg["monte_carlo"]["samples"]
    rows = []
    for (name, noise, method), run in selected_runs().items():
        formula, parts = specs[name]
        s = problem(cfg, q_std=noise_level(cfg, noise))
        s.spec = formula
        controls = plans[_key(run)]
        rate, lower, upper = monte_carlo(
            s, SimpleNamespace(controls=controls), **cfg["monte_carlo"]
        )
        flags = failure_flags(s, controls, parts, **cfg["monte_carlo"])
        classes = failure_classes(flags)
        if (classes != "").sum() != samples - round(rate * samples):
            raise RuntimeError(f"{_key(run)}: parts disagree with the spec")
        row = {k: run[k] for k in ("spec", "noise", "method", "start")}
        row.update(
            noise=noise,
            hard_score=_mean(s, controls).hard_interval[0],
            empirical_success_rate=rate,
            ci_low=lower,
            ci_high=upper,
        )
        for part in PARTS:  # fixed columns; blank where a part is absent
            held = flags.get(part)
            row[f"fail_{part}"] = (
                "" if held is None else 1 - held.float().mean().item()
            )
            row[f"class_{part}"] = (
                "" if held is None else float((classes == part).mean())
            )
        print(row, flush=True)
        rows.append(row)
    _write(rows, "monte_carlo.csv")


def failure_flags(s, controls, parts, samples, seed):
    """Whether each part holds on each rollout of runners.monte_carlo."""
    generator = torch.Generator(device=s.dyn.device).manual_seed(seed)
    paths = sample_trajectories(s.dyn, *s.state, controls, samples, generator)
    return {name: satisfied(part, paths) for name, part in parts.items()}


def failure_classes(flags):
    """The first failing part of each rollout, in `flags` order; "" if none."""
    failed = ~torch.stack(list(flags.values()), dim=1)
    first = torch.where(failed.any(1), failed.int().argmax(1) + 1, 0)
    return np.array(["", *flags])[first.cpu().numpy()]


def _read(name):
    with open(OUTPUTS / name, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# Paper outputs: read the saved results only

METHODS = {"stlcg": ("STLCG", "#D55E00"), "pdstl": ("pdSTL", "#0072B2")}
CLASS_COLORS = ("#E69F00", "#56B4E9", "#009E73", "#999999")
TIMING = (
    "Optimizer wall time on CPU, excluding diagnostic pdSTL scoring.",
    (
        "STLCG includes both seed runs; pdSTL includes both STLCG seed runs "
        "and both refinements."
    ),
)
FONTS = {
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}


def paper(cfg):
    """Figure 1, Table 1 and the supplement from the saved CSVs and plans."""
    import matplotlib.pyplot as plt

    selected = selected_runs()
    checked = {
        (r["spec"], r["noise"], r["method"]): r
        for r in _read("monte_carlo.csv")
    }
    with plt.rc_context(FONTS):
        figure_1(cfg, selected)
        failure_bars(checked)
    table(selected, checked, CASES[:3], "table1")
    table(selected, checked, CASES[3:], "table_s1", by_noise=True)


def figure_1(cfg, selected):
    """(a) Selected R2 paths at medium noise; (b) cumulative planning time."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    from visualization.figures import (
        RHO_LOWER,
        _draw_environment,
        _ellipse_parameters,
        _finish,
    )

    env = build_environment(cfg)
    plans = torch.load(OUTPUTS / "plans.pt")
    convergence = _read("convergence.csv")
    fig = plt.figure(figsize=(7.16, 3.0), layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=(1.0, 1.3))
    ax = fig.add_subplot(grid[:, 0])
    _draw_environment(ax, env)
    ax.tick_params(labelsize=FONTS["xtick.labelsize"])
    paths = []
    for method, (label, color) in METHODS.items():
        run = selected[("R2", "medium", method)]
        mean, cov = _belief(cfg, plans[_key(run)], "medium")
        paths += ax.plot(
            *mean[:, :2].T,
            color=color,
            linewidth=1.4,
            zorder=8,
            label=f"{label} ({_route(run)})",
        )
        if method != "pdstl":
            continue
        for k in range(0, len(mean), cfg["paper"]["ellipse_every"]):
            width, height, angle = _ellipse_parameters(cov[k], 0.95)
            ax.add_patch(
                Ellipse(
                    mean[k, :2],
                    width,
                    height,
                    angle=angle,
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                    zorder=7,
                )
            )
    ax.legend(handles=paths, loc="upper left", framealpha=0.9)
    ax.set_title("R2, medium noise")
    ax.text(0.02, 0.02, "(a)", transform=ax.transAxes)
    for row, spec in enumerate(("R1", "R2", "R3")):
        panel = fig.add_subplot(grid[row, 1])
        for method, (label, color) in METHODS.items():
            run = selected[(spec, "medium", method)]
            key = _key({**run, "noise": "medium"})
            points = [
                (float(r["time_s"]), float(r["score"]))
                for r in convergence
                if _key(r) == key
            ]
            panel.plot(*zip(*points), color=color, linewidth=1.2, label=label)
        panel.set(ylim=(-0.03, 1.03), ylabel=RHO_LOWER)
        panel.text(0.98, 0.08, spec, transform=panel.transAxes, ha="right")
        panel.grid(alpha=0.2, linewidth=0.5)
        if row < 2:
            panel.tick_params(labelbottom=False)
    panel.set_xlabel("cumulative optimizer time [s]")
    panel.text(0.02, 0.08, "(b)", transform=panel.transAxes)
    _finish(fig, OUTPUTS / "figure1.pdf", show=False)


def _belief(cfg, controls, noise):
    """Mean and covariance traces of open-loop controls."""
    s = problem(cfg, q_std=noise_level(cfg, noise))
    mean, cov = s.state
    means, covs = [mean], [cov]
    for control in controls:
        mean, cov = s.dyn.step(mean, cov, control)
        means.append(mean)
        covs.append(cov)
    return torch.stack(means).numpy(), torch.stack(covs).numpy()


def failure_bars(checked):
    """Stacked failure classes of R3 per noise level and method."""
    import matplotlib.pyplot as plt

    from visualization.figures import _finish

    fig, ax = plt.subplots(figsize=(3.5, 2.0), layout="constrained")
    bars = [
        (noise, method)
        for noise in ("low", "medium", "high")
        for method in METHODS
    ]
    bottom = np.zeros(len(bars))
    for part, color in zip(PARTS, CLASS_COLORS):
        height = np.array(
            [float(checked[("R3", *bar)][f"class_{part}"]) for bar in bars]
        )
        ax.bar(
            range(len(bars)), height, bottom=bottom, color=color, label=part
        )
        bottom += height
    ax.set_xticks(
        range(len(bars)),
        [f"{METHODS[m][0]}\n{noise}" for noise, m in bars],
    )
    ax.set(ylabel="failed rollouts [fraction]", ylim=(0, 1))
    ax.legend(ncol=2, frameon=False)
    _finish(fig, OUTPUTS / "failure_classes.pdf", show=False)


def _route(run):
    """Route traversed by the final mean path, rather than its seed label."""
    return "slot" if run["slot_traversed"] == "True" else "top"


def table(selected, checked, cases, name, by_noise=False):
    """CSV and booktabs LaTeX: score / time / success (route) [CI]."""
    rows = []
    for spec, noise in cases:
        row = {"spec": spec, "noise": noise}
        for method, (label, _) in METHODS.items():
            run, mc = (
                selected[(spec, noise, method)],
                checked[(spec, noise, method)],
            )
            row[f"{label} score"] = float(mc["hard_score"])
            row[f"{label} time_s"] = float(run["planning_time"])
            row[f"{label} success"] = float(mc["empirical_success_rate"])
            row[f"{label} route"] = _route(run)
            row[f"{label} ci"] = (float(mc["ci_low"]), float(mc["ci_high"]))
        rows.append(row)
    _write(rows, f"{name}.csv")
    labels = [label for label, _ in METHODS.values()]
    lines = [
        r"\begin{tabular}{l" + "ccc" * len(labels) + "}",
        r"\toprule",
        " & ".join(["", *(rf"\multicolumn{{3}}{{c}}{{{m}}}" for m in labels)])
        + r" \\",
        "".join(
            rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}"
            for k in range(len(labels))
        ),
        "Spec"
        + r" & score & time\textsuperscript{*} [s] & success [95\% CI]"
        * len(labels)
        + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [f"{row['spec']} {row['noise']}" if by_noise else row["spec"]]
        for label in labels:
            low, high = row[f"{label} ci"]
            cells += [
                f"{row[f'{label} score']:.3f}",
                f"{row[f'{label} time_s']:.1f}",
                (
                    f"{100 * row[f'{label} success']:.1f}\\% "
                    f"({row[f'{label} route']}) [{100 * low:.1f}, {100 * high:.1f}]"
                ),
            ]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines += [
        (
            rf"\multicolumn{{{1 + 3 * len(labels)}}}{{l}}"
            rf"{{\footnotesize\textsuperscript{{*}}{line}}} \\"
        )
        for line in TIMING
    ]
    lines.append(r"\end{tabular}")
    (OUTPUTS / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUTS / name}.tex", flush=True)


STEPS = {
    "check": checks,
    "checks": checks,
    "pilot": pilot,
    "runs": runs,
    "monte-carlo": validate,
    "paper": paper,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args()
    STEPS[args.step](load_config(args.config))


if __name__ == "__main__":
    main()
