"""Gap or detour: a route choice that flips as uncertainty grows.

One map, one decision. The short route passes a narrow gap (a wall on each
side) to inspect A; the long route passes a wide opening to inspect B,
driving by the dock it may not enter yet. The task is

    G[0,H] safe & (not dock) U[0,H-d] (G[0,d] A | G[0,d] B)
    & F[0,H-m] G[0,m] dock.

Centred in a gap of width w with lateral spread sigma, z = (w/2)/sigma.
Tightened STL constrains each wall separately (accepts the gap at
Phi(z) >= alpha); pdSTL's Frechet conjunction gives 2 Phi(z) - 1, which is
exact because both walls cannot be hit at once. For sigma between
w/(2 Phi^-1((1+alpha)/2)) and w/(2 Phi^-1(alpha)), tightened STL takes the
gap while its true per-step safety is below alpha, and pdSTL takes the
detour.

Each method plans from every route's warm start and takes the shortest route
it certifies: mean STL (robustness >= 0 on the mean), tightened STL
(robustness >= 0 with each margin shifted by Phi^-1(alpha) sigma), and pdSTL
(exact lower bound >= alpha). The chosen plan is checked by Monte Carlo per
time step, the quantity pointwise semantics bounds.

`run` plans the config's noise level with a live view, like the other
examples. `sweep` plans every level in `sweep.levels` in parallel CPU
processes and saves the route and safety figures under outputs/inspection/.
"""

import copy
import csv
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from statistics import NormalDist
from time import perf_counter
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
from scipy.stats import beta as beta_distribution

from baselines.det_stl import from_pdstl
from experiments.reach_avoid import straight_line_guess
from models.rollouts import (
    create_gaussian_belief_trajectory,
    sample_trajectories,
)
from pdstl.operators import Always, And, Eventually, Negation, Or, Until
from planning.environment import (
    Environment,
    RectangleRegion,
    inside,
    safety_event,
)
from planning.planner import IterationRecord
from planning.runners import RESULTS_DIR, monte_carlo, setup_problem
from utils import load_config

METHODS = ("stl_mean", "stl_tight", "pdstl")
LABELS = {
    "stl_mean": "STL on the mean",
    "stl_tight": "Tightened STL",
    "pdstl": "pdSTL",
}
METHOD_COLORS = {
    "stl_mean": "tab:gray",
    "stl_tight": "tab:orange",
    "pdstl": "tab:blue",
}
ROLES = {"inspect": "target", "dock": "goal"}  # config role -> drawing role
OUTPUT_DIR = RESULTS_DIR / "inspection"
COLUMNS = (
    "level",
    "method",
    "start",
    "certified",
    "score",
    "pdstl_lower",
    "tight_rho",
    "mean_rho",
    "crossing",
    "via_sigma",
    "planning_time_s",
)


# Scenario


def build_environment(cfg):
    """Workspace, walls, and the inspection and dock regions of a config."""
    environment = Environment()
    environment.set_bounds(**cfg["bounds"])
    for wall in cfg["walls"]:
        environment.add_obstacle(**wall)
    for name, box in cfg["regions"].items():
        if box["role"] not in ROLES:
            raise ValueError(
                f"region {name!r}: role must be one of {list(ROLES)}"
            )
        environment.add_region(
            RectangleRegion(
                name, ROLES[box["role"]], *box["x_range"], *box["y_range"]
            )
        )
    return environment


def specification(environment, cfg):
    """The safety event and the full task.

    Args:
        environment: From build_environment.
        cfg: Config with H, dwell, hold, and regions (role inspect or dock).

    Returns:
        (safe, phi): the pointwise safety event and the pdSTL task.
    """
    horizon, dwell, hold = cfg["H"], cfg["dwell"], cfg["hold"]
    names = {
        role: [n for n, box in cfg["regions"].items() if box["role"] == role]
        for role in ROLES
    }
    if not names["inspect"] or len(names["dock"]) != 1:
        raise ValueError("needs inspection regions and exactly one dock")
    zone = {name: inside(environment.region(name)) for name in cfg["regions"]}
    dock = zone[names["dock"][0]]
    inspected = reduce(
        Or, [Always(zone[name], [0, dwell]) for name in names["inspect"]]
    )
    safe = safety_event(environment)
    phi = reduce(
        And,
        [
            Always(safe, [0, horizon]),
            Until(Negation(dock), inspected, [0, horizon - dwell]),
            Eventually(Always(dock, [0, hold]), [0, horizon - hold]),
        ],
    )
    return safe, phi


def noise_level(cfg):
    """The first process-noise standard deviation, the swept quantity."""
    q_std = cfg["q_std"]
    return float(q_std[0] if isinstance(q_std, (list, tuple)) else q_std)


def with_noise(cfg, level):
    """A copy of cfg whose q_std is rescaled so its first entry is level."""
    base = noise_level(cfg)
    if base <= 0:
        raise ValueError("the first q_std entry must be positive to rescale")
    cfg = copy.deepcopy(cfg)
    q_std = cfg["q_std"]
    cfg["q_std"] = (
        [value * level / base for value in q_std]
        if isinstance(q_std, (list, tuple))
        else level
    )
    return cfg


def build(cfg, device="cpu"):
    """Problem at the config's noise level, with every route's warm start.

    Returns:
        Namespace from setup_problem plus env, safe, spec, alpha, level,
        routes (shortest first), starts, and the deterministic formulas.
    """
    s = setup_problem(cfg, device=device)
    s.env = build_environment(cfg)
    s.safe, s.spec = specification(s.env, cfg)
    s.alpha = s.planner.cfg["alpha"]
    s.level = noise_level(cfg)
    s.routes = list(cfg["routes"])
    s.starts = {
        route: straight_line_guess(cfg, s.dyn, spec["waypoints"])
        for route, spec in cfg["routes"].items()
    }
    s.robustness = {
        "stl_mean": from_pdstl(s.spec),
        "stl_tight": from_pdstl(s.spec, tighten=NormalDist().inv_cdf(s.alpha)),
    }
    return s


# Planning


def plan_stl(s, robustness, init_guess, on_iteration=None, callback_every=1):
    """Adam on deterministic STL robustness; keep the best hard iterate.

    Uses the planner's tanh-bounded parameters, control cost, learning rate,
    and smoothing schedule, so only the objective and selection differ from
    pdSTL planning. An observer sees each candidate's pdSTL interval, like
    pdSTL planning's live view.

    Args:
        s: Problem from build.
        robustness: Deterministic formula from from_pdstl.
        init_guess: [H, 2] starting controls.
        on_iteration, callback_every: Optional observer of sampled iterates.

    Returns:
        (controls, hard robustness, selected iteration, pdSTL lower bound
        of every iterate or None without an observer).
    """
    planner = s.planner
    parameters = planner._init_controls(init_guess)
    optimizer = torch.optim.Adam([parameters], lr=planner.cfg["lr"])
    best = (-math.inf, None, -1)
    history = None if on_iteration is None else []
    max_iters = planner.cfg["max_iters"]
    for step in range(max_iters + 1):
        iteration = step - 1
        optimizer.zero_grad()
        beliefs = s.rollout(parameters).belief_trajectory
        controls = s.dyn.bound_control(parameters)
        with torch.no_grad():
            rho = robustness.robustness(beliefs).item()
            interval = (
                None
                if history is None
                else tuple(s.spec.probability_interval(beliefs).tolist())
            )
        if rho >= best[0]:
            best = (rho, controls.detach().clone(), iteration)
        if history is not None:
            history.append(interval[0])
        if step == max_iters:
            break
        beta = planner._beta(step)
        smooth = robustness.robustness(beliefs, beta=beta).sum()
        control_cost = planner._control_cost(controls)
        loss = -planner.cfg["w_phi"] * smooth + control_cost
        if on_iteration is not None and iteration % callback_every == 0:
            on_iteration(
                max(iteration, 0),
                IterationRecord(
                    controls.detach().clone(),
                    loss.item(),
                    smooth.item(),
                    interval,
                    beta,
                    control_cost.item(),
                ),
            )
        loss.backward()
        optimizer.step()
    rho, controls, iteration = best
    return controls, rho, iteration, history


def plan(s, method, start, live=None, callback_every=5):
    """One method from one warm start, scored by every criterion.

    Args:
        s: Problem from build.
        method: One of METHODS.
        start: Route whose warm start is used.
        live: Optional (on_iteration, finish) of a live view.
        callback_every: Live-view cadence in iterations.

    Returns:
        Row dict (see COLUMNS) plus the selected `controls`.
    """
    on_iteration, finish = live[:2] if live else (None, None)
    clock = perf_counter()
    if method == "pdstl":
        result = s.planner.optimize_window(
            s.rollout,
            spec=s.spec,
            init_guess=s.starts[start],
            on_iteration=on_iteration,
            callback_every=callback_every,
        )
        controls = result.controls
    else:
        controls, _, iteration, history = plan_stl(
            s,
            s.robustness[method],
            s.starts[start],
            on_iteration,
            callback_every,
        )
    elapsed = perf_counter() - clock
    evaluated = s.planner.evaluate_controls(s.rollout, controls, spec=s.spec)
    if finish is not None:
        if method != "pdstl":
            result = SimpleNamespace(
                controls=controls,
                hard_interval=evaluated.hard_interval,
                hard_lower_history=history,
                selected_iteration=iteration,
            )
        finish(result)
    beliefs = evaluated.rollout.belief_trajectory
    with torch.no_grad():
        scores = {
            "pdstl_lower": evaluated.hard_interval[0],
            "tight_rho": s.robustness["stl_tight"].robustness(beliefs).item(),
            "mean_rho": s.robustness["stl_mean"].robustness(beliefs).item(),
        }
    score = {
        "pdstl": scores["pdstl_lower"],
        "stl_tight": scores["tight_rho"],
        "stl_mean": scores["mean_rho"],
    }[method]
    crossing, via_sigma = crossing_of(s, beliefs)
    return {
        "level": s.level,
        "method": method,
        "start": start,
        "certified": score >= (s.alpha if method == "pdstl" else 0.0),
        "score": score,
        **scores,
        "crossing": crossing,
        "via_sigma": via_sigma,
        "planning_time_s": elapsed,
        "controls": controls.detach().cpu(),
    }


def lateral_dim(via):
    """The axis a passage's walls bound: its `lateral` entry, x or y."""
    try:
        return {"x": 0, "y": 1}[via["lateral"]]
    except KeyError:
        raise ValueError(
            "each route's via needs `lateral: x` or `y`"
        ) from None


def crossing_of(s, beliefs):
    """Which route's passage the mean path goes through, and its spread.

    The passage's walls bind at every step the mean is level with them
    (inside the passage's extent along its long axis); the route is the one
    whose passage contains the mean's median lateral position over those
    steps.

    Returns:
        (route name, or "other" if no passage is crossed; the largest
        lateral standard deviation while level with that passage, or None).
    """
    mean = beliefs.trace.mean[0].detach().cpu()
    covariance = beliefs.trace.covariance[0].detach().cpu()
    axes = ("x_range", "y_range")
    for route, spec in s.cfg["routes"].items():
        via = spec["via"]
        lateral = lateral_dim(via)
        along = 1 - lateral
        (a0, a1), (l0, l1) = via[axes[along]], via[axes[lateral]]
        level = (mean[:, along] >= a0) & (mean[:, along] <= a1)
        steps = level.nonzero()[:, 0]
        if len(steps) and l0 <= float(mean[steps, lateral].median()) <= l1:
            sigma = covariance[steps, lateral, lateral].max().sqrt()
            return route, float(sigma)
    return "other", None


def choose(rows, method, routes):
    """The shortest route this method certifies, or None."""
    for route in routes:
        row = next(
            r for r in rows if r["method"] == method and r["start"] == route
        )
        if row["certified"]:
            return row
    return None


def plan_level(s, *, live=False, callback_every=5):
    """Every method from every warm start at one noise level.

    Returns:
        One row per (method, start), printed as it finishes.
    """
    rows = []
    for method in METHODS:
        for start in s.routes:
            view = _live_view(s, method, start) if live else None
            rows.append(plan(s, method, start, view, callback_every))
            print(_describe(rows[-1]), flush=True)
            if view is not None:
                if plt.get_backend().lower() != "agg":
                    plt.pause(1.0)  # hold the selected plan on screen
                plt.close(view[2])
    return rows


# Validation


def stepwise_safety(s, controls, samples, seed):
    """Monte Carlo P(safe at step t) of an open-loop plan, for every t.

    Returns:
        (per-step rates [H+1], the minimum rate, its 95% Clopper-Pearson
        interval as (lower, upper)).
    """
    generator = torch.Generator(device=s.dyn.device).manual_seed(seed)
    paths = sample_trajectories(
        s.dyn, *s.state, controls.to(s.dyn.device), samples, generator
    )
    beliefs = create_gaussian_belief_trajectory(paths, torch.zeros_like(paths))
    with torch.no_grad():
        safe = s.safe(beliefs)[..., 0]
    rates = safe.mean(0)
    step = int(rates.argmin())
    hits = int(safe[:, step].sum())
    lower = (
        beta_distribution.ppf(0.025, hits, samples - hits + 1) if hits else 0.0
    )
    upper = (
        beta_distribution.ppf(0.975, hits + 1, samples - hits)
        if hits < samples
        else 1.0
    )
    return rates.cpu().tolist(), float(rates[step]), (lower, upper)


def validate(s, rows):
    """Choose each method's route and Monte Carlo check the chosen plan.

    Returns:
        ({method: chosen row or None}, {method: Monte Carlo summary}).
    """
    mc = s.cfg.get("monte_carlo", {"samples": 5000, "seed": 0})
    choices, safety = {}, {}
    for method in METHODS:
        chosen = choices[method] = choose(rows, method, s.routes)
        if chosen is None:
            print(
                f"level {s.level:.4f} {LABELS[method]:>16}: no certified "
                "route",
                flush=True,
            )
            continue
        rates, minimum, interval = stepwise_safety(
            s, chosen["controls"], mc["samples"], mc["seed"]
        )
        whole = monte_carlo(
            s, SimpleNamespace(controls=chosen["controls"]), **mc
        )
        safety[method] = {
            "rates": rates,
            "min": minimum,
            "min_interval": [float(v) for v in interval],
            "whole_spec": whole[0],
        }
        print(
            f"level {s.level:.4f} {LABELS[method]:>16}: takes the "
            f"{chosen['start']} route (crosses {chosen['crossing']}); "
            f"min-step MC safety {minimum:.4f} "
            f"[{interval[0]:.4f}, {interval[1]:.4f}], whole-spec MC "
            f"{whole[0]:.4f}",
            flush=True,
        )
    return choices, safety


def band(s):
    """Noise levels where tightened STL accepts the first route's passage
    and pdSTL must not.

    Uses the first route's warm start: the spread there is level * c(k),
    independent of the controls, for a mean centred in the passage.

    Returns:
        (low, high): pdSTL needs z >= Phi^-1((1+alpha)/2), tightened STL
        z >= Phi^-1(alpha), with z = (w/2) / (level c).
    """
    route = s.routes[0]
    beliefs = s.planner.evaluate_controls(
        s.rollout, s.starts[route], spec=s.spec
    ).rollout.belief_trajectory
    crossed, sigma = crossing_of(s, beliefs)
    if crossed != route:
        raise ValueError(f"the {route!r} warm start must cross its passage")
    via = s.cfg["routes"][route]["via"]
    side = ("x_range", "y_range")[lateral_dim(via)]
    half = (via[side][1] - via[side][0]) / 2
    c = sigma / s.level
    normal = NormalDist()
    return (
        half / (normal.inv_cdf((1 + s.alpha) / 2) * c),
        half / (normal.inv_cdf(s.alpha) * c),
    )


# Entry points


def run(
    config_path,
    *,
    live=True,
    optimization_every=5,
    show=True,
    save=True,
    device="cpu",
):
    """Plan the config's noise level with all three methods, like the other
    examples: live optimization, route choice, Monte Carlo, and the map.

    Args:
        config_path: Scenario YAML.
        live, optimization_every: Stream sampled iterates to a live view.
        show, save: Display and/or save the map figure.
        device: Torch device.

    Returns:
        (rows, choices): every plan, and {method: chosen row or None}.
    """
    s = build(load_config(config_path), device)
    low, high = band(s)
    print(
        f"{s.cfg['name']}: noise {s.level:.4f}; tightened STL and pdSTL "
        f"disagree for noise in ({low:.4f}, {high:.4f})",
        flush=True,
    )
    rows = plan_level(s, live=live, callback_every=optimization_every)
    choices, _ = validate(s, rows)
    if show or save:
        plot_map(s, choices, show=show, save=save)
    return rows, choices


def sweep(
    config_path, *, levels=None, workers=None, max_iters=None, show=False
):
    """Plan every noise level in parallel; save the tables and figures.

    Args:
        config_path: Scenario YAML.
        levels: Noise levels; defaults to sweep.levels.
        workers: CPU processes; defaults to sweep.workers (1 = in process).
        max_iters: Optional iteration budget for every planner.
        show: Show figures before saving.

    Returns:
        {level: (rows, choices, safety)}.
    """
    cfg = load_config(config_path)
    if max_iters is not None:
        cfg["planner"] = {**cfg["planner"], "max_iters": max_iters}
    levels = list(cfg["sweep"]["levels"] if levels is None else levels)
    workers = workers or cfg["sweep"].get("workers", 1)
    jobs = [with_noise(cfg, level) for level in levels]
    print(
        f"{cfg['name']}: {len(levels)} noise levels x {len(METHODS)} "
        f"methods x {len(cfg['routes'])} warm starts, "
        f"{cfg['planner']['max_iters']} iterations each, on "
        f"{min(workers, len(levels))} CPU process(es).",
        flush=True,
    )
    if workers > 1:
        # Fork: workers inherit the loaded modules without re-running main.
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            min(workers, len(levels)), mp_context=context
        ) as pool:
            outcomes = list(pool.map(_sweep_level, jobs))
    else:
        outcomes = [_sweep_level(job) for job in jobs]
    results = dict(zip(levels, outcomes))
    q_band = band(build(jobs[0]))
    print(f"disagreement band: noise in ({q_band[0]:.4f}, {q_band[1]:.4f})")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _save(results)
    plot_routes(results, q_band, show=show)
    plot_safety(results, q_band, cfg["planner"]["alpha"], show=show)
    target = cfg.get("visualization", {}).get("map_level", sum(q_band) / 2)
    level = min(levels, key=lambda q: abs(q - target))
    plot_map(
        build(with_noise(cfg, level)),
        results[level][1],
        show=show,
        name="sweep_map",
    )
    return results


def _sweep_level(cfg):
    torch.set_num_threads(1)  # one core per process
    s = build(cfg)
    rows = plan_level(s)
    choices, safety = validate(s, rows)
    return rows, choices, safety


def _live_view(s, method, start):
    from visualization.live_plots import create_reach_avoid_live_view

    fig, _, on_iteration, finish = create_reach_avoid_live_view(
        s.env,
        lambda controls: s.planner.evaluate_controls(
            s.rollout, controls, spec=s.spec
        ),
        dt=s.cfg["dt"],
        title=f"{LABELS[method]}, {start} start",
        max_iters=s.planner.cfg["max_iters"],
        ellipse_every=s.cfg.get("visualization", {}).get("ellipse_every", 8),
        alpha=s.alpha,
    )
    return on_iteration, finish, fig


def _describe(row):
    return (
        f"level {row['level']:.4f} {LABELS[row['method']]:>16} from "
        f"{row['start']:>6}: crosses {row['crossing']:>6}, score "
        f"{row['score']:+.4f} "
        f"({'certified' if row['certified'] else 'not certified'}); pdSTL "
        f"lower {row['pdstl_lower']:.4f}, tight rho {row['tight_rho']:+.3f}, "
        f"mean rho {row['mean_rho']:+.3f}, {row['planning_time_s']:.1f} s"
    )


def _save(results):
    with open(OUTPUT_DIR / "runs.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(COLUMNS))
        writer.writeheader()
        for rows, _, _ in results.values():
            writer.writerows(
                {key: row[key] for key in COLUMNS} for row in rows
            )
    with open(OUTPUT_DIR / "choices.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "level",
                "method",
                "route",
                "min_step_safety",
                "min_step_ci_lower",
                "min_step_ci_upper",
                "whole_spec_mc",
            )
        )
        for level, (_, choices, safety) in results.items():
            for method in METHODS:
                check = safety.get(method, {})
                interval = check.get("min_interval", (None, None))
                writer.writerow(
                    (
                        level,
                        method,
                        _route(choices[method]),
                        check.get("min"),
                        *interval,
                        check.get("whole_spec"),
                    )
                )


# Figures


def _route(chosen):
    return "none" if chosen is None else chosen["start"]


def _shade(ax, q_band):
    ax.axvspan(*q_band, color="tab:red", alpha=0.10, label="Disagreement band")


def plot_routes(results, q_band, *, show=False):
    """Route each method takes at every noise level."""
    from visualization.figures import _finish, _style

    routes = dict.fromkeys(
        row["start"] for rows, _, _ in results.values() for row in rows
    )
    markers = {**dict(zip(routes, "os^D")), "none": "x"}
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for row, method in enumerate(METHODS):
        for route, marker in markers.items():
            levels = [
                level
                for level, (_, choices, _) in results.items()
                if _route(choices[method]) == route
            ]
            ax.scatter(
                levels,
                [row] * len(levels),
                marker=marker,
                color=METHOD_COLORS[method],
                s=46,
            )
    _shade(ax, q_band)
    ax.set_yticks(range(len(METHODS)), [LABELS[m] for m in METHODS])
    ax.set_ylim(-0.6, len(METHODS) - 0.4)
    ax.set_xlabel("Position process noise q [m per step]")
    legend = ", ".join(f"{route} ({m})" for route, m in markers.items())
    ax.set_title(f"Route taken: {legend}", fontsize=10)
    _style(ax)
    fig.tight_layout()
    return _finish(fig, OUTPUT_DIR / "route_vs_noise.png", show)


def plot_safety(results, q_band, alpha, *, show=False):
    """Monte Carlo minimum per-step safety of each method's chosen plan."""
    from visualization.figures import _finish, _style

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    plotted = [alpha]
    for method in METHODS:
        points = [
            (level, safety[method])
            for level, (_, _, safety) in results.items()
            if method in safety
        ]
        if not points:
            continue
        levels = [level for level, _ in points]
        rates = [check["min"] for _, check in points]
        errors = [
            [r - c["min_interval"][0] for r, (_, c) in zip(rates, points)],
            [c["min_interval"][1] - r for r, (_, c) in zip(rates, points)],
        ]
        ax.errorbar(
            levels,
            rates,
            yerr=errors,
            marker="o",
            markersize=4,
            capsize=2,
            color=METHOD_COLORS[method],
            label=LABELS[method],
        )
        plotted += rates
    ax.axhline(alpha, color="black", linestyle="--", linewidth=1, label="α")
    _shade(ax, q_band)
    ax.set_xlabel("Position process noise q [m per step]")
    ax.set_ylabel("min over t of MC P(safe at t)")
    ax.legend(fontsize=8)
    _style(ax, probability=True)
    ax.set_ylim(max(0.0, min(plotted) - 0.02), 1.005)
    fig.tight_layout()
    return _finish(fig, OUTPUT_DIR / "stepwise_safety_vs_noise.png", show)


def plot_map(s, choices, *, show=False, save=True, name="map"):
    """Each method's chosen plan, with belief ellipses, at one noise level."""
    from visualization.figures import (
        _draw_environment,
        _ellipse,
        _ellipse_indices,
        _finish,
    )

    fig, axes = plt.subplots(1, len(METHODS), figsize=(13.5, 5.0))
    every = s.cfg.get("visualization", {}).get("ellipse_every", 10)
    for ax, method in zip(axes, METHODS):
        _draw_environment(ax, s.env)
        chosen = choices[method]
        title = f"{LABELS[method]}: "
        if chosen is None:
            title += "no certified route"
        else:
            trace = s.planner.evaluate_controls(
                s.rollout, chosen["controls"].to(s.dyn.device), spec=s.spec
            ).rollout.belief_trajectory.trace
            mean = trace.mean[0].detach().cpu().numpy()
            covariance = trace.covariance[0].detach().cpu().numpy()
            for index in _ellipse_indices(mean, every):
                _ellipse(ax, mean[index], covariance[index])
            ax.plot(mean[:, 0], mean[:, 1], color=METHOD_COLORS[method])
            title += chosen["start"]
        ax.plot(*s.cfg["x0_mean"][:2], "o", color="black")
        ax.set_title(title)
    fig.suptitle(f"Chosen plans at q = {s.level:.3f} m per step")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _finish(fig, OUTPUT_DIR / f"{name}.png" if save else None, show)


if __name__ == "__main__":
    sweep("configs/scenarios/reach_avoid/inspection.yaml")
