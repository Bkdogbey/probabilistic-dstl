"""2D pickup-scanner-delivery study: scenario, R1-R3 tasks, starts, methods, trials.

Numbers live in configs/scenarios/pickup_delivery/warehouse.yaml. Every task
interval is in control steps; the trace has H + 1 samples (step 0 is the
initial belief). Region geometry is the robot-centre set of a point robot.
"""

from functools import reduce
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import beta as beta_dist

from baselines.det_stl import from_pdstl
from experiments import task_checker
from experiments.reach_avoid import straight_line_guess
from models.rollouts import gaussian_rollout, sample_trajectories
from pdstl import operators as pd
from planning.environment import (
    Environment,
    RectangleRegion,
    inside,
    route_through,
    safety_event,
)
from planning.planner import Planner
from planning.runners import build_dynamics, build_initial_belief

PICKUPS = ("P1", "P2")
SPECS = ("R1", "R2", "R3")


# --- Instances -------------------------------------------------------------------


def _box(block):
    (xmin, xmax), (ymin, ymax) = block["x_range"], block["y_range"]
    return (float(xmin), float(xmax), float(ymin), float(ymax))


def _shift(box, offset):
    return (
        box[0] + offset[0],
        box[1] + offset[0],
        box[2] + offset[1],
        box[3] + offset[1],
    )


def _overlap(a, b):
    """Closed boxes share a point."""
    return a[0] <= b[1] and b[0] <= a[1] and a[2] <= b[3] and b[2] <= a[3]


def _contains(outer, inner):
    return (
        outer[0] <= inner[0]
        and inner[1] <= outer[1]
        and outer[2] <= inner[2]
        and inner[3] <= outer[3]
    )


def geometry_problems(cfg, start, regions):
    """Physical validity reasons (empty when valid); never uses method results."""
    workspace = _box(cfg["bounds"])
    obstacles = {o["name"]: _box(o) for o in cfg["obstacles"]}
    reasons = []
    names = list(regions)
    for i, a in enumerate(names):
        if not _contains(workspace, regions[a]):
            reasons.append(f"{a} leaves the workspace")
        reasons += [
            f"{a} overlaps {o}"
            for o, box in obstacles.items()
            if _overlap(regions[a], box)
        ]
        reasons += [
            f"{a} overlaps {b}"
            for b in names[i + 1 :]
            if _overlap(regions[a], regions[b])
        ]
    point = (start[0], start[0], start[1], start[1])
    if not _contains(workspace, point):
        reasons.append("start leaves the workspace")
    reasons += [
        f"start in {name}"
        for name, box in {**obstacles, **regions}.items()
        if _overlap(point, box)
    ]
    return reasons


def draw_instance(cfg, instance_id):
    """Seeded start and region jitter, redrawn until physically valid.

    Returns:
        (instance dict, list of rejected draws with reasons).
    """
    spec = cfg["instances"]
    rng = np.random.default_rng(spec["seed"] + instance_id)
    nominal = {name: _box(b) for name, b in cfg["regions"].items()}
    rejected = []
    for draw in range(spec["max_draws"]):
        jitter = rng.uniform(-1, 1, 2) * spec["start_jitter"]
        start = [float(v) for v in np.add(cfg["x0_mean"][:2], jitter)]
        regions = {
            name: _shift(box, rng.uniform(-1, 1, 2) * spec["region_jitter"])
            for name, box in nominal.items()
        }
        reasons = geometry_problems(cfg, start, regions)
        if not reasons:
            return {
                "id": instance_id,
                "draw": draw,
                "start": start,
                "regions": regions,
            }, rejected
        rejected.append({"id": instance_id, "draw": draw, "reasons": reasons})
    raise ValueError(f"instance {instance_id}: no valid draw; {rejected[-1]}")


# --- Scenario --------------------------------------------------------------------


def build_environment(cfg, instance):
    """Workspace, racks, pickups P1/P2, scanner C, and delivery D."""
    env = Environment()
    env.set_bounds(**cfg["bounds"])
    for obstacle in cfg["obstacles"]:
        env.add_obstacle(**obstacle)
    for name, (xmin, xmax, ymin, ymax) in instance["regions"].items():
        role = "goal" if name == "D" else "target"
        env.add_region(RectangleRegion(name, role, xmin, xmax, ymin, ymax))
    return env


def checker_task(cfg, instance):
    """Boxes for the independent checker."""
    regions = instance["regions"]
    return {
        "workspace": _box(cfg["bounds"]),
        "obstacles": [_box(o) for o in cfg["obstacles"]],
        "pickups": {name: regions[name] for name in PICKUPS},
        "scanner": regions["C"],
        "delivery": regions["D"],
    }


def resolved_config(cfg, instance, noise, horizon=None):
    """Scenario config for one instance and noise level."""
    return {
        **cfg,
        "H": int(horizon or cfg["H"]),
        "q_std": float(cfg["noise"][noise]),
        "x0_mean": [*instance["start"], *cfg["x0_mean"][2:]],
    }


def build(cfg, instance, noise, horizon=None, device="cpu"):
    """Dynamics, initial belief, Gaussian rollout, and environment."""
    cfg = resolved_config(cfg, instance, noise, horizon)
    dyn = build_dynamics(cfg, device)
    state = build_initial_belief(cfg, device)
    return SimpleNamespace(
        cfg=cfg,
        dyn=dyn,
        state=state,
        rollout=gaussian_rollout(dyn, *state),
        env=build_environment(cfg, instance),
        task=checker_task(cfg, instance),
        instance=instance,
    )


# --- Tasks -----------------------------------------------------------------------


def specification(env, horizon, dwell, spec_id):
    """R1-R3 of the study (handoff section 5).

    S = G[0,H](W and not O1 and not O2), L = G[0,d] P1 or G[0,d] P2.
    R1 = S and F[0,H] D.
    R2 = S and F[0,H-d] L and F[0,H] D.
    R3 = S and (not C U[0,H-d] L) and (not D U[0,H] C) and F[0,H] D.
    """
    safe = pd.Always(safety_event(env), [0, horizon])
    deliver = pd.Eventually(inside(env.region("D")), [0, horizon])
    load = reduce(
        pd.Or,
        [pd.Always(inside(env.region(p)), [0, dwell]) for p in PICKUPS],
    )
    if spec_id == "R1":
        return safe & deliver
    if spec_id == "R2":
        return safe & pd.Eventually(load, [0, horizon - dwell]) & deliver
    if spec_id == "R3":
        scanner = inside(env.region("C"))
        loaded = pd.Until(~scanner, load, [0, horizon - dwell])
        scanned = pd.Until(~inside(env.region("D")), scanner, [0, horizon])
        return safe & loaded & scanned & deliver
    raise ValueError(f"unknown spec {spec_id!r}; have {SPECS}")


def lookahead(formula):
    """Future steps one evaluation of `formula` needs."""
    if isinstance(formula, pd.Predicate):
        return 0
    if hasattr(formula, "equivalent"):  # rectangles, Implies
        return lookahead(formula.equivalent)
    if isinstance(formula, pd.Negation):
        return lookahead(formula.subformula)
    if isinstance(formula, (pd.And, pd.Or)):
        return max(map(lookahead, (formula.subformula1, formula.subformula2)))
    if isinstance(formula, pd.Until):
        children = max(map(lookahead, (formula.left, formula.right)))
        return formula.lookahead + children
    if isinstance(formula, pd.Temporal_Operator):
        return formula.lookahead + lookahead(formula.subformula)
    raise TypeError(f"unknown formula node {type(formula).__name__}")


def check_lookahead(formula, horizon):
    """Raise unless the formula needs no sample after step H."""
    need = lookahead(formula)
    if need > horizon:
        raise ValueError(f"formula looks {need} steps ahead; H = {horizon}")
    return need


# --- Shared start bank -----------------------------------------------------------


def _centre(box):
    return [(box[0] + box[1]) / 2, (box[2] + box[3]) / 2]


def corridor_use(cfg, xy):
    """[N] geometric route class from the classification corridors."""
    used = {
        name: task_checker.inside(xy, _box(block)).any(axis=1)
        for name, block in cfg["corridors"].items()
    }
    labels = np.full(len(xy), "neither", dtype=object)
    labels[used["slot"]] = "slot"
    labels[used["bypass"]] = "bypass"
    labels[used["slot"] & used["bypass"]] = "both"
    return labels


def mean_positions(s, controls):
    """[1, H + 1, 2] mean positions under physical controls [H, 2]."""
    state, states = s.state[0], [s.state[0]]
    for control in controls:
        state = s.dyn.A @ state + s.dyn.B @ control
        states.append(state)
    return torch.stack(states)[None, :, :2].detach().cpu().numpy()


def start_bank(s, spec_id="R3"):
    """Ordered pickup -> C -> D starts shared by every method.

    Returns:
        {name: SimpleNamespace(controls, pickup, route, nominal)}, where
        `nominal` is the independent checker's verdict on the mean path.
    """
    cfg, regions = s.cfg, s.instance["regions"]
    route = cfg.get("route", {})
    bank = {}
    for name, start in cfg["starts"].items():
        pickup = _centre(regions[start["pickup"]])
        to_pickup = route_through(
            s.env, s.instance["start"], [*start["before"], pickup], **route
        )
        onward = [*start["after"], _centre(regions["C"])]
        onward.append(_centre(regions["D"]))
        waypoints = to_pickup + route_through(s.env, pickup, onward, **route)
        controls = straight_line_guess(
            cfg,
            s.dyn,
            waypoints,
            holds={len(to_pickup) - 1: cfg["hold_steps"]},
            **cfg["init"],
        )
        xy = mean_positions(s, controls)
        verdict = task_checker.check(
            xy, s.task, cfg["H"], cfg["dwell"], spec_id
        )
        bank[name] = SimpleNamespace(
            controls=controls,
            pickup=start["pickup"],
            route=str(corridor_use(cfg, xy)[0]),
            nominal={k: v[0].item() for k, v in verdict.items()},
        )
    return bank


# --- Methods ---------------------------------------------------------------------


class RobustnessObjective:
    """Planner adapter: deterministic STL robustness in place of pdSTL scores.

    The planner ascends `smooth_lower` and selects on `probability_interval`;
    here both are the method's native robustness (metres) at a fixed beta, so
    nominal/margin STLCG reuse the same optimizer and selection policy.
    """

    def __init__(self, formula, beta):
        self.formula, self.beta = formula, beta

    def smooth_lower(self, beliefs, beta=None):
        return self.formula.robustness(beliefs, beta=self.beta)[0]

    def probability_interval(self, beliefs):
        rho = self.formula.robustness(beliefs, beta=None)[0]
        return torch.stack((rho, rho))


def objective_for(method, spec):
    """The score a method optimizes and selects on (its native criterion)."""
    if method["kind"] == "margin":
        formula = from_pdstl(spec, margin_k=method["k"])
        return RobustnessObjective(formula, method["beta"])
    return spec


def _iteration_seconds(s, objective, controls):
    """Median wall time of one planner iteration (smooth step + hard score)."""
    parameters = torch.atanh((controls / s.dyn.u_max).clamp(-0.999, 0.999))
    parameters.requires_grad_(True)
    times = []
    for _ in range(3):
        started = perf_counter()
        beliefs = s.rollout(parameters).belief_trajectory
        objective.smooth_lower(beliefs, 10.0).backward()
        with torch.no_grad():
            objective.probability_interval(beliefs)
        times.append(perf_counter() - started)
    return float(np.median(times))


def plan_gradient(s, method, objective, controls, budget, on_iteration=None):
    """Adam on the method's smooth objective; best native hard iterate wins.

    max_iters is calibrated so the beta anneal completes inside `budget`
    (the timing probe counts against it); the budget stays a hard cap.
    `on_iteration(iteration, record)` observes every iterate.
    """
    settings = dict(method.get("planner", {}))
    probe = perf_counter()
    per_iteration = _iteration_seconds(s, objective, controls)
    ceiling = settings.get("max_iters", 300)
    settings["max_iters"] = max(
        5, min(ceiling, int(0.9 * budget / per_iteration))
    )
    settings["time_budget"] = max(budget - (perf_counter() - probe), 1e-3)
    planner = Planner(s.dyn, s.cfg["H"], settings)
    result = planner.optimize_window(
        s.rollout,
        spec=objective,
        init_guess=controls,
        on_iteration=on_iteration,
    )
    return SimpleNamespace(
        controls=result.controls,
        native_hard=result.hard_interval[0],
        native_smooth=result.smooth_lower,
        native_initial=result.initial_hard_lower,
        hard_history=result.hard_lower_history,
        smooth_history=result.smooth_lower_history,
        elapsed=result.elapsed_history,
        selected_iteration=result.selected_iteration,
        stop_reason=result.stop_reason,
        evaluations=result.evaluations,
        max_iters=settings["max_iters"],
    )


def plan_cma(s, method, spec, controls, budget, seed):
    """CMA-ES on the same smooth pdSTL objective, parameterization and effort.

    Minimizes -smooth(beta) + control cost over the tanh parameters. The hard
    score is evaluated on each generation's best candidate; the best hard
    candidate (later wins ties, start included) is returned, as in Planner.
    """
    import cma

    planner = Planner(s.dyn, s.cfg["H"])
    x0 = planner._control_parameters(controls).flatten().double().numpy()
    shape = (s.cfg["H"], planner.control_dim)
    beta, started = method["beta"], perf_counter()

    def parameters_of(x):
        return torch.as_tensor(x, dtype=s.dyn.B.dtype).reshape(shape)

    def score(x):
        parameters = parameters_of(x)
        with torch.no_grad():
            beliefs = s.rollout(parameters).belief_trajectory
            smooth = spec.smooth_lower(beliefs, beta)
            loss = planner._objective(s.dyn.bound_control(parameters), smooth)
        return float(loss), float(smooth), beliefs

    def hard(beliefs):
        with torch.no_grad():
            return float(spec.probability_interval(beliefs)[0])

    _, smooth0, beliefs0 = score(x0)
    best = (hard(beliefs0), smooth0, -1, x0)
    hard_history, smooth_history, elapsed = [best[0]], [smooth0], [0.0]
    strategy = cma.CMAEvolutionStrategy(
        x0, method["sigma0"], {"seed": seed, "verbose": -9}
    )
    evaluations, generation, stop_reason = 1, 0, "time_budget"
    while perf_counter() - started < budget:
        if strategy.stop():
            stop_reason = "cma_stop"
            break
        candidates = strategy.ask()
        scored = [score(x) for x in candidates]
        evaluations += len(candidates)
        strategy.tell(candidates, [loss for loss, _, _ in scored])
        top = int(np.argmin([loss for loss, _, _ in scored]))
        value = hard(scored[top][2])
        if value >= best[0]:
            best = (value, scored[top][1], generation, candidates[top])
        hard_history.append(value)
        smooth_history.append(scored[top][1])
        elapsed.append(perf_counter() - started)
        generation += 1
    return SimpleNamespace(
        controls=s.dyn.bound_control(parameters_of(best[3])).detach(),
        native_hard=best[0],
        native_smooth=best[1],
        native_initial=hard_history[0],
        hard_history=hard_history,
        smooth_history=smooth_history,
        elapsed=elapsed,
        selected_iteration=best[2],
        stop_reason=stop_reason,
        evaluations=evaluations,
        max_iters=generation,
    )


# --- Trials ----------------------------------------------------------------------


def clopper_pearson(hits, n, level=0.95):
    """Exact binomial interval for hits / n."""
    tail = (1 - level) / 2
    lower = beta_dist.ppf(tail, hits, n - hits + 1) if hits else 0.0
    upper = beta_dist.ppf(1 - tail, hits + 1, n - hits) if hits < n else 1.0
    return float(lower), float(upper)


def execute(s, controls, samples, seed, spec_id):
    """Fresh open-loop rollouts of the stochastic dynamics, checked independently."""
    generator = torch.Generator().manual_seed(int(seed))
    paths = sample_trajectories(
        s.dyn, *s.state, controls.detach(), samples, generator
    )
    xy = paths[..., :2].cpu().numpy()
    verdict = task_checker.check(
        xy, s.task, s.cfg["H"], s.cfg["dwell"], spec_id
    )
    hits = int(verdict["success"].sum())
    routes = corridor_use(s.cfg, xy)
    bits = verdict["failures"]
    return {
        "samples": samples,
        "successes": hits,
        "rate": hits / samples,
        "ci95": clopper_pearson(hits, samples),
        "failure_counts": {
            name: int(((bits & task_checker.BIT[name]) > 0).sum())
            for name in task_checker.FAILURES
        },
        "primary_counts": {
            name: int((verdict["primary"] == name).sum())
            for name in task_checker.FAILURES
        },
        "pickup_counts": {
            name: int((verdict["pickup"] == name).sum()) for name in PICKUPS
        },
        "route_counts": {
            name: int((routes == name).sum())
            for name in ("slot", "bypass", "both", "neither")
        },
    }


def pdstl_scores(s, spec, controls):
    """Hard pdSTL interval and smooth lower (beta_end) of physical controls."""
    result = Planner(s.dyn, s.cfg["H"]).evaluate_controls(
        s.rollout, controls, spec=spec
    )
    lower, upper = result.hard_interval
    return {
        "hard_lower": lower,
        "hard_upper": upper,
        "smooth_lower": result.smooth_lower,
        "smooth_minus_hard": result.smooth_lower - lower,
        "beta": result.smoothing_beta,
    }, result


def plan_metrics(s, controls, spec_id):
    """Effort, path length, and the mean path's task timing and route."""
    xy = mean_positions(s, controls)
    verdict = task_checker.check(
        xy, s.task, s.cfg["H"], s.cfg["dwell"], spec_id
    )
    steps = np.diff(xy[0], axis=0)
    return {
        "effort": float((controls**2).sum() * s.cfg["dt"]),
        "path_length": float(np.linalg.norm(steps, axis=1).sum()),
        "mean_success": bool(verdict["success"][0]),
        "mean_failures": task_checker.failure_names(
            int(verdict["failures"][0])
        ),
        "pickup": str(verdict["pickup"][0]),
        "route": str(corridor_use(s.cfg, xy)[0]),
        **{
            key: int(verdict[key][0])
            for key in ("load_time", "scan_time", "delivery_time")
        },
    }


def _plan_start(s, method, objective, spec, start, budget, seed):
    if method["kind"] == "cma":
        return plan_cma(s, method, spec, start.controls, budget, seed)
    return plan_gradient(s, method, objective, start.controls, budget)


def run_trial(
    cfg,
    instance,
    noise,
    spec_id,
    method_name,
    *,
    budget,
    rollouts,
    eval_seed,
    plan_seed,
    horizon=None,
    starts=None,
):
    """Plan with one method from the shared start bank, then execute.

    Args:
        cfg: Scenario config.
        instance: From `draw_instance`.
        noise: Noise level name.
        spec_id: "R1", "R2" or "R3".
        method_name: Key of cfg["methods"].
        budget: Total optimization wall time (s), split evenly over starts.
        rollouts: Monte Carlo samples for execution.
        eval_seed, plan_seed: Execution and optimizer seeds (independent).
        horizon: Optional H override; starts: optional start-name subset.

    Returns:
        (record dict, artifact dict); the artifact is empty without a plan.
    """
    method = cfg["methods"][method_name]
    torch.manual_seed(plan_seed)
    started = perf_counter()
    s = build(cfg, instance, noise, horizon)
    spec = specification(s.env, s.cfg["H"], s.cfg["dwell"], spec_id)
    check_lookahead(spec, s.cfg["H"])
    bank = start_bank(s, spec_id)
    if starts is not None:
        bank = {name: bank[name] for name in starts}
    objective = objective_for(method, spec)
    # Untimed warm-up so first-call overhead is not charged to the first start.
    _iteration_seconds(s, objective, next(iter(bank.values())).controls)
    preprocessing = perf_counter() - started

    plans, errors = {}, {}
    for k, (name, start) in enumerate(bank.items()):
        try:
            plans[name] = _plan_start(
                s,
                method,
                objective,
                spec,
                start,
                budget / len(bank),
                plan_seed + k,
            )
        except (RuntimeError, ValueError) as error:
            errors[name] = f"{type(error).__name__}: {error}"

    record = {
        "instance": instance["id"],
        "noise": noise,
        "q_std": s.cfg["q_std"],
        "spec": spec_id,
        "method": method_name,
        "method_params": method,
        "H": s.cfg["H"],
        "dt": s.cfg["dt"],
        "dwell": s.cfg["dwell"],
        "budget_s": budget,
        "eval_seed": int(eval_seed),
        "plan_seed": int(plan_seed),
        "starts": list(bank),
        "start_routes": {n: b.route for n, b in bank.items()},
        "start_nominal_success": {
            n: bool(b.nominal["success"]) for n, b in bank.items()
        },
        "start_pdstl_hard": {
            n: pdstl_scores(s, spec, b.controls)[0]["hard_lower"]
            for n, b in bank.items()
        },
        "errors": errors,
        "preprocessing_s": preprocessing,
        "optimization_s": perf_counter() - started - preprocessing,
    }
    if not plans:
        record.update(
            status="optimization_error", total_s=perf_counter() - started
        )
        return record, {}

    # Across starts: highest native hard score; the smooth score at its
    # selection beta breaks exact ties (e.g. all zero), then the earlier start.
    chosen = max(
        plans, key=lambda n: (plans[n].native_hard, plans[n].native_smooth)
    )
    plan = plans[chosen]
    scores, evaluated = pdstl_scores(s, spec, plan.controls)
    record.update(
        status="returned",
        selected_start=chosen,
        selected_iteration=plan.selected_iteration,
        stop_reasons={n: p.stop_reason for n, p in plans.items()},
        iterations={n: p.max_iters for n, p in plans.items()},
        evaluations=sum(p.evaluations for p in plans.values()),
        native_initial={n: p.native_initial for n, p in plans.items()},
        native_selected={n: p.native_hard for n, p in plans.items()},
        initial_hard_lower=record["start_pdstl_hard"][chosen],
        pdstl=scores,
        plan=plan_metrics(s, plan.controls, spec_id),
        total_s=perf_counter() - started,
    )
    executed = perf_counter()
    record["execution"] = execute(
        s, plan.controls, rollouts, eval_seed, spec_id
    )
    record["execution_s"] = perf_counter() - executed
    artifact = {
        "controls": plan.controls.cpu(),
        "mean_trace": evaluated.rollout.aux["mean_trace"].cpu(),
        "cov_trace": evaluated.rollout.aux["cov_trace"].cpu(),
        "start_controls": {n: b.controls.cpu() for n, b in bank.items()},
        "histories": {
            n: {
                "hard": p.hard_history,
                "smooth": p.smooth_history,
                "elapsed": p.elapsed,
                "selected_iteration": p.selected_iteration,
            }
            for n, p in plans.items()
        },
    }
    return record, artifact
