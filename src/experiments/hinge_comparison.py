"""Does the optional deterministic STL hinge help reach-avoid planning?

`run` plans each scenario under three planner variants (pdSTL alone, plus a
raw mean-robustness hinge, plus a z-score hinge), from the route-derived
controls (warm) and from zero controls (cold).

`ablation` diagnoses the cold start on the obstacle scenario (full formula
and goal-only formula, three hinge weights, hard and smooth auxiliary
robustness), then transfers the best full-formula setting of each hinge to
the other scenarios. It records per-iterate diagnostics through the planner's
observer, on a detached copy of the parameters, so the optimizer's updates
are unchanged.

Every run gets its own copy of the config, its own problem, and a clone of
the shared starting controls, so runs cannot affect one another. The selected
plan is still the iterate with the highest exact pdSTL lower score, and every
plan is validated with the same Monte Carlo samples and seed. Results go to a
new timestamped folder under outputs/hinge_comparison/. Hinge weights are
exploratory: raw and z-score robustness have different scales.
"""

import copy
import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch

from baselines.det_stl import DetPredicate, from_pdstl
from experiments import reach_avoid
from models.estimation import sample_gaussian
from pdstl.operators import And, Predicate
from planning.runners import RESULTS_DIR, monte_carlo
from utils import load_config

SCENARIOS = {
    name: f"configs/scenarios/reach_avoid/{name}.yaml"
    for name in ("obstacle", "narrow_passage", "either_or")
}
HINGE = {"margin_target": 0.0, "margin_beta": None}
HINGE_TYPES = {"hinge_mean": "none", "hinge_std": "std"}
PDSTL = {"w_margin": 0.0}
VARIANTS = {
    "pdstl": PDSTL,
    **{
        variant: {**HINGE, "w_margin": 0.1, "margin_normalization": norm}
        for variant, norm in HINGE_TYPES.items()
    },
}
INITIALIZATIONS = ("warm", "cold")
MONTE_CARLO = {"samples": 5000, "seed": 0}
OUTPUT_ROOT = RESULTS_DIR / "hinge_comparison"

FORMULAS = ("full", "goal_only")
SCREEN_WEIGHTS = (0.1, 1.0, 10.0)
SCREEN_BETAS = (None, 10.0)
NO_CONTROL_PENALTY = {"w_u": 0.0, "w_du": 0.0}
TRANSFER = (
    ("narrow_passage", "cold"),
    ("narrow_passage", "warm"),
    ("either_or", "cold"),
    ("either_or", "warm"),
    ("obstacle", "warm"),
)
DIAGNOSE_EVERY = 10


def run(
    scenarios=tuple(SCENARIOS),
    *,
    device="cpu",
    max_iters=None,
    monte_carlo_settings=MONTE_CARLO,
    output_root=OUTPUT_ROOT,
):
    """Plan every scenario x initialization x variant and save the results.

    Args:
        scenarios: Names from SCENARIOS.
        device: Torch device; CPU keeps runs deterministic and timings fair.
        max_iters: Optional iteration budget overriding every scenario's.
        monte_carlo_settings: Samples and seed shared by every validation.
        output_root: Parent of the new, timestamped results folder.

    Returns:
        (rows, folder): one summary dict per run, and where it was saved.
    """
    folder = _new_folder(output_root)
    context = _Context(device, max_iters, monte_carlo_settings)
    runs = [
        _execute(context, "compare", scenario, start, variant, overrides)
        for scenario in scenarios
        for start in INITIALIZATIONS
        for variant, overrides in VARIANTS.items()
    ]
    rows = _save(folder, runs, {"variants": VARIANTS, **context.settings()})
    plot_hard_lower(runs, save_path=folder / "hard_lower_history.png")
    return rows, folder


def ablation(
    previous=None,
    *,
    device="cpu",
    max_iters=None,
    monte_carlo_settings=MONTE_CARLO,
    output_root=OUTPUT_ROOT,
):
    """Obstacle cold-start screen, goal-only check, and transfer.

    Args:
        previous: Folder of an earlier `run`; runs whose full configs match
            are reused instead of re-planned, and screen runs that match are
            checked to reproduce its histories exactly.
        device, max_iters, monte_carlo_settings, output_root: As in `run`.

    Returns:
        (rows, folder, summary): the consolidated table, where it was saved,
        and the selected settings, checks, and lane-testing decision.
    """
    folder = _new_folder(output_root)
    context = _Context(device, max_iters, monte_carlo_settings, previous)
    runs = _screen(context)
    best_full = {v: _best(runs, v, "full") for v in HINGE_TYPES}
    best_goal = {v: _best(runs, v, "goal_only") for v in HINGE_TYPES}
    for variant, best in best_goal.items():
        if not best["meets_alpha"]:
            overrides = {**best["overrides"], **NO_CONTROL_PENALTY}
            runs.append(
                _execute(
                    context,
                    "goal_only_no_control_penalty",
                    "obstacle",
                    "cold",
                    variant,
                    overrides,
                    formula="goal_only",
                )
            )
    for scenario, start in TRANSFER:
        runs.append(_execute(context, "transfer", scenario, start, "pdstl"))
        for variant, best in best_full.items():
            runs.append(
                _execute(
                    context,
                    "transfer",
                    scenario,
                    start,
                    variant,
                    best["overrides"],
                )
            )
    summary = {
        **context.settings(),
        "best_full_formula": _describe(best_full),
        "best_goal_only": _describe(best_goal),
        "diagnostics_reproduce_previous": _reproduces_previous(
            runs, context.previous
        ),
        "lane_mpc": _lane_decision(runs),
    }
    rows = _save(folder, runs, summary)
    _plot_ablation(runs, folder)
    lane_settings = summary["lane_mpc"]["hinge_settings"]
    if lane_settings:
        lane_comparison(folder, lane_settings, device=device)
    return rows, folder, summary


LANE_SCENARIOS = {
    name: f"configs/scenarios/lane/{name}.yaml"
    for name in ("lane_change", "lane_merge")
}
LANE_SEEDS = (0, 1, 2)
LANE_COLUMNS = (
    "scenario",
    "seed",
    "variant",
    "w_margin",
    "margin_normalization",
    "margin_beta",
    "outcome",
    "steps",
    "first_window_hard_lower",
    "mean_window_hard_lower",
    "min_window_hard_lower",
    "windows_meeting_alpha",
    "planning_time_s",
    "control_cost",
)


def lane_comparison(
    folder, hinge_settings, *, seeds=LANE_SEEDS, device="cpu", max_steps=None
):
    """Closed-loop lane trials of pdSTL alone and each hinge setting.

    For every scenario and seed, the initial truth, process disturbances
    and observation noise are drawn once and shared by all settings; each
    trial starts from the configured warm controls and shifts them.

    Args:
        folder: Existing results folder; lane files are added to it.
        hinge_settings: {variant: planner overrides} to compare.
        seeds, device: Trial seeds and torch device.
        max_steps: Optional cap on executed steps.

    Returns:
        One summary dict per trial.
    """
    from experiments import lane

    runs = []
    for scenario, path in LANE_SCENARIOS.items():
        cfg = load_config(path)
        for seed in seeds:
            noise = _lane_noise(lane, cfg, seed, device, max_steps)
            for variant, overrides in {
                "pdstl": PDSTL,
                **hinge_settings,
            }.items():
                runs.append(
                    _lane_trial(
                        lane,
                        cfg,
                        (scenario, seed, variant, overrides),
                        noise,
                        device,
                        max_steps,
                    )
                )
    rows = [{key: run[key] for key in LANE_COLUMNS} for run in runs]
    with open(folder / "lane_results.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(LANE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    (folder / "lane_runs.json").write_text(json.dumps(runs, indent=2))
    return rows


def _lane_noise(lane, cfg, seed, device, max_steps):
    """Initial truth and every noise sequence of one seed, drawn once."""
    s = lane.build(copy.deepcopy(cfg), device)
    generator = torch.Generator(device=s.dyn.device).manual_seed(seed)
    initial_state = lane.initial_state_of(s, generator=generator)
    steps = cfg["T_SIM"] if max_steps is None else max_steps
    cars = len(cfg["traffic"])
    eye = torch.eye(4, dtype=s.dyn.B.dtype, device=s.dyn.device)

    def draw(shape, covariance):
        zeros = torch.zeros(*shape, 4, dtype=eye.dtype, device=eye.device)
        return sample_gaussian(
            zeros, covariance.expand(*shape, 4, 4), generator
        )

    return {
        "initial_state": initial_state,
        "ego_disturbances": draw((steps,), s.dyn.Q),
        "traffic_disturbances": draw(
            (steps, cars), cfg["traffic_q_std"] ** 2 * eye
        ),
        "ego_observation_noise": draw((steps,), s.ego_observation_cov),
        "traffic_observation_noise": draw(
            (steps, cars), s.traffic_observation_cov
        ),
    }


def _lane_trial(lane, cfg, labels, noise, device, max_steps):
    scenario, seed, variant, overrides = labels
    run_cfg = copy.deepcopy(cfg)
    run_cfg["planner"] = {**run_cfg.get("planner", {}), **overrides}
    s = lane.build(run_cfg, device)
    initial = noise["initial_state"]
    matched = {
        **{
            key: value.clone()
            for key, value in noise.items()
            if key != "initial_state"
        },
        "initial_state": type(initial)(
            *(v.clone() if torch.is_tensor(v) else v for v in initial)
        ),
    }
    result = lane.trial(s, max_steps=max_steps, **matched)
    plans = result.window_plans
    lowers = [plan.hard_interval[0] for plan in plans]
    cfg_planner = s.planner.cfg
    record = {
        "scenario": scenario,
        "seed": seed,
        "variant": variant,
        "w_margin": cfg_planner["w_margin"],
        "margin_normalization": (
            cfg_planner["margin_normalization"]
            if cfg_planner["w_margin"]
            else "n/a"
        ),
        "margin_beta": (
            cfg_planner["margin_beta"] if cfg_planner["w_margin"] else None
        ),
        "outcome": result.stopped_reason,
        "steps": len(plans),
        "first_window_hard_lower": lowers[0] if lowers else None,
        "mean_window_hard_lower": sum(lowers) / len(lowers)
        if lowers
        else None,
        "min_window_hard_lower": min(lowers) if lowers else None,
        "windows_meeting_alpha": (
            sum(bool(plan.threshold_met) for plan in plans) / len(plans)
            if plans
            else None
        ),
        "planning_time_s": sum(plan.planning_time for plan in plans),
        "control_cost": (
            s.planner._control_cost(result.applied_controls).item()
            if plans
            else 0.0
        ),
    }
    print(
        f"{scenario:>11} seed {seed} {variant:>10}: {record['outcome']} "
        f"after {record['steps']} steps, mean window rho_lower "
        f"{record['mean_window_hard_lower']}, "
        f"{record['planning_time_s']:.1f} s",
        flush=True,
    )
    return {
        **record,
        "overrides": overrides,
        "planner_config": cfg_planner,
        "scenario_config": run_cfg,
        "failure_detail": result.failure_detail,
        "hard_lower_by_window": lowers,
        "applied_controls": result.applied_controls.cpu().tolist(),
    }


class _Context:
    """Shared configs, starting controls and previous runs for one batch."""

    def __init__(self, device, max_iters, mc, previous=None):
        self.device, self.max_iters, self.mc = device, max_iters, mc
        self.previous_folder = None if previous is None else str(previous)
        self.previous = (
            []
            if previous is None
            else json.loads((Path(previous) / "runs.json").read_text())
        )
        self._configs, self._starts = {}, {}

    def config(self, scenario):
        if scenario not in self._configs:
            cfg = load_config(SCENARIOS[scenario])
            if self.max_iters is not None:
                cfg["planner"] = {
                    **cfg.get("planner", {}),
                    "max_iters": self.max_iters,
                }
            self._configs[scenario] = cfg
            route = reach_avoid.build(copy.deepcopy(cfg), self.device)
            self._starts[scenario] = {"warm": route.init_guess, "cold": None}
        return self._configs[scenario]

    def start(self, scenario, initialization):
        self.config(scenario)
        return self._starts[scenario][initialization]

    def settings(self):
        return {
            "initializations": {
                "warm": "route-derived controls (reach_avoid.build)",
                "cold": "init_guess=None: zero controls",
            },
            "monte_carlo": self.mc,
            "device": str(self.device),
            "selection": "highest exact pdSTL lower score over all iterates",
            "previous_folder": self.previous_folder,
        }


def _execute(
    context,
    stage,
    scenario,
    initialization,
    variant,
    overrides=PDSTL,
    *,
    formula="full",
):
    """Plan one setting on a private problem (or reuse an identical run)."""
    run_cfg = copy.deepcopy(context.config(scenario))
    run_cfg["planner"] = {**run_cfg.get("planner", {}), **overrides}
    problem = _problem(run_cfg, context, scenario, initialization, formula)
    labels = {
        "stage": stage,
        "scenario": scenario,
        "formula": formula,
        "initialization": initialization,
        "variant": variant,
        "overrides": overrides,
        "scenario_config": run_cfg,
        "planner_config": problem.planner.cfg,
    }
    reused = stage == "transfer" and _previous_match(context, labels)
    if reused:
        record = _reuse(problem, labels, reused, context)
    else:
        record = _plan(problem, labels, context.mc, stage != "compare")
    _print_row(record)
    return record


def _problem(run_cfg, context, scenario, initialization, formula):
    problem = reach_avoid.build(run_cfg, context.device)
    problem.full_spec = problem.spec
    problem.parts = _conjuncts(problem.spec)
    if formula == "goal_only":
        problem.spec = problem.parts[1]
    start = context.start(scenario, initialization)
    problem.init_guess = None if start is None else start.detach().clone()
    return problem


def _conjuncts(spec):
    """Parts of the left-nested conjunction: safety, goal, then visits."""
    if isinstance(spec, And):
        return [*_conjuncts(spec.subformula1), spec.subformula2]
    return [spec]


def _plan(problem, labels, mc, diagnose):
    tap = _RolloutTap(problem.rollout)
    diagnostics = _Diagnostics(problem, tap) if diagnose else None
    result = problem.planner.optimize_window(
        tap,
        spec=problem.spec,
        init_guess=problem.init_guess,
        on_iteration=diagnostics,
        callback_every=DIAGNOSE_EVERY,
    )
    result.monte_carlo = monte_carlo(problem, result, **mc)
    overhead = 0.0 if diagnostics is None else diagnostics.seconds
    return {
        **labels,
        **_summary(problem, result, mc),
        "planning_time_s": result.planning_time - overhead,
        "diagnostic_time_s": overhead,
        "source": "this batch",
        "hard_lower_history": result.hard_lower_history,
        "loss_history": result.loss_history,
        "diagnostics": [] if diagnostics is None else diagnostics.records,
    }


def _previous_match(context, labels):
    for record in context.previous:
        if (
            record.get("formula", "full") == labels["formula"]
            and record["initialization"] == labels["initialization"]
            and record["scenario_config"] == labels["scenario_config"]
            and record["planner_config"] == labels["planner_config"]
        ):
            return record
    return None


def _reuse(problem, labels, record, context):
    """Re-score an earlier run's saved controls; keep its histories."""
    controls = torch.tensor(
        record["selected_controls"], device=problem.dyn.device
    )
    result = problem.planner.evaluate_controls(
        problem.rollout, controls, spec=problem.spec
    )
    result.initial_hard_lower = record["initial_hard_lower"]
    result.hard_lower_history = record["hard_lower_history"]
    result.selected_iteration = record["selected_iteration"]
    result.monte_carlo = (
        record["mc_rate"],
        record["mc_ci_lower"],
        record["mc_ci_upper"],
    )
    return {
        **labels,
        **_summary(problem, result, context.mc),
        "planning_time_s": record["planning_time_s"],
        "diagnostic_time_s": 0.0,
        "source": f"reused from {context.previous_folder}",
        "hard_lower_history": record["hard_lower_history"],
        "loss_history": record["loss_history"],
        "diagnostics": [],
    }


def _summary(problem, result, mc):
    """Table columns for one selected plan."""
    rate, mc_lower, mc_upper = result.monte_carlo
    trajectory = result.rollout.belief_trajectory
    cfg = problem.planner.cfg
    row = {
        "w_margin": cfg["w_margin"],
        "margin_normalization": (
            cfg["margin_normalization"] if cfg["w_margin"] else "n/a"
        ),
        "margin_beta": cfg["margin_beta"] if cfg["w_margin"] else None,
        "control_penalty": bool(cfg["w_u"] or cfg["w_du"]),
        "alpha": cfg["alpha"],
        "initial_hard_lower": result.initial_hard_lower,
        "selected_hard_lower": result.hard_interval[0],
        "meets_alpha": result.threshold_met,
        "selected_iteration": result.selected_iteration,
        "control_cost": result.control_cost,
        **_plan_scores(problem, trajectory),
        "mc_rate": rate,
        "mc_ci_lower": mc_lower,
        "mc_ci_upper": mc_upper,
        "selected_controls": result.controls.cpu().tolist(),
    }
    if problem.spec is not problem.full_spec:
        full = SimpleNamespace(
            dyn=problem.dyn, state=problem.state, spec=problem.full_spec
        )
        row["mc_rate"], row["mc_ci_lower"], row["mc_ci_upper"] = monte_carlo(
            full, result, **mc
        )
        row["mc_goal_only"] = list(result.monte_carlo)
    return row


def _plan_scores(problem, trajectory):
    """Hard pdSTL and hard mean-STL scores of the full task and its parts."""
    with torch.no_grad():
        safety, goal = (
            from_pdstl(part).robustness(trajectory).item()
            for part in problem.parts[:2]
        )
        return {
            "full_hard_lower": problem.full_spec.probability_interval(
                trajectory
            )[0].item(),
            "goal_hard_lower": problem.parts[1]
            .probability_interval(trajectory)[0]
            .item(),
            "mean_stl_robustness": from_pdstl(problem.full_spec)
            .robustness(trajectory)
            .item(),
            "goal_rho_hard": goal,
            "safety_rho_hard": safety,
        }


class _RolloutTap:
    """Pass-through rollout that remembers the parameters it last saw."""

    def __init__(self, rollout):
        self.rollout, self.parameters = rollout, None

    def __call__(self, parameters):
        self.parameters = parameters
        return self.rollout(parameters)


class _Diagnostics:
    """Observer measuring an iterate on a detached copy of its parameters.

    The planner calls it after the forward pass and before the update, so
    the tapped parameters are exactly those the update will differentiate.
    Hard STL robustness is always reported apart from its smooth version.
    """

    def __init__(self, problem, tap):
        cfg = problem.planner.cfg
        self.problem, self.tap = problem, tap
        self.normalization = (
            cfg["margin_normalization"] if cfg["w_margin"] else "none"
        )
        self.auxiliary = problem.planner._auxiliary(problem.spec)
        self.records, self.seconds = [], 0.0

    def __call__(self, iteration, record):
        started = perf_counter()
        parameters = self.tap.parameters.detach().clone().requires_grad_()
        prediction = self.tap.rollout(parameters)
        trajectory = prediction.belief_trajectory
        norms, alignment = self._gradients(prediction, parameters, record)
        rho_hard, active = _active_predicates(
            self.problem, self._auxiliary_formula(), trajectory
        )
        beta = self.problem.planner.cfg["margin_beta"]
        with torch.no_grad():
            rho_smooth = (
                None
                if beta is None
                else self._auxiliary_formula()
                .robustness(trajectory, beta=beta)
                .item()
            )
            self.records.append(
                {
                    "iteration": iteration,
                    "beta": record.beta,
                    "hard_lower": record.hard_interval[0],
                    "smooth_lower": record.smooth_lower,
                    **_plan_scores(self.problem, trajectory),
                    "goal_atoms": _goal_atoms(self.problem, trajectory),
                    "auxiliary_rho_hard": rho_hard,
                    "auxiliary_rho_smooth": rho_smooth,
                    "active_predicates": active,
                    "gradient_norms": norms,
                    "alignment": alignment,
                }
            )
        self.seconds += perf_counter() - started

    def _auxiliary_formula(self):
        return from_pdstl(
            self.problem.spec,
            normalization=self.normalization,
            sigma_floor=self.problem.planner.cfg["margin_sigma_floor"],
        )

    def _gradients(self, prediction, parameters, record):
        """Weighted loss-term gradients with respect to the parameters."""
        planner = self.problem.planner
        trajectory = prediction.belief_trajectory
        controls = planner._applied_controls(prediction, parameters)
        terms = {
            "pdstl": -planner.cfg["w_phi"]
            * self.problem.spec.smooth_lower(trajectory, record.beta),
            "control": planner._control_cost(controls),
        }
        if self.auxiliary is not None:
            terms["hinge"] = planner.cfg["w_margin"] * (
                planner._margin_penalty(prediction, self.auxiliary)
            )
        gradients = {
            name: _gradient(term, parameters) for name, term in terms.items()
        }
        norms = {
            name: gradients[name].norm().item() if name in gradients else None
            for name in ("pdstl", "hinge", "control")
        }
        alignment = {
            "hinge_vs_control": _cosine(gradients, "hinge", "control"),
            "hinge_vs_pdstl": _cosine(gradients, "hinge", "pdstl"),
        }
        return norms, alignment


def _gradient(term, parameters):
    if not torch.is_tensor(term) or not term.requires_grad:
        return torch.zeros_like(parameters)
    (gradient,) = torch.autograd.grad(
        term, parameters, retain_graph=True, allow_unused=True
    )
    return torch.zeros_like(parameters) if gradient is None else gradient


def _cosine(gradients, first, second):
    if first not in gradients or second not in gradients:
        return None
    a, b = gradients[first].flatten(), gradients[second].flatten()
    if a.norm() == 0 or b.norm() == 0:
        return None
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _active_predicates(problem, formula, trajectory):
    """Hard robustness and the atoms/timesteps its min/max selects."""
    traces = []
    for module in formula.modules():
        if isinstance(module, DetPredicate):
            module.robustness_trace = _recording(module, traces)
    rho = formula.robustness(trajectory)
    traces = [(m, trace) for m, trace in traces if trace.requires_grad]
    gradients = torch.autograd.grad(
        rho.sum(), [trace for _, trace in traces], allow_unused=True
    )
    active = []
    for (module, trace), gradient in zip(traces, gradients):
        steps = (
            []
            if gradient is None
            else gradient[0, :, 0].nonzero().flatten().tolist()
        )
        if steps:
            active.append(
                {
                    "predicate": _label(problem.env, module),
                    "timesteps": steps,
                    "margin": trace[0, steps[0], 0].item(),
                }
            )
    return rho.item(), active


def _recording(module, traces):
    original = module.robustness_trace

    def robustness_trace(trajectory, beta=None):
        trace = original(trajectory, beta)
        traces.append((module, trace))
        return trace

    return robustness_trace


def _goal_atoms(problem, trajectory):
    """Largest probability of each goal atom over time, and when."""
    atoms = {}
    for module in problem.parts[1].modules():
        if isinstance(module, Predicate):
            probability = module(trajectory)[0, :, 0]
            step = int(probability.argmax())
            atoms[_label(problem.env, module)] = {
                "max_probability": probability[step].item(),
                "timestep": step,
            }
    return atoms


def _label(env, predicate):
    """Region and axis of an axis-interval atom, e.g. 'goal.x'."""
    if predicate.dim > 1:
        return predicate.name or f"x[{predicate.dim}]"
    axis = "xy"[predicate.dim]
    bounds = (predicate.lower, predicate.upper)
    names = [
        region.name
        for region in env.regions.values()
        if tuple(getattr(region, axis)) == bounds
    ]
    return f"{'/'.join(names) or '?'}.{axis}"


def _screen(context):
    """Obstacle from zero controls: pdSTL alone and every hinge setting."""
    runs = []
    for formula in FORMULAS:
        runs.append(
            _execute(
                context, "screen", "obstacle", "cold", "pdstl", formula=formula
            )
        )
        for variant, normalization in HINGE_TYPES.items():
            for weight in SCREEN_WEIGHTS:
                for beta in SCREEN_BETAS:
                    overrides = {
                        **HINGE,
                        "w_margin": weight,
                        "margin_normalization": normalization,
                        "margin_beta": beta,
                    }
                    runs.append(
                        _execute(
                            context,
                            "screen",
                            "obstacle",
                            "cold",
                            variant,
                            overrides,
                            formula=formula,
                        )
                    )
    return runs


def _best(runs, variant, formula):
    """Highest hard pdSTL score; hard mean-STL robustness breaks ties.

    The tie-break only picks which setting to carry forward; the planner's
    own candidate selection is unchanged.
    """
    tiebreak = (
        "goal_rho_hard" if formula == "goal_only" else ("mean_stl_robustness")
    )
    candidates = [
        run
        for run in runs
        if run["stage"] == "screen"
        and run["variant"] == variant
        and run["formula"] == formula
    ]
    return max(
        candidates, key=lambda r: (r["selected_hard_lower"], r[tiebreak])
    )


def _describe(best):
    return {
        variant: {
            "overrides": run["overrides"],
            "selected_hard_lower": run["selected_hard_lower"],
            "mean_stl_robustness": run["mean_stl_robustness"],
            "goal_rho_hard": run["goal_rho_hard"],
            "meets_alpha": run["meets_alpha"],
        }
        for variant, run in best.items()
    }


def _reproduces_previous(runs, previous):
    """Screen runs with an identical earlier run must match it exactly."""
    checks = []
    for run in runs:
        if run["stage"] != "screen" or run["formula"] != "full":
            continue
        for record in previous:
            if (
                record["initialization"] == run["initialization"]
                and record["scenario_config"] == run["scenario_config"]
                and record["planner_config"] == run["planner_config"]
            ):
                checks.append(
                    {
                        "variant": run["variant"],
                        "overrides": run["overrides"],
                        "identical_hard_lower_history": record[
                            "hard_lower_history"
                        ]
                        == run["hard_lower_history"],
                        "identical_loss_history": record["loss_history"]
                        == run["loss_history"],
                        "identical_controls": record["selected_controls"]
                        == run["selected_controls"],
                    }
                )
    return checks


def _lane_decision(runs):
    """Lane MPC is justified only if a hinge improves full-task success.

    Improvement means meeting alpha where pdSTL alone does not, or a Monte
    Carlo interval entirely above pdSTL alone's, in the same scenario and
    initialization.
    """
    baselines = {
        (run["scenario"], run["initialization"]): run
        for run in runs
        if run["variant"] == "pdstl" and run["formula"] == "full"
    }
    improvements = []
    for run in runs:
        baseline = baselines.get((run["scenario"], run["initialization"]))
        if (
            run["variant"] == "pdstl"
            or run["formula"] != "full"
            or baseline is None
        ):
            continue
        if (run["meets_alpha"] and not baseline["meets_alpha"]) or run[
            "mc_ci_lower"
        ] > baseline["mc_ci_upper"]:
            improvements.append(
                {
                    key: run[key]
                    for key in (
                        "scenario",
                        "initialization",
                        "variant",
                        "overrides",
                        "selected_hard_lower",
                        "mc_rate",
                    )
                }
            )
    settings = {}
    for improvement in improvements:
        settings.setdefault(improvement["variant"], improvement["overrides"])
    return {
        "run": bool(improvements),
        "improvements": improvements,
        "hinge_settings": settings,
        "note": (
            "a hinge improved full-formula satisfaction"
            if improvements
            else "skipped: no hinge setting improved full-formula "
            "satisfaction over pdSTL alone"
        ),
    }


COLUMNS = (
    "stage",
    "scenario",
    "formula",
    "initialization",
    "variant",
    "w_margin",
    "margin_normalization",
    "margin_beta",
    "control_penalty",
    "alpha",
    "initial_hard_lower",
    "selected_hard_lower",
    "meets_alpha",
    "selected_iteration",
    "full_hard_lower",
    "goal_hard_lower",
    "mean_stl_robustness",
    "goal_rho_hard",
    "safety_rho_hard",
    "planning_time_s",
    "control_cost",
    "mc_rate",
    "mc_ci_lower",
    "mc_ci_upper",
    "source",
)


def _print_row(row):
    print(
        f"{row['scenario']:>14} {row['initialization']:>4} "
        f"{row['formula']:>9} {row['variant']:>10} "
        f"w={row['w_margin']:<4} beta={row['margin_beta']}: "
        f"rho_lower {row['initial_hard_lower']:.3f} -> "
        f"{row['selected_hard_lower']:.3f}, goal rho "
        f"{row['goal_rho_hard']:+.2f}, safety rho "
        f"{row['safety_rho_hard']:+.2f}, MC {row['mc_rate']:.3f}, "
        f"{row['planning_time_s']:.1f} s",
        flush=True,
    )


def _new_folder(output_root):
    folder = Path(output_root) / datetime.now().strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def _save(folder, runs, settings):
    """Summary table (CSV, JSON), full per-run configs and histories."""
    rows = [{key: run[key] for key in COLUMNS} for run in runs]
    with open(folder / "results.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    (folder / "results.json").write_text(json.dumps(rows, indent=2))
    (folder / "settings.json").write_text(json.dumps(settings, indent=2))
    (folder / "runs.json").write_text(json.dumps(runs, indent=2))
    return rows


STYLES = {
    "pdstl": ("#2a78d6", "-", "pdSTL only"),
    "hinge_mean": ("#eb6834", "--", "+ mean STL hinge"),
    "hinge_std": ("#1baf7a", ":", "+ normalized hinge"),
}
WEIGHT_SHADES = {0.1: "#86b6ef", 1.0: "#2a78d6", 10.0: "#104281"}


def plot_hard_lower(runs, *, save_path=None, show=False):
    """Hard lower score vs iteration, one panel per scenario x start.

    Args:
        runs: Per-run records, with `hard_lower_history`.
        save_path: Saves a PNG there, and a PDF beside it.
        show: Display the figure.

    Returns:
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    from visualization.figures import RHO_LOWER

    scenarios = list(dict.fromkeys(run["scenario"] for run in runs))
    fig, axes = plt.subplots(
        len(scenarios),
        len(INITIALIZATIONS),
        figsize=(10, 3 * len(scenarios)),
        squeeze=False,
        sharey="row",
    )
    for run in runs:
        ax = axes[
            scenarios.index(run["scenario"]),
            INITIALIZATIONS.index(run["initialization"]),
        ]
        color, linestyle, label = STYLES[run["variant"]]
        history = run["hard_lower_history"]
        ax.plot(
            range(-1, len(history) - 1),
            history,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )
        ax.plot(
            run["selected_iteration"],
            run["selected_hard_lower"],
            "o",
            color=color,
            markersize=8,
        )
        if run["alpha"] is not None:
            ax.axhline(
                run["alpha"], color="0.45", linewidth=1, alpha=0.6, label="α"
            )
        ax.set_title(f"{run['scenario']} ({run['initialization']} start)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"Hard {RHO_LOWER}")
        ax.grid(color="tab:gray", alpha=0.16, linewidth=0.6)
    _legend(axes[0, 0], loc="lower right")
    return _finish(fig, save_path, show)


def _plot_ablation(runs, folder):
    screen = [run for run in runs if run["stage"] == "screen"]
    for formula in FORMULAS:
        plot_screen(
            [run for run in screen if run["formula"] == formula],
            title=f"Obstacle, cold start, {formula.replace('_', '-')} formula",
            save_path=folder / f"screen_{formula}.png",
        )
    plot_gradients(runs, save_path=folder / "screen_gradients.png")
    transfer = [
        run
        for run in runs
        if run["stage"] == "transfer"
        or (
            run["stage"] == "screen"
            and run["formula"] == "full"
            and (
                run["variant"] == "pdstl"
                or any(
                    t["variant"] == run["variant"]
                    and t["overrides"] == run["overrides"]
                    for t in runs
                    if t["stage"] == "transfer"
                )
            )
        )
    ]
    plot_hard_lower(transfer, save_path=folder / "transfer_hard_lower.png")


def plot_screen(runs, *, title, save_path=None, show=False):
    """Hard pdSTL score and hard goal/safety mean-STL robustness per iterate.

    Columns are hinge types; colour is the hinge weight, dashes mark smooth
    auxiliary robustness (margin_beta), and grey is pdSTL alone.
    """
    import matplotlib.pyplot as plt

    from visualization.figures import RHO_LOWER

    quantities = (
        ("hard_lower", f"Hard pdSTL {RHO_LOWER}"),
        ("goal_rho_hard", "Hard goal ρ (mean path)"),
        ("safety_rho_hard", "Hard safety ρ (mean path)"),
    )
    fig, axes = plt.subplots(
        len(quantities), len(HINGE_TYPES), figsize=(11, 9), sharex=True
    )
    for column, variant in enumerate(HINGE_TYPES):
        for row, (key, label) in enumerate(quantities):
            ax = axes[row, column]
            for run in runs:
                if run["variant"] not in ("pdstl", variant):
                    continue
                points = run["diagnostics"]
                ax.plot(
                    [p["iteration"] for p in points],
                    [p[key] for p in points],
                    **_screen_style(run),
                )
            ax.set_ylabel(label)
            ax.grid(color="tab:gray", alpha=0.16, linewidth=0.6)
            if row == 0:
                ax.set_title(STYLES[variant][2].lstrip("+ "))
            if row == len(quantities) - 1:
                ax.set_xlabel("Iteration")
        _legend(axes[0, column], fontsize=7, loc="upper left")
    fig.suptitle(title)
    return _finish(fig, save_path, show)


def _screen_style(run):
    if run["variant"] == "pdstl":
        return {"color": "0.45", "linewidth": 2, "label": "pdSTL only"}
    beta = run["margin_beta"]
    return {
        "color": WEIGHT_SHADES[run["w_margin"]],
        "linestyle": "-" if beta is None else "--",
        "linewidth": 2,
        "label": f"w={run['w_margin']:g}, "
        + ("hard" if beta is None else f"smooth β={beta:g}"),
    }


def plot_gradients(runs, *, save_path=None, show=False):
    """Weighted gradient norms and hinge alignment of the best screens."""
    import matplotlib.pyplot as plt

    chosen = [
        _best(runs, variant, formula)
        for formula in FORMULAS
        for variant in HINGE_TYPES
    ]
    fig, axes = plt.subplots(
        len(chosen), 2, figsize=(11, 3 * len(chosen)), sharex=True
    )
    colors = {"pdstl": "#2a78d6", "hinge": "#eb6834", "control": "#1baf7a"}
    for row, run in enumerate(chosen):
        points = run["diagnostics"]
        iterations = [p["iteration"] for p in points]
        for name, color in colors.items():
            norms = [p["gradient_norms"][name] for p in points]
            ax = axes[row, 0]
            ax.plot(
                iterations,
                [max(n or 0.0, 1e-12) for n in norms],
                color=color,
                linewidth=2,
                label=f"{name} term",
            )
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_ylabel("‖weighted ∇‖ (floor 1e-12 = 0)")
        for key, style in (("hinge_vs_control", "-"), ("hinge_vs_pdstl", ":")):
            values = [p["alignment"][key] for p in points]
            axes[row, 1].plot(
                iterations,
                [float("nan") if v is None else v for v in values],
                color="0.3",
                linestyle=style,
                linewidth=2,
                label=key.replace("_", " "),
            )
        axes[row, 1].set_ylim(-1.05, 1.05)
        axes[row, 1].set_ylabel("cosine")
        title = (
            f"{run['formula'].replace('_', '-')}, "
            f"{STYLES[run['variant']][2].lstrip('+ ')}: "
            f"w={run['w_margin']:g}, β={run['margin_beta']}"
        )
        for ax in axes[row]:
            ax.set_title(title, fontsize=9)
            ax.grid(color="tab:gray", alpha=0.16, linewidth=0.6)
    for ax in axes[-1]:
        ax.set_xlabel("Iteration")
    _legend(axes[0, 0], fontsize=8)
    _legend(axes[0, 1], fontsize=8)
    return _finish(fig, save_path, show)


def _legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, **kwargs)


def _finish(fig, save_path, show):
    import matplotlib.pyplot as plt

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        fig.savefig(Path(save_path).with_suffix(".pdf"))
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["ablation"]:
        ablation(*sys.argv[2:3])
    else:
        run()
