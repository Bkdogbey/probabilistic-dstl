"""Small scalar demonstrations using the canonical planner and MPC loop."""

import torch

from pdstl.operators import Always, And, Eventually, Until
from pdstl.predicates import GreaterThan, LessThan
from models.rollouts import gaussian_rollout
from planning.runners import setup_problem, _output_path
from utils import load_config


def _always(cfg, dim):
    c = cfg["threshold"]
    atom = GreaterThan(c, dim=dim)
    return (
        Always(atom, interval=cfg["interval"]),
        {f"P(x >= {c})": atom},
        [c],
        f"Always[{cfg['interval'][0]},{cfg['interval'][1]}](x >= {c})",
    )


def _eventually(cfg, dim):
    c = cfg["threshold"]
    atom = GreaterThan(c, dim=dim)
    return (
        Eventually(atom, interval=cfg["interval"]),
        {f"P(x >= {c})": atom},
        [c],
        f"Eventually[{cfg['interval'][0]},{cfg['interval'][1]}](x >= {c})",
    )


def _corridor(cfg, dim):
    lo, hi = cfg["lower_threshold"], cfg["upper_threshold"]
    low_atom, high_atom = GreaterThan(lo, dim=dim), LessThan(hi, dim=dim)
    return (
        Always(And(low_atom, high_atom), interval=cfg["interval"]),
        {f"P(x >= {lo})": low_atom, f"P(x <= {hi})": high_atom},
        [lo, hi],
        (
            f"Always[{cfg['interval'][0]},{cfg['interval'][1]}]"
            f"(x >= {lo} and x <= {hi})"
        ),
    )


def _until(cfg, dim):
    cs, cg = cfg["safe_threshold"], cfg["goal_threshold"]
    safe, goal = LessThan(cs, dim=dim), GreaterThan(cg, dim=dim)
    return (
        Until(safe, goal, interval=cfg["interval"]),
        {f"P(x <= {cs})": safe, f"P(x >= {cg})": goal},
        [cg, cs],
        f"(x <= {cs}) Until[{cfg['interval'][0]},{cfg['interval'][1]}] (x >= {cg})",
    )


def _nested(cfg, dim):
    c = cfg["threshold"]
    atom = GreaterThan(c, dim=dim)
    inner, outer = cfg["inner_interval"], cfg["outer_interval"]
    return (
        Eventually(Always(atom, interval=inner), interval=outer),
        {f"P(x >= {c})": atom},
        [c],
        f"Eventually[{outer[0]},{outer[1]}](Always[{inner[0]},{inner[1]}](x >= {c}))",
    )


CASES = {
    "always": _always,
    "eventually": _eventually,
    "corridor": _corridor,
    "until": _until,
    "nested": _nested,
}



def build_case(name, cfg):
    if name not in CASES:
        raise ValueError(f"unknown case {name!r}; known: {sorted(CASES)}")
    return CASES[name](cfg["cases"][name], cfg["dim"])


def run_case(name, *, show=False, save=True, verbose=False):
    from visualization.robustness import plot_case

    cfg = load_config("configs/scenarios/planning_examples.yaml")
    s = setup_problem(cfg)
    formula, atoms, thresholds, description = build_case(name, cfg)
    result = s.planner.optimize_window(s.rollout, spec=formula,
                                       init_guess=s.init_guess, verbose=verbose)
    if show or save:
        predicted = result.rollout
        plot_case(
            [k * cfg["dt"] for k in range(cfg["H"] + 1)],
            formula(predicted.belief_trajectory),
            mean_trace=predicted.aux["mean_trace"][0, :, cfg["dim"]],
            var_trace=predicted.aux["cov_trace"][0, :, cfg["dim"], cfg["dim"]],
            predicate_traces={label: atom(predicted.belief_trajectory)
                              for label, atom in atoms.items()},
            thresholds=thresholds, title=description,
            save_path=_output_path(f"{name}_case.png") if save else None, show=show,
        )
    return result


def run_end_to_end_reach(*, show=False, save=True, verbose=False):
    from visualization.robustness import plot_end_to_end_plans

    cfg = load_config("configs/examples.yaml")["end_to_end_reach"]
    s = setup_problem(cfg)
    atom = GreaterThan(cfg["threshold"], dim=cfg["dim"])
    spec = Eventually(atom, interval=[0, cfg["H"]])
    initial = (s.planner.evaluate_controls(
        s.rollout, torch.zeros(cfg["H"], s.planner.control_dim, device=s.dyn.device),
        spec=spec) if show or save else None)
    result = s.planner.optimize_window(s.rollout, spec=spec, verbose=verbose)
    if show or save:
        plot_end_to_end_plans(result, initial, atom, cfg, show=show,
                             save_path=_output_path("end_to_end_reach.png") if save else None)
    return result


def run_end_to_end_mpc_reach(*, show=False, save=True, verbose=False):
    from visualization.robustness import plot_mpc_reach

    cfg = load_config("configs/examples.yaml")["end_to_end_mpc_reach"]
    s = setup_problem(cfg)
    torch.manual_seed(cfg["seed"])
    spec = Eventually(GreaterThan(cfg["threshold"], dim=cfg["dim"]), interval=[0, cfg["H"]])
    result = s.planner.run_receding_horizon(
        s.state,
        make_rollout=lambda state, step: gaussian_rollout(s.dyn, *state),
        execute=lambda state, control, step: s.dyn.sample_step(*state, control),
        make_spec=lambda state, step: spec,
        is_done=lambda state, step: bool(state[0][cfg["dim"]] >= cfg["threshold"]),
        max_steps=cfg["max_steps"], init_guess=s.init_guess, verbose=verbose,
    )
    if show or save:
        plot_mpc_reach(cfg["dt"], result, cfg["threshold"], dim=cfg["dim"],
                       save_path=_output_path("end_to_end_mpc_reach.png") if save else None,
                       show=show)
    return result


def run_all(*, show=False, save=True, verbose=False):
    return {name: run_case(name, show=show, save=save, verbose=verbose) for name in CASES}
