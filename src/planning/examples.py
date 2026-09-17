"""Scalar pdSTL cases (Always, Eventually, corridor, Until, nested) plus end-to-end reach demos."""

import os

import torch

from pdstl.operators import (
    Always,
    And,
    Eventually,
    GreaterThan,
    LessThan,
    Until,
)
from planning import log_utils
from models.dynamics import SingleIntegrator
from models.rollouts import gaussian_rollout
from planning.planner import Planner
from utils import get_device, load_config
from visualization.robustness import (
    plot_case,
    plot_end_to_end,
    plot_synthesis,
)

OUTPUT_DIR = "outputs"


# --- Case specifications: each returns (formula, atoms, thresholds, description) ---


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


# --- Shared machinery ------------------------------------------------------


def load_examples_config():
    """Example parameters merged with the planner defaults."""
    cfg = load_config("configs/scenarios/planning_examples.yaml")
    planner_cfg = {**load_config("configs/planning.yaml"), **cfg.get("planner", {})}
    return cfg, planner_cfg


def build_case(name, cfg):
    """Formula, atoms, thresholds and description for a named case."""
    if name not in CASES:
        raise ValueError(f"unknown case {name!r}; known: {sorted(CASES)}")
    return CASES[name](cfg["cases"][name], cfg["dim"])


def controls_to_params(dyn, controls):
    """Invert the tanh control bound, for replaying returned physical controls."""
    return torch.atanh(torch.clamp(controls / dyn.u_max, -0.999999, 0.999999))


def evaluate_direct(formula, traj):
    """Hard pdSTL interval (scale <= 0) for formal evaluation."""
    return formula(traj, scale=-1)


def _initial_controls(cfg, device):
    """The planner's own initial guess, made reproducible by the config seed."""
    torch.manual_seed(cfg["seed"])
    H = cfg["H"]
    return torch.randn(H, 2, device=device) * 0.1 + torch.tensor(
        [0.5, 0.0], device=device
    )


def _initial_state(cfg, device):
    x0_mean = torch.tensor(cfg["x0_mean"], device=device, dtype=torch.float32)
    x0_cov = torch.eye(len(cfg["x0_mean"]), device=device) * cfg["x0_cov_scale"]
    return x0_mean, x0_cov


def run_case(name, *, show=False, save=True, verbose=True):
    """Evaluate, optimise, re-evaluate and plot one case; returns intervals, controls and history."""
    cfg, planner_cfg = load_examples_config()
    device = get_device()
    dim, H, dt = cfg["dim"], cfg["H"], cfg["dt"]

    formula, atoms, thresholds, description = build_case(name, cfg)
    dyn = SingleIntegrator(
        dt=dt, u_max=cfg["u_max"], q_std=cfg["q_std"], device=device
    )
    x0_mean, x0_cov = _initial_state(cfg, device)
    time = [t * dt for t in range(H + 1)]

    rollout = gaussian_rollout(dyn, x0_mean, x0_cov)

    v_init = _initial_controls(cfg, device)
    initial = rollout(v_init)
    interval_init = evaluate_direct(formula, initial.belief_trajectory)
    mean_init, cov_init = initial.aux["mean_trace"], initial.aux["cov_trace"]

    torch.manual_seed(cfg["seed"])
    planner = Planner(dyn, None, H, config=planner_cfg)
    plan = planner.solve(x0_mean, x0_cov, initial_controls=v_init, specification=formula)
    history = [record.loss for record in plan.history]

    final = rollout(controls_to_params(dyn, plan.controls))
    interval_final = evaluate_direct(formula, final.belief_trajectory)
    mean_final, cov_final = final.aux["mean_trace"], final.aux["cov_trace"]
    mean_mismatch = (mean_final - plan.rollout.aux["mean_trace"]).abs().max().item()

    result = {
        "case": name,
        "description": description,
        "interval_initial": interval_init.detach()[0, 0].tolist(),
        "interval_final": interval_final.detach()[0, 0].tolist(),
        "smooth_lower": plan.history[-1].smooth_lower,
        "hard_lower": plan.hard_lower,
        "controls": plan.controls,
        "history": history,
        "objective": plan.history[-1].loss,
        "trace_length": interval_final.shape[1],
        "mean_mismatch": mean_mismatch,
    }

    if verbose:
        lo_i, hi_i = result["interval_initial"]
        lo_f, hi_f = result["interval_final"]
        log_utils._log.info(
            f"[{name}] {description}\n"
            f"    hard interval  initial [{lo_i:.4f}, {hi_i:.4f}]"
            f"  ->  final [{lo_f:.4f}, {hi_f:.4f}]\n"
            f"    smooth lower {plan.history[-1].smooth_lower:.4f} | objective {plan.history[-1].loss:.4f} "
            f"| valid origins {result['trace_length']} | mean match {mean_mismatch:.2e}"
        )

    if save or show:
        atom_traces = {
            label: evaluate_direct(atom, final.belief_trajectory) for label, atom in atoms.items()
        }
        base = os.path.join(OUTPUT_DIR, name)
        plot_case(
            time,
            interval_final.detach(),
            mean_trace=mean_final.detach()[0, :, dim],
            var_trace=cov_final.detach()[0, :, dim, dim],
            predicate_traces={k: v.detach() for k, v in atom_traces.items()},
            thresholds=thresholds,
            title=f"{name}: {description}",
            save_path=f"{base}_case.png" if save else None,
            show=show,
        )
        plot_synthesis(
            time,
            {
                "mean": mean_init.detach()[0, :, dim],
                "var": cov_init.detach()[0, :, dim, dim],
                "formula": interval_init.detach(),
            },
            {
                "mean": mean_final.detach()[0, :, dim],
                "var": cov_final.detach()[0, :, dim, dim],
                "formula": interval_final.detach(),
            },
            history,
            thresholds=thresholds,
            title=f"{name}: initial vs optimised",
            save_path=f"{base}_synthesis.png" if save else None,
            show=show,
        )

    return result


def run_always_step_zero(verbose=True):
    """Always including step 0: the fixed initial belief caps the score, so it cannot improve."""
    cfg, planner_cfg = load_examples_config()
    device = get_device()
    case_cfg = cfg["cases"]["always"]
    c = case_cfg["threshold"]
    formula = Always(
        GreaterThan(c, dim=cfg["dim"]), interval=[0, case_cfg["interval"][1]]
    )

    dyn = SingleIntegrator(
        dt=cfg["dt"], u_max=cfg["u_max"], q_std=cfg["q_std"], device=device
    )
    x0_mean, x0_cov = _initial_state(cfg, device)
    v_init = _initial_controls(cfg, device)

    rollout = gaussian_rollout(dyn, x0_mean, x0_cov)
    before = evaluate_direct(formula, rollout(v_init).belief_trajectory).detach()[0, 0].tolist()

    torch.manual_seed(cfg["seed"])
    planner = Planner(dyn, None, cfg["H"], config=planner_cfg)
    plan = planner.solve(x0_mean, x0_cov, initial_controls=v_init, specification=formula)
    after = plan.hard_interval.tolist()

    if verbose:
        log_utils._log.info(
            f"[always@0] uncontrollable step-0 bottleneck: "
            f"[{before[0]:.4f}, {before[1]:.4f}] -> [{after[0]:.4f}, {after[1]:.4f}] "
            f"(optimisation cannot move a fixed initial belief)"
        )
    return {"before": before, "after": after}


# --- End-to-end planning smoke test ----------------------------------------


def load_end_to_end_config(name="end_to_end_reach"):
    """Smoke-test parameters merged with the planner defaults."""
    cfg = load_config("configs/examples.yaml")[name]
    planner_cfg = {**load_config("configs/planning.yaml"), **cfg.get("planner", {})}
    return cfg, planner_cfg


def end_to_end_setup(cfg, device):
    """Dynamics, initial Gaussian state and specification for the smoke test."""
    dyn = SingleIntegrator(dt=cfg["dt"], u_max=cfg["u_max"], q_std=cfg["q_std"], device=device)
    x0_mean, x0_cov = _initial_state(cfg, device)
    predicate = GreaterThan(cfg["threshold"], dim=cfg["dim"])
    spec = Eventually(predicate, interval=[0, cfg["H"]])
    return dyn, x0_mean, x0_cov, predicate, spec


def _evaluate_plan(rollout, v, predicate, spec):
    """Mean, covariance, atomic trace and hard pdSTL interval for v."""
    predicted = rollout(v)
    return {
        "mean": predicted.aux["mean_trace"].detach(),
        "cov": predicted.aux["cov_trace"].detach(),
        "atomic": evaluate_direct(predicate, predicted.belief_trajectory).detach(),
        "interval": evaluate_direct(spec, predicted.belief_trajectory).detach(),
    }


def run_end_to_end_reach(*, show=False, save=True, verbose=True):
    """Optimise Eventually[0,H](x >= c) from zero controls with shaping off."""
    cfg, planner_cfg = load_end_to_end_config()
    device = get_device()
    H, dim = cfg["H"], cfg["dim"]
    dyn, x0_mean, x0_cov, predicate, spec = end_to_end_setup(cfg, device)

    rollout = gaussian_rollout(dyn, x0_mean, x0_cov)
    u_init = torch.zeros(H, 2, device=device)
    initial = _evaluate_plan(rollout, torch.zeros_like(u_init), predicate, spec)

    planner = Planner(dyn, None, H, config=planner_cfg)
    plan = planner.solve(x0_mean, x0_cov, initial_controls=u_init, specification=spec)
    history = [record.loss for record in plan.history]
    final = _evaluate_plan(rollout, controls_to_params(dyn, plan.controls), predicate, spec)

    result = {
        "spec": str(spec),
        "score_initial": initial["interval"][0, 0, 0].item(),
        "score_final": final["interval"][0, 0, 0].item(),
        "interval_initial": initial["interval"][0, 0].tolist(),
        "interval_final": final["interval"][0, 0].tolist(),
        "controls": plan.controls,
        "u_max": dyn.u_max,
        "mean_initial": initial["mean"],
        "mean_final": final["mean"],
        "cov_initial": initial["cov"],
        "cov_final": final["cov"],
        "atomic_initial": initial["atomic"],
        "atomic_final": final["atomic"],
        "history": history,
        "objective": plan.history[-1].loss,
    }

    if verbose:
        log_utils._log.info(
            f"[end_to_end_reach] {spec}\n"
            f"    pdSTL score  initial {result['score_initial']:.4f}"
            f"  ->  final {result['score_final']:.4f}\n"
            f"    final mean x {final['mean'][0, -1, dim].item():.3f} | "
            f"objective {plan.history[-1].loss:.4f} | iterations {len(history)}"
        )

    if save or show:
        time = [t * cfg["dt"] for t in range(H + 1)]
        plot_end_to_end(
            time,
            {
                "mean": initial["mean"][0, :, dim],
                "var": initial["cov"][0, :, dim, dim],
                "atomic": initial["atomic"],
                "score": result["score_initial"],
            },
            {
                "mean": final["mean"][0, :, dim],
                "var": final["cov"][0, :, dim, dim],
                "atomic": final["atomic"],
                "score": result["score_final"],
            },
            cfg["threshold"],
            title=f"end_to_end_reach: {spec}",
            save_path=os.path.join(OUTPUT_DIR, "end_to_end_reach.png") if save else None,
            show=show,
        )

    return result


def run_all(*, show=False, save=True, verbose=True):
    """Run the five cases plus the step-zero bottleneck report."""
    results = {name: run_case(name, show=show, save=save, verbose=verbose)
               for name in CASES}
    results["always@0"] = run_always_step_zero(verbose=verbose)
    return results


# --- Altitude safety: a 1-D monitoring/synthesis demo -------------------------
# Not the canonical path-planning path; it lives here with the other pdSTL demonstrations.


def _altitude_setup(config_path):
    cfg = load_config(config_path)
    planner_cfg = {**load_config("configs/planning.yaml"), **cfg.get("planner", {})}
    device = get_device()
    log_utils.log_device(device)
    dyn = SingleIntegrator(
        dt=cfg["dt"], u_max=cfg["u_max"], q_std=cfg["q_std"],
        device=device, state_dim=cfg.get("state_dim", 1),
    )
    x0_mean = torch.tensor(cfg["x0_mean"], device=device, dtype=torch.float32)
    x0_cov = torch.eye(len(cfg["x0_mean"]), device=device) * cfg["x0_cov_scale"]
    return cfg, planner_cfg, dyn, x0_mean, x0_cov, device


def _altitude_plan_summary(plan, atom, dyn, x0_mean, x0_cov):
    rollout = gaussian_rollout(dyn, x0_mean, x0_cov)
    predicted = rollout(controls_to_params(dyn, plan["controls"]))
    with torch.no_grad():
        return {
            "interval": list(plan["interval"]),
            "objective": plan["objective"],
            "atomic": evaluate_direct(atom, predicted.belief_trajectory).detach()[0],
            "mean": predicted.aux["mean_trace"].detach(),
            "cov": predicted.aux["cov_trace"].detach(),
            "controls": plan["controls"],
        }


def run_altitude_safety(
    config_path="configs/scenarios/altitude_safety.yaml", *, show=True, save=True
):
    """Always_[1,H](Z >= threshold) for a 1-D stochastic altitude."""
    from visualization.planning import plot_altitude_safety

    cfg, planner_cfg, dyn, x0_mean, x0_cov, device = _altitude_setup(config_path)
    H = cfg["H"]
    atom = GreaterThan(cfg["threshold"], dim=0)
    spec = Always(atom, interval=[1, H])
    u_init = torch.tensor(cfg["init_control"], device=device).repeat(H, 1)

    planner = Planner(dyn, None, H, config=_legacy_optimizer(planner_cfg))
    rollout = gaussian_rollout(dyn, x0_mean, x0_cov)

    initial_predicted = rollout(controls_to_params(dyn, u_init))
    initial_interval = evaluate_direct(spec, initial_predicted.belief_trajectory).detach()[0, 0]
    initial = {
        "interval": initial_interval.tolist(), "objective": float("nan"),
        "atomic": evaluate_direct(atom, initial_predicted.belief_trajectory).detach()[0],
        "mean": initial_predicted.aux["mean_trace"].detach(),
        "cov": initial_predicted.aux["cov_trace"].detach(), "controls": u_init,
    }

    frames = []
    plan = planner.solve(
        x0_mean, x0_cov, initial_controls=u_init, specification=spec,
        on_iteration=lambda record, controls: frames.append(
            _altitude_plan_summary(
                {"interval": [record.hard_lower, record.hard_lower],
                 "objective": record.loss, "controls": controls},
                atom, dyn, x0_mean, x0_cov,
            )
        ),
    )
    final = _altitude_plan_summary(
        {"interval": plan.hard_interval.tolist(), "objective": plan.history[-1].loss,
         "controls": plan.controls},
        atom, dyn, x0_mean, x0_cov,
    )
    result = {
        **{f"{key}_initial": value for key, value in initial.items()},
        **{f"{key}_final": value for key, value in final.items()},
        "stored_interval": plan.hard_interval.tolist(),
        "returned_iteration": plan.best_iteration,
        "controls": final["controls"],
        "history": [r.loss for r in plan.history],
        "iterations": len(plan.history),
        "frames": frames,
        "plan": plan,
    }
    log_utils._log.info(
        f"[altitude_safety] hard pdSTL interval "
        f"[{plan.hard_lower:.4f}, {plan.hard_upper:.4f}]"
        f" | initial [{initial_interval[0]:.4f}, {initial_interval[1]:.4f}]"
        f" | iterations {len(plan.history)}"
    )

    plot_args = {"dt": cfg["dt"], "threshold": cfg["threshold"], "u_max": dyn.u_max}
    if save or show:
        from planning.runners import output_path

        figure = output_path(cfg["figure"]) if save else None
        plot_altitude_safety(result, **plot_args, save_path=figure, show=show)
    return result


def _legacy_optimizer(planner_cfg):
    """Map the legacy planner block used by the demo configs onto PlanConfig's schema."""
    from planning.runners import LEGACY_LOSS_KEYS, LEGACY_OPTIMIZER_KEYS

    optimizer = {
        new: planner_cfg[old] for old, new in LEGACY_OPTIMIZER_KEYS.items() if old in planner_cfg
    }
    optimizer["loss"] = {
        new: planner_cfg[old] for old, new in LEGACY_LOSS_KEYS.items() if old in planner_cfg
    }
    optimizer["smoothing"] = dict(planner_cfg.get("smoothing") or {})
    return optimizer
