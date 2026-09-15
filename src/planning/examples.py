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
    plot_mpc_reach,
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
    """Evaluate with scale <= 0: the reported interval, never the smooth score."""
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
    best, history = planner.optimize_window(
        rollout, spec=formula, init_guess=v_init, verbose=verbose
    )

    final = rollout(controls_to_params(dyn, best.controls))
    interval_final = evaluate_direct(formula, final.belief_trajectory)
    mean_final, cov_final = final.aux["mean_trace"], final.aux["cov_trace"]
    mean_mismatch = (mean_final - best.rollout.aux["mean_trace"]).abs().max().item()

    result = {
        "case": name,
        "description": description,
        "interval_initial": interval_init.detach()[0, 0].tolist(),
        "interval_final": interval_final.detach()[0, 0].tolist(),
        "smooth_score": best.smooth_score,
        "hard_score": best.hard_score,
        "controls": best.controls,
        "history": history,
        "objective": best.objective,
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
            f"    smooth {best.smooth_score:.4f} | objective {best.objective:.4f} "
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
    best, _ = planner.optimize_window(rollout, spec=formula, init_guess=v_init)
    after = list(best.hard_interval)

    if verbose:
        log_utils._log.info(
            f"[always@0] uncontrollable step-0 bottleneck: "
            f"[{before[0]:.4f}, {before[1]:.4f}] -> [{after[0]:.4f}, {after[1]:.4f}] "
            f"(optimisation cannot move a fixed initial belief)"
        )
    return {"before": before, "after": after}


# --- MPC consistency check -------------------------------------------------


def run_mpc_check(verbose=True):
    """Short MPC run: sampled steps, full-state observation with zero covariance, absolute deadline."""
    cfg, planner_cfg = load_examples_config()
    device = get_device()
    mpc_cfg = cfg["mpc"]
    a0, b0 = mpc_cfg["interval"]
    c = mpc_cfg["threshold"]
    dim = cfg["dim"]

    dyn = SingleIntegrator(
        dt=cfg["dt"], u_max=cfg["u_max"], q_std=cfg["q_std"], device=device
    )
    torch.manual_seed(cfg["seed"])

    # Physical state and controller belief are separate objects throughout.
    true_state = torch.tensor(cfg["x0_mean"], device=device, dtype=torch.float32)
    zero_cov = torch.zeros(len(cfg["x0_mean"]), len(cfg["x0_mean"]), device=device)

    steps = []
    warm_start = None

    for k in range(mpc_cfg["steps"]):
        a_k, b_k = max(0, a0 - k), b0 - k
        if b_k < 0:
            break  # the absolute deadline has passed

        # Horizon shrinks with the remaining obligation.
        horizon = max(1, b_k)
        spec = Eventually(GreaterThan(c, dim=dim), interval=[a_k, min(b_k, horizon)])

        # Belief for this window: the observed state with zero covariance.
        belief_mean, belief_cov = true_state.clone(), zero_cov.clone()

        planner = Planner(dyn, None, horizon, config=planner_cfg)
        guess = warm_start[:horizon] if warm_start is not None else None
        if guess is not None and guess.shape[0] < horizon:
            guess = None
        best, _ = planner.optimize_window(
            gaussian_rollout(dyn, belief_mean, belief_cov), spec=spec, init_guess=guess
        )
        direct = list(best.hard_interval)

        u0 = best.controls[0]
        next_true, _ = dyn.sample_step(true_state, zero_cov, u0)

        steps.append(
            {
                "step": k,
                "window": [a_k, b_k],
                "belief_mean": belief_mean.tolist(),
                "belief_cov_trace": float(torch.diagonal(belief_cov).sum()),
                "true_state": true_state.tolist(),
                "robustness_direct": direct,
                "robustness_smooth": best.smooth_score,
                "u0": u0.detach().tolist(),
                "satisfied": bool(true_state[dim] >= c),
            }
        )

        if verbose:
            log_utils._log.info(
                f"[mpc] step {k} | window [{a_k},{b_k}] | "
                f"true x {true_state[dim]:.3f} | belief x {belief_mean[dim]:.3f} "
                f"(cov {steps[-1]['belief_cov_trace']:.1e}) | "
                f"direct [{direct[0]:.4f}, {direct[1]:.4f}]"
                f"{'  <- target reached' if steps[-1]['satisfied'] else ''}"
            )

        true_state = next_true
        # Shift the warm start by the applied control.
        warm_start = planner._shift_controls(best.controls)

    if verbose:
        log_utils._log.info(
            f"[mpc] finished after {len(steps)} steps; deadline decremented "
            f"{b0} -> {steps[-1]['window'][1] if steps else b0}, never restarted"
        )
    return steps


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
    best, history = planner.optimize_window(rollout, spec=spec, init_guess=u_init, verbose=verbose)
    final = _evaluate_plan(rollout, controls_to_params(dyn, best.controls), predicate, spec)

    result = {
        "spec": str(spec),
        "score_initial": initial["interval"][0, 0, 0].item(),
        "score_final": final["interval"][0, 0, 0].item(),
        "interval_initial": initial["interval"][0, 0].tolist(),
        "interval_final": final["interval"][0, 0].tolist(),
        "controls": best.controls,
        "u_max": dyn.u_max,
        "mean_initial": initial["mean"],
        "mean_final": final["mean"],
        "cov_initial": initial["cov"],
        "cov_final": final["cov"],
        "atomic_initial": initial["atomic"],
        "atomic_final": final["atomic"],
        "history": history,
        "objective": best.objective,
    }

    if verbose:
        log_utils._log.info(
            f"[end_to_end_reach] {spec}\n"
            f"    pdSTL score  initial {result['score_initial']:.4f}"
            f"  ->  final {result['score_final']:.4f}\n"
            f"    final mean x {final['mean'][0, -1, dim].item():.3f} | "
            f"objective {best.objective:.4f} | iterations {len(history)}"
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


def run_end_to_end_mpc_reach(*, show=False, save=True, verbose=True):
    """Receding-horizon Eventually reach: plan, execute u_0, sample the step, replan."""
    cfg, planner_cfg = load_end_to_end_config("end_to_end_mpc_reach")
    device = get_device()
    H, dim, threshold = cfg["H"], cfg["dim"], cfg["threshold"]
    dyn, x0_mean, x0_cov, _, spec = end_to_end_setup(cfg, device)

    torch.manual_seed(cfg["seed"])  # the simulated process noise
    planner = Planner(dyn, None, H, config=planner_cfg)
    loop = planner.run_receding_horizon(
        (x0_mean, x0_cov),
        make_rollout=lambda state: gaussian_rollout(dyn, *state),
        execute=lambda state, u: dyn.sample_step(*state, u),
        is_done=lambda state: bool(state[0][dim] >= threshold),
        spec=spec,
        max_steps=cfg["max_steps"],
        init_guess=torch.tensor(cfg["init_control"], device=device).repeat(H, 1),
    )
    candidates = loop["candidates"]
    result = {
        "spec": str(spec),
        "u_max": dyn.u_max,
        "mean_trace": torch.stack([mean for mean, _ in loop["states"]]).unsqueeze(0),
        "cov_trace": torch.stack([cov for _, cov in loop["states"]]).unsqueeze(0),
        "u_trace": loop["u_trace"],
        "hard_scores": [c.hard_score for c in candidates],
        "objectives": [c.objective for c in candidates],
        "plan_mean_traces": [c.rollout.aux["mean_trace"] for c in candidates],
        "plan_controls": [c.controls for c in candidates],
        "warm_starts": loop["warm_starts"],
        "stopped_reason": loop["stopped_reason"],
    }

    if verbose:
        scores = result["hard_scores"]
        executed = result["mean_trace"][0, :, dim]
        log_utils._log.info(
            f"[end_to_end_mpc_reach] {spec} per window\n"
            f"    executed x  initial {executed[0].item():.3f}"
            f"  ->  final {executed[-1].item():.3f} | replans {len(scores)} | "
            f"stopped: {result['stopped_reason']}\n"
            f"    pdSTL score  first window {scores[0]:.4f}"
            f"  ->  last window {scores[-1]:.4f}"
        )

    if save or show:
        plot_mpc_reach(
            cfg["dt"],
            result,
            threshold,
            dim=dim,
            title=f"end_to_end_mpc_reach: {spec} per window",
            save_path=os.path.join(OUTPUT_DIR, "end_to_end_mpc_reach.png") if save else None,
            show=show,
        )

    return result


def run_all(*, show=False, save=True, verbose=True):
    """Run the five cases plus the step-zero bottleneck report."""
    results = {name: run_case(name, show=show, save=save, verbose=verbose)
               for name in CASES}
    results["always@0"] = run_always_step_zero(verbose=verbose)
    return results
