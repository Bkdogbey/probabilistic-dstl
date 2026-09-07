"""Five pdSTL example cases driven through one shared runner.

Each case runs the same pipeline: predicted Gaussian beliefs -> atomic event
probabilities -> Boolean/temporal intervals -> optimisation -> plot. The
temporal output is a stochastic robustness interval; it is not a
whole-trajectory satisfaction probability. Only the Gaussian atom values are
probabilities.
"""

import os

import torch

from pdstl.base import BeliefTrajectory
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
from planning.planner import Planner
from models.beliefs import GaussianBelief
from utils import get_device, load_config
from visualization.robustness import plot_case, plot_synthesis

OUTPUT_DIR = "outputs"


# --- Case specifications ---------------------------------------------------
# Each builder returns (formula, atoms, thresholds, description). `atoms` are
# plotted as atomic event probabilities; `thresholds` are drawn on the state.


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
    cfg = load_config("configs/scenarios/examples.yaml")
    planner_cfg = {**load_config("configs/planning.yaml"), **cfg.get("planner", {})}
    return cfg, planner_cfg


def build_case(name, cfg):
    """Formula, atoms, thresholds and description for a named case."""
    if name not in CASES:
        raise ValueError(f"unknown case {name!r}; known: {sorted(CASES)}")
    return CASES[name](cfg["cases"][name], cfg["dim"])


def belief_trajectory(mean_trace, cov_trace):
    """Wrap a predicted mean/covariance rollout as one belief per step."""
    return BeliefTrajectory(
        [
            GaussianBelief(mean_trace[:, t, :], cov_trace[:, t])
            for t in range(mean_trace.shape[1])
        ]
    )


def rollout(dyn, v_params, x0_mean, x0_cov):
    """Predicted beliefs for a control sequence."""
    mean_trace, cov_trace = dyn(v_params, x0_mean, x0_cov)
    return belief_trajectory(mean_trace, cov_trace), mean_trace, cov_trace


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
    """Evaluate, optimise, re-evaluate and plot one case.

    Returns a dict with the initial and final directly-evaluated intervals, the
    returned controls, and the objective history.
    """
    cfg, planner_cfg = load_examples_config()
    device = get_device()
    dim, H, dt = cfg["dim"], cfg["H"], cfg["dt"]

    formula, atoms, thresholds, description = build_case(name, cfg)
    dyn = SingleIntegrator(
        dt=dt, u_max=cfg["u_max"], q_std=cfg["q_std"], device=device
    )
    x0_mean, x0_cov = _initial_state(cfg, device)
    time = [t * dt for t in range(H + 1)]

    # 1. Evaluate a fixed belief trajectory directly.
    v_init = _initial_controls(cfg, device)
    traj_init, mean_init, cov_init = rollout(dyn, v_init, x0_mean, x0_cov)
    interval_init = evaluate_direct(formula, traj_init)

    # 2. Optimise through the same pipeline.
    torch.manual_seed(cfg["seed"])
    planner = Planner(dyn, None, H, config=planner_cfg)
    best_mean, _, best_u, best_r, history = planner._optimize_window(
        x0_mean, x0_cov, spec=formula, init_guess=v_init, verbose=verbose
    )

    # 3. Re-evaluate the returned controls directly and confirm consistency.
    #    best_u are physical controls, so invert the tanh squashing to recover
    #    the optimisation variable the rollout expects.
    v_best = torch.atanh(torch.clamp(best_u / dyn.u_max, -0.999999, 0.999999))
    traj_final, mean_final, cov_final = rollout(dyn, v_best, x0_mean, x0_cov)
    interval_final = evaluate_direct(formula, traj_final)

    mean_mismatch = (mean_final - best_mean).abs().max().item()

    result = {
        "case": name,
        "description": description,
        "interval_initial": interval_init.detach()[0, 0].tolist(),
        "interval_final": interval_final.detach()[0, 0].tolist(),
        "smooth_best": best_r,
        "controls": best_u.detach(),
        "history": history,
        "objective": planner.best_objective,
        "trace_length": interval_final.shape[1],
        "mean_mismatch": mean_mismatch,
    }

    if verbose:
        lo_i, hi_i = result["interval_initial"]
        lo_f, hi_f = result["interval_final"]
        log_utils._log.info(
            f"[{name}] {description}\n"
            f"    direct interval  initial [{lo_i:.4f}, {hi_i:.4f}]"
            f"  ->  final [{lo_f:.4f}, {hi_f:.4f}]\n"
            f"    smooth best {best_r:.4f} | objective {planner.best_objective:.4f} "
            f"| valid origins {result['trace_length']} | mean match {mean_mismatch:.2e}"
        )

    if save or show:
        atom_traces = {
            label: evaluate_direct(atom, traj_final) for label, atom in atoms.items()
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
    """Always including step 0, where the bottleneck is fixed at planning time.

    x(0) is the given initial belief, so no control can change P(x(0) >= c).
    Optimisation cannot improve this specification, and reporting it honestly
    is the point of the check.
    """
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

    traj_init, _, _ = rollout(dyn, v_init, x0_mean, x0_cov)
    before = evaluate_direct(formula, traj_init).detach()[0, 0].tolist()

    torch.manual_seed(cfg["seed"])
    planner = Planner(dyn, None, cfg["H"], config=planner_cfg)
    best_mean, best_cov, _, _, _ = planner._optimize_window(
        x0_mean, x0_cov, spec=formula, init_guess=v_init, verbose=False
    )
    after = evaluate_direct(
        formula, belief_trajectory(best_mean, best_cov)
    ).detach()[0, 0].tolist()

    if verbose:
        log_utils._log.info(
            f"[always@0] uncontrollable step-0 bottleneck: "
            f"[{before[0]:.4f}, {before[1]:.4f}] -> [{after[0]:.4f}, {after[1]:.4f}] "
            f"(optimisation cannot move a fixed initial belief)"
        )
    return {"before": before, "after": after}


# --- MPC consistency check -------------------------------------------------


def run_mpc_check(verbose=True):
    """Short receding-horizon run reusing one case.

    Simulation assumption, stated rather than estimated: at each replanning step
    the physical state transition is sampled, that state is observed in full,
    and the next controller belief is initialised there with zero covariance.
    Process uncertainty then grows forward from the observation. This is not a
    state estimator.

    The deadline is absolute. For Eventually(., [a, b]) fixed at the first plan,
    after k applied steps the remaining window is [max(0, a-k), b-k]; it is
    never restarted.

    Reported as an integration check -- not recursive feasibility, and not a
    closed-loop satisfaction probability.
    """
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

        # Belief for this window: observed state, zero covariance. The
        # degenerate Gaussian is handled by the atom's inclusive zero-variance
        # branch.
        belief_mean, belief_cov = true_state.clone(), zero_cov.clone()

        planner = Planner(dyn, None, horizon, config=planner_cfg)
        guess = warm_start[:horizon] if warm_start is not None else None
        if guess is not None and guess.shape[0] < horizon:
            guess = None
        best_mean, best_cov, best_u, best_r, _ = planner._optimize_window(
            belief_mean, belief_cov, spec=spec, init_guess=guess, verbose=False
        )
        # Report the directly evaluated interval; best_r is the smooth score,
        # which under scale > 0 is an approximation that can exceed 1.
        direct = evaluate_direct(
            spec, belief_trajectory(best_mean, best_cov)
        )[0, 0].tolist()

        # Apply only the first control; sample the physical transition.
        u0 = best_u[0]
        noise = torch.distributions.MultivariateNormal(
            torch.zeros_like(true_state), dyn.Q
        ).sample()
        next_true = true_state + u0 * dyn.dt + noise

        steps.append(
            {
                "step": k,
                "window": [a_k, b_k],
                "belief_mean": belief_mean.tolist(),
                "belief_cov_trace": float(torch.diagonal(belief_cov).sum()),
                "true_state": true_state.tolist(),
                "robustness_direct": direct,
                "robustness_smooth": best_r,
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
        warm_start = torch.cat([best_u[1:], best_u[-1:]], dim=0)

    if verbose:
        log_utils._log.info(
            f"[mpc] finished after {len(steps)} steps; deadline decremented "
            f"{b0} -> {steps[-1]['window'][1] if steps else b0}, never restarted"
        )
    return steps


def run_all(*, show=False, save=True, verbose=True):
    """Run the five cases plus the step-zero bottleneck report."""
    results = {name: run_case(name, show=show, save=save, verbose=verbose)
               for name in CASES}
    results["always@0"] = run_always_step_zero(verbose=verbose)
    return results
