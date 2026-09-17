"""Experiment runners: load a scenario, build the problem, call the Planner, plot."""

from functools import reduce
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from models.dynamics import DoubleIntegrator, SingleIntegrator
from models.rollouts import gaussian_rollout
from pdstl.operators import Always, And
from pdstl.predicates import GreaterThan, InsideRectangle, OutsideRectangle
from planning import log_utils
from planning.environment import RectangleRegion
from planning.scenarios.lane_merge import (
    MovingRectangleRegion,
    build_lane_merge_environment,
    clip_moving_obstacles,
)
from planning.scenarios.reach_avoid import build_reach_avoid_environment
from planning.planner import Planner
from utils import get_device, load_config
from visualization.animation import animate_altitude_optimization, animate_results
from visualization.live_plots import make_lane_change_live_callback, make_mpc_live_callback
from visualization.planning import (
    plot_altitude_safety,
    visualize_lane_change,
    visualize_reach_avoid,
    visualize_results,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "outputs"


# --- Scenario builders ---------------------------------------------------------


def load_scenario_config(cfg_path):
    """Scenario dict and planner config (scenario overrides on planning.yaml)."""
    cfg = load_config(cfg_path)
    return cfg, {**load_config("configs/planning.yaml"), **cfg.get("planner", {})}


def build_dynamics(cfg, device):
    common = {"dt": cfg["dt"], "u_max": cfg["u_max"], "q_std": cfg["q_std"], "device": device}
    if cfg.get("dynamics", "single_integrator") == "double_integrator":
        return DoubleIntegrator(**common)
    return SingleIntegrator(**common, state_dim=cfg.get("state_dim", 2))


def build_initial_belief(cfg, device):
    x0_mean = torch.tensor(cfg["x0_mean"], device=device)
    return x0_mean, torch.eye(len(x0_mean), device=device) * cfg["x0_cov_scale"]


def build_environment(cfg, device="cpu"):
    """Pick the scenario's builder from `scenario: {type: ...}`."""
    scenario = (cfg.get("scenario") or {}).get("type", "reach_avoid")
    if scenario == "lane_merge":
        return build_lane_merge_environment(cfg, device=device)
    if scenario == "reach_avoid":
        return build_reach_avoid_environment(cfg)
    raise ValueError(f"unknown scenario type {scenario!r}")


def _output_path(filename):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(RESULTS_DIR / filename)


# --- pdSTL demonstrations --------------------------------------------------------


def _setup(config_path, with_environment=False):
    """Shared single-shot setup: dynamics, Gaussian rollout from b_0, planner, initial guess."""
    device = get_device()
    log_utils.log_device(device)
    cfg, planner_cfg = load_scenario_config(config_path)
    dyn = build_dynamics(cfg, device)
    env = build_environment(cfg, device=device) if with_environment else None
    return SimpleNamespace(
        cfg=cfg,
        H=cfg["H"],
        dyn=dyn,
        env=env,
        rollout=gaussian_rollout(dyn, *build_initial_belief(cfg, device)),
        planner=Planner(dyn, env, cfg["H"], config=planner_cfg),
        u_init=torch.tensor(cfg["init_control"], device=device).repeat(cfg["H"], 1),
    )


def _log_intervals(name, spec, result):
    """Report the hard pdSTL outcome separately from differentiable semantics."""
    (lo_i, hi_i), (lo_f, hi_f) = result["interval_initial"], result["interval_final"]
    extra = ""
    if "smooth_lower" in result:
        extra = (
            f" | smooth lower {result['smooth_lower']:.4f}"
            f" (beta={result['smooth_beta']})"
            f" | control cost {result['control_cost']:.4f}"
        )
    log_utils._log.debug(f"[{name}] specification: {spec}")
    log_utils._log.info(
        f"[{name}] Hard pdSTL interval: [{lo_f:.4f}, {hi_f:.4f}]"
        f" | initial [{lo_i:.4f}, {hi_i:.4f}]"
        f"{extra} | iterations {result['iterations']}"
    )


def _event_trace(event, rollout):
    """[H+1, 2] probability interval of one event along the predicted beliefs."""
    return event(rollout.belief_trajectory).detach()[0]


def _plan_summary(candidate, atom):
    with torch.no_grad():
        return {
            "interval": list(candidate.hard_interval),
            "objective": candidate.objective,
            "atomic": _event_trace(atom, candidate.rollout),
            "mean": candidate.rollout.aux["mean_trace"],
            "cov": candidate.rollout.aux["cov_trace"],
            "controls": candidate.controls,
        }


def run_altitude_safety(config_path="configs/scenarios/altitude_safety.yaml", *, show=True, save=True):
    """Always_[1,H](Z >= threshold) for a 1-D stochastic altitude; every iterate kept as a frame."""
    s = _setup(config_path)
    atom = GreaterThan(s.cfg["threshold"], dim=0)
    spec = Always(atom, interval=[1, s.H])

    initial = _plan_summary(s.planner.evaluate_controls(s.rollout, s.u_init, spec=spec), atom)
    frames = []
    best, history = s.planner.optimize_window(
        s.rollout, spec=spec, init_guess=s.u_init, verbose=True,
        on_iteration=lambda k, candidate: frames.append(_plan_summary(candidate, atom)),
    )
    final = _plan_summary(s.planner.evaluate_controls(s.rollout, best.controls, spec=spec), atom)

    result = {
        **{f"{key}_initial": value for key, value in initial.items()},
        **{f"{key}_final": value for key, value in final.items()},
        "stored_interval": list(best.hard_interval),
        "returned_iteration": best.iteration,
        "controls": final["controls"],
        "history": history,
        "iterations": len(history),
        "frames": frames,
    }
    _log_intervals("altitude_safety", spec, result)

    plot_args = {"dt": s.cfg["dt"], "threshold": s.cfg["threshold"], "u_max": s.dyn.u_max}
    if save or show:
        figure = _output_path(s.cfg["figure"]) if save else None
        plot_altitude_safety(result, **plot_args, save_path=figure, show=show)
    if save:
        animation = s.cfg["animation"]
        animate_altitude_optimization(
            result, **plot_args, filename=_output_path(animation["filename"]), fps=animation["fps"]
        )
    return result


def _mean_clearance(mean_trace, obstacles):
    """Post-hoc diagnostic: smallest mean-to-rectangle distance over all obstacles."""
    points = mean_trace[0, 1:]
    corners = torch.tensor(
        [[o.x[0], o.y[0], o.x[1], o.y[1]] for o in obstacles],
        device=points.device,
        dtype=points.dtype,
    )
    gap = torch.maximum(corners[:, None, :2] - points, points - corners[:, None, 2:])
    outside = gap.clamp(min=0).norm(dim=2)
    return torch.where(outside > 0, outside, gap.max(dim=2).values).min().item()


def _obstacle_traces(events, rollout):
    """Per-obstacle probability intervals, for inspection alongside their conjunction."""
    return [_event_trace(event, rollout) for event in events["obstacles"]]


def reach_avoid_events(environment):
    """The named events a reach-avoid report inspects, rebuilt from the environment's regions."""
    workspace, goal = environment.region("workspace"), environment.region("goal")
    return {
        "workspace": InsideRectangle(workspace.x, workspace.y, name=workspace.name),
        "goal": InsideRectangle(goal.x, goal.y, name=goal.name),
        "obstacles": [
            OutsideRectangle(o.x, o.y, name=o.name)
            for o in environment.by_role("obstacle")
        ],
    }


def _iteration_record(candidate):
    """Scalar diagnostics only; retain no rollout or autograd graph."""
    return {
        "iteration": candidate.iteration,
        "beta": candidate.beta,
        "smooth_lower": candidate.smooth_lower,
        "hard_interval": candidate.hard_interval,
        "control_cost": candidate.control_cost,
        "objective": candidate.objective,
    }


def run_reach_avoid(config_path="configs/scenarios/reach_avoid.yaml", *, show=True, save=True):
    """Always(outside every obstacle) ∧ Eventually(inside goal) ∧ Always(inside workspace)."""
    s = _setup(config_path, with_environment=True)
    spec = s.env.get_specification(s.H)

    initial = s.planner.evaluate_controls(s.rollout, s.u_init, spec=spec)
    optimization_trace = []
    best, history = s.planner.optimize_window(
        s.rollout, spec=spec, init_guess=s.u_init, verbose=True,
        on_iteration=lambda k, candidate: optimization_trace.append(_iteration_record(candidate)),
    )
    final = s.planner.evaluate_controls(s.rollout, best.controls, spec=spec)

    events = reach_avoid_events(s.env)
    goal = _event_trace(events["goal"], final.rollout)
    safe = _event_trace(reduce(And, events["obstacles"]), final.rollout)
    goal_step, safe_step = int(goal[1:, 0].argmax()) + 1, int(safe[1:, 0].argmin()) + 1
    mean_trace = final.rollout.aux["mean_trace"]
    result = {
        "interval_initial": list(initial.hard_interval),
        "interval_final": list(final.hard_interval),
        "stored_interval": list(best.hard_interval),
        "returned_iteration": best.iteration,
        "smooth_lower": final.smooth_lower,
        "smooth_beta": final.beta,  # final replay uses beta_end, not the checkpoint's beta
        "hard_interval": list(final.hard_interval),
        "optimization_trace": optimization_trace,
        "control_cost": final.control_cost,
        "controls": final.controls,
        "history": history,
        "iterations": len(history),
        "goal_trace": goal,
        "safe_trace": safe,
        "bounds_trace": _event_trace(events["workspace"], final.rollout),
        "obstacle_traces": _obstacle_traces(events, final.rollout),
        "goal_step": goal_step,
        "goal_interval": goal[goal_step].tolist(),
        "safe_step": safe_step,
        "min_safe_interval": safe[safe_step].tolist(),
        "mean_trace": mean_trace,
        "cov_trace": final.rollout.aux["cov_trace"],
        "u_trace": final.controls.unsqueeze(0),
        "mean_initial": initial.rollout.aux["mean_trace"],
        "cov_initial": initial.rollout.aux["cov_trace"],
        "final_mean": mean_trace[0, -1].tolist(),
        "min_mean_clearance": _mean_clearance(mean_trace, s.env.by_role("obstacle")),
    }
    _log_intervals("reach_avoid", spec, result)

    if save:
        torch.save(result, _output_path(s.cfg["save_file"]))
    if save or show:
        visualize_reach_avoid(
            result, s.env, dt=s.cfg["dt"], ellipse_every=s.cfg["ellipse_every"],
            save_path=_output_path(s.cfg["figure"]) if save else None, show=show,
        )
    return result


# --- Legacy scenarios (single shot, MPC, lane change) ------------------------------


def check_collision(mean_trace, env, r_robot=1.0, moving_obs_dist=2.25):
    """Log mean-path conflicts with static (inflated) and moving obstacles."""
    traj = mean_trace.squeeze()
    if traj.ndim == 1:
        traj = traj.unsqueeze(0)
    is_safe, min_sep = True, float("inf")

    for t in range(traj.shape[0]):
        ego_pos = traj[t].cpu().numpy()
        for region in env.by_role("obstacle"):
            if isinstance(region, MovingRectangleRegion):
                centers = torch.as_tensor(region.centers)
                if t >= len(centers):
                    continue
                dist = float(np.linalg.norm(ego_pos[:2] - centers[t].cpu().numpy()))
                min_sep = min(min_sep, dist)
                if dist < moving_obs_dist:
                    log_utils.log_collision_event(t, "Moving obstacle", f"dist={dist:.2f}")
                    is_safe = False
            elif isinstance(region, RectangleRegion):
                (x_min, x_max), (y_min, y_max) = region.x, region.y
                if (x_min - r_robot <= ego_pos[0] <= x_max + r_robot) and (
                    y_min - r_robot <= ego_pos[1] <= y_max + r_robot
                ):
                    log_utils.log_collision_event(t, "Static obstacle", f"ego={ego_pos}")
                    is_safe = False

    log_utils.log_safety(is_safe, min_sep)


def _normalise_result(data):
    """Fill keys missing from older saved results."""
    result = dict(data)
    if "loss_trace" not in result and "history" in result:
        result["loss_trace"] = result["history"]
    if "history" not in result and "loss_trace" in result:
        result["history"] = result["loss_trace"]
    result.setdefault("p_sat_trace", [result.get("best_p", 0.0)])
    result.setdefault("all_plans", [])
    result.setdefault("best_p", max(result["p_sat_trace"]) if result["p_sat_trace"] else 0.0)
    result.setdefault("mode", "loaded")
    result.setdefault("stopped_reason", None)
    return result


def _load_or_solve(cfg, planner_cfg, env, *, horizon, device="cpu", load_from=None,
                   force_run=False, make_callback=None):
    """Load a saved result if present, else run Planner.solve and save it."""
    result_path = load_from or (str(RESULTS_DIR / cfg["save_file"]) if "save_file" in cfg else None)
    if not force_run and result_path and Path(result_path).exists():
        log_utils.log_load(result_path)
        return _normalise_result(torch.load(result_path, map_location=device, weights_only=False))

    planner = Planner(build_dynamics(cfg, device), env, horizon, config=planner_cfg)
    step_callback = make_callback(env) if make_callback is not None else None
    result = planner.solve(*build_initial_belief(cfg, device), step_callback=step_callback)

    if result_path:
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, result_path)
        log_utils.log_save(result_path)
    return result


def _animate(result, env, cfg, **kwargs):
    anim = cfg["animation"]
    animate_results(
        result["mean_trace"], result["cov_trace"], env, filename=_output_path(anim["filename"]),
        step=anim["step"], title=anim["title"], bounds=anim.get("bounds"), **kwargs,
    )


def run_single_shot(max_iterations=1000, load_from=None, force_run=False, *,
                    config_path="configs/scenarios/single_shot.yaml", show=True):
    device = get_device()
    log_utils.log_device(device)
    cfg, planner_cfg = load_scenario_config(config_path)
    planner_cfg["max_iters"] = max_iterations
    env = build_environment(cfg, device=device)

    log_utils._log.info("Starting single-shot optimisation...")
    result = _load_or_solve(
        cfg, planner_cfg, env, horizon=cfg["T"], device=device,
        load_from=load_from, force_run=force_run
    )
    log_utils._log.info(f"Done. Final stochastic robustness: {result['best_p']:.4f}")

    if show:
        visualize_results(result["mean_trace"], result["cov_trace"], result["u_trace"], env,
                          result["loss_trace"])
        _animate(result, env, cfg)
    return result


def run_mpc(load_from=None, force_run=False, *, config_path="configs/scenarios/mpc.yaml", show=True):
    device = get_device()
    log_utils.log_device(device)
    cfg, planner_cfg = load_scenario_config(config_path)
    env = build_environment(cfg, device=device)

    log_utils._log.info(f"Starting MPC execution (horizon={cfg['H']})...")
    result = _load_or_solve(
        cfg, {**planner_cfg, "MAX_STEPS": cfg["MAX_STEPS"]}, env, horizon=cfg["H"],
        device=device, load_from=load_from, force_run=force_run,
        make_callback=make_mpc_live_callback if show else None,
    )

    if show:
        visualize_results(result["mean_trace"], result["cov_trace"], result["u_trace"], env,
                          history=result["loss_trace"], p_sat_trace=result["p_sat_trace"])
        _animate(result, env, cfg, plan_traces=result["all_plans"])
    return result


def run_lane_change(config_path="configs/scenarios/lane_change.yaml", *, show=True):
    device = get_device()
    log_utils.log_device(device)
    cfg, planner_cfg = load_scenario_config(config_path)
    log_utils.log_scenario_start(cfg.get("label", ""))

    env = build_environment(cfg, device=device)
    planner_cfg = {**planner_cfg, "T_SIM": cfg["T_SIM"], "mpc_mode": "lane_change"}
    result = _load_or_solve(
        cfg, planner_cfg, env, horizon=cfg["H"], device=device, force_run=True,
        make_callback=make_lane_change_live_callback if show else None,
    )
    clip_moving_obstacles(env, result["mean_trace"].shape[1])
    check_collision(result["mean_trace"], env, r_robot=planner_cfg["r_robot"],
                    moving_obs_dist=planner_cfg["moving_obs_dist"])

    if show:
        visualize_lane_change(
            result["mean_trace"], result["cov_trace"], result["u_trace"], env,
            p_sat_trace=result["p_sat_trace"], dt=cfg["dt"], robot_dims=env.metadata.get("robot_dims"),
            xlim=env.metadata.get("plot_xlim"),
        )
        _animate(result, env, cfg, plan_traces=result["all_plans"], robot_dims=env.metadata.get("robot_dims"))
    return result
