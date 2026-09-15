"""Experiment runners: load a scenario, build the problem, call the Planner, plot."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from models.dynamics import DoubleIntegrator, SingleIntegrator
from models.rollouts import gaussian_rollout
from pdstl.operators import Always, GreaterThan
from planning import log_utils
from planning.environment import Environment
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


def build_environment(cfg, device):
    env = Environment(device=device)
    if "road" in cfg and "obstacle" in cfg:
        env.configure_lane_change(
            road=cfg["road"], obstacle=cfg["obstacle"], goal=cfg["goal"], success=cfg["success"],
            horizon=cfg["H"], total_steps=cfg["T_SIM"], dt=cfg["dt"], label=cfg.get("label", ""),
            plot_xlim=cfg.get("plot_xlim"), robot_dims=cfg.get("robot_dims"),
        )
        return env
    if "goal" in cfg:
        env.set_goal(**cfg["goal"])
    if "bounds" in cfg:
        env.set_bounds(**cfg["bounds"])
    for region in cfg.get("visit_regions", []):
        env.add_visit_region(**region)
    for obs in cfg.get("obstacles", []):
        if obs["type"] == "circle":
            env.add_circle_obstacle(center=obs["center"], radius=obs["radius"])
        else:
            env.add_obstacle(x_range=obs["x_range"], y_range=obs["y_range"])
    return env


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
    env = build_environment(cfg, device) if with_environment else None
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
    (lo_i, hi_i), (lo_f, hi_f) = result["interval_initial"], result["interval_final"]
    log_utils._log.info(
        f"[{name}] {spec}: initial [{lo_i:.4f}, {hi_i:.4f}] -> final [{lo_f:.4f}, {hi_f:.4f}]"
        f" | iterations {result['iterations']}"
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


def _mean_clearance(mean_trace, obstacle):
    """Post-hoc diagnostic: smallest mean-to-rectangle distance (negative inside)."""
    points = mean_trace[0, 1:]
    lower = torch.tensor([obstacle["x"][0], obstacle["y"][0]])
    upper = torch.tensor([obstacle["x"][1], obstacle["y"][1]])
    gap = torch.maximum(lower - points, points - upper)
    outside = gap.clamp(min=0).norm(dim=1)
    return torch.where(outside > 0, outside, gap.max(dim=1).values).min().item()


def run_reach_avoid(config_path="configs/scenarios/reach_avoid.yaml", *, show=True, save=True):
    """Always(outside obstacle) ∧ Eventually(inside goal), built by the Environment."""
    s = _setup(config_path, with_environment=True)
    spec = s.env.get_specification(s.H, t_goal_start=1)

    initial = s.planner.evaluate_controls(s.rollout, s.u_init, spec=spec)
    best, history = s.planner.optimize_window(s.rollout, spec=spec, init_guess=s.u_init, verbose=True)

    events = s.env.get_predicates()
    goal = _event_trace(events["goal"], best.rollout)
    safe = _event_trace(events["obstacles"][0], best.rollout)
    goal_step, safe_step = int(goal[1:, 0].argmax()) + 1, int(safe[1:, 0].argmin()) + 1
    mean_trace = best.rollout.aux["mean_trace"]
    result = {
        "interval_initial": list(initial.hard_interval),
        "interval_final": list(best.hard_interval),
        "controls": best.controls,
        "history": history,
        "iterations": len(history),
        "goal_trace": goal,
        "safe_trace": safe,
        "goal_step": goal_step,
        "goal_interval": goal[goal_step].tolist(),
        "safe_step": safe_step,
        "min_safe_interval": safe[safe_step].tolist(),
        "mean_trace": mean_trace,
        "cov_trace": best.rollout.aux["cov_trace"],
        "u_trace": best.controls.unsqueeze(0),
        "mean_initial": initial.rollout.aux["mean_trace"],
        "cov_initial": initial.rollout.aux["cov_trace"],
        "final_mean": mean_trace[0, -1].tolist(),
        "min_mean_clearance": _mean_clearance(mean_trace, s.env.obstacles[0]),
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
        for obs in env.obstacles:
            (x_min, x_max), (y_min, y_max) = obs["x"], obs["y"]
            if (x_min - r_robot <= ego_pos[0] <= x_max + r_robot) and (
                y_min - r_robot <= ego_pos[1] <= y_max + r_robot
            ):
                log_utils.log_collision_event(t, "Static obstacle", f"ego={ego_pos}")
                is_safe = False
        for obs in env.moving_obstacles:
            xt, yt = obs["x_traj"], obs["y_traj"]
            if t < len(xt):
                ox = xt[t].item() if isinstance(xt, torch.Tensor) else xt[t]
                oy = yt[t].item() if isinstance(yt, torch.Tensor) else yt[t]
                dist = np.linalg.norm(ego_pos[:2] - np.array([ox, oy]))
                min_sep = min(min_sep, dist)
                if dist < moving_obs_dist:
                    log_utils.log_collision_event(t, "Moving obstacle", f"dist={dist:.2f}")
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


def _load_or_solve(cfg, planner_cfg, env, *, horizon, load_from=None, force_run=False,
                   make_callback=None):
    """Load a saved result if present, else run Planner.solve and save it."""
    result_path = load_from or (str(RESULTS_DIR / cfg["save_file"]) if "save_file" in cfg else None)
    if not force_run and result_path and Path(result_path).exists():
        log_utils.log_load(result_path)
        return _normalise_result(torch.load(result_path, map_location=env.device, weights_only=False))

    planner = Planner(build_dynamics(cfg, env.device), env, horizon, config=planner_cfg)
    step_callback = make_callback(env) if make_callback is not None else None
    result = planner.solve(*build_initial_belief(cfg, env.device), step_callback=step_callback)

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
    env = build_environment(cfg, device)

    log_utils._log.info("Starting single-shot optimisation...")
    result = _load_or_solve(
        cfg, planner_cfg, env, horizon=cfg["T"], load_from=load_from, force_run=force_run
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
    env = build_environment(cfg, device)

    log_utils._log.info(f"Starting MPC execution (horizon={cfg['H']})...")
    result = _load_or_solve(
        cfg, {**planner_cfg, "MAX_STEPS": cfg["MAX_STEPS"]}, env, horizon=cfg["H"],
        load_from=load_from, force_run=force_run,
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

    env = build_environment(cfg, device)
    planner_cfg = {**planner_cfg, "T_SIM": cfg["T_SIM"], "mpc_mode": "lane_change"}
    result = _load_or_solve(
        cfg, planner_cfg, env, horizon=cfg["H"], force_run=True,
        make_callback=make_lane_change_live_callback if show else None,
    )
    if env.moving_obstacles:
        env.clip_moving_obstacles(result["mean_trace"].shape[1])
    check_collision(result["mean_trace"], env, r_robot=planner_cfg["r_robot"],
                    moving_obs_dist=planner_cfg["moving_obs_dist"])

    if show:
        visualize_lane_change(
            result["mean_trace"], result["cov_trace"], result["u_trace"], env,
            p_sat_trace=result["p_sat_trace"], dt=cfg["dt"], robot_dims=env.robot_dims,
            xlim=env.plot_xlim,
        )
        _animate(result, env, cfg, plan_traces=result["all_plans"], robot_dims=env.robot_dims)
    return result
