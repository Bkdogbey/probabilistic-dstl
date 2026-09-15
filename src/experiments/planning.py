import os
from pathlib import Path

import numpy as np
import torch

from utils import get_device, load_config
from planning import log_utils
from models.dynamics import DoubleIntegrator, SingleIntegrator
from models.rollouts import gaussian_rollout
from planning.environment import Environment
from planning.examples import controls_to_params
from planning.planner import Planner
from planning.specifications import inside_rectangle, reach_avoid
from visualization.animation import animate_results
from visualization.live_plots import (
    make_mpc_live_callback,
    make_lane_change_live_callback,
)
from visualization.planning import (
    plot_controls,
    plot_metrics,
    plot_reach_avoid,
    visualize_lane_change,
    visualize_results,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "outputs"


def load_scenario_config(cfg_path):
    cfg = load_config(cfg_path)
    planner_cfg = {**load_config("configs/planning.yaml"), **cfg.get("planner", {})}
    return cfg, planner_cfg


def build_environment(cfg, device):
    env = Environment(device=device)
    if "road" in cfg and "obstacle" in cfg:
        env.configure_lane_change(
            road=cfg["road"],
            obstacle=cfg["obstacle"],
            goal=cfg["goal"],
            success=cfg["success"],
            horizon=cfg["H"],
            total_steps=cfg["T_SIM"],
            dt=cfg["dt"],
            label=cfg.get("label", ""),
            plot_xlim=cfg.get("plot_xlim"),
            robot_dims=cfg.get("robot_dims"),
        )
        return env

    if "goal" in cfg:
        env.set_goal(**cfg["goal"])
    if "bounds" in cfg:
        env.set_bounds(**cfg["bounds"])
    for vr in cfg.get("visit_regions", []):
        env.add_visit_region(**vr)
    for obs in cfg.get("obstacles", []):
        if obs["type"] == "circle":
            env.add_circle_obstacle(center=obs["center"], radius=obs["radius"])
        else:
            env.add_obstacle(x_range=obs["x_range"], y_range=obs["y_range"])
    return env


def build_initial_belief(cfg, device):
    x0_mean = torch.tensor(cfg["x0_mean"], device=device)
    x0_cov = torch.eye(len(cfg["x0_mean"]), device=device) * cfg["x0_cov_scale"]
    return x0_mean, x0_cov


def build_dynamics(cfg, device):
    kind = cfg.get("dynamics", "single_integrator")
    if kind == "double_integrator":
        return DoubleIntegrator(
            dt=cfg["dt"],
            u_max=cfg["u_max"],
            q_std=cfg["q_std"],
            device=device,
        )
    return SingleIntegrator(
        dt=cfg["dt"], u_max=cfg["u_max"], q_std=cfg["q_std"], device=device
    )


def check_collision(mean_trace, env, r_robot=1.0, moving_obs_dist=2.25):
    traj = mean_trace.squeeze()  # [T, 2]
    if traj.ndim == 1:
        traj = traj.unsqueeze(0)
    T = traj.shape[0]

    is_safe = True
    min_sep = float("inf")

    for t in range(T):
        ego_pos = traj[t].cpu().numpy()

        for obs in env.obstacles:
            x_min, x_max = obs["x"]
            y_min, y_max = obs["y"]
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
                if dist < min_sep:
                    min_sep = dist
                if dist < moving_obs_dist:
                    log_utils.log_collision_event(
                        t, "Moving obstacle", f"dist={dist:.2f}"
                    )
                    is_safe = False

    log_utils.log_safety(is_safe, min_sep)


def _scenario_result_path(cfg, load_from):
    if load_from is not None:
        return load_from
    if "save_file" not in cfg:
        return None
    return os.path.join(RESULTS_DIR, cfg["save_file"])


def _normalise_result(data):
    result = dict(data)
    if "loss_trace" not in result and "history" in result:
        result["loss_trace"] = result["history"]
    if "history" not in result and "loss_trace" in result:
        result["history"] = result["loss_trace"]
    result.setdefault("p_sat_trace", [result.get("best_p", 0.0)])
    result.setdefault("all_plans", [])
    result.setdefault(
        "best_p", max(result["p_sat_trace"]) if result["p_sat_trace"] else 0.0
    )
    result.setdefault("mode", "loaded")
    result.setdefault("stopped_reason", None)
    return result


def _load_or_solve(
    cfg,
    planner_cfg,
    env,
    *,
    horizon,
    load_from=None,
    force_run=False,
    make_callback=None,
):
    result_path = _scenario_result_path(cfg, load_from)
    if not force_run and result_path and os.path.exists(result_path):
        log_utils.log_load(result_path)
        return _normalise_result(
            torch.load(result_path, map_location=env.device, weights_only=False)
        )

    step_callback = make_callback(env) if make_callback is not None else None
    dynamics = build_dynamics(cfg, env.device)
    planner = Planner(dynamics, env, horizon, config=planner_cfg)
    x0_mean, x0_cov = build_initial_belief(cfg, env.device)
    result = planner.solve(x0_mean, x0_cov, step_callback=step_callback)

    if result_path:
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, result_path)
        log_utils.log_save(result_path)
    return result


def _clip_env_to_result(env, result):
    if env.moving_obstacles and "mean_trace" in result:
        env.clip_moving_obstacles(result["mean_trace"].shape[1])


def _animation_path(filename):
    if filename is None:
        return None
    path = RESULTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def run_single_shot(
    max_iterations=1000,
    load_from=None,
    force_run=False,
    *,
    config_path="configs/scenarios/single_shot.yaml",
    show=True,
):
    device = get_device()
    log_utils.log_device(device)

    cfg, planner_cfg = load_scenario_config(config_path)

    T = cfg["T"]

    env = build_environment(cfg, device)
    planner_cfg["max_iters"] = max_iterations

    log_utils._log.info("Starting single-shot optimisation...")
    result = _load_or_solve(
        cfg, planner_cfg, env, horizon=T, load_from=load_from, force_run=force_run
    )
    log_utils._log.info(f"Done. Final stochastic robustness: {result['best_p']:.4f}")

    if show:
        visualize_results(
            result["mean_trace"],
            result["cov_trace"],
            result["u_trace"],
            env,
            result["loss_trace"],
        )

        anim = cfg["animation"]
        animate_results(
            result["mean_trace"],
            result["cov_trace"],
            env,
            filename=_animation_path(anim["filename"]),
            step=anim["step"],
            title=anim["title"],
            bounds=anim["bounds"],
        )

    return result


def run_mpc(
    load_from=None,
    force_run=False,
    *,
    config_path="configs/scenarios/mpc.yaml",
    show=True,
):
    device = get_device()
    log_utils.log_device(device)

    cfg, planner_cfg = load_scenario_config(config_path)

    H = cfg["H"]
    planner_cfg = {**planner_cfg, "MAX_STEPS": cfg["MAX_STEPS"]}

    env = build_environment(cfg, device)

    log_utils._log.info(f"Starting MPC execution (horizon={H})...")
    result = _load_or_solve(
        cfg,
        planner_cfg,
        env,
        horizon=H,
        load_from=load_from,
        force_run=force_run,
        make_callback=make_mpc_live_callback if show else None,
    )

    if show:
        visualize_results(
            result["mean_trace"],
            result["cov_trace"],
            result["u_trace"],
            env,
            history=result["loss_trace"],
            p_sat_trace=result["p_sat_trace"],
        )

        anim = cfg["animation"]
        animate_results(
            result["mean_trace"],
            result["cov_trace"],
            env,
            filename=_animation_path(anim["filename"]),
            plan_traces=result["all_plans"],
            step=anim["step"],
            title=anim["title"],
            bounds=anim.get("bounds"),
        )

    return result


def run_lane_change(config_path="configs/scenarios/lane_change.yaml", *, show=True):
    device = get_device()
    log_utils.log_device(device)

    cfg, planner_cfg = load_scenario_config(config_path)
    label = cfg.get("label", "")

    H = cfg["H"]
    dt = cfg["dt"]

    log_utils.log_scenario_start(label)

    env = build_environment(cfg, device)
    planner_cfg = {**planner_cfg, "T_SIM": cfg["T_SIM"], "mpc_mode": "lane_change"}
    result = _load_or_solve(
        cfg,
        planner_cfg,
        env,
        horizon=H,
        load_from=None,
        force_run=True,
        make_callback=make_lane_change_live_callback if show else None,
    )
    _clip_env_to_result(env, result)

    check_collision(
        result["mean_trace"],
        env,
        r_robot=planner_cfg["r_robot"],
        moving_obs_dist=planner_cfg["moving_obs_dist"],
    )

    if show:
        visualize_lane_change(
            result["mean_trace"],
            result["cov_trace"],
            result["u_trace"],
            env,
            p_sat_trace=result["p_sat_trace"],
            dt=dt,
            robot_dims=env.robot_dims,
            xlim=env.plot_xlim,
        )

        anim = cfg["animation"]
        animate_results(
            result["mean_trace"],
            result["cov_trace"],
            env,
            filename=_animation_path(anim["filename"]),
            plan_traces=result["all_plans"],
            step=anim["step"],
            robot_dims=env.robot_dims,
            title=anim["title"],
            bounds=anim.get("bounds"),
        )

    return result


def _rectangle_clearance(points, x_range, y_range):
    """Euclidean distance from each [N, 2] point to a rectangle; negative inside.

    A geometric diagnostic of the mean path only -- never part of the spec.
    """
    x, y = points[:, 0], points[:, 1]
    dx = torch.maximum(x_range[0] - x, x - x_range[1])
    dy = torch.maximum(y_range[0] - y, y - y_range[1])
    outside = torch.sqrt(dx.clamp(min=0) ** 2 + dy.clamp(min=0) ** 2)
    inside = torch.minimum(torch.maximum(dx, dy), torch.zeros_like(dx))
    return torch.where((dx > 0) | (dy > 0), outside, inside)


def _hard_interval(formula, rollout):
    """Directly evaluated [L, U] at origin 0; never the smooth surrogate."""
    return formula(rollout.belief_trajectory, scale=-1).detach()[0, 0]


def run_reach_avoid(
    max_iterations=None,
    *,
    config_path="configs/scenarios/reach_avoid.yaml",
    show=True,
    save=True,
):
    """Plan Always[1,H](outside O) and Eventually[1,H](inside G) in one window:

        v -> u -> SingleIntegrator -> gaussian_rollout -> GaussianBelief_k
          -> probability_bounds(atoms) -> Frechet And/Or -> Always/Eventually -> J

    The rectangles are primitive x/y atoms, the planner receives the formula
    directly, and every shaping heuristic is off. The Environment object only
    holds geometry for the plots; its legacy specification is never built.
    """
    device = get_device()
    log_utils.log_device(device)

    cfg, planner_cfg = load_scenario_config(config_path)
    if max_iterations is not None:
        planner_cfg["max_iters"] = max_iterations
    H = cfg["H"]

    dyn = build_dynamics(cfg, device)
    x0_mean, x0_cov = build_initial_belief(cfg, device)
    rollout = gaussian_rollout(dyn, x0_mean, x0_cov)
    spec = reach_avoid(cfg["goal"], cfg["obstacle"], H)
    goal_formula = inside_rectangle(**cfg["goal"])

    u_init = torch.tensor(cfg["init_control"], device=device).repeat(H, 1)
    initial = rollout(controls_to_params(dyn, u_init))
    interval_initial = _hard_interval(spec, initial)

    log_utils._log.info(f"Starting reach-avoid optimisation: {spec}")
    planner = Planner(dyn, None, H, config=planner_cfg)
    best, history = planner.optimize_window(
        rollout, spec=spec, init_guess=u_init, verbose=True
    )

    final = rollout(controls_to_params(dyn, best.controls))
    interval_final = _hard_interval(spec, final)

    mean_trace = final.aux["mean_trace"].detach()
    cov_trace = final.aux["cov_trace"].detach()
    path = mean_trace[0, 1:]  # steps 1..H, where the windows apply

    goal_lower = goal_formula(final.belief_trajectory, scale=-1).detach()[0, 1:, 0]
    goal_step = int(goal_lower.argmax()) + 1
    clearance = _rectangle_clearance(
        path, cfg["obstacle"]["x_range"], cfg["obstacle"]["y_range"]
    )

    result = {
        "mean_trace": mean_trace,
        "cov_trace": cov_trace,
        "u_trace": best.controls.unsqueeze(0),
        "loss_trace": history,
        "history": history,
        "best_p": interval_final[0].item(),
        "spec": spec,
        "dynamics": dyn,
        "planner_cfg": planner_cfg,
        "interval_initial": interval_initial.tolist(),
        "interval_final": interval_final.tolist(),
        "stored_hard_interval": list(best.hard_interval),
        "controls": best.controls,
        "u_init": u_init,
        "u_max": dyn.u_max,
        "mean_initial": initial.aux["mean_trace"].detach(),
        "cov_initial": initial.aux["cov_trace"].detach(),
        "iterations": len(history),
        "final_state": mean_trace[0, -1].tolist(),
        "goal_step": goal_step,
        "goal_state": mean_trace[0, goal_step].tolist(),
        "min_obstacle_clearance": clearance.min().item(),
        "goal": cfg["goal"],
        "obstacle": cfg["obstacle"],
        "mode": "reach_avoid",
    }

    lo_i, hi_i = result["interval_initial"]
    lo_f, hi_f = result["interval_final"]
    gx, gy = result["goal_state"]
    log_utils._log.info(
        f"[reach_avoid] hard interval initial [{lo_i:.4f}, {hi_i:.4f}]"
        f"  ->  final [{lo_f:.4f}, {hi_f:.4f}]\n"
        f"    goal witness step {goal_step} at ({gx:.3f}, {gy:.3f}) | "
        f"min obstacle clearance {result['min_obstacle_clearance']:.3f} | "
        f"iterations {result['iterations']}"
    )

    if save:
        path_out = RESULTS_DIR / cfg["save_file"]
        path_out.parent.mkdir(parents=True, exist_ok=True)
        saved = {k: v for k, v in result.items() if k not in ("spec", "dynamics")}
        torch.save(saved, path_out)
        log_utils.log_save(str(path_out))

    if save or show:
        env = Environment(device=device)
        env.set_goal(**cfg["goal"])
        env.add_obstacle(**cfg["obstacle"])
        plot_reach_avoid(
            result["mean_initial"],
            result["cov_initial"],
            mean_trace,
            cov_trace,
            env,
            cfg["ellipse_steps"],
            title=f"Reach-avoid: hard interval [{lo_f:.3f}, {hi_f:.3f}]",
            save_path=_animation_path(cfg["figure"]) if save else None,
            show=show,
        )

    if show:
        plot_controls(best.controls.cpu().numpy())
        plot_metrics(history, None)

        anim = cfg["animation"]
        animate_results(
            mean_trace,
            cov_trace,
            env,
            filename=_animation_path(anim["filename"]),
            step=anim["step"],
            dt=cfg["dt"],
            title=anim["title"],
            bounds=anim["bounds"],
        )

    return result
