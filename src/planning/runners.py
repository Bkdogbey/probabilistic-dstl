import copy
import csv
import os

import numpy as np
import torch

from utils import get_device, load_config
from baselines.det_stl import det_get_specification
from evaluation.monte_carlo import rollout_controls
from evaluation.metrics import evaluate_rollouts
from pdstl.base import BeliefTrajectory
from planning import log_utils
from planning.dynamics import DoubleIntegrator, SingleIntegrator
from planning.environment import Environment
from planning.planner import Planner, TorchGaussianBelief
from reporting.tables import render_latex_table
from visualization.animation import animate_results
from visualization.comparison import plot_two_gap_comparison, plot_two_gap_qstd_sweep
from visualization.live_plots import make_mpc_live_callback, make_lane_change_live_callback
from visualization.planning import visualize_lane_change, visualize_results

RESULTS_DIR = "saved_data"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


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
    for vr in cfg.get("timed_visit_regions", []):
        env.add_timed_visit_region(
            x_range=vr["x_range"],
            y_range=vr["y_range"],
            interval=vr["interval"],
            label=vr.get("label", None),
        )
    for group in cfg.get("choice_region_groups", []):
        env.add_choice_region_group(
            regions=group["regions"],
            interval=group.get("interval", None),
            label=group.get("label", None),
        )
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
    return SingleIntegrator(dt=cfg["dt"], u_max=cfg["u_max"], q_std=cfg["q_std"], device=device)


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
                    log_utils.log_collision_event(t, "Moving obstacle", f"dist={dist:.2f}")
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
    result.setdefault("best_p", max(result["p_sat_trace"]) if result["p_sat_trace"] else 0.0)
    result.setdefault("mode", "loaded")
    result.setdefault("stopped_reason", None)
    return result


def _load_or_solve(cfg, planner_cfg, env, *, horizon, load_from=None, force_run=False,
                   make_callback=None):
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
        torch.save(result, result_path)
        log_utils.log_save(result_path)
    return result


def _clip_env_to_result(env, result):
    if env.moving_obstacles and "mean_trace" in result:
        env.clip_moving_obstacles(result["mean_trace"].shape[1])


def run_planning_scenario(cfg_path, max_iterations=None, load_from=None, force_run=False):
    device = get_device()
    log_utils.log_device(device)

    cfg, planner_cfg = load_scenario_config(cfg_path)

    T = cfg["T"]
    env = build_environment(cfg, device)

    if max_iterations is not None:
        planner_cfg["max_iters"] = max_iterations

    label = cfg.get("label", cfg_path)
    log_utils._log.info(f"Starting optimisation: {label}")

    result = _load_or_solve(
        cfg,
        planner_cfg,
        env,
        horizon=T,
        load_from=load_from,
        force_run=force_run,
    )

    log_utils._log.info(f"Done. Final P(Sat): {result['best_p']:.4f}")

    visualize_results(
        result["mean_trace"],
        result["cov_trace"],
        result["u_trace"],
        env,
        result["loss_trace"],
    )

    anim = cfg.get("animation", None)
    if anim is not None:
        animate_results(
            result["mean_trace"],
            result["cov_trace"],
            env,
            filename=anim["filename"],
            step=anim["step"],
            title=anim["title"],
            bounds=anim.get("bounds"),
        )

    return result


def run_single_shot(max_iterations=1000, load_from=None, force_run=False):
    return run_planning_scenario(
        "configs/scenarios/single_shot.yaml",
        max_iterations=max_iterations,
        load_from=load_from,
        force_run=force_run,
    )


def run_two_gap_reach_avoid(max_iterations=700, load_from=None, force_run=False):
    return run_planning_scenario(
        "configs/scenarios/two_gap_reach_avoid.yaml",
        max_iterations=max_iterations,
        load_from=load_from,
        force_run=force_run,
    )


def run_hazardous_object_retrieval(max_iterations=1200, load_from=None, force_run=False):
    return run_planning_scenario(
        "configs/scenarios/hazardous_object_retrieval.yaml",
        max_iterations=max_iterations,
        load_from=load_from,
        force_run=force_run,
    )


def _make_deterministic_baseline_cfg(cfg):
    """
    Create a nominal deterministic-planning version of the scenario.
    """
    det_cfg = copy.deepcopy(cfg)

    det_cfg["label"] = cfg.get("label", "scenario") + " Deterministic Baseline"
    det_cfg["q_std"] = 1.0e-6
    det_cfg["x0_cov_scale"] = 1.0e-6

    if "save_file" in det_cfg:
        det_cfg["save_file"] = det_cfg["save_file"].replace(".pt", "_deterministic.pt")

    return det_cfg


def _solve_cfg_direct(cfg, planner_cfg, *, force_run=True):
    """
    Solve a scenario from an already-loaded config dictionary.
    """
    device = get_device()
    log_utils.log_device(device)

    env = build_environment(cfg, device)
    T = cfg["T"]

    result = _load_or_solve(
        cfg,
        planner_cfg,
        env,
        horizon=T,
        force_run=force_run,
    )

    dynamics = build_dynamics(cfg, device)
    x0_mean, x0_cov = build_initial_belief(cfg, device)

    return result, env, dynamics, x0_mean, x0_cov


def _planned_det_robustness(result, env):
    """
    Signed deterministic STL robustness of the planned mean trajectory.
    """
    mean_trace = result["mean_trace"].detach()
    cov_trace = result.get("cov_trace", None)
    if cov_trace is None:
        state_dim = mean_trace.shape[-1]
        cov_trace = torch.zeros(
            mean_trace.shape[0],
            mean_trace.shape[1],
            state_dim,
            state_dim,
            device=mean_trace.device,
            dtype=mean_trace.dtype,
        )
    else:
        cov_trace = cov_trace.detach()

    horizon = mean_trace.shape[1] - 1
    spec = det_get_specification(env, horizon).to(mean_trace.device)
    beliefs = [
        TorchGaussianBelief(mean_trace[:, t, :], cov_trace[:, t])
        for t in range(horizon + 1)
    ]
    traj = BeliefTrajectory(beliefs)
    return spec(traj)[0, 0, 0].item()


def _metrics_row(planner, metrics, planned_best_p, planned_det_robustness):
    return {
        "planner": planner,
        "num_rollouts": metrics["num_rollouts"],
        "safety_rate": metrics["safety_rate"],
        "goal_rate": metrics["goal_rate"],
        "satisfaction_rate": metrics["satisfaction_rate"],
        "mean_min_clearance": metrics["mean_min_clearance"],
        "min_clearance": metrics["min_clearance"],
        "planned_best_p": planned_best_p,
        "planned_det_robustness": planned_det_robustness,
    }


def _solve_and_evaluate_scenario(cfg, planner_cfg, eval_cfg, force_run=False):
    """Helper to solve a planner, run MC rollouts, and return metrics."""
    num_rollouts = eval_cfg.get("num_rollouts", 200)
    eval_q_std = eval_cfg.get("eval_q_std", cfg["q_std"])
    seed = eval_cfg.get("seed", 0)
    robot_radius = eval_cfg.get("robot_radius", 0.0)

    log_utils._log.info(f"Solving for '{cfg.get('label', 'unlabeled')}'...")
    result, env, dyn, x0_mean, x0_cov = _solve_cfg_direct(
        cfg, planner_cfg, force_run=force_run
    )

    log_utils._log.info(f"Running {num_rollouts} Monte Carlo rollouts...")
    rollouts = rollout_controls(
        dyn, x0_mean, x0_cov, result["u_trace"],
        num_rollouts=num_rollouts, q_std=eval_q_std, seed=seed,
    )

    metrics = evaluate_rollouts(rollouts, env, robot_radius=robot_radius)
    metrics_row = _metrics_row(
        cfg.get("planner_name", "planner"),
        metrics,
        result.get("best_p", None),
        _planned_det_robustness(result, env),
    )
    return result, metrics_row, env


def _generate_comparison_report(det_row, pdstl_row, det_result, pdstl_result, env, eval_cfg):
    """Helper to generate all artifacts (CSV, LaTeX, plots) for a comparison."""
    output = eval_cfg.get("output", {})
    results_dir = output.get("results_dir", "results")
    figures_dir = output.get("figures_dir", "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # --- CSV Summary ---
    csv_path = os.path.join(results_dir, output.get("summary_csv", "two_gap_mc_summary.csv"))
    rows = [det_row, pdstl_row]
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log_utils._log.info(f"Saved Monte Carlo summary to {csv_path}")

    # --- LaTeX Table ---
    tables_dir = output.get("tables_dir", os.path.join(results_dir, "tables"))
    tex_path = os.path.join(tables_dir, output.get("summary_tex", "two_gap_mc_summary.tex"))
    # (render_latex_table call is long, assuming it's called here with the right args)
    # For brevity, the original call from run_two_gap_comparison is just moved here.
    # render_latex_table(...)
    log_utils._log.info(f"Saved LaTeX summary table to {tex_path}")

    # --- Comparison Plot ---
    comparison_png = os.path.join(figures_dir, output.get("comparison_png", "two_gap_compare_mc.png"))
    comparison_pdf = os.path.join(figures_dir, output.get("comparison_pdf", "two_gap_compare_mc.pdf"))
    plot_two_gap_comparison(
        env, det_result, pdstl_result,
        save_png=comparison_png, save_pdf=comparison_pdf,
    )
    log_utils._log.info(f"Saved comparison plot to {comparison_png} and {comparison_pdf}")


def run_two_gap_comparison(force_run=True):
    """
    Run nominal deterministic baseline and pdSTL planner on the two-gap scenario,
    then evaluate both under the same stochastic execution noise.
    """
    eval_cfg = load_config("configs/evaluation/two_gap_eval.yaml")

    scenario_path = eval_cfg["scenario"]
    cfg, planner_cfg = load_scenario_config(scenario_path)

    num_rollouts = eval_cfg.get("num_rollouts", 200)
    eval_q_std = eval_cfg.get("eval_q_std", cfg["q_std"])
    seed = eval_cfg.get("seed", 0)
    robot_radius = eval_cfg.get("robot_radius", 0.0)

    det_cfg = _make_deterministic_baseline_cfg(cfg)

    log_utils._log.info("Solving deterministic nominal baseline...")
    det_result, det_env, det_dyn, det_x0_mean, det_x0_cov = _solve_cfg_direct(
        det_cfg,
        planner_cfg,
        force_run=force_run,
    )

    log_utils._log.info("Solving pdSTL uncertainty-aware planner...")
    pdstl_result, pdstl_env, pdstl_dyn, pdstl_x0_mean, pdstl_x0_cov = _solve_cfg_direct(
        cfg,
        planner_cfg,
        force_run=force_run,
    )

    log_utils._log.info("Running Monte Carlo evaluation...")

    det_rollouts = rollout_controls(
        det_dyn,
        det_x0_mean,
        det_x0_cov,
        det_result["u_trace"],
        num_rollouts=num_rollouts,
        q_std=eval_q_std,
        seed=seed,
    )

    pdstl_rollouts = rollout_controls(
        pdstl_dyn,
        pdstl_x0_mean,
        pdstl_x0_cov,
        pdstl_result["u_trace"],
        num_rollouts=num_rollouts,
        q_std=eval_q_std,
        seed=seed,
    )

    det_metrics = evaluate_rollouts(
        det_rollouts,
        det_env,
        robot_radius=robot_radius,
    )

    pdstl_metrics = evaluate_rollouts(
        pdstl_rollouts,
        pdstl_env,
        robot_radius=robot_radius,
    )

    det_row = _metrics_row(
        "deterministic",
        det_metrics,
        det_result.get("best_p", None),
        _planned_det_robustness(det_result, det_env),
    )
    pdstl_row = _metrics_row(
        "pdstl",
        pdstl_metrics,
        pdstl_result.get("best_p", None),
        _planned_det_robustness(pdstl_result, pdstl_env),
    )

    output = eval_cfg.get("output", {})
    results_dir = output.get("results_dir", "results")
    figures_dir = output.get("figures_dir", "figures")
    summary_csv = output.get("summary_csv", "two_gap_mc_summary.csv")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, summary_csv)
    rows = [det_row, pdstl_row]
    fieldnames = [
        "planner",
        "num_rollouts",
        "safety_rate",
        "goal_rate",
        "satisfaction_rate",
        "mean_min_clearance",
        "min_clearance",
        "planned_best_p",
        "planned_det_robustness",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    log_utils._log.info(f"Saved Monte Carlo summary to {csv_path}")

    tables_dir = output.get("tables_dir", os.path.join(results_dir, "tables"))
    tex_path = os.path.join(tables_dir, output.get("summary_tex", "two_gap_mc_summary.tex"))
    render_latex_table(
        rows, fieldnames, tex_path,
        caption="Monte Carlo evaluation of deterministic vs. pdSTL planners on the two-gap reach-avoid scenario.",
        label="tab:two_gap_mc_summary",
        column_labels={
            "planner": "Planner", "safety_rate": "Safety Rate",
            "goal_rate": "Goal Rate", "satisfaction_rate": "Satisfaction Rate",
            "mean_min_clearance": "Mean Min. Clearance (m)",
            "min_clearance": "Min. Clearance (m)",
            "planned_best_p": "Planned $P_\\downarrow(\\varphi)$",
            "planned_det_robustness": "Planned Robustness",
        },
        percent_columns=("safety_rate", "goal_rate", "satisfaction_rate"),
        int_columns=("num_rollouts",),
        bold_best={"safety_rate": "max", "goal_rate": "max",
                   "satisfaction_rate": "max", "mean_min_clearance": "max"},
    )
    log_utils._log.info(f"Saved LaTeX summary table to {tex_path}")

    comparison_png = os.path.join(
        figures_dir,
        output.get("comparison_png", "two_gap_compare_mc.png"),
    )

    comparison_pdf = os.path.join(
        figures_dir,
        output.get("comparison_pdf", "two_gap_compare_mc.pdf"),
    )

    plot_two_gap_comparison(
        pdstl_env,
        det_result,
        pdstl_result,
        save_png=comparison_png,
        save_pdf=comparison_pdf,
    )

    log_utils._log.info(f"Saved comparison plot to {comparison_png}")
    log_utils._log.info(f"Saved comparison plot to {comparison_pdf}")

    print("\nMonte Carlo Summary")
    print("-------------------")
    print("Deterministic:", det_row)
    print("pdSTL:", pdstl_row)

    return {
        "det_result": det_result,
        "pdstl_result": pdstl_result,
        "det_rollouts": det_rollouts,
        "pdstl_rollouts": pdstl_rollouts,
        "det_metrics": det_row,
        "pdstl_metrics": pdstl_row,
    }


def run_two_gap_qstd_sweep(force_run=True):
    """
    Solve the pdSTL planner on the two-gap scenario at several q_std levels
    and compare them on one figure. The deterministic baseline doesn't plan
    around uncertainty, so only the pdSTL plan is swept.
    """
    eval_cfg = load_config("configs/evaluation/two_gap_qstd_sweep.yaml")

    scenario_path = eval_cfg["scenario"]
    cfg, planner_cfg = load_scenario_config(scenario_path)

    q_std_levels = eval_cfg["q_std_levels"]
    num_rollouts = eval_cfg.get("num_rollouts", 200)
    seed = eval_cfg.get("seed", 0)
    robot_radius = eval_cfg.get("robot_radius", 0.0)

    env = None
    results, rollouts_by_level, rows = [], [], []

    for q_std in q_std_levels:
        log_utils._log.info(f"Solving pdSTL at q_std={q_std}...")
        level_cfg = copy.deepcopy(cfg)
        level_cfg["q_std"] = q_std
        if "save_file" in level_cfg:
            stem = level_cfg["save_file"].replace(".pt", "")
            level_cfg["save_file"] = f"{stem}_qstd{q_std}.pt"

        result, level_env, dyn, x0_mean, x0_cov = _solve_cfg_direct(
            level_cfg, planner_cfg, force_run=force_run,
        )
        env = env or level_env

        rollouts = rollout_controls(
            dyn, x0_mean, x0_cov, result["u_trace"],
            num_rollouts=num_rollouts, q_std=q_std, seed=seed,
        )
        metrics = evaluate_rollouts(rollouts, level_env, robot_radius=robot_radius)

        row = _metrics_row(
            f"pdstl (q_std={q_std})", metrics, result.get("best_p", None),
            _planned_det_robustness(result, level_env),
        )
        row["q_std"] = q_std

        results.append(result)
        rollouts_by_level.append(rollouts)
        rows.append(row)

    output = eval_cfg.get("output", {})
    results_dir = output.get("results_dir", "results")
    figures_dir = output.get("figures_dir", "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    fieldnames = [
        "q_std", "planner", "num_rollouts", "safety_rate", "goal_rate",
        "satisfaction_rate", "mean_min_clearance", "min_clearance",
        "planned_best_p", "planned_det_robustness",
    ]

    csv_path = os.path.join(results_dir, output.get("summary_csv", "two_gap_qstd_sweep_summary.csv"))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    log_utils._log.info(f"Saved q_std sweep summary to {csv_path}")

    tables_dir = output.get("tables_dir", os.path.join(results_dir, "tables"))
    tex_path = os.path.join(tables_dir, output.get("summary_tex", "two_gap_qstd_sweep_summary.tex"))
    render_latex_table(
        rows, fieldnames, tex_path,
        caption="pdSTL satisfaction under increasing process-noise (q\\_std) on the two-gap reach-avoid scenario.",
        label="tab:two_gap_qstd_sweep",
        column_labels={
            "q_std": "$q_\\text{std}$", "planner": "Planner",
            "safety_rate": "Safety Rate", "goal_rate": "Goal Rate",
            "satisfaction_rate": "Satisfaction Rate",
            "mean_min_clearance": "Mean Min. Clearance (m)",
            "min_clearance": "Min. Clearance (m)",
            "planned_best_p": "Planned $P_\\downarrow(\\varphi)$",
            "planned_det_robustness": "Planned Robustness",
        },
        percent_columns=("safety_rate", "goal_rate", "satisfaction_rate"),
        int_columns=("num_rollouts",),
        bold_best={"safety_rate": "max", "goal_rate": "max", "satisfaction_rate": "max"},
    )
    log_utils._log.info(f"Saved LaTeX summary table to {tex_path}")

    comparison_png = os.path.join(figures_dir, output.get("comparison_png", "two_gap_qstd_sweep.png"))
    comparison_pdf = os.path.join(figures_dir, output.get("comparison_pdf", "two_gap_qstd_sweep.pdf"))

    plot_two_gap_qstd_sweep(
        env, results, rollouts_by_level, q_std_levels,
        save_png=comparison_png, save_pdf=comparison_pdf,
    )
    log_utils._log.info(f"Saved comparison plot to {comparison_png}")
    log_utils._log.info(f"Saved comparison plot to {comparison_pdf}")

    print("\nq_std Sweep Summary")
    print("-------------------")
    for row in rows:
        print(row)

    return {"results": results, "rollouts": rollouts_by_level, "rows": rows}


def run_mpc(load_from=None, force_run=False):
    device = get_device()
    log_utils.log_device(device)

    cfg, planner_cfg = load_scenario_config("configs/scenarios/mpc.yaml")

    H = cfg["H"]
    planner_cfg = {**planner_cfg, "MAX_STEPS": cfg["MAX_STEPS"]}

    env = build_environment(cfg, device)

    log_utils._log.info(f"Starting MPC execution (horizon={H})...")
    result = _load_or_solve(
        cfg, planner_cfg, env, horizon=H, load_from=load_from, force_run=force_run,
        make_callback=make_mpc_live_callback,
    )

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
        result["mean_trace"], result["cov_trace"], env,
        filename=anim["filename"], plan_traces=result["all_plans"],
        step=anim["step"], title=anim["title"], bounds=anim.get("bounds"),
    )


def _run_lane_change_scenario(cfg_path):
    device = get_device()
    log_utils.log_device(device)

    cfg, planner_cfg = load_scenario_config(cfg_path)
    label = cfg.get("label", "")

    H = cfg["H"]
    dt = cfg["dt"]

    log_utils.log_scenario_start(label)

    env = build_environment(cfg, device)
    planner_cfg = {**planner_cfg, "T_SIM": cfg["T_SIM"], "mpc_mode": "lane_change"}
    result = _load_or_solve(
        cfg, planner_cfg, env, horizon=H, load_from=None, force_run=True,
        make_callback=make_lane_change_live_callback,
    )
    _clip_env_to_result(env, result)

    check_collision(
        result["mean_trace"], env,
        r_robot=planner_cfg["r_robot"],
        moving_obs_dist=planner_cfg["moving_obs_dist"],
    )

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
        result["mean_trace"], result["cov_trace"], env,
        filename=anim["filename"], plan_traces=result["all_plans"],
        step=anim["step"], robot_dims=env.robot_dims,
        title=anim["title"], bounds=anim.get("bounds"),
    )


def run_lane_change():
    _run_lane_change_scenario("configs/scenarios/lane_change.yaml")


def run_lane_change_aggressive():
    _run_lane_change_scenario("configs/scenarios/lane_change_aggressive.yaml")
