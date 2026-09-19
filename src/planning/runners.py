"""Load scenarios, construct problems, run the planner, and present results."""

import logging
from pathlib import Path
from types import SimpleNamespace

import torch

from models.dynamics import DoubleIntegrator, SingleIntegrator
from models.rollouts import gaussian_rollout, lane_rollout
from pdstl.operators import Always
from pdstl.predicates import GreaterThan
from planning.environment import (
    build_lane_merge_environment,
    build_reach_avoid_environment,
    lane_contains_point,
    lane_local_window,
)
from planning.planner import Planner
from utils import get_device, load_config

logger = logging.getLogger(__name__)
RESULTS_DIR = Path(__file__).resolve().parents[2] / "outputs"


def build_dynamics(cfg, device):
    common = {key: cfg[key] for key in ("dt", "u_max", "q_std")}
    if cfg.get("dynamics", "single_integrator") == "double_integrator":
        return DoubleIntegrator(**common, device=device)
    return SingleIntegrator(
        **common, device=device, state_dim=cfg.get("state_dim", 2)
    )


def build_initial_belief(cfg, device):
    mean = torch.tensor(cfg["x0_mean"], device=device, dtype=torch.float32)
    return mean, torch.eye(len(mean), device=device) * cfg["x0_cov_scale"]


def build_environment(cfg, device="cpu"):
    kind = cfg.get("scenario", {}).get("type", "reach_avoid")
    if kind == "reach_avoid":
        return build_reach_avoid_environment(cfg)
    if kind == "lane_merge":
        return build_lane_merge_environment(cfg, device)
    raise ValueError(f"unknown scenario type {kind!r}")


def setup_problem(cfg, *, device=None, with_environment=False):
    """Construct a dynamics, initial belief, planner, rollout, and optional world."""
    device = get_device() if device is None else device
    dyn = build_dynamics(cfg, device)
    state = build_initial_belief(cfg, device)
    planner = Planner(dyn, cfg["H"], cfg.get("planner", {}))
    guess = cfg.get("init_control")
    if guess is not None:
        guess = torch.as_tensor(guess, device=device, dtype=dyn.B.dtype)
        if guess.ndim == 1:
            guess = guess.repeat(cfg["H"], 1)
    return SimpleNamespace(
        cfg=cfg,
        dyn=dyn,
        state=state,
        planner=planner,
        init_guess=guess,
        rollout=gaussian_rollout(dyn, *state),
        env=build_environment(cfg, device) if with_environment else None,
    )


def _output_path(stem, suffix):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{stem}{suffix}"


def _save_result(result, stem, save):
    if save:
        torch.save(result, _output_path(stem, ".pt"))


def _optimization_view(s, label, enabled):
    if not enabled:
        return None, None
    from visualization.live_plots import create_optimization_view

    _, _, on_iteration, finish_window = create_optimization_view(
        label,
        max_iters=s.planner.cfg["max_iters"],
        control_unit="m/s²" if isinstance(s.dyn, DoubleIntegrator) else "m/s",
    )
    return on_iteration, finish_window


def run_altitude_safety(
    config_path="configs/scenarios/altitude_safety.yaml",
    *,
    show=False,
    save=True,
    verbose=False,
    live_optimization=False,
    optimization_every=20,
):
    s = setup_problem(load_config(config_path))
    spec = Always(
        GreaterThan(s.cfg["threshold"], dim=0), interval=[1, s.cfg["H"]]
    )
    initial = (
        s.planner.evaluate_controls(s.rollout, s.init_guess, spec=spec)
        if show or save
        else None
    )
    on_iteration, finish_window = _optimization_view(
        s, "Altitude safety", live_optimization
    )
    result = s.planner.optimize_window(
        s.rollout,
        spec=spec,
        init_guess=s.init_guess,
        verbose=verbose,
        on_iteration=(
            (lambda iteration, record: on_iteration(0, iteration, record))
            if on_iteration is not None
            else None
        ),
        callback_every=optimization_every if on_iteration is not None else 1,
    )
    if finish_window is not None:
        finish_window(0, result)
    if show or save:
        from visualization.animation import animate_altitude
        from visualization.planning import plot_altitude_safety

        plot_altitude_safety(
            result,
            initial=initial,
            dt=s.cfg["dt"],
            threshold=s.cfg["threshold"],
            u_max=s.dyn.u_max,
            save_path=_output_path("altitude_safety", ".png")
            if save
            else None,
            show=show,
        )
        animate_altitude(
            result,
            dt=s.cfg["dt"],
            threshold=s.cfg["threshold"],
            filename=_output_path("altitude_safety", ".gif") if save else None,
            show=show,
        )
    _save_result(result, "altitude_safety", save)
    return result


def run_reach_avoid(
    config_path="configs/scenarios/reach_avoid.yaml",
    *,
    show=False,
    save=True,
    verbose=False,
    live_optimization=False,
    optimization_every=20,
):
    s = setup_problem(load_config(config_path), with_environment=True)
    spec = s.env.get_specification(s.cfg["H"])
    on_iteration, finish_window = _optimization_view(
        s, "Reach-avoid", live_optimization
    )
    result = s.planner.optimize_window(
        s.rollout,
        spec=spec,
        init_guess=s.init_guess,
        verbose=verbose,
        on_iteration=(
            (lambda iteration, record: on_iteration(0, iteration, record))
            if on_iteration is not None
            else None
        ),
        callback_every=optimization_every if on_iteration is not None else 1,
    )
    if finish_window is not None:
        finish_window(0, result)
    if show or save:
        from visualization.animation import animate_reach_avoid
        from visualization.planning import plot_reach_avoid

        plot_reach_avoid(
            result,
            s.env,
            dt=s.cfg["dt"],
            save_path=_output_path("reach_avoid", ".png") if save else None,
            show=show,
        )
        animate_reach_avoid(
            result,
            s.env,
            dt=s.cfg["dt"],
            filename=_output_path("reach_avoid", ".gif") if save else None,
            show=show,
        )
    _save_result(result, "reach_avoid", save)
    return result


def lane_success_counter(environment, mean, counter):
    """Update the lane success streak once per executed step."""
    task = environment.metadata["task"]
    inside = (
        task["target_center"] - task["target_tolerance"]
        <= float(mean[1])
        <= task["target_center"] + task["target_tolerance"]
    )
    return counter + 1 if inside else 0


def _lane_initial_state(s):
    traffic = s.cfg["traffic"]
    mean = torch.tensor(
        [[car["x0"], car["y"], car["speed"], 0.0] for car in traffic],
        dtype=s.dyn.B.dtype,
        device=s.dyn.device,
    )
    covariance = torch.eye(4, device=s.dyn.device).expand(len(traffic), -1, -1)
    covariance = covariance.clone() * s.cfg["traffic_cov_scale"]
    return (*s.state, 0, mean, covariance, None)


def _lane_initial_controls(s):
    warm = s.cfg["warm_start"]
    controls = torch.zeros(
        s.cfg["H"], 2, dtype=s.dyn.B.dtype, device=s.dyn.device
    )
    controls[:, 0] = warm["longitudinal"]
    switch = warm["switch_step"]
    controls[:switch, 1] = warm["lateral"]
    controls[switch : 2 * switch, 1] = -warm["lateral"]
    return controls


def _live_observer(s, *, lane, live, initial_state, label):
    if not live:
        return None
    from visualization.live_plots import create_live_view

    _, _, observer = create_live_view(
        s.env,
        initial_state,
        dt=s.cfg["dt"],
        lane=lane,
        dynamics=s.dyn,
        max_iters=s.planner.cfg["max_iters"],
        label=label,
    )
    return observer


def _execution_observers(
    s,
    *,
    label,
    lane,
    live,
    live_optimization,
    optimization_every,
    initial_state,
):
    on_execution = _live_observer(
        s,
        lane=lane,
        live=live or live_optimization,
        initial_state=initial_state,
        label=label,
    )
    on_iteration = (
        on_execution.on_iteration if on_execution is not None else None
    )
    finish_window = (
        on_execution.finish_window if on_execution is not None else None
    )

    def on_step(step, state, plan):
        if finish_window is not None:
            finish_window(step, plan)
        if on_execution is not None:
            on_execution(step, state, plan)

    return (
        on_step
        if on_execution is not None or finish_window is not None
        else None,
        on_iteration,
        optimization_every if on_iteration is not None else 1,
    )


def _present_execution(result, s, *, stem, lane, show, save):
    if show or save:
        from visualization.planning import plot_lane_merge, plot_mpc_execution

        plot = plot_lane_merge if lane else plot_mpc_execution
        plot(
            result,
            s.env,
            dt=s.cfg["dt"],
            save_path=_output_path(stem, ".png") if save else None,
            show=show,
        )
    if show or save:
        from visualization.animation import animate_mpc

        animate_mpc(
            result,
            s.env,
            dt=s.cfg["dt"],
            filename=_output_path(stem, ".gif") if save else None,
            lane=lane,
            show=show,
        )


def run_mpc(
    config_path="configs/scenarios/reach_avoid.yaml",
    *,
    show=False,
    save=True,
    live=False,
    live_optimization=False,
    optimization_every=20,
    verbose=False,
    max_steps=None,
):
    s = setup_problem(load_config(config_path), with_environment=True)
    options = s.cfg["mpc"]
    s.planner = Planner(
        s.dyn,
        options.get("horizon", s.cfg["H"]),
        {**s.cfg.get("planner", {}), **options.get("planner", {})},
    )
    init_guess = (
        s.init_guess[: s.planner.horizon] if s.init_guess is not None else None
    )
    torch.manual_seed(options["seed"])
    goal = s.env.single_region("goal")
    on_step, on_iteration, callback_every = _execution_observers(
        s,
        label="MPC",
        lane=False,
        live=live,
        live_optimization=live_optimization,
        optimization_every=optimization_every,
        initial_state=s.state,
    )

    def is_done(state, step):
        mean = state[0]
        return bool(
            goal.xmin <= mean[0] <= goal.xmax
            and goal.ymin <= mean[1] <= goal.ymax
        )

    result = s.planner.run_receding_horizon(
        s.state,
        make_rollout=lambda state, step: gaussian_rollout(s.dyn, *state),
        make_spec=lambda state, step: s.env.get_specification(
            s.planner.horizon
        ),
        execute=lambda state, control, step: s.dyn.sample_step(
            *state, control
        ),
        is_done=is_done,
        max_steps=options["max_steps"] if max_steps is None else max_steps,
        init_guess=init_guess,
        verbose=verbose,
        on_step=on_step,
        on_iteration=on_iteration,
        callback_every=callback_every,
    )
    _present_execution(result, s, stem="mpc", lane=False, show=show, save=save)
    _save_result(result, "mpc", save)
    return result


def run_lane_change(
    config_path="configs/scenarios/lane_change.yaml",
    *,
    show=False,
    save=True,
    live=False,
    live_optimization=False,
    optimization_every=20,
    verbose=False,
    max_steps=None,
):
    s = setup_problem(load_config(config_path), with_environment=True)
    if "seed" in s.cfg:
        torch.manual_seed(s.cfg["seed"])
    initial_state = _lane_initial_state(s)
    traffic_q = s.cfg["traffic_q_std"] ** 2 * torch.eye(
        4, dtype=s.dyn.B.dtype, device=s.dyn.device
    )
    acceleration = s.cfg["accel_bounds"]
    accel_bounds = (
        acceleration["longitudinal"],
        acceleration["lateral"],
    )
    names = [car["name"] for car in s.cfg["traffic"]]
    scenario_label = (
        "Lane merge" if s.env.metadata.get("ramp") else "Lane change"
    )
    on_step, on_iteration, callback_every = _execution_observers(
        s,
        label=scenario_label,
        lane=True,
        live=live,
        live_optimization=live_optimization,
        optimization_every=optimization_every,
        initial_state=initial_state,
    )

    def execute(state, control, step):
        mean, covariance = s.dyn.sample_step(state[0], state[1], control)
        mean = mean.clone()
        mean[2] = mean[2].clamp(*s.cfg["speed_bounds"])
        counter = lane_success_counter(s.env, mean, state[2])
        entry = step + 1 if counter == 1 else state[5] if counter else None
        traffic_mean = state[3] @ s.dyn.A.T + (
            torch.randn_like(state[3]) * s.cfg["traffic_q_std"]
        )
        traffic_cov = s.dyn.A @ state[4] @ s.dyn.A.T + traffic_q
        if verbose and step % 5 == 0:
            logger.info("Lane step %d: position %s", step, mean[:2].tolist())
        return mean, covariance, counter, traffic_mean, traffic_cov, entry

    def is_done(state, step):
        task = s.env.metadata["task"]
        if not lane_contains_point(
            s.env, float(state[0][0]), float(state[0][1])
        ):
            return "road_violation"
        delta = state[3][:, :2] - state[0][:2]
        collision = s.env.metadata["collision"]
        if bool(
            (
                (delta[:, 0].abs() <= collision["longitudinal"])
                & (delta[:, 1].abs() <= collision["lateral"])
            ).any()
        ):
            return "collision"
        start, end = task["start_end_steps"]
        witness = max(state[5], start) if state[5] is not None else None
        if (
            witness is not None
            and witness <= end
            and step >= witness + task["dwell_steps"]
        ):
            return "goal_reached"
        if step > end and (state[2] == 0 or state[5] > end):
            return "deadline_missed"
        if step > end + task["dwell_steps"]:
            return "deadline_missed"
        return None

    result = s.planner.run_receding_horizon(
        initial_state,
        make_rollout=lambda state, step: lane_rollout(
            s.dyn,
            state[0],
            state[1],
            state[3],
            state[4],
            traffic_q,
            names,
            s.cfg["speed_bounds"],
            accel_bounds,
        ),
        make_spec=lambda state, step: lane_local_window(
            s.env, step, state[0], s.cfg, streak=state[2]
        ).get_specification(s.cfg["H"]),
        execute=execute,
        is_done=is_done,
        max_steps=s.cfg["T_SIM"] if max_steps is None else max_steps,
        init_guess=_lane_initial_controls(s),
        verbose=verbose,
        on_step=on_step,
        on_iteration=on_iteration,
        callback_every=callback_every,
    )
    stem = Path(config_path).stem
    final_bound = (
        result.window_plans[-1].hard_interval[0]
        if result.window_plans
        else float("nan")
    )
    print(
        f"{scenario_label}: {result.stopped_reason} after "
        f"{len(result.window_plans)} steps; "
        f"last hard lower bound {final_bound:.4f}",
        flush=True,
    )
    _present_execution(result, s, stem=stem, lane=True, show=show, save=save)
    _save_result(result, stem, save)
    return result
