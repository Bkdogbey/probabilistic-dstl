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
    lane_contains_footprint,
    lane_deadline_missed,
    lane_goal_reached,
    lane_has_collision,
    lane_local_window,
    lane_target_contains,
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
    """Gaussian x0: `x0_cov_scale` is one variance or one per state."""
    mean = torch.tensor(cfg["x0_mean"], device=device, dtype=torch.float32)
    variance = torch.as_tensor(
        cfg["x0_cov_scale"], device=device, dtype=torch.float32
    )
    return mean, torch.diag(variance.expand(len(mean)))


def build_environment(cfg, device="cpu"):
    kind = cfg.get("scenario", {}).get("type", "reach_avoid")
    if kind == "reach_avoid":
        return build_reach_avoid_environment(cfg)
    if kind == "lane_merge":
        return build_lane_merge_environment(cfg, device)
    raise ValueError(f"unknown scenario type {kind!r}")


def _initial_controls(cfg, dyn):
    """Repeat the configured constant control, or start from zero."""
    guess = cfg.get("init_control")
    if guess is None:
        return None
    controls = torch.as_tensor(guess, device=dyn.device, dtype=dyn.B.dtype)
    return controls.repeat(cfg["H"], 1) if controls.ndim == 1 else controls


def _route_reference(cfg, route, dtype, arrival=0.8):
    """Positions along x0 -> route at constant speed, done at `arrival` * H."""
    points = torch.tensor([cfg["x0_mean"][:2], *route], dtype=dtype)
    segments = points[1:] - points[:-1]
    ends = torch.cat((points.new_zeros(1), segments.norm(dim=1).cumsum(0)))
    progress = (torch.arange(cfg["H"] + 1) / (arrival * cfg["H"])).clamp(max=1)
    distance = ends[-1] * progress
    index = (torch.searchsorted(ends, distance, right=True) - 1).clamp(
        0, len(segments) - 1
    )
    length = (ends[index + 1] - ends[index]).clamp_min(1e-9)
    fraction = ((distance - ends[index]) / length).unsqueeze(1)
    return points[index] + fraction * segments[index]


def route_controls(cfg, dyn, route, kp=4.0, kd=4.0):
    """Controls that follow a waypoint route: a warm start for one route.

    Args:
        cfg: Scenario config (x0_mean, H, dt).
        dyn: Single- or double-integrator dynamics.
        route: Waypoints [[x, y], ...] after the start.
        kp, kd: PD gains of the double-integrator tracker.

    Returns:
        [H, 2] controls within u_max.
    """
    reference = _route_reference(cfg, route, dyn.B.dtype)
    velocity = (reference[1:] - reference[:-1]) / cfg["dt"]
    if isinstance(dyn, SingleIntegrator):
        return velocity.clamp(-dyn.u_max, dyn.u_max).to(dyn.device)
    state = torch.tensor(cfg["x0_mean"], dtype=dyn.B.dtype)
    controls = []
    for target, target_velocity in zip(reference[1:], velocity):
        error = kp * (target - state[:2]) + kd * (target_velocity - state[2:])
        control = error.clamp(-dyn.u_max, dyn.u_max)
        state = dyn.A.cpu() @ state + dyn.B.cpu() @ control
        controls.append(control)
    return torch.stack(controls).to(dyn.device)


def initial_guesses(cfg, dyn):
    """One warm start per configured route, else the constant guess."""
    routes = cfg.get("routes")
    if routes:
        return [route_controls(cfg, dyn, route) for route in routes.values()]
    return [_initial_controls(cfg, dyn)]


def setup_problem(cfg, *, device=None, with_environment=False):
    """Construct a dynamics, initial belief, planner, rollout, and optional world."""
    device = get_device() if device is None else device
    dyn = build_dynamics(cfg, device)
    state = build_initial_belief(cfg, device)
    planner = Planner(dyn, cfg["H"], cfg.get("planner", {}))
    environment = build_environment(cfg, device) if with_environment else None
    guesses = initial_guesses(cfg, dyn)
    return SimpleNamespace(
        cfg=cfg,
        dyn=dyn,
        state=state,
        planner=planner,
        init_guess=guesses[0],
        init_guesses=guesses,
        rollout=gaussian_rollout(dyn, *state),
        env=environment,
    )


def _output_path(stem, suffix):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{stem}{suffix}"


def _save_result(result, stem, save):
    if save:
        torch.save(result, _output_path(stem, ".pt"))


def _control_unit(dyn):
    return "m/s²" if isinstance(dyn, DoubleIntegrator) else "m/s"


def _optimization_view(s, label, enabled):
    if not enabled:
        return None, None
    from visualization.live_plots import create_optimization_view

    _, _, on_iteration, finish_window = create_optimization_view(
        label,
        max_iters=s.planner.cfg["max_iters"],
        control_unit=_control_unit(s.dyn),
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
    config_path,
    *,
    show=False,
    save=True,
    verbose=False,
    live=False,
    optimization_every=5,
):
    """Plan a reach-avoid scenario; keep the best-certified route.

    Args:
        config_path: Scenario YAML; outputs are named after its stem.
        show, save: Display and/or save the figure and animation.
        verbose: Log optimizer progress.
        live, optimization_every: Stream candidates to a live view.

    Returns:
        The PlanResult with the highest exact lower bound.
    """
    config = load_config(config_path)
    stem = Path(config_path).stem
    s = setup_problem(config, with_environment=True)
    spec = s.env.get_specification(s.cfg["H"])
    title = config.get("name", stem)
    visual = config.get("visualization", {})
    on_iteration, finish_live = None, None
    if live:
        from visualization.live_plots import create_reach_avoid_live_view

        _, _, on_iteration, finish_live = create_reach_avoid_live_view(
            s.env,
            lambda controls: s.planner.evaluate_controls(
                s.rollout, controls, spec=spec
            ),
            dt=s.cfg["dt"],
            title=title,
            max_iters=s.planner.cfg["max_iters"],
            ellipse_every=visual.get("ellipse_every", 4),
            alpha=s.planner.cfg["alpha"],
        )
    result, plans = s.planner.optimize_multistart(
        s.rollout,
        spec=spec,
        init_guesses=s.init_guesses,
        verbose=verbose,
        on_iteration=on_iteration,
        callback_every=optimization_every if on_iteration is not None else 1,
    )
    if finish_live is not None:
        finish_live(result)
    routes = config.get("routes") or {"initial": None}
    scores = {
        name: round(plan.hard_interval[0], 4)
        for name, plan in zip(routes, plans)
    }
    print(f"{title}: exact lower bound per route {scores}", flush=True)
    if show or save:
        from visualization.animation import animate_reach_avoid
        from visualization.planning import plot_reach_avoid

        plot_reach_avoid(
            result,
            s.env,
            dt=s.cfg["dt"],
            u_max=s.dyn.u_max,
            control_unit=_control_unit(s.dyn),
            title=title,
            ellipse_every=visual.get("ellipse_every", 4),
            ellipse_confidence=visual.get("ellipse_confidence", 0.95),
            alternatives=[plan for plan in plans if plan is not result],
            save_path=_output_path(stem, ".png") if save else None,
            show=show,
        )
        animate_reach_avoid(
            result,
            s.env,
            dt=s.cfg["dt"],
            title=title,
            fps=visual.get("animation_fps", 6),
            filename=_output_path(stem, ".gif") if save else None,
            show=show,
        )
    _save_result(result, stem, save)
    return result


def lane_success_counter(environment, mean, counter):
    """Update the lane success streak once per executed step."""
    inside = lane_target_contains(environment.metadata, mean)
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


def _present_lane_execution(result, s, *, stem, show, save):
    if show or save:
        from visualization.planning import plot_lane_merge

        if save:
            for obsolete in ("controls", "scores"):
                for suffix in (".png", ".pdf"):
                    _output_path(f"{stem}_{obsolete}", suffix).unlink(
                        missing_ok=True
                    )
        plot_lane_merge(
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
            lane=True,
            show=show,
        )


def _lane_executor(
    s, traffic_q, ego_disturbances, traffic_disturbances, verbose
):
    def execute(state, control, step):
        mean, covariance = s.dyn.step(state[0], state[1], control)
        if ego_disturbances is None:
            noise = torch.distributions.MultivariateNormal(
                torch.zeros_like(mean), s.dyn.Q
            ).sample()
        else:
            noise = ego_disturbances[step].to(mean)
        mean = mean + noise
        mean = mean.clone()
        mean[2] = mean[2].clamp(*s.cfg["speed_bounds"])
        counter = lane_success_counter(s.env, mean, state[2])
        entry = step + 1 if counter == 1 else state[5] if counter else None
        if traffic_disturbances is None:
            traffic_noise = torch.randn_like(state[3]) * s.cfg["traffic_q_std"]
        else:
            traffic_noise = traffic_disturbances[step].to(state[3])
        traffic_mean = state[3] @ s.dyn.A.T + traffic_noise
        traffic_cov = s.dyn.A @ state[4] @ s.dyn.A.T + traffic_q
        if verbose and step % 5 == 0:
            logger.info("Lane step %d: position %s", step, mean[:2].tolist())
        return mean, covariance, counter, traffic_mean, traffic_cov, entry

    return execute


def _lane_outcome(s):
    def is_done(state, step):
        if lane_has_collision(s.env, state[0][:2], state[3][:, :2]):
            return "collision"
        if not lane_contains_footprint(
            s.env, float(state[0][0]), float(state[0][1])
        ):
            return "road_violation"
        if lane_goal_reached(s.env, state[0], step, state[2]):
            return "success"
        if lane_deadline_missed(s.env, state[0], step, state[2]):
            return "deadline_missed"
        return None

    return is_done


def _lane_specification(s, factory, state, step):
    environment = lane_local_window(
        s.env, step, state[0], s.cfg, streak=state[2]
    )
    return (
        factory(environment, state, step)
        if factory is not None
        else environment.get_specification(s.cfg["H"])
    )


def run_lane_trial(
    s,
    *,
    initial_state=None,
    ego_disturbances=None,
    traffic_disturbances=None,
    make_spec=None,
    max_steps=None,
    verbose=False,
    on_step=None,
    on_iteration=None,
    callback_every=1,
):
    """Run one lane trial, optionally with predetermined paired disturbances."""
    initial_state = (
        _lane_initial_state(s) if initial_state is None else initial_state
    )
    traffic_q = s.cfg["traffic_q_std"] ** 2 * torch.eye(
        4, dtype=s.dyn.B.dtype, device=s.dyn.device
    )
    acceleration = s.cfg["accel_bounds"]
    accel_bounds = (
        acceleration["longitudinal"],
        acceleration["lateral"],
    )
    names = [car["name"] for car in s.cfg["traffic"]]
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
        make_spec=lambda state, step: _lane_specification(
            s, make_spec, state, step
        ),
        execute=_lane_executor(
            s,
            traffic_q,
            ego_disturbances,
            traffic_disturbances,
            verbose,
        ),
        is_done=_lane_outcome(s),
        max_steps=s.cfg["T_SIM"] if max_steps is None else max_steps,
        init_guess=_lane_initial_controls(s),
        verbose=verbose,
        on_step=on_step,
        on_iteration=on_iteration,
        callback_every=callback_every,
        capture_planning_failures=True,
    )
    if result.stopped_reason == "max_steps":
        result.stopped_reason = "step_limit"
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
    result = run_lane_trial(
        s,
        initial_state=initial_state,
        max_steps=max_steps,
        verbose=verbose,
        on_step=on_step,
        on_iteration=on_iteration,
        callback_every=callback_every,
    )
    stem = Path(config_path).stem
    final_interval = (
        result.window_plans[-1].hard_interval
        if result.window_plans
        else (float("nan"), float("nan"))
    )
    print(
        f"{scenario_label}: {result.stopped_reason} after "
        f"{len(result.window_plans)} steps; "
        f"final certified hard interval "
        f"[{final_interval[0]:.4f}, {final_interval[1]:.4f}]",
        flush=True,
    )
    _present_lane_execution(result, s, stem=stem, show=show, save=save)
    _save_result(result, stem, save)
    return result
