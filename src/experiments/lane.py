"""Lane change and on-ramp merge: receding-horizon pdSTL planning in traffic.

The road, traffic, task and limits come from a YAML in
configs/scenarios/lane/.
"""

import logging
from pathlib import Path

import torch

from models.rollouts import lane_rollout
from planning.environment import (
    build_lane_merge_environment,
    lane_contains_footprint,
    lane_deadline_missed,
    lane_goal_reached,
    lane_has_collision,
    lane_local_window,
    lane_target_contains,
)
from planning.runners import output_path, save_result, setup_problem
from utils import load_config

logger = logging.getLogger(__name__)


def build(cfg, device=None):
    """Dynamics, planner and road environment for one lane config."""
    s = setup_problem(cfg, device=device)
    s.env = build_lane_merge_environment(cfg, s.dyn.device)
    return s


def run(
    config_path,
    *,
    show=True,
    save=True,
    live=False,
    live_optimization=False,
    optimization_every=20,
    verbose=False,
    max_steps=None,
):
    """Execute one lane scenario closed loop, then plot and save it.

    Args:
        config_path: Lane YAML; outputs are named after its file stem.
        show, save: Display and/or save the figures and animation.
        live, live_optimization, optimization_every: Live execution view.
        verbose: Log progress.
        max_steps: Override the configured number of executed steps.

    Returns:
        The MPCResult.
    """
    s = build(load_config(config_path))
    if "seed" in s.cfg:
        torch.manual_seed(s.cfg["seed"])
    initial_state = initial_state_of(s)
    label = "Lane merge" if s.env.metadata.get("ramp") else "Lane change"
    on_step, on_iteration, callback_every = _observers(
        s,
        label=label,
        live=live or live_optimization,
        optimization_every=optimization_every,
        initial_state=initial_state,
    )
    result = trial(
        s,
        initial_state=initial_state,
        max_steps=max_steps,
        verbose=verbose,
        on_step=on_step,
        on_iteration=on_iteration,
        callback_every=callback_every,
    )
    lower, upper = (
        result.window_plans[-1].hard_interval
        if result.window_plans
        else (float("nan"), float("nan"))
    )
    print(
        f"{label}: {result.stopped_reason} after "
        f"{len(result.window_plans)} steps; last window "
        f"[rho_lower, rho_upper] = [{lower:.4f}, {upper:.4f}]",
        flush=True,
    )
    stem = Path(config_path).stem
    if show or save:
        _present(result, s, stem=stem, show=show, save=save)
    save_result(result, stem, save)
    return result


def trial(
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
    """Run one closed-loop trial, optionally with fixed disturbances."""
    initial_state = (
        initial_state_of(s) if initial_state is None else initial_state
    )
    traffic_q = s.cfg["traffic_q_std"] ** 2 * torch.eye(
        4, dtype=s.dyn.B.dtype, device=s.dyn.device
    )
    acceleration = s.cfg["accel_bounds"]
    accel_bounds = (acceleration["longitudinal"], acceleration["lateral"])
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
        make_spec=lambda state, step: _specification(
            s, make_spec, state, step
        ),
        execute=_executor(
            s, traffic_q, ego_disturbances, traffic_disturbances, verbose
        ),
        is_done=_outcome(s),
        max_steps=s.cfg["T_SIM"] if max_steps is None else max_steps,
        init_guess=initial_controls(s),
        verbose=verbose,
        on_step=on_step,
        on_iteration=on_iteration,
        callback_every=callback_every,
        capture_planning_failures=True,
    )
    if result.stopped_reason == "max_steps":
        result.stopped_reason = "step_limit"
    return result


def initial_state_of(s):
    """(ego mean, ego cov, target streak, traffic means, traffic covs, entry)."""
    traffic = s.cfg["traffic"]
    mean = torch.tensor(
        [[car["x0"], car["y"], car["speed"], 0.0] for car in traffic],
        dtype=s.dyn.B.dtype,
        device=s.dyn.device,
    )
    covariance = torch.eye(4, device=s.dyn.device).expand(len(traffic), -1, -1)
    covariance = covariance.clone() * s.cfg["traffic_cov_scale"]
    return (*s.state, 0, mean, covariance, None)


def initial_controls(s):
    """The configured first-window guess: accelerate and steer into the lane."""
    warm = s.cfg["warm_start"]
    controls = torch.zeros(
        s.cfg["H"], 2, dtype=s.dyn.B.dtype, device=s.dyn.device
    )
    controls[:, 0] = warm["longitudinal"]
    switch = warm["switch_step"]
    controls[:switch, 1] = warm["lateral"]
    controls[switch : 2 * switch, 1] = -warm["lateral"]
    return controls


def success_counter(environment, mean, counter):
    """Consecutive executed steps inside the target lane."""
    inside = lane_target_contains(environment.metadata, mean)
    return counter + 1 if inside else 0


def _observers(s, *, label, live, optimization_every, initial_state):
    """Live-view callbacks, or (None, None, 1) without a live view."""
    if not live:
        return None, None, 1
    from visualization.live_plots import create_live_view

    _, _, observer = create_live_view(
        s.env,
        initial_state,
        dt=s.cfg["dt"],
        dynamics=s.dyn,
        max_iters=s.planner.cfg["max_iters"],
        label=label,
    )

    def on_step(step, state, plan):
        observer.finish_window(step, plan)
        observer(step, state, plan)

    return on_step, observer.on_iteration, optimization_every


def _present(result, s, *, stem, show, save):
    from visualization.animation import animate_mpc
    from visualization.figures import plot_lane_merge

    plot_lane_merge(
        result,
        s.env,
        dt=s.cfg["dt"],
        save_path=output_path(stem, ".png") if save else None,
        show=show,
    )
    animate_mpc(
        result,
        s.env,
        dt=s.cfg["dt"],
        filename=output_path(stem, ".gif") if save else None,
        show=show,
    )


def _executor(s, traffic_q, ego_disturbances, traffic_disturbances, verbose):
    def execute(state, control, step):
        mean, covariance = s.dyn.step(state[0], state[1], control)
        if ego_disturbances is None:
            noise = torch.distributions.MultivariateNormal(
                torch.zeros_like(mean), s.dyn.Q
            ).sample()
        else:
            noise = ego_disturbances[step].to(mean)
        mean = mean + noise
        mean[2] = mean[2].clamp(*s.cfg["speed_bounds"])
        counter = success_counter(s.env, mean, state[2])
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


def _outcome(s):
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


def _specification(s, factory, state, step):
    environment = lane_local_window(s.env, step, streak=state[2])
    return (
        factory(environment, state, step)
        if factory is not None
        else environment.get_specification(s.cfg["H"])
    )
