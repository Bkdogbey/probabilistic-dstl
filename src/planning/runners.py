"""Load a problem, call the canonical planner, and optionally save or visualize."""

import logging
from pathlib import Path
from types import SimpleNamespace

import torch

from models.dynamics import DoubleIntegrator, SingleIntegrator
from models.rollouts import gaussian_rollout
from pdstl.operators import Always
from pdstl.predicates import GreaterThan
from planning.environment import (
    build_lane_merge_environment, build_reach_avoid_environment, lane_local_window,
)
from planning.planner import Planner
from utils import get_device, load_config

logger = logging.getLogger(__name__)
RESULTS_DIR = Path(__file__).resolve().parents[2] / "outputs"


def build_dynamics(cfg, device):
    common = {key: cfg[key] for key in ("dt", "u_max", "q_std")}
    if cfg.get("dynamics", "single_integrator") == "double_integrator":
        return DoubleIntegrator(**common, device=device)
    return SingleIntegrator(**common, device=device, state_dim=cfg.get("state_dim", 2))


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
    """Shared setup for runners and demonstrations; no planning/evaluation here."""
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
        cfg=cfg, dyn=dyn, state=state, planner=planner, init_guess=guess,
        rollout=gaussian_rollout(dyn, *state),
        env=build_environment(cfg, device) if with_environment else None,
    )


def _output_path(filename):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(RESULTS_DIR / filename)


def _save(result, cfg, save):
    if save and cfg.get("save_file"):
        torch.save(result, _output_path(cfg["save_file"]))


def run_altitude_safety(config_path="configs/scenarios/altitude_safety.yaml", *,
                        show=True, save=True, verbose=False):
    from visualization.animation import animate_altitude_optimization
    from visualization.planning import plot_altitude_safety

    s = setup_problem(load_config(config_path))
    spec = Always(GreaterThan(s.cfg["threshold"], dim=0), interval=[1, s.cfg["H"]])
    initial = (s.planner.evaluate_controls(s.rollout, s.init_guess, spec=spec)
               if show or save else None)
    frames = []
    result = s.planner.optimize_window(
        s.rollout, spec=spec, init_guess=s.init_guess, verbose=verbose,
        on_iteration=(lambda _, plan: frames.append(plan)) if save else None,
    )
    if show or save:
        args = dict(dt=s.cfg["dt"], threshold=s.cfg["threshold"], u_max=s.dyn.u_max)
        plot_altitude_safety(result, initial=initial, **args,
                             save_path=_output_path(s.cfg["figure"]) if save else None,
                             show=show)
        if save:
            animate_altitude_optimization(
                frames, initial=initial, **args,
                filename=_output_path(s.cfg["animation"]["filename"]),
                fps=s.cfg["animation"]["fps"],
            )
    _save(result, s.cfg, save)
    return result


def run_reach_avoid(config_path="configs/scenarios/reach_avoid.yaml", *,
                    show=True, save=True, verbose=False):
    from visualization.planning import visualize_reach_avoid

    s = setup_problem(load_config(config_path), with_environment=True)
    spec = s.env.get_specification(s.cfg["H"])
    result = s.planner.optimize_window(s.rollout, spec=spec, init_guess=s.init_guess,
                                       verbose=verbose)
    _save(result, s.cfg, save)
    if show or save:
        visualize_reach_avoid(
            result, s.env, dt=s.cfg["dt"], ellipse_every=s.cfg.get("ellipse_every", 10),
            save_path=_output_path(s.cfg["figure"]) if save else None, show=show,
        )
    return result


def lane_success_counter(environment, mean, counter):
    """Update the lane success streak once per executed step."""
    success = environment.metadata["success"]
    inside = success["y_min"] <= float(mean[1]) <= success["y_max"]
    return counter + 1 if inside else 0


def run_mpc(config_path="configs/scenarios/mpc.yaml", *, show=True, save=True,
            verbose=False):
    from visualization.planning import visualize_mpc
    from visualization.live_plots import make_mpc_live_callback

    s = setup_problem(load_config(config_path), with_environment=True)
    if "seed" in s.cfg:
        torch.manual_seed(s.cfg["seed"])
    goal = s.env.single_region("goal")

    def is_done(state, step):
        mean = state[0]
        return bool(goal.xmin <= mean[0] <= goal.xmax and
                    goal.ymin <= mean[1] <= goal.ymax)

    result = s.planner.run_receding_horizon(
        s.state, make_rollout=lambda state, step: gaussian_rollout(s.dyn, *state),
        make_spec=lambda state, step: s.env.get_specification(s.cfg["H"]),
        execute=lambda state, control, step: s.dyn.sample_step(*state, control),
        is_done=is_done, max_steps=s.cfg["MAX_STEPS"], init_guess=s.init_guess,
        verbose=verbose, on_step=make_mpc_live_callback(s.env) if show else None,
    )
    _save(result, s.cfg, save)
    if show or save:
        visualize_mpc(result, s.env, s.cfg, show=show, save=save, output_dir=RESULTS_DIR)
    return result


def run_lane_change(config_path="configs/scenarios/lane_change.yaml", *,
                    show=True, save=True, verbose=False):
    from visualization.planning import visualize_mpc
    from visualization.live_plots import make_lane_change_live_callback

    s = setup_problem(load_config(config_path), with_environment=True)
    if "seed" in s.cfg:
        torch.manual_seed(s.cfg["seed"])

    def execute(state, control, step):
        mean, covariance = s.dyn.sample_step(state[0], state[1], control)
        counter = lane_success_counter(s.env, mean, state[2])
        if verbose and step % 5 == 0:
            logger.info("Lane step %d: position %s", step, mean[:2].tolist())
        return mean, covariance, counter

    result = s.planner.run_receding_horizon(
        (*s.state, 0),
        make_rollout=lambda state, step: gaussian_rollout(s.dyn, state[0], state[1]),
        make_spec=lambda state, step: lane_local_window(
            s.env, step, state[0], s.cfg).get_specification(s.cfg["H"]),
        execute=execute,
        is_done=lambda state, step: state[2] >= s.cfg["success"]["consecutive_steps"],
        max_steps=s.cfg["T_SIM"], init_guess=s.init_guess, verbose=verbose,
        on_step=make_lane_change_live_callback(s.env) if show else None,
    )
    _save(result, s.cfg, save)
    if show or save:
        visualize_mpc(result, s.env, s.cfg, show=show, save=save,
                      output_dir=RESULTS_DIR, lane=True)
    return result
