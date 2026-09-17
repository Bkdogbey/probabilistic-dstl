"""Scenario wiring: load configuration, build the pieces, run one controller, report.

Thin by design. Everything here is construction and dispatch; no planning mathematics, no
result reshaping, no plotting logic. `PlanResult` and `RecedingHorizonResult` are returned
exactly as the planner and controller produced them.
"""

from pathlib import Path

import torch

from models.dynamics import DoubleIntegrator, SingleIntegrator
from planning import log_utils
from planning.controllers import RecedingHorizonController, normalize_controller_type
from planning.environment import Environment
from planning.scenarios.lane_merge import LaneMergeEnvironment
from utils import get_device, load_config

RESULTS_DIR = Path(__file__).resolve().parents[2] / "outputs"


def output_path(filename):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(RESULTS_DIR / filename)


# --- Construction ---------------------------------------------------------------


def build_dynamics(config, device):
    """Dynamics from the `dynamics:` block."""
    common = {
        "dt": config["dt"], "u_max": config["u_max"],
        "q_std": config["q_std"], "device": device,
    }
    if config.get("type", "single_integrator") == "double_integrator":
        return DoubleIntegrator(**common)
    return SingleIntegrator(**common, state_dim=config.get("state_dim", 2))


def build_initial_belief(config, device):
    """(mean [D], covariance [D, D]) from the `initial_belief:` block."""
    mean = torch.tensor(config["mean"], device=device, dtype=torch.float32)
    covariance = config["covariance"]
    if isinstance(covariance, (int, float)):
        covariance = torch.eye(len(mean), device=device) * float(covariance)
    else:
        covariance = torch.tensor(covariance, device=device, dtype=torch.float32)
        if covariance.ndim == 1:
            covariance = torch.diag(covariance)
    return mean, covariance


def initialize_toward_goal(*, mean0, goal, dynamics, horizon):
    """Constant control along the straight line from the initial mean to the goal centre.

    Deliberately simple, and deliberately not optional for the gate corridor: the pdSTL goal
    gradient is only alive within a few sigma of the goal rectangle, because further out the
    Gaussian CDF underflows to exactly zero in float32. This initialisation is what carries
    the plan into the region where the specification can steer it.
    """
    if goal is None:
        return None
    center = torch.tensor(
        [sum(goal["x"]) / 2.0, sum(goal["y"]) / 2.0],
        device=mean0.device, dtype=mean0.dtype,
    )
    offset = center - mean0[:2]
    distance = torch.linalg.norm(offset)
    if float(distance) == 0.0:
        return torch.zeros(horizon, dynamics.B.shape[1], device=mean0.device)

    speed = min(float(dynamics.u_max), float(distance) / (horizon * dynamics.dt))
    step = offset / distance * speed
    control = torch.zeros(dynamics.B.shape[1], device=mean0.device, dtype=mean0.dtype)
    control[: step.shape[0]] = step
    return control.repeat(horizon, 1)


SCENARIOS = {"reach_avoid": Environment, "lane_merge": LaneMergeEnvironment}
SCENARIO_ALIASES = {"lane_change": "lane_merge"}


def build_scenario(config, device):
    """Environment for the configured scenario type."""
    scenario = config.get("scenario") or {}
    name = SCENARIO_ALIASES.get(scenario.get("type", "reach_avoid"), scenario.get("type", "reach_avoid"))
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario type {name!r}; known: {sorted(SCENARIOS)}")
    if name == "lane_merge":
        return LaneMergeEnvironment.from_config(
            config, device=device, window_config=_window_config(config),
        )
    return Environment.from_config(config["environment"], device=device)


def build_planner(config, environment):
    """One reusable Planner. Every window, single-shot or receding, goes through it."""
    from planning.planner import Planner  # local: keeps the import graph shallow

    device = environment.device
    dynamics = build_dynamics(config["dynamics"], device)
    return Planner(dynamics, environment, config["horizon"], config=config.get("optimizer"))


# --- Runners --------------------------------------------------------------------


def run_reach_avoid(config_path="configs/scenarios/reach_avoid.yaml", *, show=True, save=True):
    """Gate-corridor reach-avoid, single-shot or receding-horizon as the config selects."""
    config = load_config(config_path)
    device = get_device()
    log_utils.log_device(device)

    environment = build_scenario(config, device)
    planner = build_planner(config, environment)
    mean0, covariance0 = build_initial_belief(config["initial_belief"], device)
    initial_controls = initialize_toward_goal(
        mean0=mean0, goal=environment.goal,
        dynamics=planner.dynamics, horizon=config["horizon"],
    )

    controller = normalize_controller_type((config.get("controller") or {}).get("type", "single_shot"))
    if controller == "single_shot":
        result = planner.solve(mean0, covariance0, initial_controls=initial_controls)
    elif controller == "receding_horizon":
        controller_config = config["controller"]
        result = RecedingHorizonController(
            planner=planner, dynamics=planner.dynamics,
            scenario=environment, config=controller_config,
        ).run(
            mean0, (mean0, covariance0),
            max_steps=controller_config["max_steps"],
            apply_steps=controller_config.get("apply_steps", 1),
            initial_controls=initial_controls,
        )
    else:
        raise ValueError(f"unknown controller type {controller!r}")

    log_utils.log_plan_summary(config["scenario"].get("name", "reach_avoid"), result)

    if save or show:
        from visualization.planning import visualize_execution, visualize_plan

        figure_path = output_path(config["output"]["figure"]) if save else None
        if controller == "single_shot":
            visualize_plan(
                result, environment, config.get("visualization") or {},
                initial_controls=initial_controls, save_path=figure_path, show=show,
            )
        else:
            visualize_execution(
                result, environment, config.get("visualization") or {},
                save_path=figure_path, show=show,
            )
    return result


def run_lane_merge(config_path="configs/scenarios/lane_change.yaml", *, show=True, save=True):
    """Existing lane-merge scenario, executed by the generic receding-horizon controller."""
    config = load_config(config_path)
    device = get_device()
    log_utils.log_device(device)
    log_utils.log_scenario_start(config.get("label", "lane merge"))

    environment = build_scenario({**config, "scenario": {"type": "lane_merge"}}, device)
    planner = build_planner(
        {
            **config,
            "horizon": config["H"],
            "dynamics": _legacy_dynamics_block(config),
            "optimizer": _legacy_optimizer_block(config),
        },
        environment,
    )
    mean0, covariance0 = build_initial_belief(
        {"mean": config["x0_mean"], "covariance": config["x0_cov_scale"]}, device
    )
    controller_config = config.get("controller") or {"max_steps": config["T_SIM"]}
    result = RecedingHorizonController(
        planner=planner, dynamics=planner.dynamics, scenario=environment, config=controller_config,
    ).run(
        mean0, (mean0, covariance0),
        max_steps=controller_config.get("max_steps", config["T_SIM"]),
        apply_steps=controller_config.get("apply_steps", 1),
    )
    log_utils.log_plan_summary(config.get("label", "lane_merge"), result)

    if show or save:
        from visualization.planning import visualize_lane_merge

        visualize_lane_merge(result, environment, config, show=show, save=save)
    return result


# --- Legacy lane-merge configuration --------------------------------------------
# lane_change*.yaml predate the scenario/controller/optimizer schema. Rather than rewriting
# those files (and moving lane-merge behaviour with them), map their keys onto the new ones.

LEGACY_OPTIMIZER_KEYS = {
    "lr": "learning_rate",
    "max_iters": "max_iterations",
    "alpha": "probability_target",
    "converge_patience": "convergence_patience",
}
LEGACY_LOSS_KEYS = {
    "w_phi": "pdstl_weight",
    "w_u": "control_effort_weight",
    "w_du": "control_smoothness_weight",
    "w_dist": "terminal_goal_weight",
}


def _legacy_dynamics_block(config):
    """Lane-merge configs predate the `dynamics:` block and keep these keys at top level."""
    return {
        "type": config.get("dynamics", "single_integrator"),
        "dt": config["dt"], "u_max": config["u_max"], "q_std": config["q_std"],
    }


def _legacy_planner_config(config):
    """planning.yaml defaults overlaid with the scenario's own `planner:` block."""
    return {**load_config("configs/planning.yaml"), **(config.get("planner") or {})}


def _window_config(config):
    """Local-window geometry and shaping parameters for the lane-merge scenario."""
    return {**_legacy_planner_config(config), **(config.get("window") or {})}


def _legacy_optimizer_block(config):
    """Translate a legacy `planner:` block into the new `optimizer:` schema."""
    legacy = _legacy_planner_config(config)
    optimizer = {new: legacy[old] for old, new in LEGACY_OPTIMIZER_KEYS.items() if old in legacy}
    optimizer["loss"] = {new: legacy[old] for old, new in LEGACY_LOSS_KEYS.items() if old in legacy}
    optimizer["smoothing"] = dict(legacy.get("smoothing") or {})
    # `extras` reaches the scenario's extra_loss hook; lane merge reads w_obs and obs_margin.
    optimizer.update({k: legacy[k] for k in ("w_obs", "obs_margin") if k in legacy})
    return optimizer


# `mpc` was the old name for receding-horizon execution; both route here.
run_lane_change = run_lane_merge
