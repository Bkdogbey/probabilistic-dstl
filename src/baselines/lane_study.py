"""Optional paired deterministic-STL versus pdSTL lane experiment."""

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

# ``planning`` is a deliberately small top-level package name.  When another
# editable research checkout provides the same name, make this study resolve
# its siblings from this distribution before importing them.
_SOURCE_ROOT = str(Path(__file__).resolve().parents[1])
if sys.path[0] != _SOURCE_ROOT:
    sys.path.insert(0, _SOURCE_ROOT)

from baselines.lane_reporting import write_study_outputs  # noqa: E402

import torch  # noqa: E402

from baselines.lane_objective import DeterministicLaneSpecification  # noqa: E402
from planning.environment import lane_target_contains  # noqa: E402
from planning.runners import (  # noqa: E402
    _lane_initial_state,
    run_lane_trial,
    setup_problem,
)
from utils import get_device, load_config  # noqa: E402


METHODS = ("deterministic_stl", "pdstl")


def scaled_config(config, factor):
    """Scale standard deviations and their corresponding initial variances."""
    scaled = deepcopy(config)
    scaled["q_std"] *= factor
    scaled["traffic_q_std"] *= factor
    scaled["x0_cov_scale"] *= factor**2
    scaled["traffic_cov_scale"] *= factor**2
    scaled["planner"] = {**scaled.get("planner", {}), "alpha": None}
    return scaled


def generate_trial_inputs(problem, seed):
    """Sample one initial state and full disturbance sequence for a paired trial."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = _lane_initial_state(problem)
    ego_sample = torch.randn(4, generator=generator).to(state[0])
    traffic_sample = torch.randn(state[3].shape, generator=generator).to(
        state[3]
    )
    ego_mean = state[0] + ego_sample * problem.cfg["x0_cov_scale"] ** 0.5
    traffic_mean = (
        state[3] + traffic_sample * problem.cfg["traffic_cov_scale"] ** 0.5
    )
    streak = int(lane_target_contains(problem.env.metadata, ego_mean))
    initial = (
        ego_mean,
        state[1].clone(),
        streak,
        traffic_mean,
        state[4].clone(),
        0 if streak else None,
    )
    steps = problem.cfg["T_SIM"]
    ego = torch.randn((steps, 4), generator=generator) * problem.cfg["q_std"]
    traffic = (
        torch.randn(
            (steps, len(problem.cfg["traffic"]), 4), generator=generator
        )
        * problem.cfg["traffic_q_std"]
    )
    return initial, ego, traffic


def _clone_state(state):
    return tuple(
        value.clone() if torch.is_tensor(value) else value for value in state
    )


def _minimum_clearance(result, environment):
    ego_size = environment.metadata["ego_vehicle"]
    minimum = float("inf")
    for state in result.states:
        for index, vehicle in enumerate(environment.metadata["traffic"]):
            delta = (state[3][index, :2] - state[0][:2]).abs()
            dx = max(
                float(delta[0]) - (ego_size["width"] + vehicle["width"]) / 2,
                0.0,
            )
            dy = max(
                float(delta[1]) - (ego_size["height"] + vehicle["height"]) / 2,
                0.0,
            )
            minimum = min(minimum, (dx * dx + dy * dy) ** 0.5)
    return minimum


def trial_metrics(result, problem):
    controls = result.applied_controls
    effort = float((controls.square().sum() * problem.cfg["dt"]).cpu())
    smoothness = (
        float(
            (
                (controls[1:] - controls[:-1]).square().sum()
                / problem.cfg["dt"]
            ).cpu()
        )
        if len(controls) > 1
        else 0.0
    )
    return {
        "outcome": result.stopped_reason,
        "completion_time": (
            (len(result.states) - 1) * problem.cfg["dt"]
            if result.stopped_reason == "success"
            else ""
        ),
        "minimum_clearance": _minimum_clearance(result, problem.env),
        "control_effort": effort,
        "control_smoothness": smoothness,
        "planning_time": sum(
            plan.planning_time for plan in result.window_plans
        ),
        "planning_failure_detail": result.failure_detail or "",
    }


def run_study(
    config_path="configs/experiments/lane_baseline.yaml", **overrides
):
    study = {**load_config(config_path), **overrides}
    trials, windows = [], []
    scenario_settings = {}
    base_seed = int(study["seed"])
    requested_device = study.get("device", "auto")
    device = (
        get_device()
        if requested_device == "auto"
        else torch.device(requested_device)
    )
    total_runs = (
        len(study["scenarios"])
        * len(study["uncertainty_factors"])
        * int(study["trials"])
        * len(METHODS)
    )
    completed = 0
    started = perf_counter()
    print(f"Lane study using {device}; {total_runs} planner runs", flush=True)
    for scenario_index, scenario_path in enumerate(study["scenarios"]):
        base = load_config(scenario_path)
        scenario = Path(scenario_path).stem
        scenario_settings[scenario] = base
        for factor in study["uncertainty_factors"]:
            config = scaled_config(base, float(factor))
            template = setup_problem(
                config, device=device, with_environment=True
            )
            for trial in range(int(study["trials"])):
                seed = base_seed + scenario_index * 100_000 + trial
                initial, ego_noise, traffic_noise = generate_trial_inputs(
                    template, seed
                )
                for method in METHODS:
                    problem = setup_problem(
                        config, device=device, with_environment=True
                    )
                    if method == "deterministic_stl":

                        def deterministic_spec(env, _state, _step):
                            return DeterministicLaneSpecification(
                                env, config["H"]
                            )

                        spec_factory = deterministic_spec
                    else:
                        spec_factory = None
                    result = run_lane_trial(
                        problem,
                        initial_state=_clone_state(initial),
                        ego_disturbances=ego_noise,
                        traffic_disturbances=traffic_noise,
                        make_spec=spec_factory,
                    )
                    identity = {
                        "scenario": scenario,
                        "uncertainty": float(factor),
                        "trial": trial,
                        "seed": seed,
                        "method": method,
                    }
                    trials.append(
                        {**identity, **trial_metrics(result, problem)}
                    )
                    for window, plan in enumerate(result.window_plans):
                        windows.append(
                            {
                                **identity,
                                "window": window,
                                "hard_lower": plan.hard_interval[0],
                                "hard_upper": plan.hard_interval[1],
                                "smooth_score": plan.smooth_lower,
                                "planning_time": plan.planning_time,
                            }
                        )
                    completed += 1
                    elapsed = perf_counter() - started
                    eta = elapsed / completed * (total_runs - completed)
                    print(
                        f"[{completed}/{total_runs}] {scenario} {factor:g}x "
                        f"trial {trial + 1} {method}; ETA {eta / 60:.1f} min",
                        flush=True,
                    )
    metadata = {
        "config": config_path,
        "scenarios": study["scenarios"],
        "trials": int(study["trials"]),
        "uncertainty_factors": study["uncertainty_factors"],
        "seed": base_seed,
        "methods": list(METHODS),
        "device": str(device),
        "scenario_settings": scenario_settings,
        "paired_initial_states_and_disturbances": True,
        "fixed_optimizer_iterations": True,
    }
    return (
        trials,
        windows,
        write_study_outputs(trials, windows, metadata, study["output_dir"]),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/experiments/lane_baseline.yaml"
    )
    parser.add_argument("--trials", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--scenarios", nargs="+")
    args = parser.parse_args(argv)
    overrides = {
        key: value
        for key, value in {
            "trials": args.trials,
            "output_dir": args.output_dir,
            "scenarios": args.scenarios,
        }.items()
        if value is not None
    }
    run_study(args.config, **overrides)


if __name__ == "__main__":
    main()
