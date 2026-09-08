import numpy as np
import torch

from models.dynamics import (
    create_gaussian_belief_trajectory,
    linear_system,
    piecewise_signal,
    sinusoidial_input,
)
from pdstl.operators import Always, Eventually, GreaterThan
from utils import load_config, skip_run, to_steps
from visualization.temporal import plot_temporal_example

CONFIG_PATH = "configs/examples.yaml"
LINEAR_PARAMETERS = ("a", "b", "g", "q", "mu", "P")


def linear_trace(model):
    """Generate a configured trace with the shared offline linear system."""
    time = np.linspace(0.0, model["t_end"], model["n_steps"])
    parameters = {name: model[name] for name in LINEAR_PARAMETERS}
    mean, variance = linear_system(
        **parameters, t=time, control_func=sinusoidial_input
    )
    return time, mean, variance


def show_results(name, example, time, mean, variance, atomic, temporal, formula):
    """Print and plot one evaluated example."""
    print(f"\n{name}")
    print(
        "atomic shape:",
        tuple(atomic.shape),
        "temporal shape:",
        tuple(temporal.shape),
    )
    print("atomic endpoints:", atomic[0, :3].detach().cpu().numpy())
    print("temporal endpoints:", temporal[0, :3].detach().cpu().numpy())
    plot_temporal_example(
        time,
        mean,
        variance,
        example["confidence_level"],
        atomic,
        temporal,
        formula.subformula,
        formula,
        formula.interval,
        show=show_plots,
    )


examples = load_config(CONFIG_PATH)
show_plots = examples["show_plots"]

# 1. Original continuous Always example
with skip_run("run", "Original continuous Always example") as check, check():
    example = examples["always"]

    time, mean, variance = linear_trace(example["model"])
    beliefs = create_gaussian_belief_trajectory(
        mean,
        variance,
        confidence_level=example["confidence_level"],
        dtype=torch.float64,
    )
    predicate = GreaterThan(example["threshold"])
    interval = to_steps(example["interval_sec"], time)
    formula = Always(predicate, interval=interval)

    atomic = predicate(beliefs)
    temporal = formula(beliefs, scale=-1)

    show_results(
        "Original Always", example, time, mean, variance, atomic, temporal, formula
    )

# 2. Discrete piecewise Always operator
with skip_run("run", "Piecewise Always operator") as check, check():
    example = examples["piecewise_always"]

    time, mean, variance = piecewise_signal(example["signal"])
    beliefs = create_gaussian_belief_trajectory(
        mean,
        variance,
        confidence_level=example["confidence_level"],
        dtype=torch.float64,
    )
    predicate = GreaterThan(example["threshold"])
    interval = example["interval_steps"]
    formula = Always(predicate, interval=interval)

    atomic = predicate(beliefs)
    temporal = formula(beliefs, scale=-1)

    show_results(
        "Piecewise Always", example, time, mean, variance, atomic, temporal, formula
    )

# 3. Continuous Eventually operator
with skip_run("run", "Continuous Eventually operator") as check, check():
    example = examples["eventually"]

    time, mean, variance = linear_trace(example["model"])
    beliefs = create_gaussian_belief_trajectory(
        mean,
        variance,
        confidence_level=example["confidence_level"],
        dtype=torch.float64,
    )
    predicate = GreaterThan(example["threshold"])
    interval = to_steps(example["interval_sec"], time)
    formula = Eventually(predicate, interval=interval)

    atomic = predicate(beliefs)
    temporal = formula(beliefs, scale=-1)

    show_results("Eventually", example, time, mean, variance, atomic, temporal, formula)
