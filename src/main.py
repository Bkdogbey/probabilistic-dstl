import numpy as np

from models.dynamics import create_gaussian_belief_trajectory, piecewise_signal
from pdstl.operators import Always, Eventually, GreaterThan
from utils import load_config, skip_run
from visualization.temporal import plot_temporal_example, print_temporal_results


examples = load_config("configs/examples.yaml")
show_plots = examples["show_plots"]


# 1. Always
with skip_run("run", "Always") as check, check():
    config = examples["always"]
    threshold = config["threshold"]
    sigma_multiplier = config["sigma_multiplier"]
    interval = config["interval_steps"]

    time, mean, variance = piecewise_signal(config["values"])
    sigma = sigma_multiplier * np.sqrt(variance)
    beliefs = create_gaussian_belief_trajectory(
        mean, variance, sigma_multiplier=sigma_multiplier
    )
    predicate = GreaterThan(threshold)
    formula = Always(predicate, interval=interval)

    atomic_trace = predicate(beliefs)
    temporal_trace = formula(beliefs, scale=-1)

    print_temporal_results("Always", atomic_trace, temporal_trace)
    plot_temporal_example(
        time, mean, sigma, threshold,
        str(predicate), atomic_trace, str(formula), temporal_trace,
        show=show_plots,
    )


# 2. Eventually
with skip_run("run", "Eventually") as check, check():
    config = examples["eventually"]
    threshold = config["threshold"]
    sigma_multiplier = config["sigma_multiplier"]
    interval = config["interval_steps"]

    time, mean, variance = piecewise_signal(config["values"])
    sigma = sigma_multiplier * np.sqrt(variance)
    beliefs = create_gaussian_belief_trajectory(
        mean, variance, sigma_multiplier=sigma_multiplier
    )
    predicate = GreaterThan(threshold)
    formula = Eventually(predicate, interval=interval)

    atomic_trace = predicate(beliefs)
    temporal_trace = formula(beliefs, scale=-1)

    print_temporal_results("Eventually", atomic_trace, temporal_trace)
    plot_temporal_example(
        time, mean, sigma, threshold,
        str(predicate), atomic_trace, str(formula), temporal_trace,
        show=show_plots,
    )


# 3. Nested Eventually(Always(predicate))
with skip_run("run", "Nested") as check, check():
    config = examples["nested"]
    threshold = config["threshold"]
    sigma_multiplier = config["sigma_multiplier"]
    always_interval = config["always_interval_steps"]
    eventually_interval = config["eventually_interval_steps"]

    time, mean, variance = piecewise_signal(config["values"])
    sigma = sigma_multiplier * np.sqrt(variance)
    beliefs = create_gaussian_belief_trajectory(
        mean, variance, sigma_multiplier=sigma_multiplier
    )
    predicate = GreaterThan(threshold)
    inner = Always(predicate, interval=always_interval)
    formula = Eventually(inner, interval=eventually_interval)

    atomic_trace = predicate(beliefs)
    inner_trace = inner(beliefs, scale=-1)
    temporal_trace = formula(beliefs, scale=-1)

    print_temporal_results(
        "Nested", atomic_trace, temporal_trace, inner_trace=inner_trace
    )
    plot_temporal_example(
        time, mean, sigma, threshold,
        str(predicate), atomic_trace, str(formula), temporal_trace,
        inner_label=str(inner), inner_trace=inner_trace,
        show=show_plots,
    )
