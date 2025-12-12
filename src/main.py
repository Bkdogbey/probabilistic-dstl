import numpy as np
import torch
import yaml
from models.dynamics import (
    GaussianBelief,
    constant_input,
    linear_system,
    sinusoidial_input,
)
from pdstl.base import BeliefTrajectory
from pdstl.operators import GreaterThan, LessThan, Always, Eventually, Until, And, Or
from pdstl.propagate import compute_bounds
from utils import skip_run
from visualization.bounds import plot_mean_with_sigma_bounds
from visualization.robustness import plot_stl_formula_bounds

config_path = "configs/config.yml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run("skip", "Data - Constant Input") as check, check():
    a = 0.1  # state
    b = 1.0  # input
    g = 0.5  # Stochastic noise
    q = 0.1  # process noise covariance

    mu = 45  # mean height
    P = 5  # initial height variance

    t = np.linspace(0, 10, 300)  # time from 0 to 10 seconds as given by stl
    mean_trace, var_trace = linear_system(a, b, g, q, mu, P, t, constant_input)
    lower_bound, upper_bound = compute_bounds(mean_trace, var_trace, t)
    plot_mean_with_sigma_bounds(t, mean_trace, var_trace)

with skip_run("skip", "Data - Sinusoidal Input") as check, check():
    a = 0.0  # zero drift
    b = 1.0  # input gain
    g = 10.5  # stochastic noise
    q = 10.1  # process noise covariance

    mu = 50  # mean height (starting at threshold)
    P = 5  # initial height variance

    t = np.linspace(0, 10, 1000)
    mean_trace, var_trace = linear_system(a, b, g, q, mu, P, t, sinusoidial_input)
    lower_bound, upper_bound = compute_bounds(mean_trace, var_trace, t)
    plot_mean_with_sigma_bounds(t, mean_trace, var_trace)


with skip_run("run", "STL Operators Verification") as check, check():
    a, b, g, q = 0.1, 1.0, 1.5, 0.1
    mu, P = 45, 10
    t = np.linspace(0, 10, 10)

    mean_trace, var_trace = linear_system(a, b, g, q, mu, P, t, sinusoidial_input)

    mean_torch = torch.tensor(mean_trace, dtype=torch.float32).reshape(1, -1, 1)
    var_torch = torch.tensor(var_trace, dtype=torch.float32).reshape(1, -1, 1)

    beliefs = []
    for mean, var in zip(mean_torch, var_torch):
        beliefs.append(GaussianBelief(mean, var))

    belief_trajectory = BeliefTrajectory(beliefs)

    threshold1 = 50.0

    phi1 = GreaterThan(threshold1)  # x >= 50
    spec = Eventually(phi1, interval=[0, 10])
    robustness_trace = spec(belief_trajectory)

    plot_stl_formula_bounds(
        t,
        robustness_trace,
        mean_trace=mean_trace,
        var_trace=var_trace,
    )
