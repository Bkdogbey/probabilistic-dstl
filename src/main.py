import numpy as np
import yaml
import torch
from models.dynamics import control_input, linear_system, sinusoidial_input
from stl.propagate import compute_bounds
from stl.pdstl import GreaterThan
from utils import skip_run
from visualization.bounds import plot_mean_with_sigma_bounds
from visualization.stlcg_robs import plot_predicate_robustness

# The configuration file
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
    mean_trace, var_trace = linear_system(a, b, g, q, mu, P, t, control_input)
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


with skip_run("run", "Greater Verification") as check, check():
    # System setup
    a, b, g, q = 0.1, 1.0, 1.5, 0.1
    mu, P = 35, 10
    threshold = 50.0

    # Generate belief trajectory
    t = np.linspace(0, 10, 300)
    mean_trace, var_trace = linear_system(a, b, g, q, mu, P, t, control_input)

    # Convert to PyTorch tensors
    mean_torch = torch.tensor(mean_trace, dtype=torch.float32).reshape(1, -1, 1)
    var_torch = torch.tensor(var_trace, dtype=torch.float32).reshape(1, -1, 1)
    belief = (mean_torch, var_torch)

    # Compute robustness using GreaterThan predicate
    predicate = GreaterThan(threshold)
    robustness_trace = predicate(belief)
    robustness = robustness_trace[..., 0].squeeze().detach().numpy()

    # Visualize trajectory AND robustness together
    plot_predicate_robustness(
        t, mean_trace, var_trace, robustness, predicate, threshold
    )
