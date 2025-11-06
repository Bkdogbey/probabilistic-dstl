import numpy as np
import yaml
import torch

from models.dynamics import control_input, linear_system, sinusoidial_input
from stl.propagate import compute_bounds
from utils import skip_run
from visualization.bounds import plot_mean_with_sigma_bounds
from visualization.stlcg_robs import plot_rnn_robustness
from stl.stlcg import Always, GreaterThan

# The configuration file
config_path = "configs/config.yml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run("run", "Data - Constant Input") as check, check():
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

    # RNN Temporal STL Robustness
    sigma = np.sqrt(var_trace)
    lower_sigma = mean_trace - sigma
    upper_sigma = mean_trace + sigma

    threshold = 50.0
    interval = [0, 50]

    lower_torch = (
        torch.tensor(lower_sigma, dtype=torch.float32).reshape(1, -1, 1).flip(1)
    )
    upper_torch = (
        torch.tensor(upper_sigma, dtype=torch.float32).reshape(1, -1, 1).flip(1)
    )

    predicate = GreaterThan(threshold)
    always_formula = Always(predicate, interval=interval)

    bounds = (lower_torch, upper_torch)
    rob_lower_torch, rob_upper_torch = always_formula(bounds, scale=-1)

    rob_lower_rnn = rob_lower_torch.flip(1).squeeze().detach().numpy()
    rob_upper_rnn = rob_upper_torch.flip(1).squeeze().detach().numpy()

    plot_rnn_robustness(t, rob_lower_rnn, rob_upper_rnn, always_formula, threshold)


with skip_run("run", "Data - Sinusoidal Input") as check, check():
    a = 0.0  # zero drift
    b = 1.0  # input gain
    g = 0.5  # stochastic noise
    q = 0.1  # process noise covariance

    mu = 45  # mean height (starting at threshold)
    P = 5  # initial height variance

    t = np.linspace(0, 10, 1000)
    mean_trace, var_trace = linear_system(a, b, g, q, mu, P, t, sinusoidial_input)
    lower_bound, upper_bound = compute_bounds(mean_trace, var_trace, t)
    plot_mean_with_sigma_bounds(t, mean_trace, var_trace)

    # RNN Temporal STL Robustness
    sigma = np.sqrt(var_trace)
    lower_sigma = mean_trace - sigma
    upper_sigma = mean_trace + sigma

    threshold = 50.0
    interval = [0, 100]

    lower_torch = (
        torch.tensor(lower_sigma, dtype=torch.float32).reshape(1, -1, 1).flip(1)
    )
    upper_torch = (
        torch.tensor(upper_sigma, dtype=torch.float32).reshape(1, -1, 1).flip(1)
    )

    predicate = GreaterThan(threshold)
    always_formula = Always(predicate, interval=interval)

    bounds = (lower_torch, upper_torch)
    rob_lower_torch, rob_upper_torch = always_formula(bounds, scale=-1)

    rob_lower_rnn = rob_lower_torch.flip(1).squeeze().detach().numpy()
    rob_upper_rnn = rob_upper_torch.flip(1).squeeze().detach().numpy()

    plot_rnn_robustness(t, rob_lower_rnn, rob_upper_rnn, always_formula, threshold)
