import numpy as np

from models.dynamics import control_input, first_order_system, sinusoidial_input
from stl.propagate import compute_bounds
from utils import skip_run
from visualization.bounds import plot_bounds_with_trace
from visualization.new_bounds import plot_bounds

# The configuration file
config_path = "configs/config.yml"
# config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run("run", "Data - Constant Input") as check, check():
    # --------- System Dynamics ------------#
    a = 0.1  # state
    b = 1.0  # input
    g = 0.5  # Stochastic noise
    q = 0.1  # process noise covariance

    mu = 45  # mean height
    P = 5  # initial height variance

    t = np.linspace(0, 5, 300)  # time from 0 to 30 seconds as given by stl
    mean_trace, var_trace = first_order_system(a, b, g, q, mu, P, t, control_input)
    lower_bound, upper_bound = compute_bounds(mean_trace, var_trace, t)
    plot_bounds_with_trace(t, mean_trace, var_trace)
    plot_bounds(t, mean_trace, var_trace)

with skip_run("run", "Data - Sinusoidal Input") as check, check():
    a = 0.0  # zero drift
    b = 1.0  # input gain
    g = 0.5  # stochastic noise
    q = 0.1  # process noise covariance

    mu = 40  # mean height (starting at threshold)
    P = 5  # initial height variance

    t = np.linspace(0, 10, 1000)
    mean_trace, var_trace = first_order_system(a, b, g, q, mu, P, t, sinusoidial_input)
    lower_bound, upper_bound = compute_bounds(mean_trace, var_trace, t)
    plot_bounds_with_trace(t, mean_trace, var_trace)
    plot_bounds(t, mean_trace, var_trace)     
    
    # I added a new plot function here, kind os imilar but indicates along the sigma lines the safe, risky and violation regions
    

