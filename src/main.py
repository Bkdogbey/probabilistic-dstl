import numpy as np

from models.dynamics import control_input, first_order_system, sinusoidial_input
from stl.propagate import compute_bounds
from utils import skip_run
from visualization.bounds import plot_bounds_with_trace

# The configuration file
config_path = "configs/config.yml"
# config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run("skip", "Data - Constant Input") as check, check():
    # --------- System Dynamics ------------#
    a = 0.1  # state
    b = 1.0  # input
    g = 0.5  # Stochastic noise
    q = 0.1  # process noise covariance

    mu = 45  # mean height
    P = 5  # initial height variance

    t = np.linspace(0, 5, 30)  # time from 0 to 30 seconds as given by stl
    mean_trace, var_trace = first_order_system(a, b, g, q, mu, P, t, control_input)
    lower_bound, upper_bound = compute_bounds(mean_trace, var_trace, t)
    plot_bounds_with_trace(t, mean_trace, var_trace, lower_bound, upper_bound)

with skip_run("run", "Data - Sinusoidal Input") as check, check():
    a = 0.0  # zero drift
    b = 1.0  # input gain
    g = 0.5  # stochastic noise
    q = 0.1  # process noise covariance

    mu = 50  # mean height (starting at threshold)
    P = 5  # initial height variance

    t = np.linspace(0, 10, 50)
    mean_trace, var_trace = first_order_system(a, b, g, q, mu, P, t, sinusoidial_input)
    lower_bound, upper_bound = compute_bounds(mean_trace, var_trace, t)

    # NOTE:
    # 1. In the plot function why are you not using lower_bound and upper_bound?
    # 2. In the plot, the band of violations (light orange) doesn't seem correct.
    # The band should start exactly when the lower sigma crosses the 50 mark. But it doesn't happen so.

    plot_bounds_with_trace(t, mean_trace, var_trace, lower_bound, upper_bound)
