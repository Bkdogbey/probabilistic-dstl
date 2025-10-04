import numpy as np
import yaml

from models.dynamics import first_order_system
from stl.propagate import compute_bounds
from utils import skip_run
from visualization.bounds import plot_bounds_with_trace

# The configuration file
config_path = "configs/config.yml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run("run", "Data") as check, check():
    # --------- System Dynamics ------------#
    a = 0.1  # state
    b = 1.0  # input
    g = 0.5  # Stochastic noise
    q = 0.1  # process noise covariance

    mu = 45  # mean height
    P = 5  # initial height variance

    t = np.linspace(0, 5, 30)  # time from 0 to 30 seconds as given by stl
    mean_trace, var_trace = first_order_system(a, b, g, q, mu, P, t)
    lower_bound, upper_bound = compute_bounds(mean_trace, var_trace, t)
    plot_bounds_with_trace(t, mean_trace, var_trace, lower_bound, upper_bound)
