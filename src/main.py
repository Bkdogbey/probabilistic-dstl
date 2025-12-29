

import numpy as np
import torch
import yaml
from models.dynamics import (
    GaussianBelief,
    constant_input,
    linear_system,
    sinusoidial_input,
    noisy_stock_input,
)
from pdstl.base import BeliefTrajectory
from pdstl.operators import GreaterThan, LessThan, Always, Eventually, Until, And, Or
from pdstl.propagate import compute_bounds
from utils import skip_run
from visualization.bounds import plot_mean_with_sigma_bounds
from visualization.robustness import plot_stl_formula_bounds

config_path = "configs/config.yml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

# =============================================================================
# HELPER FUNCTIONS FOR CREATING BELIEFS
# =============================================================================


def create_beliefs_from_trace(mean_trace, var_trace):
    """
    Create per-timestep beliefs from mean and variance traces.

    Parameters
    ----------
    mean_trace : array_like
        Mean values over time
    var_trace : array_like
        Variance values over time

    Returns
    -------
    belief_trajectory : BeliefTrajectory
        Trajectory of Gaussian beliefs
    """
    mean_torch = torch.tensor(mean_trace, dtype=torch.float32).reshape(1, -1, 1)
    var_torch = torch.tensor(var_trace, dtype=torch.float32).reshape(1, -1, 1)

    beliefs = []
    for i in range(len(mean_trace)):
        mean_i = mean_torch[:, i : i + 1, :]  # [1, 1, 1]
        var_i = var_torch[:, i : i + 1, :]  # [1, 1, 1]
        beliefs.append(GaussianBelief(mean_i, var_i))

    return BeliefTrajectory(beliefs)

def interval_seconds_to_steps(interval_sec, t):
    """
    Convert STL interval given in seconds to discrete step offsets for your STL code.

    interval_sec: [a_sec, b_sec] where b_sec can be np.inf
    t: time vector (seconds), assumed uniform spacing
    """
    a_sec, b_sec = interval_sec
    dt = float(t[1] - t[0])

    a = int(round(a_sec / dt))

    if np.isinf(b_sec):
        b = np.inf
    else:
        b = int(round(b_sec / dt))

    return [a, b]


# =============================================================================
# TEST CASE 1: Constant Input 
# =============================================================================

with skip_run("skip", "Test 1: Constant Input") as check, check():
    a = 0.1  # positive drift
    b = 1.0  # input gain
    g = 0.5  # low stochastic noise
    q = 0.1  # low process noise

    mu = 45  # start below threshold
    P = 5  # low initial variance

    t = np.linspace(0, 10, 100)
    mean_trace, var_trace = linear_system(a, b, g, q, mu, P, t, constant_input)

    # Visualize raw signal
    plot_mean_with_sigma_bounds(t, mean_trace, var_trace, threshold=50)


# =============================================================================
# TEST CASE 2: Sinusoidal Input (Oscillating System)
# =============================================================================

with skip_run("skip", "Test 2: Sinusoidal Input") as check, check():
    a = 0.0  # zero drift
    b = 1.0  # input gain
    g = 2.5  # moderate stochastic noise
    q = 0.5  # moderate process noise

    mu = 50  # start at threshold
    P = 5  # initial variance

    t = np.linspace(0, 10, 100)
    mean_trace, var_trace = linear_system(a, b, g, q, mu, P, t, noisy_stock_input)

    # Visualize raw signal
    plot_mean_with_sigma_bounds(t, mean_trace, var_trace, threshold=50)

# =============================================================================
# TEST CASE 1: Always Operator
# =============================================================================

with skip_run("run", "Test: Always Operator") as check, check():

    a, b, g, q = 0.1, 1.0, 5.0, 0.3
    mu, P = 60, 5
    t = np.linspace(0, 16, 100)

    mean_trace, var_trace = linear_system(a, b, g, q, mu, P, t, sinusoidial_input)
    belief_trajectory = create_beliefs_from_trace(mean_trace, var_trace)

    threshold = 50.0
    phi = GreaterThan(threshold)
    phi_trace = phi(belief_trajectory)
    print("Predicate trace at t=0:", phi_trace[0, 0, :])
    print("Predicate trace at t=20:", phi_trace[0, 20, :])
    print("Predicate trace at t=59:", phi_trace[0, 59, :])
    print("Shape:", phi_trace.shape)
    interval_sec = [2, 5]
    interval_steps = interval_seconds_to_steps(interval_sec, t)

    spec = Always(phi, interval=interval_steps)
    robustness_trace = spec(belief_trajectory)

    # Create custom formula string with original time interval
    formula_str = f"Always[{interval_sec[0]},{interval_sec[1]}](x >= {threshold})"

    # Plot with temporal windows
    plot_stl_formula_bounds(
        t,
        robustness_trace,
        mean_trace=mean_trace,
        var_trace=var_trace,
        thresholds=threshold,
        formula_str=formula_str, 
        show_upper=True,
        interval=interval_steps,  
        show_windows=True,
        n_example_windows=4,
    )


# =============================================================================
# TEST CASE 2: Complex Formula - Reach While Staying Safe
# =============================================================================

with skip_run("skip", "Test: Complex Formula") as check, check():
    a, b, g, q = 0.0, 1.0, 2.5, 0.5
    mu, P = 50, 8
    t = np.linspace(0, 10)

    mean_trace, var_trace = linear_system(a, b, g, q, mu, P, t, sinusoidial_input)
    belief_trajectory = create_beliefs_from_trace(mean_trace, var_trace)

    reach_target = Eventually(LessThan(75.0), interval=[0, 10])

    stay_safe = Always(GreaterThan(50.0), interval=[0, 10])

    # Combined: reach AND safe
    spec = stay_safe & reach_target

    robustness_trace = spec(belief_trajectory)

    plot_stl_formula_bounds(
        t,
        robustness_trace,
        mean_trace=mean_trace,
        var_trace=var_trace,
        thresholds=[20.0, 75.0],
        formula_str=str(spec),
        show_upper=True,
    )