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
from visualization.bounds import plot_mean_with_sigma_bounds
from visualization.robustness import plot_stl_formula_bounds
from utils import skip_run


config_path = "configs/config.yml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_beliefs_from_trace(mean_trace, var_trace, confidence_level=2.0):
    """
    Create per-timestep beliefs from mean and variance traces.
    
    Parameters
    ----------
    mean_trace : array_like
        Mean values over time
    var_trace : array_like
        Variance values over time
    confidence_level : float
        Number of standard deviations for confidence bounds 
    
    Returns
    -------
    belief_trajectory : BeliefTrajectory
        Trajectory of Gaussian beliefs with conservative bounds
    """
    mean_torch = torch.tensor(mean_trace, dtype=torch.float32).reshape(1, -1, 1)
    var_torch = torch.tensor(var_trace, dtype=torch.float32).reshape(1, -1, 1)

    beliefs = []
    for i in range(len(mean_trace)):
        mean_i = mean_torch[:, i : i + 1, :]
        var_i = var_torch[:, i : i + 1, :]
        beliefs.append(GaussianBelief(mean_i, var_i, confidence_level=confidence_level))

    return BeliefTrajectory(beliefs)


def interval_seconds_to_steps(interval_sec, t):
    """Convert STL interval from seconds to discrete steps."""
    a_sec, b_sec = interval_sec
    dt = float(t[1] - t[0])

    a = int(round(a_sec / dt))
    b = np.inf if np.isinf(b_sec) else int(round(b_sec / dt))

    return [a, b]


# =============================================================================
# TEST 1: Constant Input (Raw Signal)
# =============================================================================

with skip_run("skip", "Test 1: Constant Input") as check, check():
    a = 0.1   # positive drift
    b = 1.0   # input gain
    g = 0.5   # low stochastic noise
    q = 0.1   # low process noise

    mu = 45   # start below threshold
    P = 5     # low initial variance

    t = np.linspace(0, 10, 100)
    mean_trace, var_trace = linear_system(a, b, g, q, mu, P, t, constant_input)

    plot_mean_with_sigma_bounds(t, mean_trace, var_trace, threshold=50)


# =============================================================================
# TEST 2: Sinusoidal Input (Raw Signal)
# =============================================================================

with skip_run("skip", "Test 2: Sinusoidal Input") as check, check():
    a = 0.0   # zero drift
    b = 1.0   # input gain
    g = 2.5   # moderate stochastic noise
    q = 0.5   # moderate process noise

    mu = 50   # start at threshold
    P = 5     # initial variance

    t = np.linspace(0, 10, 100)
    mean_trace, var_trace = linear_system(a, b, g, q, mu, P, t, noisy_stock_input)

    plot_mean_with_sigma_bounds(t, mean_trace, var_trace, threshold=50)


# =============================================================================
# TEST 3: Always Operator
# =============================================================================

with skip_run("run", "Test 3: Always Operator") as check, check():

    a, b, g, q = 0.01, 1.0, 2.0, 2.5
    mu, P = 50, 2
    t = np.linspace(0, 10, 100)

    mean_trace, var_trace = linear_system(a, b, g, q, mu, P, t, sinusoidial_input)
    belief_trajectory = create_beliefs_from_trace(mean_trace, var_trace)

    # Specification: Always[2,5](x >= 50)
    threshold = 45.0
    phi = GreaterThan(threshold)
    
    interval_sec = [0, 10]
    interval_steps = interval_seconds_to_steps(interval_sec, t)

    spec = Always(phi, interval=interval_steps)
    robustness_trace = spec(belief_trajectory)

    formula_str = f"□[{interval_sec[0]},{interval_sec[1]}](x ≥ {threshold})"
    plot_stl_formula_bounds(
        t,
        robustness_trace,
        mean_trace=mean_trace,
        var_trace=var_trace,
        thresholds=threshold,
        formula_str=formula_str, 
        interval=interval_steps,
        show_windows=True,
        n_example_windows=3,
    )


# =============================================================================
# TEST 4: Boolean AND 
# =============================================================================

with skip_run("run", "Test 4: Boolean AND") as check, check():

    a, b, g, q = 0.01, 2.0, 4.0, 0.5
    mu, P = 50, 5
    t = np.linspace(0, 10, 100)

    mean_trace, var_trace = linear_system(a, b, g, q, mu, P, t, sinusoidial_input)
    belief_trajectory = create_beliefs_from_trace(mean_trace, var_trace)

    # Specification: (x >= 40) ∧ (x <= 55)
    phi_and = And(GreaterThan(40.0), LessThan(55.0))
    robustness_trace = phi_and(belief_trajectory)
 
    plot_stl_formula_bounds(
        t,
        robustness_trace,
        mean_trace=mean_trace,
        var_trace=var_trace,
        thresholds=[40.0, 55.0],
        formula_str="(x ≥ 40) ∧ (x ≤ 55)",
        interval=None,  
        show_windows=True,
    )


# =============================================================================
# TEST 5: Complex Formula - Reach Target While Staying Safe
# =============================================================================

with skip_run("run", "Test 5: Complex Formula") as check, check():

    a, b, g, q = 0.1, 1.0, 0.5, 0.5
    mu, P = 40, 2
    t = np.linspace(0, 20, 100)

    mean_trace, var_trace = linear_system(a, b, g, q, mu, P, t, sinusoidial_input)
    belief_trajectory = create_beliefs_from_trace(mean_trace, var_trace)

    interval_steps = [0, 20]
    reach_target = Eventually(GreaterThan(70.0), interval=interval_steps)
    stay_safe = Always(GreaterThan(50.0), interval=interval_steps)

    spec = reach_target & stay_safe
    robustness_trace = spec(belief_trajectory)

    plot_stl_formula_bounds(
        t,
        robustness_trace,
        mean_trace=mean_trace,
        var_trace=var_trace,
        thresholds=[40.0, 70.0],
        formula_str="◇[0,20](x ≥ 70) ∧ □[0,20](x ≥ 40)",
        interval=interval_steps,
        show_windows=True,
        n_example_windows=5,
    )

