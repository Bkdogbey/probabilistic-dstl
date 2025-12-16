"""
Main STL Verification Script
Demonstrates various STL formulas and signal inputs
"""

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
# config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

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


# =============================================================================
# TEST CASE 1: Constant Input (Stable System)
# =============================================================================

with skip_run("skip", "Test 1: Constant Input") as check, check():
    """
    System with constant negative input - should drift down
    Good for testing Always operator (stays above threshold)
    """
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

with skip_run("run", "Test 2: Sinusoidal Input") as check, check():
    """
    System with sinusoidal input - oscillates around threshold
    Good for testing Eventually/Always with bounded intervals
    """
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

with skip_run("skip", "Test: Always Operator") as check, check():
    """
    Test Always: □[0,10](x >= 48)
    Shows probability that constraint holds at ALL future times
    """
    a, b, g, q = 0.0, 1.0, 2.0, 0.03
    mu, P = 50, 5
    t = np.linspace(0, 10, 100)

    mean_trace, var_trace = linear_system(a, b, g, q, mu, P, t, sinusoidial_input)
    belief_trajectory = create_beliefs_from_trace(mean_trace, var_trace)

    threshold = 48.0
    phi = GreaterThan(threshold)
    spec = Always(phi, interval=[0, 10])
    robustness_trace = spec(belief_trajectory)

    plot_stl_formula_bounds(
        t,
        robustness_trace,
        mean_trace=mean_trace,
        var_trace=var_trace,
        thresholds=threshold,
        formula_str=str(spec),
        show_upper=True,
    )


# =============================================================================
# TEST CASE 2: Complex Formula - Reach While Staying Safe
# =============================================================================

with skip_run("run", "Test: Complex Formula") as check, check():
    """
    Test complex formula: Eventually reach target WHILE always staying safe
    ◇[0,5](x >= 55) ∧ □[0,10](x >= 40)
    
    Interpretation:
    - Must reach x >= 55 sometime in [0,5]
    - AND must stay x >= 40 for all times in [0,10]
    """
    a, b, g, q = 0.0, 1.0, 2.5, 0.5
    mu, P = 50, 8
    t = np.linspace(0, 10, 100)

    mean_trace, var_trace = linear_system(a, b, g, q, mu, P, t, noisy_stock_input)
    belief_trajectory = create_beliefs_from_trace(mean_trace, var_trace)

    # Reach target: ◇[0,5](x >= 55)
    reach_target = Eventually(GreaterThan(55.0), interval=[0, 5])

    # Stay safe: □[0,10](x >= 40)
    stay_safe = Always(GreaterThan(40.0), interval=[0, 10])

    # Combined: reach AND safe
    spec = reach_target & stay_safe

    robustness_trace = spec(belief_trajectory)

    plot_stl_formula_bounds(
        t,
        robustness_trace,
        mean_trace=mean_trace,
        var_trace=var_trace,
        thresholds=[40.0, 55.0],
        formula_str=str(spec),
        show_upper=True,
    )

