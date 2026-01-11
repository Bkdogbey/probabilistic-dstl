"""
STL Robustness Examples

Three examples:
  1. Always operator: □[a,b](x ≥ h)
  2. Eventually operator: ◇[a,b](x ≥ h)
  3. Combined: ◇[a,b](x ≥ h₁) ∧ □[a,b](x ≥ h₂)
"""

import numpy as np
import torch
from models.dynamics import GaussianBelief, linear_system, sinusoidial_input, noisy_stock_input, step_sequence_input, piecewise_input
from pdstl.base import BeliefTrajectory
from pdstl.operators import GreaterThan, Always, Eventually
from visualization.robustness import plot_stl_formula_bounds
from utils import skip_run


# =============================================================================
# HELPERS
# =============================================================================

def create_belief_trajectory(mean_trace, var_trace, confidence_level=1.0):
    """Create belief trajectory from mean and variance traces."""
    mean = torch.tensor(mean_trace, dtype=torch.float32).reshape(1, -1, 1)
    var = torch.tensor(var_trace, dtype=torch.float32).reshape(1, -1, 1)
    
    beliefs = []
    for i in range(len(mean_trace)):
        m = mean[:, i:i+1, :]
        v = var[:, i:i+1, :]
        beliefs.append(GaussianBelief(m, v, confidence_level=confidence_level))
    
    return BeliefTrajectory(beliefs)


def to_steps(interval_sec, t):
    """Convert time interval [a,b] in seconds to steps."""
    dt = float(t[1] - t[0])
    a = int(round(interval_sec[0] / dt))
    b = np.inf if np.isinf(interval_sec[1]) else int(round(interval_sec[1] / dt))
    return [a, b]


def print_trace(name, trace, time, step=10):
    """Print trace values at regular intervals."""
    if isinstance(trace, torch.Tensor):
        trace = trace.detach().cpu().numpy()
    trace = np.asarray(trace)
    
    if trace.ndim == 3:
        trace = trace[0]  # [B,T,2] -> [T,2]
    
    print(f"\n{name}:")
    print(f"{'t':>6} {'time':>8} {'lower':>10} {'upper':>10}")
    print("-" * 38)
    
    for i in range(0, len(time), step):
        print(f"{i:>6} {time[i]:>8.2f} {trace[i, 0]:>10.4f} {trace[i, 1]:>10.4f}")


# =============================================================================
# EXAMPLE 1: Always Operator
# =============================================================================

with skip_run("run", "Example 1: Always") as check, check():
    t = np.linspace(0, 10, 100)
    mean, var = linear_system(
        a=0.01, b=1.0, g=1.0, q=2.5,
        mu=50, P= 0.15,
        t=t, control_func=sinusoidial_input
    )

    beliefs = create_belief_trajectory(mean, var, confidence_level=1.0)

    # □[1,2](x ≥ 50)
    threshold = 50.0
    interval_sec = [1, 2]
    interval_steps = to_steps(interval_sec, t)

    phi = GreaterThan(threshold)
    spec = Always(phi, interval=interval_steps)

    pred_trace = phi(beliefs)
    oper_trace = spec(beliefs)

    print(f"\n{'='*50}")
    print(f"Example 1: □[{interval_sec[0]}, {interval_sec[1]}](x ≥ {threshold})")
    print(f"Interval in steps: {interval_steps}")
    print(f"{'='*50}")
    
    print_trace("Predicate P(x ≥ 50)", pred_trace, t)
    print_trace("Operator □[1,2]P(φ)", oper_trace, t)

    plot_stl_formula_bounds(
        t, oper_trace,
        mean_trace=mean,
        var_trace=var,
        predicate_trace=pred_trace,
        thresholds=threshold,
        formula_str=f"□[{interval_sec[0]}, {interval_sec[1]}](x ≥ {threshold})",
        interval=interval_steps,
        operator_type='always',
    )


# =============================================================================
# EXAMPLE 2: Eventually Operator
# =============================================================================

with skip_run("run", "Example 2: Eventually") as check, check():
    t = np.linspace(0, 20, 100)
    mean, var = linear_system(
        a=0.1, b=1.0, g=0.5, q=0.5,
        mu=40, P=2.0,
        t=t, control_func=sinusoidial_input
    )

    beliefs = create_belief_trajectory(mean, var, confidence_level=1.0)

    # ◇[0,5](x ≥ 70)
    threshold = 70.0
    interval_sec = [0, 5]
    interval_steps = to_steps(interval_sec, t)

    phi = GreaterThan(threshold)
    spec = Eventually(phi, interval=interval_steps)

    pred_trace = phi(beliefs)
    oper_trace = spec(beliefs)

    print(f"\n{'='*50}")
    print(f"Example 2: ◇[{interval_sec[0]}, {interval_sec[1]}](x ≥ {threshold})")
    print(f"Interval in steps: {interval_steps}")
    print(f"{'='*50}")
    
    print_trace("Predicate P(x ≥ 70)", pred_trace, t)
    print_trace("Operator ◇[0,5]P(φ)", oper_trace, t)

    plot_stl_formula_bounds(
        t, oper_trace,
        mean_trace=mean,
        var_trace=var,
        predicate_trace=pred_trace,
        thresholds=threshold,
        formula_str=f"◇[{interval_sec[0]}, {interval_sec[1]}](x ≥ {threshold})",
        interval=interval_steps,
        operator_type='eventually',
    )


# =============================================================================
# EXAMPLE 3: Combined - Reach While Safe
# =============================================================================

with skip_run("run", "Example 3: Combined") as check, check():
    t = np.linspace(0, 20, 100)
    mean, var = linear_system(
        a=0.05, b=1.0, g=0.5, q=0.5,
        mu=50, P=1.0,
        t=t, control_func=sinusoidial_input
    )

    beliefs = create_belief_trajectory(mean, var, confidence_level=1.0)

    # ◇[0,10](x ≥ 80) ∧ □[0,10](x ≥ 45)
    interval_steps = [0, 50]

    reach = Eventually(GreaterThan(80.0), interval=interval_steps)
    safe = Always(GreaterThan(45.0), interval=interval_steps)
    spec = reach & safe

    oper_trace = spec(beliefs)

    print(f"\n{'='*50}")
    print(f"Example 3: ◇[0,10](x ≥ 80) ∧ □[0,10](x ≥ 45)")
    print(f"Interval in steps: {interval_steps}")
    print(f"{'='*50}")
    
    print_trace("Combined operator", oper_trace, t)

    plot_stl_formula_bounds(
        t, oper_trace,
        mean_trace=mean,
        var_trace=var,
        thresholds=[45.0, 80.0],
        formula_str="◇[0,10](x ≥ 80) ∧ □[0,10](x ≥ 45)",
        interval=interval_steps,
    )

# =============================================================================
# EXAMPLE 4: Piecewise Signal for □[a,b](x ≥ 50)
# =============================================================================

with skip_run("run", "Example 4: Piecewise") as check, check():
    t = np.linspace(0, 10, 50)  # Coarse time steps for clarity
    
    mean, var = linear_system(
        a=0.0, b=1.0, g=0.5, q=0.5,
        mu=50, P=4.0,
        t=t, control_func= piecewise_input
    )
    
    beliefs = create_belief_trajectory(mean, var, confidence_level=1.0)
    
    threshold = 50.0
    interval_steps = [2, 5]
    
    phi = GreaterThan(threshold)
    spec = Always(phi, interval=interval_steps)
    
    pred_trace = phi(beliefs)
    oper_trace = spec(beliefs)
    
    print(f"\n{'='*50}")
    print(f"Example 4: □[2,5](x ≥ {threshold}) - Piecewise Input")
    print(f"{'='*50}")
    print_trace("Predicate P(x ≥ 50)", pred_trace, t, step=5)
    print_trace("Always □[2,5]P(φ)", oper_trace, t, step=5)
    
    plot_stl_formula_bounds(
        t, oper_trace,
        mean_trace=mean,
        var_trace=var,
        predicate_trace=pred_trace,
        thresholds=threshold,
        formula_str=f"□[2, 5](x ≥ {threshold})",
        interval=interval_steps,
        operator_type='always',
    )











