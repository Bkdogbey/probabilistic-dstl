"""
2D Motion Planning with Gradient-Based Optimization
====================================================
Functions for trajectory optimization under STL constraints.
"""

import numpy as np
import torch
import torch.nn as nn
from models.dynamics import GaussianBelief
from pdstl.base import BeliefTrajectory


def compute_waypoint(start, goal, obstacle):
    """
    Compute waypoint with maximum clearance from obstacle.

    Args:
        start: [2] start position
        goal: [2] goal position
        obstacle: dict with 'x' and 'y' bounds (can be None for straight-line)

    Returns:
        waypoint: [2] waypoint position
    """
    # If no obstacle provided, just go straight
    if obstacle is None:
        return (start + goal) / 2.0

    obs_center = np.array(
        [
            (obstacle["x"][0] + obstacle["x"][1]) / 2,
            (obstacle["y"][0] + obstacle["y"][1]) / 2,
        ]
    )

    clearances = [
        abs(goal[0] - obstacle["x"][1]),  # right
        abs(goal[0] - obstacle["x"][0]),  # left
        abs(goal[1] - obstacle["y"][1]),  # top
        abs(goal[1] - obstacle["y"][0]),  # bottom
    ]

    max_idx = np.argmax(clearances)
    margin = 2.0

    waypoints = [
        np.array([obstacle["x"][1] + margin, obs_center[1]]),
        np.array([obstacle["x"][0] - margin, obs_center[1]]),
        np.array([obs_center[0], obstacle["y"][1] + margin]),
        np.array([obs_center[0], obstacle["y"][0] - margin]),
    ]

    return waypoints[max_idx]


def initialize_controls_straight(start, goal, horizon, dt):
    """
    Simple straight-line control initialization: start → goal

    Args:
        start: [2] start position
        goal: [2] goal position
        horizon: int, trajectory length
        dt: float, time step

    Returns:
        controls: [1, horizon-1, 2] torch tensor
    """
    controls = torch.zeros(1, horizon - 1, 2)

    # Constant velocity toward goal
    v_to_goal = (goal - start) / ((horizon - 1) * dt)
    controls[:, :, :] = torch.tensor(v_to_goal, dtype=torch.float32).view(1, 1, 2)

    return controls


def initialize_controls(start, waypoint, goal, horizon, dt):
    """
    Two-phase control initialization: start → waypoint → goal

    Args:
        start: [2] start position
        waypoint: [2] waypoint position
        goal: [2] goal position
        horizon: int, trajectory length
        dt: float, time step

    Returns:
        controls: [1, horizon-1, 2] torch tensor
    """
    controls = torch.zeros(1, horizon - 1, 2)

    # Phase 1: to waypoint (60% of trajectory)
    n_first = int(horizon * 0.6)
    v_to_waypoint = (waypoint - start) / (n_first * dt)
    controls[:, :n_first, :] = torch.tensor(v_to_waypoint, dtype=torch.float32).view(
        1, 1, 2
    )

    # Phase 2: to goal (40% of trajectory)
    n_second = horizon - 1 - n_first
    v_to_goal = (goal - waypoint) / (n_second * dt)
    controls[:, n_first:, :] = torch.tensor(v_to_goal, dtype=torch.float32).view(
        1, 1, 2
    )

    return controls


def rollout_trajectory(initial_state, controls, dynamics, horizon):
    """
    Propagate belief through dynamics.

    Args:
        initial_state: [2] start position
        controls: [1, horizon-1, 2] control inputs
        dynamics: dynamics object with step() method
        horizon: int, trajectory length

    Returns:
        BeliefTrajectory
    """
    beliefs = []
    mu = torch.tensor(initial_state, dtype=torch.float32).view(1, 1, 2)
    var = torch.ones(1, 1, 2) * 0.01  # Initial uncertainty

    for t in range(horizon):
        belief = GaussianBelief(mu.clone(), var.clone(), confidence_level=2.0)
        beliefs.append(belief)

        if t < horizon - 1:
            u = controls[:, t : t + 1, :]
            mu, var = dynamics.step(mu, var, u)

    return BeliefTrajectory(beliefs)


def compute_loss(belief_traj, spec_safe, spec_goal, controls, goal, alpha=0.95):
    """
    Compute hinge loss for STL constraints.

    Args:
        belief_traj: BeliefTrajectory
        spec_safe: STL safety specification
        spec_goal: STL goal specification
        controls: [1, T-1, 2] control inputs
        goal: [2] goal position
        alpha: target probability threshold

    Returns:
        loss: scalar tensor
        metrics: dict with p_safe, p_goal, loss
    """
    # Evaluate STL specifications
    safe_trace = spec_safe(belief_traj)
    goal_trace = spec_goal(belief_traj)

    # Aggregate over time
    p_safe = safe_trace[:, :, 0].min(dim=1)[0].mean()  # Always: min
    p_goal = goal_trace[:, :, 0].max(dim=1)[0].mean()  # Eventually: max

    # Hinge loss (only penalize violations)
    L_safe = torch.relu(alpha - p_safe)
    L_goal = torch.relu(alpha - p_goal)

    # Terminal cost
    mu_final = belief_traj[-1].mean
    goal_tensor = torch.tensor(goal, dtype=torch.float32).view(1, 1, 2)
    L_terminal = torch.norm(mu_final - goal_tensor, p=2) ** 2

    # Regularization
    L_control = torch.sum(controls**2)
    L_smoothness = (
        torch.sum((controls[:, 1:, :] - controls[:, :-1, :]) ** 2)
        if controls.shape[1] > 1
        else torch.tensor(0.0)
    )

    # Weighted sum
    loss = (
        100.0 * L_safe
        + 100.0 * L_goal
        + 20.0 * L_terminal
        + 0.01 * L_control
        + 0.1 * L_smoothness
    )

    return loss, {"p_safe": p_safe.item(), "p_goal": p_goal.item(), "loss": loss.item()}


def optimize_trajectory(
    initial_state,
    goal_state,
    obstacle,
    dynamics,
    spec_safe,
    spec_goal,
    horizon=100,
    dt=0.15,
    num_iterations=300,
    lr=0.02,
    verbose=True,
    use_waypoint=True,
):
    """
    Gradient-based trajectory optimization.

    Args:
        initial_state: [2] start position
        goal_state: [2] goal position
        obstacle: dict with 'x' and 'y' bounds (or None for straight-line init)
        dynamics: dynamics object
        spec_safe: STL safety specification
        spec_goal: STL goal specification
        horizon: trajectory length
        dt: time step
        num_iterations: max optimization iterations
        lr: learning rate
        verbose: print progress
        use_waypoint: if True, use two-phase waypoint init, else straight-line

    Returns:
        means: [T, 2] trajectory positions
        vars: [T, 2] trajectory variances
        metrics: final metrics dict
        history: list of metrics per iteration
    """
    # Initialize controls
    if use_waypoint and obstacle is not None:
        waypoint = compute_waypoint(initial_state, goal_state, obstacle)
        if verbose:
            print(f"Waypoint: {waypoint}")
        controls = nn.Parameter(
            initialize_controls(initial_state, waypoint, goal_state, horizon, dt)
        )
    else:
        if verbose:
            print("Straight-line initialization")
        controls = nn.Parameter(
            initialize_controls_straight(initial_state, goal_state, horizon, dt)
        )

    optimizer = torch.optim.Adam([controls], lr=lr)

    if verbose:
        print("\nOptimizing...")
        print(f"{'Iter':>6} {'P_safe':>8} {'P_goal':>8}")
        print("-" * 30)

    history = []

    # Optimization loop
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        belief_traj = rollout_trajectory(initial_state, controls, dynamics, horizon)
        loss, metrics = compute_loss(
            belief_traj, spec_safe, spec_goal, controls, goal_state
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([controls], max_norm=10.0)
        optimizer.step()

        history.append(metrics)

        if verbose and (iteration % 50 == 0 or iteration == num_iterations - 1):
            print(f"{iteration:6d} {metrics['p_safe']:8.4f} {metrics['p_goal']:8.4f}")

        # Early stopping
        if metrics["p_safe"] >= 0.95 and metrics["p_goal"] >= 0.95:
            if verbose:
                print(f"\n✓ Early stop at iteration {iteration}")
            break

    if verbose:
        print("-" * 30)

    # Extract final trajectory
    with torch.no_grad():
        belief_traj = rollout_trajectory(
            initial_state, controls.detach(), dynamics, horizon
        )
        means = np.array(
            [belief_traj[t].mean.squeeze().numpy() for t in range(len(belief_traj))]
        )
        vars = np.array(
            [belief_traj[t].var.squeeze().numpy() for t in range(len(belief_traj))]
        )

    return means, vars, metrics, history
