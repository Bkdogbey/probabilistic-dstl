"""2D baseline reach-avoid mission: plain (x, y) Environment/Planner/
SingleIntegrator from src/planning/, plus the deterministic nominal path.

Altitude is never a decision variable here -- the optimizer's state/control
space is strictly (x, y); Z_HEIGHT is applied as a constant afterward.
"""

from __future__ import annotations

import numpy as np
import torch

# Imported first so components.config's sys.path insert runs before
# planning.* is imported, regardless of which module loads first.
from components.config import (
    DETERMINISTIC_VIA_POINTS,
    DT,
    END_XY,
    FLIGHT_X_BOUNDS,
    FLIGHT_Y_BOUNDS,
    GOAL,
    OBSTACLES,
    PLANNER_CONFIG,
    Q_STD,
    SIGMA0_PER_FAN,
    START_XY,
    T,
    U_MAX,
    Z_HEIGHT,
)
from planning.dynamics import SingleIntegrator
from planning.environment import Environment
from planning.planner import Planner


def nominal_safe_waypoints(n_points: int = T + 1) -> list[tuple[float, float, float]]:
    """Deterministic path and pdSTL optimizer warm start: a sine curve through
    START_XY, DETERMINISTIC_VIA_POINTS, and END_XY.

    y is linear in progress s in [0, 1]; x is the straight line plus one sine
    harmonic per via-point, x(s) = x_linear(s) + sum_k a_k*sin(k*pi*s). Every
    harmonic vanishes at s=0 and s=1, so the coefficients a_k are solved
    (closed form, one linear system) to hit each via-point exactly without
    perturbing start/end.
    """
    if n_points < 2:
        raise ValueError('n_points must be at least 2')

    y_span = END_XY[1] - START_XY[1]
    s_of_y = lambda y: (y - START_XY[1]) / y_span
    x_linear = lambda s: START_XY[0] + s * (END_XY[0] - START_XY[0])

    via = DETERMINISTIC_VIA_POINTS
    s_via = np.array([s_of_y(y) for _, y in via])
    harmonics = np.arange(1, len(via) + 1)
    basis = np.sin(np.pi * np.outer(s_via, harmonics))
    targets = np.array([x - x_linear(s) for (x, _), s in zip(via, s_via)])
    coeffs = np.linalg.solve(basis, targets)

    s = np.linspace(0.0, 1.0, n_points)
    x_pos = x_linear(s) + np.sin(np.pi * np.outer(s, harmonics)) @ coeffs
    y_pos = START_XY[1] + s * y_span
    return [(float(x), float(y), Z_HEIGHT) for x, y in zip(x_pos, y_pos)]


def build_environment_2d() -> Environment:
    """Build the pdSTL Environment for the 2D baseline reach-avoid mission."""
    env = Environment()
    env.set_bounds(x_range=FLIGHT_X_BOUNDS, y_range=FLIGHT_Y_BOUNDS)
    for obs in OBSTACLES:
        env.add_obstacle(x_range=list(obs['x']), y_range=list(obs['y']))
    env.set_goal(x_range=list(GOAL['x']), y_range=list(GOAL['y']))
    return env


def build_planner_2d(fan_speed: int) -> tuple[Planner, SingleIntegrator, Environment]:
    """Build a (Planner, dynamics, environment) for the 2D baseline mission."""
    dynamics = SingleIntegrator(dt=DT, u_max=U_MAX, q_std=Q_STD)
    env = build_environment_2d()
    planner = Planner(dynamics, env, T, config=dict(PLANNER_CONFIG))
    return planner, dynamics, env


def x0_belief_2d(fan_speed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Initial (x, y) belief (mean, cov) for the given fan level."""
    sigma0 = SIGMA0_PER_FAN[fan_speed]
    mean = torch.tensor([*START_XY], dtype=torch.float32)
    cov = torch.diag(torch.tensor([sigma0, sigma0], dtype=torch.float32))
    return mean, cov


def _nominal_init_guess_2d() -> torch.Tensor:
    """2D baseline warm start: (vx, vy) velocities from nominal_safe_waypoints."""
    wps = np.array(nominal_safe_waypoints(n_points=T + 1))
    vels_xy = np.diff(wps[:, :2], axis=0) / DT
    return torch.tensor(vels_xy, dtype=torch.float32)
