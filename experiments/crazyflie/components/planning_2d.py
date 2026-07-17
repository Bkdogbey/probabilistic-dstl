"""Everything 2D: the real 2D experiment (baseline reach-avoid mission), and
only the baseline mission.

Uses the plain, unmodified 2D Environment/Planner/SingleIntegrator from
src/planning/ -- not Environment3D/Planner3D/SingleIntegrator3D
(components/planning_3d.py, which this file never imports). Altitude is
never a decision variable here: the optimizer's state/control space is
strictly (x, y), and Z_HEIGHT is applied as a constant only after
optimizing, exactly like the deterministic nominal path already does. This
is what makes it structurally impossible for a "2D" plan to come back with
waypoints at different altitudes -- there's no z for the optimizer to move
in.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch

# Imported first: components.config does the one sys.path.insert(src/) every
# planning.* import in this file (and every other experiment module) relies
# on -- importing it before planning.* guarantees that regardless of which
# module gets imported first at runtime.
from components.config import (
    DETERMINISTIC_PATH_MARGIN,
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
    obstacle_clearance_2d,
)
from planning.dynamics import SingleIntegrator
from planning.environment import Environment
from planning.planner import Planner
from uncertainty_calibration import load_uncertainty_profile


class EmpiricalCovarianceSingleIntegrator(SingleIntegrator):
    """Single-integrator mean dynamics with a measured covariance profile."""

    def __init__(self, covariance_profile: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        profile = torch.as_tensor(covariance_profile, dtype=torch.float32)
        if profile.ndim != 3 or profile.shape[1:] != (2, 2):
            raise ValueError(f'Expected covariance profile [steps,2,2], got {profile.shape}')
        self.register_buffer('covariance_profile', profile)

    def forward(self, v_sequence, x0_mean, x0_cov):
        mean_stack, _ = super().forward(v_sequence, x0_mean, x0_cov)
        if len(self.covariance_profile) != mean_stack.shape[1]:
            raise ValueError(
                f'Covariance profile has {len(self.covariance_profile)} steps, '
                f'but rollout has {mean_stack.shape[1]}'
            )
        covariance = self.covariance_profile.to(
            device=mean_stack.device, dtype=mean_stack.dtype,
        ).unsqueeze(0)
        return mean_stack, covariance


def _obstacle_s_window(obs: dict, n_samples: int = 41) -> np.ndarray:
    """Sample points, in the sine path's normalized progress s in [0, 1],
    spanning one obstacle's y-extent (clipped to [0, 1] if the obstacle
    reaches past START_XY/END_XY)."""
    y0, y1 = obs['y']
    span = END_XY[1] - START_XY[1]
    s_lo = (min(y0, y1) - START_XY[1]) / span
    s_hi = (max(y0, y1) - START_XY[1]) / span
    return np.linspace(max(0.0, min(s_lo, s_hi)), min(1.0, max(s_lo, s_hi)), n_samples)


def _side_amplitude_bounds(obs: dict, margin: float) -> tuple[float, float]:
    """(left_min, right_max): bounds on the sine amplitude A in
    x(s) = x_linear(s) - A*sin(pi*s) for the curve to clear `obs` by
    `margin`, sampled over the obstacle's y-projected s-window --
    left_min is the smallest A that stays left of the obstacle throughout
    the window (A >= left_min); right_max is the largest A that stays right
    of it (A <= right_max). Either bound can be infeasible in practice (e.g.
    right_max < 0 means even a straight line already fails to clear the
    obstacle's right side) -- callers check the resulting interval, not
    these bounds in isolation.
    """
    s = _obstacle_s_window(obs)
    x_linear = START_XY[0] + s * (END_XY[0] - START_XY[0])
    sin_s = np.sin(np.pi * s)
    x_min, x_max = obs['x']
    mask = sin_s > 1e-9
    if not mask.any():
        return 0.0, float('inf')
    left_min = float(np.max((x_linear[mask] - x_min + margin) / sin_s[mask]))
    right_max = float(np.min((x_linear[mask] - x_max - margin) / sin_s[mask]))
    return left_min, right_max


def _calculate_sine_amplitude(margin: float = DETERMINISTIC_PATH_MARGIN) -> float:
    """Closed-form amplitude for the deterministic path's single-hump sine
    curve x(s) = x_linear(s) - A*sin(pi*s), calculated (not hand-tuned, not
    gradient-descent optimized) to clear every obstacle in OBSTACLES by
    `margin`.

    Each obstacle can be passed on its left (x(s) stays below x_min) or
    right (x(s) stays above x_max); _side_amplitude_bounds gives the A-bound
    for each side as a closed-form expression. With N obstacles there are
    only 2**N possible left/right assignments -- small enough (N=3 here) to
    just enumerate directly rather than search/optimize: for each
    assignment, intersect the per-obstacle bounds into a single feasible
    amplitude interval (or discard the assignment if the interval is
    empty), then keep whichever feasible assignment gives the largest
    worst-case clearance (checked with the real box-distance calculation,
    obstacle_clearance_2d, not just the linear bound) at the midpoint of its
    interval.
    """
    bounds = [_side_amplitude_bounds(obs, margin) for obs in OBSTACLES]
    best: tuple[float, float] | None = None  # (worst_case_clearance, amplitude)
    for combo in itertools.product((0, 1), repeat=len(OBSTACLES)):  # 0 = left, 1 = right
        lo, hi = 0.0, float('inf')
        for side, (left_min, right_max) in zip(combo, bounds):
            if side == 0:
                lo = max(lo, left_min)
            else:
                hi = min(hi, right_max)
        if lo > hi:
            continue  # this left/right assignment has no feasible amplitude
        amplitude = lo if hi == float('inf') else (lo + hi) / 2.0
        s = np.linspace(0.0, 1.0, 300)
        curve = np.column_stack([
            START_XY[0] + s * (END_XY[0] - START_XY[0]) - amplitude * np.sin(np.pi * s),
            START_XY[1] + s * (END_XY[1] - START_XY[1]),
        ])
        worst = min(obstacle_clearance_2d(curve, obs) for obs in OBSTACLES)
        if best is None or worst > best[0]:
            best = (worst, amplitude)

    if best is None or best[0] < margin - 1e-6:
        raise ValueError(
            'No single-hump sine amplitude clears every obstacle in OBSTACLES by '
            f'the required margin ({margin} m) on either side -- this obstacle '
            'layout needs hand-picked via-points instead (see planning_3d.py\'s '
            'nominal_gate_waypoints for that pattern).'
        )
    return best[1]


def nominal_safe_waypoints(n_points: int = T + 1) -> list[tuple[float, float, float]]:
    """The 2D baseline's deterministic path AND the pdSTL optimizer's warm
    start (_nominal_init_guess_2d) -- same dual role this function has
    always had. A single left/right-bending sine curve,
    x(s) = x_linear(s) - A*sin(pi*s), with A calculated by
    _calculate_sine_amplitude() to clear every obstacle by
    DETERMINISTIC_PATH_MARGIN -- this IS the curve that gets flown under
    `--condition deterministic`, unmodified; it is not further optimized.

    y(s) is exactly linear by construction, so unlike a hand-picked
    via-point path this can be evaluated directly at any n_points -- no
    resampling/caching needed.
    """
    if n_points < 2:
        raise ValueError('n_points must be at least 2')
    amplitude = _calculate_sine_amplitude()
    s = np.linspace(0.0, 1.0, n_points)
    x_pos = START_XY[0] + s * (END_XY[0] - START_XY[0]) - amplitude * np.sin(np.pi * s)
    y_pos = START_XY[1] + s * (END_XY[1] - START_XY[1])
    return [(float(x), float(y), Z_HEIGHT) for x, y in zip(x_pos, y_pos)]


def build_environment_2d() -> Environment:
    """Build the pdSTL Environment for the 2D baseline reach-avoid mission.

    Only x/y ranges are ever passed in -- OBSTACLES' 'height' and GOAL['z']
    (read by planning_3d.py's build_environment_3d) are simply not consulted
    here, since there's no altitude axis to place them on.
    """
    env = Environment()
    env.set_bounds(x_range=FLIGHT_X_BOUNDS, y_range=FLIGHT_Y_BOUNDS)
    for obs in OBSTACLES:
        env.add_obstacle(x_range=list(obs['x']), y_range=list(obs['y']))
    env.set_goal(x_range=list(GOAL['x']), y_range=list(GOAL['y']))
    return env


def build_planner_2d(fan_speed: int) -> tuple[Planner, SingleIntegrator, Environment]:
    """Build a (Planner, dynamics, environment) for the 2D baseline mission.

    A generated empirical covariance profile is used when available. Otherwise
    planning falls back to the paper Sigma0 plus shared process-noise model.
    """
    profile = load_uncertainty_profile(fan_speed, T + 1)
    if profile is None:
        dynamics = SingleIntegrator(dt=DT, u_max=U_MAX, q_std=Q_STD)
        dynamics.uncertainty_source = 'paper_sigma0_shared_process_noise'
    else:
        dynamics = EmpiricalCovarianceSingleIntegrator(
            profile['covariance'], dt=DT, u_max=U_MAX, q_std=Q_STD,
        )
        dynamics.uncertainty_source = str(profile['path'])
    env = build_environment_2d()
    planner = Planner(dynamics, env, T, config=dict(PLANNER_CONFIG))
    return planner, dynamics, env


def x0_belief_2d(fan_speed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Initial (x, y) belief (mean, cov) for the given fan level.

    The paper's empirically characterized variance for this fan setting is
    placed on each axis of the initial belief covariance.
    """
    profile = load_uncertainty_profile(fan_speed, T + 1)
    sigma0 = SIGMA0_PER_FAN[fan_speed]
    mean = torch.tensor([*START_XY], dtype=torch.float32)
    cov = (
        torch.as_tensor(profile['covariance'][0], dtype=torch.float32)
        if profile is not None
        else torch.diag(torch.tensor([sigma0, sigma0], dtype=torch.float32))
    )
    assert mean.shape == (2,), f'x0 mean must be 2D, got {tuple(mean.shape)}'
    assert cov.shape == (2, 2), f'x0 cov must be 2x2, got {tuple(cov.shape)}'
    return mean, cov


def _nominal_init_guess_2d() -> torch.Tensor:
    """2D baseline warm start: (vx, vy) velocities from nominal_safe_waypoints.

    Only two columns -- there's no z-velocity to zero out or otherwise
    account for, unlike the gate scenario's 3-column warm start.
    """
    wps = np.array(nominal_safe_waypoints(n_points=T + 1))
    vels_xy = np.diff(wps[:, :2], axis=0) / DT
    init_u = torch.tensor(vels_xy, dtype=torch.float32)
    assert init_u.shape == (T, 2), f'init_u must be [T, 2], got {tuple(init_u.shape)}'
    return init_u
