"""Single source of truth for arena geometry and how to build a planner from it.

Every other file in this experiment (offline planning, flight execution,
logging) imports geometry and pdSTL types from here instead of duplicating
constants or reaching into src/ directly. Update the arena numbers below in
this one place when the flight area is re-measured.
"""

from __future__ import annotations

import pathlib
import sys

# Use the pdSTL library from this repo's src/ (no vendored copies). Every
# other file gets Environment/SingleIntegrator/Planner/etc. through this
# module's re-exports below, so nothing else needs to touch sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'src'))

import torch

from pdstl.base import BeliefTrajectory as BeliefTrajectory
from planning.dynamics import SingleIntegrator as SingleIntegrator
from planning.environment import Environment as Environment
from planning.planner import Planner as Planner
from planning.planner import TorchGaussianBelief as TorchGaussianBelief

# ── Arena geometry — TODO: replace with the newly measured arena numbers ────
FLIGHT_X_BOUNDS: list[float] = [-0.5, 1.0]
FLIGHT_Y_BOUNDS: list[float] = [-2.0, 0.5]
Z_HEIGHT: float = 0.2  # flight height [m], fixed altitude (2D planning)

# TODO: replace with newly measured obstacle boxes. Single list consumed by
# both the planner (generate_waypoints.py) and the safety flags in the
# flight logger — no more duplicating obstacle boxes across files.
OBSTACLES: list[dict] = [
    {'name': 'obs_1', 'x': (-0.165, 0.165), 'y': (-1.144, -0.941)},  # red
    {'name': 'obs_2', 'x': (-0.432, -0.102), 'y': (-0.179, 0.049)},  # blue
    {'name': 'obs_3', 'x': (0.114, 0.343), 'y': (-0.611, -0.421)},  # green
]

GOAL: dict = {'x': (-0.30, 0.30), 'y': (0.10, 0.30)}

START_XY: tuple[float, float] = (0.0, -1.5)
# Abort the flight if the measured hover position at start differs from
# START_XY by more than this (m) — see components/calibration.py.
START_TOLERANCE: float = 0.08

# ── Planning constants (measured from the experiment setup) ─────────────────
T: int = 10  # nominal planning horizon (number of control steps)
DT: float = 0.693  # mean inter-waypoint time [s]
U_MAX: float = 0.44  # max inter-waypoint speed [m/s]

Q_STD_PER_FAN: dict[int, float] = {
    2: 0.001,
    6: 0.006,
    12: 0.020,
    16: 0.050,
}

PLANNER_ALPHA: float = 0.90
PLANNER_CONFIG: dict = {
    'w_phi': 20.0,
    'w_dist': 5.0,
    'w_obs': 5.0,
    'alpha': PLANNER_ALPHA,
    'max_iters': 1000,
    'lr': 0.02,
}

X0_COV_DIAG: tuple[float, float] = (1e-2, 1e-2)  # initial belief uncertainty
# Tighter: a live Lighthouse reading is much more certain than the initial
# pre-flight belief above, used when replanning mid-flight from measured state.
MEASURED_COV_DIAG: tuple[float, float] = (1e-3, 1e-3)

# Waypoint indices (0-based, into the T+1 = 11-point plan) at which
# components/crazyflie.py replans from the actual measured position instead
# of blindly continuing the offline plan. Append a second index for two
# mid-flight replans.
REPLAN_CHECKPOINTS: list[int] = [5]


def build_environment() -> Environment:
    """Build the pdSTL Environment (bounds, obstacles, goal) from the geometry above."""
    env = Environment()
    env.set_bounds(x_range=FLIGHT_X_BOUNDS, y_range=FLIGHT_Y_BOUNDS)
    for obs in OBSTACLES:
        env.add_obstacle(x_range=list(obs['x']), y_range=list(obs['y']))
    env.set_goal(x_range=list(GOAL['x']), y_range=list(GOAL['y']))
    return env


def build_planner(fan_speed: int, T_horizon: int | None = None) -> tuple[Planner, SingleIntegrator, Environment]:
    """Build a (Planner, dynamics, environment) sized to `T_horizon` steps.

    `T_horizon` defaults to the full nominal horizon `T`; pass a smaller
    value to size a planner for a mid-flight replan over the remaining steps.
    """
    horizon = T if T_horizon is None else T_horizon
    dynamics = SingleIntegrator(dt=DT, u_max=U_MAX, q_std=Q_STD_PER_FAN[fan_speed])
    env = build_environment()
    planner = Planner(dynamics, env, horizon, config=dict(PLANNER_CONFIG))
    return planner, dynamics, env


def x0_belief() -> tuple[torch.Tensor, torch.Tensor]:
    """Initial (mean, cov) belief for offline planning, from the assumed start."""
    mean = torch.tensor(START_XY, dtype=torch.float32)
    cov = torch.diag(torch.tensor(X0_COV_DIAG, dtype=torch.float32))
    return mean, cov


def measured_belief(xy: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean, cov) belief built from a live measured position, for mid-flight replanning."""
    mean = torch.tensor(xy, dtype=torch.float32)
    cov = torch.diag(torch.tensor(MEASURED_COV_DIAG, dtype=torch.float32))
    return mean, cov
