"""Glue shared by every experiment: set up, solve, validate, and save."""

from pathlib import Path
from types import SimpleNamespace

import torch
from scipy.stats import beta

from models.dynamics import DoubleIntegrator, SingleIntegrator
from models.rollouts import (
    create_gaussian_belief_trajectory,
    gaussian_rollout,
    sample_trajectories,
)
from planning.planner import Planner
from utils import get_device

RESULTS_DIR = Path(__file__).resolve().parents[2] / "outputs"


def build_dynamics(cfg, device):
    common = {key: cfg[key] for key in ("dt", "u_max", "q_std")}
    if cfg.get("dynamics", "single_integrator") == "double_integrator":
        return DoubleIntegrator(**common, device=device)
    return SingleIntegrator(**common, device=device)


def build_initial_belief(cfg, device):
    """Gaussian x0: `x0_cov_scale` is one variance or one per state."""
    mean = torch.tensor(cfg["x0_mean"], device=device, dtype=torch.float32)
    variance = torch.as_tensor(
        cfg["x0_cov_scale"], device=device, dtype=torch.float32
    )
    return mean, torch.diag(variance.expand(len(mean)))


def setup_problem(cfg, *, device=None):
    """Dynamics, initial belief, planner, and Gaussian rollout of a config."""
    device = get_device() if device is None else device
    dyn = build_dynamics(cfg, device)
    state = build_initial_belief(cfg, device)
    return SimpleNamespace(
        cfg=cfg,
        dyn=dyn,
        state=state,
        planner=Planner(dyn, cfg["H"], cfg.get("planner", {})),
        rollout=gaussian_rollout(dyn, *state),
        init_guess=None,
    )


def solve(s, *, label, verbose=False, on_iteration=None, callback_every=1):
    """Optimize one open-loop plan and print its lower robustness.

    Args:
        s: Problem from setup_problem, with `spec` and `init_guess` set.
        label: Name printed with the result.
        verbose: Log optimizer progress.
        on_iteration, callback_every: Optional observer of sampled iterates.

    Returns:
        The PlanResult with the highest exact lower robustness.
    """
    result = s.planner.optimize_window(
        s.rollout,
        spec=s.spec,
        init_guess=s.init_guess,
        verbose=verbose,
        on_iteration=on_iteration,
        callback_every=callback_every,
    )
    print(
        f"{label}: rho_lower {result.initial_hard_lower:.3f} -> "
        f"{result.hard_interval[0]:.3f} (smooth "
        f"{result.initial_smooth_lower:.3f} -> {result.smooth_lower:.3f}); "
        f"{result.alpha_status()}",
        flush=True,
    )
    return result


def monte_carlo(s, result, samples=5000, seed=0):
    """Fraction of sampled trajectories that satisfy the spec.

    Samples the stochastic dynamics under the plan's controls and scores each
    trajectory with the same pdSTL spec: with zero variance every atom is
    0 or 1, so the Fréchet operators reduce to Boolean logic.

    Args:
        s: Problem with `spec`.
        result: PlanResult whose controls are executed open loop.
        samples, seed: Number of trajectories and random seed.

    Returns:
        (rate, lower, upper): the rate and its 95% Clopper-Pearson interval.
    """
    generator = torch.Generator(device=s.dyn.device).manual_seed(seed)
    paths = sample_trajectories(
        s.dyn, *s.state, result.controls, samples, generator
    )
    beliefs = create_gaussian_belief_trajectory(paths, torch.zeros_like(paths))
    hits = int(s.spec(beliefs)[:, 0, 0].sum())
    lower = beta.ppf(0.025, hits, samples - hits + 1) if hits else 0.0
    upper = (
        beta.ppf(0.975, hits + 1, samples - hits) if hits < samples else 1.0
    )
    return hits / samples, float(lower), float(upper)


def output_path(stem, suffix):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{stem}{suffix}"


def save_result(result, stem, save):
    if save:
        torch.save(result, output_path(stem, ".pt"))


def control_unit(dyn):
    return "m/s²" if isinstance(dyn, DoubleIntegrator) else "m/s"
