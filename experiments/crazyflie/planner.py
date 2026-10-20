"""Offline mean-model and pdSTL planning, initialized from a frozen path."""

import math
from datetime import datetime, timezone

import numpy as np
import torch
from config import git_commit
from geometry import check_feasible, nominal_path
from trajectory import condition_name, file_hash

from baselines.det_stl import from_pdstl
from models.rollouts import BeliefRollout, create_gaussian_belief_trajectory
from planning.environment import inside, safety_event
from planning.planner import Planner


def _rollout(model):
    """Gaussian belief rollout under the model's speed and acceleration limits."""

    def rollout(params):
        controls = model.controls(params)
        mean, cov = model.predict(controls)
        return BeliefRollout(
            create_gaussian_belief_trajectory(mean, cov),
            {
                "applied_controls": controls,
                "mean_trace": mean,
                "cov_trace": cov,
            },
        )

    return rollout


def plan_pdstl(cfg, task, model, init):
    """Maximize the pdSTL lower bound with the core Planner."""
    planner = Planner(model, cfg.steps, cfg.planner)
    result = planner.optimize_window(
        _rollout(model),
        spec=task.spec,
        init_guess=torch.as_tensor(init, dtype=torch.float32),
    )
    lower = result.hard_interval[0]
    return result.controls.cpu().numpy(), {
        "pdstl_lower": lower,
        "accepted": lower >= cfg.planner["alpha"],
    }


def plan_mean(cfg, task, model, init):
    """Maximize STL robustness of the predicted mean, with matched budget.

    Same speed and acceleration limits, Adam step size, iteration count and
    control cost as `plan_pdstl`; only the task score differs. Setting the
    covariance to zero would not do: pdSTL probabilities are then 0 or 1
    and carry no gradient.
    """
    planner = Planner(model, cfg.steps, cfg.planner)
    robustness = from_pdstl(task.spec)
    params = planner._init_controls(torch.as_tensor(init, dtype=torch.float32))
    optimizer = torch.optim.Adam([params], lr=planner.cfg["lr"])
    best, best_controls = -math.inf, None
    for step in range(planner.cfg["max_iters"] + 1):
        controls = model.controls(params)
        mean = model.predict(controls)[0].unsqueeze(0)
        with torch.no_grad():
            hard = robustness.robustness(mean, beta=None).item()
        if hard >= best:
            best, best_controls = hard, controls.detach().clone()
        if step == planner.cfg["max_iters"]:
            break
        smooth = robustness.robustness(mean, beta=cfg.mean_model["beta"])[0]
        loss = -cfg.mean_model["w_rho"] * smooth + planner._control_cost(
            controls
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return best_controls.numpy(), {
        "robustness": best,
        "accepted": best >= cfg.mean_model["min_robustness"],
    }


def prediction(task, model, controls):
    """Predicted mean/covariance and per-step safe and goal probabilities."""
    with torch.no_grad():
        mean, cov = model.predict(
            torch.as_tensor(controls, dtype=torch.float32)
        )
        beliefs = create_gaussian_belief_trajectory(mean, cov)
        safe = safety_event(task.env)(beliefs)[0]
        goal = inside(task.env.region("goal"))(beliefs)[0]
    return {
        "mean": mean.tolist(),
        "cov": cov.tolist(),
        "safe_probability": safe.tolist(),
        "goal_probability": goal.tolist(),
    }


def make_plan(
    cfg,
    condition,
    name,
    start,
    velocities,
    task,
    fan=None,
    model=None,
    score=None,
    mission=True,
):
    """A self-describing plan record (not yet saved)."""
    pred = None if model is None else prediction(task, model, velocities)
    mean = None if pred is None else np.asarray(pred["mean"])[:, :2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return {
        "id": f"{condition}_{name}_{stamp}",
        "condition": condition_name(condition),
        "name": name,
        "fan": fan,
        "dt": cfg.dt,
        "altitude": cfg.altitude,
        "start": list(start),
        "velocities": np.asarray(velocities).tolist(),
        "config_hash": cfg.plan_hash,
        "execution_hash": cfg.execution_hash,
        "model_id": None if model is None else model.report_id,
        "model_hash": None if model is None else model.report_hash,
        "prediction": pred,
        "score": score,
        "min_clearance": float(
            task.clearance(nominal_path(cfg, start, velocities)).min()
        ),
        "problems": check_feasible(
            cfg, task, start, velocities, mean, mission
        ),
        "created": stamp,
        "git_commit": git_commit(),
        "reference_hash": file_hash(cfg.reference),
    }
