"""Task geometry, deterministic commands, and mean_model / pdstl plans.

A plan is a world-frame velocity sequence, one command per `dt`, plus the
provenance needed to trust it. The same `Task` geometry scores plans and,
later, the recorded flights.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from config import HERE, git_commit

# isort: split  (config puts the pdSTL core in src/ on sys.path)
from baselines.det_stl import from_pdstl
from models.rollouts import (
    create_gaussian_belief_trajectory,
    gaussian_rollout,
)
from planning.environment import Environment, inside, safety_event
from planning.planner import Planner

PLANS = HERE / "plans"
CONDITIONS = ("deterministic", "mean_model", "pdstl")


def _box_distance(xy, box):
    """Signed distance from points [..., 2] to a box: > 0 outside."""
    lo, hi = np.array([box.x[0], box.y[0]]), np.array([box.x[1], box.y[1]])
    d = np.maximum(lo - xy, xy - hi)
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    return np.where(d.max(-1) > 0, outside, d.max(-1))


class Task:
    """Stay in the workspace, avoid obstacles, end inside the goal.

    Obstacles grow and the workspace shrinks by the airframe radius, so
    every check is on the drone's centre. The goal is required only at the
    final step H, so starting inside it never counts as completion.
    """

    def __init__(self, cfg):
        r = cfg.airframe_radius
        self.workspace = cfg.workspace.grown(-r)
        self.obstacles = [box.grown(r) for box in cfg.obstacles]
        self.goal = cfg.goal
        self.env = Environment()
        self.env.set_bounds(self.workspace.x, self.workspace.y)
        for box in self.obstacles:
            self.env.add_obstacle(box.x, box.y, name=box.name)
        self.env.set_goal(
            self.goal.x, self.goal.y, interval=[cfg.steps] * 2, name="goal"
        )
        self.spec = self.env.get_specification(cfg.steps)

    def clearance(self, xy):
        """Airframe clearance [m] of positions [..., 2]; < 0 is a violation."""
        xy = np.asarray(xy, dtype=float)
        walls = -_box_distance(xy, self.workspace)
        return np.min(
            [walls, *(_box_distance(xy, box) for box in self.obstacles)],
            axis=0,
        )

    def requirements(self, path, final):
        """Each requirement on a path [N, 2] ending at `final` [2]: met or not."""
        met = {
            "workspace": bool(
                (-_box_distance(path, self.workspace) >= 0).all()
            )
        }
        for box in self.obstacles:
            met[box.name] = bool((_box_distance(path, box) >= 0).all())
        met["goal"] = bool(_box_distance(np.asarray(final), self.goal) <= 0)
        return met


# Deterministic commands


def reference_velocities(cfg):
    """Constant speed along start -> waypoints -> goal centre, then zero."""
    points = np.array(
        [cfg.start, *cfg.reference["waypoints"], cfg.goal.centre]
    )
    ends = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )
    speed = cfg.reference["speed"]
    if ends[-1] / speed > cfg.duration:
        raise ValueError("reference does not fit in the mission duration")
    s = np.minimum(speed * cfg.dt * np.arange(cfg.steps + 1), ends[-1])
    path = np.stack([np.interp(s, ends, points[:, i]) for i in (0, 1)], 1)
    return np.diff(path, axis=0) / cfg.dt


def excitation_velocities(cfg, name):
    """Piecewise-constant [vx, vy, seconds] segments at dt."""
    segments = cfg.excitation["sequences"][name]
    return np.array(
        [
            [vx, vy]
            for vx, vy, sec in segments
            for _ in range(round(sec / cfg.dt))
        ],
        dtype=float,
    )


def nominal_path(cfg, start, velocities):
    """Positions at the streaming rate if the drone tracked v exactly."""
    v = np.repeat(np.asarray(velocities), cfg.repeats, axis=0)
    steps = np.cumsum(v / cfg.stream_hz, axis=0)
    return np.vstack([start, np.asarray(start) + steps])


def check_feasible(cfg, task, start, velocities, mean=None, mission=True):
    """Physical feasibility shared by every condition; problems, or []."""
    v = np.asarray(velocities)
    problems = []
    if np.abs(v).max() > cfg.u_max + 1e-9:
        problems.append(f"command above u_max = {cfg.u_max} m/s")
    for label, path in (
        ("nominal path", nominal_path(cfg, start, v)),
        ("predicted mean", mean),
    ):
        if path is not None and task.clearance(path).min() < 0:
            problems.append(f"{label} leaves the free space")
    if mission and len(v) != cfg.steps:
        problems.append(f"{len(v)} steps, mission needs {cfg.steps}")
    return problems


# Model-based plans


def plan_pdstl(cfg, task, model, init):
    """Maximize the pdSTL lower bound with the core Planner."""
    planner = Planner(model, cfg.steps, cfg.planner)
    result = planner.optimize_window(
        gaussian_rollout(model, *model.initial),
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

    Same tanh control bound, Adam step size, iteration count and control
    cost as `plan_pdstl`; only the task score differs. Setting the
    covariance to zero would not do: pdSTL probabilities are then 0 or 1
    and carry no gradient.
    """
    planner = Planner(model, cfg.steps, cfg.planner)
    robustness = from_pdstl(task.spec)
    params = planner._init_controls(torch.as_tensor(init, dtype=torch.float32))
    optimizer = torch.optim.Adam([params], lr=planner.cfg["lr"])
    best, best_controls = -math.inf, None
    for step in range(planner.cfg["max_iters"] + 1):
        controls = model.bound_control(params)
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


# Plan files


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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return {
        "id": f"{condition}_{name}_{stamp}",
        "condition": condition,
        "name": name,
        "fan": fan,
        "dt": cfg.dt,
        "altitude": cfg.altitude,
        "start": list(start),
        "velocities": np.asarray(velocities).tolist(),
        "config_hash": cfg.hash,
        "model_id": None if model is None else model.report_id,
        "model_hash": None if model is None else model.report_hash,
        "prediction": pred,
        "score": score,
        "problems": check_feasible(
            cfg, task, start, velocities, mean, mission
        ),
        "created": stamp,
        "git_commit": git_commit(),
    }


def save_plan(plan):
    PLANS.mkdir(exist_ok=True)
    path = PLANS / f"{plan['id']}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(plan, indent=1))
    return path


def load_plan(path, cfg):
    """A saved plan, refused if the config changed or it is not flyable."""
    plan = json.loads(Path(path).read_text())
    if plan["config_hash"] != cfg.hash:
        raise ValueError(
            f"plan {plan['id']} was made with config {plan['config_hash']}, "
            f"current config is {cfg.hash}; re-plan"
        )
    if plan["problems"]:
        raise ValueError(
            f"plan {plan['id']} is infeasible: {plan['problems']}"
        )
    if plan["score"] is not None and not plan["score"]["accepted"]:
        raise ValueError(f"plan {plan['id']} missed its score threshold")
    return plan
