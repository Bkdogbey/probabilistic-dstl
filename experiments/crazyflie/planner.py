"""Offline waypoint planning for the uploaded curve: pdSTL and mean model.

Decision variables are the waypoints w_1..w_M (xy). w_0 is the start, and
the segment durations are the reference's. Each iteration maps the
waypoints to min-snap coefficients (c = K w), samples the curve on the
planning grid behind the model's delay, r(t_k - d), builds the
tracking-model belief (mean r(t_k - d) + m_k, covariance P_k) and scores it.

Both conditions run the same Adam loop, iteration count, learning-rate
sweep and penalties; only the task score differs:
  pdstl       spec.smooth_lower with the core's geometric beta schedule;
              the exact score is the pdSTL lower bound.
  mean_model  smooth STL robustness of the predicted mean r + m; the exact
              score is its hard robustness.

A probability and a robustness are on different scales, so equal weights
alone do not make the comparison fair. The limit penalty is zero inside the
limits, so it only acts on infeasible iterates. w_snap is set per condition
so the snap term's gradient is `snap_fraction` of the task score's gradient
at the reference. The learning rate is picked per condition from the same
sweep.

The reference and every iterate are candidates: each is quantized to the
uploaded float32 rows, checked densely and scored exactly on those rows.
The best feasible candidate is kept.
"""

from datetime import datetime, timezone

import minsnap
import numpy as np
import torch
from config import git_commit
from geometry import check_feasible
from trajectory import check_plan, file_hash, make_curve, new_plan

# isort: split  (config puts the pdSTL core in src/ on sys.path)
from baselines.det_stl import from_pdstl
from models.rollouts import create_gaussian_belief_trajectory
from planning.environment import inside, safety_event
from utils import load_config

CONDITIONS = ("pdstl", "mean_model")


class Problem:
    """Everything fixed during one plan: curve maps, belief moments, spec."""

    def __init__(self, cfg, task, model, reference):
        self.cfg, self.task, self.model = cfg, task, model
        self.durations = list(reference["durations"])
        waypoints = torch.tensor(reference["waypoints"], dtype=torch.float64)
        self.start, self.init = waypoints[:1], waypoints[1:]
        rows = np.asarray(reference["coefficients"], dtype=float)
        self.times = minsnap.grid_times(rows, cfg.dt)
        # The drone trails the curve by the model's delay; the curve holds
        # its start before t = 0.
        self.delayed = self.times - model.delay
        dense = minsnap.dense_times(rows)
        K = minsnap.solve_matrix(self.durations)
        self.grid = minsnap.sample_matrix(self.durations, self.delayed) @ K
        self.vel = minsnap.sample_matrix(self.durations, dense, 1) @ K
        self.acc = minsnap.sample_matrix(self.durations, dense, 2) @ K
        self.snap = K.T @ minsnap.snap_matrix(self.durations) @ K
        self.m, self.P = model.error_moments(len(self.times) - 1)
        self.spec = task.spec
        self.robustness = from_pdstl(self.spec)
        smoothing = load_config("configs/planning.yaml")["smoothing"]
        self.beta_start = smoothing["beta_start"]
        self.beta_end = smoothing["beta_end"]

    def waypoints(self, free):
        return torch.cat([self.start, free])

    def beta(self, condition, iteration, iterations):
        """pdstl: the core's geometric schedule; mean_model: a fixed beta."""
        if condition == "mean_model":
            return self.cfg.mean_model["beta"]
        fraction = iteration / max(iterations - 1, 1)
        return self.beta_start * (self.beta_end / self.beta_start) ** fraction

    def final_beta(self, condition):
        return self.beta(condition, 1, 2)

    def beliefs(self, mean):
        return create_gaussian_belief_trajectory(mean, self.P)

    def smooth_score(self, condition, mean, beta):
        if condition == "pdstl":
            return self.spec.smooth_lower(self.beliefs(mean), beta)
        return self.robustness.robustness(mean.unsqueeze(0), beta=beta)[0]

    def exact_score(self, condition, mean):
        with torch.no_grad():
            if condition == "pdstl":
                interval = self.spec.probability_interval(self.beliefs(mean))
                return float(interval[0])
            return float(self.robustness.robustness(mean.unsqueeze(0))[0])

    def limit_penalty(self, w):
        speed = torch.linalg.vector_norm(self.vel @ w, dim=1)
        accel = torch.linalg.vector_norm(self.acc @ w, dim=1)
        excess = torch.relu(speed - self.cfg.v_max).square()
        excess = excess + torch.relu(accel - self.cfg.a_max).square()
        return self.cfg.planner["w_lim"] * excess.sum()

    def snap_energy(self, w):
        return (w * (self.snap @ w)).sum()

    def rows_mean(self, rows):
        """Predicted mean [H+1, 2] of uploaded rows."""
        r = minsnap.evaluate(rows, self.delayed)[:, :2]
        return torch.as_tensor(r) + self.m

    def candidate(self, condition, free, iteration, w_snap):
        """Quantized record of an iterate, checked and scored on its rows."""
        with torch.no_grad():
            w = self.waypoints(free)
            penalty = self.limit_penalty(w) + w_snap * self.snap_energy(w)
        w = w.numpy()
        rows = make_curve(w, self.durations, self.cfg.altitude)
        mean = self.rows_mean(rows)
        problems = check_feasible(
            self.cfg,
            self.task,
            {"coefficients": rows, "waypoints": w},
            mean.numpy(),
        )
        with torch.no_grad():
            smooth = self.smooth_score(
                condition, mean, self.final_beta(condition)
            )
        return {
            "iteration": iteration,
            "waypoints": w,
            "feasible": not problems,
            "exact": self.exact_score(condition, mean),
            "smooth": float(smooth),
            "penalty": float(penalty),
        }


def select(candidates):
    """The best feasible candidate, or the best of all if none is feasible.

    Highest exact score; ties go to the higher smooth score at the final
    beta, then the lower penalty, then the earlier iterate.
    """
    pool = [c for c in candidates if c["feasible"]] or candidates
    return max(
        pool,
        key=lambda c: (
            c["exact"],
            c["smooth"],
            -c["penalty"],
            -c["iteration"],
        ),
    )


def snap_weight(problem, condition):
    """w_snap: the snap gradient is snap_fraction of the task gradient."""
    free = problem.init.clone().requires_grad_()
    beta = problem.beta(condition, 0, problem.cfg.planner["max_iters"])
    mean = problem.grid @ problem.waypoints(free) + problem.m
    (task,) = torch.autograd.grad(
        problem.smooth_score(condition, mean, beta), free
    )
    (snap,) = torch.autograd.grad(
        problem.snap_energy(problem.waypoints(free)), free
    )
    if snap.norm() == 0:
        return 0.0
    fraction = problem.cfg.planner["snap_fraction"]
    return float(fraction * task.norm() / snap.norm())


def optimize(problem, condition, lr, iterations, w_snap):
    """Adam on w_1..w_M; returns the selected candidate."""
    free = torch.nn.Parameter(problem.init.clone())
    optimizer = torch.optim.Adam([free], lr=lr)
    candidates = [problem.candidate(condition, free, -1, w_snap)]
    for i in range(iterations):
        w = problem.waypoints(free)
        mean = problem.grid @ w + problem.m
        beta = problem.beta(condition, i, iterations)
        loss = -problem.smooth_score(condition, mean, beta)
        loss = (
            loss + problem.limit_penalty(w) + w_snap * problem.snap_energy(w)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        candidates.append(problem.candidate(condition, free, i, w_snap))
    return select(candidates)


def prediction(problem, rows):
    """Predicted mean and covariance, safe and goal probability per step."""
    mean = problem.rows_mean(rows)
    with torch.no_grad():
        beliefs = problem.beliefs(mean)
        safe = safety_event(problem.task.env)(beliefs)[0]
        goal = inside(problem.task.env.region("goal"))(beliefs)[0]
    return {
        "mean": mean.tolist(),
        "cov": problem.P.tolist(),
        "safe_probability": safe.tolist(),
        "goal_probability": goal.tolist(),
    }


def plan(cfg, task, model, reference, condition, fan):
    """Sweep the learning rate, optimize, and return a checked plan record."""
    problem = Problem(cfg, task, model, reference)
    settings = cfg.planner
    w_snap = snap_weight(problem, condition)
    sweep = []
    for lr in settings["lr_grid"]:
        best = optimize(
            problem, condition, lr, settings["sweep_iters"], w_snap
        )
        keys = ("feasible", "exact", "smooth", "penalty")
        sweep.append({"lr": lr, **{k: best[k] for k in keys}})
    # Same rule as `select`; a tie keeps the smaller learning rate.
    lr = max(
        sweep,
        key=lambda s: (s["feasible"], s["exact"], s["smooth"], -s["penalty"]),
    )["lr"]
    best = optimize(problem, condition, lr, settings["max_iters"], w_snap)
    return make_plan(
        cfg,
        task,
        problem,
        condition,
        fan,
        best,
        {
            "w_lim": settings["w_lim"],
            "w_snap": w_snap,
            "snap_fraction": settings["snap_fraction"],
            "lr": lr,
            "lr_sweep": sweep,
            "sweep_iters": settings["sweep_iters"],
            "iterations": settings["max_iters"],
            "beta": (
                [problem.beta_start, problem.beta_end]
                if condition == "pdstl"
                else cfg.mean_model["beta"]
            ),
            "selected_iteration": best["iteration"],
        },
    )


def make_plan(cfg, task, problem, condition, fan, best, objective):
    """A self-describing plan whose scores are recomputed from its rows."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    record = new_plan(
        cfg,
        best["waypoints"],
        problem.durations,
        condition,
        f"{condition}_fan{fan}_{stamp}",
    )
    rows = np.asarray(record["coefficients"])
    exact = problem.exact_score(condition, problem.rows_mean(rows))
    threshold = (
        cfg.planner["alpha"]
        if condition == "pdstl"
        else cfg.mean_model["min_robustness"]
    )
    reference = torch.cat([problem.start, problem.init]).numpy()
    record.update(
        fan=fan,
        model_id=problem.model.report_id,
        model_hash=problem.model.report_hash,
        prediction=prediction(problem, rows),
        exact_score=exact,
        threshold=threshold,
        threshold_met=exact >= threshold,
        objective=objective,
        waypoint_shift_max=float(
            np.linalg.norm(best["waypoints"] - reference, axis=1).max()
        ),
        config_hash=cfg.plan_hash,
        reference_hash=file_hash(cfg.reference),
        created=stamp,
        git_commit=git_commit(),
    )
    return check_plan(cfg, task, record)
