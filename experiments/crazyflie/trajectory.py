"""Read and replay saved world-frame velocity trajectories without planning."""

import hashlib
import json
from pathlib import Path

import numpy as np
from config import HERE, fan_setting

PLANS = HERE / "plans"


def condition_name(name):
    return "reference" if name == "deterministic" else name


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def read_plan(path):
    """Read commands as supplied: no smoothing, resampling or clipping."""
    path = Path(path)
    plan = json.loads(path.read_text())
    v = np.asarray(plan["velocities"], dtype=float)
    start = np.asarray(plan["start"], dtype=float)
    if v.ndim != 2 or v.shape[1] != 2 or not len(v) or start.shape != (2,):
        raise ValueError("trajectory needs velocities [N, 2] and start [x, y]")
    if (
        not np.isfinite(v).all()
        or not np.isfinite(start).all()
        or not (0 < plan["dt"] < np.inf and 0 < plan["altitude"] < np.inf)
    ):
        raise ValueError(
            "trajectory commands, start, dt and altitude must be finite; dt/altitude positive"
        )
    plan.setdefault("id", path.stem)
    plan["condition"] = condition_name(plan.get("condition", "reference"))
    plan.setdefault("name", plan["condition"])
    for key in ("fan", "model_id", "model_hash", "prediction", "score"):
        plan.setdefault(key, None)
    if plan["fan"] is not None:
        plan["fan"] = fan_setting(plan["fan"])
    return plan


def load_plan(path, cfg):
    """Replay an imported/generated plan; optimizer tuning is irrelevant."""
    plan = read_plan(path)
    if plan.get("execution_hash") != cfg.for_plan(plan).execution_hash:
        raise ValueError(
            "trajectory needs offline import for this arena/command limits: run `import PATH`"
        )
    if plan.get("problems"):
        raise ValueError(f"trajectory is infeasible: {plan['problems']}")
    if plan["score"] is not None and not plan["score"]["accepted"]:
        raise ValueError("model-based plan missed its score threshold")
    return plan


def reference_velocities(cfg):
    return np.asarray(load_plan(cfg.reference, cfg)["velocities"], dtype=float)


def save_plan(plan, output=None):
    path = Path(output) if output else PLANS / f"{plan['id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    plan["id"] = path.stem
    path.write_text(json.dumps(plan, indent=1))
    return path


def import_plan(path, cfg, output=None):
    """One offline geometry/limit check, then freeze the original commands."""
    from geometry import Task, check_feasible, nominal_path

    source = Path(path).resolve()
    plan = read_plan(source)
    flight_cfg = cfg.for_plan(plan)
    task = Task(flight_cfg)
    mean = (
        None
        if plan["prediction"] is None
        else np.asarray(plan["prediction"]["mean"])[:, :2]
    )
    plan["problems"] = check_feasible(
        flight_cfg,
        task,
        plan["start"],
        plan["velocities"],
        mean,
        mission=False,
    )
    if plan["problems"]:
        raise ValueError(f"trajectory is infeasible: {plan['problems']}")
    plan["execution_hash"] = flight_cfg.execution_hash
    plan["source"] = str(source)
    plan["source_hash"] = file_hash(source)
    plan["min_clearance"] = float(
        task.clearance(
            nominal_path(flight_cfg, plan["start"], plan["velocities"])
        ).min()
    )
    return save_plan(plan, output or cfg.reference)
