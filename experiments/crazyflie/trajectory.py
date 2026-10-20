"""Plan records: a min-snap curve through xy waypoints, stored as uploaded.

A plan stores its waypoints [M+1, 2], durations [M] and the quantized
(float32) coefficient rows [M, 33] the drone receives. Every check and
score describes those rows.
"""

import hashlib
import json
from pathlib import Path

import minsnap
import numpy as np
import torch
from config import HERE, fan_setting

PLANS = HERE / "plans"


def condition_name(name):
    return "reference" if name == "deterministic" else name


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def make_curve(waypoints, durations, altitude):
    """Quantized [M, 33] rows of the min-snap curve through the waypoints."""
    durations = minsnap.quantize(durations)
    K = minsnap.solve_matrix(durations.tolist())
    xy = K @ torch.as_tensor(np.asarray(waypoints, dtype=float))
    return minsnap.quantize(minsnap.to_rows(durations, xy, altitude))


def new_plan(cfg, waypoints, durations, condition, name):
    """An unchecked plan record for the curve through `waypoints`."""
    waypoints = np.asarray(waypoints, dtype=float)
    rows = make_curve(waypoints, durations, cfg.altitude)
    return {
        "id": name,
        "condition": condition_name(condition),
        "name": name,
        "fan": None,
        "dt": cfg.dt,
        "altitude": cfg.altitude,
        "start": waypoints[0].tolist(),
        "waypoints": waypoints.tolist(),
        "durations": rows[:, 0].tolist(),
        "coefficients": rows.tolist(),
        "prediction": None,
        "exact_score": None,
        "threshold_met": None,
        "model_id": None,
        "model_hash": None,
    }


def check_plan(cfg, task, plan):
    """Add the sampled checks, min clearance and execution hash."""
    from geometry import check_feasible

    mean = None
    if plan["prediction"] is not None:
        mean = np.asarray(plan["prediction"]["mean"])
    rows = np.asarray(plan["coefficients"])
    xy = minsnap.evaluate(rows, minsnap.dense_times(rows))[:, :2]
    plan["problems"] = check_feasible(cfg, task, plan, mean)
    plan["feasible"] = not plan["problems"]
    plan["min_clearance"] = float(task.clearance(xy).min())
    plan["execution_hash"] = cfg.execution_hash
    return plan


def read_plan(path):
    """Read a plan and check its shapes, values and waypoints."""
    path = Path(path)
    plan = json.loads(path.read_text())
    w = np.asarray(plan["waypoints"], dtype=float)
    d = np.asarray(plan["durations"], dtype=float)
    rows = np.asarray(plan["coefficients"], dtype=float)
    M = len(d)
    if M < 1 or w.shape != (M + 1, 2) or rows.shape != (M, 33):
        raise ValueError(
            "plan needs waypoints [M+1, 2], durations [M], coefficients [M, 33]"
        )
    finite = all(np.isfinite(x).all() for x in (w, d, rows))
    if not (finite and plan["dt"] > 0 and plan["altitude"] > 0):
        raise ValueError(
            "plan values must be finite; dt and altitude positive"
        )
    if not np.array_equal(rows[:, 0], d):
        raise ValueError("coefficient rows and durations differ")
    steps = d.sum() / plan["dt"]
    if abs(steps - round(steps)) > 1e-6:
        raise ValueError("sum(durations) / dt must be a whole number")
    if not np.allclose(plan["start"], w[0]):
        raise ValueError("start must be the first waypoint")
    miss = minsnap.waypoint_error(rows, w)
    if miss > minsnap.WAYPOINT_TOL:
        raise ValueError(f"coefficients miss the waypoints by {miss:.2e} m")
    plan.setdefault("id", path.stem)
    plan["condition"] = condition_name(plan.get("condition", "reference"))
    plan.setdefault("name", plan["condition"])
    for key in ("fan", "model_id", "model_hash", "prediction"):
        plan.setdefault(key, None)
    if plan["fan"] is not None:
        plan["fan"] = fan_setting(plan["fan"])
    return plan


def load_plan(path, cfg, allow_below_threshold=False):
    """A plan that may be flown in this arena with these limits."""
    plan = read_plan(path)
    if plan.get("execution_hash") != cfg.for_plan(plan).execution_hash:
        raise ValueError(
            "plan was checked for another arena or limits; import or plan it again"
        )
    if plan.get("feasible") is not True:
        raise ValueError(f"plan is not feasible: {plan.get('problems')}")
    if plan.get("threshold_met") is False:
        print(
            f"Warning: plan score {plan['exact_score']:.3f} "
            "is below its threshold."
        )
        if not allow_below_threshold:
            raise ValueError(
                "plan misses its score threshold; "
                "pass --allow-below-threshold to fly it"
            )
    return plan


def reference_plan(cfg):
    return load_plan(cfg.reference, cfg)


def save_plan(plan, output=None):
    path = Path(output) if output else PLANS / f"{plan['id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    plan["id"] = path.stem
    path.write_text(json.dumps(plan, indent=1))
    return path


def import_plan(path, durations, cfg, output):
    """Fit the curve through supplied x,y rows, check it once, save it."""
    from geometry import Task

    output = Path(output)
    if output.exists():
        raise ValueError(f"{output} exists; saved plans are never replaced")
    source = Path(path).resolve()
    waypoints = np.loadtxt(source, delimiter=",", ndmin=2)
    if len(durations) == 1:
        durations = durations * (len(waypoints) - 1)
    if waypoints.shape[1] != 2 or len(durations) != len(waypoints) - 1:
        raise ValueError("need x,y rows and one duration per segment")
    plan = new_plan(cfg, waypoints, durations, "reference", output.stem)
    flight_cfg = cfg.for_plan(plan)
    check_plan(flight_cfg, Task(flight_cfg), plan)
    if not plan["feasible"]:
        raise ValueError(f"curve is infeasible: {plan['problems']}")
    plan["source"] = str(source)
    plan["source_hash"] = file_hash(source)
    plan["method"] = "min-snap, degree 7, through the supplied waypoints"
    return save_plan(plan, output)
