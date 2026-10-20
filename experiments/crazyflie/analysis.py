"""Mission outcomes, summary tables and figures from recorded flights.

Empirical whole-flight success (with Clopper-Pearson intervals) is
reported next to, never merged with, each plan's model-based score.
Dry runs (simulated data) never enter the summary.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import minsnap
import numpy as np
from config import HERE, fan_setting
from flight import MONITOR_S
from logger import aligned_states, state_times
from matplotlib.patches import Ellipse, Rectangle
from scipy.stats import beta
from tracking_model import COVERAGE, squared_distance
from trajectory import PLANS, condition_name, file_hash

RESULTS = HERE / "results"
METRICS = ["min_clearance", "endpoint_error", "path_length", "duration"]
ALIGNMENT_FLAG_M = 0.01  # max ctrltarget-vs-r residual worth investigating
D2_95 = 5.991  # a 2-D Gaussian ellipse with d^2 <= 5.991 holds 95%


def clopper_pearson(k, n):
    lower = beta.ppf(0.025, k, n - k + 1) if k else 0.0
    upper = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lower), float(upper)


def mission_window(meta):
    """Host times (start, end) of the flown curve, ending before landing."""
    start = meta.get("t_play", meta["mission_t0"])
    horizon = start + meta["duration"]
    return start, min(horizon, meta.get("mission_t_end", horizon))


def mission_samples(trial):
    """Mission positions only; the endpoint exists only after completion."""
    meta, s = trial["meta"], trial["state"]
    t = state_times(s)
    start, end = mission_window(meta)
    horizon = start + meta["duration"]
    keep = (t >= start) & (t <= end)
    xyz = np.stack([s["x"][keep], s["y"][keep], s["z"][keep]], 1)
    final = None
    completed = meta.get("mission_outcome", meta["outcome"]) == "completed"
    if completed and t[0] <= horizon <= t[-1] and end >= horizon:
        final = np.array(
            [np.interp(horizon, t, s["x"]), np.interp(horizon, t, s["y"])]
        )
    return xyz, final


def evaluate(trial, task):
    """Requirements and metrics of one attempt, or why it is excluded.

    Completed flights and aborts for a task violation count; data faults,
    errors, interruptions and preflight aborts are excluded with a reason.
    """
    meta = trial["meta"]
    if meta.get("purpose") in ("identification", "evaluation") and meta.get(
        "data_readiness"
    ):
        return {"excluded": meta["data_readiness"]}
    outcome = meta.get("mission_outcome", meta["outcome"])
    counted = (
        outcome == "completed" or meta.get("abort_kind") == "TaskViolation"
    )
    if not counted or "mission_t0" not in meta:
        return {"excluded": meta.get("abort_reason") or meta["outcome"]}
    xyz, final = mission_samples(trial)
    if not len(xyz):
        return {"excluded": "no mission state samples"}
    met = task.requirements(
        xyz[:, :2], final if final is not None else [np.inf] * 2
    )
    # Match the flight monitor: trailing host-time means at its checks.
    s, window = trial["state"], meta["altitude_average_s"]
    t0 = meta["mission_t0"]
    t_end = meta.get("mission_t_end", t0)
    altitude = np.array(
        [
            s["z"][
                (s["t_host"] >= tick - window) & (s["t_host"] <= tick)
            ].mean()
            for tick in np.arange(t0, t_end, MONITOR_S)
        ]
    )
    met["altitude"] = bool(
        len(altitude)
        and np.all(
            np.abs(altitude - meta["altitude"]) <= meta["altitude_tolerance"]
        )
    )
    return {
        "success": outcome == "completed" and all(met.values()),
        "requirements": met,
        "min_clearance": float(task.clearance(xyz[:, :2]).min()),
        "endpoint_error": float(np.linalg.norm(final - task.goal.centre))
        if final is not None
        else np.nan,
        "path_length": float(
            np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1).sum()
        ),
        "duration": float(t_end - t0),
    }


def _plan(meta):
    path = Path(meta.get("plan_path", PLANS / f"{meta['plan_id']}.json"))
    if not path.exists():
        # Recorded files can be moved to another machine.
        path = PLANS / f"{meta['plan_id']}.json"
    if not path.exists() or (
        meta.get("plan_hash") and file_hash(path) != meta["plan_hash"]
    ):
        return None
    return json.loads(path.read_text())


def coverage(trials, plan):
    """Held-out d^2 coverage of evaluation flights under the plan's prediction.

    Tests whether the model transfers to the path actually flown; None for
    a plan without a prediction.
    """
    if plan is None or plan.get("prediction") is None:
        return None
    mean = np.asarray(plan["prediction"]["mean"])
    cov = np.asarray(plan["prediction"]["cov"])
    d2 = []
    for trial in trials:
        p, _, _ = aligned_states(trial)
        d2.append(squared_distance(p - mean[: len(p)], cov[: len(p)]))
    if not d2:
        return None
    d2 = np.concatenate(d2)
    return {name: float(np.mean(d2 <= th)) for name, th in COVERAGE.items()}


def summarize(trials, task):
    """Evaluation flights per condition x fan -> table rows; writes results/."""
    groups = defaultdict(list)
    for trial in trials:
        meta = trial["meta"]
        if meta["purpose"] == "evaluation" and not meta.get("dry_run"):
            groups[
                (condition_name(meta["condition"]), fan_setting(meta["fan"]))
            ].append(trial)
    rows = []
    for (condition, fan), group in sorted(groups.items()):
        results = [evaluate(t, task) for t in group]
        counted = [r for r in results if "excluded" not in r]
        k, n = sum(r["success"] for r in counted), len(counted)
        failures = defaultdict(int)
        for r in counted:
            for name, ok in r["requirements"].items():
                failures[name] += not ok
        plan = _plan(group[0]["meta"]) or {}
        flown = [t for t, r in zip(group, results) if "excluded" not in r]
        row = {
            "condition": condition,
            "fan": fan,
            "flights": n,
            "successes": k,
            "success_rate": k / n if n else np.nan,
            "ci95": clopper_pearson(k, n) if n else (np.nan, np.nan),
            "failures": dict(failures),
            "excluded": [r["excluded"] for r in results if "excluded" in r],
            "planned_exact_score": plan.get("exact_score"),
            "planned_threshold_met": plan.get("threshold_met"),
            "held_out_coverage": coverage(flown, plan),
        }
        for name in METRICS:
            values = [r[name] for r in counted]
            row[name] = (
                (float(np.nanmean(values)), float(np.nanstd(values)))
                if values
                else None
            )
        rows.append(row)
    _write_tables(rows)
    return rows


def _write_tables(rows):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0]) if rows else ["condition"]
        )
        writer.writeheader()
        writer.writerows(rows)
    header = (
        "| condition | fan | success | 95% CI | min clearance [m] "
        "| endpoint error [m] | planned exact score (model) "
        "| held-out d2<=4 coverage (model, nominal 86.5%) | excluded |"
    )
    lines = [header, "|" + "---|" * 9]
    for r in rows:

        def spread(name, r=r):
            if r[name] is None:
                return "-"
            return f"{r[name][0]:.3f} ± {r[name][1]:.3f}"

        score = r["planned_exact_score"]
        score = "-" if score is None else f"{score:.3f}"
        cover = r["held_out_coverage"]
        cover = "-" if cover is None else f"{cover['d2<=4 (86.5%)']:.2f}"
        lines.append(
            f"| {r['condition']} | {r['fan']} "
            f"| {r['successes']}/{r['flights']} "
            f"| [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}] "
            f"| {spread('min_clearance')} | {spread('endpoint_error')} "
            f"| {score} | {cover} | {len(r['excluded'])} |"
        )
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")


# Figures


def draw_task(ax, task, cfg=None):
    def box(b, **kw):
        ax.add_patch(
            Rectangle((b.x[0], b.y[0]), b.x[1] - b.x[0], b.y[1] - b.y[0], **kw)
        )

    box(task.workspace, fill=False, ls="--", ec="0.5")
    for b in task.obstacles:
        box(b, fc="0.85", ec="0.4")
    if cfg is not None:
        for b in cfg.obstacles:
            box(b, fc="0.6", ec="none")
    box(task.goal, fc="tab:green", alpha=0.25, ec="tab:green")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def curve_xy(rows):
    rows = np.asarray(rows, dtype=float)
    return minsnap.evaluate(rows, minsnap.dense_times(rows))[:, :2]


def plot_plan(plan, task, cfg, path):
    """The uploaded curve and, when available, the predicted belief."""
    fig, ax = plt.subplots(figsize=(5, 6))
    draw_task(ax, task, cfg)
    curve = curve_xy(plan["coefficients"])
    ax.plot(curve[:, 0], curve[:, 1], "k--", lw=1, label="curve (uploaded)")
    w = np.asarray(plan["waypoints"])
    ax.plot(w[:, 0], w[:, 1], "k.", ms=4, label="waypoints")
    if plan["prediction"]:
        mean = np.array(plan["prediction"]["mean"])
        cov = np.array(plan["prediction"]["cov"])
        ax.plot(mean[:, 0], mean[:, 1], "tab:blue", label="predicted mean")
        every = round(1.0 / plan["dt"])  # one ellipse per second
        for k in range(0, len(mean), every):
            values, vectors = np.linalg.eigh(cov[k])
            angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
            width, height = 2 * np.sqrt(D2_95 * values[::-1])
            ax.add_patch(
                Ellipse(
                    mean[k], width, height, angle=angle,
                    fill=False, ec="tab:blue", lw=0.6,
                    label="95% region (d² = 5.991)" if k == 0 else None,
                )
            )  # fmt: skip
    ax.set_title(plan["id"], fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def tracking_errors(trial):
    """Time-aligned and cross-track errors [m] of the mission samples."""
    meta, s = trial["meta"], trial["state"]
    rows = np.asarray(meta["coefficients"], dtype=float)
    t = state_times(s)
    start, end = mission_window(meta)
    keep = (t >= start) & (t <= end)
    xy = np.stack([s["x"][keep], s["y"][keep]], 1)
    aligned = np.linalg.norm(
        xy - minsnap.evaluate(rows, t[keep] - start)[:, :2], axis=1
    )
    curve = curve_xy(rows)
    cross = np.linalg.norm(xy[:, None] - curve[None], axis=2).min(axis=1)
    return xy, aligned, cross


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def check_trial(trial, task, cfg):
    """Inspect one attempt: alignment, tracking, state data and outcome."""
    meta, s = trial["meta"], trial["state"]
    line = f"{meta['trial_id']}: {meta['outcome']}"
    if meta.get("abort_reason"):
        line += (
            f" ({meta['abort_kind']} in {meta['abort_phase']}: "
            f"{meta['abort_reason']})"
        )
    print(line)
    if meta.get("landing"):
        print(f"  landing: {meta['landing']}")
        if meta.get("landing_error"):
            print(f"  landing error: {meta['landing_error']}")
    if s is None or not len(s["t_host"]) or "mission_t0" not in meta:
        print("  no mission data")
        return None
    if meta.get("pieces"):
        print(
            f"  upload: {meta['pieces']} pieces, {meta['upload_bytes']} B, "
            f"{1e3 * meta['upload_s']:.0f} ms"
        )
    offset = 1e3 * meta["t_play_offset"]
    print(
        f"  alignment: {meta['alignment']}, t_play - t_cmd_after "
        f"= {offset:.1f} ms, identifiable: {meta['identifiable']}"
    )
    if meta.get("alignment_max") is not None:
        print(
            f"  ctrltarget vs r(t): rms {1e3 * meta['alignment_rms']:.2f} mm, "
            f"max {1e3 * meta['alignment_max']:.2f} mm"
        )
        if meta["alignment_max"] > ALIGNMENT_FLAG_M:
            print(
                "  FLAG: reference-alignment discrepancy above 1 cm; "
                "investigate upload, timing, sampling or alignment"
            )
    gaps = np.diff(state_times(s))
    period = 1.0 / meta["log_hz"]
    missing = int(np.sum(np.round(gaps / period) - 1))
    print(
        f"  state: {len(gaps) + 1} packets, interval median "
        f"{1e3 * np.median(gaps):.0f} ms, max gap {1e3 * gaps.max():.0f} ms, "
        f"missing ≈ {missing}"
    )
    print(
        f"  start state [x y z vx vy yaw]: {np.round(meta['start_state'], 3).tolist()}"
    )
    print(
        f"  battery {meta.get('battery_start')} -> {meta.get('battery_end')} V"
    )
    xy, aligned, cross = tracking_errors(trial)
    if len(xy):
        print(
            f"  time-aligned tracking error: rms {1e3 * rms(aligned):.1f} mm, "
            f"max {1e3 * aligned.max():.1f} mm"
        )
        print(
            f"  cross-track error: rms {1e3 * rms(cross):.1f} mm, "
            f"max {1e3 * cross.max():.1f} mm"
        )
    result = evaluate(trial, task)
    if "excluded" not in result:
        print(
            f"  requirements {result['requirements']}, "
            f"min clearance {result['min_clearance']:.3f} m"
        )

    t = state_times(s) - mission_window(meta)[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    draw_task(axes[0], task, cfg)
    curve = curve_xy(meta["coefficients"])
    axes[0].plot(curve[:, 0], curve[:, 1], "k--", lw=1, label="curve")
    axes[0].plot(xy[:, 0], xy[:, 1], label="flown (mission)")
    axes[0].legend()
    ax = axes[1]
    ax.plot(t, s["z"], label="measured z")
    ax.axhspan(
        meta["altitude"] - meta["altitude_tolerance"],
        meta["altitude"] + meta["altitude_tolerance"],
        color="tab:green",
        alpha=0.15,
        label="allowed band",
    )
    ax.axhline(
        meta["altitude"], c="tab:green", ls="--", lw=1, label="setpoint"
    )
    ax.axvline(meta["duration"], c="0.5", ls=":", label="curve end")
    ax.set_xlim(-1, t.max())
    ax.set_xlabel("time from playback start [s]")
    ax.set_ylabel("z [m]")
    ax.legend()
    fig.tight_layout()
    path = trial["dir"] / "flight.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
