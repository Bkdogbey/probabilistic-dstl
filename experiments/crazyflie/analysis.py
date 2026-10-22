"""Mission outcomes, the success table and figures from recorded flights.

Empirical success (with Clopper-Pearson intervals) is reported next to,
never merged with, each plan's model-based score. Dry runs are excluded.
"""

from __future__ import annotations

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
from logger import logged_time, playback_time, state_times
from matplotlib.patches import Ellipse, Rectangle
from scipy.stats import beta
from trajectory import PLANS, condition_name, file_hash

RESULTS = HERE / "results"
ALIGNMENT_FLAG_M = 0.01  # max ctrltarget-vs-r residual worth investigating
D2_95 = 5.991  # a 2-D Gaussian ellipse with d^2 <= 5.991 holds 95%


def clopper_pearson(k, n):
    lower = beta.ppf(0.025, k, n - k + 1) if k else 0.0
    upper = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lower), float(upper)


def mission_window(meta):
    """Host times (start, end) of the flown curve, ending before landing."""
    meta = {"t_play": meta["mission_t0"], **meta}
    start, horizon = logged_time(meta, [0.0, meta["duration"]])
    return start, min(horizon, meta.get("mission_t_end", horizon))


def mission_samples(trial):
    """Positions [N, 3] measured while the curve was flown."""
    s = trial["state"]
    t = state_times(s)
    start, end = mission_window(trial["meta"])
    keep = (t >= start) & (t <= end)
    return np.stack([s["x"][keep], s["y"][keep], s["z"][keep]], 1)


def evaluate(trial, task):
    """Requirements of one attempt, or why it is excluded.

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
    xyz = mission_samples(trial)
    if not len(xyz):
        return {"excluded": "no mission state samples"}
    met = task.requirements(xyz[:, :2])
    # Match the flight monitor: trailing host-time means at its checks.
    s, window = trial["state"], meta["altitude_average_s"]
    t0 = meta["mission_t0"]
    ticks = np.arange(t0, meta.get("mission_t_end", t0), MONITOR_S)
    # A tick with no packet in its window is skipped, as in flight.
    windows = [(s["t_host"] >= t - window) & (s["t_host"] <= t) for t in ticks]
    altitude = np.array([s["z"][w].mean() for w in windows if w.any()])
    error = np.abs(altitude - meta["altitude"])
    met["altitude"] = bool(
        len(altitude) and np.all(error <= meta["altitude_tolerance"])
    )
    return {
        "success": outcome == "completed" and all(met.values()),
        "requirements": met,
        "min_clearance": float(task.clearance(xyz[:, :2]).min()),
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


def summarize(trials, task):
    """Evaluation flights per condition x fan -> results/summary.md."""
    groups = defaultdict(list)
    for trial in trials:
        meta = trial["meta"]
        if meta["purpose"] == "evaluation" and not meta.get("dry_run"):
            key = (condition_name(meta["condition"]), fan_setting(meta["fan"]))
            groups[key].append(trial)
    header = (
        "| condition | fan | success | 95% CI | failed requirements "
        "| planned exact score (model) | excluded |"
    )
    lines = [header, "|" + "---|" * 7]
    for (condition, fan), group in sorted(groups.items()):
        results = [evaluate(t, task) for t in group]
        counted = [r for r in results if "excluded" not in r]
        k, n = sum(r["success"] for r in counted), len(counted)
        low, high = clopper_pearson(k, n) if n else (np.nan, np.nan)
        failures = defaultdict(int)
        for r in counted:
            for name, ok in r["requirements"].items():
                failures[name] += not ok
        failed = ", ".join(f"{name} {c}" for name, c in failures.items() if c)
        score = (_plan(group[0]["meta"]) or {}).get("exact_score")
        score = "-" if score is None else f"{score:.3f}"
        lines.append(
            f"| {condition} | {fan} | {k}/{n} | [{low:.2f}, {high:.2f}] "
            f"| {failed or '-'} | {score} | {len(results) - n} |"
        )
    RESULTS.mkdir(exist_ok=True)
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
            label = "95% region (d² = 5.991)" if k == 0 else None
            ellipse = Ellipse(mean[k], width, height, angle=angle)
            ellipse.set(fill=False, ec="tab:blue", lw=0.6, label=label)
            ax.add_patch(ellipse)
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
    reference = minsnap.evaluate(rows, playback_time(meta, t[keep]))[:, :2]
    aligned = np.linalg.norm(xy - reference, axis=1)
    curve = curve_xy(rows)
    cross = np.linalg.norm(xy[:, None] - curve[None], axis=2).min(axis=1)
    return xy, aligned, cross


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def check_trial(trial, task, cfg):
    """Inspect one attempt: alignment, tracking, state data and outcome."""
    meta, s = trial["meta"], trial["state"]
    print(f"{meta['trial_id']}: {meta['outcome']}")
    if meta.get("abort_reason"):
        print(
            f"  {meta['abort_kind']} in {meta['abort_phase']}: "
            f"{meta['abort_reason']}"
        )
    if meta.get("landing"):
        print(f"  landing: {meta['landing']}")
    if s is None or not len(s["t_host"]) or "mission_t0" not in meta:
        print("  no mission data")
        return None
    print(
        f"  alignment: {meta['alignment']}, t_play - t_cmd_after "
        f"= {1e3 * meta['t_play_offset']:.1f} ms, "
        f"rate {meta.get('t_play_rate', 1.0):.4f}, "
        f"identifiable: {meta['identifiable']}"
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
    print(
        f"  state: {len(gaps) + 1} packets, max gap {1e3 * gaps.max():.0f} ms"
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

    # Height from takeoff to landing only: the estimator reset on the ground
    # can spike by metres and would flatten the plot.
    t = state_times(s) - mission_window(meta)[0]
    shown = (t >= -2.0) & (t <= meta["duration"] + 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    draw_task(axes[0], task, cfg)
    curve = curve_xy(meta["coefficients"])
    axes[0].plot(curve[:, 0], curve[:, 1], "k--", lw=1, label="curve")
    axes[0].plot(xy[:, 0], xy[:, 1], label="flown (mission)")
    axes[0].legend()
    ax = axes[1]
    ax.plot(t[shown], s["z"][shown], label="measured z")
    tolerance = meta["altitude_tolerance"]
    ax.axhspan(
        meta["altitude"] - tolerance,
        meta["altitude"] + tolerance,
        color="tab:green",
        alpha=0.15,
        label="allowed band",
    )
    curve_end = meta["duration"] / meta.get("t_play_rate", 1.0)
    ax.axvline(curve_end, c="0.5", ls=":", label="curve end")
    ax.set_xlabel("time from playback start [s]")
    ax.set_ylabel("z [m]")
    ax.legend()
    fig.tight_layout()
    path = trial["dir"] / "flight.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
