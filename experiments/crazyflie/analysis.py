"""Mission outcomes, summary tables and figures from recorded flights.

Empirical whole-flight success (with Clopper-Pearson intervals) is
reported next to, never merged with, each plan's model-based score.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from config import HERE
from logger import state_times
from matplotlib.patches import Ellipse, Rectangle
from planner import PLANS, nominal_path
from scipy.stats import beta

RESULTS = HERE / "results"
METRICS = [
    "min_clearance",
    "endpoint_error",
    "path_length",
    "effort",
    "duration",
]


def clopper_pearson(k, n):
    lower = beta.ppf(0.025, k, n - k + 1) if k else 0.0
    upper = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lower), float(upper)


def mission_samples(trial):
    """Positions [M, 3] from mission start to step H, and the step-H XY (or None)."""
    meta, s = trial["meta"], trial["state"]
    t = state_times(s)
    t0 = meta["mission_t0"]
    t_end = t0 + meta["steps"] * meta["dt"]
    keep = (t >= t0) & (t <= t_end)
    xyz = np.stack([s["x"][keep], s["y"][keep], s["z"][keep]], 1)
    final = None
    if t[-1] >= t_end:
        final = np.array(
            [np.interp(t_end, t, s["x"]), np.interp(t_end, t, s["y"])]
        )
    return xyz, final


def evaluate(trial, task):
    """Requirements and metrics of one attempt, or why it is excluded.

    Completed flights and aborts for a task violation count; data faults,
    errors, interruptions and preflight aborts are excluded with a reason.
    """
    meta = trial["meta"]
    counted = (
        meta["outcome"] == "completed"
        or meta.get("abort_kind") == "TaskViolation"
    )
    if not counted or "mission_t0" not in meta:
        return {"excluded": meta.get("abort_reason") or meta["outcome"]}
    xyz, final = mission_samples(trial)
    met = task.requirements(
        xyz[:, :2], final if final is not None else [np.inf] * 2
    )
    met["altitude"] = bool(
        np.all(
            np.abs(xyz[:, 2] - meta["altitude"]) <= meta["altitude_tolerance"]
        )
    )
    cmd = trial["commands"]
    mission = cmd["step"] >= 0
    period = 1.0 / meta["stream_hz"]
    return {
        "success": meta["outcome"] == "completed" and all(met.values()),
        "requirements": met,
        "min_clearance": float(task.clearance(xyz[:, :2]).min()),
        "endpoint_error": float(np.linalg.norm(final - task.goal.centre))
        if final is not None
        else np.nan,
        "path_length": float(
            np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1).sum()
        ),
        "effort": float(
            np.sum(cmd["vx"][mission] ** 2 + cmd["vy"][mission] ** 2) * period
        ),
        "duration": float(
            cmd["t_sent"][mission][-1] - meta["mission_t0"] + period
        ),
    }


def _plan(plan_id):
    path = PLANS / f"{plan_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def summarize(trials, task):
    """Evaluation flights per condition x fan -> table rows; writes results/."""
    groups = defaultdict(list)
    for trial in trials:
        meta = trial["meta"]
        if meta["purpose"] == "evaluation" and not meta.get("dry_run"):
            groups[(meta["condition"], meta["fan"])].append(trial)
    rows = []
    for (condition, fan), group in sorted(groups.items()):
        results = [evaluate(t, task) for t in group]
        counted = [r for r in results if "excluded" not in r]
        k, n = sum(r["success"] for r in counted), len(counted)
        failures = defaultdict(int)
        for r in counted:
            for name, ok in r["requirements"].items():
                failures[name] += not ok
        plan = _plan(group[0]["meta"]["plan_id"])
        score = (plan or {}).get("score") or {}
        row = {
            "condition": condition,
            "fan": fan,
            "flights": n,
            "successes": k,
            "success_rate": k / n if n else np.nan,
            "ci95": clopper_pearson(k, n) if n else (np.nan, np.nan),
            "failures": dict(failures),
            "excluded": [r["excluded"] for r in results if "excluded" in r],
            "planned_pdstl_lower": score.get("pdstl_lower"),
            "planned_mean_robustness": score.get("robustness"),
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
    lines = [
        (
            "| condition | fan | success | 95% CI | min clearance [m] "
            "| endpoint error [m] | planned pdSTL lower (model) | excluded |"
        ),
        "|" + "---|" * 8,
    ]
    for r in rows:

        def spread(name, r=r):
            return (
                "-"
                if r[name] is None
                else f"{r[name][0]:.3f} ± {r[name][1]:.3f}"
            )

        planned = r["planned_pdstl_lower"]
        lines.append(
            f"| {r['condition']} | {r['fan']} | {r['successes']}/{r['flights']} "
            f"| [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}] "
            f"| {spread('min_clearance')} | {spread('endpoint_error')} "
            f"| {'-' if planned is None else f'{planned:.3f}'} | {len(r['excluded'])} |"
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


def _ellipses(ax, mean, cov, every=5):
    """2-sigma position ellipses of a predicted belief."""
    for m, c in zip(mean[::every], cov[::every]):
        w, v = np.linalg.eigh(c[:2, :2])
        ax.add_patch(
            Ellipse(
                m[:2],
                *(4 * np.sqrt(np.maximum(w, 0))),
                angle=np.degrees(np.arctan2(v[1, 1], v[0, 1])),
                fill=False,
                ec="tab:blue",
                lw=0.6,
            )
        )


def plot_plan(plan, task, cfg, path):
    """The commanded path, and the predicted mean +- 2 sigma if any."""
    fig, ax = plt.subplots(figsize=(5, 6))
    draw_task(ax, task, cfg)
    nominal = nominal_path(cfg, plan["start"], plan["velocities"])
    ax.plot(
        nominal[:, 0],
        nominal[:, 1],
        "k--",
        lw=1,
        label="commanded (integrated)",
    )
    if plan["prediction"]:
        mean = np.array(plan["prediction"]["mean"])
        ax.plot(mean[:, 0], mean[:, 1], "tab:blue", label="predicted mean")
        _ellipses(ax, mean, np.array(plan["prediction"]["cov"]))
    ax.set_title(plan["id"], fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_paths(trials, task, cfg):
    """Flown paths per condition x fan over the planned mean +- 2 sigma."""
    groups = defaultdict(list)
    for t in trials:
        if (
            t["meta"]["purpose"] == "evaluation"
            and not t["meta"].get("dry_run")
            and "mission_t0" in t["meta"]
        ):
            groups[(t["meta"]["condition"], t["meta"]["fan"])].append(t)
    if not groups:
        return None
    fig, axes = plt.subplots(
        1, len(groups), figsize=(5 * len(groups), 6), squeeze=False
    )
    for ax, ((condition, fan), group) in zip(axes[0], sorted(groups.items())):
        draw_task(ax, task, cfg)
        plan = _plan(group[0]["meta"]["plan_id"])
        if plan and plan["prediction"]:
            mean, cov = (
                np.array(plan["prediction"]["mean"]),
                np.array(plan["prediction"]["cov"]),
            )
            ax.plot(mean[:, 0], mean[:, 1], "k--", lw=1, label="planned mean")
            _ellipses(ax, mean, cov)
        for t in group:
            xyz, _ = mission_samples(t)
            ax.plot(xyz[:, 0], xyz[:, 1], lw=0.8, alpha=0.7)
        ax.set_title(f"{condition}, fan {fan} (n={len(group)})")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "paths.png", dpi=150)
    plt.close(fig)
    return RESULTS / "paths.png"


def check_trial(trial, task, cfg):
    """Milestone-1 check of one attempt: timing printout and a figure."""
    meta, cmd, s = trial["meta"], trial["commands"], trial["state"]
    print(
        f"{meta['trial_id']}: {meta['outcome']}"
        + (
            f" ({meta['abort_kind']} in {meta['abort_phase']}: {meta['abort_reason']})"
            if meta.get("abort_reason")
            else ""
        )
    )
    if s is None or not len(s["t_host"]) or "mission_t0" not in meta:
        print("  no mission data")
        return None
    period = 1.0 / meta["stream_hz"]
    gaps = np.diff(np.unique(s["device_ms"])) / 1000.0
    print(
        f"  commands: {len(cmd['k'])} sent, lateness mean {1e3 * cmd['lateness'].mean():.1f} ms, "
        f"max {1e3 * cmd['lateness'].max():.1f} ms"
    )
    print(
        f"  state: {len(gaps) + 1} packets, interval median {1e3 * np.median(gaps):.0f} ms, "
        f"max gap {1e3 * gaps.max():.0f} ms, missing ≈ {int(np.sum(np.round(gaps / period) - 1))}"
    )
    print(
        f"  start state [x y z vx vy yaw]: {np.round(meta['start_state'], 3).tolist()}"
    )
    print(
        f"  battery {meta.get('battery_start')} -> {meta.get('battery_end')} V"
    )
    if (
        meta["outcome"] == "completed"
        or meta.get("abort_kind") == "TaskViolation"
    ):
        result = evaluate(trial, task)
        print(
            f"  requirements {result['requirements']}, min clearance {result['min_clearance']:.3f} m"
        )

    t = state_times(s) - meta["mission_t0"]
    tc = cmd["t_sent"] - meta["mission_t0"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    draw_task(axes[0], task, cfg)
    plan = _plan(meta["plan_id"])
    if plan:
        nominal = nominal_path(cfg, plan["start"], plan["velocities"])
        axes[0].plot(
            nominal[:, 0],
            nominal[:, 1],
            "k--",
            lw=1,
            label="commanded (integrated)",
        )
    axes[0].plot(s["x"][t >= 0], s["y"][t >= 0], label="flown")
    axes[0].legend()
    for ax, axis in zip(axes[1:], ("x", "y")):
        ax.step(tc, cmd[f"v{axis}"], where="post", label=f"commanded v{axis}")
        ax.plot(t, s[f"v{axis}"], label=f"measured v{axis}")
        ax.set_xlim(-1, tc.max() + 1)
        ax.set_xlabel("time from mission start [s]")
        ax.set_ylabel("m/s")
        ax.legend()
    fig.tight_layout()
    path = trial["dir"] / "flight.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_validation(report, path):
    """Held-out error vs predicted spread over the horizon, and coverage."""
    fans = list(report["validation"])
    if not fans:
        return None
    fig, axes = plt.subplots(
        2, len(fans), figsize=(6 * len(fans), 7), squeeze=False
    )
    for j, fan in enumerate(fans):
        v = report["validation"][fan]
        t = report["dt"] * np.arange(len(v["error_by_step"]))
        axes[0, j].plot(t, v["error_by_step"], label="mean position error")
        axes[0, j].plot(t, v["spread_by_step"], label="predicted RMS spread")
        axes[0, j].set_xlabel("time [s]")
        axes[0, j].set_ylabel("m")
        axes[0, j].set_title(f"fan {fan}, {v['flights']} held-out flights")
        axes[0, j].legend()
        names = list(v["coverage"])
        est = np.array([v["coverage"][n] for n in names])
        axes[1, j].bar(
            names,
            est[:, 0],
            yerr=[est[:, 0] - est[:, 1], est[:, 2] - est[:, 0]],
        )
        axes[1, j].scatter(
            names, [0.393, 0.865, 0.95], color="k", zorder=3, label="nominal"
        )
        axes[1, j].set_ylim(0, 1)
        axes[1, j].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
