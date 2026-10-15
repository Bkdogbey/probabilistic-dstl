"""Mission outcomes, summary tables and figures from recorded flights.

Empirical whole-flight success (with Clopper-Pearson intervals) is
reported next to, never merged with, each plan's model-based score.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from config import HERE, fan_setting
from geometry import nominal_path
from logger import state_times
from matplotlib.patches import Rectangle
from scipy.stats import beta
from trajectory import PLANS, condition_name, file_hash

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
    """Mission positions only; the step-H endpoint exists only after completion."""
    meta, s = trial["meta"], trial["state"]
    t = state_times(s)
    t0 = meta["mission_t0"]
    horizon = t0 + meta["steps"] * meta["dt"]
    completed = meta.get("mission_outcome", meta["outcome"]) == "completed"
    t_end = min(horizon, meta.get("mission_t_end", horizon))
    if not completed and "mission_t_end" not in meta:
        # Legacy logs did not mark the abort boundary; stop after the last
        # command's interval rather than including subsequent landing.
        sent = trial["commands"]["t_sent"]
        t_end = min(
            t_end, sent[-1] + 1 / meta["stream_hz"] if len(sent) else t0
        )
    keep = (t >= t0) & (t <= t_end)
    xyz = np.stack([s["x"][keep], s["y"][keep], s["z"][keep]], 1)
    final = None
    if completed and t[0] <= horizon <= t[-1] and t_end >= horizon:
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
    cmd = trial["commands"]
    mission = cmd["step"] >= 0
    period = 1.0 / meta["stream_hz"]
    sent = cmd["t_sent"][mission]
    # Match the flight monitor: trailing host-time means at command checks.
    # Older logs without the averaging field retain their original scoring.
    altitude = xyz[:, 2]
    if window := meta.get("altitude_average_s"):
        s = trial["state"]
        checks = sent
        altitude = np.array(
            [
                s["z"][
                    (s["t_host"] >= tick - window) & (s["t_host"] <= tick)
                ].mean()
                for tick in checks
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
        "effort": float(
            np.sum(cmd["vx"][mission] ** 2 + cmd["vy"][mission] ** 2) * period
        ),
        "duration": float(
            meta.get(
                "mission_t_end",
                sent[-1] + period if len(sent) else meta["mission_t0"],
            )
            - meta["mission_t0"]
        ),
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
        plan = _plan(group[0]["meta"])
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
            "returned": sum(
                t["meta"].get("recovery") == "returned" for t in group
            ),
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


def plot_plan(plan, task, cfg, path):
    """The integrated commands and fitted-model mean, when available."""
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
    ax.set_title(plan["id"], fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def check_trial(trial, task, cfg):
    """Inspect one attempt's timing, recorded state and mission outcome."""
    meta, cmd, s = trial["meta"], trial["commands"], trial["state"]
    print(
        f"{meta['trial_id']}: {meta['outcome']}"
        + (
            f" ({meta['abort_kind']} in {meta['abort_phase']}: {meta['abort_reason']})"
            if meta.get("abort_reason")
            else ""
        )
    )
    if meta.get("recovery"):
        error = meta.get("start_error")
        print(
            f"  recovery: {meta['recovery']}"
            + (f", {error:.3f} m from the start" if error is not None else "")
        )
    if meta.get("landing"):
        print(
            f"  landing: {meta['landing']}"
            + (
                f" ({meta['landing_error']})"
                if meta.get("landing_error")
                else ""
            )
        )
    if s is None or not len(s["t_host"]) or "mission_t0" not in meta:
        print("  no mission data")
        return None
    period = 1.0 / meta["stream_hz"]
    gaps = np.diff(state_times(s))
    if not len(cmd["k"]):
        print("  no mission commands sent")
        return None
    # Derive this from timestamps for old logs too: their `lateness` was
    # measured before checks/sending and could hide a blocked send.
    delay = cmd["t_sent"] - meta["mission_t0"] - cmd["t_ref"]
    print(
        f"  commands: {len(cmd['k'])} sent, send-return delay mean {1e3 * delay.mean():.1f} ms, "
        f"max {1e3 * delay.max():.1f} ms"
    )
    print(
        f"  state: {len(gaps) + 1} packets, interval median {1e3 * np.median(gaps):.0f} ms, "
        f"max gap {1e3 * gaps.max():.0f} ms, missing ≈ {int(np.sum(np.round(gaps / period) - 1))}"
    )
    print(
        f"  host receive max gap {1e3 * np.diff(s['t_host']).max():.0f} ms"
        if len(s["t_host"]) > 1
        else "  host receive: one packet"
    )
    print(
        f"  start state [x y z vx vy yaw]: {np.round(meta['start_state'], 3).tolist()}"
    )
    print(
        f"  battery {meta.get('battery_start')} -> {meta.get('battery_end')} V"
    )
    result = evaluate(trial, task)
    if "excluded" not in result:
        print(
            f"  requirements {result['requirements']}, min clearance {result['min_clearance']:.3f} m"
        )

    t = state_times(s) - meta["mission_t0"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    draw_task(axes[0], task, cfg)
    plan = _plan(meta)
    if plan:
        nominal = nominal_path(cfg, plan["start"], plan["velocities"])
        axes[0].plot(
            nominal[:, 0],
            nominal[:, 1],
            "k--",
            lw=1,
            label="commanded (integrated)",
        )
    mission_end = (
        meta.get(
            "mission_t_end", meta["mission_t0"] + meta["steps"] * meta["dt"]
        )
        - meta["mission_t0"]
    )
    during = (t >= 0) & (t <= mission_end)
    after = t > mission_end
    axes[0].plot(s["x"][during], s["y"][during], label="flown (mission)")
    axes[0].plot(
        s["x"][after],
        s["y"][after],
        ":",
        c="tab:red",
        label="return + landing"
        if meta.get("start_error") is not None
        else "landing",
    )
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
    ax.axvline(mission_end, c="0.5", ls=":", label="mission end")
    ax.set_xlim(-1, t.max())
    ax.set_xlabel("time from mission start [s]")
    ax.set_ylabel("z [m]")
    ax.legend()
    fig.tight_layout()
    path = trial["dir"] / "flight.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
