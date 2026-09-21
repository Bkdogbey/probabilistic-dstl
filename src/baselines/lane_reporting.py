"""CSV and publication output for the optional paired lane study."""

import csv
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTCOMES = (
    "success",
    "collision",
    "road_violation",
    "deadline_missed",
    "planning_failure",
    "step_limit",
)
METRICS = (
    "completion_time",
    "minimum_clearance",
    "control_effort",
    "control_smoothness",
    "planning_time",
)


def wilson_interval(successes, total, z=1.96):
    """Two-sided Wilson score interval for one empirical rate."""
    if total == 0:
        return float("nan"), float("nan")
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    radius = (
        z
        / denominator
        * sqrt(
            probability * (1 - probability) / total
            + z * z / (4 * total * total)
        )
    )
    return center - radius, center + radius


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(trials):
    groups = defaultdict(list)
    for row in trials:
        groups[(row["scenario"], row["uncertainty"], row["method"])].append(
            row
        )
    summary = []
    for (scenario, uncertainty, method), rows in sorted(groups.items()):
        for outcome in OUTCOMES:
            count = sum(row["outcome"] == outcome for row in rows)
            low, high = wilson_interval(count, len(rows))
            record = {
                "scenario": scenario,
                "uncertainty": uncertainty,
                "method": method,
                "outcome": outcome,
                "count": count,
                "total": len(rows),
                "rate": count / len(rows),
                "wilson_low": low,
                "wilson_high": high,
            }
            summary.append(record)
    return summary


def summarize_metrics(trials):
    groups = defaultdict(list)
    for row in trials:
        groups[(row["scenario"], row["uncertainty"], row["method"])].append(
            row
        )
    summary = []
    for (scenario, uncertainty, method), rows in sorted(groups.items()):
        for metric in METRICS:
            values = [
                float(row[metric])
                for row in rows
                if row[metric] not in (None, "")
                and np.isfinite(float(row[metric]))
            ]
            summary.append(
                {
                    "scenario": scenario,
                    "uncertainty": uncertainty,
                    "method": method,
                    "metric": metric,
                    "mean": float(np.mean(values)) if values else "",
                    "std": (
                        float(np.std(values, ddof=1))
                        if len(values) > 1
                        else 0.0
                    ),
                    "median": float(np.median(values)) if values else "",
                    "n": len(values),
                }
            )
    return summary


def _plot_rates(summary, output_dir):
    scenarios = sorted({row["scenario"] for row in summary})
    methods = ("deterministic_stl", "pdstl")
    failures = OUTCOMES[1:]
    for scenario in scenarios:
        fig, axes = plt.subplots(3, 1, figsize=(7.5, 9), sharex=True)
        for method in methods:
            rows = [
                row
                for row in summary
                if row["scenario"] == scenario
                and row["method"] == method
                and row["outcome"] == "success"
            ]
            rows.sort(key=lambda row: float(row["uncertainty"]))
            axes[0].errorbar(
                [float(row["uncertainty"]) for row in rows],
                [row["rate"] for row in rows],
                yerr=[
                    [row["rate"] - row["wilson_low"] for row in rows],
                    [row["wilson_high"] - row["rate"] for row in rows],
                ],
                marker="o",
                capsize=3,
                label=method.replace("_", " ").title(),
            )
        axes[0].set(ylabel="success rate", ylim=(-0.03, 1.03))
        axes[0].legend()
        for axis, method in zip(axes[1:], methods):
            for outcome in failures:
                rows = [
                    row
                    for row in summary
                    if row["scenario"] == scenario
                    and row["method"] == method
                    and row["outcome"] == outcome
                ]
                rows.sort(key=lambda row: float(row["uncertainty"]))
                axis.plot(
                    [float(row["uncertainty"]) for row in rows],
                    [row["rate"] for row in rows],
                    marker="o",
                    label=outcome.replace("_", " "),
                )
            axis.set(
                ylabel=f"{method.replace('_', ' ')} failures",
                ylim=(-0.03, 1.03),
            )
            axis.legend(ncol=2, fontsize=8)
        axes[-1].set_xlabel("uncertainty scale")
        for axis in axes:
            axis.grid(alpha=0.2)
        stem = output_dir / f"{scenario}_rates"
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def write_study_outputs(trials, windows, metadata, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(trials)
    _write_csv(output_dir / "trials.csv", trials)
    _write_csv(output_dir / "windows.csv", windows)
    _write_csv(output_dir / "summary.csv", summary)
    _write_csv(output_dir / "metrics.csv", summarize_metrics(trials))
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    _plot_rates(summary, output_dir)
    return summary
