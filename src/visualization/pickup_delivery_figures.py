"""Pickup-delivery figures and tables, regenerated from saved records only.

Every number comes from outputs/pickup_delivery/<exp>/records.jsonl, its
manifest, and artifacts; nothing is typed in by hand.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments import pickup_delivery as pdl
from experiments import task_checker
from visualization.figures import _draw_environment, _ellipse

# Fixed categorical order (reference palette slots 1-4) plus a marker each, so
# identity never rests on colour alone.
METHODS = {
    "nominal": ("Nominal STLCG", "#2a78d6", "o"),
    "margin": ("Margin STLCG", "#eb6834", "s"),
    "pdstl": ("pdSTL", "#1baf7a", "D"),
    "cma": ("CMA-ES (pdSTL objective)", "#eda100", "^"),
}
INK, MUTED = "#222222", "#8a8a85"


def load(directory):
    """(records, manifest) of one experiment directory."""
    directory = Path(directory)
    with (directory / "records.jsonl").open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
    return records, manifest


def _returned(records):
    return [r for r in records if r.get("status") == "returned"]


def _style(ax):
    ax.grid(True, color="#e6e6e3", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# --- Figure 1: scenario before / after -------------------------------------------


def representative(records, method, noise):
    """Rule: the instance with the median execution success (lower median)."""
    rows = [
        r
        for r in _returned(records)
        if r["method"] == method and r["noise"] == noise and r["spec"] == "R3"
    ]
    if not rows:
        return None
    rows.sort(key=lambda r: (r["execution"]["rate"], r["instance"]))
    return rows[(len(rows) - 1) // 2]


def _mean_xy(s, controls):
    return pdl.mean_positions(s, controls)[0]


def _mark_sequence(ax, xy, verdict, color):
    for key, label in (
        ("load_time", "load"),
        ("scan_time", "scan"),
        ("delivery_time", "deliver"),
    ):
        t = int(verdict[key][0])
        if t >= 0:
            ax.scatter(*xy[t], s=36, color=color, edgecolor="white", zorder=9)
            ax.annotate(
                f"{label} t={t}",
                xy[t],
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=INK,
            )


def figure_scenario(directory, records, manifest, out, method="pdstl"):
    """Same map before and after optimization at low and high noise."""
    cases = [
        (noise, representative(records, method, noise))
        for noise in ("low", "high")
    ]
    cases = [(noise, r) for noise, r in cases if r and r.get("artifact")]
    if not cases:
        return None
    cfg = manifest["config"]
    fig, axes = plt.subplots(
        1, len(cases), figsize=(5.2 * len(cases), 5.2), squeeze=False
    )
    for ax, (noise, record) in zip(axes[0], cases):
        instance = manifest["instances"][str(record["instance"])]
        s = pdl.build(cfg, instance, noise, record["H"])
        artifact = torch.load(Path(directory) / record["artifact"])
        before = artifact["start_controls"][record["selected_start"]]
        after = artifact["controls"]
        _draw_environment(ax, s.env)
        cov = artifact["cov_trace"][0].numpy()
        for controls, label, color, style in (
            (before, f"start ({record['selected_start']})", MUTED, "--"),
            (after, "optimized mean", METHODS[method][1], "-"),
        ):
            xy = _mean_xy(s, controls)
            ax.plot(*xy.T, style, color=color, linewidth=2, label=label)
            verdict = task_checker.check(
                xy[None], s.task, s.cfg["H"], s.cfg["dwell"], "R3"
            )
            if style == "-":
                _mark_sequence(ax, xy, verdict, color)
                for t in range(0, len(xy), 20):
                    label = "95% pointwise covariance" if t == 0 else None
                    _ellipse(ax, xy[t], cov[t, :2, :2], label=label)
        score = record["pdstl"]["hard_lower"]
        rate = record["execution"]["rate"]
        ax.set_title(
            f"{noise} noise, instance {record['instance']}: hard "
            f"{record['initial_hard_lower']:.2f}→{score:.2f}, success "
            f"{rate:.2f}",
            fontsize=9,
        )
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(
        "R3 plans (median-success instance per noise level); ellipses are "
        "pointwise covariance contours, not trajectory tubes",
        fontsize=9,
    )
    return _save(fig, out / "fig1_scenario.png")


# --- Figure 2: optimization progress ---------------------------------------------


def _best_so_far(histories, key):
    """Concatenate per-start histories in run order; running best hard."""
    times, values, offset = [], [], 0.0
    for history in histories.values():
        elapsed = np.asarray(history["elapsed"], dtype=float) + offset
        times.extend(elapsed)
        values.extend(history[key])
        offset = elapsed[-1] if len(elapsed) else offset
    if key == "hard":
        values = np.maximum.accumulate(values)
    return np.asarray(times), np.asarray(values)


def figure_optimization(directory, records, out):
    """Best hard pdSTL score vs optimization wall time, per specification.

    Only methods whose objective is pdSTL (pdSTL, CMA-ES) have pdSTL
    histories; the STLCG baselines optimize metres and are shown as their
    returned plans' final hard pdSTL score at the budget.
    """
    specs = sorted({r["spec"] for r in _returned(records)})
    if not specs:
        return None
    fig, axes = plt.subplots(
        2, len(specs), figsize=(4.4 * len(specs), 6.4), squeeze=False
    )
    for column, spec in enumerate(specs):
        hard_ax, smooth_ax = axes[0, column], axes[1, column]
        for method, (label, color, marker) in METHODS.items():
            rows = [
                r
                for r in _returned(records)
                if r["spec"] == spec and r["method"] == method
            ]
            for k, r in enumerate(rows):
                if method in ("pdstl", "cma") and r.get("artifact"):
                    histories = torch.load(Path(directory) / r["artifact"])[
                        "histories"
                    ]
                    t, hard = _best_so_far(histories, "hard")
                    hard_ax.plot(
                        t,
                        hard,
                        color=color,
                        linewidth=1,
                        alpha=0.6,
                        label=label if k == 0 else None,
                    )
                    t, smooth = _best_so_far(histories, "smooth")
                    smooth_ax.plot(
                        t,
                        smooth,
                        color=color,
                        linewidth=0.8,
                        alpha=0.6,
                        label=label if k == 0 else None,
                    )
                else:
                    hard_ax.scatter(
                        r["optimization_s"],
                        r["pdstl"]["hard_lower"],
                        color=color,
                        marker=marker,
                        s=30,
                        edgecolor="white",
                        zorder=5,
                        label=label if k == 0 else None,
                    )
        hard_ax.set_title(f"{spec}", fontsize=10)
        hard_ax.set_ylabel("best hard pdSTL lower score")
        smooth_ax.set_ylabel("smooth lower of current iterate\n(β annealed)")
        smooth_ax.set_xlabel("optimization wall time (s), starts in sequence")
        for ax in (hard_ax, smooth_ax):
            _style(ax)
        hard_ax.legend(fontsize=7)
    fig.suptitle(
        "Hard (top) and smooth (bottom) pdSTL scores are different "
        "quantities; STLCG baselines are shown only as final hard scores",
        fontsize=9,
    )
    return _save(fig, out / "fig2_optimization.png")


# --- Figure 3: execution ---------------------------------------------------------


def figure_execution(records, out, spec="R3"):
    """Full-task success vs noise (per instance, with CIs) and effort."""
    rows = [r for r in _returned(records) if r["spec"] == spec]
    if not rows:
        return None
    noises = [
        n
        for n in ("low", "medium", "high")
        if any(r["noise"] == n for r in rows)
    ]
    methods = [m for m in METHODS if any(r["method"] == m for r in rows)]
    fig, (ax, effort_ax) = plt.subplots(1, 2, figsize=(10, 4.2))
    width = 0.8 / max(len(methods), 1)
    for k, method in enumerate(methods):
        label, color, marker = METHODS[method]
        for j, noise in enumerate(noises):
            group = [
                r
                for r in rows
                if r["method"] == method and r["noise"] == noise
            ]
            x = j + (k - (len(methods) - 1) / 2) * width
            rates = [r["execution"]["rate"] for r in group]
            errors = np.array(
                [
                    [
                        r["execution"]["rate"] - r["execution"]["ci95"][0],
                        r["execution"]["ci95"][1] - r["execution"]["rate"],
                    ]
                    for r in group
                ]
            ).T
            jitter = np.linspace(-0.25, 0.25, len(group)) * width
            ax.errorbar(
                x + jitter,
                rates,
                yerr=errors,
                fmt=marker,
                color=color,
                markersize=5,
                elinewidth=0.8,
                capsize=0,
                label=label if j == 0 else None,
            )
            if rates:
                ax.hlines(
                    np.median(rates),
                    x - width / 2,
                    x + width / 2,
                    color=INK,
                    linewidth=1.5,
                )
            efforts = [r["plan"]["effort"] for r in group]
            effort_ax.scatter(
                x + jitter,
                efforts,
                color=color,
                marker=marker,
                s=24,
                label=label if j == 0 else None,
            )
    for axis in (ax, effort_ax):
        axis.set_xticks(range(len(noises)), noises)
        axis.set_xlabel("process noise level")
        _style(axis)
    ax.set_ylabel(f"{spec} full-task success (95% Clopper–Pearson)")
    ax.set_ylim(-0.02, 1.02)
    effort_ax.set_ylabel("control effort Σ‖u‖² dt")
    ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(
        "Each point is one instance's returned plan; bars are medians across "
        "instances (instances are not pooled)",
        fontsize=9,
    )
    return _save(fig, out / "fig3_execution.png")


# --- Figure 4: monitor scaling ---------------------------------------------------


def figure_scaling(records, out):
    rows = [r for r in records if "family" in r]
    if not rows:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    keys = (
        ("smooth_forward_s", "forward time (s)"),
        ("backward_s", "backward time (s)"),
        ("peak_bytes_forward_backward", "peak tensor memory (MB)"),
    )
    styles = {
        "GF": ("#2a78d6", "o"),
        "FG": ("#eb6834", "s"),
        "until": ("#1baf7a", "D"),
    }
    for ax, (key, label) in zip(axes, keys):
        for family, (color, marker) in styles.items():
            group = sorted(
                (r for r in rows if r["family"] == family),
                key=lambda r: r["length"],
            )
            if not group:
                continue
            values = [r[key] / (1e6 if "bytes" in key else 1) for r in group]
            ax.plot(
                [r["length"] for r in group],
                values,
                marker=marker,
                color=color,
                linewidth=2,
                label=family,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("trace length T (steps)")
        ax.set_ylabel(label)
        _style(ax)
    axes[0].legend(fontsize=8, title="formula family", title_fontsize=8)
    fig.suptitle(
        "Monitor only (rollout excluded), origin-0 score, batch 1, one thread, "
        "float32, median of repeats",
        fontsize=9,
    )
    return _save(fig, out / "fig4_scaling.png")


# --- Tables ----------------------------------------------------------------------


def _median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else float("nan")


def main_table(records):
    """Specification x method rows (medians across instances and noise)."""
    # Every attempted trial, returned or not; diagnostic candidates excluded.
    trials = [r for r in records if "method" in r and "candidate" not in r]
    table = []
    names = {r["method"] for r in trials}
    order = [m for m in METHODS if m in names] + sorted(names - set(METHODS))
    for spec in sorted({r.get("spec") for r in trials if r.get("spec")}):
        for method in order:
            group = [
                r
                for r in trials
                if r.get("spec") == spec and r.get("method") == method
            ]
            if not group:
                continue
            ok = _returned(group)
            initial = [r["initial_hard_lower"] for r in ok]
            final = [r["pdstl"]["hard_lower"] for r in ok]
            success = [r["execution"]["rate"] for r in ok]
            table.append(
                {
                    "spec": spec,
                    "method": METHODS.get(method, (method,))[0],
                    "attempted": len(group),
                    "returned": len(ok),
                    "median_initial_hard": _median(initial),
                    "median_final_hard": _median(final),
                    "median_hard_gain": _median(
                        np.subtract(final, initial).tolist()
                    ),
                    "median_success_returned": _median(success),
                    # No-plan trials count as zero success end to end.
                    "mean_success_attempted": float(sum(success) / len(group)),
                    "median_total_s": _median([r.get("total_s") for r in ok]),
                    "median_smooth_minus_hard": _median(
                        [r["pdstl"]["smooth_minus_hard"] for r in ok]
                    ),
                }
            )
    return table


def failure_table(records):
    rows = []
    for r in _returned(records):
        counts = r["execution"]["failure_counts"]
        rows.append(
            {
                "instance": r["instance"],
                "noise": r["noise"],
                "spec": r["spec"],
                "method": r["method"],
                "samples": r["execution"]["samples"],
                **counts,
            }
        )
    return rows


def write_rows(rows, path):
    if not rows:
        return None
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown = path.with_suffix(".md")
    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [
            f"{v:.3f}" if isinstance(v, float) else str(v)
            for v in row.values()
        ]
        lines.append("| " + " | ".join(cells) + " |")
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_all(directories, config_path=None):
    """Every figure and table each experiment's records support."""
    from experiments.pickup_delivery_study import summarize_scores

    for directory in directories:
        directory = Path(directory)
        records, manifest = load(directory)
        out = directory / "figures"
        out.mkdir(exist_ok=True)
        made = [
            figure_scenario(directory, records, manifest, out),
            figure_optimization(directory, records, out),
            figure_execution(records, out),
            figure_scaling(records, out),
            write_rows(main_table(records), out / "table_main.csv"),
            write_rows(failure_table(records), out / "table_failures.csv"),
        ]
        if any("candidate" in r for r in records):
            made.append(
                write_rows(
                    summarize_scores(records), out / "table_score_ranking.csv"
                )
            )
        for path in filter(None, made):
            print(f"wrote {path}")
