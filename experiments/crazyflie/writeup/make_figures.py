"""Figures and numbers for the hardware write-up, from the recorded flights.

Run from experiments/crazyflie:  python writeup/make_figures.py
Writes writeup/figures/*.pdf and writeup/numbers.tex. Each group (fan x
condition) keeps its first 20 flights that reached the curve, in flight
order, as in `python run.py analyze --first 20`.
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import analysis  # noqa: E402
import config  # noqa: E402
from logger import all_trials  # noqa: E402

FIRST = 20
FANS = ("low", "medium", "high")
CONDITIONS = ("arc", "pdstl")
LABEL = {"arc": "Nominal arc", "pdstl": "pdSTL"}
COLOR = {"arc": "#eb6834", "pdstl": "#2a78d6"}  # categorical slots 2 and 1
MARKER = {"arc": "s", "pdstl": "o"}
FAILED = "#9a9a96"
INK = "#2b2b2a"
FIGURES = HERE / "figures"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": "#8a8a86",
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
    }
)


def condition(meta):
    return "pdstl" if meta["plan_id"].startswith("pdstl") else "arc"


def groups(cfg, task):
    """(fan, condition) -> [(trial, result)], the first FIRST that count."""
    out = defaultdict(list)
    for trial in all_trials():
        meta = trial["meta"]
        if meta["purpose"] != "evaluation" or meta.get("dry_run"):
            continue
        out[(meta["fan"], condition(meta))].append(
            (trial, analysis.evaluate(trial, task))
        )
    for key, pairs in out.items():
        kept = analysis.first_counted([r for _, r in pairs], FIRST)
        out[key] = pairs[: len(kept)]
    return out


def counted(pairs):
    return [(t, r) for t, r in pairs if "excluded" not in r]


def draw_arena(ax, cfg):
    """Measured boxes and the goal."""
    for box in cfg.obstacles:
        (x0, x1), (y0, y1) = box.x, box.y
        ax.add_patch(
            Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#8a8a86", ec="none", zorder=2)
        )
    (x0, x1), (y0, y1) = cfg.goal.x, cfg.goal.y
    ax.add_patch(
        Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fc="none", ec=INK, lw=0.8, ls=(0, (3, 2)), zorder=2,
        )
    )


def paths_figure(cfg, data):
    fig, axes = plt.subplots(
        2, 3, figsize=(7.0, 5.6), sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.06, "hspace": 0.12},
    )
    for row, cond in enumerate(CONDITIONS):
        for col, fan in enumerate(FANS):
            ax = axes[row, col]
            draw_arena(ax, cfg)
            pairs = counted(data[(fan, cond)])
            for trial, result in pairs:
                xy = analysis.mission_samples(trial)[:, :2]
                ok = result["success"]
                ax.plot(
                    xy[:, 0], xy[:, 1], lw=0.7, zorder=4 if ok else 3,
                    color=COLOR[cond] if ok else FAILED, alpha=0.85 if ok else 0.6,
                )
            rows = np.asarray(pairs[0][0]["meta"]["coefficients"])
            curve = analysis.curve_xy(rows)
            ax.plot(curve[:, 0], curve[:, 1], color=INK, lw=1.0, ls="--", zorder=5)
            k = sum(r["success"] for _, r in pairs)
            ax.set_title(f"{LABEL[cond]}, fan {fan}: {k}/{len(pairs)}", color=INK)
            ax.set_xlim(-0.25, 1.15)
            ax.set_ylim(-2.15, -0.02)
            ax.set_aspect("equal")
            ax.grid(False)
            if row == 1:
                ax.set_xlabel("x [m]")
            if col == 0:
                ax.set_ylabel("y [m]")
    handles = [
        plt.Line2D([], [], color=INK, ls="--", lw=1.0, label="uploaded curve"),
        plt.Line2D([], [], color=COLOR["arc"], lw=1.2, label="success (nominal arc)"),
        plt.Line2D([], [], color=COLOR["pdstl"], lw=1.2, label="success (pdSTL)"),
        plt.Line2D([], [], color=FAILED, lw=1.2, label="failure"),
        Rectangle((0, 0), 1, 1, fc="#8a8a86", label="box"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handlelength=1.6, columnspacing=1.0)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.96, bottom=0.11)
    fig.savefig(FIGURES / "flown_paths.pdf")
    plt.close(fig)


def success_figure(data):
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    x = np.arange(len(FANS))
    for i, cond in enumerate(CONDITIONS):
        rates, lows, highs = [], [], []
        for fan in FANS:
            pairs = counted(data[(fan, cond)])
            k, n = sum(r["success"] for _, r in pairs), len(pairs)
            lo, hi = analysis.clopper_pearson(k, n)
            rates.append(k / n)
            lows.append(k / n - lo)
            highs.append(hi - k / n)
        offset = (i - 0.5) * 0.22
        ax.errorbar(
            x + offset, rates, yerr=[lows, highs], fmt=MARKER[cond], ms=5,
            color=COLOR[cond], ecolor=COLOR[cond], elinewidth=1.2, capsize=0,
            mec="white", mew=0.8, label=LABEL[cond], zorder=3,
        )
    ax.set_xticks(x, [f"fan {f}" for f in FANS])
    ax.set_xlim(-0.5, len(FANS) - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.yaxis.grid(True, color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES / "success_rates.pdf")
    plt.close(fig)


def write_table(data):
    """The results table body as LaTeX rows, one per fan level and condition."""
    lines = []
    for fan in FANS:
        for cond in CONDITIONS:
            pairs = counted(data[(fan, cond)])
            results = [r for _, r in pairs]
            n = len(results)
            k = sum(r["success"] for r in results)
            lo, hi = analysis.clopper_pearson(k, n)
            excluded = len(data[(fan, cond)]) - n
            lines.append(
                f"{fan if cond == 'arc' else ''} & {LABEL[cond]} & "
                f"{k}/{n} & [{lo:.2f}, {hi:.2f}] & "
                f"{sum(r['fell'] for r in results)}/{n} & "
                f"{sum(r['height_held'] for r in results)}/{n} & "
                f"{excluded} \\\\"
            )
        lines.append("\\midrule" if fan != FANS[-1] else "")
    (HERE / "results_table.tex").write_text("\n".join(lines).rstrip() + "\n")


def main():
    cfg = config.load(config.HERE / "config.yml")
    task = analysis.scoring_task(cfg)
    FIGURES.mkdir(exist_ok=True)
    data = groups(cfg, task)
    paths_figure(cfg, data)
    success_figure(data)
    write_table(data)
    print(f"wrote {FIGURES} and {HERE / 'results_table.tex'}")


if __name__ == "__main__":
    main()
