"""Slit or around: does pdSTL put safety first as noise grows?

A floor block and an upper block leave a narrow slit between them (the short
way); above the upper block the room is open (a longer way around). STLCG
requires a fixed margin on the mean path; pdSTL requires an exact score. Both
then spend the least effort. Each method plans from both routes, keeps one
plan by its own rule, and the plan is executed open loop under noise (Monte
Carlo, the same draws for every plan).

Steps (CPU):

    PDSTL_DEVICE=cpu PYTHONPATH=src python -m experiments.two_gap run
    PDSTL_DEVICE=cpu PYTHONPATH=src python -m experiments.two_gap report

`run` writes candidates.csv, selected.csv and plans.pt; `report` reads them
and writes the table and figure. Outputs go to outputs/two_gap/.
"""

import argparse
import copy
import csv

import numpy as np
import torch

from baselines.det_stl import from_pdstl
from experiments.reach_avoid import straight_line_guess
from models.rollouts import (
    create_gaussian_belief_trajectory,
    sample_trajectories,
)
from planning.environment import build_reach_avoid_environment
from planning.runners import RESULTS_DIR, setup_problem
from utils import load_config
from visualization.figures import COLORS, _ellipse_parameters

CONFIG = "configs/scenarios/reach_avoid/two_gap.yaml"
OUTPUTS = RESULTS_DIR / "two_gap"


def problem(cfg, method, noise):
    """Dynamics, planner, environment and spec for one method and level.

    STLCG plans the mean path only, so it gets zero noise and a known start.

    Args:
        cfg: Scenario config.
        method: Key of `cfg["methods"]`.
        noise: Key of `cfg["noise"]`, or None for noise-free.

    Returns:
        SimpleNamespace from setup_problem, plus `env` and `spec`.
    """
    settings = cfg["methods"][method]
    c = copy.deepcopy(cfg)
    c["planner"] = settings["planner"]
    if settings["kind"] == "stlcg" or noise is None:
        c["q_std"], c["x0_cov_scale"] = 0.0, 0.0
    else:
        c["q_std"] = cfg["noise"][noise]
    s = setup_problem(c)
    s.env = build_reach_avoid_environment(c)
    s.spec = s.env.get_specification(c["H"])
    return s


def evaluation_problem(cfg, noise):
    """Noisy problem used to score and simulate any plan at one level."""
    return problem(cfg, "pdstl", noise)


def gap_used(cfg, mean):
    """'slit' or 'around' from the mean path's height at the blocks.

    The slit lies between the floor block and the upper block; the way
    around passes above the upper block.
    """
    block = {o["name"]: o for o in cfg["obstacles"]}
    floor, upper = block["floor"], block["upper"]
    k = int(np.argmin(np.abs(mean[:, 0] - np.mean(upper["x_range"]))))
    y = float(mean[k, 1])
    if floor["y_range"][1] < y < upper["y_range"][0]:
        return "slit"
    return "around" if y > upper["y_range"][1] else "none"


def effort(controls):
    """Control effort, sum of squared accelerations."""
    return float(torch.as_tensor(controls).square().sum())


def mean_robustness(spec, mean_trace):
    """Hard STL robustness of the mean path (STLCG's criterion)."""
    return from_pdstl(spec).robustness(mean_trace, beta=None).item()


def plan(cfg, method, noise):
    """Optimize from each gap's start; return one row per candidate."""
    rows = []
    for gap, waypoints in cfg["gaps"].items():
        s = problem(cfg, method, noise)
        guess = straight_line_guess(s.cfg, s.dyn, waypoints)
        result = s.planner.optimize_window(
            s.rollout, spec=s.spec, init_guess=guess
        )
        mean = result.rollout.aux["mean_trace"]
        rows.append(
            {
                "method": method,
                "noise": noise,
                "start": gap,
                "gap": gap_used(cfg, mean[0].cpu().numpy()),
                "robustness": mean_robustness(s.spec, mean),
                "score": result.hard_interval[0],
                "effort": effort(result.controls),
                "time_s": result.planning_time,
                "controls": result.controls.detach().cpu(),
            }
        )
    return rows


def select(cfg, method, rows):
    """Meet the method's requirement, then least effort.

    STLCG: hard mean robustness >= margin_target. pdSTL: exact score >=
    score_threshold. If no candidate qualifies, keep the best criterion.
    """
    settings = cfg["methods"][method]
    if settings["kind"] == "stlcg":
        key, need = "robustness", settings["planner"]["margin_target"]
    else:
        key, need = "score", settings["planner"]["score_threshold"]
    feasible = [r for r in rows if r[key] >= need]
    if feasible:
        return min(feasible, key=lambda r: r["effort"])
    return max(rows, key=lambda r: r[key])


def clearance(paths, obstacles):
    """Per-path minimum distance from the robot to any block, in metres."""
    xy = paths[..., :2]
    distance = torch.full(xy.shape[:2], float("inf"))
    for block in obstacles:
        (x0, x1), (y0, y1) = block["x_range"], block["y_range"]
        dx = torch.clamp(torch.maximum(x0 - xy[..., 0], xy[..., 0] - x1), 0)
        dy = torch.clamp(torch.maximum(y0 - xy[..., 1], xy[..., 1] - y1), 0)
        distance = torch.minimum(distance, torch.hypot(dx, dy))
    return distance.min(dim=1).values


def evaluate(cfg, controls, noise):
    """Monte Carlo success, its 95% interval, score and clearance."""
    from scipy.stats import beta

    s = evaluation_problem(cfg, noise)
    mc = cfg["monte_carlo"]
    generator = torch.Generator(device=s.dyn.device).manual_seed(mc["seed"])
    paths = sample_trajectories(
        s.dyn, *s.state, controls.to(s.dyn.device), mc["samples"], generator
    )
    beliefs = create_gaussian_belief_trajectory(paths, torch.zeros_like(paths))
    satisfied = s.spec(beliefs)[:, 0, 0] > 0.5
    hits, n = int(satisfied.sum()), mc["samples"]
    lower = beta.ppf(0.025, hits, n - hits + 1) if hits else 0.0
    upper = beta.ppf(0.975, hits + 1, n - hits) if hits < n else 1.0
    gaps = clearance(paths.cpu(), cfg["obstacles"])
    score = s.planner.evaluate_controls(
        s.rollout, controls.to(s.dyn.device), spec=s.spec
    ).hard_interval[0]
    return {
        "success": hits / n,
        "ci_low": float(lower),
        "ci_high": float(upper),
        "score_eval": score,
        "clearance_median": float(gaps.median()),
        "clearance_p05": float(torch.quantile(gaps, 0.05)),
    }


def run(cfg):
    """Plan every method at every level, select, evaluate; save CSVs."""
    candidates, selected, plans = [], [], {}
    for method, settings in cfg["methods"].items():
        # STLCG ignores noise: plan once, evaluate at every level.
        shared = (
            plan(cfg, method, None) if settings["kind"] == "stlcg" else None
        )
        for noise in cfg["noise"]:
            rows = shared if shared is not None else plan(cfg, method, noise)
            pick = select(cfg, method, rows)
            for row in rows:
                candidates.append(
                    {**_public(row), "noise": noise, "selected": row is pick}
                )
            result = evaluate(cfg, pick["controls"], noise)
            selected.append(
                {
                    **_public(pick),
                    "noise": noise,
                    "time_s": sum(r["time_s"] for r in rows),
                    **result,
                }
            )
            plans[f"{method}/{noise}"] = pick["controls"]
            print(
                f"{method:9s} {noise:6s} {pick['gap']:6s} "
                f"success {100 * result['success']:5.1f}% "
                f"[{100 * result['ci_low']:.1f}, "
                f"{100 * result['ci_high']:.1f}] "
                f"score {result['score_eval']:.3f}",
                flush=True,
            )
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    _write(candidates, "candidates.csv")
    _write(selected, "selected.csv")
    torch.save(plans, OUTPUTS / "plans.pt")


def _public(row):
    return {k: v for k, v in row.items() if k != "controls"}


def _write(rows, name):
    with open(OUTPUTS / name, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUTPUTS / name}", flush=True)


def _read(name):
    with open(OUTPUTS / name, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# Report: table and figure from the saved results

LABELS = {"stlcg": "STLCG", "pdstl": "pdSTL"}
# The repo's figure palette (visualization.figures.COLORS): pdSTL in the
# belief blue, the deterministic plan in the planned orange.
METHOD_COLOR = {"pdstl": COLORS["mean"], "stlcg": COLORS["planned"]}


def table(cfg):
    """One row per noise level and method; CSV and Markdown."""
    rows = []
    for r in _read("selected.csv"):
        rows.append(
            {
                "noise": r["noise"],
                "method": LABELS[r["method"]],
                "gap": r["gap"],
                "success_pct": round(100 * float(r["success"]), 1),
                "ci_pct": (
                    f"[{100 * float(r['ci_low']):.1f}, "
                    f"{100 * float(r['ci_high']):.1f}]"
                ),
                "pdstl_score": round(float(r["score_eval"]), 3),
                "effort": round(float(r["effort"]), 1),
                "clearance_p05_cm": round(100 * float(r["clearance_p05"]), 1),
            }
        )
    noise = {n: i for i, n in enumerate(cfg["noise"])}
    method = {LABELS[m]: i for i, m in enumerate(cfg["methods"])}
    rows.sort(key=lambda r: (noise[r["noise"]], method[r["method"]]))
    _write(rows, "table.csv")
    header = list(rows[0])
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(str(r[h]) for h in header) + " |" for r in rows),
    ]
    (OUTPUTS / "table.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUTS / 'table.md'}", flush=True)


def belief(cfg, controls, noise):
    """Mean and covariance of the open-loop plan at one noise level.

    Cut at arrival, the step closest to the goal centre: the formula asks
    only to reach the goal, so the plan is free to drift afterwards.
    """
    s = evaluation_problem(cfg, noise)
    mean, cov = s.state
    means, covs = [mean], [cov]
    for control in controls.to(s.dyn.device):
        mean, cov = s.dyn.step(mean, cov, control)
        means.append(mean)
        covs.append(cov)
    mean = torch.stack(means).cpu().numpy()[:, :2]
    cov = torch.stack(covs).cpu().numpy()[:, :2, :2]
    (gx0, gx1), (gy0, gy1) = cfg["goal"]["x_range"], cfg["goal"]["y_range"]
    centre = np.array([(gx0 + gx1) / 2, (gy0 + gy1) / 2])
    arrival = int(np.argmin(np.linalg.norm(mean - centre, axis=1)))
    return mean[: arrival + 1], cov[: arrival + 1]


def _draw_room(ax, cfg):
    """Walls, obstacles and goal in the repo's style; start marked S."""
    from matplotlib.patches import Rectangle

    (x0, x1), (y0, y1) = cfg["bounds"]["x_range"], cfg["bounds"]["y_range"]
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            ec=COLORS["workspace"],
            lw=1.3,
        )
    )
    for block in cfg["obstacles"]:
        (bx0, bx1), (by0, by1) = block["x_range"], block["y_range"]
        ax.add_patch(
            Rectangle(
                (bx0, by0),
                bx1 - bx0,
                by1 - by0,
                fc=COLORS["obstacle"],
                ec=COLORS["obstacle"],
                alpha=0.45,
                hatch="//",
            )
        )
    (gx0, gx1), (gy0, gy1) = cfg["goal"]["x_range"], cfg["goal"]["y_range"]
    ax.add_patch(
        Rectangle(
            (gx0, gy0),
            gx1 - gx0,
            gy1 - gy0,
            fc=COLORS["goal"],
            ec=COLORS["goal"],
            alpha=0.35,
        )
    )
    ax.text(
        (gx0 + gx1) / 2,
        gy1 + 0.15,
        "G",
        color=COLORS["goal"],
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    sx, sy = cfg["x0_mean"][:2]
    ax.plot(sx, sy, "o", color="k", ms=3.5, zorder=5)
    ax.text(sx, sy + 0.25, "S", fontweight="bold", ha="center", va="bottom")


def _draw_plan(ax, mean, cov, method, every):
    """Mean path; pdSTL also gets its 95% belief ellipses every `every`.

    STLCG plans the noise-free mean, so it has no belief to draw; its
    executed spread shows in its Monte Carlo success instead.
    """
    from matplotlib.patches import Ellipse

    color = METHOD_COLOR[method]
    ax.plot(*mean.T, color=color, lw=1.6, zorder=4)
    if method != "pdstl":
        return
    for k in [*range(every, len(mean) - 1, every), len(mean) - 1]:
        width, height, angle = _ellipse_parameters(cov[k], 0.95)
        ax.add_patch(
            Ellipse(
                mean[k],
                width,
                height,
                angle=angle,
                fc=color,
                ec=color,
                alpha=0.22,
                lw=0.6,
                zorder=3,
            )
        )


def figure(cfg, methods=("stlcg", "pdstl")):
    """One panel per noise level: both plans and their MC success.

    Sized for a two-column IEEE figure (7.16 in) with the panels filling
    the width; the explanation of the marks belongs in the caption.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plans = torch.load(OUTPUTS / "plans.pt")
    picked = {(r["method"], r["noise"]): r for r in _read("selected.csv")}
    levels = list(cfg["noise"])
    every = cfg["figure"]["ellipse_every"]
    with plt.rc_context({"font.size": 8}):
        fig, axes = plt.subplots(
            1,
            len(levels),
            figsize=(7.16, 2.95),
            sharex=True,
            sharey=True,
            gridspec_kw={"wspace": 0.05},
        )
        for i, (ax, noise) in enumerate(zip(axes, levels)):
            _draw_room(ax, cfg)
            for method in methods:
                mean, cov = belief(cfg, plans[f"{method}/{noise}"], noise)
                _draw_plan(ax, mean, cov, method, every)
            success = "\n".join(
                f"{LABELS[m]}: "
                f"{100 * float(picked[(m, noise)]['success']):.1f}%"
                for m in reversed(methods)
            )
            ax.text(
                0.25,
                9.75,
                success,
                va="top",
                fontsize=7.5,
                linespacing=1.35,
                bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1.5},
            )
            ax.set_title(
                f"({'abc'[i]}) {noise} noise, q = {cfg['noise'][noise]:g}",
                fontsize=8,
                pad=3,
            )
            ax.set(
                xlim=cfg["bounds"]["x_range"],
                ylim=cfg["bounds"]["y_range"],
                aspect="equal",
                xticks=range(0, 11, 2),
                yticks=range(0, 11, 2),
            )
            ax.set_xlabel("x [m]", labelpad=1)
            ax.grid(color=COLORS["workspace"], alpha=0.16, lw=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=7, length=2, pad=1.5)
        axes[0].set_ylabel("y [m]", labelpad=1)
        handles = [
            Line2D([], [], color=METHOD_COLOR["pdstl"], lw=1.6, label="pdSTL"),
            Patch(
                fc=METHOD_COLOR["pdstl"],
                ec=METHOD_COLOR["pdstl"],
                alpha=0.3,
                label="pdSTL 95% belief",
            ),
            Line2D([], [], color=METHOD_COLOR["stlcg"], lw=1.6, label="STLCG"),
            Patch(
                fc=COLORS["obstacle"],
                ec=COLORS["obstacle"],
                alpha=0.45,
                hatch="//",
                label="Obstacle",
            ),
            Patch(
                fc=COLORS["goal"], ec=COLORS["goal"], alpha=0.35, label="Goal"
            ),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=5,
            fontsize=7.5,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.4,
            borderaxespad=0.1,
        )
        fig.subplots_adjust(left=0.05, right=0.998, bottom=0.1, top=0.88)
        for suffix in (".png", ".pdf"):
            fig.savefig(
                OUTPUTS / f"figure{suffix}",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.02,
            )
        plt.close(fig)
    print(f"wrote {OUTPUTS / 'figure.png'}", flush=True)


def report(cfg):
    """Table and figure from the saved results."""
    table(cfg)
    figure(cfg)


STEPS = {"run": run, "report": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args()
    STEPS[args.step](load_config(args.config))


if __name__ == "__main__":
    main()
