"""Two-gap reach-avoid: does pdSTL put safety first as noise grows?

Three blocks leave a narrow gap level with the start and goal (the short way)
and a wide gap that needs a detour. STLCG requires a fixed margin on the mean
path; pdSTL requires an exact score. Both then spend the least effort. Each
method plans from both gaps, keeps one plan by its own rule, and the plan is
executed open loop under noise (Monte Carlo, the same draws for every plan).

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
    """'narrow' or 'wide' from the mean path's height at the blocks.

    The narrow gap is above the middle block, the wide gap below it.
    """
    mid = next(o for o in cfg["obstacles"] if o["name"] == "mid")
    k = int(np.argmin(np.abs(mean[:, 0] - np.mean(mid["x_range"]))))
    y = float(mean[k, 1])
    if y > mid["y_range"][1]:
        return "narrow"
    return "wide" if y < mid["y_range"][0] else "none"


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

LABELS = {"stlcg": "STLCG", "pdstl": "pdSTL", "stlcg_m30": "STLCG, 30 cm"}
COLORS = {"stlcg": "#D55E00", "pdstl": "#0072B2", "stlcg_m30": "#999999"}


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


def figure(cfg, methods=("stlcg", "pdstl")):
    """Map at each noise level with each method's mean path and spread.

    The formula asks only to reach the goal, so each path is drawn up to its
    arrival (closest approach to the goal centre); afterwards the plan is
    free to drift.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Rectangle

    from visualization.figures import _ellipse_parameters

    plans = torch.load(OUTPUTS / "plans.pt")
    picked = {(r["method"], r["noise"]): r for r in _read("selected.csv")}
    levels = list(cfg["noise"])
    fig, axes = plt.subplots(
        1, len(levels), figsize=(4.2 * len(levels), 3.6), sharey=True
    )
    for ax, noise in zip(axes, levels):
        for block in cfg["obstacles"]:
            (x0, x1), (y0, y1) = block["x_range"], block["y_range"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="0.6"))
        (gx0, gx1), (gy0, gy1) = cfg["goal"]["x_range"], cfg["goal"]["y_range"]
        ax.add_patch(
            Rectangle(
                (gx0, gy0), gx1 - gx0, gy1 - gy0, color="#2e7d32", alpha=0.3
            )
        )
        ax.plot(*cfg["x0_mean"][:2], "ko", ms=4)
        goal_centre = np.array([(gx0 + gx1) / 2, (gy0 + gy1) / 2])
        s = evaluation_problem(cfg, noise)
        for method in methods:
            controls = plans[f"{method}/{noise}"].to(s.dyn.device)
            mean, cov = s.state
            means, covs = [mean], [cov]
            for control in controls:
                mean, cov = s.dyn.step(mean, cov, control)
                means.append(mean)
                covs.append(cov)
            mean = torch.stack(means).cpu().numpy()
            cov = torch.stack(covs).cpu().numpy()
            arrival = int(np.argmin(np.hypot(*(mean[:, :2] - goal_centre).T)))
            mean, cov = mean[: arrival + 1], cov[: arrival + 1]
            r = picked[(method, noise)]
            ax.plot(
                *mean[:, :2].T,
                color=COLORS[method],
                lw=1.6,
                label=(f"{LABELS[method]}: {100 * float(r['success']):.1f}%"),
            )
            every = cfg["figure"]["ellipse_every"]
            for k in [*range(0, arrival, every), arrival]:
                width, height, angle = _ellipse_parameters(
                    cov[k, :2, :2], 0.95
                )
                ax.add_patch(
                    Ellipse(
                        mean[k, :2],
                        width,
                        height,
                        angle=angle,
                        color=COLORS[method],
                        alpha=0.15,
                        lw=0,
                    )
                )
        ax.set(
            xlim=cfg["bounds"]["x_range"],
            ylim=cfg["bounds"]["y_range"],
            aspect="equal",
            title=f"{noise} noise (q = {cfg['noise'][noise]})",
        )
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(alpha=0.2)
    fig.text(
        0.5,
        0.01,
        "Mean paths up to arrival at the goal; 95% position ellipses every "
        f"{every * cfg['dt']:g} s and at arrival. Legend: Monte Carlo "
        "success.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for suffix in (".png", ".pdf"):
        fig.savefig(OUTPUTS / f"figure{suffix}", dpi=150)
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
