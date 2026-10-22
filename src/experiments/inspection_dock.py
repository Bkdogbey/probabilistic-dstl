"""Inspection and dock: a nested rule under uncertainty.

A shelf wall splits the room; the gap between its shelves is a contaminated
shortcut C. The robot must visit A or B, end at the dock D, and, if it
passed C, spend 1 s at the inspection station I before docking:

    phi = G safe & (not D U (A or B)) & G(C -> (not D U G[0,1 s] I)) & F D

The shortcut (C, inspection at I, then A) is shorter but squeezes through
0.6 m and pays an inspection stop; the clean detour over the shelf (via B)
is longer. The robot follows its plan with a PD tracker, so its uncertainty
is a steady tracking error (config `uncertainty`). STLCG and pdSTL each plan
from both routes, keep one plan by their own rule, and the plan is executed
under noise (Monte Carlo, the same draws for every plan).

Steps (CPU):

    PDSTL_DEVICE=cpu PYTHONPATH=src python -m experiments.inspection_dock run
    PDSTL_DEVICE=cpu PYTHONPATH=src python -m experiments.inspection_dock report

Outputs go to outputs/inspection_dock/.
"""

import argparse
import copy
import csv
from functools import reduce
from itertools import pairwise

import numpy as np
import torch

from baselines.det_stl import from_pdstl
from experiments.two_gap import (
    LABELS,
    METHOD_COLOR,
    clearance,
    effort,
    select,
)
from models.rollouts import (
    create_gaussian_belief_trajectory,
    sample_trajectories,
)
from pdstl.operators import Always, And, Eventually, Implies, Or, Until
from planning.environment import RectangleRegion, inside, outside
from planning.runners import RESULTS_DIR, build_dynamics, setup_problem
from utils import load_config
from visualization.figures import COLORS, _ellipse_parameters

CONFIG = "configs/scenarios/reach_avoid/inspection_dock.yaml"
OUTPUTS = RESULTS_DIR / "inspection_dock"
PARTS = ("safe", "visit", "rule", "dock")


def noise_for_tracking_error(cfg, sigma):
    """Process noise q whose steady 1-sigma tracking error is `sigma` (m).

    The error covariance obeys P' = A_cl P A_cl^T + q^2 I, so its steady
    state is q^2 S with S fixed by the tracker; q follows in closed form.
    """
    dyn = build_dynamics({**cfg, "q_std": 0.0}, "cpu")
    a = dyn.A_cl.double()
    steady = torch.zeros_like(a)
    for _ in range(2000):
        steady = a @ steady @ a.T + torch.eye(len(a), dtype=a.dtype)
    return sigma / float(steady[1, 1]) ** 0.5


def load(path=CONFIG):
    """Config with `noise` (q per level) set from `uncertainty`."""
    cfg = load_config(path)
    cfg["noise"] = {
        level: noise_for_tracking_error(cfg, sigma)
        for level, sigma in cfg["uncertainty"]["tracking_std_m"].items()
    }
    return cfg


def steps(cfg, seconds):
    return round(seconds / cfg["dt"])


def _region(name, box):
    return RectangleRegion(name, "target", *box["x_range"], *box["y_range"])


def specification(cfg):
    """The task formula and its four parts.

    Returns:
        (phi, parts): phi is the conjunction of parts["safe"], ["visit"],
        ["rule"] and ["dock"]; every window is in steps.
    """
    H = cfg["H"]
    times = cfg["times"]
    inspect, within = (
        steps(cfg, times["inspect_s"]),
        steps(cfg, times["within_s"]),
    )
    watch = min(steps(cfg, times["rule_s"]), H - within - inspect)
    room = inside(_region("room", cfg["bounds"]))
    shelves = [outside(_region(o["name"], o)) for o in cfg["obstacles"]]
    p = {n: inside(_region(n, box)) for n, box in cfg["regions"].items()}
    parts = {
        "safe": Always(reduce(And, [room, *shelves]), [0, H]),
        "visit": Until(~p["D"], Or(p["A"], p["B"]), [0, H]),
        "rule": Always(
            Implies(
                p["C"],
                Until(~p["D"], Always(p["I"], [0, inspect]), [0, within]),
            ),
            [0, watch],
        ),
        "dock": Eventually(p["D"], [0, H]),
    }
    return reduce(And, parts.values()), parts


def problem(cfg, method, level):
    """Dynamics, planner and spec for one method and uncertainty level.

    STLCG plans the mean path only, so it gets zero noise and a known start.
    """
    settings = cfg["methods"][method]
    c = copy.deepcopy(cfg)
    c["planner"] = settings["planner"]
    if settings["kind"] == "stlcg" or level is None:
        c["q_std"], c["x0_cov_scale"] = 0.0, 0.0
    else:
        c["q_std"] = cfg["noise"][level]
    s = setup_problem(c)
    s.spec, s.parts = specification(c)
    return s


def evaluation_problem(cfg, level):
    """Noisy problem used to score and simulate any plan at one level."""
    return problem(cfg, "pdstl", level)


# Initial guesses: smooth rest-to-rest motion along each route's legs


def _rounded(points, passes=4):
    """Chaikin corner cutting: a polyline with rounded corners, same ends."""
    p = np.asarray(points, float)
    for _ in range(passes):
        cut = [p[0]]
        for a, b in pairwise(p):
            cut += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        p = np.vstack([*cut, p[-1]])
    return p


def _trapezoid(tau, ramp=0.2):
    """Fraction of a leg covered at normalized time tau: accelerate over
    `ramp`, cruise, decelerate over `ramp` (rest to rest)."""
    peak = 1.0 / (1.0 - ramp)
    up = 0.5 * peak * tau**2 / ramp
    down = 1.0 - 0.5 * peak * (1.0 - tau) ** 2 / ramp
    cruise = peak * (tau - 0.5 * ramp)
    return np.where(tau < ramp, up, np.where(tau > 1 - ramp, down, cruise))


def reference(cfg, legs, arrive=0.9):
    """Positions along the legs, dwelling after each, done at arrive*T."""
    dt, H = cfg["dt"], cfg["H"]
    start = np.array(cfg["x0_mean"][:2], float)
    move = arrive * H * dt - sum(dwell for _, dwell in legs)
    paths, last = [], start
    for points, _ in legs:
        p = _rounded(np.vstack([last, np.array(points, float)]))
        arc = np.r_[0, np.linalg.norm(np.diff(p, axis=0), axis=1).cumsum()]
        paths.append((p, arc))
        last = p[-1]
    total = sum(arc[-1] for _, arc in paths)
    out = [start]
    for (p, arc), (_, dwell) in zip(paths, legs):
        n = max(2, round(move * arc[-1] / total / dt))
        tau = np.linspace(0, 1, n + 1)[1:]
        s = _trapezoid(tau) * arc[-1]
        out += [
            np.array([np.interp(v, arc, p[:, 0]), np.interp(v, arc, p[:, 1])])
            for v in s
        ]
        out += [out[-1].copy() for _ in range(steps(cfg, dwell))]
    out = np.array(out)
    pad = max(0, H + 1 - len(out))
    out = np.vstack([out, np.repeat(out[-1:], pad, axis=0)])
    return torch.tensor(out[: H + 1], dtype=torch.float32)


def initial_guess(cfg, dyn, legs, kp=4.0, kd=4.0):
    """Feed-forward accelerations whose mean path follows `reference`."""
    ref = reference(cfg, legs)
    velocity = torch.cat([(ref[1:] - ref[:-1]) / cfg["dt"], torch.zeros(1, 2)])
    state = torch.tensor(cfg["x0_mean"], dtype=torch.float32)
    a, b = dyn.A.cpu(), dyn.B.cpu()
    controls = []
    for t in range(cfg["H"]):
        u = kp * (ref[t + 1] - state[:2]) + kd * (velocity[t + 1] - state[2:])
        u = u.clamp(-0.999 * dyn.u_max, 0.999 * dyn.u_max)
        state = a @ state + b @ u
        controls.append(u)
    return torch.stack(controls).to(dyn.device)


# What a mean path does


def describe(cfg, mean):
    """Route ('shortcut'/'detour'), target visited ('A'/'B'/'-'), inspected."""
    shelf = cfg["obstacles"][0]["x_range"]
    k = int(np.argmin(np.abs(mean[:, 0] - np.mean(shelf))))
    y = float(mean[k, 1])
    c, top = cfg["regions"]["C"], cfg["obstacles"][1]["y_range"][1]
    route = (
        "shortcut"
        if c["y_range"][0] <= y <= c["y_range"][1]
        else "detour"
        if y > top
        else "none"
    )

    def inside_box(name):
        box = cfg["regions"][name]
        return (
            (mean[:, 0] >= box["x_range"][0])
            & (mean[:, 0] <= box["x_range"][1])
            & (mean[:, 1] >= box["y_range"][0])
            & (mean[:, 1] <= box["y_range"][1])
        )

    dock = np.flatnonzero(inside_box("D"))
    end = dock[0] if len(dock) else len(mean)
    first = {n: np.flatnonzero(inside_box(n)[:end]) for n in ("A", "B")}
    visited = [n for n in ("A", "B") if len(first[n])]
    target = min(visited, key=lambda n: first[n][0]) if visited else "-"
    run = steps(cfg, cfg["times"]["inspect_s"]) + 1
    at_i = inside_box("I")[:end].astype(int)
    inspected = bool(
        len(at_i) >= run
        and np.convolve(at_i, np.ones(run, int), "valid").max() == run
    )
    return route, target, inspected


def plan(cfg, method, level):
    """Optimize from each route's start; one row per candidate."""
    rows = []
    for route, legs in cfg["routes"].items():
        s = problem(cfg, method, level)
        guess = initial_guess(s.cfg, s.dyn, legs)
        result = s.planner.optimize_window(
            s.rollout, spec=s.spec, init_guess=guess
        )
        mean = result.rollout.aux["mean_trace"]
        took, target, inspected = describe(cfg, mean[0].cpu().numpy())
        rows.append(
            {
                "method": method,
                "noise": level,
                "start": route,
                "gap": took,
                "target": target,
                "inspected": inspected,
                "robustness": from_pdstl(s.spec)
                .robustness(mean, beta=None)
                .item(),
                "score": result.hard_interval[0],
                "effort": effort(result.controls),
                "time_s": result.planning_time,
                "controls": result.controls.detach().cpu(),
            }
        )
    return rows


def evaluate(cfg, controls, level):
    """Monte Carlo success (and its 95% interval), failures by part, score."""
    from scipy.stats import beta

    s = evaluation_problem(cfg, level)
    mc = cfg["monte_carlo"]
    generator = torch.Generator(device=s.dyn.device).manual_seed(mc["seed"])
    paths = sample_trajectories(
        s.dyn, *s.state, controls.to(s.dyn.device), mc["samples"], generator
    )
    beliefs = create_gaussian_belief_trajectory(paths, torch.zeros_like(paths))
    satisfied = s.spec(beliefs)[:, 0, 0] > 0.5
    hits, n = int(satisfied.sum()), mc["samples"]
    out = {
        "success": hits / n,
        "ci_low": float(beta.ppf(0.025, hits, n - hits + 1)) if hits else 0.0,
        "ci_high": (
            float(beta.ppf(0.975, hits + 1, n - hits)) if hits < n else 1.0
        ),
    }
    for name, part in s.parts.items():
        failed = part(beliefs)[:, 0, 0] <= 0.5
        out[f"fail_{name}"] = float(failed.float().mean())
    out["score_eval"] = s.planner.evaluate_controls(
        s.rollout, controls.to(s.dyn.device), spec=s.spec
    ).hard_interval[0]
    gaps = clearance(paths.cpu(), cfg["obstacles"])
    out["clearance_p05"] = float(torch.quantile(gaps, 0.05))
    return out


def run(cfg):
    """Plan every method at every level, select, evaluate; save CSVs."""
    candidates, selected, plans = [], [], {}
    for method, settings in cfg["methods"].items():
        # STLCG ignores noise: plan once, evaluate at every level.
        shared = (
            plan(cfg, method, None) if settings["kind"] == "stlcg" else None
        )
        for level in cfg["noise"]:
            rows = shared if shared is not None else plan(cfg, method, level)
            pick = select(cfg, method, rows)
            for row in rows:
                candidates.append(
                    {**_public(row), "noise": level, "selected": row is pick}
                )
            result = evaluate(cfg, pick["controls"], level)
            selected.append(
                {
                    **_public(pick),
                    "noise": level,
                    "time_s": sum(r["time_s"] for r in rows),
                    **result,
                }
            )
            plans[f"{method}/{level}"] = pick["controls"]
            print(
                f"{method:6s} {level:6s} {pick['gap']:8s} {pick['target']} "
                f"inspected={pick['inspected']!s:5s} "
                f"success {100 * result['success']:5.1f}% "
                f"score {result['score_eval']:.3f} | fail "
                + " ".join(
                    f"{p}={100 * result[f'fail_{p}']:.1f}%" for p in PARTS
                ),
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


def table(cfg):
    """One row per uncertainty level and method; CSV and Markdown."""
    sigma = cfg["uncertainty"]["tracking_std_m"]
    rows = []
    for r in _read("selected.csv"):
        rows.append(
            {
                "uncertainty": r["noise"],
                "tracking_std_cm": round(100 * sigma[r["noise"]]),
                "method": LABELS[r["method"]],
                "plan": (
                    f"{r['gap']}, {r['target']}"
                    + (", inspect" if r["inspected"] == "True" else "")
                ),
                "success_pct": round(100 * float(r["success"]), 1),
                "ci_pct": (
                    f"[{100 * float(r['ci_low']):.1f}, "
                    f"{100 * float(r['ci_high']):.1f}]"
                ),
                **{
                    f"fail_{p}_pct": round(100 * float(r[f"fail_{p}"]), 1)
                    for p in PARTS
                },
                "pdstl_score": round(float(r["score_eval"]), 3),
                "effort": round(float(r["effort"]), 1),
            }
        )
    order = {n: i for i, n in enumerate(cfg["noise"])}
    method = {LABELS[m]: i for i, m in enumerate(cfg["methods"])}
    rows.sort(key=lambda r: (order[r["uncertainty"]], method[r["method"]]))
    _write(rows, "table.csv")
    header = list(rows[0])
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(str(r[h]) for h in header) + " |" for r in rows),
    ]
    (OUTPUTS / "table.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUTS / 'table.md'}", flush=True)


def belief(cfg, controls, level):
    """Mean and covariance of a plan, cut when it first reaches the dock."""
    s = evaluation_problem(cfg, level)
    mean, cov = s.state
    means, covs = [mean], [cov]
    for control in controls.to(s.dyn.device):
        mean, cov = s.dyn.step(mean, cov, control)
        means.append(mean)
        covs.append(cov)
    mean = torch.stack(means).cpu().numpy()[:, :2]
    cov = torch.stack(covs).cpu().numpy()[:, :2, :2]
    d = cfg["regions"]["D"]
    docked = np.flatnonzero(
        (mean[:, 0] >= d["x_range"][0])
        & (mean[:, 1] >= d["y_range"][0])
        & (mean[:, 1] <= d["y_range"][1])
    )
    end = docked[0] + 1 if len(docked) else len(mean)
    return mean[:end], cov[:end]


REGION_STYLE = {
    "C": {"fc": "none", "ec": COLORS["obstacle"], "hatch": "xxxx", "lw": 0.8},
    "A": {"fc": COLORS["target"], "ec": COLORS["target"], "alpha": 0.35},
    "B": {"fc": COLORS["target"], "ec": COLORS["target"], "alpha": 0.35},
    "I": {"fc": COLORS["score"], "ec": COLORS["score"], "alpha": 0.3},
    "D": {"fc": COLORS["goal"], "ec": COLORS["goal"], "alpha": 0.35},
}
REGION_TEXT = {
    "C": COLORS["obstacle"],
    "A": "darkgoldenrod",
    "B": "darkgoldenrod",
    "I": COLORS["score"],
    "D": COLORS["goal"],
}


def _draw_room(ax, cfg):
    """Walls, hatched shelves, contaminated gap and the task regions."""
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
    for name, box in cfg["regions"].items():
        (rx0, rx1), (ry0, ry1) = box["x_range"], box["y_range"]
        ax.add_patch(
            Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, **REGION_STYLE[name])
        )
        tx = rx0 - 0.15 if name == "C" else (rx0 + rx1) / 2
        ax.text(
            tx,
            (ry0 + ry1) / 2 if name == "C" else ry1 + 0.12,
            name,
            color=REGION_TEXT[name],
            fontweight="bold",
            ha="right" if name == "C" else "center",
            va="center" if name == "C" else "bottom",
        )
    sx, sy = cfg["x0_mean"][:2]
    ax.plot(sx, sy, "o", color="k", ms=3.5, zorder=5)
    ax.text(sx, sy + 0.25, "S", fontweight="bold", ha="center", va="bottom")


def figure(cfg, methods=("stlcg", "pdstl")):
    """One panel per uncertainty level: both plans and their MC success."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse, Patch

    plans = torch.load(OUTPUTS / "plans.pt")
    picked = {(r["method"], r["noise"]): r for r in _read("selected.csv")}
    levels = list(cfg["noise"])
    every = cfg["figure"]["ellipse_every"]
    sigma = cfg["uncertainty"]["tracking_std_m"]
    with plt.rc_context({"font.size": 8}):
        fig, axes = plt.subplots(
            1,
            len(levels),
            figsize=(7.16, 2.95),
            sharey=True,
            gridspec_kw={"wspace": 0.05},
        )
        for i, (ax, level) in enumerate(zip(axes, levels)):
            _draw_room(ax, cfg)
            for method in methods:
                mean, cov = belief(cfg, plans[f"{method}/{level}"], level)
                color = METHOD_COLOR[method]
                ax.plot(*mean.T, color=color, lw=1.6, zorder=4)
                if method != "pdstl":
                    continue
                for k in [*range(every, len(mean) - 1, every), len(mean) - 1]:
                    w, h, angle = _ellipse_parameters(cov[k], 0.95)
                    ax.add_patch(
                        Ellipse(
                            mean[k],
                            w,
                            h,
                            angle=angle,
                            fc=color,
                            ec=color,
                            alpha=0.22,
                            lw=0.6,
                            zorder=3,
                        )
                    )
            success = "\n".join(
                f"{LABELS[m]}: "
                f"{100 * float(picked[(m, level)]['success']):.1f}%"
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
                f"({'abc'[i]}) {level} uncertainty: "
                f"{100 * sigma[level]:.0f} cm tracking error",
                fontsize=8,
                pad=3,
            )
            last = i == len(levels) - 1
            ax.set(
                xlim=cfg["bounds"]["x_range"],
                ylim=cfg["bounds"]["y_range"],
                aspect="equal",
                xticks=range(0, 11 if last else 10, 2),
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
                label="Shelf",
            ),
            Patch(**REGION_STYLE["C"], label="Contaminated gap C"),
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
    STEPS[args.step](load(args.config))


if __name__ == "__main__":
    main()
