"""Check a path or flights against the task (no hardware, no ROS).

    python analyze.py --preview          # planned figure-8: length, time, task
    python analyze.py [flights/x.csv]    # one flight: tracking error, task, plot
    python analyze.py --all              # every flight in one table
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402

import arena  # noqa: E402
import config  # noqa: E402
from paths import Reference, figure8  # noqa: E402


FLIGHTS = config.HERE / "flights"


def load(path):
    """A flight's rows and its saved metadata (config and outcome)."""
    rows = np.genfromtxt(path, delimiter=",", names=True)
    meta = Path(path).with_suffix(".json")
    return rows, json.loads(meta.read_text()) if meta.exists() else {}


def distance_to_polyline(xyz, line):
    """Distance from each point to the nearest segment of `line`."""
    a, b = line[:-1], line[1:]
    ab = b - a
    denom = np.maximum((ab * ab).sum(1), 1e-12)
    s = np.clip(((xyz[:, None] - a) * ab).sum(2) / denom, 0.0, 1.0)
    nearest = a + s[..., None] * ab
    return np.linalg.norm(xyz[:, None] - nearest, axis=2).min(1)


def box(ax, x, y, **style):
    ax.add_patch(Rectangle((x[0], y[0]), x[1] - x[0], y[1] - y[0], **style))


def draw(ax, cfg, ref_xy, flown_xy=None):
    box(ax, cfg.workspace.x, cfg.workspace.y, fill=False, ls=":", ec="k")
    for b in cfg.obstacles:
        box(ax, b.x, b.y, color="0.5")
    box(ax, cfg.goal.x, cfg.goal.y, color="tab:green", alpha=0.4)
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], "--", label="reference")
    if flown_xy is not None:
        ax.plot(flown_xy[:, 0], flown_xy[:, 1], label="flown")
    ax.set_aspect("equal")
    ax.legend()


def report_task(cfg, t, xy):
    s = arena.score(cfg, arena.sample(t, xy, cfg.dt))
    verdict = "satisfied" if s["robustness"] > 0 else "VIOLATED"
    print(f"task {verdict}: robustness {s['robustness']:+.3f} m, "
          f"closest approach {s['clearance']:+.3f} m")


def metrics(cfg, d):
    """Tracking error and task score of one flight."""
    pos = np.column_stack([d["x"], d["y"], d["z"]])
    ref = np.column_stack([d["rx"], d["ry"], d["rz"]])
    lag = np.linalg.norm(pos - ref, axis=1)
    cross = distance_to_polyline(pos, ref)
    return {
        "duration": d["t"][-1],
        "lag_rms": np.sqrt((lag**2).mean()), "lag_max": lag.max(),
        "cross_rms": np.sqrt((cross**2).mean()), "cross_max": cross.max(),
        "z_mean": pos[:, 2].mean(), "z_std": pos[:, 2].std(),
        **arena.score(cfg, arena.sample(d["t"], pos[:, :2], cfg.dt)),
    }


def preview(cfg):
    points = figure8(cfg)
    ref = Reference(points, cfg.path.speed)
    seg = np.diff(points, axis=0)
    heading = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    curv = np.abs(np.diff(heading)) / np.maximum(
        np.linalg.norm(seg[1:], axis=1), 1e-9)
    print(f"figure-8: {ref.length:.2f} m at {cfg.path.speed} m/s "
          f"= {ref.duration:.0f} s")
    print(f"tightest turn radius {1 / max(curv.max(), 1e-9):.2f} m; "
          f"command limit {cfg.control.v_max} m/s")
    t = np.linspace(0.0, ref.duration, 2000)
    report_task(cfg, t, np.array([ref.at(s)[0][:2] for s in t]))
    fig, ax = plt.subplots(figsize=(4, 7))
    draw(ax, cfg, points[:, :2])
    out = config.HERE / "path_preview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def flight(cfg, path):
    d, meta = load(path)
    m = metrics(cfg, d)
    verdict = "satisfied" if m["robustness"] > 0 else "VIOLATED"
    print(f"{path.name}: {len(d)} steps, {m['duration']:.1f} s, "
          f"outcome: {meta.get('outcome', '?')}")
    print(f"distance to reference point: rms {m['lag_rms']:.3f} m"
          f"  max {m['lag_max']:.3f} m")
    print(f"distance to path:            rms {m['cross_rms']:.3f} m"
          f"  max {m['cross_max']:.3f} m")
    print(f"height: mean {m['z_mean']:.3f} m  std {m['z_std']:.3f} m")
    print(f"task {verdict}: robustness {m['robustness']:+.3f} m, "
          f"closest approach {m['clearance']:+.3f} m")
    fig, ax = plt.subplots(figsize=(4, 7))
    draw(ax, cfg, np.column_stack([d["rx"], d["ry"]]),
         np.column_stack([d["x"], d["y"]]))
    out = path.with_suffix(".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def table(cfg):
    """One line per flight, then the success rate of the task."""
    print(f"{'flight':17} {'outcome':10} {'time':>5} {'rms':>6} {'max':>6} "
          f"{'robust':>7} {'clear':>7}")
    done = satisfied = 0
    for path in sorted(FLIGHTS.glob("*.csv")):
        d, meta = load(path)
        m = metrics(cfg, d)
        outcome = meta.get("outcome", "?").split(":")[0]
        done += outcome == "done"
        satisfied += outcome == "done" and m["robustness"] > 0
        print(f"{path.stem:17} {outcome:10} {m['duration']:5.1f} "
              f"{m['lag_rms']:6.3f} {m['lag_max']:6.3f} "
              f"{m['robustness']:+7.3f} {m['clearance']:+7.3f}")
    print(f"task satisfied in {satisfied} of {done} completed flights "
          "(rms/max: distance to the reference point [m])")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--config", default=config.HERE / "config.yml")
    args = parser.parse_args()
    cfg = config.load(args.config)
    if args.preview:
        preview(cfg)
    elif args.all:
        table(cfg)
    else:
        found = sorted(FLIGHTS.glob("*.csv"))
        flight(cfg, Path(args.csv) if args.csv else found[-1])
