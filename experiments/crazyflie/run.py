"""Crazyflie experiment: fly an uploaded min-snap curve, fit, plan, analyze.

python run.py fly --fan 0
python run.py analyze --trial latest
python run.py fit
python run.py plan --condition pdstl --fan 0 --output plans/pdstl_fan0.json
python run.py export plans/reference.json --csv /tmp/reference.csv

The configured reference is already registered. Use import only for a new
waypoint file.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

import config
import minsnap
import numpy as np
from logger import TrialLog, all_trials, find_trial
from trajectory import load_plan, read_plan, save_plan

# Imported by a real flight; each entry says how to provide it.
FLIGHT_STACK = {
    "ros_sugar": "conda activate ros_env && source ~/sugar_ws/install/setup.bash",
    "cflib": "pip install cflib",
    "irobot": "pip install -e ~/Research/irobot --no-deps",
}


def cmd_import(cfg, args):
    import analysis
    from geometry import Task
    from trajectory import import_plan

    durations = args.durations or [args.segment]
    path = import_plan(args.path, durations, cfg, args.output)
    plan = read_plan(path)
    flight_cfg = cfg.for_plan(plan)
    analysis.plot_plan(
        plan, Task(flight_cfg), flight_cfg, path.with_suffix(".png")
    )
    print(f"Imported {len(plan['durations'])} pieces: {path}")
    return path


def cmd_export(cfg, args):
    plan = read_plan(args.plan)
    minsnap.write_csv(args.csv, plan["coefficients"])
    print(f"Wrote {len(plan['coefficients'])} pieces to {args.csv}")


def cmd_plan(cfg, args):
    import analysis
    from geometry import Task
    from planner import plan
    from tracking_model import load_model
    from trajectory import reference_plan

    if (
        args.output
        and Path(args.output).resolve() == Path(cfg.reference).resolve()
    ):
        sys.exit(
            "choose a separate model-plan filename; the reference stays frozen"
        )
    model_path = args.model or cfg.model_report
    if model_path is None:
        sys.exit(
            "set model_report in config.yml or supply --model; "
            "no model is selected automatically"
        )
    model = load_model(model_path, args.fan, cfg)
    task = Task(cfg)
    record = plan(
        cfg, task, model, reference_plan(cfg), args.condition, args.fan
    )
    path = save_plan(record, args.output)
    analysis.plot_plan(record, task, cfg, path.with_suffix(".png"))
    print(f"{path}")
    print(f"  feasible: {record['feasible']} {record['problems'] or ''}")
    print(
        f"  exact score {record['exact_score']:.4f}, "
        f"threshold {record['threshold']} met: {record['threshold_met']}"
    )
    print(
        f"  min clearance {record['min_clearance']:.3f} m, largest waypoint "
        f"move from the reference {record['waypoint_shift_max']:.3f} m"
    )
    return path


def missing_flight_stack():
    missing = []
    for module, hint in FLIGHT_STACK.items():
        try:
            importlib.import_module(module)
        except ImportError as exc:
            missing.append(f"{module}: {hint} [{exc}]")
    return missing


def select_plan(cfg, args):
    """An explicit file or the configured frozen reference; never newest."""
    path = Path(args.plan or cfg.reference)
    if not path.exists() and not path.is_absolute():
        path = config.HERE / path
    if not path.exists():
        sys.exit(f"no trajectory {path}; select its saved file")
    try:
        plan = load_plan(path, cfg, args.allow_below_threshold)
    except ValueError as exc:
        sys.exit(str(exc))
    if plan["fan"] is not None and plan["fan"] != args.fan:
        sys.exit(f"plan is for fan {plan['fan']!r}, not {args.fan!r}")
    return path, plan


def describe(plan, path):
    """What will be uploaded and flown."""
    rows = np.asarray(plan["coefficients"])
    t = minsnap.dense_times(rows)
    speed = np.linalg.norm(minsnap.evaluate(rows, t, 1)[:, :2], axis=1)
    accel = np.linalg.norm(minsnap.evaluate(rows, t, 2)[:, :2], axis=1)
    return (
        f"  {len(rows)} pieces, duration {minsnap.total_duration(rows):.1f} s, "
        f"altitude {plan['altitude']} m, start {np.round(plan['start'], 3).tolist()}\n"
        f"  max speed {speed.max():.3f} m/s, "
        f"max acceleration {accel.max():.3f} m/s^2, "
        f"min clearance {plan['min_clearance']:.3f} m\n"
        f"  plan figure {path.with_suffix('.png')}"
    )


def cmd_fly(cfg, args):
    from flight import launch, run_flight

    path, plan = select_plan(cfg, args)
    cfg = cfg.for_plan(plan)
    if not args.dry_run and (missing := missing_flight_stack()):
        print("Cannot fly from this Python environment; missing:")
        print("\n".join(f"  - {line}" for line in missing))
        sys.exit("Nothing was flown or logged. Use --dry-run to simulate.")

    dry = " [DRY RUN]" if args.dry_run else ""
    print(
        f"{plan['id']} ({plan['condition']}), purpose {args.purpose}, "
        f"fan {args.fan}{dry}"
    )
    print(describe(plan, path))
    log = TrialLog.for_flight(
        cfg,
        plan,
        path,
        purpose=args.purpose,
        fan=args.fan,
        dry_run=args.dry_run,
        note=args.note,
    )
    if args.dry_run:
        from simulator import SimDrone

        home = (*plan["start"], 0.0)
        meta = run_flight(cfg, plan, log, connect_fn=lambda _: SimDrone(home))
    else:
        meta = launch(cfg, plan, log)
    print(f"Trial {meta['trial_id']}: {meta['outcome']}")


def cmd_fit(cfg, args):
    from tracking_model import fit, save_report

    try:
        report = fit(cfg, all_trials())
    except ValueError as exc:
        sys.exit(
            f"cannot fit yet: {exc} (identification flights that completed "
            "or aborted on a task violation; dry runs do not count)"
        )
    path = save_report(report)
    print(f"{report['id']}: {path}")
    for warning in report["warnings"]:
        print(f"  warning: {warning}")
    for fan, f in report["fans"].items():
        print(
            f"  fan {fan}: {f['fit_flights']} fit flights, "
            f"rho {np.round(f['rho'], 3).tolist()}, "
            f"b {np.round(f['b'], 4).tolist()} m, "
            f"stationary spread {np.round(f['stationary_spread'], 4).tolist()} m"
        )
        if any(f["b_poorly_determined"]):
            print("    b is poorly determined: rho is close to 1")
        v = report["validation"].get(fan)
        if v:
            coverage = ", ".join(
                f"{k}: {c[0]:.2f} [{c[1]:.2f}, {c[2]:.2f}]"
                for k, c in v["coverage"].items()
            )
            print(f"    {v['flights']} held out; {coverage}")


def cmd_analyze(cfg, args):
    import analysis
    from geometry import Task

    task = Task(cfg)
    if args.trial:
        print(analysis.check_trial(find_trial(args.trial), task, cfg))
        return
    trials = all_trials()
    analysis.summarize(trials, task)
    print((analysis.RESULTS / "summary.md").read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=config.HERE / "config.yml")
    sub = parser.add_subparsers(dest="action", required=True)
    imp = sub.add_parser(
        "import", help="fit a curve through x,y waypoint rows and freeze it"
    )
    imp.add_argument("path", help="CSV of x,y rows, the first is the start")
    times = imp.add_mutually_exclusive_group(required=True)
    times.add_argument("--durations", type=float, nargs="+")
    times.add_argument("--segment", type=float, help="one duration for all")
    imp.add_argument("--output", required=True, help="new plan file")
    exp = sub.add_parser("export", help="write a plan in uav_trajectories CSV")
    exp.add_argument("plan")
    exp.add_argument("--csv", required=True)
    p = sub.add_parser("plan", help="optimize a model-based plan offline")
    p.add_argument(
        "--condition", choices=("mean_model", "pdstl"), required=True
    )
    p.add_argument(
        "--fan",
        type=int,
        choices=config.FAN_SETTINGS,
        help="default: fan in config.yml",
    )
    p.add_argument("--model", help="default: configured model_report")
    p.add_argument("--output", help="optional saved plan filename")
    f = sub.add_parser("fly", help="upload and fly a saved curve")
    f.add_argument("--plan", help="saved file; default: configured reference")
    f.add_argument(
        "--purpose",
        choices=("pilot", "identification", "evaluation"),
        default="pilot",
    )
    f.add_argument(
        "--fan",
        type=int,
        choices=config.FAN_SETTINGS,
        help="physical setting; default: fan in config.yml",
    )
    f.add_argument("--dry-run", action="store_true")
    f.add_argument("--allow-below-threshold", action="store_true")
    f.add_argument("--note", default="")
    sub.add_parser("fit", help="fit and validate the error model offline")
    a = sub.add_parser("analyze", help="score and plot recorded flights")
    a.add_argument(
        "--trial", help='trial id, a unique part of it, or "latest"'
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        cfg = config.load(args.config)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        sys.exit(str(exc))
    if args.action in ("plan", "fly") and args.fan is None:
        args.fan = cfg.fan
    {
        "import": cmd_import,
        "export": cmd_export,
        "plan": cmd_plan,
        "fly": cmd_fly,
        "fit": cmd_fit,
        "analyze": cmd_analyze,
    }[args.action](cfg, args)


if __name__ == "__main__":
    main()
