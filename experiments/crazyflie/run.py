"""Crazyflie experiment: replay supplied commands, fit, plan and analyze.

python run.py fly --fan 0
python run.py analyze --trial latest
python run.py fit

The configured reference is already registered. Use import only for a new file.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

import config
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

    path = import_plan(args.path, cfg, args.output)
    plan = read_plan(path)
    cfg = cfg.for_plan(plan)
    analysis.plot_plan(
        plan,
        Task(cfg),
        cfg,
        path.with_suffix(".png"),
    )
    print(f"Imported unchanged commands: {path}")
    return path


def cmd_plan(cfg, args):
    import analysis
    from geometry import Task
    from planner import make_plan, plan_mean, plan_pdstl
    from tracking_model import load_model
    from trajectory import reference_velocities

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
            "set model_report in config.yml or supply --model; no model is selected automatically"
        )
    model = load_model(model_path, args.fan, cfg)
    task = Task(cfg)
    optimize = plan_pdstl if args.condition == "pdstl" else plan_mean
    controls, score = optimize(cfg, task, model, reference_velocities(cfg))
    plan = make_plan(
        cfg,
        args.condition,
        f"fan-{args.fan}",
        cfg.start,
        controls,
        task,
        args.fan,
        model,
        score,
    )
    path = save_plan(plan, args.output)
    analysis.plot_plan(plan, task, cfg, path.with_suffix(".png"))
    print(
        f"{path}: clearance {plan['min_clearance']:.3f} m, score {score}, problems {plan['problems'] or 'none'}"
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
    plan = load_plan(path, cfg)
    if plan["fan"] is not None and plan["fan"] != args.fan:
        sys.exit(f"plan is for fan {plan['fan']!r}, not {args.fan!r}")
    return path, plan


def cmd_fly(cfg, args):
    from flight import launch, run_flight

    path, plan = select_plan(cfg, args)
    cfg = cfg.for_plan(plan)
    if not args.dry_run and (missing := missing_flight_stack()):
        print("Cannot fly from this Python environment; missing:")
        print("\n".join(f"  - {line}" for line in missing))
        sys.exit("Nothing was flown or logged. Use --dry-run to simulate.")

    v = np.asarray(plan["velocities"])
    print(
        f"{plan['id']} ({plan['condition']}), purpose {args.purpose}, "
        f"fan {args.fan}"
        + (" [DRY RUN]" if args.dry_run else "")
        + f"\n  duration {len(v) * cfg.dt:.1f} s, altitude {cfg.altitude} m, "
        f"start {plan['start']}, max speed {np.linalg.norm(v, axis=1).max():.2f} m/s, "
        f"clearance {plan['min_clearance']:.3f} m"
        f"\n  plan figure {path.with_suffix('.png')}"
    )
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
            f"cannot fit yet: {exc} (completed identification flights; dry runs do not count)"
        )
    path = save_report(report)
    print(
        f"{report['id']}: tau {np.round(report['tau'], 3).tolist()} s, "
        f"gain {np.round(report['gain'], 3).tolist()}\n  {path}"
    )
    for fan, v in report["validation"].items():
        coverage = ", ".join(
            f"{k}: {c[0]:.2f} [{c[1]:.2f}, {c[2]:.2f}]"
            for k, c in v["coverage"].items()
        )
        print(
            f"  fan {fan}: {v['flights']} held out; final error "
            f"{v['final_error'][0]:.3f} m; {coverage}"
        )


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
        "import", help="check a supplied trajectory once and freeze it"
    )
    imp.add_argument("path")
    imp.add_argument("--output", help="default: configured reference file")
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
    f = sub.add_parser("fly", help="replay a saved velocity trajectory")
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
    f.add_argument("--note", default="")
    sub.add_parser("fit", help="fit and validate the flight model offline")
    a = sub.add_parser("analyze", help="score and plot recorded flights")
    a.add_argument(
        "--trial", help='trial id, a unique part of it, or "latest"'
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        cfg = (
            config.load(args.config, trajectory_source=args.path)
            if args.action == "import"
            else config.load(args.config)
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        sys.exit(str(exc))
    if args.action in ("plan", "fly") and args.fan is None:
        args.fan = cfg.fan
    {
        "import": cmd_import,
        "plan": cmd_plan,
        "fly": cmd_fly,
        "fit": cmd_fit,
        "analyze": cmd_analyze,
    }[args.action](cfg, args)


if __name__ == "__main__":
    main()
