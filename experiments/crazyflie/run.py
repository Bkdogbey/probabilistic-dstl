"""Crazyflie pdSTL experiment: plan, fly, fit, analyze.

python run.py plan --condition deterministic [--sequence reference|<excitation>]
python run.py plan --condition mean_model|pdstl --fan off [--model reports/X.json]
python run.py fly --plan plans/X.json --purpose pilot --fan off [--dry-run]
python run.py fit
python run.py analyze [--trial TRIAL_ID]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import config
import numpy as np
from logger import LOGS, TrialLog, all_trials, load_trial
from planner import (
    CONDITIONS,
    Task,
    excitation_velocities,
    load_plan,
    make_plan,
    plan_mean,
    plan_pdstl,
    reference_velocities,
    save_plan,
)

# Imported by a real flight; each entry says how to provide it.
FLIGHT_STACK = {
    "ros_sugar": "conda activate ros_env && source ~/sugar_ws/install/setup.bash",
    "cflib": "pip install cflib",
    "irobot": "pip install -e ~/Research/irobot",
}


def cmd_plan(cfg, args):
    import analysis

    task = Task(cfg)
    if args.condition == "deterministic":
        if args.sequence == "reference":
            v, start, mission = reference_velocities(cfg), cfg.start, True
        else:
            v = excitation_velocities(cfg, args.sequence)
            start, mission = cfg.excitation["start"], False
        plan = make_plan(
            cfg,
            "deterministic",
            args.sequence,
            start,
            v,
            task,
            mission=mission,
        )
    else:
        from tracking_model import REPORTS, load_model

        if args.fan is None:
            sys.exit("model plans need --fan")
        reports = sorted(REPORTS.glob("model_*.json"))
        path = args.model or (reports[-1] if reports else None)
        if path is None:
            sys.exit("no model report; run `fit` first")
        model = load_model(path, args.fan, cfg)
        init = np.clip(reference_velocities(cfg), -cfg.u_max, cfg.u_max)
        optimize = plan_pdstl if args.condition == "pdstl" else plan_mean
        controls, score = optimize(cfg, task, model, init)
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
    path = save_plan(plan)
    figure = analysis.plot_plan(plan, task, cfg, path.with_suffix(".png"))
    print(
        f"{plan['id']}: {len(plan['velocities'])} steps of {cfg.dt} s, "
        f"score {plan['score']}, problems {plan['problems'] or 'none'}"
    )
    print(f"  {path}\n  {figure}")


def missing_flight_stack():
    missing = []
    for module, hint in FLIGHT_STACK.items():
        try:
            importlib.import_module(module)
        except ImportError as exc:
            missing.append(f"{module}: {hint} [{exc}]")
    return missing


def cmd_fly(cfg, args):
    import analysis
    from flight import SimDrone, launch, run_flight

    plan = load_plan(args.plan, cfg)
    if plan["fan"] is not None and plan["fan"] != args.fan:
        sys.exit(
            f"plan {plan['id']} is for fan {plan['fan']!r}, not {args.fan!r}"
        )
    if args.purpose != "pilot" and (reason := cfg.ready_for_data(args.fan)):
        sys.exit(f"refusing a {args.purpose} flight: {reason}")
    if not args.dry_run and (missing := missing_flight_stack()):
        print("Cannot fly from this Python environment; missing:")
        print("\n".join(f"  - {line}" for line in missing))
        sys.exit("Nothing was flown or logged. Use --dry-run to simulate.")

    setting = cfg.fans.get(args.fan)
    print(
        f"{plan['id']} ({plan['condition']}), {len(plan['velocities'])} x "
        f"{cfg.dt} s at z = {cfg.altitude} m, start {plan['start']}, "
        f"fan {args.fan} (setting {setting}), purpose {args.purpose}"
        + (" [DRY RUN]" if args.dry_run else "")
    )
    if (
        not args.dry_run
        and input("Fan set, arena clear? [y/N] ").lower() != "y"
    ):
        sys.exit("aborted before connecting")

    log = TrialLog(
        {
            "purpose": args.purpose,
            "condition": plan["condition"],
            "plan_id": plan["id"],
            "plan_hash": hashlib.sha256(
                Path(args.plan).read_bytes()
            ).hexdigest()[:12],
            "model_id": plan["model_id"],
            "fan": args.fan,
            "fan_setting": setting,
            "fan_position": cfg.fan_position,
            "fan_direction": cfg.fan_direction,
            "dry_run": args.dry_run,
            "note": args.note,
            "dt": cfg.dt,
            "steps": len(plan["velocities"]),
            "repeats": cfg.repeats,
            "stream_hz": cfg.stream_hz,
            "altitude": cfg.altitude,
            "altitude_tolerance": cfg.altitude_tolerance,
            "config_hash": cfg.hash,
            "config": asdict(cfg),
            "git_commit": config.git_commit(),
        }
    )
    if args.dry_run:
        home = (*plan["start"], 0.0)
        meta = run_flight(cfg, plan, log, connect_fn=lambda _: SimDrone(home))
    else:
        meta = launch(cfg, plan, log)
    print(f"Trial {meta['trial_id']}: {meta['outcome']}")
    figure = analysis.check_trial(load_trial(log.dir), Task(cfg), cfg)
    if figure:
        print(f"  {figure}")


def cmd_fit(cfg, args):
    from analysis import plot_validation
    from tracking_model import fit, save_report

    report = fit(cfg, all_trials())
    path = save_report(report)
    plot_validation(report, path.with_suffix(".png"))
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
            f"{v['final_error'][0]:.3f} m; {coverage}; brier {v['brier'][0]:.3f}"
        )


def cmd_analyze(cfg, args):
    import analysis

    task = Task(cfg)
    if args.trial:
        print(analysis.check_trial(load_trial(LOGS / args.trial), task, cfg))
        return
    trials = all_trials()
    analysis.summarize(trials, task)
    analysis.plot_paths(trials, task, cfg)
    print((analysis.RESULTS / "summary.md").read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=config.HERE / "config.yml")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--condition", choices=CONDITIONS, required=True)
    p.add_argument("--sequence", default="reference")
    p.add_argument("--fan")
    p.add_argument("--model")
    f = sub.add_parser("fly")
    f.add_argument("--plan", required=True)
    f.add_argument(
        "--purpose",
        choices=("pilot", "identification", "evaluation"),
        required=True,
    )
    f.add_argument("--fan", required=True)
    f.add_argument("--dry-run", action="store_true")
    f.add_argument("--note", default="")
    sub.add_parser("fit")
    a = sub.add_parser("analyze")
    a.add_argument("--trial")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = config.load(args.config)
    {"plan": cmd_plan, "fly": cmd_fly, "fit": cmd_fit, "analyze": cmd_analyze}[
        args.action
    ](cfg, args)


if __name__ == "__main__":
    main()
