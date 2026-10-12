"""Crazyflie pipeline: plan -> fly -> analyze.

    python experiments/crazyflie/run.py plan                 # plan + figure
    python experiments/crazyflie/run.py fly --dry-run        # simulated drone
    python experiments/crazyflie/run.py fly                  # real flight
    python experiments/crazyflie/run.py analyze [RUN] [--all]

``--method s_path|pdstl`` overrides ``planning.method`` for one command.

plan     config.yaml -> planners.plan -> trajectory.check
         -> plans/<name>/<method>/{trajectory.csv, plan.json, plan.png}
fly      reuse that plan while its signature matches the config (else plan
         again) -> show it and confirm -> FlightLog -> ROS Sugar component
         (or SimDrone) -> logs/<run_id>/ with summary.json and flight.png
analyze  logs/<run_id>/ -> metrics, flight figure; --all -> runs.csv and
         the repeated-runs overlay

Exits 0 on success; ``fly`` only when the attempt completed.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import analysis
import config
import flight
import planners
import plots
import trajectory
from flight_log import FlightLog

logger = logging.getLogger(__name__)


# --- plan ------------------------------------------------------------------


def get_plan(cfg, replan=False):
    """The config's plan: reused if saved and in sync, else planned.

    A new plan is validated with ``trajectory.check`` before it is saved,
    so an invalid plan is never stored or flown.

    Returns:
        ``(t, xyz, info, directory)``.
    """
    directory = planners.plan_dir(cfg, config.resolve(cfg, "plans_dir"))
    saved = None if replan else planners.load_plan(directory)
    if saved and saved[2]["signature"] == planners.signature(cfg):
        print(f"Using saved plan {directory}")
        t, xyz, info = saved
        check(cfg, t, xyz)
        return t, xyz, info, directory
    reason = (
        "replan requested"
        if replan
        else (
            "no saved plan"
            if saved is None
            else "config changed since last plan"
        )
    )
    print(f"Planning with {cfg['planning']['method']} ({reason})")
    t, xyz, info = planners.plan(cfg)
    check(cfg, t, xyz)
    info = planners.save_plan(directory, cfg, t, xyz, info)
    plots.plot_plan(cfg, t, xyz, info, directory / "plan")
    return t, xyz, info, directory


def check(cfg, t, xyz):
    """Raise ValueError unless the plan is flyable in this arena."""
    trajectory.check(
        t, xyz, cfg["bounds"], cfg["flight"]["max_speed"], cfg["environment"]
    )


def describe(info, t):
    """One line about a plan."""
    text = f"{info['method']}: {len(t)} points, {t[-1]:.1f} s"
    if "p_lower" in info:
        text += f", P(phi) in [{info['p_lower']:.3f}, {info['p_upper']:.3f}]"
    if "monte_carlo" in info:
        text += f", Monte Carlo {info['monte_carlo']['rate']:.3f}"
    if info.get("alpha_met") is False:
        text += f" (below alpha = {info['alpha']})"
    return text


def cmd_plan(cfg, args):
    t, xyz, info, directory = get_plan(cfg, replan=True)
    print(describe(info, t))
    print(f"Plan saved in {directory}")
    if not args.no_show:
        plots.plot_plan(cfg, t, xyz, info, show=True)
    return 0


# --- fly -------------------------------------------------------------------


def confirm(cfg, t, xyz, info):
    """Show the plan and ask before flying; True to fly."""
    plots.plot_plan(cfg, t, xyz, info, show=True)
    return input("Fly this plan? [y/N] ").strip().lower() == "y"


def launch(cfg, t, xyz, log):
    """Fly through ROS Sugar; returns the run summary."""
    from ros_sugar import Launcher

    from component import FlightComponent

    component = FlightComponent(
        component_name="crazyflie_flight", cfg=cfg, t=t, xyz=xyz, log=log
    )
    launcher = Launcher()
    launcher.add_pkg(
        components=[component], activate_all_components_on_start=True
    )
    try:
        launcher.bringup()
    except KeyboardInterrupt:
        pass
    component.stop.set()
    summary_file = log.dir / "summary.json"
    if component.summary is None and not summary_file.exists():
        return log.finish("error", "flight component did not run")
    return component.summary or json.loads(summary_file.read_text())


def plot_flight(run_dir):
    """Save the post-flight figure; a plotting error never masks the run."""
    try:
        plots.plot_run(analysis.load_run(run_dir), run_dir / "flight")
    except Exception:
        logger.exception("Could not plot %s", run_dir)


def cmd_fly(cfg, args):
    t, xyz, info, directory = get_plan(cfg, replan=args.replan)
    print(describe(info, t))
    if not (args.dry_run or args.yes or confirm(cfg, t, xyz, info)):
        print("Not flown.")
        return 1

    log = FlightLog(cfg, t, xyz, plan_info=info, dry_run=args.dry_run)
    shutil.copy(directory / "plan.png", log.dir / "plan.png")
    print(f"Run {log.run_id} -> {log.dir}")
    if args.dry_run:
        home = (xyz[0, 0], xyz[0, 1], 0.0)
        summary = flight.run_flight(
            cfg, t, xyz, log, connect_fn=lambda c: flight.SimDrone(home=home)
        )
    else:
        summary = launch(cfg, t, xyz, log)
    plot_flight(log.dir)

    print(
        f"Outcome: {summary['outcome']} (last phase {summary['last_phase']})"
    )
    if summary.get("error"):
        print(f"Error: {summary['error']}")
    if summary.get("tracking"):
        print(f"Tracking RMSE: {summary['tracking']['rmse_m']:.3f} m")
    print(f"Logs: {log.dir}")
    return 0 if summary["outcome"] == "completed" else 1


# --- analyze ---------------------------------------------------------------


def cmd_analyze(cfg, args):
    return analysis.report(
        config.resolve(cfg, "dir"),
        run_dir=args.run,
        all_runs=args.all,
        show=not args.no_show,
    )


# --- entry point -----------------------------------------------------------


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=config.DEFAULT)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="plan and plot, no flight")
    fly = commands.add_parser("fly", help="fly the plan (plans if needed)")
    for sub in (plan, fly):
        sub.add_argument("--method", choices=list(planners.PLANNERS))
    fly.add_argument(
        "--dry-run", action="store_true", help="simulated drone, no ROS/radio"
    )
    fly.add_argument("--yes", action="store_true", help="skip the prompt")
    fly.add_argument("--replan", action="store_true", help="ignore saved plan")

    analyze = commands.add_parser("analyze", help="metrics and figures")
    analyze.add_argument("run", nargs="?", type=Path, help="run directory")
    analyze.add_argument("--all", action="store_true", help="every run")
    for sub in (plan, analyze):
        sub.add_argument("--no-show", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    """CLI entry point; returns the process exit code."""
    args = parse(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg = config.load(args.config, method=getattr(args, "method", None))
    command = {"plan": cmd_plan, "fly": cmd_fly, "analyze": cmd_analyze}
    return command[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
