#!/usr/bin/env python3
"""Single entry point for the Crazyflie reach-avoid experiment.

Two subcommands:

    # 1. Plan (offline, no hardware): optimise waypoints for a fan level and
    #    write waypoints/pdstl_fan<L>.json  (+ plots/fan<L>_comparison.png)
    python run.py plan --fan 12 --plot

    # 2. Fly (needs hardware + ROS): fly a plan start->finish, no replanning
    python run.py fly --condition pdstl --fan 12          # optimised plan
    python run.py fly --condition deterministic --fan 6   # nominal safe path

All configuration lives in components/config.py. `plan` needs only
torch/numpy/matplotlib; `fly` additionally needs cflib, ros_sugar and the
irobot package (imported lazily, so `plan` works without them).
"""

from __future__ import annotations

import argparse

from components.config import VALID_FANS


def _plan(args: argparse.Namespace) -> None:
    # Lazy import: keeps torch/matplotlib off the `fly` path.
    from waypoint_planning import run_plan

    run_plan(fan=args.fan, plot=args.plot)


def _fly(args: argparse.Namespace) -> None:
    # Lazy import: cflib/ros_sugar/irobot are only needed to actually fly.
    from ros_sugar import Launcher

    from components.crazyflie import CrazyflieConfig, CrazyfliePlanning

    component = CrazyfliePlanning(
        component_name='crazyflie_planning',
        config=CrazyflieConfig(z_hold=0.3, condition=args.condition, fan_speed=args.fan),
    )
    launcher = Launcher()
    launcher.add_pkg(components=[component], activate_all_components_on_start=True)
    launcher.bringup()


def main() -> None:
    parser = argparse.ArgumentParser(description='Crazyflie pdSTL reach-avoid experiment')
    sub = parser.add_subparsers(dest='command', required=True)

    p_plan = sub.add_parser('plan', help='Optimise waypoints for a fan level (offline)')
    p_plan.add_argument('--fan', type=int, default=2, choices=VALID_FANS,
                        help='Fan level; selects the initial belief covariance Σ0')
    p_plan.add_argument('--plot', action='store_true',
                        help='Also save plots/fan<L>_comparison.png')
    p_plan.set_defaults(func=_plan)

    p_fly = sub.add_parser('fly', help='Fly a plan start->finish (needs hardware)')
    p_fly.add_argument('--condition', choices=['pdstl', 'deterministic'], default='pdstl',
                       help="'pdstl' flies the optimised plan for --fan; "
                            "'deterministic' flies the nominal safe path")
    p_fly.add_argument('--fan', type=int, default=12, choices=VALID_FANS,
                       help='Fan level; selects which optimised plan to fly and tags the logs')
    p_fly.set_defaults(func=_fly)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
