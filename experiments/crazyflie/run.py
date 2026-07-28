#!/usr/bin/env python3
"""Single entry point for the Crazyflie reach-avoid experiment.

Three subcommands:

    # 0. Trial selection (fan/condition/scenario) defaults to config.yml's
    #    trial: section -- edit that instead of typing flags every run:
    python run.py plan            # plans config.yml's trial.fan/scenario
    python run.py fly             # flies config.yml's trial.fan/condition/scenario
    python run.py analyze         # analyzes config.yml's trial.fan/condition

    # Every flag below is an override for one-off runs (e.g. batch loops);
    # none of them are required.

    # 1. Plan (offline, no hardware): optimise waypoints for a fan level and
    #    write waypoints/pdstl_fan<L>.json  (+ plots/fan<L>_comparison.png)
    python run.py plan --fan 12 --plot
    python run.py plan --fan 12 --scenario gate --plot     # gate/descend/avoid/land

    # 2. Fly (needs hardware + ROS): fly a plan start->finish
    python run.py fly --condition pdstl --fan 12          # optimised plan
    python run.py fly --condition deterministic --fan 6   # nominal safe path
    python run.py fly --condition pdstl --fan 12 --scenario gate

    # 3. Analyze (offline, no hardware): plot a logged flight against its plan
    python run.py analyze --condition pdstl --fan 12      # latest run
    python run.py analyze --all                           # every condition/fan pair
    python run.py analyze --summary                       # cross-run table + rollup, no plots

All configuration, including which fan/condition/scenario to act on by
default, lives in config.yml (loaded by components/config.py). `plan`/
`analyze` need only torch/numpy/matplotlib; `fly` additionally needs cflib,
ros_sugar and the irobot package (imported lazily, so `plan`/`analyze` work
without them).
"""

from __future__ import annotations

import argparse

from components.config import (
    Q_STD,
    SIGMA0_PER_FAN,
    TRIAL_CONDITION,
    TRIAL_FAN,
    TRIAL_SCENARIO,
    VALID_FANS,
    VALID_SCENARIOS,
    pdstl_plan_meta,
)


def _add_scenario_arg(parser: argparse.ArgumentParser) -> None:
    """--scenario is identical on `plan`, `fly` and `analyze`; defined once here."""
    parser.add_argument('--scenario', choices=list(VALID_SCENARIOS), default=TRIAL_SCENARIO,
                        help=f"'baseline' = 2D single-phase reach-avoid; 'gate' = 3D climb "
                             f"through a gate, descend to 0.3m, avoid, land; 'figure8' = 3D "
                             f"closed-loop figure-eight with an oscillating altitude profile "
                             f"(default: config.yml's trial.scenario = {TRIAL_SCENARIO!r})")


def _check_pdstl_converged(fan: int, scenario: str) -> None:
    """Refuse to fly a pdstl plan generated under a different uncertainty
    model, or whose optimizer never found a satisfying trajectory
    (rho_after <= 0), before any hardware connection is attempted.
    """
    meta = pdstl_plan_meta(fan, scenario)
    if meta.get('sigma0') != SIGMA0_PER_FAN[fan] or meta.get('q_std') != Q_STD:
        raise SystemExit(
            f'Refusing to fly stale pdstl fan {fan} waypoints (scenario={scenario}): '
            'they were generated with a different uncertainty model. '
            f'Re-run `python run.py plan --fan {fan}'
            f'{"" if scenario == "baseline" else f" --scenario {scenario}"}`.'
        )
    rho_after = meta.get('rho_after')
    if rho_after is None or rho_after <= 0:
        raise SystemExit(
            f'Refusing to fly pdstl fan {fan} (scenario={scenario}): '
            f'rho_after={rho_after} -- no converged plan. '
            f'Re-run `python run.py plan --fan {fan}'
            f'{"" if scenario == "baseline" else f" --scenario {scenario}"}` '
            f'after tuning, or use --condition deterministic.'
        )


def _plan(args: argparse.Namespace) -> None:
    # Lazy import: keeps torch/matplotlib off the `fly` path.
    from waypoint_planning import run_plan

    run_plan(fan=args.fan, scenario=args.scenario, plot=args.plot)


def _analyze(args: argparse.Namespace) -> None:
    # Lazy import: keeps torch/matplotlib off the `fly` path.
    from analyze_logs import run_analyze

    run_analyze(
        condition=args.condition, fan=args.fan, run=args.run, all_=args.all, summary=args.summary,
        scenario=args.scenario,
    )


def _fly(args: argparse.Namespace) -> None:
    if args.condition == 'pdstl':
        _check_pdstl_converged(args.fan, args.scenario)

    # Lazy import: cflib/ros_sugar/irobot are only needed to actually fly.
    from ros_sugar import Launcher

    from components.crazyflie import CrazyflieConfig, CrazyfliePlanning

    # Drone radio address comes from config.yml's DRONE_URI (set it there once
    # for your drone); CrazyflieConfig's hw_config default factory picks it up
    # automatically, falling back to irobot's own default if unset. z_hold
    # also defaults from config.yml (shared with the gate scenario's
    # post-gate descent altitude) -- no need to pass it here.
    component = CrazyfliePlanning(
        component_name='crazyflie_planning',
        config=CrazyflieConfig(
            condition=args.condition, fan_speed=args.fan, scenario=args.scenario,
        ),
    )
    launcher = Launcher()
    launcher.add_pkg(components=[component], activate_all_components_on_start=True)
    launcher.bringup()


def main() -> None:
    parser = argparse.ArgumentParser(description='Crazyflie pdSTL reach-avoid experiment')
    sub = parser.add_subparsers(dest='command', required=True)

    p_plan = sub.add_parser('plan', help='Optimise waypoints for a fan level (offline)')
    p_plan.add_argument('--fan', type=int, default=TRIAL_FAN, choices=VALID_FANS,
                        help='Fan level; selects the empirically characterized initial '
                             f"covariance (default: config.yml's trial.fan = {TRIAL_FAN})")
    _add_scenario_arg(p_plan)
    p_plan.add_argument('--plot', action='store_true',
                        help='Also save plots/fan<L>[_<scenario>]_comparison.png')
    p_plan.set_defaults(func=_plan)

    p_fly = sub.add_parser('fly', help='Fly a plan start->finish (needs hardware)')
    p_fly.add_argument('--condition', choices=['pdstl', 'deterministic'], default=TRIAL_CONDITION,
                       help="'pdstl' flies the optimised plan for --fan; 'deterministic' flies "
                            f"the nominal safe path (default: config.yml's trial.condition = "
                            f'{TRIAL_CONDITION!r})')
    p_fly.add_argument('--fan', type=int, default=TRIAL_FAN, choices=VALID_FANS,
                       help='Fan level; selects which optimised plan to fly and tags the logs '
                            f"(default: config.yml's trial.fan = {TRIAL_FAN})")
    _add_scenario_arg(p_fly)
    p_fly.set_defaults(func=_fly)

    p_analyze = sub.add_parser(
        'analyze', help='Plot a logged flight against its planned path (offline)'
    )
    p_analyze.add_argument('--condition', choices=['pdstl', 'deterministic'], default=TRIAL_CONDITION,
                           help=f"default: config.yml's trial.condition = {TRIAL_CONDITION!r}; "
                                f'ignored if --all or --summary is given')
    p_analyze.add_argument('--fan', type=int, choices=VALID_FANS, default=TRIAL_FAN,
                           help=f"default: config.yml's trial.fan = {TRIAL_FAN}; "
                                f'ignored if --all or --summary is given')
    p_analyze.add_argument('--run', type=int, default=None,
                           help='Run number; defaults to the latest logged run')
    p_analyze.add_argument('--all', action='store_true',
                           help='Plot the latest run of every condition/fan pair with logs')
    p_analyze.add_argument('--summary', action='store_true',
                           help='Print a cross-run table + per-condition/fan rollup (crash rate, '
                                'mean unsafe fraction, mean duration) instead of plotting')
    _add_scenario_arg(p_analyze)
    p_analyze.set_defaults(func=_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
