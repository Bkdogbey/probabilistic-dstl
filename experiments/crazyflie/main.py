from __future__ import annotations

import argparse

from ros_sugar import Launcher

from components.config import CrazyflieConfig
from components.crazyflie import CrazyfliePlanning

parser = argparse.ArgumentParser(description='Fly a Crazyflie reach-avoid trial')
parser.add_argument(
    '--condition',
    choices=['pdstl', 'deterministic'],
    default='pdstl',
    help="'pdstl' flies the optimised plan (components/opt_waypoints.py); "
    "'deterministic' flies the nominal safe path (no optimisation)",
)
parser.add_argument(
    '--fan',
    type=int,
    choices=[2, 6, 12, 16],
    default=12,
    help='Fan speed setting; selects process noise for mid-flight replanning',
)
args = parser.parse_args()

my_component = CrazyfliePlanning(
    component_name='crazyflie_planning',
    config=CrazyflieConfig(z_hold=0.3, condition=args.condition, fan_speed=args.fan),
)

launcher = Launcher()
launcher.add_pkg(components=[my_component], activate_all_components_on_start=True)
launcher.bringup()
