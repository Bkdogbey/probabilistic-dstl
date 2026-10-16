"""Fly the figure-8; all settings come from config.yml.

    conda activate ros_env
    source ~/sugar_ws/install/setup.bash
    cd experiments/crazyflie
    PYTHONPATH=~/Research/irobot python run.py
"""

from ros_sugar import Launcher

import config
from flight import Flight

component = Flight(component_name="flight", cfg=config.load())
launcher = Launcher()
launcher.add_pkg(components=[component], activate_all_components_on_start=True)
launcher.bringup()
