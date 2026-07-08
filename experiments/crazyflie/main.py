from __future__ import annotations

from ros_sugar import Launcher

from components.config import CrazyflieConfig
from components.crazyflie import CrazyfliePlanning

my_component = CrazyfliePlanning(
    component_name='crazyflie_planning',
    config=CrazyflieConfig(z_hold=0.3),
)

launcher = Launcher()
launcher.add_pkg(components=[my_component], activate_all_components_on_start=True)
launcher.bringup()
