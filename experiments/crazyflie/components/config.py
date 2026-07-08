from __future__ import annotations

from attrs import define
from ros_sugar.config import BaseComponentConfig


@define(kw_only=True)
class CrazyflieConfig(BaseComponentConfig):
    """Configuration for the waypoint-based Crazyflie runner.

    Trial settings (condition, fan_speed) live here instead of as constants
    edited in crazyflie.py before each run — set them from main.py's CLI
    args, e.g. `python main.py --condition pdstl --fan 12`.
    """

    z_hold: float = 0.3
    condition: str = 'pdstl'  # 'pdstl' or 'deterministic'
    fan_speed: int = 12  # 2, 6, 12, or 16 -- selects process noise (Q_STD_PER_FAN)
