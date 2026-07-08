from __future__ import annotations

from attrs import define
from ros_sugar.config import BaseComponentConfig


@define(kw_only=True)
class CrazyflieConfig(BaseComponentConfig):
    """Configuration for the waypoint-based Crazyflie runner."""

    z_hold: float = 0.3
