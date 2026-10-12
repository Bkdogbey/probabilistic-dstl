"""ROS Sugar component that executes one flight attempt.

Same pattern as irobot's ``examples/crazyflie_ros_component.py``: the
mission runs once in ``_execute_once`` when the component is activated.
The radio is opened inside the mission (not in ``__init__``) so every
hardware step is covered by ``flight.run_flight``'s cleanup.
"""

from __future__ import annotations

import os
import signal
import threading

from ros_sugar.core import BaseComponent

import flight


class FlightComponent(BaseComponent):
    def __init__(self, *, component_name, cfg, t, xyz, log, **kwargs):
        self.cfg, self.t, self.xyz, self.log = cfg, t, xyz, log
        self.stop = threading.Event()
        self.summary = None
        super().__init__(component_name=component_name, **kwargs)

    def _execute_once(self):
        try:
            self.summary = flight.run_flight(
                self.cfg, self.t, self.xyz, self.log, stop=self.stop
            )
        finally:
            # One attempt per launch: end Launcher.bringup() once done.
            os.kill(os.getpid(), signal.SIGINT)

    def _execution_step(self):
        pass

    def custom_on_deactivate(self):
        self.stop.set()
        parent = getattr(super(), "custom_on_deactivate", None)
        return parent() if parent else None

    def custom_on_shutdown(self):
        self.stop.set()
        parent = getattr(super(), "custom_on_shutdown", None)
        return parent() if parent else None
