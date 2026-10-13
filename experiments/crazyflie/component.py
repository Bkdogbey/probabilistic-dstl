"""ROS Sugar component that executes one flight attempt.

ROS Sugar calls ``_execute_once`` from the component's timer callback when
it is activated. The mission must not block that callback (it would also
block the lifecycle callbacks that deliver shutdown), so ``_execute_once``
only starts a dedicated, non-daemon mission thread and returns.

Stopping: ROS shutdown (``custom_on_deactivate``/``custom_on_shutdown``)
or ``run.launch`` sets ``stop``; ``flight.run_flight`` then lands and
cleans up. ``run.launch`` always waits for the mission thread, so the
process never exits while the drone is still airborne. The radio is
opened inside the mission, so every hardware step is covered by
``run_flight``'s cleanup.
"""

from __future__ import annotations

import os
import signal
import threading

from ros_sugar.core import BaseComponent

import flight


class FlightComponent(BaseComponent):
    def __init__(
        self,
        *,
        component_name,
        cfg,
        t,
        xyz,
        log,
        connect_fn=flight.connect,
        **kwargs,
    ):
        self.cfg, self.t, self.xyz, self.log = cfg, t, xyz, log
        self.connect_fn = connect_fn
        self.stop = threading.Event()
        self.summary = None
        self.mission = None
        super().__init__(component_name=component_name, **kwargs)

    def _execute_once(self):
        self.mission = threading.Thread(
            target=self._fly, name="crazyflie-mission"
        )
        self.mission.start()

    def _fly(self):
        try:
            self.summary = flight.run_flight(
                self.cfg,
                self.t,
                self.xyz,
                self.log,
                stop=self.stop,
                connect_fn=self.connect_fn,
            )
        finally:
            if not self.stop.is_set():
                # Finished on its own: end Launcher.bringup().
                os.kill(os.getpid(), signal.SIGINT)

    def _execution_step(self):
        pass

    def custom_on_deactivate(self):
        self.stop.set()
        return super().custom_on_deactivate()

    def custom_on_shutdown(self):
        self.stop.set()
        return super().custom_on_shutdown()
