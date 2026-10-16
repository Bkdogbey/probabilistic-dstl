"""ROS Sugar component: one flight with velocity commands.

Connection, arming and Lighthouse state come from irobot's
CrazyflieController: take off where the drone sits, fly straight to the start
of the figure-8, follow it, land.
Every control step goes to flights/<time>.csv, next to a JSON copy of the
config.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from ros_sugar.core import BaseComponent

from irobot.src.robots.crazyflie import CrazyflieConfig, CrazyflieController
from paths import Reference, figure8, transit
from tracking import Abort, inside_workspace, settle, track

COLUMNS = ["t", "x", "y", "z", "rx", "ry", "rz", "vx", "vy", "vz"]


class Drone(CrazyflieController):
    """Adds a timestamp to each Lighthouse state packet."""

    def __init__(self, config):
        self.state_time = 0.0
        super().__init__(config)

    def _publish_state(self):
        self.state_time = time.monotonic()

    def state(self):
        pos = np.array([self.current_x, self.current_y, self.current_z])
        return pos, time.monotonic() - self.state_time


class Flight(BaseComponent):
    def __init__(self, *, component_name, cfg, **kwargs):
        self.cfg = cfg
        super().__init__(component_name=component_name, **kwargs)

    def _execute_once(self):
        cfg, fl = self.cfg, self.cfg.flight
        points = figure8(cfg)  # fails before connecting if the path is bad
        ref = Reference(points, cfg.path.speed)
        print(f"path: {ref.length:.2f} m, {ref.duration:.0f} s")
        rows = []
        drone = Drone(CrazyflieConfig(
            uri=cfg.radio_uri, connect_timeout=fl.connect_timeout,
            hover_z=cfg.altitude))
        airborne = False
        outcome = "error"  # replaced below unless an exception escapes
        try:
            if not drone.is_connected:
                raise Abort("could not connect")
            here = settle(drone, fl)
            print(f"drone at {np.round(here[:2], 2)}, path starts at "
                  f"{np.round(points[0][:2], 2)}")
            if not inside_workspace(here, cfg.workspace, fl.fence_margin):
                raise Abort("the drone is outside the workspace")
            try:
                to_start = Reference(transit(cfg, here, points[0]),
                                     fl.transit_speed)
            except ValueError as err:
                raise Abort(str(err)) from None

            commander = drone.cf.commander

            def send(v):
                commander.send_velocity_world_setpoint(*v, 0.0)

            airborne = True
            drone.takeoff(z_hold=cfg.altitude, duration=fl.takeoff_s)
            track(to_start, drone.state, send, cfg)
            print("at the start; following the figure-8")
            track(ref, drone.state, send, cfg,
                  lambda t, p, r, v: rows.append([t, *p, *r, *v]))
            outcome = "done"
        except Abort as err:
            outcome = f"aborted: {err}"
        except KeyboardInterrupt:
            outcome = "interrupted"
        finally:
            try:
                if airborne:
                    drone.land(z_hold=max(drone.current_z, 0.1))
            finally:
                if drone.is_connected:
                    drone.cf.commander.send_notify_setpoint_stop()
                    drone.disarm()
                    drone.disconnect()
                print(outcome)
                self._save(rows, outcome)

    def _save(self, rows, outcome):
        if not rows:
            return
        folder = Path(__file__).parent / "flights"
        folder.mkdir(exist_ok=True)
        stem = folder / time.strftime("%Y%m%d_%H%M%S")
        with stem.with_suffix(".csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        stem.with_suffix(".json").write_text(
            json.dumps({"outcome": outcome, **asdict(self.cfg)}, indent=2))
        print(f"saved {stem.with_suffix('.csv')}")

    def _execution_step(self):
        pass
