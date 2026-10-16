from dataclasses import replace

import numpy as np
import pytest

import config
from paths import Reference, figure8, transit
from tracking import (
    Abort, inside_workspace, settle, track, velocity_command,
)


def test_command_adds_feedforward_and_error():
    v = velocity_command([0.2, 0, 0], [0.1, 0, 0], gain=1.0, v_max=0.4)
    assert np.allclose(v, [0.3, 0, 0])


def test_command_is_clamped_and_keeps_direction():
    v = velocity_command([0, 0, 0], [3, 4, 0], gain=1.0, v_max=0.3)
    assert np.linalg.norm(v) == pytest.approx(0.3)
    assert np.allclose(v / 0.3, [0.6, 0.8, 0])


def test_workspace_and_margin():
    ws = config.Workspace()
    assert inside_workspace((0.5, -1.0, 0.3), ws)
    assert not inside_workspace((1.2, -1.0, 0.3), ws)
    assert inside_workspace((1.03, -1.0, 0.3), ws, margin=0.05)
    assert not inside_workspace((0.5, -1.0, 0.8), ws)


def test_default_config_path_fits_the_arena():
    cfg = config.load()
    pts = figure8(cfg)
    assert np.allclose(pts[0], pts[-1])  # closed
    assert np.allclose(pts[0][:2], [0.5, -1.8])  # bottom tip


def test_path_outside_the_arena_is_rejected():
    cfg = replace(config.Config(), path=config.PathSpec(height=2.4))
    with pytest.raises(ValueError, match="leaves the workspace"):
        figure8(cfg)


def test_unknown_config_key_is_rejected(tmp_path):
    f = tmp_path / "c.yml"
    f.write_text("radio_url: radio://0/80/2M/E7E7E7E7E7\n")
    with pytest.raises(TypeError):
        config.load(f)


def test_reference_runs_at_constant_speed():
    ref = Reference(figure8(config.Config()), speed=0.25)
    pos = np.array([ref.at(t)[0] for t in np.linspace(0, ref.duration, 200)])
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    dt = ref.duration / 199
    assert step.max() / dt == pytest.approx(0.25, rel=0.05)
    assert np.allclose(ref.at(ref.duration + 5)[0], ref.end)
    assert np.allclose(ref.at(ref.duration + 5)[1], 0)


class Plant:
    """Drone that follows the commanded velocity with a first-order lag."""

    def __init__(self, start, lag=0.3, dt=0.05):
        self.pos = np.array(start, float)
        self.vel = np.zeros(3)
        self.lag, self.dt, self.cmd, self.t = lag, dt, np.zeros(3), 0.0

    def send(self, v):
        self.cmd = np.asarray(v, float)

    def sleep(self, _):
        self.vel += (self.cmd - self.vel) * self.dt / self.lag
        self.pos = self.pos + self.vel * self.dt
        self.t += self.dt

    def state(self):
        return self.pos.copy(), 0.0


def fly(plant, ref, cfg, on_tick=None):
    track(ref, plant.state, plant.send, cfg, on_tick,
          clock=lambda: plant.t, sleep=plant.sleep)


def test_figure8_is_tracked_closely():
    cfg = config.load()
    ref = Reference(figure8(cfg), cfg.path.speed)
    plant = Plant(ref.points[0])
    errors, speeds = [], []

    def tick(t, pos, r, v):
        errors.append(np.linalg.norm(pos - r))
        speeds.append(np.linalg.norm(v))

    fly(plant, ref, cfg, tick)
    assert max(errors) < 0.15
    assert max(speeds) <= cfg.control.v_max + 1e-9
    assert np.linalg.norm(plant.pos - ref.end) < cfg.control.tol


def test_leaving_the_workspace_aborts():
    cfg = config.load()
    ref = Reference(figure8(cfg), cfg.path.speed)
    with pytest.raises(Abort, match="outside"):
        fly(Plant((1.5, -1.0, 0.3)), ref, cfg)


def test_stale_state_aborts():
    cfg = config.load()
    ref = Reference(figure8(cfg), cfg.path.speed)
    with pytest.raises(Abort, match="old"):
        track(ref, lambda: (ref.points[0], 1.0), lambda v: None, cfg)


def test_lagging_behind_the_reference_aborts():
    cfg = config.load()
    ref = Reference(figure8(cfg), cfg.path.speed)
    plant = Plant(ref.points[0])
    plant.send = lambda v: None  # commands are ignored
    with pytest.raises(Abort, match="from the reference"):
        fly(plant, ref, cfg)


class FakeDrone:
    """Position jumps around for `unsteady_s` seconds, then holds still."""

    def __init__(self, unsteady_s):
        self.now, self.unsteady_s = 0.0, unsteady_s
        self.cf = type("cf", (), {"param": type(
            "p", (), {"set_value": lambda *a: None})()})()

    def sleep(self, s):
        self.now += s

    def state(self):
        if self.now < self.unsteady_s:
            return np.array([0.5, -1.8 + 0.5 * np.sin(40 * self.now), 0.3]), 0.0
        return np.array([0.5, -1.8, 0.0]), 0.0


def test_settle_waits_for_the_estimate_to_converge():
    drone = FakeDrone(unsteady_s=5.0)
    pos = settle(drone, config.Flight(), clock=lambda: drone.now,
                 sleep=drone.sleep)
    assert np.allclose(pos, [0.5, -1.8, 0.0], atol=0.01)


def test_settle_gives_up_if_the_estimate_never_converges():
    drone = FakeDrone(unsteady_s=1e9)
    with pytest.raises(Abort, match="never settled"):
        settle(drone, config.Flight(), clock=lambda: drone.now,
               sleep=drone.sleep)


def test_transit_goes_straight_to_the_start():
    cfg = config.load()
    line = transit(cfg, (0.91, -2.0, 0.0), (0.5, -1.8, 0.3))
    assert np.allclose(line, [[0.91, -2.0, 0.3], [0.5, -1.8, 0.3]])


def test_transit_through_an_obstacle_is_refused():
    cfg = config.load()  # obstacle 'left' at x [0, 0.2], y [-1.1, -0.9]
    with pytest.raises(ValueError, match="left"):
        transit(cfg, (0.1, -0.5, 0.0), (0.1, -1.8, 0.3))


def test_transit_then_figure8_from_a_far_corner():
    cfg = config.load()
    path = Reference(figure8(cfg), cfg.path.speed)
    plant = Plant((0.91, -2.0, 0.3))
    fly(plant, Reference(transit(cfg, plant.pos, path.points[0]), 0.2), cfg)
    plant.t = 0.0
    fly(plant, path, cfg)
    assert np.linalg.norm(plant.pos - path.end) < cfg.control.tol


def test_transit_when_already_at_the_start():
    cfg = config.load()
    start = figure8(cfg)[0]
    plant = Plant(start)
    fly(plant, Reference(transit(cfg, start, start), 0.2), cfg)
    assert np.allclose(plant.pos, start)
