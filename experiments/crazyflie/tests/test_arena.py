from dataclasses import replace

import numpy as np
import pytest

import arena
import config
from paths import Reference, figure8


def planned_xy(cfg):
    ref = Reference(figure8(cfg), cfg.path.speed)
    t = np.linspace(0.0, ref.duration, 2000)
    xy = np.array([ref.at(s)[0][:2] for s in t])
    return arena.sample(t, xy, cfg.dt)


def test_sample_takes_one_point_per_step():
    t = np.linspace(0.0, 1.0, 101)
    xy = np.column_stack([t, -t])
    out = arena.sample(t, xy, 0.2)
    assert np.allclose(out[:, 0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def test_default_figure8_satisfies_the_task():
    s = arena.score(config.load(), planned_xy(config.load()))
    assert s["robustness"] > 0
    assert s["clearance"] > 0


def test_obstacle_on_the_path_violates_the_task():
    cfg = config.load()
    on_path = config.Box("crossing", (0.45, 0.55), (-1.05, -0.95))
    cfg = replace(cfg, obstacles=cfg.obstacles + (on_path,))
    s = arena.score(cfg, planned_xy(cfg))
    assert s["robustness"] < 0
    assert s["clearance"] < 0


def test_goal_off_the_path_violates_the_task():
    cfg = config.load()
    cfg = replace(cfg, goal=config.Box("goal", (0.85, 0.95), (-0.15, -0.05)))
    assert arena.score(cfg, planned_xy(cfg))["robustness"] < 0


def test_obstacles_and_goal_are_read_from_yaml(tmp_path):
    f = tmp_path / "c.yml"
    f.write_text(
        "obstacles:\n  - {name: a, x: [0.1, 0.2], y: [-1.0, -0.9]}\n"
        "goal: {x: [0.4, 0.6], y: [-0.3, -0.1]}\n"
    )
    cfg = config.load(f)
    assert cfg.obstacles == (config.Box("a", (0.1, 0.2), (-1.0, -0.9)),)
    assert cfg.goal == config.Box("goal", (0.4, 0.6), (-0.3, -0.1))


def test_unknown_obstacle_key_is_rejected(tmp_path):
    f = tmp_path / "c.yml"
    f.write_text("obstacles:\n  - {name: a, x: [0, 1], y: [0, 1], z: 1}\n")
    with pytest.raises(TypeError):
        config.load(f)
