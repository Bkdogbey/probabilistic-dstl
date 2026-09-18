"""Altitude behavior and on-demand plotting/animation diagnostics."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch
from PIL import Image

from pdstl.operators import Always
from pdstl.predicates import GreaterThan
from planning.runners import run_altitude_safety, setup_problem
from utils import load_config
from visualization.animation import animate_altitude_optimization
from visualization.planning import plot_altitude_safety


@pytest.fixture(scope="module")
def problem():
    s = setup_problem(load_config("configs/scenarios/altitude_safety.yaml"), device="cpu")
    spec = Always(GreaterThan(s.cfg["threshold"]), interval=[1, s.cfg["H"]])
    initial = s.planner.evaluate_controls(s.rollout, s.init_guess, spec=spec)
    return s, spec, initial


@pytest.fixture(scope="module")
def result():
    return run_altitude_safety(show=False, save=False)


def test_altitude_improves_and_replays(problem, result):
    s, spec, initial = problem
    assert result.controls.shape == (s.cfg["H"], 1)
    assert initial.hard_interval[0] < 0.5
    assert result.hard_interval[0] > 0.95
    assert result.smooth_lower > initial.smooth_lower
    assert result.controls.abs().max() <= s.dyn.u_max
    replay = s.planner.evaluate_controls(s.rollout, result.controls, spec=spec)
    assert replay.hard_interval == pytest.approx(result.hard_interval, abs=1e-6)


def test_altitude_three_panels(problem, result, tmp_path):
    s, _, initial = problem
    fig, axes = plot_altitude_safety(result, initial=initial, dt=s.cfg["dt"],
                                    threshold=s.cfg["threshold"], u_max=s.dyn.u_max,
                                    show=False, save_path=str(tmp_path / "altitude.png"))
    assert len(axes) == 3
    assert (tmp_path / "altitude.png").exists()
    plt.close(fig)


def test_animation_uses_post_update_frames_including_final(problem, tmp_path):
    s, spec, initial = problem
    s.planner.cfg["max_iters"] = 2
    frames = []
    result = s.planner.optimize_window(s.rollout, spec=spec, init_guess=s.init_guess,
                                       on_iteration=lambda k, p: frames.append(p))
    assert len(frames) == 2
    torch.testing.assert_close(frames[-1].controls, result.controls)
    filename = tmp_path / "altitude.gif"
    animate_altitude_optimization(frames, initial=initial, dt=s.cfg["dt"],
                                  threshold=s.cfg["threshold"], u_max=s.dyn.u_max,
                                  filename=str(filename), fps=2)
    with Image.open(filename) as image:
        assert image.n_frames >= 2
