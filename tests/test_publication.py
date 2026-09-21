"""Headless publication and live views for a bounded reach-avoid MPC run."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from planning.runners import (
    build_environment,
    run_altitude_safety,
    run_mpc,
    run_reach_avoid,
)
from planning.planner import IterationRecord
from utils import load_config
from visualization.live_plots import (
    create_live_view,
    create_optimization_view,
)
from visualization.planning import _finish, plot_mpc_execution


def test_mpc_uses_shared_configuration_and_saves_publication_outputs(
    tmp_path,
    monkeypatch,
    capsys,
):
    import planning.runners as runners

    cfg = load_config("configs/scenarios/reach_avoid.yaml")
    cfg["H"] = 8
    cfg["planner"]["max_iters"] = 2
    cfg["mpc"]["max_steps"] = 1
    cfg["mpc"]["horizon"] = 8
    cfg["mpc"]["planner"]["max_iters"] = 2
    path = tmp_path / "reach_avoid.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(runners, "RESULTS_DIR", tmp_path)
    result = run_mpc(str(path), show=False, save=True)
    assert len(result.states) == 2
    assert result.applied_controls.shape == (1, 2)
    for suffix in (".pt", ".png", ".pdf", ".gif"):
        assert (tmp_path / f"mpc{suffix}").exists()
    with Image.open(tmp_path / "mpc.gif") as movie:
        assert movie.n_frames >= 2
        first = movie.convert("RGB").tobytes()
        movie.seek(movie.n_frames - 1)
        assert movie.convert("RGB").tobytes() != first
    env = build_environment(cfg)
    fig, axes = plot_mpc_execution(result, env, dt=cfg["dt"], show=False)
    for ax in axes:
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert len(labels) == len(set(labels))
    plt.close(fig)

    initial = result.states[0]
    live_fig, live_axes, observe = create_live_view(env, initial, dt=cfg["dt"])
    executed = next(
        line for line in live_axes[0].lines if line.get_label() == "Executed"
    )
    assert len(executed.get_xdata()) == 1
    assert executed.get_xdata()[0] == float(initial[0][0])
    assert executed.get_ydata()[0] == float(initial[0][1])
    observe(0, result.states[1], result.window_plans[0])
    assert len(executed.get_xdata()) == 2
    plt.close(live_fig)

    viewed = run_mpc(str(path), show=False, save=False, live=True)
    torch.testing.assert_close(
        viewed.applied_controls, result.applied_controls
    )
    for viewed_state, state in zip(viewed.states, result.states):
        torch.testing.assert_close(viewed_state[0], state[0])
        torch.testing.assert_close(viewed_state[1], state[1])
    assert [plan.hard_interval for plan in viewed.window_plans] == [
        plan.hard_interval for plan in result.window_plans
    ]
    optimized_view = run_mpc(
        str(path), show=False, save=False, live_optimization=True
    )
    torch.testing.assert_close(
        optimized_view.applied_controls, result.applied_controls
    )
    progress = capsys.readouterr().out
    assert "MPC window 1, iteration 1/2" in progress
    assert "MPC window 1 complete after 2 iterations" in progress
    plt.close("all")


def test_preview_happens_before_publication_files_are_saved(
    tmp_path, monkeypatch
):
    fig, _ = plt.subplots()
    calls = []
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "QtAgg")
    monkeypatch.setattr(plt, "show", lambda **kwargs: calls.append("show"))
    monkeypatch.setattr(
        fig,
        "savefig",
        lambda path, **kwargs: calls.append(path.suffix),
    )
    _finish(fig, tmp_path / "preview.png", show=True)
    assert calls == ["show", ".png", ".pdf"]


def test_optimizer_view_shows_control_updates():
    fig, axes, observe, _ = create_optimization_view(
        "test", max_iters=2, control_unit="m/s²"
    )
    observe(
        0,
        0,
        IterationRecord(
            torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            1.0,
            0.5,
            (0.4, 0.6),
            2.0,
        ),
    )
    observe(
        0,
        1,
        IterationRecord(
            torch.tensor([[0.2, 0.1], [0.5, 0.3]]),
            0.7,
            0.6,
            (0.5, 0.7),
            3.0,
        ),
    )
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), [1.0, 0.7])
    np.testing.assert_allclose(axes[2].lines[0].get_ydata(), [0.2, 0.5])
    assert axes[2].get_ylabel() == "current control [m/s²]"
    plt.close(fig)


def test_one_shot_animations_reveal_more_than_one_step(
    tmp_path, monkeypatch, capsys
):
    import planning.runners as runners

    monkeypatch.setattr(runners, "RESULTS_DIR", tmp_path)
    for scenario, runner, stem in (
        ("altitude_safety", run_altitude_safety, "altitude_safety"),
        ("reach_avoid", run_reach_avoid, "reach_avoid"),
    ):
        cfg = load_config(f"configs/scenarios/{scenario}.yaml")
        cfg["H"] = 4
        cfg["planner"]["max_iters"] = 2
        path = tmp_path / f"{scenario}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        runner(str(path), show=False, save=True, live_optimization=True)
        for suffix in (".pt", ".png", ".pdf", ".gif"):
            assert (tmp_path / f"{stem}{suffix}").exists()
        with Image.open(tmp_path / f"{stem}.gif") as movie:
            assert movie.n_frames >= 2
    progress = capsys.readouterr().out
    assert "Altitude safety window 1, iteration 1/2" in progress
    assert "Reach-avoid window 1, iteration 1/2" in progress
    plt.close("all")
