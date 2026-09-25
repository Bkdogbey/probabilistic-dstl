"""Saved figures, animations, notebooks, and main.py switches."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from planning.planner import IterationRecord
from planning.runners import run_altitude_safety, run_reach_avoid
from utils import load_config, skip_run
from visualization.live_plots import create_optimization_view
from visualization.planning import _finish


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
        ("narrow_passage", run_reach_avoid, "narrow_passage"),
    ):
        cfg = load_config(f"configs/scenarios/{scenario}.yaml")
        cfg["H"] = 4
        cfg["planner"]["max_iters"] = 2
        path = tmp_path / f"{scenario}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        kwargs = (
            {"live_optimization": True}
            if scenario == "altitude_safety"
            else {"live": True, "optimization_every": 1}
        )
        runner(str(path), show=False, save=True, **kwargs)
        for suffix in (".pt", ".png", ".pdf", ".gif"):
            assert (tmp_path / f"{stem}{suffix}").exists()
        with Image.open(tmp_path / f"{stem}.gif") as movie:
            assert movie.n_frames >= 2
            first = movie.convert("RGB").tobytes()
            movie.seek(movie.n_frames - 1)
            assert movie.convert("RGB").tobytes() != first
    progress = capsys.readouterr().out
    assert "Altitude safety window 1, iteration 1/2" in progress
    assert "Narrow Passage iteration 1/2" in progress
    assert "Reach-avoid window" not in progress
    plt.close("all")


def _code(notebook):
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_reach_avoid_notebook_runs_both_examples_through_the_runner():
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments/reach_avoid_examples.ipynb"
    notebook = json.loads(path.read_text())
    source = _code(notebook)

    assert notebook["nbformat"] == 4
    assert all(
        not cell.get("outputs") and cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "run_reach_avoid(" in source
    assert "example('narrow_passage')" in source
    assert "example('either_or')" in source
    assert "hard_interval" in source


def test_lane_notebook_walks_from_dynamics_through_both_executions():
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments/lane_change_merge_demo.ipynb"
    notebook = json.loads(path.read_text())
    source = _code(notebook)

    assert notebook["nbformat"] == 4
    assert all(
        not cell.get("outputs") and cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
    assert "configs/scenarios/lane_change.yaml" in source
    assert "configs/scenarios/lane_merge.yaml" in source
    assert "lane_rollout(" in source
    assert "lane_subformulas(" in source
    assert "optimize_window(" in source
    assert "run_lane_change(" in source
    assert "plot_lane_merge(" in source
    assert "animate_mpc(" in source
    assert "baseline" not in source.lower()


def test_project_blocks_can_be_run_or_skipped():
    visited = []
    with skip_run("run", "active") as check, check():
        visited.append("active")
    with skip_run("skip", "inactive") as check, check():
        visited.append("inactive")
    assert visited == ["active"]
