"""Experiment notebooks are output-free and use the production pipelines."""

import json
from pathlib import Path


def _code(notebook):
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_reach_avoid_notebook_exposes_one_shot_certificate_workflow():
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments/reach_avoid_demo.ipynb"
    notebook = json.loads(path.read_text())
    source = _code(notebook)

    assert notebook["nbformat"] == 4
    assert all(
        not cell.get("outputs") and cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "run_reach_avoid(" in source
    assert "plot_reach_avoid(" in source
    assert "animate_reach_avoid(" in source
    assert "hard_interval[0]" in source
    assert "visit" not in source.lower()
    assert "run_mpc" not in source


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
