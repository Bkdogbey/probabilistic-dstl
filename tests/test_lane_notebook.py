"""The explanatory lane notebook remains valid and uses the real pipeline."""

import json
from pathlib import Path


def test_lane_pipeline_notebook_is_valid_and_uses_project_apis():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/lane_pipeline_demo.ipynb"
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    assert notebook["cells"]

    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "run_lane_change(" in source
    assert "formula.probability_interval(" in source
    assert "plot_lane_merge(" in source
    assert "animate_mpc(" in source
    assert "configs/scenarios/lane_change.yaml" in source
    assert "configs/scenarios/lane_merge.yaml" in source
