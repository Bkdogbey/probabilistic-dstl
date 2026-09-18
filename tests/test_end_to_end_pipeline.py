"""The scalar one-shot demonstration uses the canonical result and score."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch

from planning.examples import run_end_to_end_reach
from planning.runners import setup_problem
from pdstl.predicates import GreaterThan
from pdstl.operators import Eventually
from utils import load_config
from visualization.robustness import plot_end_to_end_plans


@pytest.fixture(scope="module")
def result():
    return run_end_to_end_reach(show=False, save=False)


def test_scalar_pipeline_improves_replays_and_has_valid_beliefs(result, tmp_path):
    cfg = load_config("configs/examples.yaml")["end_to_end_reach"]
    s = setup_problem(cfg, device="cpu")
    atom = GreaterThan(cfg["threshold"], dim=cfg["dim"])
    spec = Eventually(atom, interval=[0, cfg["H"]])
    initial = s.planner.evaluate_controls(s.rollout, torch.zeros(cfg["H"], 2), spec=spec)
    assert result.hard_interval[0] > initial.hard_interval[0] + 0.5
    replay = s.planner.evaluate_controls(s.rollout, result.controls, spec=spec)
    assert replay.hard_interval == pytest.approx(result.hard_interval, abs=1e-6)
    assert torch.isfinite(result.rollout.aux["cov_trace"]).all()
    assert torch.linalg.eigvalsh(result.rollout.aux["cov_trace"]).min() >= 0
    fig, axes = plot_end_to_end_plans(result, initial, atom, cfg, show=False,
                                     save_path=str(tmp_path / "reach.png"))
    assert len(axes) == 2
    plt.close(fig)
