"""Scalar MPC and rectangular runner integration, using structured results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch
import yaml

from planning.examples import run_end_to_end_mpc_reach
from planning.runners import run_mpc, setup_problem
from utils import load_config
from visualization.robustness import plot_mpc_reach


@pytest.fixture(scope="module")
def result():
    return run_end_to_end_mpc_reach(show=False, save=False)


def test_scalar_mpc_reaches_target_and_executes_first_controls(result, tmp_path):
    cfg = load_config("configs/examples.yaml")["end_to_end_mpc_reach"]
    assert len(result.window_plans) > 1
    assert result.stopped_reason == "goal_reached"
    assert result.states[-1][0][cfg["dim"]] >= cfg["threshold"]
    assert len(result.states) == len(result.applied_controls) + 1
    for control, plan in zip(result.applied_controls, result.window_plans):
        torch.testing.assert_close(control, plan.controls[0])
    assert torch.isfinite(result.applied_controls).all()
    assert result.applied_controls.abs().max() <= cfg["u_max"]
    fig, axes = plot_mpc_reach(cfg["dt"], result, cfg["threshold"],
                              save_path=str(tmp_path / "mpc.png"), show=False)
    assert len(axes) == 3
    plt.close(fig)
    path = tmp_path / "mpc.pt"
    torch.save(result, path)
    loaded = torch.load(path, weights_only=False)
    torch.testing.assert_close(loaded.applied_controls, result.applied_controls)


def test_rectangular_mpc_runner_runs_one_window_with_same_schema(tmp_path):
    cfg = load_config("configs/scenarios/mpc.yaml")
    cfg["MAX_STEPS"] = 1
    cfg["planner"]["max_iters"] = 2
    s = setup_problem(cfg, device="cpu", with_environment=True)
    assert len(s.env.by_role("obstacle")) == 2
    path = tmp_path / "mpc.yaml"
    path.write_text(yaml.safe_dump(cfg))
    result = run_mpc(str(path), show=False, save=False)
    assert len(result.window_plans) == 1
    assert len(result.states) == 2
    torch.testing.assert_close(result.applied_controls[0], result.window_plans[0].controls[0])
