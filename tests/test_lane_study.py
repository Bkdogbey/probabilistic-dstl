"""The optional paired study stays reproducible and isolated from core runners."""

from pathlib import Path

import pytest
import torch
import yaml

from baselines.lane_reporting import wilson_interval
from baselines.lane_study import (
    generate_trial_inputs,
    run_study,
    scaled_config,
)
from planning.runners import setup_problem
from utils import load_config


def test_uncertainty_scaling_and_paired_random_draws():
    base = load_config("configs/scenarios/lane_change.yaml")
    low = setup_problem(
        scaled_config(base, 0.5), device="cpu", with_environment=True
    )
    high = setup_problem(
        scaled_config(base, 2.0), device="cpu", with_environment=True
    )
    low_initial, low_ego, low_traffic = generate_trial_inputs(low, 17)
    high_initial, high_ego, high_traffic = generate_trial_inputs(high, 17)
    base_mean = torch.tensor(base["x0_mean"])
    torch.testing.assert_close(
        high_initial[0] - base_mean, 4 * (low_initial[0] - base_mean)
    )
    torch.testing.assert_close(high_ego, 4 * low_ego)
    torch.testing.assert_close(high_traffic, 4 * low_traffic)


def test_wilson_interval_contains_the_observed_rate():
    low, high = wilson_interval(30, 50)
    assert low < 0.6 < high
    assert (low, high) == pytest.approx((0.4618, 0.7239), abs=1e-3)


def test_tiny_study_writes_isolated_outputs(tmp_path):
    scenario = load_config("configs/scenarios/lane_change.yaml")
    scenario["H"] = 6
    scenario["T_SIM"] = 1
    scenario["task"]["start_window_seconds"] = [0.2, 0.6]
    scenario["task"]["dwell_seconds"] = 0.2
    scenario["planner"]["max_iters"] = 1
    scenario_path = tmp_path / "lane_change.yaml"
    scenario_path.write_text(yaml.safe_dump(scenario))
    output = tmp_path / "study"
    study_path = tmp_path / "study.yaml"
    study_path.write_text(
        yaml.safe_dump(
            {
                "scenarios": [str(scenario_path)],
                "trials": 1,
                "uncertainty_factors": [1.0],
                "seed": 4,
                "device": "cpu",
                "output_dir": str(output),
            }
        )
    )
    trials, windows, summary = run_study(str(study_path))
    assert {row["method"] for row in trials} == {"deterministic_stl", "pdstl"}
    assert len(windows) == 2
    assert summary
    for name in (
        "trials.csv",
        "windows.csv",
        "summary.csv",
        "metrics.csv",
        "metadata.json",
        "lane_change_rates.png",
        "lane_change_rates.pdf",
    ):
        assert (output / name).exists()


def test_core_and_main_do_not_import_the_optional_study():
    root = Path(__file__).resolve().parents[1]
    for relative in ("src/main.py", "src/planning/runners.py"):
        assert "lane_study" not in (root / relative).read_text()


def test_packaged_config_loads_outside_the_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config("configs/planning.yaml")["max_iters"] > 0


def test_configs_have_one_editable_source_tree():
    root = Path(__file__).resolve().parents[1]
    assert (root / "configs/experiments/lane_baseline.yaml").is_file()
    assert not (root / "src/pdstl/resources").exists()
