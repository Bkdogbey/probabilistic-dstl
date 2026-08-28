"""Tests for the three real offline example pipelines in src/main.py."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

import main as main_module
from main import (
    DEFAULT_CONFIG,
    run_always_example,
    run_boolean_example,
    run_eventually_example,
)
from utils import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _always_hand_calculation(atomic_bounds, interval):
    a, b = interval
    outputs = []
    for anchor in range(atomic_bounds.shape[1] - b):
        window = atomic_bounds[:, anchor + a : anchor + b + 1, :]
        lower = torch.clamp(window[..., 0].sum(dim=1) - (b - a), min=0.0)
        upper = window[..., 1].amin(dim=1)
        outputs.append(torch.stack((lower, upper), dim=-1))
    return torch.stack(outputs, dim=1)


def _eventually_hand_calculation(atomic_bounds, interval):
    a, b = interval
    outputs = []
    for anchor in range(atomic_bounds.shape[1] - b):
        window = atomic_bounds[:, anchor + a : anchor + b + 1, :]
        lower = window[..., 0].amax(dim=1)
        upper = torch.clamp(window[..., 1].sum(dim=1), max=1.0)
        outputs.append(torch.stack((lower, upper), dim=-1))
    return torch.stack(outputs, dim=1)


def test_boolean_runner_returns_explicit_expected_bounds():
    not_a, and_ab, or_ab = run_boolean_example()
    dtype = not_a.dtype

    torch.testing.assert_close(not_a, torch.tensor([[[0.10, 0.40]]], dtype=dtype))
    torch.testing.assert_close(and_ab, torch.tensor([[[0.30, 0.90]]], dtype=dtype))
    torch.testing.assert_close(or_ab, torch.tensor([[[0.70, 1.00]]], dtype=dtype))


def test_always_default_interval_matches_every_hand_calculated_output():
    _, atomic, temporal, figure = run_always_example(50.0, (0, 2), show=False)

    assert temporal.shape == (1, 5, 2)
    torch.testing.assert_close(temporal, _always_hand_calculation(atomic, (0, 2)))
    plt.close(figure)


def test_eventually_default_interval_matches_every_hand_calculated_output():
    _, atomic, temporal, figure = run_eventually_example(
        55.0, (0, 2), show=False
    )

    assert temporal.shape == (1, 5, 2)
    torch.testing.assert_close(
        temporal,
        _eventually_hand_calculation(atomic, (0, 2)),
    )
    plt.close(figure)


def test_always_interval_changes_output_count():
    _, atomic, temporal, figure = run_always_example(50.0, (0, 1), show=False)

    assert temporal.shape == (1, 6, 2)
    torch.testing.assert_close(temporal, _always_hand_calculation(atomic, (0, 1)))
    assert len(figure.axes[2].lines[0].get_xdata()) == 6
    assert len(figure.axes[2].lines[1].get_xdata()) == 6
    plt.close(figure)


def test_eventually_nonzero_lower_endpoint_changes_output_count(capsys):
    _, atomic, temporal, figure = run_eventually_example(
        55.0, (1, 3), show=False
    )

    assert temporal.shape == (1, 4, 2)
    torch.testing.assert_close(
        temporal,
        _eventually_hand_calculation(atomic, (1, 3)),
    )
    assert len(figure.axes[2].lines[0].get_xdata()) == 4
    assert len(figure.axes[2].lines[1].get_xdata()) == 4
    assert "k=0, window=[1,3]" in capsys.readouterr().out
    plt.close(figure)


def test_importing_main_has_no_execution_side_effects(capsys):
    importlib.reload(main_module)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_default_yaml_config_contains_each_independent_example():
    config = load_config(DEFAULT_CONFIG)

    assert set(config["experiments"]) == {
        "boolean",
        "always",
        "eventually",
        "mission",
        "sliding_always",
        "streaming_always",
        "streaming_animation",
        "streaming_eventually",
        "until",
    }
    assert config["experiments"]["always"]["interval"] == [0, 1]
    assert config["experiments"]["streaming_always"]["interval"] == [0, 2]
    assert config["experiments"]["streaming_animation"]["run"] is False


def test_real_main_pipeline_runs_default_yaml_examples(capsys):
    main_module.main(show=False)

    output = capsys.readouterr().out
    for expected in (
        "A AND B",
        "Offline sliding Always",
        "Streaming Always",
        "Streaming Eventually",
        "Mission = Always",
        "Until mission",
        "Online step() outputs match",
        "mission outputs match the offline graph",
        "Streaming Until outputs match",
        "t=0",
        "k=0",
    ):
        assert expected in output
    plt.close("all")


def test_yaml_config_can_select_one_example(tmp_path, capsys):
    config = load_config(DEFAULT_CONFIG)
    for settings in config["experiments"].values():
        settings["run"] = False
    config["experiments"]["boolean"]["run"] = True
    config_path = tmp_path / "examples.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    main_module.main(config_path, show=False)

    output = capsys.readouterr().out
    assert "A AND B" in output
    assert "Offline Always" not in output
    assert "Streaming Always" not in output


def test_executable_entry_point_runs_real_pipeline(tmp_path):
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(tmp_path)

    completed = subprocess.run(
        [sys.executable, "src/main.py"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "A AND B" in completed.stdout
    assert "Offline sliding Always" in completed.stdout
    assert "Streaming Always" in completed.stdout
    assert "Streaming Eventually" in completed.stdout
