"""Tests for the demonstration runners and the src/main.py pipeline."""

import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from experiments.offline import (
    run_always_example,
    run_boolean_example,
    run_eventually_example,
)
from utils import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/examples.yml"


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


def test_default_yaml_config_contains_parameters_for_three_configured_examples():
    config = load_config(DEFAULT_CONFIG)

    assert set(config["experiments"]) == {
        "offline_temporal",
        "streaming",
        "until",
    }
    offline = config["experiments"]["offline_temporal"]
    assert set(offline) == {"always", "eventually"}
    assert offline["always"] == {"threshold": 50.0, "interval": [0, 1]}
    assert offline["eventually"] == {"threshold": 55.0, "interval": [0, 1]}
    assert isinstance(offline["always"]["threshold"], float)
    assert isinstance(offline["eventually"]["threshold"], float)
    assert all(isinstance(value, int) for value in offline["always"]["interval"])
    assert all(
        isinstance(value, int) for value in offline["eventually"]["interval"]
    )

    streaming = config["experiments"]["streaming"]
    assert streaming == {
        "interval": [0, 2],
        "animate": True,
        "frame_interval_ms": 900,
        "repeat": True,
    }
    assert isinstance(streaming["animate"], bool)
    assert isinstance(streaming["frame_interval_ms"], int)
    assert isinstance(streaming["repeat"], bool)
    assert all(isinstance(value, int) for value in streaming["interval"])
    assert config["experiments"]["until"] == {"interval": [1, 2]}
    assert all(
        isinstance(value, int)
        for value in config["experiments"]["until"]["interval"]
    )


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
    assert "Always[0,1](altitude_at_least_50)" not in completed.stdout
    assert "Online step() outputs match" not in completed.stdout
    assert "Until mission" not in completed.stdout
    assert completed.stderr.count("Skipping the block") == 4
