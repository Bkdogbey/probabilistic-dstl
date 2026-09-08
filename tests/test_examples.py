"""Offline examples exercise the same belief-to-pdSTL pipeline as main.py."""

import ast
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.special import ndtr
import torch
import yaml

from offline import evaluate_offline_example
from models.dynamics import (
    create_gaussian_belief_trajectory,
    create_signal_trace,
)
from pdstl.operators import Always, GreaterThan
from utils import to_steps
from visualization.temporal import plot_temporal_example


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return yaml.safe_load((ROOT / "configs/examples.yaml").read_text())


def _continuous(name):
    return evaluate_offline_example(name.title(), _config()[name])


@pytest.mark.parametrize("name,reduction", [("always", np.min), ("eventually", np.max)])
def test_continuous_examples_match_endpointwise_cdf_reductions(name, reduction):
    result = _continuous(name)
    sigma = np.sqrt(result.variance)
    z = (result.mean - result.predicate.threshold) / sigma
    lower = ndtr(z - result.sigma_multiplier)
    upper = ndtr(z + result.sigma_multiplier)
    a, b = result.formula.interval
    expected = np.array(
        [
            [
                reduction(lower[i + a : i + b + 1]),
                reduction(upper[i + a : i + b + 1]),
            ]
            for i in range(len(result.time) - b)
        ]
    )

    np.testing.assert_allclose(
        result.atomic_trace[0].numpy(),
        np.stack((lower, upper), axis=-1),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.temporal_trace[0].numpy(), expected, atol=1e-12
    )
    assert result.temporal_trace.shape == (1, len(result.time) - b, 2)

    figure = plot_temporal_example(result, show=False)
    assert len(figure.axes) == 3
    assert "pdSTL stochastic robustness" == figure.axes[-1].get_ylabel()
    figure.canvas.draw()
    plt.close(figure)


def test_continuous_examples_retain_representative_outputs():
    always = _continuous("always")
    eventually = _continuous("eventually")

    assert always.atomic_trace.shape == (1, 100, 2)
    assert always.temporal_trace.shape == (1, 80, 2)
    np.testing.assert_allclose(
        always.atomic_trace[0, 0].numpy(),
        [0.1586552539, 0.8413447461],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        always.temporal_trace[0, 0].numpy(),
        [0.2397146171, 0.9019562671],
        atol=1e-10,
    )

    assert eventually.atomic_trace.shape == (1, 100, 2)
    assert eventually.temporal_trace.shape == (1, 90, 2)
    np.testing.assert_allclose(
        eventually.temporal_trace[0, 0].numpy(),
        [0.8250194548, 0.9983304600],
        atol=1e-10,
    )


def test_piecewise_always_remains_a_focused_unit_example():
    example = _config()["nested"]
    _, mean, variance = create_signal_trace(example["signal"])
    beliefs = create_gaussian_belief_trajectory(
        mean, variance, sigma_multiplier=example["sigma_multiplier"]
    )
    predicate = GreaterThan(example["threshold"])
    interval = example["formula"]["child"]["interval_steps"]
    formula = Always(predicate, interval=interval)
    atomic = predicate(beliefs)[0].numpy()
    got = formula(beliefs)[0].numpy()

    expected = np.minimum(atomic[:-1], atomic[1:])
    np.testing.assert_allclose(got, expected)


def test_nested_example_values_length_and_four_panel_plot():
    result = evaluate_offline_example("Nested", _config()["nested"])
    atomic = result.atomic_trace
    inner = result.inner_trace
    temporal = result.temporal_trace

    expected_inner = np.minimum(atomic[0, :-1].numpy(), atomic[0, 1:].numpy())
    expected_final = np.maximum(expected_inner[:-1], expected_inner[1:])
    np.testing.assert_allclose(inner[0].numpy(), expected_inner)
    np.testing.assert_allclose(temporal[0].numpy(), expected_final)
    assert atomic.shape == (1, 7, 2)
    assert inner.shape == (1, 6, 2)
    assert temporal.shape == (1, 5, 2)

    figure = plot_temporal_example(result, show=False)
    assert len(figure.axes) == 4
    assert figure.axes[2].lines[0].get_xdata().tolist() == result.time[:6].tolist()
    assert figure.axes[3].lines[0].get_xdata().tolist() == result.time[:5].tolist()
    figure.canvas.draw()
    plt.close(figure)


def test_signal_selection_is_independent_of_the_formula():
    config = _config()

    always_piecewise = deepcopy(config["always"])
    always_piecewise["signal"] = config["nested"]["signal"]
    piecewise_result = evaluate_offline_example("Always", always_piecewise)
    assert piecewise_result.signal_type == "piecewise"
    assert piecewise_result.temporal_trace.shape == (1, 5, 2)

    nested_linear = deepcopy(config["nested"])
    nested_linear["signal"] = config["always"]["signal"]
    linear_result = evaluate_offline_example("Nested", nested_linear)
    assert linear_result.signal_type == "linear"
    assert linear_result.temporal_trace.shape == (1, 98, 2)


def test_gaussian_trajectory_factory_preserves_supported_shapes_and_types():
    scalar_mean = np.array([1.0, 2.0], dtype=np.float64)
    scalar = create_gaussian_belief_trajectory(
        scalar_mean, np.array([0.25, 1.0]), sigma_multiplier=1.5
    )
    assert scalar[0].mean.shape == (1, 1)
    assert scalar[0].mean.dtype == torch.float64

    vector = create_gaussian_belief_trajectory(
        np.zeros((3, 2)), np.ones((3, 2, 2)), sigma_multiplier=1.0
    )
    assert len(vector) == 3 and vector[0].var.shape == (1, 2, 2)

    mean = torch.zeros(2, 3, 2, dtype=torch.float64, requires_grad=True)
    variance = torch.ones(2, 3, 2, dtype=torch.float64)
    batched = create_gaussian_belief_trajectory(
        mean, variance, sigma_multiplier=0.0
    )
    assert len(batched) == 3 and batched[0].mean.shape == (2, 2)
    assert batched[0].mean.device == mean.device
    GreaterThan(0.0)(batched).sum().backward()
    assert mean.grad is not None and torch.isfinite(mean.grad).all()


@pytest.mark.parametrize(
    "mean,variance",
    [
        (np.zeros(3), np.zeros((3, 1))),
        (np.zeros((3, 2)), np.zeros((3, 3))),
        (np.zeros((3, 2)), np.zeros((3, 2, 3))),
        (np.zeros((2, 3, 2)), np.zeros((2, 3, 3))),
        (np.zeros((2, 3, 2)), np.zeros((2, 3, 2, 3))),
    ],
)
def test_gaussian_trajectory_factory_rejects_inexact_shapes(mean, variance):
    with pytest.raises(ValueError, match="exactly match"):
        create_gaussian_belief_trajectory(mean, variance, sigma_multiplier=1.0)


@pytest.mark.parametrize(
    "time,message",
    [
        ([0.0], "at least two"),
        ([0.0, 1.0, 0.5], "strictly increasing"),
        ([0.0, 1.0, 2.1], "uniformly spaced"),
    ],
)
def test_to_steps_rejects_invalid_time_grids(time, message):
    with pytest.raises(ValueError, match=message):
        to_steps([0.0, 1.0], time)


def test_examples_configuration_contains_only_the_three_blocks():
    config = _config()
    assert set(config) == {"show_plots", "always", "eventually", "nested"}
    assert all("enabled" not in config[name] for name in config if name != "show_plots")
    assert config["always"]["signal"]["type"] == "linear"
    assert config["eventually"]["signal"]["type"] == "linear"
    assert config["nested"]["signal"]["type"] == "piecewise"


def test_reusable_modules_do_not_run_examples_or_eagerly_import_planning():
    code = (
        "import sys; import models.dynamics; import offline; "
        "assert 'visualization.planning' not in sys.modules; "
        "assert 'visualization.live_plots' not in sys.modules; "
        "assert 'visualization.animation' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_main_is_direct_and_runs_exactly_three_literal_blocks(tmp_path):
    source = (ROOT / "src/main.py").read_text()
    tree = ast.parse(source)
    assert source.count('with skip_run("run",') == 3
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body)
    assert not any(isinstance(node, ast.If) for node in tree.body)

    config = _config()
    config["show_plots"] = False
    config_path = tmp_path / "examples.yaml"
    config_path.write_text(yaml.safe_dump(config))
    script = tmp_path / "main.py"
    script.write_text(
        source.replace('"configs/examples.yaml"', repr(str(config_path)))
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Running the block") == 3
    assert "\nAlways\n" in result.stdout
    assert "\nEventually\n" in result.stdout
    assert "\nNested\n" in result.stdout
    assert "Piecewise Always" not in result.stdout
