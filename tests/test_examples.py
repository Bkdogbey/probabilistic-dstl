"""Offline examples exercise the same belief-to-pdSTL pipeline as main.py."""

import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr
import torch
import yaml

from models.dynamics import (
    create_gaussian_belief_trajectory,
    linear_system,
    piecewise_signal,
    sinusoidial_input,
)
from pdstl.operators import Always, Eventually, GreaterThan
from utils import to_steps
from visualization.temporal import plot_temporal_example

ROOT = Path(__file__).resolve().parents[1]


def _evaluate(name):
    config = yaml.safe_load((ROOT / "configs/examples.yaml").read_text())
    example = config[name]
    if name == "piecewise_always":
        time, mean, variance = piecewise_signal(example["signal"])
        interval = example["interval_steps"]
    else:
        model = example["model"]
        time = np.linspace(0.0, model["t_end"], model["n_steps"])
        mean, variance = linear_system(
            **{key: model[key] for key in ("a", "b", "g", "q", "mu", "P")},
            t=time,
            control_func=sinusoidial_input,
        )
        interval = to_steps(example["interval_sec"], time)
    beliefs = create_gaussian_belief_trajectory(
        mean, variance, example["confidence_level"], dtype=torch.float64
    )
    predicate = GreaterThan(example["threshold"])
    formula = Eventually(predicate, interval) if name == "eventually" else Always(predicate, interval)
    return example, time, mean, variance, interval, predicate, formula, predicate(beliefs), formula(beliefs)


def test_offline_examples_match_endpointwise_cdf_reductions():
    for name, reduction in (("always", np.min), ("piecewise_always", np.min), ("eventually", np.max)):
        example, time, mean, variance, interval, predicate, formula, atomic, temporal = _evaluate(name)
        sigma = np.sqrt(variance)
        z = (mean - predicate.threshold) / sigma
        lower, upper = ndtr(z - example["confidence_level"]), ndtr(z + example["confidence_level"])
        a, b = interval
        expected = np.array([
            [reduction(lower[i + a : i + b + 1]), reduction(upper[i + a : i + b + 1])]
            for i in range(len(time) - b)
        ])
        np.testing.assert_allclose(atomic[0].numpy(), np.stack((lower, upper), axis=-1), atol=1e-12)
        np.testing.assert_allclose(temporal[0].numpy(), expected, atol=1e-12)
        assert temporal.shape == (1, len(time) - b, 2)

        figure = plot_temporal_example(
            time, mean, variance, example["confidence_level"], atomic, temporal,
            predicate, formula, interval, show=False,
        )
        assert len(figure.axes) == 3
        figure.canvas.draw()
        plt.close(figure)


def test_gaussian_trajectory_factory_normalizes_scalar_and_batched_traces():
    scalar = create_gaussian_belief_trajectory([1.0, 2.0], [0.25, 1.0], 1.5)
    assert len(scalar) == 2
    assert scalar[0].mean.shape == (1, 1)
    batched = create_gaussian_belief_trajectory(
        torch.zeros(2, 3, 1), torch.ones(2, 3, 1), 1.0
    )
    assert len(batched) == 3
    assert batched[0].mean.shape == (2, 1)


def test_reusable_modules_do_not_run_examples():
    result = subprocess.run(
        [sys.executable, "-c", "import models.dynamics; import visualization.temporal"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_main_runs_only_the_three_offline_examples(tmp_path):
    config = yaml.safe_load((ROOT / "configs/examples.yaml").read_text())
    config["show_plots"] = False
    config_path = tmp_path / "examples.yaml"
    config_path.write_text(yaml.safe_dump(config))
    script = tmp_path / "main.py"
    script.write_text((ROOT / "src/main.py").read_text().replace('"configs/examples.yaml"', repr(str(config_path))))
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Always" in result.stdout
    assert "Piecewise Always" in result.stdout
    assert "Eventually" in result.stdout
