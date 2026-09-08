"""Offline examples exercise the same belief-to-pdSTL pipeline as main.py."""

import ast
import os
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import yaml

from models.dynamics import IntervalBelief, create_interval_belief_trajectory
from pdstl.operators import Always, Eventually, GreaterThan, LessThan
from utils import to_steps
from visualization.temporal import plot_temporal_example, print_temporal_results


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return yaml.safe_load((ROOT / "configs/examples.yaml").read_text())


# ---------------------------------------------------------------------------
# IntervalBelief input validation
# ---------------------------------------------------------------------------


def test_interval_belief_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="matching shapes"):
        IntervalBelief(torch.zeros(1, 2), torch.zeros(1, 3))


def test_interval_belief_rejects_non_2d():
    with pytest.raises(ValueError, match=r"\[B,D\]"):
        IntervalBelief(torch.zeros(3), torch.zeros(3))


def test_interval_belief_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        IntervalBelief(torch.tensor([[float("nan")]]), torch.tensor([[1.0]]))


def test_interval_belief_rejects_lower_greater_than_upper():
    with pytest.raises(ValueError, match="lower <= upper"):
        IntervalBelief(torch.tensor([[2.0]]), torch.tensor([[1.0]]))


def test_interval_belief_rejects_invalid_predicate_dimension():
    belief = IntervalBelief(torch.zeros(1, 1), torch.ones(1, 1))
    predicate = GreaterThan(0.5, dim=5)
    with pytest.raises(ValueError, match="outside the state"):
        belief.probability_bounds(predicate)


# ---------------------------------------------------------------------------
# Inclusive threshold semantics for GreaterThan / LessThan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lower,upper,expected",
    [
        (5.0, 5.0, [1.0, 1.0]),  # x_lower == threshold -> guaranteed
        (6.0, 8.0, [1.0, 1.0]),  # entirely above -> guaranteed
        (3.0, 5.0, [0.0, 1.0]),  # x_upper == threshold -> not violated
        (3.0, 7.0, [0.0, 1.0]),  # crossing
        (1.0, 4.9, [0.0, 0.0]),  # entirely below -> violated
    ],
)
def test_greater_than_inclusive_bounds(lower, upper, expected):
    belief = IntervalBelief(torch.tensor([[lower]]), torch.tensor([[upper]]))
    got = belief.probability_bounds(GreaterThan(5.0))
    torch.testing.assert_close(got[0], torch.tensor(expected))


@pytest.mark.parametrize(
    "lower,upper,expected",
    [
        (5.0, 5.0, [1.0, 1.0]),  # x_upper == threshold -> guaranteed
        (1.0, 4.0, [1.0, 1.0]),  # entirely below -> guaranteed
        (5.0, 7.0, [0.0, 1.0]),  # x_lower == threshold -> not violated
        (3.0, 7.0, [0.0, 1.0]),  # crossing
        (5.1, 8.0, [0.0, 0.0]),  # entirely above -> violated
    ],
)
def test_less_than_inclusive_bounds(lower, upper, expected):
    belief = IntervalBelief(torch.tensor([[lower]]), torch.tensor([[upper]]))
    got = belief.probability_bounds(LessThan(5.0))
    torch.testing.assert_close(got[0], torch.tensor(expected))


# ---------------------------------------------------------------------------
# create_interval_belief_trajectory
# ---------------------------------------------------------------------------


def test_interval_trajectory_factory_builds_one_belief_per_step():
    trajectory = create_interval_belief_trajectory([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
    assert len(trajectory) == 3
    assert trajectory[0].lower.shape == (1, 1)
    assert trajectory[0].upper.shape == (1, 1)


def test_interval_trajectory_factory_rejects_mismatched_traces():
    with pytest.raises(ValueError, match="matching shapes"):
        create_interval_belief_trajectory([1.0, 2.0], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Configured examples: hand-computed atomic/temporal traces
# ---------------------------------------------------------------------------


def test_always_example_matches_hand_computed_traces():
    config = _config()["always"]
    lower, upper = config["signal"]["lower"], config["signal"]["upper"]
    time = np.arange(len(lower))
    beliefs = create_interval_belief_trajectory(lower, upper)
    predicate = GreaterThan(config["threshold"])
    formula = Always(predicate, interval=config["interval_steps"])

    atomic = predicate(beliefs)
    temporal = formula(beliefs, scale=-1)

    expected_atomic = torch.tensor(
        [[1, 1], [1, 1], [1, 1], [0, 1], [0, 0], [1, 1], [1, 1], [1, 1]],
        dtype=atomic.dtype,
    )
    expected_temporal = torch.tensor(
        [[1, 1], [0, 1], [0, 0], [0, 0], [0, 0], [1, 1]], dtype=temporal.dtype
    )

    torch.testing.assert_close(atomic[0], expected_atomic)
    torch.testing.assert_close(temporal[0], expected_temporal)
    assert atomic.shape == (1, 8, 2)
    assert temporal.shape == (1, 6, 2)

    figure = plot_temporal_example(
        "Always", time, lower, upper, config["threshold"], atomic, temporal,
        show=False,
    )
    assert len(figure.axes) == 3
    assert figure.axes[-1].get_ylabel() == "pdSTL stochastic robustness"
    figure.canvas.draw()
    plt.close(figure)


def test_eventually_example_matches_hand_computed_traces():
    config = _config()["eventually"]
    lower, upper = config["signal"]["lower"], config["signal"]["upper"]
    time = np.arange(len(lower))
    beliefs = create_interval_belief_trajectory(lower, upper)
    predicate = GreaterThan(config["threshold"])
    formula = Eventually(predicate, interval=config["interval_steps"])

    atomic = predicate(beliefs)
    temporal = formula(beliefs, scale=-1)

    expected_atomic = torch.tensor(
        [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1], [0, 0], [0, 0]],
        dtype=atomic.dtype,
    )
    expected_temporal = torch.tensor(
        [[0, 0], [0, 1], [0, 1], [1, 1], [1, 1], [0, 0]], dtype=temporal.dtype
    )

    torch.testing.assert_close(atomic[0], expected_atomic)
    torch.testing.assert_close(temporal[0], expected_temporal)
    assert atomic.shape == (1, 7, 2)
    assert temporal.shape == (1, 6, 2)

    figure = plot_temporal_example(
        "Eventually", time, lower, upper, config["threshold"], atomic, temporal,
        show=False,
    )
    assert len(figure.axes) == 3
    assert figure.axes[-1].get_ylabel() == "pdSTL stochastic robustness"
    figure.canvas.draw()
    plt.close(figure)


def test_nested_example_matches_hand_computed_traces():
    config = _config()["nested"]
    lower, upper = config["signal"]["lower"], config["signal"]["upper"]
    time = np.arange(len(lower))
    beliefs = create_interval_belief_trajectory(lower, upper)
    predicate = GreaterThan(config["threshold"])
    inner_formula = Always(predicate, interval=config["always_interval_steps"])
    formula = Eventually(inner_formula, interval=config["eventually_interval_steps"])

    atomic = predicate(beliefs)
    inner = inner_formula(beliefs, scale=-1)
    temporal = formula(beliefs, scale=-1)

    expected_atomic = torch.tensor(
        [[0, 0], [1, 1], [1, 1], [0, 0], [0, 1], [1, 1]], dtype=atomic.dtype
    )
    expected_inner = torch.tensor(
        [[0, 0], [1, 1], [0, 0], [0, 0], [0, 1]], dtype=inner.dtype
    )
    expected_temporal = torch.tensor(
        [[1, 1], [1, 1], [0, 0], [0, 1]], dtype=temporal.dtype
    )

    torch.testing.assert_close(atomic[0], expected_atomic)
    torch.testing.assert_close(inner[0], expected_inner)
    torch.testing.assert_close(temporal[0], expected_temporal)
    assert atomic.shape == (1, 6, 2)
    assert inner.shape == (1, 5, 2)
    assert temporal.shape == (1, 4, 2)

    figure = plot_temporal_example(
        "Nested",
        time,
        lower,
        upper,
        config["threshold"],
        atomic,
        temporal,
        inner_trace=inner,
        show=False,
    )
    assert len(figure.axes) == 4
    assert figure.axes[-1].get_ylabel() == "pdSTL stochastic robustness"
    figure.canvas.draw()
    plt.close(figure)


# ---------------------------------------------------------------------------
# to_steps stays valid, unrelated utility code
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Structural / hygiene tests
# ---------------------------------------------------------------------------


def test_examples_configuration_contains_only_the_three_blocks():
    config = _config()
    assert set(config) == {"show_plots", "always", "eventually", "nested"}
    assert set(config["always"]) == {"threshold", "interval_steps", "signal"}
    assert set(config["eventually"]) == {"threshold", "interval_steps", "signal"}
    assert set(config["nested"]) == {
        "threshold",
        "always_interval_steps",
        "eventually_interval_steps",
        "signal",
    }
    for name in ("always", "eventually", "nested"):
        assert set(config[name]["signal"]) == {"lower", "upper"}


def test_offline_module_has_been_removed():
    assert not (ROOT / "src/offline.py").exists()


def test_reusable_modules_do_not_run_examples_or_eagerly_import_planning():
    code = (
        "import sys; import models.dynamics; "
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
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body
    )
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
