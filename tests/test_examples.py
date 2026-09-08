"""Offline examples pass supplied probability bounds straight through pdSTL."""

import ast
import os
from pathlib import Path
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch
import yaml

import models.dynamics
from pdstl.base import create_probability_belief_trajectory
from pdstl.operators import Always, Eventually, GreaterThan
from utils import to_steps
from visualization.temporal import plot_temporal_example


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return yaml.safe_load((ROOT / "configs/examples.yaml").read_text())


def _beliefs(config):
    """Build the example trajectory exactly as main.py does."""
    predicate = GreaterThan(config["threshold"])
    beliefs = create_probability_belief_trajectory(
        predicate, config["probability_bounds"]
    )
    return predicate, beliefs


def _expected(rows, dtype):
    return torch.tensor(rows, dtype=dtype)


# ---------------------------------------------------------------------------
# The atomic trace is exactly what the configuration supplied
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["always", "eventually", "nested"])
def test_atomic_trace_equals_the_configured_probability_bounds(name):
    config = _config()[name]
    predicate, beliefs = _beliefs(config)

    atomic = predicate(beliefs)

    expected = _expected(config["probability_bounds"], atomic.dtype)
    torch.testing.assert_close(atomic[0], expected)
    assert atomic.shape == (1, len(config["probability_bounds"]), 2)


# ---------------------------------------------------------------------------
# Temporal operators over the configured bounds
# ---------------------------------------------------------------------------


def test_always_applies_the_endpointwise_minimum():
    config = _config()["always"]
    predicate, beliefs = _beliefs(config)
    formula = Always(predicate, interval=config["interval_steps"])

    atomic = predicate(beliefs)
    temporal = formula(beliefs, scale=-1)

    torch.testing.assert_close(
        temporal[0],
        _expected(
            [
                [0.70, 0.90],
                [0.40, 0.75],
                [0.20, 0.55],
                [0.20, 0.55],
                [0.20, 0.55],
                [0.75, 0.90],
            ],
            temporal.dtype,
        ),
    )
    assert temporal.shape == (1, 6, 2)

    figure = plot_temporal_example(
        str(predicate), atomic, str(formula), temporal, show=False
    )
    assert len(figure.axes) == 2
    figure.canvas.draw()
    plt.close(figure)


def test_eventually_applies_the_endpointwise_maximum():
    config = _config()["eventually"]
    predicate, beliefs = _beliefs(config)
    formula = Eventually(predicate, interval=config["interval_steps"])

    atomic = predicate(beliefs)
    temporal = formula(beliefs, scale=-1)

    torch.testing.assert_close(
        temporal[0],
        _expected(
            [
                [0.20, 0.40],
                [0.45, 0.70],
                [0.45, 0.70],
                [0.80, 0.95],
                [0.80, 0.95],
                [0.25, 0.45],
            ],
            temporal.dtype,
        ),
    )
    assert temporal.shape == (1, 6, 2)

    figure = plot_temporal_example(
        str(predicate), atomic, str(formula), temporal, show=False
    )
    assert len(figure.axes) == 2
    figure.canvas.draw()
    plt.close(figure)


def test_nested_eventually_always_matches_hand_computation():
    config = _config()["nested"]
    predicate, beliefs = _beliefs(config)
    inner = Always(predicate, interval=config["always_interval_steps"])
    formula = Eventually(inner, interval=config["eventually_interval_steps"])

    atomic = predicate(beliefs)
    inner_trace = inner(beliefs, scale=-1)
    temporal = formula(beliefs, scale=-1)

    torch.testing.assert_close(
        inner_trace[0],
        _expected(
            [
                [0.20, 0.40],
                [0.70, 0.90],
                [0.30, 0.50],
                [0.30, 0.50],
                [0.60, 0.80],
            ],
            inner_trace.dtype,
        ),
    )
    torch.testing.assert_close(
        temporal[0],
        _expected(
            [[0.70, 0.90], [0.70, 0.90], [0.30, 0.50], [0.60, 0.80]],
            temporal.dtype,
        ),
    )
    assert inner_trace.shape == (1, 5, 2)
    assert temporal.shape == (1, 4, 2)

    figure = plot_temporal_example(
        str(predicate),
        atomic,
        str(formula),
        temporal,
        inner_label=str(inner),
        inner_trace=inner_trace,
        show=False,
    )
    assert len(figure.axes) == 3
    figure.canvas.draw()
    plt.close(figure)


def test_only_complete_temporal_windows_are_returned():
    config = _config()["always"]
    predicate, beliefs = _beliefs(config)
    steps = len(config["probability_bounds"])

    for interval in ([0, 1], [0, 2], [1, 3]):
        temporal = Always(predicate, interval=interval)(beliefs, scale=-1)
        assert temporal.shape == (1, steps - interval[1], 2)


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


def test_examples_configuration_holds_only_numerical_example_data():
    config = _config()
    assert set(config) == {"show_plots", "always", "eventually", "nested"}
    assert set(config["always"]) == {
        "threshold",
        "interval_steps",
        "probability_bounds",
    }
    assert set(config["eventually"]) == {
        "threshold",
        "interval_steps",
        "probability_bounds",
    }
    assert set(config["nested"]) == {
        "threshold",
        "always_interval_steps",
        "eventually_interval_steps",
        "probability_bounds",
    }
    for name in ("always", "eventually", "nested"):
        assert all(len(pair) == 2 for pair in config[name]["probability_bounds"])


def test_offline_module_and_signal_dispatcher_remain_absent():
    assert not (ROOT / "src/offline.py").exists()
    assert not hasattr(models.dynamics, "create_signal_trace")


def test_reusable_modules_do_not_run_examples_or_eagerly_import_planning():
    code = (
        "import sys; import models.dynamics; import pdstl.base; "
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
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def _main_blocks(source):
    """Return the (flag, name) pair of each literal skip_run block."""
    return re.findall(r'with skip_run\("(run|skip)", "(\w+)"\)', source)


def test_main_is_direct_and_holds_three_literal_skip_run_blocks():
    source = (ROOT / "src/main.py").read_text()
    tree = ast.parse(source)

    assert [name for _, name in _main_blocks(source)] == [
        "Always",
        "Eventually",
        "Nested",
    ]
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body
    )
    assert not any(isinstance(node, ast.If) for node in tree.body)


@pytest.mark.parametrize("flags", [("run", "run", "run"), ("run", "skip", "run")])
def test_main_runs_whichever_blocks_the_user_selected(tmp_path, flags):
    source = (ROOT / "src/main.py").read_text()
    for (_, name), flag in zip(_main_blocks(source), flags):
        source = re.sub(
            rf'skip_run\("(?:run|skip)", "{name}"\)',
            f'skip_run("{flag}", "{name}")',
            source,
        )

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
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    ran = sum(1 for flag in flags if flag == "run")
    assert result.stdout.count("Running the block") == ran
    assert result.stderr.count("Skipping the block") == len(flags) - ran
    for (_, name), flag in zip(_main_blocks(source), flags):
        assert (f"\n{name}\n" in result.stdout) == (flag == "run")
