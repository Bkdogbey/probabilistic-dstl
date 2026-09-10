"""Offline examples drive pdSTL from an upstream scalar state model."""

import ast
import os
from pathlib import Path
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.special import ndtr
import yaml

import models.dynamics
from models.dynamics import create_gaussian_belief_trajectory, piecewise_signal
from pdstl.operators import Always, Eventually, GreaterThan
from utils import to_steps
from visualization.temporal import plot_temporal_example


ROOT = Path(__file__).resolve().parents[1]

EXAMPLES = ("always", "eventually", "nested")


def _config():
    return yaml.safe_load((ROOT / "configs/examples.yaml").read_text())


def _example(name):
    """Build one example's state trace, predicate, and beliefs as main.py does."""
    config = _config()[name]
    time, mean, variance = piecewise_signal(config["values"])
    predicate = GreaterThan(config["threshold"])
    sigma = config["sigma_multiplier"] * np.sqrt(variance)
    beliefs = create_gaussian_belief_trajectory(mean - sigma, mean + sigma, variance)
    return config, (time, mean, variance), predicate, beliefs


def _expected_atomic(config, mean, variance):
    """The sigma-displaced Gaussian CDF the belief is supposed to produce."""
    sigma = np.sqrt(variance)
    displacement = config["sigma_multiplier"] * sigma
    threshold = config["threshold"]
    return np.stack(
        (
            ndtr((mean - displacement - threshold) / sigma),
            ndtr((mean + displacement - threshold) / sigma),
        ),
        axis=-1,
    )


def _reduce(trace, interval, reduction):
    """Endpointwise window reduction at every complete evaluation origin."""
    a, b = interval
    return np.array(
        [
            reduction(trace[origin + a : origin + b + 1], axis=0)
            for origin in range(len(trace) - b)
        ]
    )


# ---------------------------------------------------------------------------
# The atomic bounds come from the state model, not from the file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", EXAMPLES)
def test_atomic_bounds_are_derived_from_the_configured_state_model(name):
    config, (_, mean, variance), predicate, beliefs = _example(name)

    atomic = predicate(beliefs)

    np.testing.assert_allclose(
        atomic[0].numpy(), _expected_atomic(config, mean, variance), atol=1e-12
    )
    assert atomic.shape == (1, len(config["values"]), 2)


@pytest.mark.parametrize("name", EXAMPLES)
def test_every_example_produces_a_valid_state_signal(name):
    config, (time, mean, variance), _, _ = _example(name)

    assert time.shape == mean.shape == variance.shape
    assert len(time) == len(config["values"])
    np.testing.assert_allclose(time, np.arange(len(config["values"])))
    assert np.all(variance > 0)
    assert np.isfinite(mean).all()


@pytest.mark.parametrize("name", EXAMPLES)
def test_every_example_plots_aligned_panels_over_the_state_trace(name):
    config, (time, mean, variance), predicate, beliefs = _example(name)
    sigma = config["sigma_multiplier"] * np.sqrt(variance)

    if name == "nested":
        inner = Always(predicate, interval=config["always_interval_steps"])
        formula = Eventually(inner, interval=config["eventually_interval_steps"])
        extra = {"inner_label": str(inner), "inner_trace": inner(beliefs, scale=-1)}
    else:
        formula = Always(predicate, interval=config["interval_steps"])
        extra = {}

    temporal = formula(beliefs, scale=-1)
    figure = plot_temporal_example(
        time, mean, sigma, config["threshold"],
        str(predicate), predicate(beliefs), str(formula), temporal,
        show=False, **extra,
    )

    assert len(figure.axes) == (4 if name == "nested" else 3)
    assert figure.axes[0].get_ylabel() == "state"
    assert figure.axes[1].get_ylabel() == "probability bounds"
    assert figure.axes[-1].get_ylabel() == "pdSTL stochastic robustness"

    # Every panel shares one time axis, and the shorter temporal traces stop at
    # their last valid origin instead of being stretched over the horizon.
    limits = {ax.get_xlim() for ax in figure.axes}
    assert len(limits) == 1
    assert figure.axes[0].lines[-2].get_xdata().max() == time[-1]
    assert figure.axes[-1].lines[0].get_xdata().max() == time[len(temporal[0]) - 1]

    figure.canvas.draw()
    plt.close(figure)


# ---------------------------------------------------------------------------
# Temporal operators over the derived bounds
# ---------------------------------------------------------------------------


def test_always_applies_the_endpointwise_minimum():
    config, (_, mean, variance), predicate, beliefs = _example("always")
    interval = config["interval_steps"]

    temporal = Always(predicate, interval=interval)(beliefs, scale=-1)

    expected = _reduce(_expected_atomic(config, mean, variance), interval, np.min)
    np.testing.assert_allclose(temporal[0].numpy(), expected, atol=1e-12)
    assert temporal.shape == (1, 6, 2)
    # The window covering the sub-threshold dip is pinned by its worst step.
    np.testing.assert_allclose(
        temporal[0, 2].numpy(), [0.02275013, 0.5], atol=1e-8
    )


def test_eventually_applies_the_endpointwise_maximum():
    config, (_, mean, variance), predicate, beliefs = _example("eventually")
    interval = config["interval_steps"]

    temporal = Eventually(predicate, interval=interval)(beliefs, scale=-1)

    expected = _reduce(_expected_atomic(config, mean, variance), interval, np.max)
    np.testing.assert_allclose(temporal[0].numpy(), expected, atol=1e-12)
    assert temporal.shape == (1, 6, 2)
    # The window reaching the above-threshold peak takes that step's bounds.
    np.testing.assert_allclose(
        temporal[0, 3].numpy(), [0.84134475, 0.9986501], atol=1e-8
    )


def test_nested_eventually_always_matches_hand_computation():
    config, (_, mean, variance), predicate, beliefs = _example("nested")
    always_interval = config["always_interval_steps"]
    eventually_interval = config["eventually_interval_steps"]

    inner = Always(predicate, interval=always_interval)
    formula = Eventually(inner, interval=eventually_interval)
    inner_trace = inner(beliefs, scale=-1)
    temporal = formula(beliefs, scale=-1)

    atomic = _expected_atomic(config, mean, variance)
    expected_inner = _reduce(atomic, always_interval, np.min)
    expected_outer = _reduce(expected_inner, eventually_interval, np.max)

    np.testing.assert_allclose(inner_trace[0].numpy(), expected_inner, atol=1e-12)
    np.testing.assert_allclose(temporal[0].numpy(), expected_outer, atol=1e-12)
    assert inner_trace.shape == (1, 5, 2)
    assert temporal.shape == (1, 4, 2)


def test_only_complete_temporal_windows_are_returned():
    config, _, predicate, beliefs = _example("always")
    steps = len(config["values"])

    for interval in ([0, 1], [0, 2], [1, 3]):
        temporal = Always(predicate, interval=interval)(beliefs, scale=-1)
        assert temporal.shape == (1, steps - interval[1], 2)


def test_state_panel_rejects_a_signal_that_does_not_span_the_atomic_trace():
    _, _, predicate, beliefs = _example("always")
    atomic = predicate(beliefs)

    with pytest.raises(ValueError, match="match each other and the atomic trace"):
        plot_temporal_example(
            np.arange(3.0), np.zeros(3), np.ones(3), 0.5,
            str(predicate), atomic, "label", atomic, show=False,
        )


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
    assert set(config) == {"show_plots", "enclosure_reach", *EXAMPLES}

    shared = {"threshold", "sigma_multiplier", "values"}
    assert set(config["always"]) == shared | {"interval_steps"}
    assert set(config["eventually"]) == shared | {"interval_steps"}
    assert set(config["nested"]) == shared | {
        "always_interval_steps",
        "eventually_interval_steps",
    }
    for name in EXAMPLES:
        assert all(len(pair) == 2 for pair in config[name]["values"])

    enclosure = config["enclosure_reach"]
    assert set(enclosure) == {
        "threshold", "interval_steps", "H", "dt", "u_max", "q_std",
        "lower0", "upper0", "covariance0", "d_lower", "d_upper", "seed", "planner",
    }
    assert len(enclosure["lower0"]) == len(enclosure["upper0"]) == 2


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


def test_main_is_direct_and_holds_one_literal_skip_run_block_per_example():
    source = (ROOT / "src/main.py").read_text()
    tree = ast.parse(source)

    assert [name for _, name in _main_blocks(source)] == [
        "Always",
        "Eventually",
        "Nested",
        "EnclosureReach",
    ]
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body
    )
    assert not any(isinstance(node, ast.If) for node in tree.body)


@pytest.mark.parametrize(
    "flags",
    [
        ("run", "run", "run", "run"),
        ("run", "skip", "run", "skip"),
        ("skip", "skip", "skip", "skip"),
    ],
)
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
