"""Real monitoring examples, valid plot origins, and the script entry point."""

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
import yaml
from experiments.offline import run_always_example, run_eventually_example

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("interval", [(0, 1), (1, 2)])
@pytest.mark.parametrize(
    "runner, threshold, reduction",
    [
        (run_always_example, 50.0, min),
        (run_eventually_example, 55.0, max),
    ],
)
def test_examples_match_cdf_and_window_references(
    runner, threshold, reduction, interval
):
    time, mean, variance, atomic, temporal, figure = runner(
        threshold, interval, show=False
    )
    try:
        probability = ndtr((mean.numpy() - threshold) / np.sqrt(variance.numpy()))
        a, b = interval
        expected = np.array(
            [reduction(probability[k + a : k + b + 1]) for k in range(7 - b)]
        )
        assert atomic.shape == (1, 7, 2)
        assert temporal.shape == (1, 7 - b, 2)
        for endpoint in range(2):
            np.testing.assert_allclose(atomic[0, :, endpoint], probability, atol=1e-12)
            np.testing.assert_allclose(temporal[0, :, endpoint], expected, atol=1e-12)
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_xdata(), time)
        np.testing.assert_array_equal(figure.axes[1].lines[0].get_xdata(), time)
        np.testing.assert_array_equal(
            figure.axes[2].lines[0].get_xdata(), time[: 7 - b]
        )
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_example_propagates_insufficient_horizon_error():
    with pytest.raises(ValueError, match="needs 8 steps"):
        run_always_example(50.0, (0, 7), show=False)
    assert not plt.get_fignums()


def _environment(tmp_path):
    return {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def test_reusable_imports_do_not_run_experiments(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import experiments.offline; import experiments.planning; "
            "import matplotlib.pyplot as plt; assert not plt.get_fignums()",
        ],
        cwd=tmp_path,
        env=_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""
    assert not (tmp_path / "saved_data").exists()
    assert not (tmp_path / "outputs").exists()


def test_script_runs_monitoring_and_skips_planning(tmp_path):
    config = yaml.safe_load((ROOT / "configs/stl_demos.yaml").read_text())
    config["show_plots"] = False
    config_path = tmp_path / "monitoring.yaml"
    config_path.write_text(yaml.safe_dump(config))
    script = tmp_path / "main.py"
    script.write_text(
        (ROOT / "src/main.py")
        .read_text()
        .replace('"configs/stl_demos.yaml"', repr(str(config_path)))
    )
    code = """
import runpy
import sys
import matplotlib.pyplot as plt

def forbid_planning_config(event, args):
    if event == "open" and isinstance(args[0], (str, bytes)):
        path = str(args[0])
        if "configs/scenarios/" in path or path.endswith("configs/planning.yaml"):
            raise AssertionError("A skipped planner loaded configuration")
sys.addaudithook(forbid_planning_config)

def unexpected_show():
    raise AssertionError("show=False displayed a window")
plt.show = unexpected_show
runpy.run_path(sys.argv[1], run_name="__main__")
assert len(plt.get_fignums()) == 2
plt.close("all")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)],
        cwd=tmp_path,
        env=_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Always[0,1]" in result.stdout
    assert "Eventually[0,1]" in result.stdout
    assert result.stdout.count("origin  window_start") == 2
    assert result.stderr.count("Skipping the block") == 4
    assert not (tmp_path / "saved_data").exists()
    assert not (tmp_path / "outputs").exists()
