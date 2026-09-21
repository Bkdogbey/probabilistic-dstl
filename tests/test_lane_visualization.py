"""Publication styling for lane results."""

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import yaml

from planning.runners import run_lane_change, setup_problem
from utils import load_config
from visualization.animation import animate_mpc
from visualization.live_plots import create_live_view
from visualization.planning import COLORS, _draw_environment, plot_lane_merge


def test_lane_figures_have_time_labels_red_traffic_and_final_score(
    tmp_path,
):
    config = load_config("configs/scenarios/lane_change.yaml")
    config["H"] = 8
    config["T_SIM"] = 1
    config["task"]["start_window_seconds"] = [0.2, 1.0]
    config["task"]["dwell_seconds"] = 0.2
    config["planner"]["max_iters"] = 1
    path = tmp_path / "lane_change.yaml"
    path.write_text(yaml.safe_dump(config))
    result = run_lane_change(str(path), max_steps=1, show=False, save=False)
    problem = setup_problem(config, device="cpu", with_environment=True)
    figures = plot_lane_merge(result, problem.env, dt=config["dt"])
    trajectory = figures["trajectory"][1]
    trajectory_text = {text.get_text() for text in trajectory.texts}
    assert "t=0 s" in trajectory_text
    assert any("Final pdSTL: [" in text for text in trajectory_text)
    assert -3.0 < trajectory.get_xlim()[0] <= 0.0
    traffic = [
        patch
        for patch in trajectory.patches
        if patch.get_label() in {"Traffic", "_nolegend_"}
    ]
    expected = colors.to_rgb(COLORS["traffic"])
    assert traffic and all(
        patch.get_facecolor()[:3] == expected for patch in traffic
    )
    combined_axes = figures["combined"][1]
    score_labels = {
        text.get_text() for text in combined_axes[1].get_legend().get_texts()
    }
    assert {"Hard interval", "Smooth score"} <= score_labels
    assert {"Dwell-start window", "Completion deadline"} <= score_labels
    assert any(
        text.get_text().startswith("Final pdSTL: [")
        for text in combined_axes[1].texts
    )
    animation_fig, animation_axes, movie = animate_mpc(
        result, problem.env, dt=config["dt"], lane=True
    )
    animation_anchor = (
        animation_axes[0]
        .get_legend()
        .get_bbox_to_anchor()
        .transformed(animation_axes[0].transAxes.inverted())
    )
    assert animation_anchor.y0 > 1
    movie._draw_was_started = True
    plt.close(animation_fig)
    plt.close("all")


def test_merge_shades_only_the_taper_triangle():
    config = load_config("configs/scenarios/lane_merge.yaml")
    problem = setup_problem(config, device="cpu", with_environment=True)
    fig, ax = plt.subplots()
    _draw_environment(ax, problem.env, lane=True)
    shaded = [
        collection
        for collection in ax.collections
        if collection.get_label() == "Non-drivable"
    ]
    assert len(shaded) == 1
    vertices = shaded[0].get_paths()[0].vertices
    assert vertices[:, 0].min() == config["ramp"]["start_x"]
    assert vertices[:, 0].max() == config["ramp"]["end_x"]
    plt.close(fig)


def test_live_map_legend_is_outside_axes():
    config = load_config("configs/scenarios/lane_change.yaml")
    problem = setup_problem(config, device="cpu", with_environment=True)
    from planning.runners import _lane_initial_state

    fig, axes, _ = create_live_view(
        problem.env,
        _lane_initial_state(problem),
        dt=config["dt"],
        lane=True,
        dynamics=problem.dyn,
    )
    anchor = (
        axes[0]
        .get_legend()
        .get_bbox_to_anchor()
        .transformed(axes[0].transAxes.inverted())
    )
    assert anchor.y0 > 1
    plt.close(fig)
