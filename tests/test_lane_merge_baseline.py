"""Lane geometry is unchanged; execution uses the canonical smooth objective."""

import matplotlib

matplotlib.use("Agg")
import pytest
import torch

import yaml
from tempfile import TemporaryDirectory
from pathlib import Path
from planning import environment as lane_merge
from planning.runners import build_environment, lane_success_counter, run_lane_change
from utils import load_config

CONFIG = "configs/scenarios/lane_change.yaml"

# Exact values before the AxisInterval change, seed 0, T_SIM=5, max_iters=20.
# Recorded for drift inspection; only the geometry tier asserts against exact numbers.
BASELINE = {
    "stopped_reason": "max_steps",
    "applied_steps": 5,
    "p_sat": [0.964276, 0.959312, 0.952691, 0.976214, 0.980404],
    "final_state": [3.534585, 0.01286, 3.624797, -0.010911],
}


@pytest.fixture(scope="module")
def setup():
    cfg = load_config(CONFIG)
    return cfg, cfg, build_environment(cfg, device="cpu")


def _execute_lane_merge(cfg, planner_cfg, env, *, steps, iters, seed=0):
    """Run `steps` lane-merge MPC windows on the existing path; returns a plain summary."""
    cfg = {**cfg, "T_SIM": steps, "seed": seed,
           "planner": {**cfg["planner"], "max_iters": iters}}
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        path = Path(directory) / "lane.yaml"
        path.write_text(yaml.safe_dump(cfg))
        result = run_lane_change(str(path), show=False, save=False)
    return {
        "stopped_reason": result.stopped_reason,
        "applied_steps": len(result.applied_controls),
        "p_sat": [plan.hard_interval[0] for plan in result.window_plans],
        "states": torch.stack([state[0] for state in result.states]),
    }


class TestLaneMergeGeometry:
    """Pure functions of the configuration. These must not move."""

    def test_config_loads_with_the_keys_the_scenario_needs(self, setup):
        cfg, _, _ = setup
        assert cfg["dynamics"] == "double_integrator"
        for key in ("road", "obstacle", "goal", "success", "H", "T_SIM"):
            assert key in cfg, f"lane-merge config lost {key!r}"

    def test_specification_is_safety_and_eventual_goal(self, setup):
        _, _, env = setup
        spec = str(env.get_specification(40))
        assert "MovingRectangularObstaclePredicate" in spec
        assert "□_[1, 40]" in spec and "♢_[0, 40]" in spec

    def test_local_window_goal_tracks_the_ego_by_the_configured_lookahead(self, setup):
        _, planner_cfg, env = setup
        local = lane_merge.lane_local_window(env, 7, torch.tensor([12.0, 1.3, 3.5, 0.0]), planner_cfg)
        goal = local.region("goal")
        # ego_x + mpc_goal_lookahead, + mpc_goal_window_width; y inset on both sides.
        assert goal.x == pytest.approx((16.0, 76.0))
        assert goal.y == pytest.approx((2.1, 5.9))

    def test_local_window_floor_lifts_to_the_divider_once_the_ego_is_committed(self, setup):
        _, planner_cfg, env = setup
        below = lane_merge.lane_local_window(env, 7, torch.tensor([12.0, 1.3, 3.5, 0.0]), planner_cfg)
        above = lane_merge.lane_local_window(env, 7, torch.tensor([12.0, 1.9, 3.5, 0.0]), planner_cfg)
        assert below.region("workspace").y == pytest.approx((-1.5, 6.0))  # road y_min + margin
        assert above.region("workspace").y == pytest.approx((2.0, 6.0))   # lifted to lane_divider

    def test_local_window_carries_one_horizon_of_moving_obstacle(self, setup):
        cfg, planner_cfg, env = setup
        local = lane_merge.lane_local_window(env, 7, torch.tensor([12.0, 1.3, 3.5, 0.0]), planner_cfg)
        (vehicle,) = local.by_role("obstacle")
        assert len(vehicle.centers) == cfg["H"] + 1
        assert float(vehicle.centers[0, 0]) == pytest.approx(3.42)
        assert float(vehicle.centers[0, 1]) == pytest.approx(4.0)

    def test_moving_obstacle_position_is_constant_speed(self, setup):
        cfg, _, env = setup
        obstacle, dt = cfg["obstacle"], cfg["dt"]
        for step in (0, 7, 25):
            expected = obstacle["x0"] + obstacle["speed"] * step * dt
            position = lane_merge.obstacle_position(env, step)
            assert position[0] == pytest.approx(expected, rel=1e-6)

    def test_success_needs_consecutive_steps_inside_the_target_band(self, setup):
        cfg, planner_cfg, env = setup
        success = cfg["success"]
        inside = torch.tensor([0.0, (success["y_min"] + success["y_max"]) / 2, 3.5, 0.0])
        outside = torch.tensor([0.0, success["y_min"] - 1.0, 3.5, 0.0])

        counter = lane_success_counter(env, inside, success["consecutive_steps"] - 2)
        done = counter >= success["consecutive_steps"]
        assert counter == success["consecutive_steps"] - 1 and not done
        counter = lane_success_counter(env, inside, success["consecutive_steps"] - 1)
        done = counter >= success["consecutive_steps"]
        assert done, "consecutive_steps inside the band must trigger success"
        counter = lane_success_counter(env, outside, success["consecutive_steps"] - 1)
        done = counter >= success["consecutive_steps"]
        assert counter == 0 and not done, "leaving the band must reset the counter"


class TestLaneMergeExecution:
    """End-to-end outcome. Bounded, not pinned -- see the module docstring."""

    @pytest.fixture(scope="class")
    def run(self, setup):
        cfg, planner_cfg, env = setup
        return _execute_lane_merge(cfg, planner_cfg, env, steps=5, iters=20)

    def test_runs_the_requested_number_of_windows(self, run):
        assert run["stopped_reason"] == BASELINE["stopped_reason"]
        assert run["applied_steps"] == BASELINE["applied_steps"]
        assert run["states"].shape == (BASELINE["applied_steps"] + 1, 4)

    def test_every_window_reports_a_high_safety_probability(self, run):
        assert len(run["p_sat"]) == BASELINE["applied_steps"]
        assert min(run["p_sat"]) > 0.90, f"safety probability collapsed: {run['p_sat']}"
        assert all(0.0 <= p <= 1.0 for p in run["p_sat"])

    def test_ego_makes_forward_progress_at_roughly_the_initial_speed(self, run, setup):
        cfg, _, _ = setup
        x0, vx0, dt = cfg["x0_mean"][0], cfg["x0_mean"][2], cfg["dt"]
        travelled = run["states"][-1, 0].item() - x0
        assert travelled == pytest.approx(vx0 * dt * BASELINE["applied_steps"], rel=0.25)

    def test_ego_has_not_left_its_lane_this_early(self, run):
        # Five steps is far too few to merge; the ego should still be near the lane-1 centre.
        assert abs(run["states"][-1, 1].item()) < 0.5

    def test_ego_stays_inside_the_road(self, run, setup):
        cfg, _, _ = setup
        road = cfg["road"]
        y = run["states"][:, 1]
        assert bool((y >= road["y_min"]).all() and (y <= road["y_max"]).all())


def test_lane_runner_saves_headless_figures_animation_and_structured_result(tmp_path, monkeypatch):
    import planning.runners as runners
    import matplotlib.pyplot as plt

    cfg = load_config(CONFIG)
    cfg["T_SIM"] = 1
    cfg["planner"]["max_iters"] = 2
    path = tmp_path / "lane.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(runners, "RESULTS_DIR", tmp_path)
    result = runners.run_lane_change(str(path), show=False, save=True)
    assert len(result.window_plans) == 1
    assert (tmp_path / "lane_change.png").exists()
    assert (tmp_path / "lane_change_metrics.png").exists()
    assert (tmp_path / cfg["animation"]["filename"]).exists()
    loaded = torch.load(tmp_path / cfg["save_file"], weights_only=False)
    torch.testing.assert_close(loaded.applied_controls, result.applied_controls)
    plt.close("all")


def test_lane_live_callback_accepts_the_canonical_observer_arguments(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from planning.runners import run_lane_change
    from visualization.live_plots import make_lane_change_live_callback

    cfg = load_config(CONFIG)
    cfg["T_SIM"] = 1
    cfg["planner"]["max_iters"] = 1
    path = tmp_path / "lane.yaml"
    path.write_text(yaml.safe_dump(cfg))
    result = run_lane_change(str(path), show=False, save=False)
    env = build_environment(cfg, "cpu")
    monkeypatch.setattr(plt, "pause", lambda *_: None)
    callback = make_lane_change_live_callback(env)
    callback(0, result.states[1], result.window_plans[0])
    plt.close("all")
