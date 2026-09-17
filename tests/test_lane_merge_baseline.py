"""Characterization baseline for the lane-merge scenario and its MPC execution.

Lane merge had no test coverage at all: `run_lane_change`, `Planner._run_mpc`,
`make_local_lane_change_window` and the success counter were all unpinned. This pins them
while lane merge stays on its existing working path, so "lane merge still works" is
falsifiable before anything touches it.

Two tiers, deliberately:

* `TestLaneMergeGeometry` pins pure functions of the configuration exactly. These are
  geometry, not optimization, and nothing should move them.

* `TestLaneMergeExecution` bounds the end-to-end outcome instead of pinning a float. It
  cannot be exact: `InsideRectangle` is now a conjunction of two axis intervals, so the
  lane-merge goal's *smooth* lower bound is a softplus where it used to be a hard clamp.
  That shifts the Adam trajectory by design. BASELINE records the pre-change values so the
  drift stays visible to a reader.
"""

import matplotlib

matplotlib.use("Agg")
import pytest
import torch

from planning.planner import Planner
from planning.scenarios import lane_merge
from planning.runners import build_dynamics, build_environment, build_initial_belief
from utils import load_config

CONFIG = "configs/scenarios/lane_change.yaml"

# Exact values before the AxisInterval change, seed 0, T_SIM=5, max_iters=20.
# Recorded for drift inspection; only the geometry tier asserts against exact numbers.
BASELINE = {
    "stopped_reason": "T_SIM",
    "applied_steps": 5,
    "p_sat": [0.964276, 0.959312, 0.952691, 0.976214, 0.980404],
    "final_state": [3.534585, 0.01286, 3.624797, -0.010911],
}


@pytest.fixture(scope="module")
def setup():
    cfg = load_config(CONFIG)
    planner_cfg = {**load_config("configs/planning.yaml"), **cfg["planner"]}
    return cfg, planner_cfg, build_environment(cfg, device="cpu")


def _execute_lane_merge(cfg, planner_cfg, env, *, steps, iters, seed=0):
    """Run `steps` lane-merge MPC windows on the existing path; returns a plain summary."""
    planner = Planner(
        build_dynamics(cfg, "cpu"),
        env,
        cfg["H"],
        config={**planner_cfg, "T_SIM": steps, "mpc_mode": "lane_change", "max_iters": iters},
    )
    torch.manual_seed(seed)
    result = planner.solve(*build_initial_belief(cfg, "cpu"), verbose=False)
    return {
        "stopped_reason": result["stopped_reason"],
        "applied_steps": result["u_trace"].shape[1],
        "p_sat": list(result["p_sat_trace"]),
        "states": result["mean_trace"][0],
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
        local = lane_merge.local_window(env, 7, torch.tensor([12.0, 1.3, 3.5, 0.0]), planner_cfg)
        goal = local.region("goal")
        # ego_x + mpc_goal_lookahead, + mpc_goal_window_width; y inset on both sides.
        assert goal.x == pytest.approx((16.0, 76.0))
        assert goal.y == pytest.approx((2.1, 5.9))

    def test_local_window_floor_lifts_to_the_divider_once_the_ego_is_committed(self, setup):
        _, planner_cfg, env = setup
        below = lane_merge.local_window(env, 7, torch.tensor([12.0, 1.3, 3.5, 0.0]), planner_cfg)
        above = lane_merge.local_window(env, 7, torch.tensor([12.0, 1.9, 3.5, 0.0]), planner_cfg)
        assert below.region("workspace").y == pytest.approx((-1.5, 6.0))  # road y_min + margin
        assert above.region("workspace").y == pytest.approx((2.0, 6.0))   # lifted to lane_divider

    def test_local_window_carries_one_horizon_of_moving_obstacle(self, setup):
        cfg, planner_cfg, env = setup
        local = lane_merge.local_window(env, 7, torch.tensor([12.0, 1.3, 3.5, 0.0]), planner_cfg)
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
        planner = Planner(build_dynamics(cfg, "cpu"), env, cfg["H"], config=planner_cfg)
        success = cfg["success"]
        inside = torch.tensor([0.0, (success["y_min"] + success["y_max"]) / 2, 3.5, 0.0])
        outside = torch.tensor([0.0, success["y_min"] - 1.0, 3.5, 0.0])

        counter, done = planner._lane_change_success(inside, success["consecutive_steps"] - 2)
        assert counter == success["consecutive_steps"] - 1 and not done
        counter, done = planner._lane_change_success(inside, success["consecutive_steps"] - 1)
        assert done, "consecutive_steps inside the band must trigger success"
        counter, done = planner._lane_change_success(outside, success["consecutive_steps"] - 1)
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
