"""Characterization baseline for the lane-merge scenario and its receding-horizon execution.

Written BEFORE the reach-avoid refactor, against the behaviour on RA_L-planning at
commit 9301ef9, because `run_lane_change`, `Planner._run_mpc`, `make_local_lane_change_window`
and the success counter had no test coverage at all. Its job is to make
"lane-merge behavior remains operational" falsifiable.

Two tiers, deliberately:

* `TestLaneMergeGeometry` pins pure functions of the configuration exactly. Nothing in the
  refactor may move these numbers -- they are geometry, not optimization.

* `TestLaneMergeExecution` pins the end-to-end outcome loosely. It cannot be exact: the
  refactor rebuilds `InsideRectangle` as a conjunction of axis intervals, so the lane-merge
  goal's *smooth* lower bound changes from a hard clamp to softplus. That shifts the Adam
  trajectory by design. The assertions below bound the outcome (lane kept, safety probability
  high, forward progress) rather than reproducing a float. BASELINE records the exact
  pre-refactor values so drift stays visible to a human reader.

The end-to-end run goes through `_execute_lane_merge`. That helper is the ONLY thing the
refactor is allowed to rewrite; every assertion must survive untouched.
"""

import matplotlib

matplotlib.use("Agg")
import pytest
import torch

from planning.controllers import RecedingHorizonController
from planning.runners import (
    _legacy_optimizer_block,
    _window_config,
    build_initial_belief,
    build_planner,
    build_scenario,
)
from utils import load_config

CONFIG = "configs/scenarios/lane_change.yaml"

# Exact pre-refactor values, seed 0, T_SIM=5, max_iters=20. Recorded for drift inspection;
# only the geometry tier asserts against exact numbers.
BASELINE = {
    "stopped_reason": "T_SIM",
    "applied_steps": 5,
    "p_sat": [0.964276, 0.959312, 0.952691, 0.976214, 0.980404],
    "final_state": [3.534585, 0.01286, 3.624797, -0.010911],
}


@pytest.fixture(scope="module")
def setup():
    cfg = load_config(CONFIG)
    env = build_scenario({**cfg, "scenario": {"type": "lane_merge"}}, "cpu")
    return cfg, _window_config(cfg), env


def _execute_lane_merge(cfg, planner_cfg, env, *, steps, iters, seed=0):
    """Run `steps` lane-merge receding-horizon windows and return a plain summary.

    REFACTOR NOTE: reroute this body to RecedingHorizonController.run(). The returned
    dict keys are the contract; the assertions below must not change.
    """
    optimizer = {**_legacy_optimizer_block(cfg), "max_iterations": iters}
    planner = build_planner(
        {
            "horizon": cfg["H"],
            "dynamics": {
                "type": cfg["dynamics"], "dt": cfg["dt"],
                "u_max": cfg["u_max"], "q_std": cfg["q_std"],
            },
            "optimizer": optimizer,
        },
        env,
    )
    mean0, covariance0 = build_initial_belief(
        {"mean": cfg["x0_mean"], "covariance": cfg["x0_cov_scale"]}, "cpu"
    )
    torch.manual_seed(seed)
    result = RecedingHorizonController(
        planner=planner, dynamics=planner.dynamics, scenario=env,
        config={"warm_start": True},
    ).run(mean0, (mean0, covariance0), max_steps=steps, apply_steps=1)
    return {
        "stopped_reason": "T_SIM" if result.stopped_reason == "max_steps" else result.stopped_reason,
        "applied_steps": result.applied_controls.shape[0],
        "p_sat": result.hard_lowers,
        "states": result.states,
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
        spec = str(env.specification(40))
        assert "MovingRectangularObstaclePredicate" in spec
        assert "□_[1, 40]" in spec and "♢_[0, 40]" in spec

    def test_local_window_goal_tracks_the_ego_by_the_configured_lookahead(self, setup):
        _, planner_cfg, env = setup
        local = env.local_window(7, torch.tensor([12.0, 1.3, 3.5, 0.0]), planner_cfg)
        # curr_x + mpc_goal_lookahead, + mpc_goal_window_width; y inset on both sides.
        assert local.goal["x"] == pytest.approx([16.0, 76.0])
        assert local.goal["y"] == pytest.approx([2.1, 5.9])

    def test_local_window_floor_lifts_to_the_divider_once_the_ego_is_committed(self, setup):
        _, planner_cfg, env = setup
        below = env.local_window(7, torch.tensor([12.0, 1.3, 3.5, 0.0]), planner_cfg)
        above = env.local_window(7, torch.tensor([12.0, 1.9, 3.5, 0.0]), planner_cfg)
        assert below.bounds["y"] == pytest.approx([-1.5, 6.0])   # road y_min + margin
        assert above.bounds["y"] == pytest.approx([2.0, 6.0])    # lifted to lane_divider

    def test_local_window_carries_one_horizon_of_moving_obstacle(self, setup):
        cfg, planner_cfg, env = setup
        local = env.local_window(7, torch.tensor([12.0, 1.3, 3.5, 0.0]), planner_cfg)
        (obs,) = local.moving_obstacles
        assert len(obs["x_traj"]) == cfg["H"] + 1
        assert float(obs["x_traj"][0]) == pytest.approx(3.42)
        assert float(obs["y_traj"][0]) == pytest.approx(4.0)

    def test_moving_obstacle_position_is_constant_speed(self, setup):
        cfg, _, env = setup
        obstacle, dt = cfg["obstacle"], cfg["dt"]
        for step in (0, 7, 25):
            expected = obstacle["x0"] + obstacle["speed"] * step * dt
            assert env.moving_obstacle_position(step)[0] == pytest.approx(expected, rel=1e-6)

    def test_success_needs_consecutive_steps_inside_the_target_band(self, setup):
        cfg, _, env = setup
        success = cfg["success"]
        inside = torch.tensor([0.0, (success["y_min"] + success["y_max"]) / 2, 3.5, 0.0])
        outside = torch.tensor([0.0, success["y_min"] - 1.0, 3.5, 0.0])

        env.reset_progress()
        for _ in range(success["consecutive_steps"] - 1):
            assert not env.is_complete(inside), "must not fire before consecutive_steps"
        assert env.is_complete(inside), "consecutive_steps inside the band must trigger success"

        env.reset_progress()
        for _ in range(success["consecutive_steps"] - 1):
            env.is_complete(inside)
        assert not env.is_complete(outside), "leaving the band must reset the counter"
        assert not env.is_complete(inside), "and the count must restart from one"


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
