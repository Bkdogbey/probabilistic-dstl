"""Four-vehicle lane MPC, relative beliefs, physical limits, and live views."""

import matplotlib.pyplot as plt
import pytest
import torch
import yaml
from PIL import Image

from baselines.det_stl import compare_lane_window
from models.beliefs import GaussianBeliefTrajectory
from models.rollouts import lane_rollout
from pdstl.predicates import AxisInterval, RelativeAxisInterval
from planning.environment import lane_local_window, lane_subformulas
from planning.planner import IterationRecord
from planning.runners import (
    _lane_initial_controls,
    _lane_initial_state,
    build_environment,
    run_lane_change,
    setup_problem,
)
from utils import load_config
from visualization.live_plots import create_live_view


@pytest.fixture
def lane_config(tmp_path):
    cfg = load_config("configs/scenarios/lane_change.yaml")
    cfg["H"] = 8
    cfg["task"]["start_window_seconds"] = [0.6, 1.0]
    cfg["task"]["dwell_seconds"] = 0.2
    cfg["planner"]["max_iters"] = 2
    cfg["seed"] = 0
    path = tmp_path / "lane_change.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return cfg, path


def test_relative_beliefs_and_separate_formulae(lane_config):
    cfg, _ = lane_config
    setup = setup_problem(cfg, with_environment=True)
    state = _lane_initial_state(setup)
    names = [car["name"] for car in cfg["traffic"]]
    rollout = lane_rollout(
        setup.dyn,
        state[0],
        state[1],
        state[3],
        state[4],
        cfg["traffic_q_std"] ** 2 * torch.eye(4),
        names,
        cfg["speed_bounds"],
        (cfg["accel_bounds"]["longitudinal"], cfg["accel_bounds"]["lateral"]),
    )
    prediction = rollout(
        setup.planner._control_parameters(_lane_initial_controls(setup))
    )
    belief = prediction.belief_trajectory
    relative = belief.relative_belief("lead")
    torch.testing.assert_close(
        relative.trace.mean[0, 0], state[3][0, :2] - state[0][:2]
    )
    torch.testing.assert_close(
        relative.trace.covariance[0, 0],
        state[4][0, :2, :2] + state[1][:2, :2],
    )
    event = RelativeAxisInterval("lead", -5.0, 5.0, dim=0)
    torch.testing.assert_close(
        belief.probability_bounds(event),
        relative.probability_bounds(AxisInterval(-5.0, 5.0, dim=0)),
    )
    with pytest.raises(ValueError, match="lane belief"):
        GaussianBeliefTrajectory(
            prediction.aux["mean_trace"], prediction.aux["cov_trace"]
        ).probability_bounds(event)
    local = lane_local_window(setup.env, 0, state[0], cfg)
    formulae = lane_subformulas(local, cfg["H"])
    assert {"road", "safe", "complete", "overall"} <= formulae.keys()
    assert sum(name.startswith("safety_") for name in formulae) == 3
    lower, upper = formulae["overall"].probability_interval(belief)
    assert 0 <= lower <= upper <= 1


def test_lane_save_and_live_observer(
    lane_config, tmp_path, monkeypatch, capsys
):
    import planning.runners as runners

    cfg, path = lane_config
    monkeypatch.setattr(runners, "RESULTS_DIR", tmp_path)
    result = run_lane_change(str(path), max_steps=1, show=False, save=True)
    for suffix in (".pt", ".png", ".pdf", ".gif"):
        assert (tmp_path / f"lane_change{suffix}").exists()
    with Image.open(tmp_path / "lane_change.gif") as movie:
        assert movie.n_frames >= 2
        first = movie.convert("RGB").tobytes()
        movie.seek(movie.n_frames - 1)
        assert movie.convert("RGB").tobytes() != first

    setup = setup_problem(cfg, with_environment=True)
    fig, axes, observe = create_live_view(
        setup.env,
        result.states[0],
        dt=cfg["dt"],
        lane=True,
        dynamics=setup.dyn,
        max_iters=2,
    )
    ax_map = axes[0]
    labels = [patch.get_label() for patch in ax_map.patches]
    assert {"Ego", "Lead", "Target Front", "Target Rear"} <= set(labels)
    for vehicle in cfg["traffic"]:
        patch = next(
            item
            for item in ax_map.patches
            if item.get_label() == vehicle["name"].replace("_", " ").title()
        )
        assert patch.get_x() + patch.get_width() / 2 == pytest.approx(
            vehicle["x0"]
        )
    legend = [item.get_text() for item in ax_map.get_legend().get_texts()]
    assert len(legend) == len(set(legend))
    window_labels = [
        item.get_text() for item in axes[3].get_legend().get_texts()
    ]
    assert "Lane entry window" in window_labels
    assert "Entry deadline" in window_labels
    executed = next(
        line
        for line in ax_map.lines
        if line.get_label() == "Executed trajectory"
    )
    planned = next(
        line
        for line in ax_map.lines
        if line.get_label() == "Predicted planning window"
    )
    assert list(executed.get_xdata()) == [cfg["x0_mean"][0]]
    plan = result.window_plans[0]
    observe.on_iteration(
        0,
        0,
        IterationRecord(
            plan.controls,
            plan.final_loss,
            plan.smooth_lower,
            plan.hard_interval,
            plan.smoothing_beta,
        ),
    )
    assert len(planned.get_xdata()) == cfg["H"] + 1
    first_candidate = list(planned.get_ydata())
    observe.on_iteration(
        0,
        1,
        IterationRecord(
            plan.controls * 0.5,
            plan.final_loss,
            plan.smooth_lower,
            plan.hard_interval,
            plan.smoothing_beta,
        ),
    )
    assert list(planned.get_ydata()) != first_candidate
    observe(0, result.states[1], plan)
    assert len(executed.get_xdata()) == 2
    plt.close(fig)

    viewed = run_lane_change(
        str(path), max_steps=1, show=False, save=False, live=True
    )
    torch.testing.assert_close(
        viewed.applied_controls, result.applied_controls
    )
    for live_state, state in zip(viewed.states, result.states):
        torch.testing.assert_close(live_state[0], state[0])
        torch.testing.assert_close(live_state[3], state[3])
    assert [p.hard_interval for p in viewed.window_plans] == [
        p.hard_interval for p in result.window_plans
    ]
    assert "Lane change window 1, iteration 1/2" in capsys.readouterr().out
    plt.close("all")


def test_default_lane_completes_with_physical_controls():
    cfg = load_config("configs/scenarios/lane_change.yaml")
    result = run_lane_change(show=False, save=False)
    assert result.stopped_reason == "goal_reached"
    assert 1 <= len(result.window_plans) <= cfg["T_SIM"]
    assert (
        len(result.window_plans[0].loss_history) < cfg["planner"]["max_iters"]
    )
    assert result.window_plans[0].hard_interval[0] >= cfg["planner"]["alpha"]
    for index, plan in enumerate(result.window_plans):
        torch.testing.assert_close(
            result.applied_controls[index], plan.controls[0]
        )
        torch.testing.assert_close(
            plan.controls, plan.rollout.aux["applied_controls"]
        )
    speeds = torch.stack([state[0][2] for state in result.states])
    assert bool(((speeds >= 0) & (speeds <= 25)).all())
    assert bool((result.applied_controls[:, 0].abs() <= 2.5).all())
    assert bool((result.applied_controls[:, 1].abs() <= 2.0).all())
    setup = setup_problem(cfg, with_environment=True)
    first_state = _lane_initial_state(setup)
    replay_rollout = lane_rollout(
        setup.dyn,
        first_state[0],
        first_state[1],
        first_state[3],
        first_state[4],
        cfg["traffic_q_std"] ** 2 * torch.eye(4),
        [car["name"] for car in cfg["traffic"]],
        cfg["speed_bounds"],
        (cfg["accel_bounds"]["longitudinal"], cfg["accel_bounds"]["lateral"]),
    )
    first = result.window_plans[0]
    spec = lane_local_window(
        setup.env, 0, first_state[0], cfg
    ).get_specification(cfg["H"])
    replay = setup.planner.evaluate_controls(
        replay_rollout, first.controls, spec=spec
    )
    torch.testing.assert_close(
        replay.rollout.aux["mean_trace"], first.rollout.aux["mean_trace"]
    )
    assert replay.hard_interval == pytest.approx(first.hard_interval, abs=1e-6)
    comparison = compare_lane_window(first, setup.env, cfg["H"])
    assert comparison["pdstl_probability_interval"] == first.hard_interval
    assert comparison["deterministic_signed_distance"] > 0
    task = build_environment(cfg).metadata["task"]
    witness = max(result.states[-1][5], task["start_end_steps"][0])
    assert witness <= task["start_end_steps"][1]
    assert len(result.window_plans) >= witness + task["dwell_steps"]


def test_deadline_missed_is_reported(lane_config):
    cfg, path = lane_config
    cfg["task"]["start_window_seconds"] = [0.0, 0.2]
    path.write_text(yaml.safe_dump(cfg))
    result = run_lane_change(str(path), max_steps=5, show=False, save=False)
    assert result.stopped_reason == "deadline_missed"
    assert len(result.window_plans) <= 2


def test_on_ramp_merge_tapers_and_completes():
    cfg = load_config("configs/scenarios/lane_merge.yaml")
    env = build_environment(cfg)
    ramp = env.metadata["ramp"]
    road = env.metadata["road"]
    assert ramp["start_x"] < ramp["end_x"]
    assert env.metadata["traffic"][0]["y"] == road["lane_divider"] * 2
    result = run_lane_change(
        "configs/scenarios/lane_merge.yaml", show=False, save=False
    )
    assert result.stopped_reason == "goal_reached"
    assert len(result.window_plans) <= cfg["T_SIM"]
    assert float(result.states[-1][0][1]) >= road["lane_divider"]
    assert result.window_plans[-1].hard_interval[0] > 0.5
