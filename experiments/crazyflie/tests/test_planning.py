"""Config, environment, planners (S-path and pdSTL) and plan storage."""

import copy

import numpy as np
import pytest
import yaml

import config
import planners
import plots
import run
import trajectory

# --- configuration ---------------------------------------------------------


def test_shipped_config_is_valid():
    cfg = config.load()
    assert cfg["radio"]["uri"].startswith("radio://")
    assert config.load(method="pdstl")["planning"]["method"] == "pdstl"


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda c: c["flight"].pop("land_speed"), "missing"),
        (lambda c: c["planning"]["pdstl"].pop("H"), "planning.pdstl"),
        (lambda c: c["bounds"].update(x=[1.0, 0.0]), "bounds.x"),
        (lambda c: c["planning"]["s_path"].update(speed=0), "s_path.speed"),
        (lambda c: c["flight"].update(transit_z=2.0), "transit_z"),
        (lambda c: c["environment"].pop("rings"), "missing"),
        (lambda c: c["planning"].update(goal=[1.0, -2.0]), "must differ"),
    ],
)
def test_invalid_config_rejected(tmp_path, edit, message):
    c = config.load()
    edit(c)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match=message):
        config.load(path)


# --- S-path and trajectory checks --------------------------------------------


def test_s_path_is_deterministic_and_flyable():
    cfg = config.load()
    p, sp = cfg["planning"], cfg["planning"]["s_path"]
    t, xyz, info = planners.s_path(cfg)
    t2, xyz2, _ = planners.s_path(cfg)
    assert info == {"method": "s_path"}
    assert np.array_equal(t, t2) and np.array_equal(xyz, xyz2)
    assert np.allclose(xyz[:, 2], p["height"])
    assert np.allclose(xyz[0, :2], p["start"])
    assert np.allclose(xyz[-1, :2], p["goal"])
    assert np.allclose(np.diff(t[:-1]), sp["dt"])
    assert np.allclose(trajectory.speeds(t, xyz), sp["speed"], rtol=0.02)
    assert np.ptp(xyz[:, 0]) == pytest.approx(2 * sp["amplitude"], abs=0.01)
    run.check(cfg, t, xyz)


def test_s_path_follows_any_start_goal_line():
    cfg = config.load()
    cfg["planning"].update(start=[0.0, -1.5], goal=[1.2, -0.3])
    t, xyz, _ = planners.s_path(cfg)
    assert np.allclose(xyz[0, :2], [0.0, -1.5])
    assert np.allclose(xyz[-1, :2], [1.2, -0.3])
    speed = trajectory.speeds(t, xyz)
    assert np.allclose(speed, cfg["planning"]["s_path"]["speed"], rtol=0.02)


def test_check_rejects_bad_trajectories():
    cfg = config.load()
    bounds, vmax = cfg["bounds"], cfg["flight"]["max_speed"]
    t, xyz, _ = planners.s_path(cfg)
    outside = xyz.copy()
    outside[5, 0] = 5.0
    with pytest.raises(ValueError, match="x bounds"):
        trajectory.check(t, outside, bounds, vmax)
    with pytest.raises(ValueError, match="speed"):
        trajectory.check(t * 0.1, xyz, bounds, vmax)
    with pytest.raises(ValueError, match="increasing"):
        trajectory.check(
            np.r_[t[:3], t[2:]], np.r_[xyz[:3], xyz[2:]], bounds, vmax
        )
    with pytest.raises(ValueError, match="t=0"):
        trajectory.check(t + 1.0, xyz, bounds, vmax)


def _env(**edits):
    env = copy.deepcopy(config.load()["environment"])
    for group, value in edits.items():
        env[group].update(value)
    return env


def test_obstacles_only_enforced_when_enabled():
    cfg = config.load()
    bounds, vmax = cfg["bounds"], cfg["flight"]["max_speed"]
    t, xyz, _ = planners.s_path(cfg)
    trajectory.check(t, xyz, bounds, vmax, cfg["environment"])  # disabled
    env = _env(obstacles={"enabled": True})
    with pytest.raises(ValueError, match="obs_1"):
        trajectory.check(t, xyz, bounds, vmax, env)
    high = xyz.copy()
    high[:, 2] = 0.7  # above every box top + clearance
    trajectory.check(t, high, bounds, vmax, env)


def test_rings_must_be_passed_in_order():
    cfg = config.load()
    bounds, vmax = cfg["bounds"], cfg["flight"]["max_speed"]
    t, xyz, _ = planners.s_path(cfg)
    regions = cfg["environment"]["rings"]["regions"]
    rings_on = _env(rings={"enabled": True})
    trajectory.check(t, xyz, bounds, vmax, rings_on)
    reverse = _env(rings={"enabled": True, "regions": regions[::-1]})
    with pytest.raises(ValueError, match="out of order"):
        trajectory.check(t, xyz, bounds, vmax, reverse)
    moved = copy.deepcopy(regions)
    moved[1]["center"] = [0.0, -1.0, 0.5]
    with pytest.raises(ValueError, match="misses ring_2"):
        trajectory.check(
            t,
            xyz,
            bounds,
            vmax,
            _env(rings={"enabled": True, "regions": moved}),
        )
    # A coarse plan (e.g. pdSTL at dt 0.5 s) is densified before checking.
    keep = np.unique(np.r_[0 : len(t) : 7, len(t) - 1])
    trajectory.check(t[keep], xyz[keep], bounds, vmax, rings_on)


def test_csv_round_trip_and_sampling(tmp_path):
    t, xyz, _ = planners.s_path(config.load())
    trajectory.save_csv(tmp_path / "plan.csv", t, xyz)
    t2, xyz2 = trajectory.load_csv(tmp_path / "plan.csv")
    assert np.allclose(t, t2) and np.allclose(xyz, xyz2)
    assert np.allclose(trajectory.at(t, xyz, -1.0), xyz[0])
    assert np.allclose(trajectory.at(t, xyz, t[-1] + 5.0), xyz[-1])
    mid = 0.5 * (t[3] + t[4])
    assert np.allclose(trajectory.at(t, xyz, mid), 0.5 * (xyz[3] + xyz[4]))


# --- pdSTL ---------------------------------------------------------------------


def test_pdstl_scenario_without_rings_is_reach_only():
    scenario, route = planners.pdstl_scenario(config.load())
    assert scenario["visit_regions"] == [] and scenario["obstacles"] == []
    assert route == [[1.0, 0.0]]
    assert scenario["x0_mean"] == [1.0, -2.0]
    assert scenario["bounds"] == {
        "x_range": [-0.5, 1.5],
        "y_range": [-2.0, 0.5],
    }


def test_pdstl_scenario_maps_rings_and_obstacles():
    cfg = config.load()
    cfg["environment"]["rings"]["enabled"] = True
    cfg["environment"]["obstacles"]["enabled"] = True
    scenario, route = planners.pdstl_scenario(cfg)
    rings = cfg["environment"]["rings"]["regions"]
    visits = scenario["visit_regions"]
    assert [v["name"] for v in visits] == ["ring_1", "ring_2", "ring_3"]
    assert route[:3] == [r["center"][:2] for r in rings]
    windows = [v["interval"] for v in visits]
    assert (
        windows[0][0] == 1 and windows[-1][1] == cfg["planning"]["pdstl"]["H"]
    )
    for (_, end), (start, _) in zip(windows, windows[1:]):
        assert end <= start  # ordered, non-overlapping
    # Square inscribed in each ring's disc at the plan height.
    half = np.diff(visits[0]["x_range"])[0] / 2
    assert half == pytest.approx(rings[0]["radius"] / np.sqrt(2))
    # Every box top + clearance is above 0.5 m, so all four are avoided,
    # inflated by the clearance.
    assert [o["name"] for o in scenario["obstacles"]] == [
        "obs_1",
        "obs_2",
        "obs_3",
        "obs_4",
    ]
    assert scenario["obstacles"][0]["x_range"] == pytest.approx([0.58, 1.14])
    cfg["planning"]["height"] = 0.7
    cfg["environment"]["rings"]["enabled"] = False
    assert planners.pdstl_scenario(cfg)[0]["obstacles"] == []


def test_pdstl_rejects_ring_out_of_reach():
    cfg = config.load()
    cfg["environment"]["rings"]["enabled"] = True
    cfg["planning"]["height"] = 0.8  # 0.3 m above the rings, radius 0.15
    with pytest.raises(ValueError, match="cannot be passed"):
        planners.pdstl_scenario(cfg)


def test_pdstl_plan_through_rings(cfg):
    cfg["environment"]["rings"]["enabled"] = True
    cfg["planning"]["method"] = "pdstl"
    t, xyz, info = planners.plan(cfg)
    pd = cfg["planning"]["pdstl"]
    assert xyz.shape == (pd["H"] + 1, 3)
    assert np.allclose(t, np.arange(pd["H"] + 1) * pd["dt"])
    assert np.allclose(xyz[0], [*cfg["planning"]["start"], 0.5])
    assert np.allclose(xyz[:, 2], cfg["planning"]["height"])
    assert 0.0 <= info["p_lower"] <= info["p_upper"] <= 1.0
    assert len(info["cov_xy"]) == pd["H"] + 1
    assert 0.0 <= info["monte_carlo"]["rate"] <= 1.0
    run.check(cfg, t, xyz)  # bounds, speed and ring order


def test_unknown_method_rejected():
    cfg = config.load()
    cfg["planning"]["method"] = "magic"
    with pytest.raises(ValueError, match="unknown planning.method"):
        planners.plan(cfg)


# --- plan storage and sync -----------------------------------------------------


def test_signature_tracks_plan_inputs_only():
    cfg = config.load()
    base = planners.signature(cfg)
    other = copy.deepcopy(cfg)
    other["radio"]["uri"] = "radio://0/80/2M/E7E7E7E7E7"
    other["positioning"]["timeout_s"] = 99
    assert planners.signature(other) == base
    for edit in (
        lambda c: c["planning"].update(height=0.6),
        lambda c: c["environment"]["rings"].update(enabled=True),
        lambda c: c["bounds"].update(margin=0.2),
        lambda c: c["flight"].update(max_speed=0.4),
    ):
        changed = copy.deepcopy(cfg)
        edit(changed)
        assert planners.signature(changed) != base


def test_saved_plan_is_reused_until_config_changes(cfg, monkeypatch):
    t, xyz, info, directory = run.get_plan(cfg)
    assert (directory / "plan.png").exists()
    loaded = planners.load_plan(directory)
    assert np.allclose(loaded[1], xyz) and loaded[2] == info

    monkeypatch.setattr(planners, "plan", lambda c: pytest.fail("replanned"))
    assert np.allclose(run.get_plan(cfg)[1], xyz)  # in sync: reused

    calls = []
    monkeypatch.setattr(
        planners, "plan", lambda c: calls.append(1) or planners.s_path(c)
    )
    cfg["planning"]["s_path"]["amplitude"] = 0.2
    run.get_plan(cfg)  # out of sync: planned again
    assert calls == [1]


def test_invalid_plan_is_never_saved(cfg):
    cfg["planning"]["s_path"]["amplitude"] = 0.8  # x beyond 1.5 m
    with pytest.raises(ValueError, match="x bounds"):
        run.get_plan(cfg)
    directory = planners.plan_dir(cfg, config.resolve(cfg, "plans_dir"))
    assert planners.load_plan(directory) is None


def test_plan_figures(cfg, tmp_path):
    cfg["environment"]["rings"]["enabled"] = True
    t, xyz, info = planners.s_path(cfg)
    plots.plot_plan(cfg, t, xyz, info, tmp_path / "s")
    cov = [np.eye(2).tolist()] * len(t)
    pd_info = {
        "method": "pdstl",
        "p_lower": 0.5,
        "p_upper": 0.9,
        "cov_xy": cov,
    }
    plots.plot_plan(cfg, t, xyz, pd_info, tmp_path / "p")
    for stem in ("s", "p"):
        assert (tmp_path / f"{stem}.png").exists()
        assert (tmp_path / f"{stem}.pdf").exists()
