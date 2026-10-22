"""Inspection and dock: levels, the nested rule's logic, route detection."""

import numpy as np
import pytest
import torch

from experiments import inspection_dock as m
from models.rollouts import create_gaussian_belief_trajectory


@pytest.fixture(scope="module")
def cfg():
    return m.load()


def _holds(formula, path):
    """Exact verdict of a formula on one deterministic [T, 2] path."""
    states = torch.cat([path, torch.zeros_like(path)], dim=1)[None]
    beliefs = create_gaussian_belief_trajectory(
        states, torch.zeros_like(states)
    )
    return bool(formula(beliefs)[0, 0, 0] > 0.5)


def _path(cfg, points, dwell=None):
    """Piecewise-linear [H+1, 2] path through points; optional pause."""
    points = np.asarray(points, float)
    arc = np.r_[0, np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
    n = cfg["H"] + 1
    if dwell is not None:  # (index of point, steps to wait there)
        index, wait = dwell
        stop = arc[index] / arc[-1]
        s = np.linspace(0, 1, n - wait)
        s = np.r_[s[s <= stop], np.full(wait, stop), s[s > stop]]
    else:
        s = np.linspace(0, 1, n)
    xy = np.column_stack(
        [np.interp(s * arc[-1], arc, points[:, k]) for k in (0, 1)]
    )
    return torch.tensor(xy[:n], dtype=torch.float32)


def test_levels_give_the_stated_steady_tracking_error(cfg):
    for level, sigma in cfg["uncertainty"]["tracking_std_m"].items():
        s = m.evaluation_problem(cfg, level)
        mean, cov = s.state
        for _ in range(300):
            mean, cov = s.dyn.step(mean, cov, torch.zeros(2))
        assert cov[1, 1].sqrt().item() == pytest.approx(sigma, rel=1e-3)


def test_rule_needs_inspection_after_the_contaminated_gap(cfg):
    _, parts = m.specification(cfg)
    gap = [[0.55, 2.5], [3.0, 3.9], [4.2, 3.9]]
    a, dock = [8.1, 4.1], [9.35, 5.6]
    i = [5.9, 3.8]
    inspect = m.steps(cfg, cfg["times"]["inspect_s"]) + 2
    skipped = _path(cfg, [*gap, [6.0, 2.0], a, dock])
    inspected = _path(cfg, [*gap, i, a, dock], dwell=(3, inspect))
    assert not _holds(parts["rule"], skipped)
    assert _holds(parts["rule"], inspected)
    assert _holds(parts["visit"], inspected)
    assert _holds(parts["dock"], inspected)


def test_clean_detour_never_triggers_the_rule(cfg):
    _, parts = m.specification(cfg)
    detour = _path(
        cfg,
        [[0.55, 2.5], [1.5, 8.0], [3.6, 9.3], [6.9, 7.6], [9.35, 5.6]],
    )
    assert _holds(parts["rule"], detour)
    assert _holds(parts["visit"], detour)  # passes B


def test_visiting_neither_a_nor_b_before_the_dock_fails(cfg):
    _, parts = m.specification(cfg)
    straight = _path(cfg, [[0.55, 2.5], [1.5, 9.3], [9.35, 9.3], [9.35, 5.6]])
    assert not _holds(parts["visit"], straight)


@pytest.mark.parametrize(
    ("route", "expected"),
    [
        ("shortcut", ("shortcut", "A", True)),
        ("detour", ("detour", "B", False)),
    ],
)
def test_each_initial_guess_takes_its_route(cfg, route, expected):
    s = m.problem(cfg, "stlcg", None)
    controls = m.initial_guess(s.cfg, s.dyn, cfg["routes"][route])
    result = s.planner.evaluate_controls(s.rollout, controls, spec=s.spec)
    mean = result.rollout.aux["mean_trace"][0].numpy()
    assert m.describe(cfg, mean) == expected
    assert result.hard_interval[0] == pytest.approx(1.0)


def test_report_builds_table_and_figure_from_saved_results(
    cfg, tmp_path, monkeypatch
):
    import shutil

    saved = m.OUTPUTS
    if not (saved / "selected.csv").exists():
        pytest.skip("no saved results; run `inspection_dock run` first")
    for name in ("selected.csv", "plans.pt"):
        shutil.copy(saved / name, tmp_path / name)
    monkeypatch.setattr(m, "OUTPUTS", tmp_path)
    m.report(cfg)
    for name in ("table.csv", "table.md", "figure.png", "figure.pdf"):
        assert (tmp_path / name).stat().st_size > 0
