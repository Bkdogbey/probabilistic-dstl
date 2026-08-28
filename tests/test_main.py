"""Tests for src/main.py: the pdSTL demonstration pipeline."""

import importlib

import torch

from pdstl import OnlineSource

import pytest

import main as main_module
from main import example_always, example_boolean, example_predicate, run_examples


@pytest.mark.parametrize("example", [example_predicate, example_boolean, example_always])
def test_example_runs_without_raising_and_prints_output(example, capsys):
    example()

    captured = capsys.readouterr()
    assert captured.out.strip() != ""


def test_always_matches_the_independent_reference():
    state = run_examples()

    torch.testing.assert_close(state["always_safe_bounds"], state["reference_always"])


def test_eventually_matches_the_independent_reference():
    state = run_examples()

    torch.testing.assert_close(state["eventually_goal_bounds"], state["reference_eventually"])


def test_mission_conjunction_matches_the_independent_reference():
    state = run_examples()

    torch.testing.assert_close(state["mission_bounds"], state["reference_mission"])


def test_output_shapes_are_one_ten_two():
    state = run_examples()

    assert state["always_safe_bounds"].shape == (1, 10, 2)
    assert state["eventually_goal_bounds"].shape == (1, 10, 2)
    assert state["mission_bounds"].shape == (1, 10, 2)


def test_online_outputs_equal_offline_prefixes():
    state = run_examples()
    safe, goal = state["safe"], state["goal"]
    safe_bounds, goal_bounds = state["safe_bounds"], state["goal_bounds"]

    online = OnlineSource()
    for k in range(safe_bounds.shape[1]):
        online.append({safe: safe_bounds[:, k, :], goal: goal_bounds[:, k, :]})

        for formula, offline_bounds in (
            (state["always_safe"], state["always_safe_bounds"]),
            (state["eventually_goal"], state["eventually_goal_bounds"]),
            (state["mission"], state["mission_bounds"]),
        ):
            online_bounds = formula(online)
            t = online_bounds.shape[1]
            torch.testing.assert_close(online_bounds, offline_bounds[:, :t, :])


def test_importing_main_has_no_side_effects(capsys):
    importlib.reload(main_module)

    captured = capsys.readouterr()
    assert captured.out == ""
