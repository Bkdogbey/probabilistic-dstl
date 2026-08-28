"""Tests for src/main.py: the three independent pdSTL experiments."""

import importlib

import torch

from models.boolean import boolean_example
from models.drone import always_altitude_example, eventually_altitude_example
from pdstl import And, Always, Eventually, Not, OfflineSource, Or, Predicate

import main as main_module
from main import run_always_example, run_boolean_example, run_eventually_example


# ---------------------------------------------------------------------------
# Boolean
# ---------------------------------------------------------------------------


def test_predicate_a_returns_its_supplied_bounds():
    bounds_a, bounds_b = boolean_example()
    a, b = Predicate("A"), Predicate("B")
    source = OfflineSource({a: bounds_a, b: bounds_b})

    torch.testing.assert_close(a(source), bounds_a)


def test_predicate_b_returns_its_supplied_bounds():
    bounds_a, bounds_b = boolean_example()
    a, b = Predicate("A"), Predicate("B")
    source = OfflineSource({a: bounds_a, b: bounds_b})

    torch.testing.assert_close(b(source), bounds_b)


def test_not_a():
    bounds_a, bounds_b = boolean_example()
    a, b = Predicate("A"), Predicate("B")
    source = OfflineSource({a: bounds_a, b: bounds_b})

    torch.testing.assert_close(Not(a)(source), torch.tensor([[[0.10, 0.40]]], dtype=bounds_a.dtype))


def test_a_and_b():
    bounds_a, bounds_b = boolean_example()
    a, b = Predicate("A"), Predicate("B")
    source = OfflineSource({a: bounds_a, b: bounds_b})

    torch.testing.assert_close(And(a, b)(source), torch.tensor([[[0.30, 0.90]]], dtype=bounds_a.dtype))


def test_a_or_b():
    bounds_a, bounds_b = boolean_example()
    a, b = Predicate("A"), Predicate("B")
    source = OfflineSource({a: bounds_a, b: bounds_b})

    torch.testing.assert_close(Or(a, b)(source), torch.tensor([[[0.70, 1.00]]], dtype=bounds_a.dtype))


# ---------------------------------------------------------------------------
# Always
# ---------------------------------------------------------------------------


def test_always_matches_the_explicit_hand_calculation():
    model = always_altitude_example()
    predicate = Predicate("altitude >= 50 m")
    source = OfflineSource({predicate: model.probability_bounds})

    p = predicate(source)[0, :, 0]
    expected = torch.stack([torch.clamp(p.sum() - 2, min=0.0), p.amin()]).reshape(1, 1, 2)

    torch.testing.assert_close(Always(predicate, (0, 2))(source), expected)


def test_always_of_three_certain_events_is_one():
    predicate = Predicate("certain")
    bounds = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]])
    source = OfflineSource({predicate: bounds})

    result = Always(predicate, (0, 2))(source)

    torch.testing.assert_close(result, torch.tensor([[[1.0, 1.0]]]))


# ---------------------------------------------------------------------------
# Eventually
# ---------------------------------------------------------------------------


def test_eventually_matches_the_explicit_hand_calculation():
    model = eventually_altitude_example()
    predicate = Predicate("altitude >= 55 m")
    source = OfflineSource({predicate: model.probability_bounds})

    p = predicate(source)[0, :, 0]
    expected = torch.stack([p.amax(), torch.clamp(p.sum(), max=1.0)]).reshape(1, 1, 2)

    torch.testing.assert_close(Eventually(predicate, (0, 2))(source), expected)


def test_eventually_of_one_certain_event_is_one():
    predicate = Predicate("certain-once")
    bounds = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]])
    source = OfflineSource({predicate: bounds})

    result = Eventually(predicate, (0, 2))(source)

    torch.testing.assert_close(result, torch.tensor([[[1.0, 1.0]]]))


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def test_run_boolean_example_runs_independently():
    run_boolean_example()


def test_run_always_example_runs_independently():
    run_always_example()


def test_run_eventually_example_runs_independently():
    run_eventually_example()


def test_importing_main_has_no_side_effects(capsys):
    importlib.reload(main_module)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_skipped_experiments_are_not_executed(monkeypatch):
    called = []
    monkeypatch.setattr(main_module, "run_boolean_example", lambda: called.append("boolean"))
    monkeypatch.setattr(main_module, "run_always_example", lambda: called.append("always"))
    monkeypatch.setattr(main_module, "run_eventually_example", lambda: called.append("eventually"))

    main_module.main()

    assert called == ["boolean"]
