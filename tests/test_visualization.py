"""Tests for src/visualization/probability_bounds.py."""

import matplotlib

matplotlib.use("Agg")

import torch  # noqa: E402

from visualization.probability_bounds import plot_formula_bounds  # noqa: E402
import visualization.probability_bounds as probability_bounds  # noqa: E402


def _series():
    time = torch.arange(6)
    bounds = torch.rand(1, 6, 2).sort(dim=-1).values
    output_time = torch.arange(4)
    output_bounds = torch.rand(1, 4, 2).sort(dim=-1).values
    return [
        (time, bounds, "raw", "tab:blue"),
        (output_time, output_bounds, "reduced", "purple"),
    ]


def test_plot_formula_bounds_accepts_one_or_more_traces(monkeypatch):
    monkeypatch.setattr(probability_bounds.plt, "show", lambda: None)

    plot_formula_bounds("title", _series()[:1])
    plot_formula_bounds("title", _series())


def test_plot_formula_bounds_does_not_call_show_when_show_is_false(monkeypatch):
    called = []
    monkeypatch.setattr(probability_bounds.plt, "show", lambda: called.append(True))

    plot_formula_bounds("title", _series(), show=False)

    assert called == []


def test_plot_formula_bounds_calls_show_when_show_is_true(monkeypatch):
    called = []
    monkeypatch.setattr(probability_bounds.plt, "show", lambda: called.append(True))

    plot_formula_bounds("title", _series(), show=True)

    assert called == [True]
