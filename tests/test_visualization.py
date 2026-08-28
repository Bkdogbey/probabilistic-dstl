"""Tests for src/visualization/temporal.py."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure  # noqa: E402
import torch  # noqa: E402

import visualization.temporal as viz  # noqa: E402


def _signal():
    time = torch.arange(6)
    mean = torch.tensor([58.0, 56.0, 52.0, 54.0, 57.0, 60.0])
    std = torch.tensor([1.5, 1.5, 2.0, 1.5, 1.2, 1.0])
    return time, mean, std


def _bounds(t):
    return torch.rand(1, t, 2).sort(dim=-1).values


def test_plot_predicates_and_boolean_returns_a_figure(monkeypatch):
    monkeypatch.setattr(viz.plt, "show", lambda: None)
    time, mean, std = _signal()

    fig = viz.plot_predicates_and_boolean(
        time, mean, std, _bounds(6), _bounds(6), [("~a", _bounds(6), "tab:green")]
    )

    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_temporal_operator_returns_a_figure_with_shorter_output_axis(monkeypatch):
    monkeypatch.setattr(viz.plt, "show", lambda: None)
    time, mean, std = _signal()

    fig = viz.plot_temporal_operator(
        time, mean, std, 50, _bounds(6), _bounds(4), time[:4], "Always[0,2]"
    )

    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_online_window_returns_a_figure():
    snapshots = [(0, 0, False), (1, 0, False), (2, 0, True), (3, 1, True)]

    fig = viz.plot_online_window(torch.arange(6), _bounds(6), snapshots, show=False)

    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_functions_do_not_call_show_when_show_is_false(monkeypatch):
    called = []
    monkeypatch.setattr(viz.plt, "show", lambda: called.append(True))
    time, mean, std = _signal()

    viz.plot_predicates_and_boolean(
        time, mean, std, _bounds(6), _bounds(6), [("~a", _bounds(6), "tab:green")], show=False
    )
    viz.plot_temporal_operator(time, mean, std, 50, _bounds(6), _bounds(4), time[:4], "Always", show=False)
    viz.plot_online_window(time, _bounds(6), [(0, 0, False)], show=False)

    assert called == []


def test_plot_functions_call_show_by_default(monkeypatch):
    called = []
    monkeypatch.setattr(viz.plt, "show", lambda: called.append(True))
    time, mean, std = _signal()

    viz.plot_predicates_and_boolean(
        time, mean, std, _bounds(6), _bounds(6), [("~a", _bounds(6), "tab:green")]
    )

    assert called == [True]
