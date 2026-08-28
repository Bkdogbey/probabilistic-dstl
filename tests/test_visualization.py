"""Tests for src/visualization/temporal.py."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure  # noqa: E402

from models.drone import always_altitude_example  # noqa: E402
import visualization.temporal as viz  # noqa: E402


def _example():
    model = always_altitude_example()
    atomic = model.probability_bounds
    temporal = atomic[:, :1, :]
    return model, atomic, temporal


def test_plot_temporal_example_returns_a_figure(monkeypatch):
    monkeypatch.setattr(viz.plt, "show", lambda: None)
    model, atomic, temporal = _example()

    fig = viz.plot_temporal_example(model, atomic, temporal, "Always[0,2]")

    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_temporal_example_does_not_call_show_when_show_is_false(monkeypatch):
    called = []
    monkeypatch.setattr(viz.plt, "show", lambda: called.append(True))
    model, atomic, temporal = _example()

    viz.plot_temporal_example(model, atomic, temporal, "Always[0,2]", show=False)

    assert called == []


def test_plot_temporal_example_calls_show_by_default(monkeypatch):
    called = []
    monkeypatch.setattr(viz.plt, "show", lambda: called.append(True))
    model, atomic, temporal = _example()

    viz.plot_temporal_example(model, atomic, temporal, "Always[0,2]")

    assert called == [True]
