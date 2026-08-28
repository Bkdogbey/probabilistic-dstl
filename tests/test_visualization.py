"""Tests for the offline temporal-example visualization."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt

from models.drone import always_altitude_example
from pdstl import Always, OfflineSource, Predicate
from visualization.temporal import plot_temporal_example


def _evaluated_example(interval):
    model = always_altitude_example()
    predicate = Predicate("altitude >= 50 m")
    source = OfflineSource({predicate: model.probability_bounds})
    atomic = predicate(source)
    temporal = Always(predicate, interval)(source)
    formula = f"Always[{interval[0]},{interval[1]}]({predicate})"
    return model, atomic, temporal, formula


def test_plot_returns_figure_with_every_temporal_anchor():
    model, atomic, temporal, formula = _evaluated_example((0, 2))

    figure = plot_temporal_example(
        model,
        atomic,
        temporal,
        (0, 2),
        formula,
        show=False,
    )

    assert isinstance(figure, matplotlib.figure.Figure)
    assert len(figure.axes) == 3
    assert len(figure.axes[1].lines[0].get_xdata()) == 7
    assert len(figure.axes[1].lines[1].get_xdata()) == 7
    assert len(figure.axes[2].lines[0].get_xdata()) == 5
    assert len(figure.axes[2].lines[1].get_xdata()) == 5
    plt.close(figure)


def test_plot_handles_an_empty_temporal_result():
    interval = (0, 10)
    model, atomic, temporal, formula = _evaluated_example(interval)

    figure = plot_temporal_example(
        model,
        atomic,
        temporal,
        interval,
        formula,
        show=False,
    )

    assert temporal.shape == (1, 0, 2)
    assert isinstance(figure, matplotlib.figure.Figure)
    assert "No complete windows" in figure.axes[2].texts[0].get_text()
    plt.close(figure)
