"""End-to-end checks for offline sliding and incremental temporal examples."""

import matplotlib.pyplot as plt
import torch

from experiments.streaming import (
    run_sliding_always_example,
    run_streaming_always_animation,
    run_streaming_always_example,
    run_streaming_eventually_example,
)
from models.streaming import sliding_always_example, sliding_eventually_example


def test_streaming_models_have_eleven_valid_probability_intervals():
    for trace in (sliding_always_example(), sliding_eventually_example()):
        assert trace.time.shape == (11,)
        assert trace.bounds.shape == (1, 11, 2)
        assert bool((trace.bounds[..., 0] <= trace.bounds[..., 1]).all())
        assert bool((trace.bounds >= 0).all())
        assert bool((trace.bounds <= 1).all())


def test_offline_sliding_always_matches_hand_calculated_windows():
    _, temporal, figure = run_sliding_always_example((0, 5), show=False)
    expected = torch.tensor(
        [
            [0.42, 0.85],
            [0.41, 0.85],
            [0.42, 0.85],
            [0.42, 0.85],
            [0.44, 0.85],
            [0.68, 0.97],
        ],
        dtype=temporal.dtype,
    ).unsqueeze(0)

    torch.testing.assert_close(temporal, expected)
    plt.close(figure)


def test_streaming_always_state_fills_then_remains_six_entries():
    _, offline, online, updates, figure = run_streaming_always_example(
        (0, 5), show=False
    )

    assert [update.window_state.shape[1] for update in updates] == [
        1,
        2,
        3,
        4,
        5,
        6,
        6,
        6,
        6,
        6,
        6,
    ]
    assert all(update.output is None for update in updates[:5])
    assert all(update.output is not None for update in updates[5:])
    torch.testing.assert_close(online, offline)
    plt.close(figure)


def test_streaming_always_discards_the_expired_entry():
    trace, _, _, updates, figure = run_streaming_always_example(
        (0, 5), show=False
    )

    torch.testing.assert_close(updates[5].window_state, trace.bounds[:, 0:6, :])
    torch.testing.assert_close(updates[6].window_state, trace.bounds[:, 1:7, :])
    torch.testing.assert_close(updates[10].window_state, trace.bounds[:, 5:11, :])
    plt.close(figure)


def test_streaming_eventually_matches_hand_calculated_windows():
    _, offline, online, updates, figure = run_streaming_eventually_example(
        (0, 5), show=False
    )
    expected = torch.tensor(
        [
            [0.75, 1.00],
            [0.75, 1.00],
            [0.75, 1.00],
            [0.75, 1.00],
            [0.75, 1.00],
            [0.10, 0.75],
        ],
        dtype=offline.dtype,
    ).unsqueeze(0)

    torch.testing.assert_close(offline, expected)
    torch.testing.assert_close(online, expected)
    assert [update.output is not None for update in updates] == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    plt.close(figure)


def test_streaming_figures_show_every_completed_output():
    _, temporal, sliding_figure = run_sliding_always_example((0, 5), show=False)
    _, offline, online, _, streaming_figure = run_streaming_always_example(
        (0, 5), show=False
    )

    assert len(sliding_figure.axes[1].lines[0].get_xdata()) == temporal.shape[1]
    comparison = streaming_figure.axes[-1]
    assert len(comparison.lines[0].get_xdata()) == offline.shape[1]
    assert comparison.collections[0].get_offsets().shape[0] == online.shape[1]
    plt.close(sliding_figure)
    plt.close(streaming_figure)


def test_streaming_animation_contains_state_and_output_panels():
    *_, figure, movie = run_streaming_always_animation(show=False)

    assert len(figure.axes) == 2
    assert "Window filling: 1/3" in figure.axes[0].get_title()
    assert movie.event_source.interval == 900
    movie._draw_was_started = True
    plt.close(figure)
