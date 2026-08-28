"""End-to-end checks for the safe-until-goal example."""

import matplotlib.pyplot as plt
import torch

from experiments.until import run_until_example


def test_until_example_matches_hand_calculated_bounds():
    _, candidates, offline, online, updates, figure = run_until_example(
        (1, 2), show=False
    )
    expected = torch.tensor(
        [
            [0.00, 0.17],
            [0.00, 0.19],
            [0.62, 0.98],
            [0.68, 0.97],
            [0.00, 0.21],
            [0.00, 0.23],
            [0.02, 0.25],
            [0.04, 0.27],
            [0.03, 0.29],
        ],
        dtype=offline.dtype,
    ).unsqueeze(0)

    torch.testing.assert_close(offline, expected)
    torch.testing.assert_close(online, expected)
    assert candidates.shape == (1, 9, 2, 2)
    assert all(update.output is None for update in updates[:2])
    assert all(update.output is not None for update in updates[2:])
    plt.close(figure)


def test_until_example_retains_two_named_three_entry_states():
    _, _, _, _, updates, figure = run_until_example((1, 2), show=False)
    expected_lengths = [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3]

    assert [update.state.left.shape[1] for update in updates] == expected_lengths
    assert [update.state.right.shape[1] for update in updates] == expected_lengths
    assert len(figure.axes) == 3
    plt.close(figure)
