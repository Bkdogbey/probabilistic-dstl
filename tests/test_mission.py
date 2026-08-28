"""Verification of the composed Always-and-Eventually mission graph."""

import matplotlib.pyplot as plt
import torch

from experiments.mission import run_mission_example


def test_mission_matches_hand_calculated_branch_and_conjunction_bounds():
    _, offline, online_mission, updates, figure = run_mission_example(
        (0, 2), show=False
    )
    always_bounds, eventually_bounds, mission_bounds = offline
    dtype = mission_bounds.dtype

    expected_always = torch.tensor(
        [
            [0.85, 0.98],
            [0.82, 0.97],
            [0.59, 0.85],
            [0.57, 0.85],
            [0.59, 0.85],
            [0.83, 0.97],
            [0.85, 0.98],
            [0.85, 0.98],
            [0.85, 0.98],
        ],
        dtype=dtype,
    ).unsqueeze(0)
    expected_eventually = torch.tensor(
        [
            [0.05, 0.24],
            [0.06, 0.27],
            [0.75, 1.00],
            [0.75, 1.00],
            [0.75, 1.00],
            [0.07, 0.33],
            [0.08, 0.36],
            [0.09, 0.39],
            [0.10, 0.42],
        ],
        dtype=dtype,
    ).unsqueeze(0)
    expected_mission = torch.tensor(
        [
            [0.00, 0.24],
            [0.00, 0.27],
            [0.34, 0.85],
            [0.32, 0.85],
            [0.34, 0.85],
            [0.00, 0.33],
            [0.00, 0.36],
            [0.00, 0.39],
            [0.00, 0.42],
        ],
        dtype=dtype,
    ).unsqueeze(0)

    torch.testing.assert_close(always_bounds, expected_always)
    torch.testing.assert_close(eventually_bounds, expected_eventually)
    torch.testing.assert_close(mission_bounds, expected_mission)
    torch.testing.assert_close(online_mission, expected_mission)
    assert all(update.mission_output is None for update in updates[:2])
    assert all(update.mission_output is not None for update in updates[2:])
    plt.close(figure)


def test_mission_streaming_states_fill_then_slide_independently():
    _, _, _, updates, figure = run_mission_example((0, 2), show=False)
    expected_lengths = [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3]

    assert [update.always_state.shape[1] for update in updates] == expected_lengths
    assert [update.eventually_state.shape[1] for update in updates] == expected_lengths
    assert len(figure.axes) == 3
    plt.close(figure)
