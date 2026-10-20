"""The min-snap curve, its uploaded rows and the uav_trajectories format."""

import minsnap
import numpy as np
import torch
from trajectory import make_curve

W = np.array([[0.0, 0.0], [0.5, 0.2], [0.8, 0.9], [0.3, 1.2]])
D = [1.5, 2.0, 1.0]


def exact_rows():
    """Unquantized rows, for the smoothness checks."""
    K = minsnap.solve_matrix(D)
    return minsnap.to_rows(D, K @ torch.tensor(W), 0.3)


def horner(p, t):
    """uav_trajectories' Polynomial.eval."""
    x = 0.0
    for i in range(len(p)):
        x = x * t + p[len(p) - 1 - i]
    return x


def test_curve_passes_through_the_waypoints():
    rows = make_curve(W, D, 0.3)
    knots = np.concatenate([[0.0], np.cumsum(D)])
    xyz = minsnap.evaluate(rows, knots)
    assert np.abs(xyz[:, :2] - W).max() < 1e-5
    assert np.allclose(xyz[:, 2], 0.3)
    assert minsnap.problems(rows, W) == []


def test_curve_is_at_rest_at_both_ends():
    rows = exact_rows()
    for order in (1, 2, 3):
        start, end = minsnap.piece_ends(rows, order)
        assert np.abs(start[0]).max() < 1e-9 and np.abs(end[-1]).max() < 1e-9


def test_curve_is_smooth_at_the_knots():
    rows = exact_rows()
    for order in range(7):
        start, end = minsnap.piece_ends(rows, order)
        tolerance = 1e-8 if order <= 4 else 1e-4  # C5, C6 loosely
        assert np.abs(end[:-1] - start[1:]).max() < tolerance, order


def test_coefficients_are_differentiable_in_the_waypoints():
    K = minsnap.solve_matrix(D)
    S = minsnap.sample_matrix(D, np.linspace(0, sum(D), 17))
    w = torch.tensor(W, requires_grad=True)
    assert torch.autograd.gradcheck(lambda w: S @ (K @ w), (w,))


def test_csv_round_trip_is_exact(tmp_path):
    rows = make_curve(W, D, 0.3)
    path = tmp_path / "curve.csv"
    minsnap.write_csv(path, rows)
    header = path.read_text().splitlines()[0].split(",")
    assert len(header) == 33 and header[:2] == ["duration", "x^0"]
    assert np.array_equal(minsnap.read_csv(path), rows)


def test_positions_match_the_uav_trajectories_evaluator():
    rows = make_curve(W, D, 0.3)
    start = 0.0
    for row in rows:
        for tau in np.linspace(0, row[0], 7)[:-1]:  # the end is the next piece
            expected = [
                horner(row[1 + 8 * a : 9 + 8 * a], tau) for a in range(3)
            ]
            assert np.allclose(
                minsnap.evaluate(rows, start + tau)[0], expected, atol=1e-12
            )
        start += row[0]


def test_float32_rounding_moves_positions_less_than_10_microns():
    rows = exact_rows()
    t = minsnap.dense_times(rows)
    moved = minsnap.evaluate(minsnap.quantize(rows), t) - minsnap.evaluate(
        rows, t
    )
    assert np.abs(moved).max() < 1e-5
