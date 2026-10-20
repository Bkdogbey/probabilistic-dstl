"""Minimum-snap curves: degree-7 pieces through xy waypoints, fixed durations.

Coefficients are lowest order first (c[0] is the constant term), as in
cflib's Poly4D and in uav_trajectories CSV rows:
duration, x0..x7, y0..y7, z0..z7, yaw0..yaw7 (33 columns).

For fixed durations the curve is linear in the waypoints, c = K w, so K is
solved once and the curve stays differentiable in the waypoints.
"""

import csv
import math

import numpy as np
import torch

N = 8  # coefficients per polynomial (degree 7)
PIECE_BYTES = 132  # 33 float32 per uploaded piece
MAX_PIECES = 31  # 4096-byte trajectory memory // 132; rechecked at upload
DENSE_HZ = 100
WAYPOINT_TOL = 1e-4  # m
KNOT_TOL = 1e-4  # derivative jumps, in SI units of each order
HEADER = ["duration"] + [
    f"{a}^{j}" for a in ("x", "y", "z", "yaw") for j in range(N)
]


def basis(t, order=0):
    """[..., 8]: the order-th derivative of [1, t, ..., t^7] at times t."""
    t = torch.as_tensor(t, dtype=torch.float64).unsqueeze(-1)
    scale = torch.tensor(
        [math.perm(j, order) for j in range(N)], dtype=torch.float64
    )
    power = (torch.arange(N) - order).clamp(min=0)
    return scale * t**power


def cost_matrix(T):
    """Integral of the squared 4th derivative over [0, T], as c^T Q c."""
    Q = torch.zeros(N, N, dtype=torch.float64)
    for j in range(4, N):
        for k in range(4, N):
            p = j + k - 7
            Q[j, k] = math.perm(j, 4) * math.perm(k, 4) * T**p / p
    return Q


def snap_matrix(durations):
    return torch.block_diag(*[cost_matrix(float(T)) for T in durations])


def solve_matrix(durations):
    """K [8M, M+1] with per-axis coefficients c = K @ waypoints.

    Minimizes the snap energy subject to: positions at both ends of every
    piece, 1st-3rd derivatives zero at the start and the end, 1st-4th
    derivatives continuous at interior knots. One KKT solve.
    """
    M = len(durations)
    n = N * M
    rows, picks = [], []  # constraint rows A and which waypoint each fixes

    def row(piece, t, order):
        r = torch.zeros(n, dtype=torch.float64)
        r[N * piece : N * piece + N] = basis(t, order)
        return r

    for i, T in enumerate(durations):
        rows += [row(i, 0.0, 0), row(i, T, 0)]
        picks += [i, i + 1]
    for order in (1, 2, 3):
        rows += [row(0, 0.0, order), row(M - 1, durations[-1], order)]
        picks += [None, None]
    for i in range(M - 1):
        for order in (1, 2, 3, 4):
            rows.append(row(i, durations[i], order) - row(i + 1, 0.0, order))
            picks.append(None)

    A = torch.stack(rows)
    B = torch.zeros(len(rows), M + 1, dtype=torch.float64)
    for r, pick in enumerate(picks):
        if pick is not None:
            B[r, pick] = 1.0
    zeros = torch.zeros(len(rows), len(rows), dtype=torch.float64)
    kkt = torch.cat(
        [
            torch.cat([2 * snap_matrix(durations), A.T], 1),
            torch.cat([A, zeros], 1),
        ]
    )
    rhs = torch.cat([torch.zeros(n, M + 1, dtype=torch.float64), B])
    return torch.linalg.solve(kkt, rhs)[:n]


def sample_matrix(durations, times, order=0):
    """S [len(times), 8M] with per-axis samples S @ c; times clamp to [0, T]."""
    durations = torch.as_tensor(durations, dtype=torch.float64)
    knots = torch.cat(
        [torch.zeros(1, dtype=torch.float64), durations.cumsum(0)]
    )
    t = torch.as_tensor(times, dtype=torch.float64).clamp(
        0.0, float(knots[-1])
    )
    piece = (torch.searchsorted(knots, t, right=True) - 1).clamp(
        0, len(durations) - 1
    )
    S = torch.zeros(len(t), N * len(durations), dtype=torch.float64)
    values = basis(t - knots[piece], order)
    for k in range(N):
        S[torch.arange(len(t)), N * piece + k] = values[:, k]
    return S


def to_rows(durations, xy, altitude):
    """[M, 33] rows from xy coefficients [8M, 2]: z fixed, yaw zero."""
    M = len(durations)
    xy = np.asarray(xy, dtype=float).reshape(M, N, 2)
    z = np.zeros((M, N))
    z[:, 0] = altitude
    return np.column_stack(
        [
            np.asarray(durations, float),
            xy[..., 0],
            xy[..., 1],
            z,
            np.zeros((M, N)),
        ]
    )


def quantize(rows):
    """What the drone receives: every value rounded to float32."""
    return np.asarray(rows, dtype=np.float32).astype(float)


def axes(rows):
    """Durations [M] and per-axis coefficients [3, 8M] (x, y, z)."""
    rows = np.asarray(rows, dtype=float)
    c = rows[:, 1:25].reshape(len(rows), 3, N)
    return rows[:, 0], c.transpose(1, 0, 2).reshape(3, -1)


def evaluate(rows, times, order=0):
    """xyz (or a derivative) [len(times), 3]; times outside [0, T] hold the ends."""
    durations, c = axes(rows)
    S = sample_matrix(durations, np.atleast_1d(times), order)
    return (S @ torch.as_tensor(c.T)).numpy()


def total_duration(rows):
    return float(np.sum(np.asarray(rows)[:, 0]))


def dense_times(rows):
    """Samples at DENSE_HZ plus every piece boundary."""
    knots = np.concatenate([[0.0], np.cumsum(np.asarray(rows)[:, 0])])
    return np.union1d(np.arange(0.0, knots[-1], 1.0 / DENSE_HZ), knots)


def grid_times(rows, dt):
    return dt * np.arange(round(total_duration(rows) / dt) + 1)


def piece_ends(rows, order):
    """Each piece's xyz derivative at its start and at its end: [M, 3] each."""
    rows = np.asarray(rows, dtype=float)
    c = rows[:, 1:25].reshape(len(rows), 3, N)
    start = c @ basis(0.0, order).numpy()
    end = np.einsum("mj,maj->ma", basis(rows[:, 0], order).numpy(), c)
    return start, end


def waypoint_error(rows, waypoints):
    """Largest distance [m] from a piece end to its waypoint, per axis."""
    start, end = piece_ends(rows, 0)
    w = np.asarray(waypoints, dtype=float)
    return max(
        np.abs(start[:, :2] - w[:-1]).max(), np.abs(end[:, :2] - w[1:]).max()
    )


def problems(rows, waypoints):
    """Checks of an uploaded (quantized) record; [] when it is sound."""
    rows = np.asarray(rows, dtype=float)
    if not (rows[:, 0] > 0).all():
        return ["piece durations must be positive"]
    out = []
    miss = waypoint_error(rows, waypoints)
    if miss > WAYPOINT_TOL:
        out.append(f"curve misses a waypoint by {miss:.2e} m")
    for order in range(5):
        start, end = piece_ends(rows, order)
        jump = np.abs(end[:-1] - start[1:]).max(initial=0.0)
        if jump > KNOT_TOL:
            out.append(f"derivative {order} jumps by {jump:.2e} at a knot")
        if (
            order
            and order <= 3
            and max(np.abs(start[0]).max(), np.abs(end[-1]).max()) > KNOT_TOL
        ):
            out.append(f"derivative {order} is not zero at the ends")
    return out


def write_csv(path, rows):
    """uav_trajectories format: one header row, then 33 columns per piece."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(np.asarray(rows, dtype=float).tolist())


def read_csv(path):
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
