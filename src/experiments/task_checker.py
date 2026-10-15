"""Deterministic STL checker for the pickup-delivery tasks, independent of pdSTL.

Implements R1-R3 directly on sampled positions (no pdSTL import), so execution
success never depends on a pdSTL score. Regions are closed boxes
(xmin, xmax, ymin, ymax); time is in steps 0..H. Until is strong and inclusive
(the left formula holds through the witness), matching the pdSTL operators.
"""

import numpy as np

FAILURES = (
    "WORKSPACE",
    "OBSTACLE",
    "NO_LOAD",
    "SCAN_BEFORE_LOAD",
    "NO_SCAN",
    "DELIVER_BEFORE_SCAN",
    "NO_DELIVERY",
)
BIT = {name: 1 << k for k, name in enumerate(FAILURES)}
REQUIRED = {
    "R1": ("WORKSPACE", "OBSTACLE", "NO_DELIVERY"),
    "R2": ("WORKSPACE", "OBSTACLE", "NO_LOAD", "NO_DELIVERY"),
    "R3": FAILURES,
}


def inside(xy, box):
    """[N, T] closed-box membership of positions [N, T, 2]."""
    xmin, xmax, ymin, ymax = box
    x, y = xy[..., 0], xy[..., 1]
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def dwell_starts(member, dwell):
    """[N, T - dwell] True where `member` holds on every step of [t, t + dwell]."""
    run = np.cumsum(np.pad(member, ((0, 0), (1, 0))), axis=1)
    return run[:, dwell + 1 :] - run[:, : -dwell - 1] == dwell + 1


def first_index(mask):
    """[N] first True index along time, or -1."""
    return np.where(mask.any(axis=1), mask.argmax(axis=1), -1)


def strong_until(left, right, end):
    """[N] first witness tau <= end with right[tau] and left on [0, tau], or -1."""
    right = right[:, : end + 1]
    held = np.cumprod(left[:, : right.shape[1]], axis=1).astype(bool)
    return first_index(right & held)


def check(xy, task, horizon, dwell, spec="R3"):
    """Check sampled trajectories against one task.

    Args:
        xy: [N, H + 1, 2] positions.
        task: dict with `workspace`, `obstacles` (list), `pickups` (dict
            name -> box), `scanner`, `delivery` boxes.
        horizon, dwell: H and d in steps.
        spec: "R1", "R2" or "R3".

    Returns:
        dict of [N] arrays: success, failures (bitmask), primary (first
        failure name or ""), pickup (name or ""), load/scan/delivery times
        (-1 when absent).
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 3 or xy.shape[1] != horizon + 1:
        raise ValueError(f"positions must be [N, {horizon + 1}, 2]")
    names = list(task["pickups"])
    member = {name: inside(xy, task["pickups"][name]) for name in names}
    starts = np.stack([dwell_starts(member[n], dwell) for n in names])
    load = starts.any(axis=0)  # L at t = 0..H-d
    scanner = inside(xy, task["scanner"])
    delivery = inside(xy, task["delivery"])

    failures = np.zeros(len(xy), dtype=int)
    failures |= BIT["WORKSPACE"] * ~inside(xy, task["workspace"]).all(axis=1)
    hit = np.zeros(xy.shape[:2], dtype=bool)
    for box in task["obstacles"]:
        hit |= inside(xy, box)
    failures |= BIT["OBSTACLE"] * hit.any(axis=1)

    any_load = first_index(load)
    ordered_load = strong_until(~scanner, load, horizon - dwell)
    scan = strong_until(~delivery, scanner, horizon)
    deliver = first_index(delivery)
    load_time = any_load if spec == "R2" else ordered_load

    failures |= BIT["NO_LOAD"] * (any_load < 0)
    failures |= BIT["NO_DELIVERY"] * (deliver < 0)
    failures |= _order_failures(any_load, ordered_load, scanner, scan)

    required = sum(BIT[name] for name in REQUIRED[spec])
    failures &= required
    pickup = np.array(
        [
            names[int(starts[:, n, t].argmax())] if t >= 0 else ""
            for n, t in enumerate(load_time)
        ]
    )
    return {
        "success": failures == 0,
        "failures": failures,
        "primary": np.array([primary_failure(f) for f in failures]),
        "pickup": pickup if spec != "R1" else np.full(len(xy), ""),
        "load_time": load_time if spec != "R1" else np.full(len(xy), -1),
        "scan_time": scan if spec == "R3" else np.full(len(xy), -1),
        "delivery_time": deliver,
    }


def _order_failures(any_load, ordered_load, scanner, scan):
    """Bits for the R3 ordering requirements (masked out for R1/R2)."""
    failures = np.zeros(len(any_load), dtype=int)
    failures |= BIT["SCAN_BEFORE_LOAD"] * (
        (any_load >= 0) & (ordered_load < 0)
    )
    scanned = scanner.any(axis=1)
    failures |= BIT["NO_SCAN"] * ~scanned
    failures |= BIT["DELIVER_BEFORE_SCAN"] * (scanned & (scan < 0))
    return failures


def primary_failure(bits):
    """First set failure in FAILURES order (a fixed reporting priority)."""
    for name in FAILURES:
        if bits & BIT[name]:
            return name
    return ""


def failure_names(bits):
    """Every requirement a bitmask marks as violated."""
    return [name for name in FAILURES if bits & BIT[name]]
