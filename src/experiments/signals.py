"""Scalar mean/variance data for the offline examples."""

import numpy as np


def piecewise_signal(values=None):
    """Return a configurable discrete scalar mean/variance signal."""
    if values is None:
        values = ((45, 4), (55, 4), (60, 4), (48, 4), (42, 9), (58, 4), (52, 4))
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("piecewise values must have shape [T, 2] as (mean, variance)")
    return np.arange(len(values), dtype=float), values[:, 0], values[:, 1]
