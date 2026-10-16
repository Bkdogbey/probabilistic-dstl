import numpy as np
import pytest

from model import fit, residuals, tracking_error


def flight(xy, v, ref=None, dt=0.05):
    """A flight record like flight.py writes, from positions and commands."""
    n = len(xy)
    ref = xy if ref is None else ref
    names = ["t", "x", "y", "rx", "ry", "vx", "vy"]
    cols = [np.arange(n) * dt, xy[:, 0], xy[:, 1], ref[:, 0], ref[:, 1],
            v[:, 0], v[:, 1]]
    return np.rec.fromarrays(cols, names=names)


def integrate(v, dt=0.05, noise=None):
    steps = v * dt + (0 if noise is None else noise)
    return np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])[:-1]


def test_perfect_tracking_has_no_residual():
    v = np.tile([0.2, -0.1], (200, 1))
    w = residuals(flight(integrate(v), v), dt=0.2)
    assert len(w) > 40
    assert np.allclose(w, 0, atol=1e-9)


def test_constant_drift_shows_as_bias():
    v = np.tile([0.2, 0.0], (200, 1))
    xy = integrate(v + [0.0, 0.05])  # pushed sideways at 5 cm/s
    f = fit(residuals(flight(xy, v), dt=0.2))
    assert f["bias"] == pytest.approx([0.0, 0.05 * 0.2], abs=1e-9)


def test_noise_level_is_recovered():
    rng = np.random.default_rng(0)
    v = np.zeros((20000, 2))
    sigma = 0.01  # per 0.05 s row, so 0.02 per 0.2 s step
    xy = integrate(v, noise=rng.normal(0, sigma, (20000, 2)))
    f = fit(residuals(flight(xy, v), dt=0.2))
    assert f["q_std"] == pytest.approx(0.02, rel=0.05)


def test_tracking_error_is_sampled_every_step():
    xy = np.zeros((100, 2))
    ref = xy + [0.1, -0.05]
    e = tracking_error(flight(xy, np.zeros((100, 2)), ref), dt=0.2)
    assert e.shape == (25, 2)
    assert np.allclose(e, [-0.1, 0.05])
