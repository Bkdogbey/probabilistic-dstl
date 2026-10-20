"""Tracking-error model on synthetic data: plumbing and recovery only.

Data drawn from the model the fit assumes cannot show whether the model
suits the hardware; that is the held-out hardware validation's job.
"""

import json
from dataclasses import replace

import minsnap
import numpy as np
import pytest
import tracking_model
from logger import align_playback, aligned_states
from simulator import TICKS, per_tick
from trajectory import reference_plan

TRUE = {
    "rho": np.array([0.7, 0.85]),
    "b": np.array([0.02, -0.01]),
    "Q": np.array([[4e-5, 1e-5], [1e-5, 3e-5]]),
    "m0": np.array([0.005, -0.003]),
    "P0": np.diag([2e-4, 1.5e-4]),
}


def errors(rng, steps, p=TRUE):
    e = [rng.multivariate_normal(p["m0"], p["P0"])]
    for _ in range(steps):
        w = rng.multivariate_normal([0.0, 0.0], p["Q"])
        e.append(p["rho"] * e[-1] + (1 - p["rho"]) * p["b"] + w)
    return np.array(e)


def synthetic_flights(count, rng, steps=90):
    """Identification flights with errors drawn from TRUE."""
    flights = []
    for i in range(count):
        meta = {
            "purpose": "identification",
            "outcome": "completed",
            "mission_outcome": "completed",
            "fan": 0,
            "trial_id": f"t{i:02d}",
        }
        flights.append({"meta": meta, "errors": errors(rng, steps)})
    return flights


@pytest.fixture
def synthetic(monkeypatch):
    monkeypatch.setattr(tracking_model, "flight_errors", lambda t: t["errors"])


def test_fit_recovers_the_model_and_held_out_coverage_is_nominal(
    cfg, synthetic
):
    cfg = replace(cfg, model={**cfg.model, "bootstrap": 50})
    report = tracking_model.fit(
        cfg, synthetic_flights(60, np.random.default_rng(0))
    )
    fan = report["fans"]["0"]
    assert fan["fit_flights"] == 40 and len(fan["held_out_trials"]) == 20
    assert not set(fan["held_out_trials"]) & set(fan["fit_trials"])
    assert np.allclose(fan["rho"], TRUE["rho"], atol=0.04)
    assert np.allclose(fan["b"], TRUE["b"], atol=0.004)
    assert np.allclose(fan["Q"], TRUE["Q"], atol=4e-6)
    assert np.allclose(fan["m0"], TRUE["m0"], atol=0.008)
    assert np.allclose(np.diag(fan["P0"]), np.diag(TRUE["P0"]), rtol=0.6)
    lo, hi = np.array(fan["intervals"]["rho"])
    assert (lo <= fan["rho"]).all() and (fan["rho"] <= hi).all()
    assert fan["b_poorly_determined"] == [False, False]
    assert report["warnings"] == []
    validation = report["validation"]["0"]
    assert validation["flights_by_step"] == [20] * 91
    for (name, (estimate, *_)), nominal in zip(
        validation["coverage"].items(), (0.393, 0.865, 0.95)
    ):
        assert abs(estimate - nominal) < 0.05, name


def test_few_flights_warn_and_a_single_flight_is_refused(cfg, synthetic):
    cfg = replace(cfg, model={**cfg.model, "bootstrap": 5})
    report = tracking_model.fit(
        cfg, synthetic_flights(3, np.random.default_rng(1))
    )
    assert report["fans"]["0"]["fit_flights"] == 2
    assert "2 fit flights" in report["warnings"][0]
    with pytest.raises(ValueError, match="at least two"):
        tracking_model.fit(cfg, synthetic_flights(1, np.random.default_rng(1)))


def test_clipped_rho_refits_the_intercept_and_the_residuals():
    k = np.arange(40.0)
    # x grows (slope 1.01 > 0.999); y alternates (slope < 0).
    e = np.stack([0.01 * 1.01**k, 0.01 * (-0.5) ** k + 0.003], 1)
    params = tracking_model.fit_errors([e, e])  # two flights: P0 exists
    x, y = e[:-1], e[1:]
    assert params["clipped"] == [True, True]
    assert params["rho"].tolist() == [0.999, 0.0]
    intercept = (y - params["rho"] * x).mean(0)
    assert np.allclose(params["b"], intercept / (1 - params["rho"]))
    residual = y - params["rho"] * x - intercept
    assert np.allclose(params["Q"], np.cov(np.vstack([residual, residual]).T))
    assert np.allclose(tracking_model.residuals(params, e), residual)


@pytest.mark.parametrize(
    "rho", [(0.8, 0.8), (0.3, 0.95), (0.0, 0.6), (1.0, 0.5)]
)
def test_tick_rate_process_subsampled_reproduces_q(rho):
    Q = np.array([[4e-5, 1e-5], [1e-5, 3e-5]])
    r, Q_h = per_tick(rho, Q, TICKS)
    assert np.allclose(r**TICKS, rho)
    subsampled = sum(np.diag(r**j) @ Q_h @ np.diag(r**j) for j in range(TICKS))
    assert np.allclose(subsampled, Q, rtol=1e-12, atol=0)
    assert not np.allclose(Q_h, Q / TICKS)  # dividing by 20 is wrong


def test_task_violations_count_up_to_the_abort_and_faults_do_not(cfg):
    def trial(**meta):
        base = {
            "purpose": "identification",
            "outcome": "completed",
            "mission_outcome": "completed",
            "fan": 0,
        }
        return {"meta": {**base, **meta}}

    kept = [
        trial(),
        trial(
            outcome="aborted",
            mission_outcome="aborted",
            abort_kind="TaskViolation",
        ),
    ]
    dropped = [
        trial(
            outcome="aborted",
            mission_outcome="aborted",
            abort_kind="DataFault",
        ),
        trial(outcome="interrupted", mission_outcome="interrupted"),
        trial(outcome="error", mission_outcome="error"),
        trial(
            outcome="aborted",
            mission_outcome="not_started",
            abort_kind="TaskViolation",
        ),
        trial(dry_run=True),
        trial(data_readiness="config.yml has measured: false"),
        trial(purpose="pilot"),
    ]
    assert [tracking_model.usable(t) for t in kept + dropped] == [True] * 2 + [
        False
    ] * len(dropped)

    # An aborted flight's data stops at the abort.
    rows = reference_plan(cfg)["coefficients"]
    t = np.arange(0.0, 5.0, 0.05)
    state = {"t_host": t, "device_ms": 1000 * t, "x": t, "y": t}
    meta = {
        "coefficients": rows,
        "dt": cfg.dt,
        "t_play": 1.0,
        "mission_t_end": 2.0,
    }
    _, _, e = aligned_states({"meta": meta, "state": state})
    assert len(e) == 6  # t = 1.0, 1.2, ..., 2.0


def test_alignment_recovers_a_shifted_playback_start(cfg):
    rows = np.asarray(reference_plan(cfg)["coefficients"])
    duration = minsnap.total_duration(rows)
    t_cmd, t_play = 100.0, 100.137
    t = np.arange(99.0, t_cmd + duration + 0.3, 0.05)
    delay = np.random.default_rng(0).uniform(0.0, 0.01, len(t))
    delay[0] = 0.0
    xyz = minsnap.evaluate(rows, t - t_play).astype(np.float32)
    target = {
        "t_host": t + delay,
        "device_ms": np.round(1000 * (t - 50.0)),
        "tx": xyz[:, 0],
        "ty": xyz[:, 1],
    }
    meta = {
        "coefficients": rows,
        "t_cmd_after": t_cmd,
        "duration": duration,
        "mission_t_end": t_cmd + duration + 0.25,
    }
    result = align_playback(meta, None, target)
    assert result["alignment"] == "ctrltarget" and result["identifiable"]
    assert abs(result["t_play"] - t_play) <= 0.002
    assert result["alignment_max"] < 1e-3
    fallback = align_playback(meta, None, None)
    assert fallback["alignment"] == "host_fallback"
    assert fallback["t_play"] == t_cmd


def test_model_prediction_approaches_the_stationary_covariance():
    model = tracking_model.make_model(0.2, {**TRUE, "clipped": None})
    mean, cov = model.predict(np.zeros((200, 2)))
    stationary = model.stationary_cov().numpy()
    assert np.allclose(cov[-1].numpy(), stationary, rtol=1e-6)
    assert np.allclose(mean[-1].numpy(), TRUE["b"])
    rho = TRUE["rho"]
    assert np.allclose(
        stationary, np.diag(rho) @ stationary @ np.diag(rho) + TRUE["Q"]
    )


@pytest.mark.parametrize("setting", [0, 6, 12, 18])
def test_model_report_loads_by_numeric_fan_setting(cfg, tmp_path, setting):
    report = {
        "id": "numeric-fan-model",
        "hash": "test",
        "dt": cfg.dt,
        "fans": {
            str(setting): {
                "rho": [0.5, 0.5],
                "b": [setting / 1000] * 2,
                "Q": (1e-4 * np.eye(2)).tolist(),
                "m0": [0, 0],
                "P0": (1e-4 * np.eye(2)).tolist(),
            }
        },
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(report))
    loaded = tracking_model.load_model(path, setting, cfg)
    assert np.allclose(loaded.b.numpy(), setting / 1000)
    with pytest.raises(ValueError, match="no fan"):
        tracking_model.load_model(path, 6 if setting != 6 else 12, cfg)
