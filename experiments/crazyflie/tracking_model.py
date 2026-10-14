"""Tracking model s' = A s + B u + c_f + w, w ~ N(0, Q_f): fit, validate, load.

State s = [px, py, vx, vy], control u = commanded world velocity. Each
axis is a first-order velocity lag (time constant tau, gain g) under a
zero-order hold, discretized exactly at dt, so position integrates the
velocity consistently:

    a  = exp(-dt / tau)
    v' = a v + g (1 - a) u
    p' = p + tau (1 - a) v + g (dt - tau (1 - a)) u

tau and g are shared across fan settings; the bias c_f, residual
covariance Q_f and initial belief are per fan.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from config import HERE
from logger import aligned_states

# isort: split  (config puts the pdSTL core in src/ on sys.path)
from models.beliefs import interval_probability
from models.dynamics import Dynamics

REPORTS = HERE / "reports"
COVERAGE = {
    "d2<=1 (39.3%)": 1.0,
    "d2<=4 (86.5%)": 4.0,
    "d2<=5.991 (95%)": 5.991,
}


def response_matrices(dt, tau, gain):
    A, B = np.eye(4), np.zeros((4, 2))
    for i in (0, 1):
        a = math.exp(-dt / tau[i])
        A[i, 2 + i], A[2 + i, 2 + i] = tau[i] * (1 - a), a
        B[i, i] = gain[i] * (dt - tau[i] * (1 - a))
        B[2 + i, i] = gain[i] * (1 - a)
    return A, B


class TrackingModel(Dynamics):
    """Linear Gaussian tracking model with a constant bias c."""

    def __init__(
        self,
        dt,
        tau,
        gain,
        c,
        Q,
        initial,
        u_max,
        report_id=None,
        report_hash=None,
    ):
        super().__init__(dt, u_max)
        A, B = response_matrices(dt, tau, gain)

        def f32(x):
            return torch.as_tensor(np.asarray(x), dtype=torch.float32)

        for name, value in {"A": A, "B": B, "c": c, "Q": Q}.items():
            self.register_buffer(name, f32(value))
        self.initial = (f32(initial[0]), f32(initial[1]))
        self.report_id, self.report_hash = report_id, report_hash

    def step(self, x, P, u):
        return self.A @ x + self.B @ u + self.c, self.A @ P @ self.A.T + self.Q

    def predict(self, controls, mean0=None, cov0=None):
        """Mean [N+1, 4] and covariance [N+1, 4, 4] under physical controls."""
        mean = (
            self.initial[0]
            if mean0 is None
            else torch.as_tensor(mean0, dtype=torch.float32)
        )
        cov = (
            self.initial[1]
            if cov0 is None
            else torch.as_tensor(cov0, dtype=torch.float32)
        )
        means, covs = [mean], [cov]
        for u in torch.as_tensor(controls, dtype=torch.float32):
            mean, cov = self.step(mean, cov, u)
            means.append(mean)
            covs.append(cov)
        return torch.stack(means), torch.stack(covs)


# Fitting


def split(trials, holdout_every):
    """Identification flights per fan, in time order: (fit, held_out)."""
    fit, held = {}, {}
    usable = [
        t
        for t in trials
        if t["meta"]["purpose"] == "identification"
        and not t["meta"].get("dry_run")
        and t["meta"]["outcome"] == "completed"
    ]
    for fan in sorted({t["meta"]["fan"] for t in usable}):
        ordered = [t for t in usable if t["meta"]["fan"] == fan]
        held[fan] = ordered[holdout_every - 1 :: holdout_every]
        fit[fan] = [t for t in ordered if t not in held[fan]]
    return fit, held


def fit(cfg, trials):
    """Fit tau, g (shared) and c_f, Q_f, initial belief (per fan); validate."""
    from planner import Task

    fit_trials, held = split(trials, cfg.model["holdout_every"])
    data = {
        fan: [aligned_states(t) for t in ts] for fan, ts in fit_trials.items()
    }
    fans = sorted(data)
    if not fans or min(len(d) for d in data.values()) < 2:
        raise ValueError("need at least two fitting flights per fan")

    tau, gain = [], []
    for i in (0, 1):  # velocity regression with a per-fan intercept
        X, y = [], []
        for j, fan in enumerate(fans):
            for s, u in data[fan]:
                onehot = np.zeros((len(u), len(fans)))
                onehot[:, j] = 1
                X.append(np.column_stack([s[:-1, 2 + i], u[:, i], onehot]))
                y.append(s[1:, 2 + i])
        (a, b, *_), *_ = np.linalg.lstsq(
            np.vstack(X), np.concatenate(y), rcond=None
        )
        if not 0 < a < 1:
            raise ValueError(
                f"axis {'xy'[i]}: velocity is not a stable lag (a = {a:.3f})"
            )
        tau.append(-cfg.dt / math.log(a))
        gain.append(b / (1 - a))
    A, B = response_matrices(cfg.dt, tau, gain)

    report = {
        "dt": cfg.dt,
        "tau": tau,
        "gain": gain,
        "config_hash": cfg.hash,
        "fans": {},
    }
    for fan in fans:
        r = np.vstack([s[1:] - s[:-1] @ A.T - u @ B.T for s, u in data[fan]])
        starts = np.array([s[0] for s, _ in data[fan]])
        report["fans"][fan] = {
            "c": r.mean(0).tolist(),
            "Q": np.cov(r, rowvar=False).tolist(),
            "initial_mean": starts.mean(0).tolist(),
            "initial_cov": np.cov(starts, rowvar=False).tolist(),
            "fit_trials": [t["meta"]["trial_id"] for t in fit_trials[fan]],
            "held_out_trials": [t["meta"]["trial_id"] for t in held[fan]],
        }
    params = json.dumps(report, sort_keys=True)
    report["hash"] = hashlib.sha256(params.encode()).hexdigest()[:12]
    report["id"] = "model_" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    rng = np.random.default_rng(cfg.model["seed"])
    task = Task(cfg)
    report["validation"] = {
        fan: validate(
            model_from(report, fan, cfg.u_max),
            [aligned_states(t) for t in held[fan]],
            task,
            cfg.model["bootstrap"],
            rng,
        )
        for fan in fans
        if held[fan]
    }
    return report


def save_report(report):
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"{report['id']}.json"
    path.write_text(json.dumps(report, indent=1))
    return path


def model_from(report, fan, u_max):
    f = report["fans"][fan]
    return TrackingModel(
        report["dt"],
        report["tau"],
        report["gain"],
        f["c"],
        f["Q"],
        (f["initial_mean"], f["initial_cov"]),
        u_max,
        report["id"],
        report["hash"],
    )


def load_model(path, fan, cfg):
    report = json.loads(Path(path).read_text())
    if report["dt"] != cfg.dt:
        raise ValueError(f"model dt {report['dt']} != planning dt {cfg.dt}")
    if fan not in report["fans"]:
        raise ValueError(f"model {report['id']} has no fan {fan!r}")
    return model_from(report, fan, cfg.u_max)


# Validation on held-out flights


def _bootstrap(per_flight, statistic, samples, rng):
    """Estimate and 95% interval, resampling whole flights."""
    value = statistic(per_flight)
    draws = [
        statistic(
            [
                per_flight[i]
                for i in rng.integers(len(per_flight), size=len(per_flight))
            ]
        )
        for _ in range(samples)
    ]
    return [float(value), *map(float, np.percentile(draws, [2.5, 97.5]))]


def _flight_checks(model, s, u, task):
    """Per-flight arrays used by every validation statistic.

    The prediction starts from the model's initial belief, exactly as a
    plan's does, so coverage checks the probabilities plans rely on.
    """
    with torch.no_grad():
        mean, cov = (x.numpy() for x in model.predict(u))
    err = s[:, :2] - mean[:, :2]
    inv = np.linalg.inv(cov[1:, :2, :2])
    d2 = np.einsum("ki,kij,kj->k", err[1:], inv, err[1:])
    A, B, c = (x.numpy() for x in (model.A, model.B, model.c))
    residual = s[1:] - s[:-1] @ A.T - u @ B.T - c
    probs, hits = [], []
    for box in [task.workspace, *task.obstacles, task.goal]:
        for axis, (lo, hi) in enumerate((box.x, box.y)):
            p = interval_probability(
                lo,
                hi,
                torch.tensor(mean[1:, axis]),
                torch.tensor(cov[1:, axis, axis]),
            )
            probs.append(p.numpy())
            hits.append((lo <= s[1:, axis]) & (s[1:, axis] <= hi))
    return {
        "error": np.linalg.norm(err, axis=1),
        "spread": np.sqrt(np.trace(cov[:, :2, :2], axis1=1, axis2=2)),
        "d2": d2,
        "residual": residual,
        "xy": s[1:, :2],
        "p": np.concatenate(probs),
        "hit": np.concatenate(hits),
    }


def validate(model, flights, task, samples, rng):
    """Held-out checks: horizon error, spread, residuals, coverage, calibration."""
    checks = [_flight_checks(model, s, u, task) for s, u in flights]

    def pooled(fs, key):
        return np.concatenate([f[key] for f in fs])

    def interval(statistic):
        return _bootstrap(checks, statistic, samples, rng)

    n = min(len(f["error"]) for f in checks)
    residuals = [
        f["residual"] - pooled(checks, "residual").mean(0) for f in checks
    ]
    lag1 = sum((r[:-1] * r[1:]).sum(0) for r in residuals) / sum(
        (r**2).sum(0) for r in residuals
    )
    thirds = np.array_split(np.arange(n - 1), 3)
    centre = np.array(task.workspace.centre)
    quadrant = np.concatenate(
        [
            2 * (f["xy"][:, 0] > centre[0]) + (f["xy"][:, 1] > centre[1])
            for f in checks
        ]
    )
    residual = pooled(checks, "residual")
    p, hit = pooled(checks, "p"), pooled(checks, "hit")
    bins = np.minimum((p * 10).astype(int), 9)
    return {
        "flights": len(checks),
        "error_by_step": np.mean([f["error"][:n] for f in checks], 0).tolist(),
        "spread_by_step": np.mean(
            [f["spread"][:n] for f in checks], 0
        ).tolist(),
        "final_error": interval(
            lambda fs: np.mean([f["error"][n - 1] for f in fs])
        ),
        "coverage": {
            name: interval(lambda fs, th=th: np.mean(pooled(fs, "d2") <= th))
            for name, th in COVERAGE.items()
        },
        "residual_bias": residual.mean(0).tolist(),
        "residual_lag1_corr": lag1.tolist(),
        "residual_std_by_time": [
            np.vstack([f["residual"][i] for f in checks]).std(0).tolist()
            for i in thirds
        ],
        "residual_std_by_quadrant": {
            int(q): residual[quadrant == q].std(0).tolist()
            for q in np.unique(quadrant)
        },
        "calibration": [  # [mean predicted, observed frequency, count] per bin
            [
                float(p[bins == b].mean()),
                float(hit[bins == b].mean()),
                int((bins == b).sum()),
            ]
            for b in range(10)
            if (bins == b).any()
        ],
        "brier": interval(
            lambda fs: np.mean((pooled(fs, "p") - pooled(fs, "hit")) ** 2)
        ),
    }
