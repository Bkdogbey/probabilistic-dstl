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
from config import HERE, fan_setting
from logger import aligned_states
from scipy.optimize import least_squares, minimize

# isort: split  (config puts the pdSTL core in src/ on sys.path)
from models.beliefs import interval_probability
from models.dynamics import Dynamics

REPORTS = HERE / "reports"
MIN_VELOCITY_VARIANCE = (
    1e-8  # (1e-4 m/s)^2 per step: never a degenerate covariance
)
START_COV = torch.diag(
    torch.tensor([1e-4, 1e-4, 0.05**2, 0.05**2])
)  # measured start: ~1 cm, at rest
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
        a_max,
        report_id=None,
        report_hash=None,
    ):
        super().__init__(dt, u_max)
        self.a_max = a_max
        A, B = response_matrices(dt, tau, gain)

        def f32(x):
            return torch.as_tensor(np.asarray(x), dtype=torch.float32)

        for name, value in {"A": A, "B": B, "c": c, "Q": Q}.items():
            self.register_buffer(name, f32(value))
        self.initial = (f32(initial[0]), f32(initial[1]))
        self.report_id, self.report_hash = report_id, report_hash

    def controls(self, params):
        """Physical controls [T, 2] from unconstrained parameters.

        Speed is bounded by u_max (tanh) and the change per step by
        a_max * dt, starting from rest. A sequence already within both
        limits maps to itself, so planners can re-evaluate their output.
        """
        step = self.a_max * self.dt
        u, out = torch.zeros_like(params[0]), []
        for target in self.bound_control(params):
            u = u + torch.clamp(target - u, -step, step)
            out.append(u)
        return torch.stack(out)

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
        and not t["meta"].get("data_readiness")
        and t["meta"]["outcome"] == "completed"
    ]
    for fan in sorted(
        {str(fan_setting(t["meta"]["fan"])) for t in usable}, key=int
    ):
        ordered = [
            t for t in usable if str(fan_setting(t["meta"]["fan"])) == fan
        ]
        held[fan] = ordered[holdout_every - 1 :: holdout_every]
        fit[fan] = [t for t in ordered if t not in held[fan]]
    return fit, held


def _response(dt, tau, gain, drift, u, p0):
    """Positions [N, 2] of the velocity-lag model driven by u [N, 2] from rest.

    `drift` is a steady velocity offset [m/s]. This is `response_matrices`
    rolled out, written per axis so it can be fitted quickly.
    """
    tau, gain, drift = map(np.asarray, (tau, gain, drift))
    a = np.exp(-dt / tau)
    lag = tau * (1 - a)
    v, p, out = np.zeros(2), np.asarray(p0, float), []
    for uk in u:
        p = p + lag * v + gain * (dt - lag) * uk + (dt - lag) * drift
        v = a * v + gain * (1 - a) * uk + (1 - a) * drift
        out.append(p)
    return np.array(out)


def _fit_response(cfg, data):
    """tau, gain (shared) and per-fan drift, by matching measured positions.

    The onboard velocity estimate is too noisy (about 0.15 m/s) to regress
    on, so the model is simulated from rest over each whole flight and its
    positions are matched to the measured ones (robust least squares).
    """
    fans = sorted(data, key=int)

    def unpack(theta):
        return (
            theta[:2],
            theta[2:4],
            {f: theta[4 + 2 * j : 6 + 2 * j] for j, f in enumerate(fans)},
        )

    def residuals(theta):
        tau, gain, drift = unpack(theta)
        return np.concatenate(
            [
                (
                    _response(cfg.dt, tau, gain, drift[f], u, s[0, :2])
                    - s[1:, :2]
                ).ravel()
                for f in fans
                for s, u in data[f]
            ]
        )

    k = 2 * len(fans)
    fitted = least_squares(
        residuals,
        x0=[0.5, 0.5, 1.0, 1.0] + [0.0] * k,
        bounds=(
            [0.05] * 2 + [0.3] * 2 + [-0.3] * k,
            [3.0] * 2 + [1.7] * 2 + [0.3] * k,
        ),
        loss="soft_l1",
        f_scale=0.05,
    )
    return unpack(fitted.x)


def _velocity_noise(dt, tau, errors, v0_var=0.05**2):
    """Per-step velocity process variance of one axis, by maximum likelihood.

    `errors` is a list of measured-minus-simulated position traces. The
    position variance is propagated through the model and an unknown
    measurement variance is added, so measurement noise is not absorbed
    into the process noise.
    """
    a = math.exp(-dt / tau)
    A = np.array([[1.0, tau * (1 - a)], [0.0, a]])
    # Supplied trajectories may have different horizons. Pool
    # only observations that exist at each step, without padding residuals.
    by_step = [
        np.array([e[k] for e in errors if k < len(e)])
        for k in range(max(map(len, errors)))
    ]

    def nll(log_params):
        q, r = np.exp(log_params)
        P, total = np.diag([0.0, v0_var]), 0.0
        for observed in by_step:
            P = A @ P @ A.T + np.diag([1e-6, q])
            var = P[0, 0] + r
            total += np.sum(np.log(var) + observed**2 / var)
        return total

    best = minimize(nll, np.log([1e-4, 1e-4]), method="Nelder-Mead")
    return float(max(np.exp(best.x[0]), MIN_VELOCITY_VARIANCE))


def fit(cfg, trials):
    """Fit tau, g (shared) and drift c_f, Q_f, initial belief (per fan); validate."""
    from geometry import Task

    fit_trials, held = split(trials, cfg.model["holdout_every"])
    data = {
        fan: [aligned_states(t) for t in ts] for fan, ts in fit_trials.items()
    }
    fans = sorted(data, key=int)
    if not fans or min(len(d) for d in data.values()) < 2:
        raise ValueError("need at least two fitting flights per fan")

    tau, gain, drift = _fit_response(cfg, data)
    a = np.exp(-cfg.dt / tau)
    lag = tau * (1 - a)
    report = {
        "dt": cfg.dt,
        "tau": tau.tolist(),
        "gain": gain.tolist(),
        "config_hash": cfg.hash,
        "fans": {},
    }
    for fan in fans:
        errors = [
            s[1:, :2] - _response(cfg.dt, tau, gain, drift[fan], u, s[0, :2])
            for s, u in data[fan]
        ]
        q = [
            _velocity_noise(cfg.dt, tau[i], [e[:, i] for e in errors])
            for i in (0, 1)
        ]
        # The belief the planner starts from: flights that begin at the start.
        starts = np.array(
            [
                s[0, :2]
                for s, _ in data[fan]
                if np.linalg.norm(s[0, :2] - cfg.start) < 0.3
            ]
        )
        var = (
            np.maximum(starts.var(axis=0), 0.01**2)
            if len(starts) > 1
            else np.full(2, 0.01**2)
        )
        report["fans"][fan] = {
            "drift": drift[fan].tolist(),
            "c": np.concatenate(
                [(cfg.dt - lag) * drift[fan], (1 - a) * drift[fan]]
            ).tolist(),
            "Q": np.diag([1e-6, 1e-6, *q]).tolist(),
            "initial_mean": [*starts.mean(axis=0), 0.0, 0.0]
            if len(starts)
            else [*cfg.start, 0.0, 0.0],
            "initial_cov": np.diag([*var, 0.05**2, 0.05**2]).tolist(),
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
            model_from(report, fan, cfg),
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


def model_from(report, fan, cfg):
    # JSON object keys are strings; older fan-off reports used the key 'off'.
    key = str(fan_setting(fan))
    f = report["fans"].get(
        key, report["fans"].get("off") if key == "0" else None
    )
    if f is None:
        raise ValueError(f"model {report['id']} has no fan {fan!r}")
    return TrackingModel(
        report["dt"],
        report["tau"],
        report["gain"],
        f["c"],
        f["Q"],
        (f["initial_mean"], f["initial_cov"]),
        cfg.u_max,
        cfg.a_max,
        report["id"],
        report["hash"],
    )


def load_model(path, fan, cfg):
    report = json.loads(Path(path).read_text())
    if report["dt"] != cfg.dt:
        raise ValueError(f"model dt {report['dt']} != planning dt {cfg.dt}")
    return model_from(report, fan, cfg)


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


WINDOW = 5  # plan steps (1 s) per residual: long enough to rise above position noise


def _flight_checks(model, s, u, task):
    """Per-flight arrays used by every validation statistic.

    The prediction starts at the flight's measured position (at rest, as the
    start gate ensures) with the initial uncertainty, like a plan's does.
    Residuals are the increments of the open-loop position error over
    WINDOW steps. The measured velocity is too noisy to be a regressor.
    """
    mean0 = [*s[0, :2], 0.0, 0.0]
    with torch.no_grad():
        mean, cov = (x.numpy() for x in model.predict(u, mean0, START_COV))
    err = s[:, :2] - mean[:, :2]
    inv = np.linalg.inv(cov[1:, :2, :2])
    d2 = np.einsum("ki,kij,kj->k", err[1:], inv, err[1:])
    residual = np.diff(err[::WINDOW], axis=0)
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
        "residual_xy": s[WINDOW::WINDOW, :2][: len(residual)],
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

    n = max(len(f["error"]) for f in checks)

    def by_step(key):
        return [
            float(np.mean([f[key][k] for f in checks if k < len(f[key])]))
            for k in range(n)
        ]

    residual = pooled(checks, "residual")
    centred = [f["residual"] - residual.mean(0) for f in checks]
    lag1 = sum((r[:-1] * r[1:]).sum(0) for r in centred) / sum(
        (r**2).sum(0) for r in centred
    )
    thirds = [np.array_split(f["residual"], 3) for f in checks]
    centre = np.array(task.workspace.centre)
    xy = pooled(checks, "residual_xy")
    quadrant = 2 * (xy[:, 0] > centre[0]) + (xy[:, 1] > centre[1])
    p, hit = pooled(checks, "p"), pooled(checks, "hit")
    bins = np.minimum((p * 10).astype(int), 9)
    return {
        "flights": len(checks),
        "error_by_step": by_step("error"),
        "spread_by_step": by_step("spread"),
        "flights_by_step": [
            sum(k < len(f["error"]) for f in checks) for k in range(n)
        ],
        "final_error": interval(
            lambda fs: np.mean([f["error"][-1] for f in fs])
        ),
        "coverage": {
            name: interval(lambda fs, th=th: np.mean(pooled(fs, "d2") <= th))
            for name, th in COVERAGE.items()
        },
        "residual_window_s": WINDOW * model.dt,
        "residual_bias": residual.mean(0).tolist(),
        "residual_lag1_corr": lag1.tolist(),
        "residual_std_by_time": [
            np.vstack([t[i] for t in thirds]).std(0).tolist() for i in range(3)
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
