"""Stable stochastic tracking-error model: fit, validate, load.

The onboard controller flies the uploaded curve r(t). On the planning grid
t_k = t_play + k dt the time-aligned tracking error e_k = p_k - r(t_k) is
modelled, per fan setting, as

    e[k+1] = R e[k] + (I - R) b + w[k],   R = diag(rho_x, rho_y),
    w ~ N(0, Q) (full 2x2),   e[0] ~ N(m0, P0).

This is an empirical error model, not a physical model of the drone. Q is
the effective innovation covariance: it also absorbs state-estimation error
and model mismatch, so it does not isolate the physical fan noise. With
every |rho| < 1 the covariance approaches the stationary P_inf, which
solves P_inf = R P_inf R^T + Q. A Gaussian error still has unbounded
support. Whether real flights behave this way is what the held-out and
transfer checks test.

Assumption: at a given fan setting the error statistics are the same for
every candidate path, so a plan only moves r. This is a strong assumption.
The residual diagnostics (signed reference velocity, acceleration and
position relative to the fan) check it; they are reported, never fitted.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import minsnap
import numpy as np
import torch
from config import HERE, fan_setting
from logger import aligned_states
from trajectory import file_hash

# isort: split  (config puts the pdSTL core in src/ on sys.path)
from models.beliefs import interval_probability

REPORTS = HERE / "reports"
RHO_RANGE = (0.0, 0.999)
RHO_NEAR_ONE = 0.95  # above this, b = c / (1 - rho) is poorly determined
P0_FLOOR = 0.01**2  # m^2, on the initial variances
COVERAGE = {
    "d2<=1 (39.3%)": 1.0,
    "d2<=4 (86.5%)": 4.0,
    "d2<=5.991 (95%)": 5.991,
}


class TrackingModel:
    """The error model of one fan setting; beliefs are r + m_k, P_k."""

    def __init__(
        self, dt, rho, b, Q, m0, P0, report_id=None, report_hash=None
    ):
        def tensor(x):
            return torch.as_tensor(np.asarray(x, float), dtype=torch.float64)

        self.dt = dt
        self.rho, self.b, self.Q = tensor(rho), tensor(b), tensor(Q)
        self.m0, self.P0 = tensor(m0), tensor(P0)
        self.report_id, self.report_hash = report_id, report_hash

    def error_moments(self, steps):
        """Error mean m_k [steps+1, 2] and covariance P_k [steps+1, 2, 2]."""
        R = torch.diag(self.rho)
        m, P = self.m0, self.P0
        means, covs = [m], [P]
        for _ in range(steps):
            m = self.rho * m + (1 - self.rho) * self.b
            P = R @ P @ R.T + self.Q
            means.append(m)
            covs.append(P)
        return torch.stack(means), torch.stack(covs)

    def predict(self, reference):
        """Belief mean r_k + m_k [N, 2] and covariance P_k for r [N, 2]."""
        reference = torch.as_tensor(reference, dtype=torch.float64)
        m, P = self.error_moments(len(reference) - 1)
        return reference + m, P

    def stationary_cov(self):
        """P_inf = R P_inf R^T + Q; elementwise for diagonal R."""
        return self.Q / (1 - torch.outer(self.rho, self.rho))


# Which flights count


def usable(trial):
    """Identification flights that completed or aborted on a task violation.

    Task-violation aborts count up to the abort: dropping large deviations
    would understate the noise. Data faults, errors and interruptions do not.
    """
    meta = trial["meta"]
    mission = meta.get("mission_outcome")
    return (
        meta["purpose"] == "identification"
        and not meta.get("dry_run")
        and not meta.get("data_readiness")
        and (
            mission == "completed"
            or (
                mission == "aborted"
                and meta.get("abort_kind") == "TaskViolation"
            )
        )
    )


def by_fan(trials):
    groups = defaultdict(list)
    for t in trials:
        if usable(t):
            groups[str(fan_setting(t["meta"]["fan"]))].append(t)
    return dict(sorted(groups.items(), key=lambda item: int(item[0])))


def split(trials, holdout_every):
    """Usable flights per fan, in time order: (fit, held_out)."""
    fit, held = {}, {}
    for fan, ordered in by_fan(trials).items():
        held[fan] = ordered[holdout_every - 1 :: holdout_every]
        fit[fan] = [t for t in ordered if t not in held[fan]]
    return fit, held


def flight_data(trial):
    """Error e, reference r and its velocity and acceleration on the grid."""
    _, r, e = aligned_states(trial)
    rows = np.asarray(trial["meta"]["coefficients"], dtype=float)
    t = trial["meta"]["dt"] * np.arange(len(e))
    return {
        "e": e,
        "r": r,
        "v": minsnap.evaluate(rows, t, 1)[:, :2],
        "a": minsnap.evaluate(rows, t, 2)[:, :2],
    }


# Fitting


def fit_errors(errors):
    """rho, b, Q, m0, P0 from the errors [N_i, 2] of whole flights.

    Per axis, e[k+1] is regressed on [e[k], 1]. A slope outside RHO_RANGE
    is clipped and the intercept refitted with rho fixed, so b and Q
    describe the model actually used for prediction.
    """
    x = np.concatenate([e[:-1] for e in errors])
    y = np.concatenate([e[1:] for e in errors])
    rho, intercept, clipped = np.zeros(2), np.zeros(2), [False, False]
    for i in (0, 1):
        slope, c = np.polyfit(x[:, i], y[:, i], 1)
        if not RHO_RANGE[0] <= slope <= RHO_RANGE[1]:
            slope = float(np.clip(slope, *RHO_RANGE))
            c = np.mean(y[:, i] - slope * x[:, i])
            clipped[i] = True
        rho[i], intercept[i] = slope, c
    residual = y - rho * x - intercept
    starts = np.array([e[0] for e in errors])
    P0 = np.cov(starts.T)
    P0[np.diag_indices(2)] = np.maximum(np.diag(P0), P0_FLOOR)
    return {
        "rho": rho,
        "b": intercept / (1 - rho),
        "Q": np.cov(residual.T),
        "m0": starts.mean(0),
        "P0": P0,
        "clipped": clipped,
    }


def residuals(params, e):
    rho, b = np.asarray(params["rho"]), np.asarray(params["b"])
    return e[1:] - rho * e[:-1] - (1 - rho) * b


def _resample(items, rng):
    return [items[i] for i in rng.integers(len(items), size=len(items))]


def _parameter_intervals(errors, samples, rng):
    """95% intervals of rho, b and Q, resampling whole flights."""
    draws = [fit_errors(_resample(errors, rng)) for _ in range(samples)]
    return {
        key: np.percentile(
            [d[key] for d in draws], [2.5, 97.5], axis=0
        ).tolist()
        for key in ("rho", "b", "Q")
    }


def _features(cfg, d):
    """Regressors at step k: signed reference velocity, acceleration, fan."""
    names = ["vx", "vy", "ax", "ay"]
    columns = [d["v"][:-1], d["a"][:-1]]
    if cfg.fan_position is not None and cfg.fan_direction is not None:
        u = np.asarray(cfg.fan_direction, float)[:2]
        u = u / np.linalg.norm(u)
        relative = d["r"][:-1] - np.asarray(cfg.fan_position, float)[:2]
        columns.append(np.stack([relative @ u, relative @ [-u[1], u[0]]], 1))
        names += ["fan_along", "fan_across"]
    columns.append(np.ones((len(d["e"]) - 1, 1)))
    return np.column_stack(columns), names + ["constant"]


def diagnostics(cfg, params, flights, samples, rng):
    """One-step residuals per axis regressed on the reference; report only.

    A delay delta gives e ~ -delta * r_dot, which changes sign with
    direction, so the velocity regressors are signed.
    """

    def coefficients(fs):
        X = np.vstack([_features(cfg, d)[0] for d in fs])
        y = np.vstack([residuals(params, d["e"]) for d in fs])
        return np.linalg.lstsq(X, y, rcond=None)[0]

    names = _features(cfg, flights[0])[1]
    estimate = coefficients(flights)
    draws = np.array(
        [coefficients(_resample(flights, rng)) for _ in range(samples)]
    )
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    return {
        axis: {
            name: [estimate[j, i], low[j, i], high[j, i]]
            for j, name in enumerate(names)
        }
        for i, axis in enumerate(("x", "y"))
    }


def make_model(dt, params, report_id=None, report_hash=None):
    return TrackingModel(
        dt,
        params["rho"],
        params["b"],
        params["Q"],
        params["m0"],
        params["P0"],
        report_id,
        report_hash,
    )


def fit(cfg, trials):
    """Fit the error model per fan, then validate it on held-out flights."""
    from geometry import Task

    fit_trials, held = split(trials, cfg.model["holdout_every"])
    data = {
        fan: [flight_data(t) for t in ts] for fan, ts in fit_trials.items()
    }
    if not data or min(len(d) for d in data.values()) < 2:
        raise ValueError("need at least two fitting flights per fan")
    samples, rng = (
        cfg.model["bootstrap"],
        np.random.default_rng(cfg.model["seed"]),
    )
    report = {"dt": cfg.dt, "config_hash": cfg.hash, "fans": {}}
    warnings = []
    for fan, flights in data.items():
        params = fit_errors([d["e"] for d in flights])
        if len(flights) < cfg.model["min_flights_warning"]:
            warnings.append(
                f"fan {fan}: {len(flights)} fit flights; plan for at least "
                f"{cfg.model['min_flights_warning']} before trusting the fit"
            )
        report["fans"][fan] = {
            **{k: np.asarray(v).tolist() for k, v in params.items()},
            "fit_flights": len(flights),
            "fit_trials": [t["meta"]["trial_id"] for t in fit_trials[fan]],
            "held_out_trials": [t["meta"]["trial_id"] for t in held[fan]],
        }
    text = json.dumps(report, sort_keys=True)
    report["hash"] = hashlib.sha256(text.encode()).hexdigest()[:12]
    report["id"] = "model_" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    report["warnings"] = warnings

    task = Task(cfg)
    report["validation"] = {}
    for fan, flights in data.items():
        entry = report["fans"][fan]
        model = model_from(report, fan, cfg)
        entry["intervals"] = _parameter_intervals(
            [d["e"] for d in flights], samples, rng
        )
        entry["b_poorly_determined"] = [
            bool(r > RHO_NEAR_ONE) for r in entry["rho"]
        ]
        entry["stationary_spread"] = np.sqrt(
            np.linalg.eigvalsh(model.stationary_cov().numpy())
        ).tolist()
        entry["diagnostics"] = diagnostics(cfg, entry, flights, samples, rng)
        if held[fan]:
            report["validation"][fan] = validate(
                model, [flight_data(t) for t in held[fan]], task, samples, rng
            )
    report["transfer"] = transfer(cfg, trials, task, samples, rng)
    return report


def transfer(cfg, trials, task, samples, rng):
    """Fit on the configured reference's flights, validate on other paths.

    Reported separately from the same-path held-out results.
    """
    reference = file_hash(cfg.reference)
    out = {}
    for fan, flights in by_fan(trials).items():
        own = [t for t in flights if t["meta"].get("plan_hash") == reference]
        others = defaultdict(list)
        for t in flights:
            if t["meta"].get("plan_hash") != reference:
                others[t["meta"]["plan_id"]].append(t)
        if len(own) < 2 or not others:
            continue
        params = fit_errors([flight_data(t)["e"] for t in own])
        model = make_model(cfg.dt, params)
        out[fan] = {
            plan_id: validate(
                model, [flight_data(t) for t in group], task, samples, rng
            )
            for plan_id, group in others.items()
        }
    return out


def save_report(report):
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"{report['id']}.json"
    path.write_text(json.dumps(report, indent=1))
    return path


def model_from(report, fan, cfg):
    key = str(fan_setting(fan))
    if key not in report["fans"]:
        raise ValueError(f"model {report['id']} has no fan {fan!r}")
    return make_model(
        report["dt"], report["fans"][key], report["id"], report["hash"]
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
    draws = [statistic(_resample(per_flight, rng)) for _ in range(samples)]
    return [float(value), *map(float, np.percentile(draws, [2.5, 97.5]))]


def squared_distance(deviation, cov):
    """d^2 = x^T P^-1 x per step, for x [N, 2] and P [N, 2, 2]."""
    return np.einsum("ki,kij,kj->k", deviation, np.linalg.inv(cov), deviation)


def _flight_checks(model, d, task):
    """Per-flight arrays used by every validation statistic."""
    e = d["e"]
    m, P = (x.numpy() for x in model.error_moments(len(e) - 1))
    mean, measured = d["r"] + m, d["r"] + e
    probs, hits = [], []
    for box in [task.workspace, *task.obstacles, task.goal]:
        for axis, (lo, hi) in enumerate((box.x, box.y)):
            p = interval_probability(
                lo,
                hi,
                torch.tensor(mean[:, axis]),
                torch.tensor(P[:, axis, axis]),
            )
            probs.append(p.numpy())
            hits.append((lo <= measured[:, axis]) & (measured[:, axis] <= hi))
    return {
        "error": np.linalg.norm(e, axis=1),
        "spread": np.sqrt(np.trace(P, axis1=1, axis2=2)),
        "d2": squared_distance(e - m, P),
        "residual": residuals(
            {"rho": model.rho.numpy(), "b": model.b.numpy()}, e
        ),
        "p": np.concatenate(probs),
        "hit": np.concatenate(hits),
    }


def validate(model, flights, task, samples, rng):
    """Held-out checks: coverage, calibration, Brier score and residuals.

    Flights aborted early drop out of later steps: `flights_by_step` says
    how many flights each step's statistics describe.
    """
    checks = [_flight_checks(model, d, task) for d in flights]

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
        "residual_bias": residual.mean(0).tolist(),
        "residual_std": residual.std(0).tolist(),
        "residual_lag1_corr": lag1.tolist(),
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
