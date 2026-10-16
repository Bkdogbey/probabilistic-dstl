"""Fit the planner's motion model to the completed flights.

The pdSTL planner uses a single integrator (src/models/dynamics.py):

    x[k+1] = x[k] + u[k] dt + w[k],   w ~ N(0, q_std^2 I)

Each flight gives one residual w per step of dt: where the drone went minus
where the commanded velocity, averaged over that step, would have taken it.
The residuals include Lighthouse noise and the drone's response lag, so the
fit describes the system as flown, not an ideal drone.

In this model the noise adds up, so the predicted spread grows as
q_std * sqrt(k). In flight the tracker's feedback keeps the error bounded,
so the file also keeps `track_cov`: the covariance of position minus
reference, which is what feedback tracking actually achieves.

    python model.py          # fit, print, save model.json
"""

import json

import numpy as np

import config
from analyze import FLIGHTS, load


def residuals(d, dt):
    """One-step residuals [n, 2] of x[k+1] - x[k] - u[k] dt for one flight."""
    t = d["t"]
    xy = np.column_stack([d["x"], d["y"]])
    v = np.column_stack([d["vx"], d["vy"]])
    steps = np.arange(0.0, t[-1] + 1e-9, dt)
    idx = np.searchsorted(t, steps - 1e-9)
    out = []
    for a, b in zip(idx[:-1], idx[1:]):
        u = v[a:b].mean(axis=0)  # commands sent during this step
        out.append(xy[b] - xy[a] - u * (t[b] - t[a]))
    return np.array(out)


def fit(w):
    """Noise of residuals w [n, 2]: covariance about zero and q_std.

    The planner has no bias term, so Q is taken about zero and keeps any
    drift. q_std is the largest standard deviation over directions.
    """
    Q = w.T @ w / len(w)
    return {
        "q_std": float(np.sqrt(np.linalg.eigvalsh(Q).max())),
        "Q": Q.tolist(),
        "bias": w.mean(axis=0).tolist(),
        "steps": len(w),
    }


def tracking_error(d, dt):
    """Position minus reference [n, 2], one sample per step of dt."""
    steps = np.arange(0.0, d["t"][-1] + 1e-9, dt)
    idx = np.searchsorted(d["t"], steps - 1e-9)
    return np.column_stack([d["x"] - d["rx"], d["y"] - d["ry"]])[idx]


def start_covariance(starts):
    """Covariance [2, 2] of the start error over flights; None if < 2."""
    if len(starts) < 2:
        return None
    return np.cov(np.array(starts).T).tolist()


def main(cfg):
    dt = cfg.dt
    per_flight, errors, starts = [], [], []
    print(f"{'flight':17} {'steps':>5} {'q_std':>6}  bias [m per step]")
    for path in sorted(FLIGHTS.glob("*.csv")):
        d, meta = load(path)
        if meta.get("outcome") != "done":
            continue  # aborted flights end in the fault that stopped them
        w = residuals(d, dt)
        f = fit(w)
        per_flight.append(w)
        errors.append(tracking_error(d, dt))
        starts.append([d["x"][0] - d["rx"][0], d["y"][0] - d["ry"][0]])
        print(f"{path.stem:17} {f['steps']:5d} {f['q_std']:6.3f}  "
              f"{np.round(f['bias'], 4)}")
    if not per_flight:
        print("no completed flights to fit")
        return
    model = {
        "dt": dt,
        "flights": len(per_flight),
        **fit(np.vstack(per_flight)),
        "x0_cov": start_covariance(starts),
        "track_cov": np.cov(np.vstack(errors).T).tolist(),
    }
    track_std = np.sqrt(np.linalg.eigvalsh(model["track_cov"]).max())
    print(f"all {model['flights']} flights: q_std {model['q_std']:.3f} m "
          f"per {dt} s step, bias {np.round(model['bias'], 4)}")
    print(f"tracking error: std {track_std:.3f} m (largest direction)")
    out = config.HERE / "model.json"
    out.write_text(json.dumps(model, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main(config.load())
