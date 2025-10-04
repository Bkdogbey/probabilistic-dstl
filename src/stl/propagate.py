import numpy as np
from scipy.stats import norm


def compute_bounds(
    mean_trace,
    var_trace,
    time,
    min_height=50,
    start_time=0,
    end_time=10,
):
    mask = (time >= start_time) & (time <= end_time)
    idxs = np.where(mask)[0]
    if np.sum(mask) == 0:
        return 0.0, 1.0  # no time in the interval

    probs = []  # P(z(time) >= min_height) at each time in [start_time, end_time]
    for i in idxs:
        m = mean_trace[i]
        v = var_trace[i]
        if v <= 0:
            p = 1.0 if m >= min_height else 0.0
        else:
            std = np.sqrt(v)
            # P(Z >= h) = 1 - Phi((h - m)/std)
            p = 1.0 - norm.cdf((min_height - m) / std)
        probs.append(p)

    min_prob = float(
        np.min(probs)
    )  # Choosing the minimum probability over the interval due the G operator
    # Here we don't compute a separate loose upper bound; set both to min_prob.
    return min_prob, min_prob  # return the bounds
